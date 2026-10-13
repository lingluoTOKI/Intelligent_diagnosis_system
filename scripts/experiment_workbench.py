"""Notebook helpers for reproducible four-group classification experiments."""
from pathlib import Path
import csv,json,time,subprocess,hashlib
import yaml
TAGS=('baseline','akconv','yolo26n','yolo26n_akconv')
LABELS=dict(zip(TAGS,('YOLO11n-cls','YOLO11n-cls + AKConv','YOLO26n-cls','YOLO26n-cls + AKConv')))
def dataset_snapshot(data,classes):
    counts={};files=[];digest=hashlib.sha256()
    for split in ('train','val'):
        folder=Path(data)/split
        if not folder.is_dir():raise FileNotFoundError(folder)
        actual=sorted(p.name for p in folder.iterdir() if p.is_dir())
        if actual!=classes:raise ValueError(f'{split} classes: {actual}')
        counts[split]={}
        for label in classes:
            images=sorted(p for p in (folder/label).rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'})
            if not images:raise ValueError(f'Empty class: {split}/{label}')
            counts[split][label]=len(images)
            for path in images:
                relative=path.relative_to(data).as_posix()
                content=hashlib.sha256(path.read_bytes()).hexdigest()
                digest.update((relative+'\0'+content+'\n').encode())
                files.append({'path':relative,'sha256':content})
    return {'counts':counts,'content_fingerprint_sha256':digest.hexdigest(),'files':files}
def complete_segments(path,epochs=50):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows=[{k.strip():float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    segments=[]
    for row in rows:
        if not segments or row['epoch']<=segments[-1][-1]['epoch']:segments.append([])
        segments[-1].append(row)
    return [(i,s) for i,s in enumerate(segments,1) if [int(r['epoch']) for r in s]==list(range(1,epochs+1))]
def overview(root,settings):
    records=[]
    paths=list((Path(root)/'runs'/'comparison_notebook').glob('*/*/results.csv'))+list((Path(root)/'runs'/'ablation').glob('*/results.csv'))
    for path in sorted(paths):
        tag=path.parent.name
        if tag not in TAGS:continue
        args=yaml.safe_load((path.parent/'args.yaml').read_text(encoding='utf-8'))
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        different=[k for k,v in settings.items() if args.get(k)!=v]
        full=complete_segments(path,settings['epochs'])
        evidence=path.parent.parent/(tag+'_manifest.json')
        manifest=json.loads(evidence.read_text(encoding='utf-8')) if evidence.exists() else {}
        status='完整同配置' if full and not different else ('历史/异配置' if full else '未完成')
        if manifest.get('status') in ('failed','interrupted'):status='训练/最终验证失败'
        records.append({'model':LABELS[tag],'status':status,'last_epoch':int(float(rows[-1]['epoch'])) if rows else 0,'differences':','.join(different),'manifest':bool(manifest),'path':str(path)})
    return records
def run_training(tag,source,yaml_path,root,output,python,env,settings,snapshot):
    root,output=Path(root),Path(output)
    if tag not in TAGS:raise ValueError(tag)
    if not 0<settings['fraction']<=1 or settings['epochs']<1:raise ValueError('Invalid epochs/fraction')
    output.mkdir(parents=True,exist_ok=True)
    log=output/(tag+'.log');manifest=output/(tag+'_manifest.json')
    if log.exists() or (output/tag).exists():raise FileExistsError('该实验已存在；重新运行配置单元创建新的RUN_ID，不能覆盖')
    spec={'tag':tag,'source':str(source),'yaml':str(yaml_path),'yaml_sha256':hashlib.sha256(Path(yaml_path).read_bytes()).hexdigest(),'settings':settings,'data_fingerprint':snapshot['content_fingerprint_sha256'],'counts':snapshot['counts'],'status':'pending','created_at':time.strftime('%Y-%m-%d %H:%M:%S')}
    manifest.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    runner=output/(tag+'_runner.py')
    code='''import sys,os,json,time,multiprocessing,traceback
from pathlib import Path
spec_path=Path(SPEC_PATH)
spec=json.loads(spec_path.read_text(encoding='utf-8'))
sys.path.insert(0,spec['source'])
from ultralytics import YOLO
import torch,ultralytics
if __name__=='__main__':
    multiprocessing.freeze_support()
    start=time.perf_counter()
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
        spec.update(status='running',python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,framework_file=ultralytics.__file__,gpu=torch.cuda.get_device_name(0))
        print('GPU:',spec['gpu'],'AMP:',spec['settings']['amp'],'workers:',spec['settings']['workers'],flush=True)
        spec_path.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
        model=YOLO(spec['yaml'],task='classify')
        count=sum(type(x).__name__=='AKConv' for x in model.model.modules())
        if ('akconv' in spec['tag']) != (count>0):raise RuntimeError('模型AKConv结构与实验组不一致')
        settings=dict(spec['settings'],data=DATA_PATH,device=0,project=str(spec_path.parent),name=spec['tag'],exist_ok=False,plots=True)
        model.train(**settings)
        best=spec_path.parent/spec['tag']/'weights'/'best.pt'
        saved=YOLO(str(best))
        actual=sum(type(x).__name__=='AKConv' for x in saved.model.modules())
        if actual!=count:raise RuntimeError('保存权重的AKConv结构不一致')
        from ultralytics.utils.torch_utils import get_flops
        spec.update(status='complete',akconv_layers=actual,parameters=sum(p.numel() for p in saved.model.parameters()),gflops_at_input=get_flops(saved.model,imgsz=settings['imgsz']),wall_seconds=time.perf_counter()-start,best_checkpoint=str(best))
    except BaseException as error:
        spec.update(status='failed',error=str(error),wall_seconds=time.perf_counter()-start)
        traceback.print_exc()
        raise
    finally:
        spec_path.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
'''.replace('SPEC_PATH',repr(str(manifest))).replace('DATA_PATH',repr(str(root/'data')))
    compile(code,str(runner),'exec');runner.write_text(code,encoding='utf-8')
    start=time.monotonic()
    print(f'{tag}: workers={settings["workers"]}, AMP={settings["amp"]}; log={log}',flush=True)
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen([str(python),str(runner)],env=env,cwd=str(root),stdout=stream,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                time.sleep(5)
                try:
                    from IPython import get_ipython
                    from IPython.display import clear_output
                    if get_ipython() is not None:clear_output(wait=True)
                except ImportError:pass
                lines=log.read_text(encoding='utf-8',errors='replace').splitlines()
                print(f'{tag} | {time.monotonic()-start:.0f}s | workers={settings["workers"]} | AMP={settings["amp"]}\n'+ '\n'.join(lines[-3:]),flush=True)
        except KeyboardInterrupt:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
            process.wait()
            record=json.loads(manifest.read_text(encoding='utf-8'));record['status']='interrupted';manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            raise
    if process.returncode:
        print(log.read_text(encoding='utf-8',errors='replace'))
        raise RuntimeError(f'{tag}失败。完整日志: {log}；若报错与数据加载进程有关，将WORKERS改为4再试，不自动回退')
    record=json.loads(manifest.read_text(encoding='utf-8'))
    if record['status']!='complete':raise RuntimeError('进程结束但未通过完成验证')
    print('完成:',record['best_checkpoint'],'参数量:',record['parameters'],'AKConv层:',record['akconv_layers'])
    return record
