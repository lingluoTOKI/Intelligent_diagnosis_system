"""Export representative complete runs with explicit comparison conditions."""
from pathlib import Path
import csv,json
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'runs'
TARGET_SETTINGS=globals().get('TARGET_SETTINGS',dict(epochs=50,imgsz=224,batch=32,amp=False,workers=8,fraction=1.0,optimizer='AdamW',lr0=0.000833,momentum=0.9,weight_decay=0.0005,seed=0))
def segments(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=[{k.strip():float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    out=[]
    for row in rows:
        if not out or row['epoch']<=out[-1][-1]['epoch']:out.append([])
        out[-1].append(row)
    return [(i,s) for i,s in enumerate(out,1) if [int(r['epoch']) for r in s]==list(range(1,51))]
selected=[]
for tag,label,color in [('baseline','YOLO11n-cls','#2563EB'),('akconv','YOLO11n-cls + AKConv','#E07A3F'),('yolo26n','YOLO26n-cls','#169B83'),('yolo26n_akconv','YOLO26n-cls + AKConv','#8B5CF6')]:
    candidates=sorted((RUNS/'comparison_notebook').glob(f'*/{tag}/results.csv'),key=lambda p:p.stat().st_mtime,reverse=True)
    if tag in {'baseline','akconv'}: candidates.append(RUNS/'ablation'/tag/'results.csv')
    def candidate_rank(path):
        if not path.exists():return (-1,0)
        args_path=path.parent/'args.yaml'
        args=yaml.safe_load(args_path.read_text(encoding='utf-8')) if args_path.exists() else {}
        matched=all(args.get(k)==v for k,v in TARGET_SETTINGS.items())
        return (int(matched),path.stat().st_mtime)
    candidates.sort(key=candidate_rank,reverse=True)
    for path in candidates:
        if not path.exists():continue
        full=segments(path)
        if not full:continue
        index,rows=full[0] if path.parent.parent.name=='ablation' else full[-1]
        args=yaml.safe_load((path.parent/'args.yaml').read_text(encoding='utf-8'))
        manifest_path=path.parent.parent/(tag+'_manifest.json')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        if manifest.get('status') in ('failed','interrupted','running'):continue
        differences=[k for k,v in TARGET_SETTINGS.items() if args.get(k)!=v]
        best=max(rows,key=lambda r:r['metrics/accuracy_top1'])
        record={'model':label,'epochs':50,'best_top1_pct':round(best['metrics/accuracy_top1']*100,3),'best_top1_epoch':int(best['epoch']),'top5_at_best_top1_pct':round(best['metrics/accuracy_top5']*100,3),'last_top1_pct':round(rows[-1]['metrics/accuracy_top1']*100,3),'last_top5_pct':round(rows[-1]['metrics/accuracy_top5']*100,3),'max_top5_pct':round(max(r['metrics/accuracy_top5'] for r in rows)*100,3),'csv_training_seconds':rows[-1]['time'],'amp':args.get('amp'),'workers':args.get('workers'),'batch':args.get('batch'),'imgsz':args.get('imgsz'),'segment':index,'source_csv':str(path.relative_to(ROOT)),'comparison_scope':'historical AMP ablation' if path.parent.parent.name=='ablation' else 'FP32 2x2 framework experiment'}
        record.update(matching_settings=not differences,configuration_differences=','.join(differences),completion_verified=manifest.get('status')=='complete',data_fingerprint=manifest.get('data_fingerprint',''),parameters=manifest.get('parameters',''),gflops_at_input=manifest.get('gflops_at_input',''),wall_seconds=manifest.get('wall_seconds',''),akconv_layers=manifest.get('akconv_layers',''),framework_version=manifest.get('ultralytics',''))
        selected.append((tag,label,color,rows,record));break
if not selected:raise RuntimeError('没有完整50轮结果')
if len(selected)<4:print('四组尚未全部完成；只展示真实已有结果，不填补缺失组。')
with (RUNS/'comparison_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(selected[0][4]));w.writeheader();w.writerows(s[4] for s in selected)
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'#F8FAFC','grid.color':'#E2E8F0','axes.unicode_minus':False})
fig,axes=plt.subplots(2,2,figsize=(12,8.5))
metrics=[('metrics/accuracy_top1','Top-1 准确率',True),('metrics/accuracy_top5','Top-5 准确率',True),('train/loss','训练损失',False),('val/loss','验证损失',False)]
for ax,(metric,title,pct) in zip(axes.flat,metrics):
    for tag,label,color,rows,record in selected:
        legend=label+(' · AMP' if record['amp'] else ' · FP32')
        ax.plot([r['epoch'] for r in rows],[r[metric] for r in rows],color=color,lw=2,label=legend,linestyle='--' if tag.startswith('yolo26') else '-')
        if metric=='metrics/accuracy_top1':
            best=max(rows,key=lambda r:r[metric]);ax.scatter(best['epoch'],best[metric],s=30,color=color,zorder=4)
            offset={'baseline':(5,8),'akconv':(5,-18),'yolo26n':(5,5),'yolo26n_akconv':(5,-15)}[tag]
            ax.annotate(f"{best[metric]*100:.2f}%",(best['epoch'],best[metric]),xytext=offset,textcoords='offset points',color=color,fontweight='bold',fontsize=10)
    ax.set_title(title,loc='left',fontweight='bold',pad=10);ax.set_xlabel('训练轮次');ax.grid(alpha=.8);ax.set_xlim(0,56)
    if pct:ax.yaxis.set_major_formatter(PercentFormatter(1))
fig.suptitle('眼科分类 2×2 框架与AKConv实验',fontsize=18,fontweight='bold',y=.98)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=2,frameon=False,fontsize=10)
fig.text(.5,.045,'每组仅选一段完整50轮结果 · 输入224 · batch 32 · workers 8',ha='center',fontsize=10)
series_by_tag={item[0]:item for item in selected}
overlapping=[]
for first,second in [('baseline','yolo26n'),('akconv','yolo26n_akconv')]:
    if first in series_by_tag and second in series_by_tag:
        left,right=series_by_tag[first][3],series_by_tag[second][3]
        if all(a[key]==b[key] for a,b in zip(left,right) for key,_,_ in metrics):
            overlapping.append((first,second))
caption='两框架对应模型的损失、准确率曲线完全重合；本地分类配置结构相同。' if len(overlapping)==2 else '跨版本结果仅适用于当前本地分类配置；单次训练不代表统计显著收益。'
fig.text(.5,.018,caption,ha='center',fontsize=9,color='#64748B')
fig.subplots_adjust(left=.08,right=.93,bottom=.13,top=.83,wspace=.24,hspace=.4)
fig.savefig(RUNS/'comparison_curves.png',dpi=240,bbox_inches='tight')
fig.savefig(RUNS/'comparison_curves.pdf',bbox_inches='tight')
plt.close(fig)
for tag,label,color,rows,r in selected:print(f"{label}: best Top-1={r['best_top1_pct']:.2f}%, epoch={r['best_top1_epoch']}, Top-5同轮={r['top5_at_best_top1_pct']:.2f}%, AMP={r['amp']}, CSV段={r['segment']}")
print('已刷新comparison_summary.csv与comparison_curves.png；不含测试/中断/重复段。')

# Separate resource and accuracy panels keep identical trajectories readable.
fig,axes=plt.subplots(1,3,figsize=(13,5.6))
short_labels=['11 原生' if tag=='baseline' else '11 + AKConv' if tag=='akconv' else '26 原生' if tag=='yolo26n' else '26 + AKConv' for tag,_,_,_,_ in selected]
colors=[item[2] for item in selected]
for ax,key,title,scale,unit in [
    (axes[0],'best_top1_pct','最佳 Top-1',1,'%'),
    (axes[1],'parameters','训练后参数量',1e-6,' M'),
    (axes[2],'csv_training_seconds','CSV 记录训练耗时',1,' s')]:
    values=[float(item[4][key])*scale if item[4][key]!='' else float('nan') for item in selected]
    bars=ax.bar(short_labels,values,color=colors,width=.62)
    for bar,value in zip(bars,values):
        if value==value:
            ax.annotate(f'{value:.3f}{unit}' if key!='csv_training_seconds' else f'{value:.1f}{unit}',
                (bar.get_x()+bar.get_width()/2,value),xytext=(0,6),textcoords='offset points',ha='center',fontsize=10)
    ax.set_title(title,loc='left',fontweight='bold',pad=15)
    ax.set_ylim(bottom=0,top=max(v for v in values if v==v)*1.2)
    ax.tick_params(axis='x',labelrotation=22)
    ax.grid(axis='y',alpha=.8);ax.set_axisbelow(True)
fig.suptitle('四组实验：准确率、参数量与训练成本',fontsize=17,fontweight='bold')
fig.text(.5,.035,'50轮 · 224像素 · batch 32 · FP32 · workers 8 · 单个随机种子；耗时包含训练内验证，非推理延迟。',ha='center',fontsize=10)
fig.subplots_adjust(top=.82,bottom=.22,wspace=.25)
fig.savefig(RUNS/'comparison_relationships.png',dpi=240,bbox_inches='tight')
fig.savefig(RUNS/'comparison_relationships.pdf',bbox_inches='tight')
plt.close(fig)
print('已生成 comparison_relationships.png：指标分开呈现，不把单位混在一起。')
