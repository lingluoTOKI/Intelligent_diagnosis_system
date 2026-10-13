# 仅读取真实CSV；按运行配置分组展示，不覆盖训练产物。
import json, pathlib, csv
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import yaml
ROOT=pathlib.Path(r'C:/Users/47449/Desktop/yolo/Intelligent_diagnosis_system')
FIGURES=ROOT/'runs'/'report_figures'
FIGURES.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','axes.facecolor':'#FAFBFD','axes.edgecolor':'#CBD5E1','grid.color':'#E2E8F0','axes.unicode_minus':False,'savefig.bbox':'tight'})
COLORS={'baseline':'#2563EB','akconv':'#E07A3F','yolo26n':'#169B83','yolo26n_akconv':'#8B5CF6'}
LABELS={'baseline':'YOLO11n-cls','akconv':'YOLO11n-cls + AKConv','yolo26n':'YOLO26n-cls','yolo26n_akconv':'YOLO26n-cls + AKConv'}
def segments(path):
 with path.open(encoding='utf-8-sig',newline='') as f:
  rows=[{k.strip():float(v) for k,v in r.items()} for r in csv.DictReader(f)]
 out=[]
 for r in rows:
  if not out or r['epoch']<=out[-1][-1]['epoch']:out.append([])
  out[-1].append(r)
 return out
paths=list((ROOT/'runs'/'ablation').glob('*/results.csv'))+list((ROOT/'runs'/'comparison_notebook').glob('*/*/results.csv'))
groups={}
for path in sorted(paths,key=lambda p:p.stat().st_mtime):
 tag=path.parent.name
 if tag not in COLORS:continue
 a=path.parent/'args.yaml'
 args=yaml.safe_load(a.read_text(encoding='utf-8')) if a.exists() else {}
 legacy=path.parent.parent.name=='ablation'
 # 历史单独列出；新运行严格按精度、线程、batch和输入分组。
 key=('historical' if legacy else 'new',args.get('amp'),args.get('workers'),args.get('batch'),args.get('imgsz'))
 full=[s for s in segments(path) if len(s)==50 and s[-1]['epoch']==50]
 if not full:continue
 rows=full[0] if legacy else full[-1]
 groups.setdefault(key,{})[tag]=(path,rows)
print('仅展示完整50轮实验；未完成/冒烟实验不纳入最终对比。')
manifest=[]
for gi,(key,models) in enumerate(groups.items(),1):
 caption=f'50 epochs · input {key[4]} · batch {key[3]} · AMP {key[1]} · workers {key[2]}'
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 metrics=[('train/loss','训练损失',False),('val/loss','验证损失',False),('metrics/accuracy_top1','Top-1 准确率',True),('metrics/accuracy_top5','Top-5 准确率',True)]
 for ax,(metric,title,pct) in zip(axes.flat,metrics):
  for tag,(path,rows) in models.items():
   ax.plot([r['epoch'] for r in rows],[r[metric] for r in rows],color=COLORS[tag],lw=2,label=LABELS[tag])
   ax.scatter(rows[-1]['epoch'],rows[-1][metric],color=COLORS[tag],s=22,zorder=3)
  ax.set_title(title,loc='left',pad=12);ax.set_xlabel('训练轮次');ax.grid(alpha=.8);ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
  if pct:ax.yaxis.set_major_formatter(PercentFormatter(1))
 handles,labels=axes[0,0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='outside lower center',ncol=min(3,len(models)),frameon=False)
 fig.suptitle('眼科图像分类训练曲线\n'+caption,fontsize=15)
 stem=FIGURES/f'curves_group_{gi}'
 fig.savefig(stem.with_suffix('.png'),dpi=240);fig.savefig(stem.with_suffix('.pdf'));plt.show();plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 tags=list(models)
 for ax,metric,title in zip(axes,['metrics/accuracy_top1','metrics/accuracy_top5'],['最佳 Top-1 准确率','最佳 Top-1 所在轮次的 Top-5']):
  values=[]
  for tag in tags:
   rows=models[tag][1];best=max(rows,key=lambda r:r['metrics/accuracy_top1']);values.append(best[metric]*100)
  bars=ax.barh([LABELS[t] for t in tags],values,color=[COLORS[t] for t in tags],height=.5)
  ax.set_xlim(0,105);ax.set_title(title,loc='left');ax.set_xlabel('准确率 (%)');ax.grid(axis='x',alpha=.8);ax.set_axisbelow(True)
  for bar,v in zip(bars,values):ax.text(v+1,bar.get_y()+bar.get_height()/2,f'{v:.2f}%',va='center',fontsize=11,fontweight='bold')
 fig.suptitle(caption,fontsize=11)
 fig.savefig(FIGURES/f'accuracy_group_{gi}.png',dpi=240);fig.savefig(FIGURES/f'accuracy_group_{gi}.pdf');plt.show();plt.close(fig)
 manifest.append({'group':list(key),'models':{t:str(p) for t,(p,r) in models.items()}})
 # 展示本组实际已有的归一化混淆矩阵，保留原图数据和轴方向。
 images=[(t,p.parent/'confusion_matrix_normalized.png') for t,(p,r) in models.items() if (p.parent/'confusion_matrix_normalized.png').exists()]
 if images:
  fig,axes=plt.subplots(1,len(images),figsize=(7*len(images),6),squeeze=False,layout='constrained')
  for ax,(tag,path) in zip(axes.flat,images):ax.imshow(plt.imread(path));ax.set_title(LABELS[tag],pad=12);ax.axis('off')
  fig.savefig(FIGURES/f'confusion_group_{gi}.png',dpi=200);plt.show();plt.close(fig)
(FIGURES/'sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('图表输出:',FIGURES)
