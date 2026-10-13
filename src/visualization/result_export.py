"""Scoped, structured exports independent of live widgets."""
import csv,json,threading
from pathlib import Path
from datetime import datetime
from collections import Counter
_LOCK=threading.Lock()

def export_bundle(folder,mode,rows,filters,batch_id=None,family="DengXian"):
    target=Path(folder)/(mode+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"));target.mkdir()
    fields=['path','timestamp','batch_id','disease_name','confidence','status','save_status','error','seconds','probabilities','model_info']
    normalized=[]
    for row in rows:
        item=dict(row);item.setdefault('path',row.get('image_path',''));item.setdefault('status','success');item.setdefault('save_status','saved');normalized.append(item)
    def cell(value):
        if isinstance(value,(dict,list)):return json.dumps(value,ensure_ascii=False)
        if isinstance(value,str) and value.startswith(('=','+','-','@')):return "'"+value
        return value
    with (target/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(fields)
        writer.writerows([[cell(row.get(key,'')) for key in fields] for row in normalized])
    counts=Counter(r.get('disease_name') for r in normalized if r['status']=='success')
    metadata={'scope':mode,'batch_id':batch_id,'filters':filters,'total':len(rows),'classification_success':sum(counts.values()),'classification_failed':sum(r['status']=='failed' for r in normalized),'save_failed':sum(r['save_status']=='failed' for r in normalized),'note':'分类分布不是模型准确率，也不代表人群患病率。'}
    (target/'scope.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    with (target/'statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['类别','数量']);writer.writerows(counts.items())
    with _LOCK:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib import rc_context
        with rc_context({'font.sans-serif':[family,'Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False}):
            figure=Figure(figsize=(11,7),layout='constrained');FigureCanvasAgg(figure);ax=figure.subplots()
            names=list(counts);values=list(counts.values())
            if names:ax.barh(names,values,color='#249C85');ax.invert_yaxis();ax.set_xlabel('分类数量')
            else:ax.text(.5,.5,'当前范围暂无成功分类',ha='center',transform=ax.transAxes)
            ax.set_title('本次批量分类分布' if mode=='batch' else '历史筛选结果分布')
            figure.savefig(target/'distribution.png',dpi=200);figure.savefig(target/'distribution.pdf')
            with PdfPages(target/'report.pdf') as pdf:
                cover=Figure(figsize=(8.27,11.69));FigureCanvasAgg(cover)
                text='眼科图像分类分析报告\n\n'
                text+='统计范围：'+('本次批量任务' if mode=='batch' else '历史筛选结果')+'\n'
                text+='导出时间：'+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'\n'
                text+='批次编号：'+str(batch_id or filters.get('batch') or '全部符合条件的记录')+'\n'
                text+='日期范围：'+str(filters.get('start') or '不限')+' 至 '+str(filters.get('end') or '不限')+'（结束边界不含）\n'
                text+='类别：'+str(filters.get('disease') or '全部')+'\n'
                text+='文件筛选：'+str(filters.get('search') or '不限')+'\n\n'
                text+=f"记录数：{metadata['total']}\n分类成功：{metadata['classification_success']}\n分类失败：{metadata['classification_failed']}\n保存失败：{metadata['save_failed']}\n\n"
                text+='本报告的数量和分布仅描述当前记录，不代表模型准确率或人群患病率。\n完整逐项结果及模型信息见同目录 results.csv。'
                cover.text(.08,.93,text,va='top',fontsize=12,wrap=True);pdf.savefig(cover);pdf.savefig(figure)
                if mode=='history':
                    daily=Counter(str(r.get('timestamp',''))[:10] for r in normalized)
                    trend=Figure(figsize=(11,7),layout='constrained');FigureCanvasAgg(trend);axis=trend.subplots()
                    days=sorted(daily);axis.plot(days,[daily[d] for d in days],color='#249C85');axis.tick_params(axis='x',rotation=45);axis.set_title('当前筛选范围每日分类记录数量')
                    trend.savefig(target/'daily.png',dpi=200);pdf.savefig(trend)
    return str(target)
