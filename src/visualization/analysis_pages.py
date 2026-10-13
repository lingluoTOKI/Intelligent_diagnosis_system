"""Inline result workspaces with scoped filters and one export contract."""
from pathlib import Path
from collections import Counter
import json,csv,uuid,time,shutil
from datetime import datetime,timedelta
from PyQt5.QtCore import Qt,QTimer,QPointF,QDate
from PyQt5.QtGui import QPainter,QColor,QPolygonF,QPen
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QLabel,QLineEdit,QComboBox,QSplitter,QTableWidget,QTableWidgetItem,QHeaderView,QFileDialog,QTextEdit,QMessageBox,QScrollArea,QDialog,QCalendarWidget,QStyle)
from .background_jobs import BackgroundJob
from .workbench_layout import NAMES

class DateFilter(QLineEdit):
    """Optional date filter with a calendar and an explicit unrestricted value."""
    def __init__(self):
        super().__init__()
        self.setPlaceholderText('不限日期')
        self.setReadOnly(True)
        self.setToolTip('点击选择日期；在日历中选择“不限日期”可清除筛选')
        action=self.addAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView),QLineEdit.TrailingPosition)
        action.setToolTip('打开日历');action.triggered.connect(self.choose_date)
    def mousePressEvent(self,event):
        super().mousePressEvent(event)
        if event.button()==Qt.LeftButton:self.choose_date()
    def choose_date(self):
        dialog=QDialog(self);dialog.setWindowTitle('选择日期');layout=QVBoxLayout(dialog)
        calendar=QCalendarWidget();calendar.setGridVisible(False)
        from PyQt5.QtGui import QTextCharFormat,QPalette
        calendar.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
        calendar.setHorizontalHeaderFormat(QCalendarWidget.ShortDayNames)
        palette=calendar.palette()
        for role,color in ((QPalette.Base,'#242C37'),(QPalette.AlternateBase,'#303B49'),(QPalette.Text,'#EEF2F6'),(QPalette.ButtonText,'#EEF2F6'),(QPalette.Highlight,'#226F98'),(QPalette.HighlightedText,'#FFFFFF')):
            palette.setColor(role,QColor(color))
        calendar.setPalette(palette)
        header=QTextCharFormat();header.setBackground(QColor('#303B49'));header.setForeground(QColor('#ADB8C7'));calendar.setHeaderTextFormat(header)
        weekend=QTextCharFormat();weekend.setForeground(QColor('#83D8EA'))
        calendar.setWeekdayTextFormat(Qt.Saturday,weekend);calendar.setWeekdayTextFormat(Qt.Sunday,weekend)
        calendar.setStyleSheet("QCalendarWidget QWidget { background: #242C37; } QCalendarWidget QAbstractItemView { background: #242C37; color: #EEF2F6; alternate-background-color: #303B49; selection-background-color: #226F98; selection-color: white; border: none; } QCalendarWidget QToolButton { color: #EEF2F6; padding: 6px; border: none; } QCalendarWidget QMenu { background: #303B49; color: #EEF2F6; }")
        selected=QDate.fromString(self.text(),'yyyy-MM-dd')
        calendar.setSelectedDate(selected if selected.isValid() else QDate.currentDate())
        layout.addWidget(calendar)
        buttons=QHBoxLayout();layout.addLayout(buttons)
        def accept_date(date):
            self.setText(date.toString('yyyy-MM-dd'));dialog.accept()
        clear=QPushButton('不限日期');clear.clicked.connect(lambda:(self.clear(),dialog.accept()));buttons.addWidget(clear)
        today=QPushButton('今天');today.clicked.connect(lambda:accept_date(QDate.currentDate()));buttons.addWidget(today)
        confirm=QPushButton('确定');confirm.clicked.connect(lambda:accept_date(calendar.selectedDate()));buttons.addWidget(confirm)
        calendar.activated.connect(accept_date)
        dialog.resize(480,400);dialog.exec_()

class ProbabilityItem(QTableWidgetItem):
    def __lt__(self,other):
        left=self.data(Qt.UserRole+1);right=other.data(Qt.UserRole+1)
        return (left if left is not None else -1)<(right if right is not None else -1)

class Distribution(QWidget):
    COLORS = ('#4DD6B4','#51C5E3','#A49AE8','#E8B96B','#75B8A4','#DA91B3','#8AA6D1','#B8C580')
    def __init__(self):
        super().__init__();self.counts={};self.daily={};self.mode=0
        self.adapt_geometry()
    def adapt_geometry(self):
        fm=self.fontMetrics();self.row_height=fm.height()+30
        count=max(1,len(self.counts))
        height=70+count*self.row_height if self.mode==0 else (270+count*(fm.height()+12) if self.mode==1 else 350)
        self.setMinimumSize(max(300,fm.horizontalAdvance('糖尿病视网膜病变  10000')+48),height)
        self.updateGeometry();self.update()
    def set_mode(self,index):self.mode=index;self.adapt_geometry()
    def set_counts(self,counts):self.counts=counts;self.adapt_geometry()
    def paintEvent(self,event):
        from PyQt5.QtCore import QRectF
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(),QColor('#242C37'));p.setPen(QColor('#EEF2F6'))
        fm=p.fontMetrics();items=sorted(self.counts.items(),key=lambda x:-x[1]);total=sum(self.counts.values())
        p.drawText(18,32,'每日记录数量' if self.mode==2 else f'分类分布 · {total} 张')
        if self.mode==2:
            if not self.daily:p.drawText(18,75,'当前范围暂无日期数据');return
            days=sorted(self.daily);values=[self.daily[d] for d in days];peak=max(1,max(values))
            left=60;top=75;width=self.width()-85;height=self.height()-145
            for i in range(5):
                y=top+height*i/4;p.setPen(QPen(QColor('#394451'),1));p.drawLine(QPointF(left,y),QPointF(left+width,y))
                p.setPen(QColor('#ADB8C7'));p.drawText(8,int(y+5),str(round(peak*(1-i/4))))
            points=[QPointF(left+i*width/max(1,len(days)-1),top+height*(1-v/peak)) for i,v in enumerate(values)]
            p.setPen(QPen(QColor('#4DD6B4'),2));p.drawPolyline(QPolygonF(points))
            for point in points:p.drawEllipse(point,3,3)
            p.setPen(QColor('#EEF2F6'));p.drawText(18,self.height()-38,days[0]);p.drawText(18,self.height()-12,'至 '+days[-1]);return
        if not total:p.drawText(18,75,'暂无分类结果');return
        if self.mode==1:
            side=180;rect=QRectF((self.width()-side)/2,55,side,side);angle=0
            for i,(name,count) in enumerate(items):
                span=round(count/total*5760);p.setPen(QPen(QColor('#242C37'),2));p.setBrush(QColor(self.COLORS[i%8]));p.drawPie(rect,angle,span);angle+=span
            for i,(name,count) in enumerate(items):
                y=270+i*(fm.height()+12);p.fillRect(18,y-fm.ascent()+3,12,12,QColor(self.COLORS[i%8]));p.setPen(QColor('#EEF2F6'))
                p.drawText(40,y,f'{NAMES.get(name,name)}  {count} ({count/total:.1%})')
            return
        maximum=max(self.counts.values());width=self.width()-36
        for i,(name,count) in enumerate(items):
            y=65+i*self.row_height;p.setPen(QColor('#EEF2F6'))
            label=NAMES.get(name,name);value=str(count)
            p.drawText(18,y,label);p.drawText(self.width()-18-fm.horizontalAdvance(value),y,value)
            p.fillRect(18,y+10,width,10,QColor('#394451'))
            p.fillRect(18,y+10,round(width*count/maximum),10,QColor(self.COLORS[i%8]))

class AnalysisPage(QWidget):
    def __init__(self,window,db,mode):
        super().__init__();self.window=window;self.db=db;self.mode=mode;self.rows=[];self.pending=[];self.page=0;self.jobs=[];self.generation=0;self.batch_id=None;self.running=False
        layout=QVBoxLayout(self);toolbar=QHBoxLayout();layout.addLayout(toolbar)
        def button(text,fn):
            b=QPushButton(text);b.clicked.connect(fn);toolbar.addWidget(b);return b
        if mode=="batch":
            self.add_button=button("添加图像",self.add_files);self.folder_button=button("添加文件夹",self.add_folder)
            self.remove_button=button("移除所选",self.remove_selected)
            self.start_button=button("开始分类",window.batch_process)
            self.retry_button=button("重试失败项",self.retry)
            self.batch_choice=QComboBox();self.batch_choice.setMinimumWidth(180);toolbar.addWidget(self.batch_choice)
            self.batch_choice.activated.connect(self.restore_batch);self.reload_batches()
        else:
            self.start=DateFilter()
            self.end=DateFilter()
            self.batch_filter=QLineEdit();self.batch_filter.setPlaceholderText("批次编号（可选）")
            toolbar.addWidget(QLabel("开始日期"));toolbar.addWidget(self.start)
            toolbar.addWidget(QLabel("结束日期"));toolbar.addWidget(self.end)
            toolbar.addWidget(self.batch_filter,1)
            button("筛选 / 刷新",self.refresh)
        self.export_button=button("导出当前范围",self.export)
        filters=QHBoxLayout();layout.addLayout(filters)
        self.search=QLineEdit();self.search.setPlaceholderText("按图像路径或文件名搜索")
        self.category=QComboBox();self.category.addItem("全部类别","")
        for name,title in NAMES.items():self.category.addItem(title,name)
        filters.addWidget(self.search,1);filters.addWidget(self.category)
        self.state=QComboBox();self.state.addItems(["全部状态","分类成功","分类失败","保存失败","待处理"])
        if mode=="batch":filters.addWidget(self.state)
        self.timer=QTimer(self);self.timer.setSingleShot(True);self.timer.setInterval(250);self.timer.timeout.connect(self.reset_page)
        self.search.textChanged.connect(lambda:self.timer.start());self.category.currentIndexChanged.connect(self.reset_page);self.state.currentIndexChanged.connect(self.reset_page)
        self.summary=QLabel("尚无数据");self.summary.setWordWrap(True);layout.addWidget(self.summary)
        split=QSplitter();layout.addWidget(split,1)
        self.table=QTableWidget(0,6);self.table.setHorizontalHeaderLabels(["图像","类别","概率","分类状态","保存状态","时间 / 批次"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers);self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive);self.table.horizontalHeader().setStretchLastSection(True);self.table.setAlternatingRowColors(True);self.table.setWordWrap(False);self.table.verticalHeader().hide();self.table.setSortingEnabled(True)
        split.addWidget(self.table)
        right=QWidget();r=QVBoxLayout(right);self.chart=Distribution()
        self.chart_choice=QComboBox();self.chart_choice.addItems(['柱状图 · 类别数量','饼图 · 类别占比'])
        if mode=='history':self.chart_choice.addItem('趋势图 · 每日数量')
        self.chart_choice.currentIndexChanged.connect(self.chart.set_mode);r.addWidget(self.chart_choice)
        self.chart_scroll=QScrollArea();self.chart_scroll.setWidgetResizable(True);self.chart_scroll.setWidget(self.chart);r.addWidget(self.chart_scroll,1)
        self.detail_toggle=QPushButton('展开所选记录详情');self.detail_toggle.setCheckable(True);r.addWidget(self.detail_toggle)
        self.detail=QTextEdit();self.detail.setReadOnly(True);self.detail.setPlaceholderText('选择记录后展开，查看完整路径、概率与模型信息。');self.detail.setMinimumHeight(120);self.detail.setMaximumHeight(220);self.detail.hide()
        self.detail_toggle.toggled.connect(self.detail.setVisible)
        self.detail_toggle.toggled.connect(lambda checked:self.detail_toggle.setText('收起记录详情' if checked else '展开所选记录详情'))
        r.addWidget(self.detail)
        split.addWidget(right);split.setChildrenCollapsible(False);split.setSizes([900,480]);self.table.itemSelectionChanged.connect(self.show_detail)
        footer=QHBoxLayout();layout.addLayout(footer)
        self.previous=QPushButton("上一页");self.next=QPushButton("下一页");self.page_label=QLabel()
        self.previous.setEnabled(False);self.next.setEnabled(False)
        self.previous.clicked.connect(lambda:self.move(-1));self.next.clicked.connect(lambda:self.move(1))
        for w in (self.previous,self.page_label,self.next):footer.addWidget(w)
        if mode=="history":
            delete=QPushButton("删除所选记录");delete.clicked.connect(self.delete_selected);footer.addWidget(delete)
            details=QPushButton("查看图像详情");details.clicked.connect(self.open_detail);footer.addWidget(details)
        footer.addStretch();self.export_path=None
        open_folder=QPushButton("打开导出目录");open_folder.clicked.connect(self.open_export);footer.addWidget(open_folder)
        self.export_button.setEnabled(False)
        if mode=="batch":self.render_batch()
    def adapt_typography(self):
        fm=self.table.fontMetrics();height=fm.height()+18
        self.table.verticalHeader().setMinimumSectionSize(height)
        self.table.verticalHeader().setDefaultSectionSize(height)
        self.table.horizontalHeader().setMinimumHeight(height+4)
        for col,label in enumerate(('图像文件名称示例','糖尿病视网膜病变','100.0%','分类状态','保存状态','2026-10-13 13:00:00')):
            self.table.setColumnWidth(col,fm.horizontalAdvance(label)+28)
        if self.mode=='history':
            for field in (self.start,self.end):field.setFixedWidth(field.fontMetrics().horizontalAdvance('YYYY-MM-DD')+60)
        self.chart.adapt_geometry()
    def submit(self,operation,callback):
        job=BackgroundJob(operation,self);self.jobs.append(job)
        def complete(payload):
            if payload['error']:
                self.export_button.setEnabled(True)
                self.window.show_message_box("操作失败",payload['error'])
            else:callback(payload['result'])
        job.completed.connect(complete)
        job.finished.connect(lambda:self.jobs.remove(job));job.finished.connect(job.deleteLater);job.start()
    def filters(self):
        result={"search":self.search.text().strip(),"disease":self.category.currentData(),"real_only":True}
        if self.mode=="history":
            for key,widget in (("start",self.start),("end",self.end)):
                if widget.text().strip():
                    day=datetime.strptime(widget.text().strip(),"%Y-%m-%d")
                    result[key]=(day+(timedelta(days=1) if key=="end" else timedelta())).strftime("%Y-%m-%d")
            if self.start.text() and self.end.text() and self.start.text()>self.end.text():
                raise ValueError('开始日期不能晚于结束日期')
            result['batch']=self.batch_filter.text().strip()
        return result
    def refresh(self,*args):
        if self.mode=="batch":self.render_batch();return
        try:filters=self.filters()
        except ValueError:self.summary.setText("请检查日期：开始日期不能晚于结束日期，格式为 YYYY-MM-DD");return
        self.generation+=1;generation=self.generation;page=self.page
        def ready(data):
            if generation!=self.generation:return
            rows,total,self.page,counts,daily=data
            self.chart.daily=daily
            self.rows=[dict(r,path=r['image_path'],status='success',save_status='saved') for r in rows]
            self.paint_rows(self.rows,total,counts)
        self.submit(lambda cancel,progress:self.db.filtered(filters,page),ready)
    def reset_page(self,*args):self.page=0;self.refresh()
    def move(self,delta):self.page=max(0,self.page+delta);self.refresh()
    def selected(self):
        ids={i.row() for i in self.table.selectedItems()}
        return [self.table.item(i,0).data(Qt.UserRole) for i in sorted(ids)]
    def paint_rows(self,rows,total,counts):
        self.table.setSortingEnabled(False);self.table.setRowCount(len(rows))
        for i,row in enumerate(rows):
            probability=row.get('confidence');name=row.get('disease_name','')
            values=[Path(row.get('path') or '').name,NAMES.get(name,name),f"{probability:.1%}" if probability is not None else '—',{'success':'成功','failed':'失败','pending':'待处理'}.get(row.get('status'),'—'),{'saved':'已保存','failed':'保存失败','pending':'—'}.get(row.get('save_status'),'—'),row.get('timestamp','')+' '+(row.get('batch_id') or '')]
            for col,value in enumerate(values):
                item=ProbabilityItem(value) if col==2 else QTableWidgetItem(value)
                if col==2:item.setData(Qt.UserRole+1,probability)
                item.setToolTip(value);self.table.setItem(i,col,item)
            self.table.item(i,0).setData(Qt.UserRole,row)
        self.table.setSortingEnabled(True);self.chart.set_counts(counts)
        self.export_button.setEnabled(total>0)
        self.page_label.setText(f"第 {self.page+1} 页 · 当前范围 {total} 条")
        self.previous.setEnabled(self.page>0);self.next.setEnabled((self.page+1)*100<total)
        if self.mode=='history':self.summary.setText(f"当前筛选 {total} 条记录（排除标注演示）· 图表与导出范围一致")
    def render_batch(self):
        for widget in (self.add_button,self.folder_button,self.remove_button,self.start_button,self.retry_button,self.batch_choice):widget.setEnabled(not self.running)
        ready=self.window.detector is not None and self.window.detector.model is not None
        self.start_button.setEnabled(not self.running and ready and any(r["status"]=="pending" for r in self.pending))
        self.retry_button.setEnabled(not self.running and ready and any(r["status"]=="failed" or r["save_status"]=="failed" for r in self.pending))
        search=self.search.text().casefold();disease=self.category.currentData();state=self.state.currentIndex()
        rows=[r for r in self.pending if search in r.get('path','').casefold() and (not disease or disease==r.get('disease_name')) and (state==0 or (state==1 and r['status']=='success') or (state==2 and r['status']=='failed') or (state==3 and r['save_status']=='failed') or (state==4 and r['status']=='pending'))]
        self.rows=rows;total=len(rows);self.page=min(self.page,max(0,(total-1)//100))
        self.paint_rows(rows[self.page*100:(self.page+1)*100],total,dict(Counter(r['disease_name'] for r in rows if r['status']=='success')))
        success=sum(r['status']=='success' for r in self.pending);failed=sum(r['status']=='failed' for r in self.pending);save_failed=sum(r['save_status']=='failed' for r in self.pending)
        self.summary.setText(f"总数 {len(self.pending)} · 分类成功 {success} · 分类失败 {failed} · 待处理 {len(self.pending)-success-failed} · 保存失败 {save_failed} · 累计推理 {sum(r.get('seconds',0) for r in self.pending):.2f}s · 批次 {self.batch_id or '未开始'}")
    def add_paths(self,paths):
        if self.running:return
        existing={r['path'] for r in self.pending}
        for path in paths:
            if str(path) not in existing:self.pending.append({'path':str(path),'status':'pending','save_status':'pending'});existing.add(str(path))
        self.render_batch()
    def add_files(self):
        paths,_=QFileDialog.getOpenFileNames(self,"添加图像","","图像 (*.png *.jpg *.jpeg *.bmp)");self.add_paths(paths)
    def add_folder(self):
        folder=QFileDialog.getExistingDirectory(self,"添加文件夹")
        if folder:self.add_paths(sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp')))
    def remove_selected(self):
        if self.running:return
        selected=self.selected();self.pending=[r for r in self.pending if r not in selected];self.render_batch()
    def retry(self):
        if self.running:return
        failed=[r['path'] for r in self.pending if r['status']=='failed' or r['save_status']=='failed']
        if not failed:return
        self.pending=[];self.add_paths(failed);self.window.batch_process()
    def reload_batches(self):
        self.batch_choice.clear();self.batch_choice.addItem("选择已保存批次",None)
        for b in self.db.list_batches():self.batch_choice.addItem(b['created']+' '+b['state'],b['batch_id'])
    def restore_batch(self):
        if self.running:return
        bid=self.batch_choice.currentData()
        if bid:self.batch_id=bid;self.pending=self.db.batch_items(bid);self.render_batch()
    def show_detail(self):
        rows=self.selected()
        if rows:
            row=rows[0];prob=row.get('probabilities');model=row.get('model_info')
            for key,value in [('probabilities',prob),('model_info',model)]:
                if isinstance(value,str):
                    try:row=dict(row);row[key]=json.loads(value)
                    except ValueError:pass
            text=f"图像：{row.get('path','')}\n时间：{row.get('timestamp','')}\n批次：{row.get('batch_id') or '旧记录 / 单图'}\n错误原因：{row.get('error') or '无'}\n"
            text+="八类概率：\n"+json.dumps(row.get('probabilities') or "旧记录未保存完整概率",ensure_ascii=False,indent=2)
            text+="\n模型信息：\n"+json.dumps(row.get('model_info') or "旧记录未保存模型信息",ensure_ascii=False,indent=2)
            self.detail.setPlainText(text)
    def open_detail(self):
        rows=self.selected()
        if rows:self.window.view_history_record(rows[0])
    def delete_selected(self):
        rows=self.selected()
        if rows and QMessageBox.question(self,"删除记录",f"删除选中的 {len(rows)} 条记录？",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)==QMessageBox.Yes:
            for row in rows:self.db.delete_by_record_id(row['record_id'])
            self.refresh()
    def export(self):
        folder=QFileDialog.getExistingDirectory(self,"导出当前筛选结果、图表及报告")
        if not folder:return
        try:filters=self.filters()
        except ValueError:self.window.show_message_box("日期无效","请先修正日期");return
        snapshot=[dict(r) for r in self.rows];mode=self.mode;db=self.db;bid=self.batch_id;family=self.window.font_family_combo.currentData()
        self.export_button.setEnabled(False)
        def work(cancel,progress):
            from .result_export import export_bundle
            rows=db.filtered(filters,export=True)[0] if mode=='history' else snapshot
            return export_bundle(folder,mode,rows,filters,bid,family)
        def done(path):self.export_path=path;self.export_button.setEnabled(True);self.window.show_message_box("导出完成",str(path))
        self.submit(work,done)
    def open_export(self):
        if self.export_path:
            from PyQt5.QtGui import QDesktopServices
            from PyQt5.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.export_path)))
