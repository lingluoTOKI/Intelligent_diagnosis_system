"""Image-first desktop layout and inexpensive native Qt probability displays."""
from pathlib import Path
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap, QFont, QFontDatabase
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                            QFrame, QProgressBar, QSizePolicy, QSplitter, QStackedWidget, QScrollArea, QComboBox, QGroupBox, QGridLayout, QTabWidget, QTextEdit, QFileDialog)

NAMES = {"AMD": "黄斑变性", "Cataract": "白内障", "Diabetic Retinopathy": "糖尿病视网膜病变",
         "Glaucoma": "青光眼", "Hypertensive Retinopathy": "高血压性视网膜病变",
         "Myopia": "近视性黄斑病变", "Normal": "正常", "Other": "其他"}


class ImageCanvas(QLabel):
    open_requested = pyqtSignal()
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._image = QPixmap()
        self._scaled = QPixmap()
        self._cache_key = None
        self.setMinimumSize(260, 150)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("双击导入图像，也可以将图像拖到这里")

    def setPixmap(self, pixmap):
        self._image = pixmap
        self._cache_key = None
        self.update()

    def pixmap(self):
        return self._image

    def clear(self):
        self._image = QPixmap()
        self._cache_key = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1B222B"))
        # Quiet grid is useful in the empty state; never paint over the image.
        if self._image.isNull():
            painter.setPen(QColor("#27313D"))
            for x in range(0, self.width(), 32): painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), 32): painter.drawLine(0, y, self.width(), y)
            cx, cy = self.width() / 2, self.height() / 2 - 48
            painter.setPen(QPen(QColor("#51C5E3"), 2))
            painter.drawEllipse(QRectF(cx-29, cy-20, 58, 40))
            painter.drawEllipse(QRectF(cx-9, cy-9, 18, 18))
            painter.setFont(QFont("Microsoft YaHei", self.font().pointSize()+1, QFont.Bold))
            painter.setPen(QColor("#EEF2F6"))
            painter.drawText(QRectF(0, cy+38, self.width(), 32), Qt.AlignCenter, "拖入眼科图像")
            painter.setFont(QFont("Microsoft YaHei", max(12,self.font().pointSize()-2)))
            painter.setPen(QColor("#ADB8C7"))
            painter.drawText(QRectF(0, cy+80, self.width(), 25), Qt.AlignCenter, "也可点击下方「加载图像」")
        else:
            size = self.size()
            key = (self._image.cacheKey(), size.width(), size.height())
            if key != self._cache_key:
                self._scaled = self._image.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self._cache_key = key
            painter.drawPixmap((self.width()-self._scaled.width())//2,
                               (self.height()-self._scaled.height())//2, self._scaled)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton: self.open_requested.emit()

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile() and Path(urls[0].toLocalFile()).suffix.lower() in {'.png','.jpg','.jpeg','.bmp'}:
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.file_dropped.emit(event.mimeData().urls()[0].toLocalFile())
        event.acceptProposedAction()


def label(text, name):
    item = QLabel(text)
    item.setObjectName(name)
    return item


def build_workbench(window, config):
    """Retain the original tabbed controls, with a smaller image and inline results."""
    legacy = window.takeCentralWidget()
    legacy.setParent(window)
    legacy.hide()
    window._optional_controls = legacy
    root = QWidget()
    root_layout = QVBoxLayout(root)
    root_layout.setContentsMargins(20,16,20,12)
    root_layout.setSpacing(16)
    header = QHBoxLayout()
    header.addWidget(label("眼科图像分类辅助诊断系统", "pageTitle"))
    header.addStretch()
    window.font_family_combo=QComboBox()
    available=set(QFontDatabase().families())
    for title,family in [("微软雅黑","Microsoft YaHei"),("等线","DengXian"),("思源黑体","Noto Sans SC")]:
        if family in available:window.font_family_combo.addItem(title,family)
    if not window.font_family_combo.count():window.font_family_combo.addItem("微软雅黑","Microsoft YaHei")
    window.font_family_combo.setToolTip("界面字体，切换后立即生效")
    window.font_family_combo.setParent(legacy);window.font_family_combo.hide()
    window.font_size_label=label("字号", "toolbarLabel")
    window.font_size_label.setParent(legacy);window.font_size_label.hide()
    window.font_size_combo = QComboBox()
    for title,size in [("标准 14",14),("大字 16",16),("特大 18",18)]:window.font_size_combo.addItem(title,size)
    window.font_size_combo.setParent(legacy);window.font_size_combo.hide()
    window.runtime_badge = label("模型未加载", "badge")
    window.runtime_badge.setAlignment(Qt.AlignCenter)
    window.runtime_badge.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Maximum)
    header.addWidget(window.runtime_badge)
    root_layout.addLayout(header)
    split=QSplitter(Qt.Horizontal)
    split.setChildrenCollapsible(False)
    split.setHandleWidth(12)
    window.workbench_splitter=split
    left=QWidget()
    left.setMinimumWidth(340)
    ll=QVBoxLayout(left)
    ll.setContentsMargins(0,0,0,0)
    ll.setSpacing(12)
    ll.addWidget(label("图像与操作", "sectionTitle"))
    # The camera page is the original page with all original signal connections.
    local=QWidget()
    local_layout=QVBoxLayout(local)
    local_layout.setContentsMargins(12,12,12,12)
    window.original_image_label=ImageCanvas()
    window.original_image_label.open_requested.connect(window.load_image)
    window.original_image_label.file_dropped.connect(window.load_image_path)
    local_layout.addWidget(window.original_image_label,1)
    meta=QHBoxLayout()
    window.image_meta_label=label("未选择图像","muted")
    window.image_meta_label.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
    window.image_size_label=label("— × — px","muted")
    meta.addWidget(window.image_meta_label,1);meta.addWidget(window.image_size_label)
    local_layout.addLayout(meta)
    old_local=window.image_tab_widget.widget(0)
    window.image_tab_widget.removeTab(0)
    old_local.setParent(legacy);old_local.hide()
    window.image_tab_widget.insertTab(0,local,"本地图像")
    window.image_tab_widget.setTabText(1,"开发板摄像头")
    from .local_camera import LocalCameraPage
    window.local_camera_page=LocalCameraPage(window)
    window.image_tab_widget.insertTab(1,window.local_camera_page,"本机摄像头")
    window.image_tab_widget.currentChanged.connect(lambda index:window.local_camera_page.stop() if index!=1 else None)
    window.image_tab_widget.setCurrentIndex(0)
    window.image_tab_widget.setMinimumWidth(0)
    window.image_tab_widget.setMaximumHeight(520)
    ll.addWidget(window.image_tab_widget,1)
    controls=QFrame();controls.setObjectName("card")
    grid=QGridLayout(controls);grid.setContentsMargins(14,14,14,14);grid.setSpacing(10)
    buttons=[(window.model_button,"加载模型"),(window.image_button,"加载图像"),
             (window.detect_button,"开始分类"),
             (window.board_interaction_button,"开发板交互"),(window.board_voice_button,"板端语音")]
    for index,(button,text) in enumerate(buttons):
        button.setText(text);button.setVisible(True);button.setMinimumHeight(42)
        if index < 2:
            grid.addWidget(button,0,index)
        elif index == 2:
            grid.addWidget(button,1,0,1,2)
        else:
            grid.addWidget(button,2,index-3)
    for button in (window.results_button, window.batch_button, window.history_button):
        button.hide()
    ll.addWidget(controls)
    status=QHBoxLayout()
    window.task_status_label=label("加载图像后开始分类","muted")
    window.task_status_label.setWordWrap(True)
    status.addWidget(window.task_status_label,1)
    window.cancel_task_button=QPushButton("取消")
    window.cancel_task_button.clicked.connect(window.cancel_current_task)
    window.cancel_task_button.hide();status.addWidget(window.cancel_task_button)
    root_layout.addLayout(status)
    window.task_progress=QProgressBar()
    window.task_progress.setFixedHeight(4);window.task_progress.setTextVisible(False);window.task_progress.hide()
    root_layout.addWidget(window.task_progress)
    split.addWidget(left)
    right=QWidget();right.setMinimumWidth(540)
    rl=QVBoxLayout(right);rl.setContentsMargins(0,0,0,0);rl.setSpacing(12)
    result=QFrame();result.setObjectName("card");window.result_card=result
    result_layout=QVBoxLayout(result);result_layout.setContentsMargins(20,16,20,16);result_layout.setSpacing(12)
    summary=QHBoxLayout()
    summary.addWidget(label("分类结果","sectionTitle"));summary.addStretch()
    window.inference_time_label=label("等待分析","muted");summary.addWidget(window.inference_time_label)
    result_layout.addLayout(summary)
    prediction=QHBoxLayout()
    window.classification_summary_label=label("尚未分类","prediction")
    window.classification_summary_label.setWordWrap(True)
    window.classification_score_label=label("—","score")
    prediction.addWidget(window.classification_summary_label,1)
    prediction.addWidget(label("模型概率","muted"));prediction.addWidget(window.classification_score_label)
    result_layout.addLayout(prediction)
    details=QWidget();details_layout=QVBoxLayout(details);details_layout.setContentsMargins(0,0,0,0)
    window.probability_details=details
    window.probability_toggle=QPushButton("收起八类概率")
    window.probability_toggle.setCheckable(True);window.probability_toggle.setChecked(True)
    summary.addWidget(window.probability_toggle)
    def toggle_details(checked):
        details.setVisible(checked)
        size=window.font_size_combo.currentData() or 14
        window.workbench_splitter.setMinimumHeight(round((560 if checked else 400)*size/14))
        window.probability_toggle.setText("收起八类概率" if checked else "展开八类概率")
    window.probability_toggle.toggled.connect(toggle_details)
    window.probability_hint=label("八类概率 · 分类后显示实际结果","muted");details_layout.addWidget(window.probability_hint)
    probabilities=QGridLayout();probabilities.setHorizontalSpacing(28);probabilities.setVerticalSpacing(8)
    window.probability_rows=[]
    for index in range(8):
        row=QHBoxLayout();row.setSpacing(8)
        name=label(list(NAMES.values())[index],"probName")
        name.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        bar=QProgressBar();bar.setRange(0,1000);bar.setValue(0);bar.setFixedHeight(7);bar.setTextVisible(False);bar.setObjectName("probabilityBar")
        score=label("—","muted");score.setAlignment(Qt.AlignRight|Qt.AlignVCenter);score.setMinimumWidth(66)
        row.addWidget(name,5);row.addWidget(bar,3);row.addWidget(score)
        probabilities.addLayout(row,index%4,index//4)
        window.probability_rows.append((name,bar,score))
    details_layout.addLayout(probabilities)
    result_layout.addWidget(details)
    notice=label("概率未经校准，不代表诊断准确率；分类结果仅供辅助参考。","notice")
    notice.setWordWrap(True)
    result_layout.addWidget(notice)
    rl.addWidget(result)
    # Original three tabs remain familiar; settings gain a scroll container.
    window.right_tab_widget.setTabText(0,"辅助建议")
    window.right_tab_widget.setTabText(1,"多轮问答")
    window.right_tab_widget.setTabText(2,"系统设置")
    settings=window.right_tab_widget.widget(2)
    window.right_tab_widget.removeTab(2)
    settings.setParent(legacy);settings.hide();window.connection_page=settings
    window.right_tab_widget.setMinimumHeight(300)
    window.right_tab_widget.currentChanged.connect(lambda index: window.probability_toggle.setChecked(False) if index==1 else None)
    window.chat_input.setMaximumHeight(72)
    from .conversation import install_chat_keys
    install_chat_keys(window)
    chat_layout=window.right_tab_widget.widget(1).layout()
    def empty_layout(layout):
        while layout.count():
            item=layout.takeAt(0)
            if item.layout():empty_layout(item.layout())
    empty_layout(chat_layout)
    chat_layout.addWidget(window.chat_display,1)
    chat_layout.addWidget(window.ai_progress_bar)
    composer=QHBoxLayout();composer.addWidget(window.chat_input,1)
    window.chat_input.setFixedHeight(58)
    composer.addWidget(window.send_chat_button)
    more=QPushButton('更多');more.setCheckable(True);composer.addWidget(more)
    chat_layout.addLayout(composer)
    extras=QWidget();extra_layout=QHBoxLayout(extras);extra_layout.setContentsMargins(0,0,0,0)
    for control in (window.voice_chat_enabled,window.voice_input_button,window.clear_chat_button):extra_layout.addWidget(control)
    extra_layout.addStretch();chat_layout.addWidget(extras);extras.hide()
    more.toggled.connect(extras.setVisible)
    more.toggled.connect(lambda checked:more.setText('收起' if checked else '更多'))
    reading_bar=QWidget();reading_layout=QHBoxLayout(reading_bar);reading_layout.setContentsMargins(0,0,0,0)
    reading_layout.addWidget(QLabel('阅读字号'))
    window.reading_size_combo=QComboBox()
    for value in (12,14,16,18,20,22,24):window.reading_size_combo.addItem(str(value),value)
    window.reading_size_combo.setCurrentIndex(1)
    reading_layout.addWidget(window.reading_size_combo)
    window.right_tab_widget.setCornerWidget(reading_bar,Qt.TopRightCorner)
    def reading_changed():
        size=window.reading_size_combo.currentData()
        for editor in (window.advice_text,window.chat_display):
            from .rich_text import apply_reading_font
            apply_reading_font(editor,size)
    window.reading_size_combo.currentIndexChanged.connect(reading_changed)
    window.apply_reading_size=reading_changed
    for editor in (window.advice_text,window.chat_display):
        editor.document().contentsChanged.connect(reading_changed)
    rl.addWidget(window.right_tab_widget,1)
    window.advice_button.setText("生成辅助建议");window.advice_button.setEnabled(False)
    window.fullscreen_advice_btn.setText("展开阅读")
    window.toggle_password_button.setText("显示")
    window.advice_text.setPlainText("完成分类后，点击「生成辅助建议」查看当前图像的说明。可在「多轮问答」继续提问，在顶部「设置」配置在线服务和语音。")
    window.voice_input_button.setText("语音输入")
    window.voice_chat_enabled.setText("自动朗读回复")
    for button in (window.voice_input_button,window.voice_chat_enabled):button.setVisible(True);button.setEnabled(config.enable_voice)
    for group in settings.findChildren(QGroupBox):group.setVisible(True)
    window.mic_test_button.setEnabled(config.enable_voice)
    window.connect_camera_button.setEnabled(config.enable_board)
    for button in (window.board_interaction_button,window.board_voice_button):
        button.setEnabled(config.enable_board)
        button.setToolTip("开发板功能；可在运行与网络设置中启用")
    split.addWidget(right);split.setStretchFactor(0,3);split.setStretchFactor(1,7);split.setSizes([400,900])
    window.workspace_pages=QTabWidget()
    single_scroll=QScrollArea();single_scroll.setWidgetResizable(True);single_scroll.setFrameShape(QFrame.NoFrame);single_scroll.setWidget(split)
    window.workspace_pages.addTab(single_scroll,"单图分析")
    settings_button=QPushButton("系统设置")
    settings_button.clicked.connect(window.show_runtime_settings)
    header.addWidget(settings_button)
    from .analysis_pages import AnalysisPage
    import sys
    app=sys.modules[window.__class__.__module__]
    db=app.get_history_db()
    window.batch_page=AnalysisPage(window,db,"batch")
    window.history_page=AnalysisPage(window,db,"history")
    window.workspace_pages.addTab(window.batch_page,"批量分析")
    window.workspace_pages.addTab(window.history_page,"历史记录")
    window.workspace_pages.currentChanged.connect(lambda index:window.history_page.refresh() if index==2 else None)
    window.batch_output=QTextEdit();window.batch_output.hide()
    root_layout.insertWidget(1,window.workspace_pages,1)
    window.setCentralWidget(root)
    from configs.runtime_config import PROJECT_ROOT
    window._ui_preferences_path=PROJECT_ROOT/"configs"/"ui_preferences.json"
    try:
        import json
        preferences=json.loads(window._ui_preferences_path.read_text(encoding="utf-8"))
        size=preferences.get("font_size",14)
        window.font_family_combo.setCurrentIndex(max(0,window.font_family_combo.findData(preferences.get("font_family","Microsoft YaHei"))))
    except (OSError,ValueError):size=14
    window.font_size_combo.setCurrentIndex(max(0,window.font_size_combo.findData(size)))
    window.font_size_combo.currentIndexChanged.connect(lambda _:change_font_size(window))
    window.font_family_combo.currentIndexChanged.connect(lambda _:change_font_size(window))


def update_probability_panel(window, probabilities=None):
    probabilities=probabilities or {}
    ranked=sorted(probabilities.items(),key=lambda x:x[1],reverse=True) if probabilities else [(name,None) for name in NAMES]
    for (name,bar,score),(disease,value) in zip(window.probability_rows,ranked):
        name.setText(NAMES.get(disease,disease));name.setToolTip(disease+" / "+NAMES.get(disease,disease))
        bar.setValue(round(value*1000) if value is not None else 0)
        score.setText(f"{value:.1%}" if value is not None else "—")
    window.probability_hint.setText("八类概率 · 按模型概率排序" if probabilities else "八类概率 · 分类后显示实际结果")


def change_font_size(window):
    import json
    from .workbench_theme import apply_workbench_theme
    apply_workbench_theme(window)
    try:
        path=window._ui_preferences_path
        settings=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        settings["font_size"]=window.font_size_combo.currentData()
        settings["font_family"]=window.font_family_combo.currentData()
        path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding="utf-8")
    except (OSError,ValueError) as error:
        window.status_bar.showMessage(f"字号已调整，偏好保存失败：{error}")
