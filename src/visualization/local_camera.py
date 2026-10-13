"""Optional native PC camera preview and capture; no device opens at startup."""
from pathlib import Path
import uuid
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QComboBox,QPushButton,QLabel

class LocalCameraPage(QWidget):
    def __init__(self,window):
        super().__init__();self.window=window;self.camera=None;self.capture=None
        layout=QVBoxLayout(self);bar=QHBoxLayout();layout.addLayout(bar)
        self.devices=QComboBox();bar.addWidget(self.devices,1)
        refresh=QPushButton('刷新设备');refresh.clicked.connect(self.refresh);bar.addWidget(refresh)
        self.status=QLabel('选择本机摄像头后打开预览');self.status.setWordWrap(True)
        try:
            from PyQt5.QtMultimediaWidgets import QCameraViewfinder
            self.preview=QCameraViewfinder();layout.addWidget(self.preview,1)
            self.available=True
        except ImportError:
            self.available=False
            layout.addWidget(QLabel('未安装 Qt 多媒体组件，本机摄像头暂不可用。'),1)
        layout.addWidget(self.status);actions=QHBoxLayout();layout.addLayout(actions)
        self.open_button=QPushButton('打开摄像头');self.open_button.clicked.connect(self.open_camera);actions.addWidget(self.open_button)
        stop=QPushButton('关闭预览');stop.clicked.connect(self.stop);actions.addWidget(stop)
        self.photo=QPushButton('拍照并载入');self.photo.setEnabled(False);self.photo.clicked.connect(self.take_photo);actions.addWidget(self.photo)
        hint=QLabel('拍照后切换到本地图像，点击“开始分类”。普通摄像头无法替代眼底成像设备。');hint.setWordWrap(True);layout.addWidget(hint)
        self.refresh()
    def refresh(self):
        self.stop();self.devices.clear()
        if not self.available:return
        from PyQt5.QtMultimedia import QCameraInfo
        for info in QCameraInfo.availableCameras():self.devices.addItem(info.description(),info)
        self.open_button.setEnabled(self.devices.count()>0)
        if not self.devices.count():self.status.setText('未发现摄像头；请连接设备，并检查 Windows 摄像头权限。')
    def stop(self):
        self.photo.setEnabled(False)
        if self.camera:
            self.camera.stop()
            self.camera.deleteLater();self.camera=None;self.capture=None
        self.status.setText('预览已关闭')
    def open_camera(self):
        self.stop()
        info=self.devices.currentData()
        if info is None:return
        from PyQt5.QtMultimedia import QCamera,QCameraImageCapture
        self.camera=QCamera(info,self);self.camera.setViewfinder(self.preview);self.camera.setCaptureMode(QCamera.CaptureStillImage)
        self.capture=QCameraImageCapture(self.camera)
        self.capture.readyForCaptureChanged.connect(self.photo.setEnabled)
        self.capture.imageSaved.connect(self.saved)
        self.capture.error.connect(lambda _,code,message:self.status.setText('拍照失败：'+message))
        self.camera.error.connect(lambda _:self.status.setText('摄像头错误：'+self.camera.errorString()) if self.camera else None)
        self.camera.start();self.status.setText('正在打开 '+info.description())
    def take_photo(self):
        if getattr(self.window,'_busy',False):self.status.setText('请等待当前任务完成再拍照');return
        if self.capture and self.capture.isReadyForCapture():
            folder=Path(__file__).resolve().parents[2]/'runtime'/'camera_captures';folder.mkdir(parents=True,exist_ok=True)
            self.capture.capture(str(folder/(uuid.uuid4().hex+'.jpg')))
    def saved(self,identifier,path):
        self.stop();self.window.load_image_path(path);self.window.image_tab_widget.setCurrentIndex(0)
