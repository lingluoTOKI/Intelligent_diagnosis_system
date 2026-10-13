"""Quiet, scalable placeholder that retains QLabel video-frame behaviour."""
from pathlib import Path
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QIcon
from PyQt5.QtWidgets import QLabel

class CameraPreview(QLabel):
    def __init__(self):
        super().__init__("摄像头未连接")
        self._icon = QIcon(str(Path(__file__).resolve().parents[2] / "assets/brand/device.svg"))

    def paintEvent(self, event):
        pixmap = self.pixmap()
        if pixmap is not None and not pixmap.isNull():
            return super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1e252e"))
        height = self.fontMetrics().height()
        center = self.height() / 2
        painter.setOpacity(.65)
        painter.drawPixmap((self.width()-40)//2, int(center-height-48), self._icon.pixmap(40,40))
        painter.setOpacity(1)
        painter.setFont(self.font())
        painter.setPen(QColor("#adb8c7"))
        painter.drawText(QRectF(12,center-height,self.width()-24,height+6),Qt.AlignCenter,self.text())
        painter.setPen(QColor("#8292a5"))
        painter.drawText(QRectF(12,center+12,self.width()-24,height*2),Qt.AlignHCenter|Qt.AlignTop|Qt.TextWordWrap,"连接开发板后显示实时画面")
