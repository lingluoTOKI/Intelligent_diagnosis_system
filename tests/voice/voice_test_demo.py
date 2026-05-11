#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语音功能演示脚本
展示语音识别和合成功能的使用方法
"""

import sys
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QTextEdit, QMessageBox,
                             QProgressBar, QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

# 导入语音管理器
from enhanced_voice_recognition import VoiceManager

class VoiceDemoWindow(QMainWindow):
    """语音功能演示窗口"""
    
    def __init__(self):
        super().__init__()
        self.voice_manager = VoiceManager()
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("AI眼科疾病智诊系统 - 语音功能演示")
        self.setGeometry(100, 100, 800, 600)
        
        # 设置深色主题
        self.set_dark_theme()
        
        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("🎤 语音功能演示")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #4299e1; margin: 10px;")
        layout.addWidget(title_label)
        
        # 语音识别区域
        recognition_group = QGroupBox("语音识别")
        recognition_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2a3441;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #e2e8f0;
            }
        """)
        
        recognition_layout = QVBoxLayout(recognition_group)
        
        # 语音识别按钮
        self.voice_button = QPushButton("🎤 开始语音识别")
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #4299e1;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3182ce;
            }
            QPushButton:pressed {
                background-color: #2b6cb0;
            }
            QPushButton:disabled {
                background-color: #4a5568;
                color: #a0aec0;
            }
        """)
        self.voice_button.clicked.connect(self.start_voice_recognition)
        recognition_layout.addWidget(self.voice_button)
        
        # 识别结果显示
        self.result_label = QLabel("识别结果将显示在这里...")
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #2a3441;
                border: 1px solid #4a5568;
                border-radius: 4px;
                padding: 10px;
                color: #e2e8f0;
                font-size: 12px;
            }
        """)
        self.result_label.setWordWrap(True)
        recognition_layout.addWidget(self.result_label)
        
        layout.addWidget(recognition_group)
        
        # 语音合成区域
        tts_group = QGroupBox("语音合成")
        tts_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2a3441;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #e2e8f0;
            }
        """)
        
        tts_layout = QVBoxLayout(tts_group)
        
        # 文本输入
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("输入要转换为语音的文本...")
        self.text_input.setMaximumHeight(100)
        self.text_input.setStyleSheet("""
            QTextEdit {
                background-color: #2a3441;
                border: 1px solid #4a5568;
                border-radius: 4px;
                color: #e2e8f0;
                padding: 8px;
                font-size: 12px;
            }
        """)
        tts_layout.addWidget(self.text_input)
        
        # 语音播放按钮
        self.speak_button = QPushButton("🔊 播放语音")
        self.speak_button.setStyleSheet("""
            QPushButton {
                background-color: #48bb78;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #38a169;
            }
            QPushButton:pressed {
                background-color: #2f855a;
            }
            QPushButton:disabled {
                background-color: #4a5568;
                color: #a0aec0;
            }
        """)
        self.speak_button.clicked.connect(self.speak_text)
        tts_layout.addWidget(self.speak_button)
        
        layout.addWidget(tts_group)
        
        # 状态显示区域
        status_group = QGroupBox("系统状态")
        status_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2a3441;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #e2e8f0;
            }
        """)
        
        status_layout = QVBoxLayout(status_group)
        
        # 状态标签
        self.status_label = QLabel("系统就绪")
        self.status_label.setStyleSheet("color: #e2e8f0; font-size: 12px;")
        status_layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #4a5568;
                border-radius: 4px;
                text-align: center;
                background-color: #2a3441;
                color: #e2e8f0;
            }
            QProgressBar::chunk {
                background-color: #4299e1;
                border-radius: 3px;
            }
        """)
        status_layout.addWidget(self.progress_bar)
        
        layout.addWidget(status_group)
        
        # 测试按钮区域
        test_layout = QHBoxLayout()
        
        # 麦克风测试按钮
        self.test_mic_button = QPushButton("🔧 测试麦克风")
        self.test_mic_button.setStyleSheet("""
            QPushButton {
                background-color: #ed8936;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #dd6b20;
            }
        """)
        self.test_mic_button.clicked.connect(self.test_microphone)
        test_layout.addWidget(self.test_mic_button)
        
        # 状态检查按钮
        self.status_button = QPushButton("📊 检查状态")
        self.status_button.setStyleSheet("""
            QPushButton {
                background-color: #9f7aea;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #805ad5;
            }
        """)
        self.status_button.clicked.connect(self.check_status)
        test_layout.addWidget(self.status_button)
        
        test_layout.addStretch()
        layout.addLayout(test_layout)
        
    def set_dark_theme(self):
        """设置深色主题"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#1a202c"))
        palette.setColor(QPalette.WindowText, QColor("#e2e8f0"))
        palette.setColor(QPalette.Base, QColor("#2a3441"))
        palette.setColor(QPalette.AlternateBase, QColor("#2a3441"))
        palette.setColor(QPalette.ToolTipBase, QColor("#2a3441"))
        palette.setColor(QPalette.ToolTipText, QColor("#e2e8f0"))
        palette.setColor(QPalette.Text, QColor("#e2e8f0"))
        palette.setColor(QPalette.Button, QColor("#2a3441"))
        palette.setColor(QPalette.ButtonText, QColor("#e2e8f0"))
        palette.setColor(QPalette.BrightText, QColor("#4299e1"))
        palette.setColor(QPalette.Link, QColor("#4299e1"))
        palette.setColor(QPalette.Highlight, QColor("#4299e1"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        
        self.setPalette(palette)
        
    def connect_signals(self):
        """连接信号"""
        # 语音识别信号
        self.voice_manager.voice_recognized.connect(self.on_voice_recognized)
        self.voice_manager.voice_error.connect(self.on_voice_error)
        self.voice_manager.voice_timeout.connect(self.on_voice_timeout)
        self.voice_manager.voice_unknown.connect(self.on_voice_unknown)
        
        # TTS信号
        self.voice_manager.tts_started.connect(self.on_tts_started)
        self.voice_manager.tts_finished.connect(self.on_tts_finished)
        self.voice_manager.tts_error.connect(self.on_tts_error)
        
    def start_voice_recognition(self):
        """开始语音识别"""
        self.voice_button.setText("🎤 正在录音...")
        self.voice_button.setEnabled(False)
        self.status_label.setText("正在录音，请说话...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        
        self.voice_manager.start_voice_recognition()
        
    def on_voice_recognized(self, text):
        """语音识别成功"""
        self.voice_button.setText("🎤 开始语音识别")
        self.voice_button.setEnabled(True)
        self.result_label.setText(f"识别结果: {text}")
        self.status_label.setText("语音识别成功")
        self.progress_bar.setVisible(False)
        
        # 自动填入文本输入框
        self.text_input.setPlainText(text)
        
    def on_voice_error(self, error):
        """语音识别错误"""
        self.voice_button.setText("🎤 开始语音识别")
        self.voice_button.setEnabled(True)
        self.status_label.setText("语音识别失败")
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "错误", f"语音识别失败：{error}")
        
    def on_voice_timeout(self):
        """语音识别超时"""
        self.voice_button.setText("🎤 开始语音识别")
        self.voice_button.setEnabled(True)
        self.status_label.setText("录音超时")
        self.progress_bar.setVisible(False)
        
        QMessageBox.information(self, "提示", "录音超时，请重试")
        
    def on_voice_unknown(self):
        """语音无法识别"""
        self.voice_button.setText("🎤 开始语音识别")
        self.voice_button.setEnabled(True)
        self.status_label.setText("无法识别语音")
        self.progress_bar.setVisible(False)
        
        QMessageBox.information(self, "提示", "无法识别您的语音，请清晰地说话后重试")
        
    def speak_text(self):
        """播放文本语音"""
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "警告", "请输入要播放的文本")
            return
            
        self.speak_button.setText("🔊 正在播放...")
        self.speak_button.setEnabled(False)
        self.status_label.setText("正在播放语音...")
        
        self.voice_manager.speak_text(text)
        
    def on_tts_started(self):
        """TTS开始"""
        self.status_label.setText("开始播放语音...")
        
    def on_tts_finished(self):
        """TTS完成"""
        self.speak_button.setText("🔊 播放语音")
        self.speak_button.setEnabled(True)
        self.status_label.setText("语音播放完成")
        
    def on_tts_error(self, error):
        """TTS错误"""
        self.speak_button.setText("🔊 播放语音")
        self.speak_button.setEnabled(True)
        self.status_label.setText("语音播放失败")
        
        QMessageBox.critical(self, "错误", f"语音播放失败：{error}")
        
    def test_microphone(self):
        """测试麦克风"""
        success, message = self.voice_manager.test_microphone()
        
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.critical(self, "错误", message)
            
    def check_status(self):
        """检查语音功能状态"""
        status = self.voice_manager.get_voice_status()
        
        status_text = f"""
语音功能状态检查：

✅ 语音识别器: {'可用' if status['recognizer_available'] else '不可用'}
✅ 麦克风: {'可用' if status['microphone_available'] else '不可用'}
✅ TTS引擎: {'可用' if status['tts_available'] else '不可用'}
🎤 正在录音: {'是' if status['is_listening'] else '否'}
🔊 正在播放: {'是' if status['is_speaking'] else '否'}
        """
        
        QMessageBox.information(self, "状态检查", status_text.strip())

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("AI眼科疾病智诊系统 - 语音演示")
    app.setApplicationVersion("1.0")
    
    # 创建并显示窗口
    window = VoiceDemoWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 