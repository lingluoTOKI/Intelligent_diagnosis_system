#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能语音识别演示程序
展示有网络时使用百度API，无网络时使用本地识别的功能
"""

import sys
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QTextEdit, QMessageBox,
                             QProgressBar, QCheckBox, QGroupBox, QTabWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

# 导入智能语音系统
from smart_voice_system import SmartVoiceManager, NetworkDetector

class SmartVoiceDemoWindow(QMainWindow):
    """智能语音识别演示窗口"""
    
    def __init__(self):
        super().__init__()
        self.voice_manager = SmartVoiceManager()
        self.init_ui()
        self.connect_signals()
        
        # 启动状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(2000)  # 每2秒更新一次状态
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("AI眼科疾病智诊系统 - 智能语音识别演示")
        self.setGeometry(100, 100, 900, 700)
        
        # 设置深色主题
        self.set_dark_theme()
        
        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("🎤 智能语音识别系统")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #4299e1; margin: 15px;")
        layout.addWidget(title_label)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #4a5568;
                background-color: #2a3441;
            }
            QTabBar::tab {
                background-color: #4a5568;
                color: #e2e8f0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4299e1;
            }
            QTabBar::tab:hover {
                background-color: #5a6c7d;
            }
        """)
        
        # 语音识别选项卡
        recognition_tab = self.create_recognition_tab()
        tab_widget.addTab(recognition_tab, "🎤 语音识别")
        
        # 系统状态选项卡
        status_tab = self.create_status_tab()
        tab_widget.addTab(status_tab, "📊 系统状态")
        
        # 网络监控选项卡
        network_tab = self.create_network_tab()
        tab_widget.addTab(network_tab, "🌐 网络监控")
        
        layout.addWidget(tab_widget)
        
    def create_recognition_tab(self):
        """创建语音识别选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 识别模式显示
        mode_group = QGroupBox("识别模式")
        mode_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #48bb78;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #48bb78;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_label = QLabel("检测中...")
        self.mode_label.setStyleSheet("""
            QLabel {
                background-color: #2a3441;
                border: 1px solid #4a5568;
                border-radius: 4px;
                padding: 10px;
                color: #e2e8f0;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.mode_label.setAlignment(Qt.AlignCenter)
        mode_layout.addWidget(self.mode_label)
        
        layout.addWidget(mode_group)
        
        # 语音识别区域
        recognition_group = QGroupBox("语音识别")
        recognition_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2a3441;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #e2e8f0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
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
                color: #e2e8f0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
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
        layout.addStretch()
        
        return widget
        
    def create_status_tab(self):
        """创建系统状态选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 系统状态显示
        status_group = QGroupBox("系统状态")
        status_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2a3441;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #e2e8f0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        status_layout = QVBoxLayout(status_group)
        
        # 状态标签
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a202c;
                border: 1px solid #4a5568;
                border-radius: 4px;
                color: #e2e8f0;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        status_layout.addWidget(self.status_text)
        
        layout.addWidget(status_group)
        
        # 测试按钮
        test_group = QGroupBox("功能测试")
        test_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2a3441;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #e2e8f0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        test_layout = QHBoxLayout(test_group)
        
        # 麦克风测试按钮
        self.test_mic_button = QPushButton("🔧 测试麦克风")
        self.test_mic_button.setStyleSheet("""
            QPushButton {
                background-color: #ed8936;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #dd6b20;
            }
        """)
        self.test_mic_button.clicked.connect(self.test_microphone)
        test_layout.addWidget(self.test_mic_button)
        
        # 网络测试按钮
        self.test_network_button = QPushButton("🌐 测试网络")
        self.test_network_button.setStyleSheet("""
            QPushButton {
                background-color: #9f7aea;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #805ad5;
            }
        """)
        self.test_network_button.clicked.connect(self.test_network)
        test_layout.addWidget(self.test_network_button)
        
        test_layout.addStretch()
        layout.addWidget(test_group)
        
        return widget
        
    def create_network_tab(self):
        """创建网络监控选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 网络状态显示
        network_group = QGroupBox("网络监控")
        network_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2a3441;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #e2e8f0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        network_layout = QVBoxLayout(network_group)
        
        # 网络状态标签
        self.network_status_label = QLabel("检测中...")
        self.network_status_label.setStyleSheet("""
            QLabel {
                background-color: #2a3441;
                border: 1px solid #4a5568;
                border-radius: 4px;
                padding: 10px;
                color: #e2e8f0;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.network_status_label.setAlignment(Qt.AlignCenter)
        network_layout.addWidget(self.network_status_label)
        
        # 网络日志
        self.network_log = QTextEdit()
        self.network_log.setReadOnly(True)
        self.network_log.setMaximumHeight(200)
        self.network_log.setStyleSheet("""
            QTextEdit {
                background-color: #1a202c;
                border: 1px solid #4a5568;
                border-radius: 4px;
                color: #e2e8f0;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        network_layout.addWidget(self.network_log)
        
        layout.addWidget(network_group)
        layout.addStretch()
        
        return widget
        
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
        
        # 网络状态信号
        self.voice_manager.network_status_changed.connect(self.on_network_status_changed)
        
        # TTS信号
        self.voice_manager.tts_started.connect(self.on_tts_started)
        self.voice_manager.tts_finished.connect(self.on_tts_finished)
        self.voice_manager.tts_error.connect(self.on_tts_error)
        
    def start_voice_recognition(self):
        """开始语音识别"""
        self.voice_button.setText("🎤 正在录音...")
        self.voice_button.setEnabled(False)
        self.result_label.setText("正在录音，请说话...")
        
        self.voice_manager.start_voice_recognition()
        
    def on_voice_recognized(self, text):
        """语音识别成功"""
        self.voice_button.setText("🎤 开始语音识别")
        self.voice_button.setEnabled(True)
        self.result_label.setText(f"识别结果: {text}")
        
        # 自动填入文本输入框
        self.text_input.setPlainText(text)
        
    def on_voice_error(self, error):
        """语音识别错误"""
        self.voice_button.setText("🎤 开始语音识别")
        self.voice_button.setEnabled(True)
        self.result_label.setText("识别失败")
        
        QMessageBox.critical(self, "错误", f"语音识别失败：{error}")
        
    def on_voice_timeout(self):
        """语音识别超时"""
        self.voice_button.setText("🎤 开始语音识别")
        self.voice_button.setEnabled(True)
        self.result_label.setText("录音超时")
        
        QMessageBox.information(self, "提示", "录音超时，请重试")
        
    def on_voice_unknown(self):
        """语音无法识别"""
        self.voice_button.setText("🎤 开始语音识别")
        self.voice_button.setEnabled(True)
        self.result_label.setText("无法识别语音")
        
        QMessageBox.information(self, "提示", "无法识别您的语音，请清晰地说话后重试")
        
    def on_network_status_changed(self, is_online, message):
        """网络状态变化"""
        if is_online:
            self.mode_label.setText("🌐 在线模式 - 使用百度API识别")
            self.mode_label.setStyleSheet("""
                QLabel {
                    background-color: #2a3441;
                    border: 2px solid #48bb78;
                    border-radius: 4px;
                    padding: 10px;
                    color: #48bb78;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
        else:
            self.mode_label.setText("💻 离线模式 - 使用本地识别")
            self.mode_label.setStyleSheet("""
                QLabel {
                    background-color: #2a3441;
                    border: 2px solid #ed8936;
                    border-radius: 4px;
                    padding: 10px;
                    color: #ed8936;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
        
        # 添加到网络日志
        timestamp = time.strftime("%H:%M:%S")
        self.network_log.append(f"[{timestamp}] {message}")
        
    def speak_text(self):
        """播放文本语音"""
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "警告", "请输入要播放的文本")
            return
            
        self.speak_button.setText("🔊 正在播放...")
        self.speak_button.setEnabled(False)
        
        self.voice_manager.speak_text(text)
        
    def on_tts_started(self):
        """TTS开始"""
        self.result_label.setText("开始播放语音...")
        
    def on_tts_finished(self):
        """TTS完成"""
        self.speak_button.setText("🔊 播放语音")
        self.speak_button.setEnabled(True)
        self.result_label.setText("语音播放完成")
        
    def on_tts_error(self, error):
        """TTS错误"""
        self.speak_button.setText("🔊 播放语音")
        self.speak_button.setEnabled(True)
        self.result_label.setText("语音播放失败")
        
        QMessageBox.critical(self, "错误", f"语音播放失败：{error}")
        
    def test_microphone(self):
        """测试麦克风"""
        success, message = self.voice_manager.test_microphone()
        
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.critical(self, "错误", message)
            
    def test_network(self):
        """测试网络"""
        is_online, status = NetworkDetector.check_network()
        
        if is_online:
            QMessageBox.information(self, "成功", f"网络连接正常: {status}")
        else:
            QMessageBox.warning(self, "警告", f"网络连接异常: {status}")
            
    def update_status(self):
        """更新状态显示"""
        status = self.voice_manager.get_voice_status()
        
        status_text = f"""
系统状态信息:

✅ 语音识别器: {'可用' if status['recognizer_available'] else '不可用'}
✅ 麦克风: {'可用' if status['microphone_available'] else '不可用'}
✅ TTS引擎: {'可用' if status['tts_available'] else '不可用'}
✅ 百度API: {'可用' if status['baidu_api_available'] else '不可用'}
✅ 本地识别: {'可用' if status['local_recognizer_available'] else '不可用'}

🎤 当前模式: {status['current_mode']}
🎤 正在录音: {'是' if status['is_listening'] else '否'}
🔊 正在播放: {'是' if status['is_speaking'] else '否'}

最后更新: {time.strftime("%H:%M:%S")}
        """
        
        self.status_text.setPlainText(status_text.strip())
        
        # 更新网络状态
        is_online, network_status = NetworkDetector.check_network()
        if is_online:
            self.network_status_label.setText("🌐 网络连接正常")
            self.network_status_label.setStyleSheet("""
                QLabel {
                    background-color: #2a3441;
                    border: 2px solid #48bb78;
                    border-radius: 4px;
                    padding: 10px;
                    color: #48bb78;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
        else:
            self.network_status_label.setText("❌ 网络连接异常")
            self.network_status_label.setStyleSheet("""
                QLabel {
                    background-color: #2a3441;
                    border: 2px solid #e53e3e;
                    border-radius: 4px;
                    padding: 10px;
                    color: #e53e3e;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("AI眼科疾病智诊系统 - 智能语音演示")
    app.setApplicationVersion("1.0")
    
    # 创建并显示窗口
    window = SmartVoiceDemoWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 