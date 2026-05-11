#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能语音识别系统集成测试
测试智能语音识别系统在主程序中的集成情况
"""

import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QTextEdit, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# 导入智能语音系统
from smart_voice_system import SmartVoiceManager, NetworkDetector

class SmartVoiceIntegrationTest(QMainWindow):
    """智能语音识别系统集成测试窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能语音识别系统集成测试")
        self.setGeometry(100, 100, 600, 500)
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a202c;
                color: #e2e8f0;
            }
            QLabel {
                color: #e2e8f0;
                font-family: 'Microsoft YaHei', sans-serif;
            }
            QPushButton {
                background-color: #4299e1;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3182ce;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
            QTextEdit {
                background-color: #2a3441;
                color: #e2e8f0;
                border: 1px solid #4299e1;
                border-radius: 6px;
                padding: 10px;
                font-size: 12px;
            }
        """)
        
        # 初始化智能语音管理器
        self.voice_manager = SmartVoiceManager()
        
        # 初始化UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
        
        # 启动网络监控
        self.voice_manager.start_network_monitoring()
        
        # 状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(2000)  # 每2秒更新一次状态
        
    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title = QLabel("🎤 智能语音识别系统集成测试")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #4299e1; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # 状态显示
        self.status_label = QLabel("正在检测系统状态...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            padding: 10px;
            background-color: #2a3441;
            border-radius: 6px;
            border: 1px solid #4299e1;
            font-weight: bold;
        """)
        layout.addWidget(self.status_label)
        
        # 网络状态
        self.network_label = QLabel("网络状态: 检测中...")
        self.network_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.network_label)
        
        # 语音识别模式
        self.mode_label = QLabel("识别模式: 检测中...")
        self.mode_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.mode_label)
        
        # 测试按钮
        self.test_voice_button = QPushButton("🎤 测试语音识别")
        self.test_voice_button.clicked.connect(self.test_voice_recognition)
        layout.addWidget(self.test_voice_button)
        
        # 测试麦克风按钮
        self.test_mic_button = QPushButton("🔧 测试麦克风")
        self.test_mic_button.clicked.connect(self.test_microphone)
        layout.addWidget(self.test_mic_button)
        
        # 测试TTS按钮
        self.test_tts_button = QPushButton("🔊 测试语音合成")
        self.test_tts_button.clicked.connect(self.test_tts)
        layout.addWidget(self.test_tts_button)
        
        # 日志显示
        self.log_text = QTextEdit()
        self.log_text.setPlaceholderText("系统日志将显示在这里...")
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)
        
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
        
    def update_status(self):
        """更新状态显示"""
        try:
            status = self.voice_manager.get_voice_status()
            
            # 更新网络状态
            if status.get('network_online', False):
                self.network_label.setText("🌐 网络状态: 在线")
                self.network_label.setStyleSheet("color: #48bb78; font-weight: bold;")
            else:
                self.network_label.setText("💻 网络状态: 离线")
                self.network_label.setStyleSheet("color: #ed8936; font-weight: bold;")
            
            # 更新识别模式
            current_mode = status.get('current_mode', '未知')
            if current_mode == 'online':
                self.mode_label.setText("🌐 识别模式: 百度API")
                self.mode_label.setStyleSheet("color: #4299e1; font-weight: bold;")
            elif current_mode == 'offline':
                self.mode_label.setText("💻 识别模式: 本地识别")
                self.mode_label.setStyleSheet("color: #ed64a6; font-weight: bold;")
            else:
                self.mode_label.setText("❓ 识别模式: 未知")
                self.mode_label.setStyleSheet("color: #a0aec0; font-weight: bold;")
                
        except Exception as e:
            self.log_message(f"状态更新错误: {e}")
            
    def test_voice_recognition(self):
        """测试语音识别"""
        self.log_message("开始测试语音识别...")
        self.test_voice_button.setEnabled(False)
        self.test_voice_button.setText("🎤 正在录音...")
        
        # 启动语音识别
        self.voice_manager.start_voice_recognition()
        
    def test_microphone(self):
        """测试麦克风"""
        self.log_message("开始测试麦克风...")
        success, message = self.voice_manager.test_microphone()
        
        if success:
            self.log_message(f"麦克风测试成功: {message}")
            QMessageBox.information(self, "测试成功", message)
        else:
            self.log_message(f"麦克风测试失败: {message}")
            QMessageBox.critical(self, "测试失败", message)
            
    def test_tts(self):
        """测试语音合成"""
        self.log_message("开始测试语音合成...")
        test_text = "智能语音识别系统测试成功！"
        self.voice_manager.speak_text(test_text)
        
    def on_voice_recognized(self, text):
        """语音识别成功"""
        self.log_message(f"语音识别成功: {text}")
        self.test_voice_button.setEnabled(True)
        self.test_voice_button.setText("🎤 测试语音识别")
        QMessageBox.information(self, "识别成功", f"识别结果: {text}")
        
    def on_voice_error(self, error):
        """语音识别错误"""
        self.log_message(f"语音识别错误: {error}")
        self.test_voice_button.setEnabled(True)
        self.test_voice_button.setText("🎤 测试语音识别")
        QMessageBox.critical(self, "识别失败", f"错误信息: {error}")
        
    def on_voice_timeout(self):
        """语音识别超时"""
        self.log_message("语音识别超时")
        self.test_voice_button.setEnabled(True)
        self.test_voice_button.setText("🎤 测试语音识别")
        QMessageBox.warning(self, "识别超时", "录音超时，请重试")
        
    def on_voice_unknown(self):
        """语音无法识别"""
        self.log_message("无法识别语音")
        self.test_voice_button.setEnabled(True)
        self.test_voice_button.setText("🎤 测试语音识别")
        QMessageBox.warning(self, "识别失败", "无法识别您的语音，请清晰地说话后重试")
        
    def on_network_status_changed(self, is_online, message):
        """网络状态变化"""
        if is_online:
            self.log_message(f"🌐 网络状态变化: {message}")
        else:
            self.log_message(f"💻 网络状态变化: {message}")
            
    def on_tts_started(self):
        """TTS开始"""
        self.log_message("开始播放语音...")
        
    def on_tts_finished(self):
        """TTS完成"""
        self.log_message("语音播放完成")
        
    def on_tts_error(self, error):
        """TTS错误"""
        self.log_message(f"语音播放错误: {error}")
        QMessageBox.critical(self, "TTS错误", f"语音播放失败: {error}")
        
    def log_message(self, message):
        """记录日志消息"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("智能语音识别系统集成测试")
    app.setApplicationVersion("1.0.0")
    
    # 创建并显示主窗口
    window = SmartVoiceIntegrationTest()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 