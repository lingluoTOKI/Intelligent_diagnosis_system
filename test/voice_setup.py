#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语音功能安装和配置脚本
帮助用户快速安装和配置语音识别功能
"""

import sys
import subprocess
import os
import platform
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QTextEdit, QMessageBox,
                             QProgressBar, QCheckBox, QGroupBox, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

class InstallWorker(QThread):
    """安装工作线程"""
    
    progress_updated = pyqtSignal(int, str)
    installation_completed = pyqtSignal(bool, str)
    
    def __init__(self, packages):
        super().__init__()
        self.packages = packages
        
    def run(self):
        """运行安装任务"""
        try:
            total_packages = len(self.packages)
            
            for i, package in enumerate(self.packages):
                self.progress_updated.emit(
                    int((i / total_packages) * 100),
                    f"正在安装 {package}..."
                )
                
                # 安装包
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.installation_completed.emit(False, f"安装 {package} 失败: {result.stderr}")
                    return
                    
            self.progress_updated.emit(100, "安装完成")
            self.installation_completed.emit(True, "所有包安装成功")
            
        except Exception as e:
            self.installation_completed.emit(False, f"安装过程异常: {str(e)}")

class VoiceSetupWindow(QMainWindow):
    """语音功能设置窗口"""
    
    def __init__(self):
        super().__init__()
        self.install_worker = None
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("AI眼科疾病智诊系统 - 语音功能设置")
        self.setGeometry(100, 100, 900, 700)
        
        # 设置深色主题
        self.set_dark_theme()
        
        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("🎤 语音功能设置向导")
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
        
        # 安装选项卡
        install_tab = self.create_install_tab()
        tab_widget.addTab(install_tab, "📦 依赖安装")
        
        # 测试选项卡
        test_tab = self.create_test_tab()
        tab_widget.addTab(test_tab, "🧪 功能测试")
        
        # 配置选项卡
        config_tab = self.create_config_tab()
        tab_widget.addTab(config_tab, "⚙️ 系统配置")
        
        layout.addWidget(tab_widget)
        
    def create_install_tab(self):
        """创建安装选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 说明文字
        info_label = QLabel("""
本向导将帮助您安装语音功能所需的依赖包。

需要安装的包：
• SpeechRecognition - 语音识别库
• pyttsx3 - 文字转语音库
• pyaudio - 音频处理库

请点击"开始安装"按钮开始安装过程。
        """)
        info_label.setStyleSheet("""
            QLabel {
                background-color: #2a3441;
                border: 1px solid #4a5568;
                border-radius: 6px;
                padding: 15px;
                color: #e2e8f0;
                font-size: 12px;
            }
        """)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 安装按钮
        self.install_button = QPushButton("🚀 开始安装")
        self.install_button.setStyleSheet("""
            QPushButton {
                background-color: #4299e1;
                color: white;
                border: none;
                padding: 12px 24px;
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
        self.install_button.clicked.connect(self.start_installation)
        layout.addWidget(self.install_button)
        
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
        layout.addWidget(self.progress_bar)
        
        # 日志显示
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("""
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
        layout.addWidget(self.log_text)
        
        layout.addStretch()
        return widget
        
    def create_test_tab(self):
        """创建测试选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 测试说明
        test_info = QLabel("""
测试语音功能是否正常工作。

测试项目：
• 语音识别库导入
• 麦克风访问
• TTS引擎初始化
• 网络连接（语音识别需要）

请点击"开始测试"按钮进行功能测试。
        """)
        test_info.setStyleSheet("""
            QLabel {
                background-color: #2a3441;
                border: 1px solid #4a5568;
                border-radius: 6px;
                padding: 15px;
                color: #e2e8f0;
                font-size: 12px;
            }
        """)
        test_info.setWordWrap(True)
        layout.addWidget(test_info)
        
        # 测试按钮
        self.test_button = QPushButton("🧪 开始测试")
        self.test_button.setStyleSheet("""
            QPushButton {
                background-color: #48bb78;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #38a169;
            }
            QPushButton:pressed {
                background-color: #2f855a;
            }
        """)
        self.test_button.clicked.connect(self.run_tests)
        layout.addWidget(self.test_button)
        
        # 测试结果显示
        self.test_result = QTextEdit()
        self.test_result.setStyleSheet("""
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
        layout.addWidget(self.test_result)
        
        return widget
        
    def create_config_tab(self):
        """创建配置选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 配置说明
        config_info = QLabel("""
系统配置和优化建议。

配置项目：
• 音频设备选择
• 语音识别参数调整
• TTS语音设置
• 网络代理配置（如需要）

请根据您的系统环境进行相应配置。
        """)
        config_info.setStyleSheet("""
            QLabel {
                background-color: #2a3441;
                border: 1px solid #4a5568;
                border-radius: 6px;
                padding: 15px;
                color: #e2e8f0;
                font-size: 12px;
            }
        """)
        config_info.setWordWrap(True)
        layout.addWidget(config_info)
        
        # 配置选项
        config_group = QGroupBox("配置选项")
        config_group.setStyleSheet("""
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
        
        config_layout = QVBoxLayout(config_group)
        
        # 音频设备检测
        self.detect_audio_button = QPushButton("🔍 检测音频设备")
        self.detect_audio_button.setStyleSheet("""
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
        self.detect_audio_button.clicked.connect(self.detect_audio_devices)
        config_layout.addWidget(self.detect_audio_button)
        
        # 网络测试
        self.test_network_button = QPushButton("🌐 测试网络连接")
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
        config_layout.addWidget(self.test_network_button)
        
        # 配置结果显示
        self.config_result = QTextEdit()
        self.config_result.setMaximumHeight(150)
        self.config_result.setStyleSheet("""
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
        config_layout.addWidget(self.config_result)
        
        layout.addWidget(config_group)
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
        
    def start_installation(self):
        """开始安装"""
        packages = ["SpeechRecognition", "pyttsx3", "pyaudio"]
        
        self.install_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        self.install_worker = InstallWorker(packages)
        self.install_worker.progress_updated.connect(self.update_progress)
        self.install_worker.installation_completed.connect(self.installation_finished)
        self.install_worker.start()
        
    def update_progress(self, value, message):
        """更新进度"""
        self.progress_bar.setValue(value)
        self.log_text.append(f"[{value}%] {message}")
        
    def installation_finished(self, success, message):
        """安装完成"""
        self.install_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.log_text.append(f"✅ {message}")
            QMessageBox.information(self, "成功", "依赖包安装完成！")
        else:
            self.log_text.append(f"❌ {message}")
            QMessageBox.critical(self, "错误", f"安装失败：{message}")
            
    def run_tests(self):
        """运行测试"""
        self.test_button.setEnabled(False)
        self.test_result.clear()
        
        # 运行测试
        tests = [
            ("语音识别库", self.test_speech_recognition),
            ("TTS引擎", self.test_tts_engine),
            ("音频设备", self.test_audio_devices),
            ("网络连接", self.test_network_connection)
        ]
        
        for test_name, test_func in tests:
            self.test_result.append(f"🧪 测试 {test_name}...")
            try:
                result = test_func()
                if result:
                    self.test_result.append(f"✅ {test_name} 测试通过")
                else:
                    self.test_result.append(f"❌ {test_name} 测试失败")
            except Exception as e:
                self.test_result.append(f"❌ {test_name} 测试异常: {e}")
                
        self.test_button.setEnabled(True)
        self.test_result.append("\n🎉 测试完成！")
        
    def test_speech_recognition(self):
        """测试语音识别库"""
        try:
            import speech_recognition as sr
            return True
        except ImportError:
            return False
            
    def test_tts_engine(self):
        """测试TTS引擎"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            return True
        except Exception:
            return False
            
    def test_audio_devices(self):
        """测试音频设备"""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            p.terminate()
            return device_count > 0
        except Exception:
            return False
            
    def test_network_connection(self):
        """测试网络连接"""
        try:
            import requests
            response = requests.get("https://www.google.com", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
            
    def detect_audio_devices(self):
        """检测音频设备"""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            
            result = "音频设备检测结果：\n\n"
            
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    result += f"输入设备 {i}: {info['name']}\n"
                if info['maxOutputChannels'] > 0:
                    result += f"输出设备 {i}: {info['name']}\n"
                    
            p.terminate()
            self.config_result.setText(result)
            
        except Exception as e:
            self.config_result.setText(f"音频设备检测失败: {e}")
            
    def test_network(self):
        """测试网络连接"""
        import requests
        
        result = "网络连接测试结果：\n\n"
        
        # 测试Google连接
        try:
            response = requests.get("https://www.google.com", timeout=5)
            result += f"Google连接: {'✅ 正常' if response.status_code == 200 else '❌ 异常'}\n"
        except Exception as e:
            result += f"Google连接: ❌ 失败 ({e})\n"
            
        # 测试百度连接
        try:
            response = requests.get("https://www.baidu.com", timeout=5)
            result += f"百度连接: {'✅ 正常' if response.status_code == 200 else '❌ 异常'}\n"
        except Exception as e:
            result += f"百度连接: ❌ 失败 ({e})\n"
            
        self.config_result.setText(result)

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("AI眼科疾病智诊系统 - 语音设置")
    app.setApplicationVersion("1.0")
    
    # 创建并显示窗口
    window = VoiceSetupWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 