#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音状态指示器测试程序
"""

import sys
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

class VoiceStatusTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音状态指示器测试")
        self.setGeometry(100, 100, 600, 400)
        
        # 颜色定义
        self.accent_color = "#4299e1"
        self.highlight_color = "#e53e3e"
        self.secondary_bg = "#2a3441"
        
        # 设置窗口样式
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #1a202c;
                color: #e2e8f0;
            }}
        """)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 语音状态指示器
        self.voice_status_indicator = QLabel("⭕")
        self.voice_status_indicator.setStyleSheet(f"""
            QLabel {{
                color: #666;
                font-size: 24px;
                font-weight: bold;
                padding: 8px;
                border-radius: 50%;
                background-color: rgba(102, 102, 102, 0.1);
                border: 2px solid #666;
                min-width: 40px;
                min-height: 40px;
                text-align: center;
            }}
        """)
        self.voice_status_indicator.setAlignment(Qt.AlignCenter)
        
        # 语音进度指示器
        self.voice_progress_indicator = QLabel("待机中")
        self.voice_progress_indicator.setStyleSheet(f"""
            QLabel {{
                color: {self.accent_color};
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                border-radius: 4px;
                background-color: {self.secondary_bg};
                border: 1px solid {self.accent_color};
                min-width: 80px;
                text-align: center;
            }}
        """)
        self.voice_progress_indicator.setAlignment(Qt.AlignCenter)
        
        # 状态显示布局
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.voice_status_indicator)
        status_layout.addWidget(self.voice_progress_indicator)
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
        
        # 测试按钮
        buttons_layout = QHBoxLayout()
        
        # 录音状态按钮
        recording_button = QPushButton("🔴 录音状态")
        recording_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #dc2626;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: #b91c1c;
            }}
        """)
        recording_button.clicked.connect(self.test_recording_status)
        buttons_layout.addWidget(recording_button)
        
        # 处理状态按钮
        processing_button = QPushButton("🟡 处理状态")
        processing_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #f59e0b;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: #d97706;
            }}
        """)
        processing_button.clicked.connect(self.test_processing_status)
        buttons_layout.addWidget(processing_button)
        
        # 成功状态按钮
        success_button = QPushButton("🟢 成功状态")
        success_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: #059669;
            }}
        """)
        success_button.clicked.connect(self.test_success_status)
        buttons_layout.addWidget(success_button)
        
        # 失败状态按钮
        error_button = QPushButton("❌ 失败状态")
        error_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #ef4444;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: #dc2626;
            }}
        """)
        error_button.clicked.connect(self.test_error_status)
        buttons_layout.addWidget(error_button)
        
        layout.addLayout(buttons_layout)
        
        # 说明文本
        info_text = """
        <h3>语音状态指示器测试</h3>
        <p><b>状态说明：</b></p>
        <ul>
            <li>🔴 录音状态 - 红色，表示正在录音</li>
            <li>🟡 处理状态 - 黄色，表示正在识别</li>
            <li>🟢 成功状态 - 绿色，表示识别成功</li>
            <li>❌ 失败状态 - 红色，表示识别失败</li>
            <li>⭕ 待机状态 - 灰色，表示等待开始</li>
        </ul>
        <p><b>测试方法：</b></p>
        <p>点击不同按钮查看状态指示器的变化效果</p>
        """
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("""
            QLabel {
                background-color: #2a3441;
                border-radius: 8px;
                padding: 15px;
                color: #e2e8f0;
                font-size: 14px;
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            }
        """)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 动画定时器
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_counter = 0
        self.current_status = "idle"
    
    def test_recording_status(self):
        """测试录音状态"""
        self.current_status = "recording"
        self.voice_status_indicator.setText("🔴")
        self.voice_status_indicator.setStyleSheet(f"""
            QLabel {{
                color: #dc2626;
                font-size: 24px;
                font-weight: bold;
                padding: 8px;
                border-radius: 50%;
                background-color: rgba(220, 38, 38, 0.2);
                border: 2px solid #dc2626;
                min-width: 40px;
                min-height: 40px;
                text-align: center;
            }}
        """)
        self.voice_progress_indicator.setText("录音中...")
        self.voice_progress_indicator.setStyleSheet(f"""
            QLabel {{
                color: #dc2626;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                border-radius: 4px;
                background-color: {self.secondary_bg};
                border: 1px solid #dc2626;
                min-width: 80px;
                text-align: center;
            }}
        """)
        self.animation_timer.start(500)
    
    def test_processing_status(self):
        """测试处理状态"""
        self.current_status = "processing"
        self.voice_status_indicator.setText("🟡")
        self.voice_status_indicator.setStyleSheet(f"""
            QLabel {{
                color: #f59e0b;
                font-size: 24px;
                font-weight: bold;
                padding: 8px;
                border-radius: 50%;
                background-color: rgba(245, 158, 11, 0.2);
                border: 2px solid #f59e0b;
                min-width: 40px;
                min-height: 40px;
                text-align: center;
            }}
        """)
        self.voice_progress_indicator.setText("识别中...")
        self.voice_progress_indicator.setStyleSheet(f"""
            QLabel {{
                color: #f59e0b;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                border-radius: 4px;
                background-color: {self.secondary_bg};
                border: 1px solid #f59e0b;
                min-width: 80px;
                text-align: center;
            }}
        """)
        self.animation_timer.stop()
    
    def test_success_status(self):
        """测试成功状态"""
        self.current_status = "success"
        self.voice_status_indicator.setText("🟢")
        self.voice_status_indicator.setStyleSheet(f"""
            QLabel {{
                color: #10b981;
                font-size: 24px;
                font-weight: bold;
                padding: 8px;
                border-radius: 50%;
                background-color: rgba(16, 185, 129, 0.2);
                border: 2px solid #10b981;
                min-width: 40px;
                min-height: 40px;
                text-align: center;
            }}
        """)
        self.voice_progress_indicator.setText("识别成功!")
        self.voice_progress_indicator.setStyleSheet(f"""
            QLabel {{
                color: #10b981;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                border-radius: 4px;
                background-color: {self.secondary_bg};
                border: 1px solid #10b981;
                min-width: 80px;
                text-align: center;
            }}
        """)
        self.animation_timer.stop()
    
    def test_error_status(self):
        """测试失败状态"""
        self.current_status = "error"
        self.voice_status_indicator.setText("❌")
        self.voice_status_indicator.setStyleSheet(f"""
            QLabel {{
                color: #ef4444;
                font-size: 24px;
                font-weight: bold;
                padding: 8px;
                border-radius: 50%;
                background-color: rgba(239, 68, 68, 0.2);
                border: 2px solid #ef4444;
                min-width: 40px;
                min-height: 40px;
                text-align: center;
            }}
        """)
        self.voice_progress_indicator.setText("识别失败")
        self.voice_progress_indicator.setStyleSheet(f"""
            QLabel {{
                color: #ef4444;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                border-radius: 4px;
                background-color: {self.secondary_bg};
                border: 1px solid #ef4444;
                min-width: 80px;
                text-align: center;
            }}
        """)
        self.animation_timer.stop()
    
    def update_animation(self):
        """更新动画效果"""
        self.animation_counter += 1
        if self.current_status == "recording":
            animation_chars = ["🔴", "🟠", "🟡", "🟢", "🔵", "🟣"]
            current_char = animation_chars[self.animation_counter % len(animation_chars)]
            self.voice_status_indicator.setText(current_char)
            
            progress_texts = ["录音中...", "正在录音...", "请说话...", "继续录音..."]
            current_text = progress_texts[self.animation_counter % len(progress_texts)]
            self.voice_progress_indicator.setText(current_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceStatusTestWindow()
    window.show()
    sys.exit(app.exec_()) 