#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按钮样式测试程序
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class ButtonTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("按钮样式测试")
        self.setGeometry(100, 100, 800, 600)
        
        # 颜色定义
        self.accent_color = "#4299e1"
        self.highlight_color = "#e53e3e"
        
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
        
        # 基础按钮样式
        button_style = f"""
            QPushButton {{
                background-color: {self.accent_color};
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px 15px;
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
                border: 2px solid {self.accent_color};
                min-height: 20px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #3182ce;
                border: 2px solid #3182ce;
            }}
            QPushButton:pressed {{
                background-color: #2b6cb0;
                border: 2px solid #2b6cb0;
            }}
            QPushButton:disabled {{
                background-color: #718096;
                color: #a0aec0;
                border: 2px solid #718096;
            }}
        """
        
        # AI建议按钮样式
        advice_style = f"""
            QPushButton {{
                background-color: {self.highlight_color};
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px 15px;
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
                border: 2px solid {self.highlight_color};
                min-height: 20px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #2c5aa0;
                border: 2px solid #2c5aa0;
            }}
            QPushButton:pressed {{
                background-color: #1e3f5f;
                border: 2px solid #1e3f5f;
            }}
            QPushButton:disabled {{
                background-color: #718096;
                color: #a0aec0;
                border: 2px solid #718096;
            }}
        """
        
        # 批量处理按钮样式
        batch_style = f"""
            QPushButton {{
                background-color: #805ad5;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px 15px;
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
                border: 2px solid #805ad5;
                min-height: 20px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #9f7aea;
                border: 2px solid #9f7aea;
            }}
            QPushButton:pressed {{
                background-color: #553c9a;
                border: 2px solid #553c9a;
            }}
            QPushButton:disabled {{
                background-color: #718096;
                color: #a0aec0;
                border: 2px solid #718096;
            }}
        """
        
        # 历史记录按钮样式
        history_style = f"""
            QPushButton {{
                background-color: #d69e2e;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 10px;
                padding: 10px 15px;
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
                border: 2px solid #d69e2e;
                min-height: 20px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #ed8936;
                border: 2px solid #ed8936;
            }}
            QPushButton:pressed {{
                background-color: #b7791f;
                border: 2px solid #b7791f;
            }}
            QPushButton:disabled {{
                background-color: #718096;
                color: #a0aec0;
                border: 2px solid #718096;
            }}
        """
        
        # 创建按钮
        buttons_layout = QHBoxLayout()
        
        # 基础按钮
        basic_button = QPushButton("🖼️ 加载图像")
        basic_button.setStyleSheet(button_style)
        basic_button.setCursor(Qt.PointingHandCursor)
        buttons_layout.addWidget(basic_button)
        
        # AI建议按钮
        advice_button = QPushButton("🤖 AI治疗建议")
        advice_button.setStyleSheet(advice_style)
        advice_button.setCursor(Qt.PointingHandCursor)
        buttons_layout.addWidget(advice_button)
        
        # 批量处理按钮
        batch_button = QPushButton("📁 批量处理")
        batch_button.setStyleSheet(batch_style)
        batch_button.setCursor(Qt.PointingHandCursor)
        buttons_layout.addWidget(batch_button)
        
        # 历史记录按钮
        history_button = QPushButton("📜 历史记录")
        history_button.setStyleSheet(history_style)
        history_button.setCursor(Qt.PointingHandCursor)
        buttons_layout.addWidget(history_button)
        
        layout.addLayout(buttons_layout)
        
        # 添加说明文本
        info_label = QWidget()
        info_label.setStyleSheet("""
            QWidget {
                background-color: #2a3441;
                border-radius: 8px;
                padding: 15px;
                color: #e2e8f0;
                font-size: 14px;
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            }
        """)
        info_layout = QVBoxLayout(info_label)
        info_layout.addWidget(QWidget())  # 占位符
        layout.addWidget(info_label)
        
        # 设置说明文本
        info_text = """
        <h3>按钮样式测试</h3>
        <p>✅ 基础按钮 - 蓝色主题</p>
        <p>✅ AI建议按钮 - 红色主题</p>
        <p>✅ 批量处理按钮 - 紫色主题</p>
        <p>✅ 历史记录按钮 - 橙色主题</p>
        <p><b>测试要点：</b></p>
        <ul>
            <li>悬停效果 - 鼠标悬停时颜色变化</li>
            <li>按下效果 - 点击时颜色变深</li>
            <li>圆角设计 - 10px圆角</li>
            <li>边框效果 - 2px边框</li>
            <li>字体设置 - 微软雅黑</li>
        </ul>
        """
        
        from PyQt5.QtWidgets import QLabel
        info_label_widget = QLabel(info_text)
        info_label_widget.setWordWrap(True)
        info_layout.addWidget(info_label_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ButtonTestWindow()
    window.show()
    sys.exit(app.exec_()) 