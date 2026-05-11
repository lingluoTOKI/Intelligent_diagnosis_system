#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试批量检测报告的全屏和退出按钮功能
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QDialog, QLabel, QTextEdit
from PyQt5.QtCore import Qt

class TestBatchReport(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("测试批量检测报告")
        self.setGeometry(100, 100, 400, 300)
        
        # 创建测试按钮
        test_btn = QPushButton("测试批量检测报告")
        test_btn.clicked.connect(self.test_batch_report)
        
        # 设置布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(test_btn)
    
    def test_batch_report(self):
        """测试批量检测报告对话框"""
        # 模拟数据
        disease_counter = {
            "AMD": 5,
            "糖尿病视网膜病变": 3,
            "青光眼": 2,
            "白内障": 1
        }
        
        results_summary = [
            "检测完成，共处理10张图像",
            "AMD: 5例 (50%)",
            "糖尿病视网膜病变: 3例 (30%)",
            "青光眼: 2例 (20%)",
            "白内障: 1例 (10%)"
        ]
        
        # 调用批量检测报告
        self.show_batch_report(disease_counter, results_summary)
    
    def show_batch_report(self, disease_counter, results_summary):
        """显示批量检测统计报告"""
        dialog = QDialog(self)
        dialog.setWindowTitle("批量检测统计报告")
        dialog.resize(800, 600)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d3748;
                color: #e2e8f0;
            }
        """)

        layout = QVBoxLayout(dialog)

        # 统计文本
        stat_text = QLabel(
            "<b>各疾病检测数量统计：</b><br>" + "<br>".join([f"{k}: {v}" for k, v in disease_counter.items()]))
        stat_text.setWordWrap(True)
        stat_text.setStyleSheet("color: #e2e8f0;")
        layout.addWidget(stat_text)

        # 结果摘要
        result_box = QTextEdit()
        result_box.setReadOnly(True)
        result_box.setText("\n".join(results_summary))
        result_box.setStyleSheet("""
            QTextEdit {
                background-color: #4a5568;
                color: #e2e8f0;
                border: 1px solid #3182ce;
                border-radius: 4px;
            }
        """)
        layout.addWidget(result_box)

        # 按钮布局
        button_layout = QVBoxLayout() # Changed from QHBoxLayout to QVBoxLayout
        
        # 全屏按钮
        fullscreen_btn = QPushButton("🖥️ 全屏")
        fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: #3182ce;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                margin-top: 10px;
                margin-right: 10px;
            }
            QPushButton:hover {
                background-color: #3182ce;
            }
        """)
        fullscreen_btn.clicked.connect(lambda: self.toggle_batch_report_fullscreen(dialog))
        button_layout.addWidget(fullscreen_btn)
        
        # 关闭按钮
        close_btn = QPushButton("❌ 关闭")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3182ce;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                margin-top: 10px;
                margin-right: 10px;
            }
            QPushButton:hover {
                background-color: #3182ce;
            }
        """)
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        # 退出按钮
        exit_btn = QPushButton("🚪 退出")
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #e53e3e;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #c53030;
            }
        """)
        exit_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(exit_btn)
        
        # 添加按钮布局
        layout.addLayout(button_layout)

        dialog.exec_()
    
    def toggle_batch_report_fullscreen(self, dialog):
        """切换批量检测报告的全屏模式"""
        if dialog.isMaximized():
            dialog.showNormal()
        else:
            dialog.showMaximized()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestBatchReport()
    window.show()
    sys.exit(app.exec_())

