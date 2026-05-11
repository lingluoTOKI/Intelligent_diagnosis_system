#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JupyterLab环境专用开发板启动脚本
解决JupyterLab中无法进行命令行交互的问题
"""

import time
import threading
from board_integrated_system import BoardMedicalSystem

class JupyterBoardLauncher:
    """JupyterLab环境专用启动器"""
    
    def __init__(self):
        self.system = BoardMedicalSystem()
        self.auto_mode = True
        self.command_queue = []
        
    def start_auto_mode(self):
        """启动自动模式，模拟用户操作"""
        print("🚀 [Jupyter] 启动自动模式")
        print("=" * 50)
        
        # 初始化系统
        if not self.system.initialize():
            print("❌ [失败] 系统初始化失败")
            return
        
        print("✅ [成功] 系统初始化完成")
        
        # 启动自动测试流程
        self._run_auto_test_sequence()
    
    def _run_auto_test_sequence(self):
        """运行自动测试序列"""
        print("\n🤖 [自动测试] 开始执行测试序列")
        
        # 测试1: 网络连接
        print("\n📡 [测试1] 测试网络连接...")
        if self.system.network_manager.test_connection():
            print("✅ 网络连接正常")
        else:
            print("❌ 网络连接失败")
            return
        
        # 测试2: 启动视频流
        print("\n📹 [测试2] 启动视频流传输...")
        self.system.start_video_streaming()
        time.sleep(2)  # 等待启动
        
        # 测试3: 启动摄像头预览
        print("\n📷 [测试3] 启动摄像头预览...")
        self.system.camera_manager.start_preview()
        
        # 测试4: 模拟拍照诊断
        print("\n🔍 [测试4] 模拟拍照诊断...")
        time.sleep(3)  # 等待摄像头稳定
        
        # 获取当前帧进行诊断
        if self.system.camera_manager.camera_thread:
            frame = self.system.camera_manager.camera_thread.capture_frame()
            if frame is not None:
                print("✅ 成功获取图像帧")
                # 发送诊断请求，并保存到PC端指定目录
                request_id = self.system.network_manager.send_image_for_diagnosis(frame, save_to_pc=True)
                if request_id:
                    print(f"✅ 诊断请求已发送，ID: {request_id}")
                    # 等待诊断结果
                    result = self.system.network_manager.wait_for_diagnosis_result(request_id)
                    if result:
                        print("✅ 收到诊断结果")
                        self._display_diagnosis_result(result)
                    else:
                        print("⚠️  诊断结果等待超时")
                else:
                    print("❌ 诊断请求发送失败")
            else:
                print("❌ 无法获取图像帧")
        
        print("\n🎉 [自动测试] 测试序列完成")
        print("💡 提示: 摄像头预览窗口已打开，可以进行手动操作")
        print("   键盘快捷键:")
        print("   - 空格键: 拍照诊断")
        print("   - 's'键: 保存照片")
        print("   - 'r'键: 语音对话")
        print("   - 'q'键: 退出预览")
        
        # 保持系统运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 [中断] 收到停止信号")
            self.system.shutdown()
    
    def _display_diagnosis_result(self, result):
        """显示诊断结果"""
        print("\n" + "="*50)
        print("🏥 [AI诊断结果]")
        print("="*50)
        
        if 'disease_name' in result:
            print(f"疾病名称: {result['disease_name']}")
        
        if 'confidence' in result:
            confidence = result['confidence']
            print(f"置信度: {confidence:.2%}")
            
            if confidence > 0.8:
                print("评级: 高置信度诊断")
            elif confidence > 0.6:
                print("评级: 中等置信度诊断")
            else:
                print("评级: 低置信度诊断，建议专业医生确认")
        
        if 'advice' in result:
            print(f"医疗建议: {result['advice']}")
        
        if 'emergency' in result and result['emergency']:
            print("⚠️  紧急情况！建议立即就医")
        
        print("="*50)
    
    def start_interactive_mode(self):
        """启动交互模式（如果可能）"""
        print("🎮 [Jupyter] 启动交互模式")
        print("注意: 在JupyterLab中，建议使用自动模式")
        
        # 尝试启动触摸界面
        try:
            self.system.run()
        except Exception as e:
            print(f"❌ 交互模式启动失败: {e}")
            print("🔄 切换到自动模式...")
            self.start_auto_mode()

def main():
    """主函数"""
    print("🏥 JupyterLab环境专用开发板启动器")
    print("=" * 50)
    print("功能特性:")
    print("  ✅ 自动网络连接测试")
    print("  ✅ 自动视频流启动")
    print("  ✅ 自动摄像头预览")
    print("  ✅ 自动拍照诊断测试")
    print("  ✅ JupyterLab环境兼容")
    print("=" * 50)
    
    launcher = JupyterBoardLauncher()
    
    try:
        # 启动自动模式
        launcher.start_auto_mode()
        
    except Exception as e:
        print(f"❌ [错误] 启动器运行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
