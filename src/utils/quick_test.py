#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证开发板端和PC端通信
"""

import time
import cv2
import numpy as np
from board_integrated_system import NetworkManager, CameraManager

def quick_network_test():
    """快速网络测试"""
    print("🔍 快速网络测试")
    print("=" * 40)
    
    # 创建网络管理器
    network = NetworkManager()
    
    # 测试连接
    print("📡 测试PC端连接...")
    if network.test_connection():
        print("✅ 网络连接正常")
        
        # 测试心跳
        print("💓 测试心跳包...")
        network._send_heartbeat()
        time.sleep(1)
        
        # 测试视频流
        print("📹 测试视频流传输...")
        network.start_streaming()
        
        # 创建测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Frame", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # 发送测试帧
        print("📤 发送测试图像帧...")
        success = network.send_stream_frame(test_image)
        if success:
            print("✅ 测试帧发送成功")
        else:
            print("❌ 测试帧发送失败")
        
        network.stop_streaming()
        
    else:
        print("❌ 网络连接失败")
    
    network.close()

def quick_camera_test():
    """快速摄像头测试"""
    print("\n📷 快速摄像头测试")
    print("=" * 40)
    
    # 创建摄像头管理器
    camera = CameraManager()
    
    # 初始化摄像头
    print("🔧 初始化摄像头...")
    if camera.initialize():
        print("✅ 摄像头初始化成功")
        
        # 获取状态
        status = camera.get_camera_status()
        print(f"📊 摄像头状态: {status}")
        
        # 测试图像捕获
        print("📸 测试图像捕获...")
        if camera.camera_thread:
            frame = camera.camera_thread.capture_frame()
            if frame is not None:
                print(f"✅ 成功捕获图像，尺寸: {frame.shape}")
                
                # 测试图像增强
                print("✨ 测试图像增强...")
                enhanced = camera.enhance_image(frame)
                print(f"✅ 图像增强完成，尺寸: {enhanced.shape}")
                
                # 测试图像保存
                print("💾 测试图像保存...")
                filename = f"test_frame_{int(time.time())}.jpg"
                filepath = camera.save_image(enhanced, filename)
                if filepath:
                    print(f"✅ 图像已保存: {filepath}")
                else:
                    print("❌ 图像保存失败")
            else:
                print("❌ 无法捕获图像")
        else:
            print("❌ 摄像头线程未初始化")
    else:
        print("❌ 摄像头初始化失败")
    
    # 释放资源
    camera.release()

def main():
    """主函数"""
    print("🚀 开发板端快速测试")
    print("=" * 50)
    
    try:
        # 网络测试
        quick_network_test()
        
        # 摄像头测试
        quick_camera_test()
        
        print("\n🎉 快速测试完成")
        print("💡 如果测试成功，说明基本功能正常")
        print("💡 如果测试失败，请检查:")
        print("   1. PC端是否已启动")
        print("   2. 网络配置是否正确")
        print("   3. 摄像头是否可用")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        print("💡 请检查错误信息并修复问题")

if __name__ == "__main__":
    main()
