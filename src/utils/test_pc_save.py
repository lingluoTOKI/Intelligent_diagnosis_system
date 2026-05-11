#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC端图像保存测试脚本
验证开发板端图像是否能正确保存到指定目录
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime

def test_pc_save_directory():
    """测试PC端保存目录"""
    print("🔍 测试PC端保存目录")
    print("=" * 50)
    
    # 目标保存路径
    save_path = r"C:\Users\47449\Desktop\yolo\Intelligent_diagnosis_system\ultralytics-main\datasets\test"
    
    print(f"📁 目标保存路径: {save_path}")
    
    # 检查目录是否存在
    if os.path.exists(save_path):
        print(f"✅ 目录已存在: {save_path}")
    else:
        print(f"❌ 目录不存在: {save_path}")
        print("🔄 尝试创建目录...")
        try:
            os.makedirs(save_path, exist_ok=True)
            print(f"✅ 目录创建成功: {save_path}")
        except Exception as e:
            print(f"❌ 目录创建失败: {e}")
            return False
    
    # 检查目录权限
    try:
        test_file = os.path.join(save_path, "test_permission.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✅ 目录写入权限正常")
    except Exception as e:
        print(f"❌ 目录写入权限异常: {e}")
        return False
    
    return True

def create_test_image():
    """创建测试图像"""
    print("\n📸 创建测试图像")
    print("=" * 50)
    
    # 创建测试图像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加一些测试内容
    cv2.putText(image, "Test Image", (200, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(image, "Board Camera Test", (150, 250), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (100, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    print(f"✅ 测试图像创建成功，尺寸: {image.shape}")
    return image

def simulate_board_save(image, save_path):
    """模拟开发板保存过程"""
    print("\n💾 模拟开发板保存过程")
    print("=" * 50)
    
    try:
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        request_id = f"test_req_{timestamp}"
        
        # 保存图像
        image_filename = f"board_image_{timestamp}_{request_id}.jpg"
        image_path = os.path.join(save_path, image_filename)
        cv2.imwrite(image_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 保存信息文件
        info_filename = f"board_info_{timestamp}_{request_id}.json"
        info_path = os.path.join(save_path, info_filename)
        
        info_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "image_size": image.shape,
            "image_path": image_path,
            "source": "board_camera_simulation",
            "pc_save_time": datetime.now().isoformat(),
            "test_mode": True
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 模拟保存成功:")
        print(f"   图像文件: {image_path}")
        print(f"   信息文件: {info_path}")
        
        # 验证文件
        if os.path.exists(image_path):
            print(f"✅ 图像文件验证成功: {os.path.getsize(image_path)} 字节")
        else:
            print("❌ 图像文件验证失败")
        
        if os.path.exists(info_path):
            print(f"✅ 信息文件验证成功: {os.path.getsize(info_path)} 字节")
        else:
            print("❌ 信息文件验证失败")
        
        return True
        
    except Exception as e:
        print(f"❌ 模拟保存失败: {e}")
        return False

def list_saved_files(save_path):
    """列出已保存的文件"""
    print("\n📋 列出已保存的文件")
    print("=" * 50)
    
    try:
        files = os.listdir(save_path)
        board_files = [f for f in files if f.startswith('board_')]
        
        if board_files:
            print(f"✅ 找到 {len(board_files)} 个开发板相关文件:")
            for file in sorted(board_files):
                file_path = os.path.join(save_path, file)
                file_size = os.path.getsize(file_path)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"   📄 {file} ({file_size} 字节, {file_time.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print("📭 未找到开发板相关文件")
            
    except Exception as e:
        print(f"❌ 列出文件失败: {e}")

def main():
    """主函数"""
    print("🚀 PC端图像保存功能测试")
    print("=" * 60)
    
    # 目标保存路径
    save_path = r"C:\Users\47449\Desktop\yolo\Intelligent_diagnosis_system\ultralytics-main\datasets\test"
    
    try:
        # 测试1: 检查保存目录
        if not test_pc_save_directory():
            print("❌ 目录测试失败，无法继续")
            return
        
        # 测试2: 创建测试图像
        test_image = create_test_image()
        
        # 测试3: 模拟开发板保存
        if simulate_board_save(test_image, save_path):
            print("✅ 模拟保存测试成功")
        else:
            print("❌ 模拟保存测试失败")
        
        # 测试4: 列出已保存文件
        list_saved_files(save_path)
        
        print("\n🎉 测试完成")
        print("💡 如果所有测试都通过，说明PC端保存功能正常")
        print(f"💡 开发板图像将保存到: {save_path}")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")

if __name__ == "__main__":
    main()
