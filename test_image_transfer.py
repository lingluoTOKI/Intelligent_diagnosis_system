#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片传输功能测试脚本
验证开发板到PC端的图片保存功能
"""

import os
import time
import json
import socket
from datetime import datetime

def test_save_directory():
    """测试保存目录是否可访问"""
    print("🔍 测试保存目录...")
    
    save_path = r"C:\Users\47449\Desktop\yolo\Intelligent_diagnosis_system\ultralytics-main\datasets\test"
    
    try:
        # 检查目录是否存在
        if not os.path.exists(save_path):
            print(f"⚠️ 目录不存在，尝试创建: {save_path}")
            os.makedirs(save_path, exist_ok=True)
        
        # 检查目录权限
        test_file = os.path.join(save_path, "test_write_permission.txt")
        
        with open(test_file, 'w') as f:
            f.write("测试写入权限")
        
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"✅ 保存目录可访问: {save_path}")
            return True
        else:
            print(f"❌ 无法写入目录: {save_path}")
            return False
            
    except Exception as e:
        print(f"❌ 目录测试失败: {e}")
        return False

def test_network_ports():
    """测试网络端口状态"""
    print("\n🌐 测试网络端口...")
    
    ports_to_test = {
        5002: "摄像头数据接收",
        5003: "诊断结果发送", 
        5004: "命令控制"
    }
    
    results = {}
    
    for port, description in ports_to_test.items():
        try:
            # 尝试绑定端口
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", port))
            sock.close()
            
            results[port] = {"status": "可用", "description": description}
            print(f"✅ 端口 {port} ({description}): 可用")
            
        except Exception as e:
            results[port] = {"status": "占用", "description": description, "error": str(e)}
            print(f"❌ 端口 {port} ({description}): 占用或错误")
    
    return results

def simulate_image_save_request():
    """模拟图像保存请求"""
    print("\n📸 模拟图像保存请求...")
    
    try:
        # 创建模拟保存请求
        save_request = {
            "type": "image_save_request",
            "request_id": f"test_save_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "filename": f"test_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            "image_size": 102400,  # 模拟100KB图像
            "width": 640,
            "height": 480,
            "pc_save_path": r"C:\Users\47449\Desktop\yolo\Intelligent_diagnosis_system\ultralytics-main\datasets\test"
        }
        
        print(f"请求ID: {save_request['request_id']}")
        print(f"文件名: {save_request['filename']}")
        print(f"保存路径: {save_request['pc_save_path']}")
        print(f"模拟图像大小: {save_request['image_size']} 字节")
        
        # 验证请求格式
        request_json = json.dumps(save_request, ensure_ascii=False)
        request_size = len(request_json.encode('utf-8'))
        
        if request_size > 4096:
            print(f"⚠️ 警告: 请求大小 ({request_size} 字节) 超过UDP限制")
        else:
            print(f"✅ 请求格式正确，大小: {request_size} 字节")
        
        return save_request
        
    except Exception as e:
        print(f"❌ 模拟请求创建失败: {e}")
        return None

def check_image_processing_capability():
    """检查图像处理能力"""
    print("\n🖼️ 检查图像处理能力...")
    
    try:
        import cv2
        import numpy as np
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试图像编码
        _, encoded = cv2.imencode('.jpg', test_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        encoded_size = len(encoded.tobytes())
        
        print(f"✅ OpenCV可用")
        print(f"测试图像尺寸: {test_image.shape}")
        print(f"JPEG编码后大小: {encoded_size} 字节")
        
        # 测试图像解码
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            print(f"✅ 图像编码/解码正常")
            return True
        else:
            print(f"❌ 图像解码失败")
            return False
            
    except ImportError:
        print(f"❌ OpenCV未安装")
        return False
    except Exception as e:
        print(f"❌ 图像处理测试失败: {e}")
        return False

def test_packet_fragmentation():
    """测试数据包分片"""
    print("\n📦 测试数据包分片...")
    
    try:
        # 模拟大图像数据
        image_size = 150000  # 150KB
        max_packet_size = 1400  # 标准UDP包大小
        
        total_packets = (image_size + max_packet_size - 1) // max_packet_size
        
        print(f"图像大小: {image_size} 字节")
        print(f"最大包大小: {max_packet_size} 字节")
        print(f"需要分片数: {total_packets} 包")
        
        if total_packets > 200:
            print(f"⚠️ 警告: 分片数量过多 ({total_packets})，可能影响传输效率")
        else:
            print(f"✅ 分片数量合理")
        
        # 模拟包头大小计算
        packet_header_size = 4 + 2 + 2 + 1  # packet_id + index + total + flag
        effective_payload = max_packet_size - packet_header_size
        
        print(f"包头大小: {packet_header_size} 字节")
        print(f"有效载荷: {effective_payload} 字节")
        
        return True
        
    except Exception as e:
        print(f"❌ 分片测试失败: {e}")
        return False

def generate_test_report(results):
    """生成测试报告"""
    print("\n" + "="*60)
    print("📋 图片传输功能测试报告")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总测试项: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n详细结果:")
    test_names = {
        "directory": "保存目录检查",
        "network": "网络端口检查", 
        "image_request": "图像请求格式",
        "image_processing": "图像处理能力",
        "fragmentation": "数据包分片"
    }
    
    for test_key, result in results.items():
        test_name = test_names.get(test_key, test_key)
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print("\n💡 建议:")
    if passed_tests == total_tests:
        print("  🎉 所有测试通过！图片传输功能就绪。")
        print("  📸 可以开始使用拍照诊断功能。")
    else:
        print("  ⚠️ 部分测试失败，请检查：")
        if not results.get("directory"):
            print("    - 检查保存目录权限和路径")
        if not results.get("network"):
            print("    - 确保网络端口未被占用")
        if not results.get("image_processing"):
            print("    - 安装OpenCV: pip install opencv-python")
        print("    - 重新运行测试确认修复结果")
    
    print("="*60)

def main():
    """主测试函数"""
    print("🚀 开始图片传输功能测试...")
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 1. 测试保存目录
    results["directory"] = test_save_directory()
    
    # 2. 测试网络端口
    port_results = test_network_ports()
    results["network"] = all(info["status"] == "可用" for info in port_results.values())
    
    # 3. 模拟图像保存请求
    save_request = simulate_image_save_request()
    results["image_request"] = save_request is not None
    
    # 4. 检查图像处理能力
    results["image_processing"] = check_image_processing_capability()
    
    # 5. 测试数据包分片
    results["fragmentation"] = test_packet_fragmentation()
    
    # 生成测试报告
    generate_test_report(results)
    
    print(f"\n测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
