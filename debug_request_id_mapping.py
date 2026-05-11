#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试 request_id 映射问题的脚本
模拟开发板发送诊断请求的过程
"""

import socket
import json
import time
import cv2
import numpy as np
from datetime import datetime

def send_diagnosis_request(pc_ip="172.20.10.3"):
    """发送诊断请求命令"""
    try:
        # 创建命令套接字
        command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 生成请求ID
        timestamp = int(time.time() * 1000)
        request_id = f"debug_req_{timestamp}"
        
        # 创建诊断请求命令
        command_data = {
            "type": "diagnosis_request",
            "request_id": request_id,
            "timestamp": timestamp,
            "image_size": 0,  # 稍后填充
            "width": 640,
            "height": 480,
            "total_packets": 0,  # 稍后填充
            "compression_quality": 95,
            "save_to_pc": True,
            "pc_save_path": r"C:\Users\47449\Desktop\yolo\Intelligent_diagnosis_system\ultralytics-main\datasets\test"
        }
        
        # 发送命令到PC端
        command_json = json.dumps(command_data)
        command_socket.sendto(command_json.encode('utf-8'), (pc_ip, 5004))
        command_socket.close()
        
        print(f"✅ 诊断请求命令已发送，request_id: {request_id}")
        return request_id
        
    except Exception as e:
        print(f"❌ 发送诊断请求命令失败: {e}")
        return None

def create_test_image():
    """创建测试图像"""
    # 创建测试图像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加测试内容
    cv2.putText(image, "DEBUG TEST IMAGE", (150, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(image, f"Time: {datetime.now().strftime('%H:%M:%S')}", (200, 250), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, "Request ID Mapping Test", (120, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    return image

def send_image_data(request_id, image, pc_ip="172.20.10.3"):
    """发送图像数据"""
    try:
        # 编码图像
        _, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_data = img_encoded.tobytes()
        
        # 创建图像数据套接字
        camera_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 分片参数
        MAX_PACKET_SIZE = 1400
        total_packets = (len(img_data) + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
        
        print(f"📸 开始发送图像数据，大小: {len(img_data)} 字节，分片: {total_packets}")
        
        # 计算packet_id（与开发板端相同的哈希方法）
        packet_id = hash(request_id) & 0xFFFFFFFF
        print(f"🔑 packet_id: {packet_id}, request_id: {request_id}")
        
        for i in range(total_packets):
            start = i * MAX_PACKET_SIZE
            end = min(start + MAX_PACKET_SIZE, len(img_data))
            packet_data = img_data[start:end]
            
            # 构建包头：[4字节请求ID哈希][2字节包索引][2字节总包数][1字节标志位]
            packet_header = (
                packet_id.to_bytes(4, 'big') +
                i.to_bytes(2, 'big') +
                total_packets.to_bytes(2, 'big') +
                (1 if i == total_packets - 1 else 0).to_bytes(1, 'big')
            )
            
            # 在第一个包中包含request_id信息
            if i == 0:
                request_info = request_id.encode('utf-8')
                packet = packet_header + len(request_info).to_bytes(2, 'big') + request_info + packet_data
                print(f"📦 第一个包包含request_id: {request_id} (长度: {len(request_info)})")
            else:
                packet = packet_header + packet_data
            
            # 发送数据包
            camera_socket.sendto(packet, (pc_ip, 5002))  # 5002是摄像头数据端口
            
            if i < total_packets - 1:
                time.sleep(0.001)  # 1ms间隔
        
        camera_socket.close()
        print(f"✅ 图像数据发送完成，request_id: {request_id}")
        
    except Exception as e:
        print(f"❌ 发送图像数据失败: {e}")

def main():
    """主函数"""
    print("🐛 request_id 映射问题调试脚本")
    print("=" * 50)
    
    pc_ip = "172.20.10.3"
    
    try:
        # 步骤1: 发送诊断请求命令
        print("📡 步骤1: 发送诊断请求命令")
        request_id = send_diagnosis_request(pc_ip)
        
        if not request_id:
            print("❌ 无法发送诊断请求命令")
            return
        
        # 等待PC端处理命令
        print("⏳ 等待PC端处理命令...")
        time.sleep(1)
        
        # 步骤2: 创建测试图像
        print("🖼️ 步骤2: 创建测试图像")
        test_image = create_test_image()
        print(f"✅ 测试图像创建成功，尺寸: {test_image.shape}")
        
        # 步骤3: 发送图像数据
        print("📸 步骤3: 发送图像数据")
        send_image_data(request_id, test_image, pc_ip)
        
        print("\n🎯 调试完成")
        print("💡 请查看PC端的日志输出，观察以下信息：")
        print("   1. 是否收到并存储了诊断请求头")
        print("   2. 是否正确解析了图像包中的request_id")
        print("   3. 是否成功建立了packet_id到request_id的映射")
        print("   4. 是否能找到对应的请求头信息")
        
    except Exception as e:
        print(f"❌ 调试过程中发生错误: {e}")

if __name__ == "__main__":
    main()
