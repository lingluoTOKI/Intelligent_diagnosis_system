#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发板连接测试脚本
用于测试开发板与PC端的网络连接和摄像头功能
"""

import socket
import cv2
import numpy as np
import json
import time
import threading
from datetime import datetime

# 配置参数
PC_IP = "192.168.1.100"  # 请修改为你的PC端实际IP地址
CAMERA_PORT = 5002
COMMAND_PORT = 5004
DIAGNOSIS_PORT = 5003

def test_network_connectivity():
    """测试网络连通性"""
    print("🔍 [测试] 开始网络连通性测试...")
    
    # 测试ping
    import subprocess
    try:
        result = subprocess.run(['ping', '-c', '1', PC_IP], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ [网络] Ping {PC_IP} 成功")
        else:
            print(f"❌ [网络] Ping {PC_IP} 失败")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ [网络] Ping {PC_IP} 超时")
        return False
    except Exception as e:
        print(f"⚠️ [网络] Ping测试异常: {e}")
    
    # 测试端口连通性
    ports_to_test = [CAMERA_PORT, COMMAND_PORT, DIAGNOSIS_PORT]
    
    for port in ports_to_test:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(2)
            
            # 发送测试数据包
            test_data = b"connection_test"
            sock.sendto(test_data, (PC_IP, port))
            print(f"📡 [端口] 向 {PC_IP}:{port} 发送测试包")
            
            sock.close()
            
        except Exception as e:
            print(f"❌ [端口] {PC_IP}:{port} 连接失败: {e}")
    
    return True

def test_camera_functionality():
    """测试摄像头功能"""
    print("\n📹 [测试] 开始摄像头功能测试...")
    
    try:
        # 尝试打开摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ [摄像头] 无法打开摄像头设备")
            return False
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 获取实际参数
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✅ [摄像头] 摄像头已打开")
        print(f"📐 [分辨率] {width}x{height}")
        print(f"🎬 [帧率] {fps} FPS")
        
        # 测试图像捕获
        ret, frame = cap.read()
        if ret:
            print(f"✅ [图像] 成功捕获图像，大小: {frame.shape}")
            
            # 测试图像编码
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"✅ [编码] 图像编码成功，大小: {len(encoded)} 字节")
            
        else:
            print("❌ [图像] 图像捕获失败")
            cap.release()
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ [摄像头] 摄像头测试失败: {e}")
        return False

def test_pc_communication():
    """测试与PC端的通信"""
    print("\n💬 [测试] 开始PC端通信测试...")
    
    try:
        # 创建UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5.0)
        
        # 发送连接测试包
        test_data = {
            "type": "connection_test",
            "timestamp": datetime.now().isoformat(),
            "board_id": "medical_board_test",
            "version": "2.0"
        }
        
        test_bytes = json.dumps(test_data).encode('utf-8')
        sock.sendto(test_bytes, (PC_IP, COMMAND_PORT))
        print(f"📤 [发送] 已发送连接测试包到 {PC_IP}:{COMMAND_PORT}")
        
        # 等待响应
        try:
            response, addr = sock.recvfrom(1024)
            response_data = json.loads(response.decode('utf-8'))
            
            if response_data.get('type') == 'connection_test_response':
                print(f"✅ [响应] 收到PC端响应: {response_data}")
                return True
            else:
                print(f"⚠️ [响应] 收到意外响应: {response_data}")
                return False
                
        except socket.timeout:
            print("❌ [响应] 等待PC端响应超时")
            return False
            
    except Exception as e:
        print(f"❌ [通信] PC端通信测试失败: {e}")
        return False
    finally:
        sock.close()

def send_test_image():
    """发送测试图像"""
    print("\n🖼️ [测试] 开始测试图像发送...")
    
    try:
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ [摄像头] 无法打开摄像头")
            return False
        
        # 捕获一帧
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ [图像] 无法捕获图像")
            return False
        
        # 压缩图像
        _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_data = img_encoded.tobytes()
        
        print(f"📊 [图像] 图像大小: {len(img_data)} 字节")
        
        # 创建UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 分片发送
        MAX_PACKET_SIZE = 1400
        total_packets = (len(img_data) + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
        
        print(f"📦 [分片] 将分成 {total_packets} 个数据包发送")
        
        timestamp = int(time.time() * 1000)
        request_id = f"test_req_{timestamp}"
        
        # 发送头部信息
        header = {
            "type": "diagnosis_request",
            "request_id": request_id,
            "timestamp": timestamp,
            "image_size": len(img_data),
            "width": frame.shape[1],
            "height": frame.shape[0],
            "total_packets": total_packets
        }
        
        header_data = json.dumps(header).encode('utf-8')
        sock.sendto(header_data, (PC_IP, COMMAND_PORT))
        print(f"📤 [头部] 已发送图像头部信息")
        
        # 分片发送图像数据
        for i in range(total_packets):
            start = i * MAX_PACKET_SIZE
            end = min(start + MAX_PACKET_SIZE, len(img_data))
            packet_data = img_data[start:end]
            
            # 包头格式
            packet_id = hash(request_id) & 0xFFFFFFFF
            packet_header = (
                packet_id.to_bytes(4, 'big') +
                i.to_bytes(2, 'big') +
                total_packets.to_bytes(2, 'big') +
                (1 if i == total_packets - 1 else 0).to_bytes(1, 'big')
            )
            packet = packet_header + packet_data
            
            sock.sendto(packet, (PC_IP, CAMERA_PORT))
            
            if i % 10 == 0:  # 每10个包显示一次进度
                print(f"📤 [进度] 已发送 {i+1}/{total_packets} 个数据包")
            
            time.sleep(0.001)  # 1ms延迟
        
        print(f"✅ [发送] 图像发送完成，请求ID: {request_id}")
        sock.close()
        
        return True
        
    except Exception as e:
        print(f"❌ [发送] 图像发送失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔧 开发板连接测试工具")
    print("=" * 50)
    print(f"📡 目标PC IP: {PC_IP}")
    print(f"📹 摄像头端口: {CAMERA_PORT}")
    print(f"⚙️ 命令端口: {COMMAND_PORT}")
    print(f"🔍 诊断端口: {DIAGNOSIS_PORT}")
    print("=" * 50)
    
    # 执行测试
    tests = [
        ("网络连通性", test_network_connectivity),
        ("摄像头功能", test_camera_functionality),
        ("PC端通信", test_pc_communication),
        ("图像发送", send_test_image)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 开始测试: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 显示测试总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print("=" * 50)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📈 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！开发板连接正常")
    else:
        print("⚠️ 部分测试失败，请检查网络配置和PC端程序")
        print("\n💡 故障排除建议:")
        print("1. 确认PC端医疗诊断程序已启动")
        print("2. 检查防火墙设置，允许相关端口通信")
        print(f"3. 确认PC IP地址是否正确: {PC_IP}")
        print("4. 检查开发板与PC是否在同一局域网")

if __name__ == "__main__":
    main()
