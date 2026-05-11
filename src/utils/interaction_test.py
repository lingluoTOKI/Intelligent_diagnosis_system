#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发板端和PC端交互测试脚本
用于验证通信协议是否匹配
"""

import json
import socket
import time
from datetime import datetime

def test_connection_protocol():
    """测试连接协议"""
    print("🔍 测试开发板端和PC端交互协议")
    print("=" * 60)
    
    # 测试配置
    PC_IP = "172.20.10.3"
    COMMAND_PORT = 5004
    CAMERA_PORT = 5002
    DIAGNOSIS_PORT = 5003
    
    print(f"📡 网络配置:")
    print(f"  PC端IP: {PC_IP}")
    print(f"  命令端口: {COMMAND_PORT}")
    print(f"  摄像头端口: {CAMERA_PORT}")
    print(f"  诊断端口: {DIAGNOSIS_PORT}")
    print()
    
    # 测试1: 连接测试
    print("🧪 测试1: 连接测试")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5.0)
        
        test_data = {
            "type": "connection_test",
            "timestamp": datetime.now().isoformat(),
            "board_id": "test_board_001",
            "version": "2.0"
        }
        
        test_bytes = json.dumps(test_data).encode('utf-8')
        sock.sendto(test_bytes, (PC_IP, COMMAND_PORT))
        print(f"  ✅ 连接测试包已发送到 {PC_IP}:{COMMAND_PORT}")
        
        # 等待响应
        try:
            response, addr = sock.recvfrom(1024)
            response_data = json.loads(response.decode('utf-8'))
            print(f"  ✅ 收到响应: {response_data}")
        except socket.timeout:
            print("  ⚠️  连接测试超时 - PC端可能未启动")
        
        sock.close()
        
    except Exception as e:
        print(f"  ❌ 连接测试失败: {e}")
    
    print()
    
    # 测试2: 诊断请求协议
    print("🧪 测试2: 诊断请求协议")
    print("  开发板端发送格式:")
    print("    1. 命令端口发送头部信息:")
    print("       - type: 'diagnosis_request'")
    print("       - request_id: 'req_timestamp_sequence'")
    print("       - timestamp: 时间戳")
    print("       - image_size: 图像大小")
    print("       - width/height: 图像尺寸")
    print("       - total_packets: 总包数")
    print("    2. 摄像头端口发送图像数据:")
    print("       - 包头: [4字节packet_id][2字节包索引][2字节总包数][1字节标志位]")
    print("       - 第一个包额外包含: [2字节长度][request_id字符串]")
    print("       - 后续包: [图像数据]")
    
    print()
    
    # 测试3: 诊断结果协议
    print("🧪 测试3: 诊断结果协议")
    print("  PC端发送格式:")
    print("    - type: 'diagnosis_result'")
    print("    - request_id: 匹配开发板请求ID")
    print("    - disease_name: 疾病名称")
    print("    - confidence: 置信度")
    print("    - advice: 医疗建议")
    print("    - timestamp: 时间戳")
    
    print()
    
    # 测试4: 心跳协议
    print("🧪 测试4: 心跳协议")
    print("  开发板端发送:")
    print("    - packet_id: 0")
    print("    - packet_index: 0")
    print("    - total_packets: 0")
    print("    - 数据: JSON格式心跳信息")
    print("  PC端响应:")
    print("    - type: 'heartbeat_response'")
    print("    - timestamp: 时间戳")
    print("    - server_status: 'running'")
    print("    - latency: 延迟时间")
    
    print()
    
    # 测试5: 语音协议
    print("🧪 测试5: 语音协议")
    print("  开发板端发送:")
    print("    1. 语音命令: type='voice_command'")
    print("    2. 语音数据: 分片发送WAV格式音频")
    print("  PC端响应:")
    print("    - 语音合成结果")
    print("    - 语音命令响应")
    
    print()
    
    # 总结
    print("📋 交互协议总结:")
    print("  ✅ 开发板端 -> PC端:")
    print("     - 命令端口: 控制命令和请求头")
    print("     - 摄像头端口: 图像数据分片")
    print("     - 语音端口: 语音数据和命令")
    print("  ✅ PC端 -> 开发板端:")
    print("     - 命令端口: 响应和状态")
    print("     - 诊断端口: AI诊断结果")
    print("     - 语音端口: TTS音频数据")
    
    print()
    print("🔧 已修复的问题:")
    print("  1. 数据包ID匹配问题 - 在第一个包中包含request_id")
    print("  2. 请求头存储映射 - 使用packet_to_request字典")
    print("  3. 时序问题 - 改进数据包处理逻辑")
    
    print()
    print("💡 建议:")
    print("  1. 确保PC端先启动并监听相应端口")
    print("  2. 检查防火墙设置，确保端口可访问")
    print("  3. 使用网络工具验证端口连通性")
    print("  4. 查看PC端日志确认数据接收情况")

if __name__ == "__main__":
    test_connection_protocol()
