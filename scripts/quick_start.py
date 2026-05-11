#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本 - 智能医疗诊断系统
自动检测网络配置并启动相应的系统组件
"""

import os
import sys
import socket
import subprocess
import time
import json
from datetime import datetime

def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                     🏥 快速启动助手                          ║
║                  智能医疗诊断系统 v2.0                       ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def get_local_ip():
    """获取本机IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def detect_environment():
    """检测运行环境"""
    print("🔍 检测运行环境...")
    
    env_info = {
        "local_ip": get_local_ip(),
        "has_display": os.environ.get('DISPLAY') is not None,
        "is_jupyter": 'ipykernel' in sys.modules,
        "platform": sys.platform
    }
    
    print(f"📡 本机IP: {env_info['local_ip']}")
    print(f"🖥️ 图形界面: {'✅' if env_info['has_display'] else '❌'}")
    print(f"📓 Jupyter环境: {'✅' if env_info['is_jupyter'] else '❌'}")
    print(f"💻 平台: {env_info['platform']}")
    
    return env_info

def test_pc_connection(ip, port=5004):
    """测试PC端连接"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        
        # 发送测试包
        test_data = json.dumps({
            "type": "connection_test",
            "timestamp": datetime.now().isoformat()
        }).encode('utf-8')
        
        sock.sendto(test_data, (ip, port))
        
        # 等待响应
        response, addr = sock.recvfrom(1024)
        response_data = json.loads(response.decode('utf-8'))
        
        sock.close()
        return response_data.get('type') == 'connection_test_response'
        
    except:
        return False

def find_pc_ip():
    """查找PC端IP地址"""
    print("\n🔍 搜索PC端...")
    
    local_ip = get_local_ip()
    ip_parts = local_ip.split('.')
    ip_base = '.'.join(ip_parts[:3])
    
    # 候选IP列表
    candidate_ips = [
        local_ip,  # 本机（可能是同一设备）
        f"{ip_base}.1",    # 路由器
        f"{ip_base}.100",  # 常见PC IP
        f"{ip_base}.101",
        f"{ip_base}.102",
        f"{ip_base}.10",
        f"{ip_base}.2",
        f"{ip_base}.3",
    ]
    
    for ip in candidate_ips:
        print(f"   测试 {ip}...", end=" ")
        if test_pc_connection(ip):
            print("✅")
            return ip
        else:
            print("❌")
    
    return None

def choose_mode():
    """选择运行模式"""
    print("\n🎯 选择运行模式:")
    print("1. 🖥️  PC端模式 (主界面和AI诊断)")
    print("2. 📱 开发板模式 (摄像头和触摸界面)")
    print("3. 🔄 智能模式 (自动检测并启动)")
    print("4. ⚙️  网络配置")
    
    while True:
        try:
            choice = input("\n请选择 (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("❌ 请输入 1-4")
        except KeyboardInterrupt:
            print("\n👋 退出")
            sys.exit(0)

def start_pc_mode():
    """启动PC端模式"""
    print("\n🖥️ 启动PC端...")
    
    if not os.path.exists('visualization_test2.py'):
        print("❌ 找不到 visualization_test2.py")
        return False
    
    try:
        process = subprocess.Popen([sys.executable, 'visualization_test2.py'])
        print(f"✅ PC端已启动 (PID: {process.pid})")
        print("💡 PC端窗口将在几秒内出现")
        return True
    except Exception as e:
        print(f"❌ PC端启动失败: {e}")
        return False

def start_board_mode():
    """启动开发板模式"""
    print("\n📱 启动开发板...")
    
    if not os.path.exists('src/board/board_integrated_system.py'):
        print("❌ 找不到 src/board/board_integrated_system.py")
        return False
    
    try:
        process = subprocess.Popen([sys.executable, 'src/board/board_integrated_system.py'])
        print(f"✅ 开发板已启动 (PID: {process.pid})")
        print("📷 摄像头和触摸界面应该已激活")
        return True
    except Exception as e:
        print(f"❌ 开发板启动失败: {e}")
        return False

def smart_mode():
    """智能模式 - 自动检测环境并启动"""
    print("\n🤖 智能模式启动...")
    
    env = detect_environment()
    
    # 检测是否已有PC端运行
    pc_ip = find_pc_ip()
    
    if pc_ip:
        print(f"✅ 发现PC端: {pc_ip}")
        print("📱 启动开发板模式连接到PC端")
        
        # 创建临时配置文件
        create_temp_config(pc_ip)
        return start_board_mode()
    else:
        print("❌ 未发现PC端")
        
        if env['has_display']:
            print("🖥️ 检测到图形界面，启动PC端模式")
            return start_pc_mode()
        else:
            print("📱 无图形界面，启动开发板控制台模式")
            return start_board_mode()

def create_temp_config(pc_ip):
    """创建临时配置文件"""
    config = {
        "pc_ip": pc_ip,
        "auto_detected": True,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        with open("network_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"📝 已创建网络配置: {pc_ip}")
    except Exception as e:
        print(f"⚠️ 配置文件创建失败: {e}")

def configure_network():
    """配置网络"""
    print("\n⚙️ 网络配置")
    
    local_ip = get_local_ip()
    print(f"📡 本机IP: {local_ip}")
    
    # 搜索PC端
    pc_ip = find_pc_ip()
    
    if pc_ip:
        print(f"\n✅ 自动发现PC端: {pc_ip}")
        create_temp_config(pc_ip)
    else:
        print("\n❌ 未能自动发现PC端")
        print("请手动输入PC端IP地址:")
        
        while True:
            try:
                manual_ip = input("PC端IP: ").strip()
                if not manual_ip:
                    break
                
                # 验证IP格式
                parts = manual_ip.split('.')
                if len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts):
                    if test_pc_connection(manual_ip):
                        print(f"✅ 连接成功: {manual_ip}")
                        create_temp_config(manual_ip)
                        break
                    else:
                        print(f"❌ 无法连接到: {manual_ip}")
                else:
                    print("❌ IP格式无效")
                    
            except ValueError:
                print("❌ IP格式无效")
            except KeyboardInterrupt:
                print("\n取消配置")
                break

def main():
    """主函数"""
    print_banner()
    
    # 检测环境
    env = detect_environment()
    
    # 选择模式
    mode = choose_mode()
    
    success = False
    
    if mode == 1:
        success = start_pc_mode()
    elif mode == 2:
        success = start_board_mode()
    elif mode == 3:
        success = smart_mode()
    elif mode == 4:
        configure_network()
        return
    
    if success:
        print("\n✅ 系统启动成功！")
        print("\n💡 使用提示:")
        print("   - PC端: 使用图形界面进行AI诊断")
        print("   - 开发板: 使用触摸屏或控制台命令")
        print("   - 网络: 确保两端在同一网络")
        
        try:
            input("\n按回车键退出...")
        except KeyboardInterrupt:
            pass
    else:
        print("\n❌ 系统启动失败")
        print("\n🔧 故障排除:")
        print("   1. 检查文件是否存在")
        print("   2. 检查Python环境和依赖")
        print("   3. 检查网络连接")
        print("   4. 运行模式4配置网络")

if __name__ == "__main__":
    main()












