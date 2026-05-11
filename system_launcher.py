#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能医疗诊断系统启动器
协调PC端和开发板端的启动和通信
"""

import sys
import os
import time
import json
import threading
import subprocess
from datetime import datetime

# 检查并设置环境
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def print_banner():
    """打印系统横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    智能医疗诊断系统                          ║
║                  AI Eye Disease Diagnosis System             ║
║                                                              ║
║  PC端: visualization_test2.py - 主界面和AI诊断               ║
║  开发板: board_integrated_system.py - 摄像头和触摸界面       ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """检查系统依赖"""
    print("🔍 检查系统依赖...")
    
    required_modules = [
        'PyQt5', 'cv2', 'numpy', 'requests', 'ultralytics',
        'matplotlib', 'speech_recognition', 'pyttsx3'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module}")
    
    if missing_modules:
        print(f"\n⚠️ 缺少依赖模块: {', '.join(missing_modules)}")
        print("请使用以下命令安装:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    
    print("✅ 所有依赖检查完成\n")
    return True

def check_system_files():
    """检查系统文件"""
    print("📁 检查系统文件...")
    
    files_to_check = [
        'visualization_test2.py',
        'board_integrated_system.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在")
            return False
    
    print("✅ 系统文件检查完成\n")
    return True

def get_system_mode():
    """获取系统运行模式"""
    print("🎯 选择运行模式:")
    print("1. PC端模式 (运行主界面)")
    print("2. 开发板模式 (运行开发板系统)")
    print("3. 双端模式 (同时启动两端)")
    print("4. 仅检查配置")
    
    while True:
        try:
            choice = input("\n请选择模式 (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("❌ 无效选择，请输入 1-4")
        except KeyboardInterrupt:
            print("\n\n👋 用户取消操作")
            sys.exit(0)

def create_default_config():
    """创建默认配置文件"""
    config_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统统一配置文件
"""

import socket

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

# 网络配置
PC_IP = get_local_ip()  # PC端IP地址，自动获取
BOARD_IP = "192.168.1.100"  # 开发板IP地址，需要根据实际情况修改

# 端口配置
NETWORK_PORTS = {
    "camera": 5002,        # 摄像头数据端口
    "diagnosis": 5003,     # 诊断结果端口
    "command": 5004,       # 命令控制端口
    "voice_send": 5005,    # 语音发送端口
    "voice_receive": 5006, # 语音接收端口
    "voice_command": 5007, # 语音命令端口
    "touch_control": 5008  # 触摸控制端口
}

# 摄像头配置
CAMERA_CONFIG = {
    "width": 640,
    "height": 480,
    "fps": 30,
    "quality": 85
}

# 音频配置
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024,
    "format": "wav"
}

# 系统配置
SYSTEM_CONFIG = {
    "debug_mode": True,
    "log_level": "INFO",
    "auto_connect": True,
    "heartbeat_interval": 5.0
}

# 连接管理器（占位符）
connection_manager = None

print(f"📡 系统配置加载完成，PC IP: {PC_IP}")
'''
    
    with open('system_config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✅ 已创建默认配置文件 system_config.py")

def check_config():
    """检查配置文件"""
    print("⚙️ 检查配置文件...")
    
    try:
        import system_config
        print(f"✅ 配置文件已加载")
        print(f"📡 PC IP: {system_config.PC_IP}")
        print(f"🔌 端口配置: {system_config.NETWORK_PORTS}")
        return True
    except ImportError:
        print("❌ 未找到 system_config.py")
        print("🔧 正在创建默认配置文件...")
        create_default_config()
        return True

def launch_pc_mode():
    """启动PC端模式"""
    print("🖥️ 启动PC端主界面...")
    print("📝 启动命令: python visualization_test2.py")
    print("⏳ 正在启动，请稍候...")
    
    try:
        # 使用subprocess启动PC端
        process = subprocess.Popen(
            [sys.executable, 'visualization_test2.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ PC端已启动 (PID: {process.pid})")
        print("💡 PC端窗口应该会在几秒内出现")
        print("📱 如需连接开发板，请在PC端界面中启用开发板交互功能")
        
        return process
        
    except Exception as e:
        print(f"❌ PC端启动失败: {e}")
        return None

def launch_board_mode():
    """启动开发板模式"""
    print("📱 启动开发板系统...")
    print("📝 启动命令: python board_integrated_system.py")
    print("⏳ 正在启动，请稍候...")
    
    try:
        # 使用subprocess启动开发板端
        process = subprocess.Popen(
            [sys.executable, 'board_integrated_system.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ 开发板系统已启动 (PID: {process.pid})")
        print("📷 摄像头和触摸界面应该已激活")
        print("🔗 系统将自动尝试连接PC端")
        
        return process
        
    except Exception as e:
        print(f"❌ 开发板系统启动失败: {e}")
        return None

def launch_dual_mode():
    """启动双端模式"""
    print("🔄 启动双端模式...")
    
    # 先启动PC端
    pc_process = launch_pc_mode()
    if not pc_process:
        return None, None
    
    # 等待PC端启动
    print("⏳ 等待PC端完全启动...")
    time.sleep(3)
    
    # 启动开发板端
    board_process = launch_board_mode()
    if not board_process:
        print("⚠️ 开发板启动失败，但PC端仍在运行")
        return pc_process, None
    
    print("✅ 双端模式启动完成")
    print("🔗 系统将自动建立连接")
    
    return pc_process, board_process

def monitor_processes(processes):
    """监控进程状态"""
    print("\n📊 进程监控已启动")
    print("💡 按 Ctrl+C 停止所有服务")
    
    try:
        while True:
            time.sleep(5)
            
            # 检查进程状态
            active_processes = []
            for name, process in processes.items():
                if process and process.poll() is None:
                    active_processes.append(name)
                elif process:
                    print(f"⚠️ {name} 进程已退出 (返回码: {process.returncode})")
            
            if active_processes:
                print(f"✅ 运行中: {', '.join(active_processes)}", end='\r')
            else:
                print("\n❌ 所有进程已退出")
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号，正在关闭所有服务...")
        
        # 终止所有进程
        for name, process in processes.items():
            if process and process.poll() is None:
                print(f"🔄 停止 {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                    print(f"✅ {name} 已停止")
                except subprocess.TimeoutExpired:
                    print(f"⚠️ 强制停止 {name}...")
                    process.kill()
        
        print("✅ 所有服务已停止")

def main():
    """主函数"""
    print_banner()
    
    # 检查依赖和文件
    if not check_dependencies():
        return
    
    if not check_system_files():
        return
    
    # 检查配置
    if not check_config():
        return
    
    # 获取运行模式
    mode = get_system_mode()
    
    if mode == 4:
        print("✅ 配置检查完成")
        return
    
    print(f"\n🚀 启动模式: {mode}")
    
    # 根据模式启动服务
    processes = {}
    
    if mode == 1:
        pc_process = launch_pc_mode()
        if pc_process:
            processes['PC端'] = pc_process
    
    elif mode == 2:
        board_process = launch_board_mode()
        if board_process:
            processes['开发板'] = board_process
    
    elif mode == 3:
        pc_process, board_process = launch_dual_mode()
        if pc_process:
            processes['PC端'] = pc_process
        if board_process:
            processes['开发板'] = board_process
    
    # 监控进程
    if processes:
        monitor_processes(processes)
    else:
        print("❌ 没有成功启动的服务")

if __name__ == "__main__":
    main()
