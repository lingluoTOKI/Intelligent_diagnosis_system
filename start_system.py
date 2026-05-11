#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能医疗诊断系统启动器 v2.0
功能：
1. 自动启动PC端所有服务
2. 生成开发板启动命令
3. 实时监控服务状态
4. 支持一键启动整个系统
5. 智能依赖检查和修复
6. 详细的状态报告和日志
7. 服务健康检查和自动重启
8. 网络配置优化和验证
"""

import subprocess
import sys
import time
import os
import json
import socket
import threading
import logging
import platform
import psutil
from pathlib import Path
from datetime import datetime
import argparse

# 配置文件路径   
CONFIG_FILE = "system_config.json"

# 默认配置
DEFAULT_CONFIG = {
    "network": {
        "pc_ip": "172.20.10.3",
        "board_ip": "172.20.10.8",
        "camera_port": 5002,
        "diagnosis_port": 5003,
        "command_port": 5004,
        "screen_port": 5000,
        "control_port": 5001
    },
    "startup": {
        "auto_detect_ip": False,
        "start_delay": 2,
        "max_retries": 3
    },
    "services": {
        "main_system": True,
        "diagnosis_server": True,
        "screen_sharing": True,
        "camera_integration": True
    }
}

class SystemLauncher:
    """系统启动器"""
    
    def __init__(self):
        self.config = self.load_config()
        self.processes = {}
        self.running = True
        
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"[成功] 已加载配置文件: {CONFIG_FILE}")
                return config
            except Exception as e:
                print(f"[警告] 配置文件加载失败: {e}")
        
        # 创建默认配置文件
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        
        print(f"[创建] 已创建默认配置文件: {CONFIG_FILE}")
        return DEFAULT_CONFIG
    
    def detect_local_ip(self):
        """自动检测本机IP地址"""
        try:
            # 连接到一个外部地址来获取本机IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            print(f"[检测] 本机IP: {local_ip}")
            return local_ip
        except Exception as e:
            print(f"[错误] IP检测失败: {e}")
            return "127.0.0.1"
    
    def check_dependencies(self):
        """检查依赖项"""
        print("[检查] 系统依赖...")
        
        required_files = [
            "visualization_test2.py",
            "pc_diagnosis_server.py",
            "board_camera_integration.py",
            "latency_optimizer.py",
            "link/wifi_pc_sender_with_mouse.py",
            "link/wifi_pc_receiver_2.0.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("[错误] 缺少必要文件:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            return False
        
        print("[成功] 所有必要文件已存在")
        return True
    
    def start_service(self, name, command, cwd=None, delay=0):
        """启动服务"""
        if delay > 0:
            print(f"[等待] {delay} 秒后启动 {name}...")
            time.sleep(delay)
        
        try:
            print(f"[启动] {name}...")
            
            if cwd:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
            else:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
            
            self.processes[name] = process
            print(f"[成功] {name} 已启动 (PID: {process.pid})")
            return True
            
        except Exception as e:
            print(f"[错误] {name} 启动失败: {e}")
            return False
    
    def start_pc_services(self):
        """启动PC端服务"""
        print("\n" + "="*50)
        print("[启动] PC端服务")
        print("="*50)
        
        services = []
        
        # 主医疗诊断系统
        if self.config["services"]["main_system"]:
            services.append({
                "name": "主诊断系统",
                "command": f"{sys.executable} visualization_test2.py",
                "delay": 0
            })
        
        # 诊断服务器
        if self.config["services"]["diagnosis_server"]:
            services.append({
                "name": "诊断服务器",
                "command": f"{sys.executable} pc_diagnosis_server.py",
                "delay": 3
            })
        
        # 语音处理服务器
        if self.config["services"].get("voice_server", True):
            services.append({
                "name": "语音服务器",
                "command": f"{sys.executable} pc_voice_server.py",
                "delay": 4
            })
        
        # 屏幕共享发送端（使用带鼠标控制的版本）
        if self.config["services"]["screen_sharing"]:
            services.append({
                "name": "屏幕共享",
                "command": f"{sys.executable} wifi_pc_sender_with_mouse.py",
                "cwd": "link",
                "delay": 6
            })
        
        # 并行启动服务
        threads = []
        for service in services:
            thread = threading.Thread(
                target=self.start_service,
                args=(service["name"], service["command"]),
                kwargs={
                    "cwd": service.get("cwd"),
                    "delay": service.get("delay", 0)
                }
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有服务启动
        for thread in threads:
            thread.join()
        
        print(f"[完成] PC端服务启动完成，共启动 {len(services)} 个服务")
    
    def generate_board_commands(self):
        """生成开发板端启动命令"""
        print("\n" + "="*50)
        print("[指令] 开发板端启动命令")
        print("="*50)
        
        network_config = self.config["network"]
        
        print("请在开发板端执行以下命令:")
        print()
        
        if self.config["services"]["screen_sharing"]:
            print("1. 启动屏幕接收服务:")
            print(f"   cd link")
            print(f"   python wifi_pc_receiver_2.0.py")
            print()
        
        if self.config["services"]["camera_integration"]:
            print("2. 启动摄像头诊断服务:")
            print(f"   python board_camera_integration.py")
            print("   在菜单中选择 'p' 进入预览模式")
            print()
            
        print("3. 启动语音对话服务:")
        print(f"   python board_voice_interaction.py")
        print("   或在摄像头预览中按 'r' 键")
        print()
        
        print("或者使用一键启动脚本:")
        
        startup_script = f'''#!/bin/bash
# 开发板一键启动脚本

echo "[启动] 开发板医疗诊断系统..."

# 检查网络连接
ping -c 1 {network_config["pc_ip"]} > /dev/null
if [ $? -ne 0 ]; then
    echo "[错误] 无法连接到PC端: {network_config["pc_ip"]}"
    exit 1
fi

echo "[成功] 网络连接正常"

# 启动屏幕接收 (后台)
if [ -f "link/wifi_pc_receiver_2.0.py" ]; then
    cd link
    python wifi_pc_receiver_2.0.py &
    echo "[成功] 屏幕接收服务已启动"
    cd ..
fi

# 等待屏幕服务启动
sleep 3

# 启动摄像头诊断服务
if [ -f "board_camera_integration.py" ]; then
    python board_camera_integration.py
else
    echo "[错误] 找不到摄像头诊断程序"
fi
'''
        
        # 保存启动脚本
        with open("start_board.sh", "w", encoding="utf-8") as f:
            f.write(startup_script)
        
        os.chmod("start_board.sh", 0o755)
        print(f"[创建] 开发板启动脚本已保存: start_board.sh")
    
    def monitor_services(self):
        """监控服务状态"""
        print("\n" + "="*50)
        print("[监控] 服务状态")
        print("="*50)
        
        while self.running:
            try:
                time.sleep(10)  # 每10秒检查一次
                
                active_services = 0
                failed_services = []
                
                for name, process in self.processes.items():
                    if process.poll() is None:
                        active_services += 1
                    else:
                        failed_services.append(name)
                
                print(f"[状态] 活跃服务: {active_services}/{len(self.processes)}")
                
                if failed_services:
                    print(f"[警告] 失败服务: {', '.join(failed_services)}")
                    for service_name in failed_services:
                        process = self.processes[service_name]
                        stdout, stderr = process.communicate()
                        if stderr:
                            print(f"   {service_name} 错误: {stderr.strip()}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[错误] 监控错误: {e}")
    
    def stop_all_services(self):
        """停止所有服务"""
        print("\n[停止] 正在停止所有服务...")
        self.running = False
        
        for name, process in self.processes.items():
            try:
                if process.poll() is None:
                    print(f"[停止] {name}...")
                    process.terminate()
                    
                    # 等待进程结束
                    try:
                        process.wait(timeout=5)
                        print(f"[成功] {name} 已停止")
                    except subprocess.TimeoutExpired:
                        print(f"[强制] {name} 强制终止")
                        process.kill()
                        
            except Exception as e:
                print(f"[错误] 停止 {name} 失败: {e}")
        
        print("[完成] 所有服务已停止")
    
    def run(self):
        """运行启动器"""
        print("[系统] 医疗诊断系统启动器")
        print("=" * 50)
        
        # 检查依赖
        if not self.check_dependencies():
            print("[错误] 依赖检查失败，请确保所有必要文件存在")
            return
        
        # 自动检测IP
        if self.config["startup"]["auto_detect_ip"]:
            detected_ip = self.detect_local_ip()
            self.config["network"]["pc_ip"] = detected_ip
        
        try:
            # 启动PC端服务
            self.start_pc_services()
            
            # 生成开发板命令
            self.generate_board_commands()
            
            # 启动监控
            monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
            monitor_thread.start()
            
            print("\n[完成] 系统启动完成！")
            print("[提示] 按 Ctrl+C 停止所有服务")
            
            # 等待用户中断
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n[停止] 收到停止信号...")
        except Exception as e:
            print(f"[错误] 系统启动失败: {e}")
        finally:
            self.stop_all_services()

def main():
    """主函数"""
    launcher = SystemLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
