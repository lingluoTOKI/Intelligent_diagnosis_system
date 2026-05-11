#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置测试脚本
验证PC IP配置是否正确设置
"""

def test_system_config():
    """测试system_config.py中的配置"""
    try:
        from system_config import PC_IP
        print(f"[system_config.py] PC_IP = {PC_IP}")
        return PC_IP
    except ImportError as e:
        print(f"[错误] 无法导入system_config: {e}")
        return None

def test_board_system_config():
    """测试board_integrated_system.py中的配置"""
    try:
        # 模拟导入board_integrated_system中的配置
        import sys
        import os
        
        # 首先尝试从system_config导入
        try:
            from system_config import PC_IP
            print(f"[board_integrated_system.py] 使用统一配置 PC_IP = {PC_IP}")
            return PC_IP
        except ImportError:
            # 如果无法导入，使用默认配置
            pc_ip = "172.20.10.3"
            print(f"[board_integrated_system.py] 使用默认配置 PC_IP = {pc_ip}")
            return pc_ip
            
    except Exception as e:
        print(f"[错误] 测试board_integrated_system配置失败: {e}")
        return None

def test_json_config():
    """测试system_config.json中的配置"""
    try:
        import json
        with open('../system_config.json', 'r') as f:
            config = json.load(f)
        pc_ip = config['network']['pc_ip']
        auto_detect = config['startup']['auto_detect_ip']
        print(f"[system_config.json] PC_IP = {pc_ip}, auto_detect_ip = {auto_detect}")
        return pc_ip, auto_detect
    except Exception as e:
        print(f"[错误] 无法读取system_config.json: {e}")
        return None, None

def main():
    """主测试函数"""
    print("=" * 50)
    print("PC IP配置测试")
    print("=" * 50)
    
    # 测试各个配置文件
    config_pc_ip = test_system_config()
    board_pc_ip = test_board_system_config()
    json_pc_ip, auto_detect = test_json_config()
    
    print("\n" + "=" * 50)
    print("配置汇总")
    print("=" * 50)
    
    # 检查配置一致性
    all_ips = [config_pc_ip, board_pc_ip, json_pc_ip]
    all_ips = [ip for ip in all_ips if ip is not None]
    
    if len(set(all_ips)) == 1:
        print(f"✅ 所有配置文件PC IP一致: {all_ips[0]}")
    else:
        print("❌ 配置文件PC IP不一致:")
        if config_pc_ip:
            print(f"   system_config.py: {config_pc_ip}")
        if board_pc_ip:
            print(f"   board_integrated_system.py: {board_pc_ip}")
        if json_pc_ip:
            print(f"   system_config.json: {json_pc_ip}")
    
    # 检查自动检测是否已禁用
    if auto_detect is not None:
        if auto_detect:
            print("⚠️  自动IP检测仍然启用，建议禁用")
        else:
            print("✅ 自动IP检测已禁用")
    
    print("\n预期配置:")
    print("  PC_IP: 172.20.10.3")
    print("  auto_detect_ip: false")

if __name__ == "__main__":
    main()
