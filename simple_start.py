#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化启动器 - 直接跳过有问题的依赖
"""

import os
import sys

def main():
    print("🏥 医疗诊断系统 - 简化启动")
    print("=" * 40)
    
    # 直接禁用可能有问题的功能
    os.environ['DISABLE_AUDIO'] = '1'
    os.environ['DISABLE_PYGAME'] = '1'
    
    print("⚡ 快速启动模式:")
    print("   - 音频功能: 禁用")
    print("   - 图形界面: 禁用（控制台模式）")
    print("   - 摄像头功能: 启用")
    print("   - 网络通信: 启用")
    print()
    
    try:
        # 检查OpenCV
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV不可用，请先安装: pip3 install opencv-python")
        return 1
    
    # 启动主系统
    try:
        print("🚀 启动系统...")
        exec(open('board_integrated_system.py').read())
    except FileNotFoundError:
        print("❌ 未找到 board_integrated_system.py")
        return 1
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
