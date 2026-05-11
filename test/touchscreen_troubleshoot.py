#!/usr/bin/env python3
"""
触摸屏问题诊断和测试工具
帮助解决开发板触摸屏无法控制PC的问题
"""

import socket
import time
import subprocess
import os
import sys

def test_network_connectivity():
    """测试网络连通性"""
    print("[测试] 网络连通性...")
    
    pc_ip = "172.20.10.3"  # 修改为您的PC IP
    
    try:
        # 测试ping连通性
        result = subprocess.run(['ping', '-c', '1', pc_ip], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print(f"[成功] 网络连通正常 - {pc_ip}")
            return True
        else:
            print(f"[错误] 无法连接到PC - {pc_ip}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[错误] ping超时 - {pc_ip}")
        return False
    except Exception as e:
        print(f"[错误] 网络测试失败: {e}")
        return False

def test_video_port():
    """测试视频端口连接"""
    print("[测试] 视频端口连接...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 5000))
        sock.settimeout(3.0)
        
        print("[成功] 视频端口5000可以绑定")
        
        # 尝试接收数据
        print("[等待] 等待PC端视频数据...")
        try:
            data, addr = sock.recvfrom(1024)
            print(f"[成功] 收到来自 {addr[0]} 的数据: {len(data)} 字节")
            return True
        except socket.timeout:
            print("[警告] 3秒内未收到视频数据")
            return False
            
    except Exception as e:
        print(f"[错误] 视频端口测试失败: {e}")
        return False
    finally:
        try:
            sock.close()
        except:
            pass

def test_control_port():
    """测试控制端口发送"""
    print("[测试] 控制端口发送...")
    
    pc_ip = "172.20.10.3"
    control_port = 5001
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 发送测试命令（鼠标移动到屏幕中央）
        test_cmd = b'\x00' + (960).to_bytes(2, 'big') + (540).to_bytes(2, 'big')
        
        sock.sendto(test_cmd, (pc_ip, control_port))
        print(f"[成功] 测试命令已发送到 {pc_ip}:{control_port}")
        
        # 发送几个移动命令测试
        for i in range(5):
            x = 960 + i * 10
            y = 540 + i * 10
            cmd = b'\x00' + x.to_bytes(2, 'big') + y.to_bytes(2, 'big')
            sock.sendto(cmd, (pc_ip, control_port))
            time.sleep(0.1)
        
        print("[成功] 控制命令发送完成")
        return True
        
    except Exception as e:
        print(f"[错误] 控制端口测试失败: {e}")
        return False
    finally:
        try:
            sock.close()
        except:
            pass

def check_opencv_mouse():
    """检查OpenCV鼠标事件"""
    print("[测试] OpenCV鼠标事件...")
    
    try:
        import cv2
        import numpy as np
        
        # 创建测试窗口
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Touch Test - Click anywhere", 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        click_detected = False
        
        def test_mouse_callback(event, x, y, flags, param):
            nonlocal click_detected
            if event == cv2.EVENT_LBUTTONDOWN:
                click_detected = True
                print(f"[成功] 检测到鼠标点击: ({x}, {y})")
                cv2.circle(test_image, (x, y), 10, (0, 255, 0), -1)
                cv2.imshow("Touch Test", test_image)
        
        cv2.namedWindow("Touch Test")
        cv2.setMouseCallback("Touch Test", test_mouse_callback)
        
        print("[测试] 请点击测试窗口，按ESC退出...")
        
        start_time = time.time()
        while time.time() - start_time < 10:  # 10秒测试时间
            cv2.imshow("Touch Test", test_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                break
            if click_detected:
                break
        
        cv2.destroyAllWindows()
        
        if click_detected:
            print("[成功] OpenCV鼠标事件正常")
            return True
        else:
            print("[警告] 未检测到鼠标点击")
            return False
            
    except Exception as e:
        print(f"[错误] OpenCV测试失败: {e}")
        return False

def display_solution():
    """显示解决方案"""
    print("\n" + "="*60)
    print("[解决方案] 触摸屏无法使用的常见解决方法")
    print("="*60)
    
    print("\n1. [网络问题]")
    print("   - 确保开发板和PC在同一网络")
    print("   - 检查IP地址配置是否正确")
    print("   - 确认防火墙未阻止端口5000和5001")
    
    print("\n2. [PC端程序问题]")
    print("   - 确保PC端运行了带鼠标控制的程序:")
    print("     python link/wifi_pc_sender_with_mouse.py")
    print("   - 或者修改原程序添加鼠标控制功能")
    
    print("\n3. [开发板端问题]")
    print("   - 确保正确运行接收程序:")
    print("     python link/wifi_pc_receiver_2.0.py")
    print("   - 检查触摸屏驱动是否正常")

def main():
    """主诊断流程"""
    print("触摸屏问题诊断工具")
    print("=" * 40)
    
    # 测试网络
    if not test_network_connectivity():
        print("[建议] 请检查网络连接和IP配置")
    
    # 测试视频端口
    print("\n[提示] 请确保PC端已启动发送程序，然后按Enter继续...")
    input()
    
    if test_video_port():
        print("[成功] 视频数据正常")
    else:
        print("[建议] 请启动PC端发送程序")
    
    # 测试控制端口
    test_control_port()
    
    # 显示解决方案
    display_solution()

if __name__ == "__main__":
    main()
