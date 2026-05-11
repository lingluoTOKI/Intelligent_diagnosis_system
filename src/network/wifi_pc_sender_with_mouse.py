#!/usr/bin/env python3
"""
PC端屏幕共享服务
功能：
1. 发送屏幕画面到开发板
2. 接收开发板的鼠标/触摸控制指令
3. 在PC上执行鼠标操作
4. 支持实时屏幕共享和远程控制
"""

import socket
import time
import numpy as np
import cv2
import mss
import math
import threading
import pyautogui

# ===== 参数设置 =====
BOARD_IP = "172.20.10.8"
VIDEO_PORT = 5000        # 发送视频数据的端口
CONTROL_PORT = 5001      # 接收控制指令的端口
TARGET_FPS = 30
MIN_FPS = 20
SCALE_FACTOR = 1.0
MAX_PACKET_SIZE = 1400
JPEG_QUALITY = 75
DIFF_THRESHOLD = 0.02
KEYFRAME_INTERVAL = 15
# ===================

# 设置pyautogui
pyautogui.FAILSAFE = False  # 禁用安全模式
pyautogui.PAUSE = 0.01      # 减少操作间隔

class MouseController:
    """鼠标控制器"""
    
    def __init__(self):
        self.last_move_time = 0
        self.move_threshold = 0.01  # 鼠标移动节流时间(秒)
        self.is_running = True
        
    def handle_mouse_command(self, command_data):
        """处理鼠标控制命令"""
        try:
            if len(command_data) < 5:
                return
            
            # 解析命令：1字节命令类型 + 2字节X坐标 + 2字节Y坐标
            cmd_type = command_data[0]
            x = int.from_bytes(command_data[1:3], 'big')
            y = int.from_bytes(command_data[3:5], 'big')
            
            current_time = time.time()
            
            if cmd_type == 0x00:  # 鼠标移动
                # 节流控制，避免过于频繁的移动
                if current_time - self.last_move_time > self.move_threshold:
                    pyautogui.moveTo(x, y, duration=0)
                    self.last_move_time = current_time
                    
            elif cmd_type == 0x01:  # 鼠标按下
                pyautogui.mouseDown(x, y, button='left')
                print(f"[鼠标] 按下 ({x}, {y})")
                
            elif cmd_type == 0x02:  # 鼠标释放
                pyautogui.mouseUp(x, y, button='left')
                print(f"[鼠标] 释放 ({x}, {y})")
                
        except Exception as e:
            print(f"[错误] 鼠标控制处理失败: {e}")

class VideoEncoder:
    """视频编码器"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.jpeg_quality = JPEG_QUALITY
        self.prev_frame = None
        self.frame_counter = 0
        self.encode = self._encode_software
        print(f"[编码] 使用软件编码 | 固定JPEG质量: {self.jpeg_quality}")

    def _frame_change_detection(self, frame):
        """帧变化检测"""
        if self.prev_frame is None:
            return True

        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        edge_current = cv2.Canny(gray_current, 50, 150)
        edge_prev = cv2.Canny(gray_prev, 50, 150)
        edge_diff = cv2.absdiff(edge_current, edge_prev)

        gray_diff = cv2.absdiff(gray_current, gray_prev)
        combined_diff = cv2.bitwise_or(gray_diff, edge_diff)
        change_ratio = np.count_nonzero(combined_diff) / (gray_current.size)

        is_keyframe = (self.frame_counter % KEYFRAME_INTERVAL == 0)
        return is_keyframe or (change_ratio > DIFF_THRESHOLD)

    def _encode_software(self, frame):
        """软件编码"""
        if not self._frame_change_detection(frame):
            return [b'\x00']

        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ]
        _, img_encoded = cv2.imencode(".jpg", frame, encode_params)
        encoded_data = img_encoded.tobytes()
        self.prev_frame = frame.copy()
        self.frame_counter += 1

        total_packets = math.ceil(len(encoded_data) / MAX_PACKET_SIZE)
        packets = []
        is_keyframe = (self.frame_counter % KEYFRAME_INTERVAL == 0)
        keyframe_flag = b'\x01' if is_keyframe else b'\x00'

        for i in range(total_packets):
            start = i * MAX_PACKET_SIZE
            end = start + MAX_PACKET_SIZE
            payload = encoded_data[start:end]
            frame_seq = (self.frame_counter % 65536).to_bytes(2, 'big')
            packet_header = keyframe_flag + frame_seq + bytes([i]) + bytes([total_packets])
            packets.append(packet_header + payload)

        return packets

def get_screen_size():
    """获取屏幕尺寸信息"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        original_width, original_height = monitor["width"], monitor["height"]
        scaled_width = int(original_width * SCALE_FACTOR)
        scaled_height = int(original_height * SCALE_FACTOR)
        
        print(f"[屏幕] 原始分辨率: {original_width}x{original_height}")
        print(f"[分辨率] 使用分辨率: {scaled_width}x{scaled_height} (缩放比例: {SCALE_FACTOR})")
        print(f"[网络] 单包最大数据量: {MAX_PACKET_SIZE}字节（不含4字节头部）")
        return scaled_width, scaled_height

def mouse_control_server(mouse_controller):
    """鼠标控制服务器线程"""
    print("[鼠标] 启动鼠标控制接收服务器...")
    
    try:
        # 创建控制指令接收套接字
        control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        control_sock.bind(("0.0.0.0", CONTROL_PORT))
        control_sock.settimeout(0.1)  # 短超时，避免阻塞
        
        print(f"[鼠标] 鼠标控制服务器已启动，监听端口 {CONTROL_PORT}")
        
        while mouse_controller.is_running:
            try:
                data, addr = control_sock.recvfrom(16)  # 控制命令很短
                mouse_controller.handle_mouse_command(data)
                
            except socket.timeout:
                continue
            except Exception as e:
                if mouse_controller.is_running:
                    print(f"[错误] 鼠标控制接收错误: {e}")
                    
    except Exception as e:
        print(f"[错误] 鼠标控制服务器启动失败: {e}")
    finally:
        try:
            control_sock.close()
        except:
            pass
        print("[鼠标] 鼠标控制服务器已关闭")

def main():
    """主函数"""
    # 获取屏幕尺寸
    WIDTH, HEIGHT = get_screen_size()

    # 初始化编码器和网络
    encoder = VideoEncoder(WIDTH, HEIGHT)
    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)

    # 初始化鼠标控制器
    mouse_controller = MouseController()
    
    # 启动鼠标控制服务器线程
    mouse_thread = threading.Thread(target=mouse_control_server, args=(mouse_controller,), daemon=True)
    mouse_thread.start()

    # 屏幕捕获设置
    monitor = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}
    sct = mss.mss()

    # 性能统计
    stats = {
        "frame_count": 0,
        "start_time": time.time(),
        "total_data": 0,
        "loss_count": 0,
        "last_delay": 0
    }

    # 帧率控制
    frame_interval = 1.0 / TARGET_FPS
    next_frame_time = time.time()
    drift_compensation = 0.0

    print(f"[启动] 开始传输到 {BOARD_IP}:{VIDEO_PORT}...（按Ctrl+C终止）")
    print(f"[鼠标] 鼠标控制已启用，开发板可控制PC鼠标")

    try:
        while True:
            current_time = time.time()
            
            # 帧率控制与延迟计算
            sleep_time = next_frame_time - current_time + drift_compensation
            if sleep_time > 0:
                time.sleep(max(0, sleep_time - 0.0005))
                actual_sleep = time.time() - current_time
                drift_compensation = sleep_time - actual_sleep
            else:
                drift_compensation = 0

            frame_delay = (time.time() - current_time) * 1000
            stats["last_delay"] = frame_delay
            next_frame_time += frame_interval

            # 捕获屏幕
            screen = sct.grab(monitor)
            frame = np.frombuffer(screen.bgra, dtype=np.uint8).reshape(
                screen.height, screen.width, 4)[:, :, :3]

            if (frame.shape[1], frame.shape[0]) != (WIDTH, HEIGHT):
                frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

            # 编码发送
            packets = encoder.encode(frame)
            if not packets:
                continue

            # 发送数据包
            for i, packet in enumerate(packets):
                try:
                    video_sock.sendto(packet, (BOARD_IP, VIDEO_PORT))
                    stats["total_data"] += len(packet)
                    if len(packet) > 1000 and i % 2 == 0:
                        time.sleep(0.0005)
                except Exception as e:
                    stats["loss_count"] += 1
                    if packet[0] == 1:
                        print(f"\n[警告] 关键帧丢包（累计{stats['loss_count']}次）")

            # 性能统计
            stats["frame_count"] += 1
            elapsed = time.time() - stats["start_time"]
            if elapsed >= 2.0:
                fps = stats["frame_count"] / elapsed
                mbps = (stats["total_data"] * 8 / (1024 * 1024)) / elapsed
                loss_rate = (stats["loss_count"] / (stats["frame_count"] * len(packets))) * 100 if stats["frame_count"] > 0 else 0

                print(
                    f"[状态] FPS: {fps:.1f} | 延迟: {stats['last_delay']:.1f}ms | "
                    f"带宽: {mbps:.2f} Mbps | 丢包率: {loss_rate:.1f}% | 模式: JPEG",
                    end='\r'
                )

                stats = {
                    "frame_count": 0,
                    "start_time": time.time(),
                    "total_data": 0,
                    "loss_count": stats["loss_count"],
                    "last_delay": stats["last_delay"]
                }

    except KeyboardInterrupt:
        print("\n[停止] 传输终止")
    except Exception as e:
        print(f"\n[错误] 错误: {e}")
    finally:
        # 关闭鼠标控制
        mouse_controller.is_running = False
        
        # 关闭资源
        sct.close()
        video_sock.close()
        
        print("[停止] 所有服务已停止")

if __name__ == "__main__":
    main()
