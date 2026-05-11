import socket
import time
import numpy as np
import cv2
import mss
import math

# ===== 参数设置 =====
BOARD_IP = "172.20.10.8"
PORT = 5000
TARGET_FPS = 30
MIN_FPS = 20
SCALE_FACTOR = 1.0
MAX_PACKET_SIZE = 1400
JPEG_QUALITY = 75
DIFF_THRESHOLD = 0.02  # 灵敏的变化检测
KEYFRAME_INTERVAL = 15
# ===================

class VideoEncoder:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.jpeg_quality = JPEG_QUALITY
        self.prev_frame = None  # 仅用于变化检测
        self.frame_counter = 0
        self.encode = self._encode_software
        # 初始化时只输出一次编码信息
        print(f"[编码] 使用软件编码 | 固定JPEG质量: {self.jpeg_quality}")

    def _frame_change_detection(self, frame):
        """仅保留变化检测，用于减少冗余传输"""
        if self.prev_frame is None:
            return True  # 第一帧强制发送

        # 基于边缘和灰度的变化检测
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        # 边缘差异增强变化感知
        edge_current = cv2.Canny(gray_current, 50, 150)
        edge_prev = cv2.Canny(gray_prev, 50, 150)
        edge_diff = cv2.absdiff(edge_current, edge_prev)

        # 综合灰度和边缘差异
        gray_diff = cv2.absdiff(gray_current, gray_prev)
        combined_diff = cv2.bitwise_or(gray_diff, edge_diff)
        change_ratio = np.count_nonzero(combined_diff) / (gray_current.size)

        is_keyframe = (self.frame_counter % KEYFRAME_INTERVAL == 0)
        return is_keyframe or (change_ratio > DIFF_THRESHOLD)

    def _encode_software(self, frame):
        """直接编码原始帧，无任何平滑处理"""
        # 1. 变化检测（保留，减少冗余）
        if not self._frame_change_detection(frame):
            return [b'\x00']  # 空帧标记

        # 2. 直接编码原始帧
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1  # 优化编码速度
        ]
        _, img_encoded = cv2.imencode(".jpg", frame, encode_params)
        encoded_data = img_encoded.tobytes()
        self.prev_frame = frame.copy()  # 更新上一帧（仅用于变化检测）
        self.frame_counter += 1

        # 3. 分片处理
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
    """获取屏幕尺寸信息，用于终端输出"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主显示器
        original_width, original_height = monitor["width"], monitor["height"]
        scaled_width = int(original_width * SCALE_FACTOR)
        scaled_height = int(original_height * SCALE_FACTOR)
        # 输出分辨率信息
        print(f"[屏幕] 原始分辨率: {original_width}x{original_height}")
        print(f"[分辨率] 使用分辨率: {scaled_width}x{scaled_height} (缩放比例: {SCALE_FACTOR})")
        print(f"[网络] 单包最大数据量: {MAX_PACKET_SIZE}字节（不含4字节头部）")
        return scaled_width, scaled_height

def main():
    # 获取屏幕尺寸并输出信息
    WIDTH, HEIGHT = get_screen_size()

    # 初始化编码器和网络
    encoder = VideoEncoder(WIDTH, HEIGHT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)

    # 屏幕捕获设置
    monitor = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}
    sct = mss.mss()

    # 性能统计
    stats = {
        "frame_count": 0,
        "start_time": time.time(),
        "total_data": 0,
        "loss_count": 0,
        "last_delay": 0  # 记录延迟信息
    }

    # 帧率控制
    frame_interval = 1.0 / TARGET_FPS
    next_frame_time = time.time()
    drift_compensation = 0.0

    # 开始传输提示
    print(f"[启动] 开始传输到 {BOARD_IP}:{PORT}...（按Ctrl+C终止）")

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

            # 计算单帧延迟（毫秒）
            frame_delay = (time.time() - current_time) * 1000
            stats["last_delay"] = frame_delay

            next_frame_time += frame_interval

            # 捕获屏幕
            screen = sct.grab(monitor)
            frame = np.frombuffer(screen.bgra, dtype=np.uint8).reshape(
                screen.height, screen.width, 4)[:, :, :3]

            # 调整大小（如需）
            if (frame.shape[1], frame.shape[0]) != (WIDTH, HEIGHT):
                frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

            # 编码发送
            packets = encoder.encode(frame)
            if not packets:
                continue

            # 发送数据包
            for i, packet in enumerate(packets):
                try:
                    sock.sendto(packet, (BOARD_IP, PORT))
                    stats["total_data"] += len(packet)
                    if len(packet) > 1000 and i % 2 == 0:
                        time.sleep(0.0005)
                except Exception as e:
                    stats["loss_count"] += 1
                    if packet[0] == 1:
                        print(f"\n[警告] 关键帧丢包（累计{stats['loss_count']}次）")

            # 性能统计与终端输出
            stats["frame_count"] += 1
            elapsed = time.time() - stats["start_time"]
            if elapsed >= 2.0:  # 每2秒更新一次统计
                fps = stats["frame_count"] / elapsed
                mbps = (stats["total_data"] * 8 / (1024 * 1024)) / elapsed
                loss_rate = (stats["loss_count"] / (stats["frame_count"] * len(packets))) * 100 if stats["frame_count"] > 0 else 0

                # 格式化输出，与示例样式保持一致
                print(
                    f"[状态] FPS: {fps:.1f} | 延迟: {stats['last_delay']:.1f}ms | "
                    f"带宽: {mbps:.2f} Mbps | 丢包率: {loss_rate:.1f}% | 模式: JPEG",
                    end='\r'
                )

                # 重置统计
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
        sct.close()
        sock.close()

if __name__ == "__main__":
    main()
