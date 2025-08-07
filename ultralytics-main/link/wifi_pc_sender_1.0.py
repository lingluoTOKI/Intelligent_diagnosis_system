# WiFi发送端
import socket
import time
import numpy as np
import cv2
import mss
import math

# ===== 可调参数（优化后）=====
BOARD_IP = "172.20.10.8"  # 开发板IP
PORT = 5000               # 端口（与接收端一致）
FPS = 30                  # 降低帧率至30（平衡流畅度与稳定性）
USE_HARDWARE_ENCODER = False  # 禁用硬件编码，优先保证兼容性
SCALE_FACTOR = 0.9        # 分辨率缩放因子
MAX_PACKET_SIZE = 1400    # 减小单包大小，降低丢包风险（不含头部）
JPEG_QUALITY = 60         # 固定JPEG质量，避免波动
# ===================

class VideoEncoder:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.bitrate = int(width * height * FPS * 0.07)  # 动态计算码率

        if USE_HARDWARE_ENCODER:
            self._init_hardware_encoder()
        else:
            self._init_software_encoder()

    def _init_hardware_encoder(self):
        """初始化NVIDIA硬件编码器（支持分片）"""
        try:
            from PyNvCodec import NvEncoder, CudaContext
            self.cuda_ctx = CudaContext()
            self.encoder = NvEncoder(
                self.cuda_ctx,
                self.width, self.height,
                codec="h264",
                preset="p7",
                tune="ull",
                bitrate=self.bitrate,
                rc_mode="cbr",  # 恒定码率，减少质量波动
                gop_size=15
            )
            self.encode = self._encode_hardware
            print(f"✅ 硬件编码器已启用 | 码率: {self.bitrate//1000}kbps")
        except Exception as e:
            print(f"⚠️ 硬件编码初始化失败: {e}，回退到软件编码")
            self._init_software_encoder()

    def _init_software_encoder(self):
        """初始化OpenCV软件编码器（固定质量，减少波动）"""
        self.encode = self._encode_software
        self.jpeg_quality = JPEG_QUALITY  # 使用固定质量
        print(f"🖥️ 使用软件编码 | 固定JPEG质量: {self.jpeg_quality}")

    def _encode_hardware(self, frame):
        """硬件编码并分片（优化分片策略）"""
        cuda_frame = cv2.cuda_GpuMat()
        cuda_frame.upload(frame)
        encoded_data = self.encoder.encode(cuda_frame)

        # 分片处理
        total_packets = math.ceil(len(encoded_data) / MAX_PACKET_SIZE)
        packets = []
        for i in range(total_packets):
            start = i * MAX_PACKET_SIZE
            end = start + MAX_PACKET_SIZE
            payload = encoded_data[start:end]

            # 3字节头部：1字节序号 + 2字节总分片数（大端模式）
            total_packets_bytes = total_packets.to_bytes(2, byteorder='big')
            header = bytes([i]) + total_packets_bytes
            packets.append(header + payload)
        return packets

    def _encode_software(self, frame):
        """软件编码（JPEG）并分片（固定质量）"""
        # 编码为JPEG（固定质量）
        _, img_encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        encoded_data = img_encoded.tobytes()

        # 分片处理
        total_packets = math.ceil(len(encoded_data) / MAX_PACKET_SIZE)
        packets = []
        for i in range(total_packets):
            start = i * MAX_PACKET_SIZE
            end = start + MAX_PACKET_SIZE
            payload = encoded_data[start:end]

            # 3字节头部（与接收端匹配）
            total_packets_bytes = total_packets.to_bytes(2, byteorder='big')
            header = bytes([i]) + total_packets_bytes
            packets.append(header + payload)
        return packets

def get_screen_size():
    """获取缩放后的屏幕尺寸"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主显示器
        return {
            "width": int(monitor["width"] * SCALE_FACTOR),
            "height": int(monitor["height"] * SCALE_FACTOR),
            "original": (monitor["width"], monitor["height"])
        }

def main():
    # 获取屏幕信息
    screen_info = get_screen_size()
    WIDTH, HEIGHT = screen_info["width"], screen_info["height"]
    original_res = screen_info["original"]

    print(f"🖥️ 屏幕原始分辨率: {original_res[0]}x{original_res[1]}")
    print(f"📏 使用分辨率: {WIDTH}x{HEIGHT} (缩放比例: {SCALE_FACTOR})")
    print(f"📦 单包最大数据量: {MAX_PACKET_SIZE}字节（不含3字节头部）")

    # 初始化
    encoder = VideoEncoder(WIDTH, HEIGHT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)  # 4MB缓冲区

    # 屏幕捕获区域
    monitor = {"top": 0, "left": 0, "width": original_res[0], "height": original_res[1]}
    sct = mss.mss()

    # 性能统计
    stats = {
        "frame_count": 0,
        "start_time": time.time(),
        "total_data": 0,
        "loss_count": 0  # 新增：记录发送失败次数
    }

    print(f"🚀 开始传输到 {BOARD_IP}:{PORT}...（按Ctrl+C终止）")

    try:
        while True:
            frame_start = time.time()

            # 1. 捕获屏幕
            screen = sct.grab(monitor)
            frame = np.array(screen)[:, :, :3]  # 移除alpha通道

            # 2. 调整大小（使用平滑缩放算法）
            if (frame.shape[1], frame.shape[0]) != (WIDTH, HEIGHT):
                # INTER_AREA算法：缩放后图像更平滑，减少失真
                frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

            # 3. 编码帧（自动分片）
            packets = encoder.encode(frame)
            if not packets:
                continue  # 空帧跳过

            # 4. 发送所有分片（增加间隔，避免网络拥塞）
            for i, packet in enumerate(packets):
                try:
                    sock.sendto(packet, (BOARD_IP, PORT))
                    stats["total_data"] += len(packet)
                    # 每发送5个分片，增加1ms间隔，降低网络压力
                    if i % 5 == 0 and i != 0:
                        time.sleep(0.001)
                except Exception as e:
                    stats["loss_count"] += 1
                    print(f"\n⚠️ 发送失败（累计{stats['loss_count']}次）: {e}")
                    continue

            # 5. 性能统计（每2秒更新）
            stats["frame_count"] += 1
            elapsed = time.time() - stats["start_time"]
            if elapsed >= 2.0:
                fps = stats["frame_count"] / elapsed
                mbps = (stats["total_data"] * 8 / (1024 * 1024)) / elapsed
                loss_rate = (stats["loss_count"] / (stats["frame_count"] * len(packets))) * 100 if stats["frame_count"] > 0 else 0
                print(
                    f"📊 FPS: {fps:.1f} | 延迟: {(1/fps)*1000:.1f}ms | "
                    f"带宽: {mbps:.2f} Mbps | 丢包率: {loss_rate:.1f}% | "
                    f"模式: {'NVENC' if USE_HARDWARE_ENCODER else 'JPEG'}",
                    end='\r'
                )
                # 重置统计
                stats = {
                    "frame_count": 0,
                    "start_time": time.time(),
                    "total_data": 0,
                    "loss_count": stats["loss_count"]  # 保留累计丢包数
                }

            # 6. 帧率控制（精确控制发送间隔）
            frame_time = time.time() - frame_start
            sleep_time = (1/FPS) - frame_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 传输终止")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        sct.close()  # 关闭屏幕捕获
        sock.close()

if __name__ == "__main__":
    main()