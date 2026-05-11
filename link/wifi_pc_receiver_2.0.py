import socket
import cv2
import numpy as np
import time

# ===== 核心配置（需修改为实际环境）=====
PC_IP = "172.20.10.3"  # PC的IP地址（需与发送端一致）
VIDEO_PORT = 5000       # 接收视频的端口（与PC端一致）
CONTROL_PORT = 5001     # 发送控制指令的端口
MAX_PACKET_SIZE = 1400  # 与PC端保持一致
PC_SCREEN_WIDTH = 1920  # PC原始宽度（需与发送端实际屏幕一致）
PC_SCREEN_HEIGHT = 1080 # PC原始高度
# =====================================

# 初始化Socket
video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_sock.bind(("0.0.0.0", VIDEO_PORT))
video_sock.settimeout(0.03)  # 短超时，提升响应速度
video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)

control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
control_sock.setblocking(False)  # 非阻塞发送（避免指令发送阻塞）

# 帧缓存（处理视频帧的存储与显示）
class FrameBuffer:
    def __init__(self):
        self.current_frame = None  # 当前帧（numpy数组）
        self.last_valid_frame = None  # 上一有效帧（用于超时显示）
        self.window_initialized = False
        self.last_display_time = 0  # 最后一次显示时间
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps  # 显示间隔
        self.last_frame_seq = -1  # 最后处理的帧序号（防重复）

    def update(self, frame, frame_seq):
        """仅更新序号更大的帧（保证时序正确）"""
        if frame_seq > self.last_frame_seq:
            self.last_valid_frame = self.current_frame
            self.current_frame = frame
            self.last_frame_seq = frame_seq

    def should_display(self):
        """控制显示帧率（避免频繁刷新）"""
        return time.time() - self.last_display_time >= self.frame_interval

    def get_frame(self):
        """获取当前可显示的帧"""
        return self.current_frame if self.current_frame is not None else self.last_valid_frame

frame_buffer = FrameBuffer()

# 触摸事件相关全局变量（用于预测与节流）
last_touch_pos = None  # 上一次触摸位置 (x,y)
last_touch_time = 0    # 上一次触摸时间
last_send_time = 0     # 上一次发送指令时间
SEND_THROTTLE = 0.005  # 指令发送节流（5ms内不重复发送移动指令）

def mouse_callback(event, x, y, flags, param):
    """优化的触摸事件处理：预测位置+节流发送"""
    if frame_buffer.current_frame is None:
        return  # 未收到PC画面时不处理

    global last_touch_pos, last_touch_time, last_send_time
    current_time = time.time()
    h, w = frame_buffer.current_frame.shape[:2]  # 触摸屏显示的帧尺寸

    # 1. 坐标映射：触摸屏坐标 -> PC原始坐标
    pc_x = int(x * PC_SCREEN_WIDTH / w)
    pc_y = int(y * PC_SCREEN_HEIGHT / h)

    # 2. 触摸预测（补偿显示延迟）
    # 计算显示延迟（当前时间 - 最后一帧显示时间）
    display_delay = current_time - frame_buffer.last_display_time
    # 根据延迟和移动趋势预测位置（限制预测幅度）
    if event == cv2.EVENT_MOUSEMOVE and last_touch_pos is not None:
        dx = pc_x - last_touch_pos[0]  # X方向移动量
        dy = pc_y - last_touch_pos[1]  # Y方向移动量
        # 预测因子：延迟越长，预测越多（最多补偿50%移动量）
        predict_factor = min(display_delay * 3, 0.5)
        pc_x += int(dx * predict_factor)
        pc_y += int(dy * predict_factor)

    # 3. 指令节流（移动事件5ms内只发一次，减少冗余指令）
    if event == cv2.EVENT_MOUSEMOVE:
        if current_time - last_send_time < SEND_THROTTLE:
            return  # 节流：短时间内不重复发送

    # 4. 构造指令并发送
    try:
        if event == cv2.EVENT_MOUSEMOVE:
            cmd = b'\x00' + pc_x.to_bytes(2, 'big') + pc_y.to_bytes(2, 'big')
        elif event == cv2.EVENT_LBUTTONDOWN:
            cmd = b'\x01' + pc_x.to_bytes(2, 'big') + pc_y.to_bytes(2, 'big')
        elif event == cv2.EVENT_LBUTTONUP:
            cmd = b'\x02' + pc_x.to_bytes(2, 'big') + pc_y.to_bytes(2, 'big')
        else:
            return  # 忽略右键等其他事件

        # 非阻塞发送（不阻塞主线程）
        control_sock.sendto(cmd, (PC_IP, CONTROL_PORT))
        last_send_time = current_time  # 更新发送时间
        last_touch_pos = (pc_x, pc_y)  # 更新触摸位置
        last_touch_time = current_time

    except BlockingIOError:
        pass  # 发送缓冲区满时忽略（避免阻塞）
    except Exception as e:
        print(f"❌ 控制指令发送失败: {e}")

# 初始化显示窗口
def init_window():
    cv2.namedWindow("PC Stream", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("PC Stream", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("PC Stream", mouse_callback)  # 绑定触摸事件
    frame_buffer.window_initialized = True
    print("✅ 触摸屏控制已启用，可操作PC鼠标")

# 解码并显示视频帧
def decode_and_display(data, frame_seq):
    try:
        # 解码JPEG数据
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            frame_buffer.update(frame, frame_seq)
            if not frame_buffer.window_initialized:
                init_window()
        
        # 按目标帧率显示
        if frame_buffer.should_display():
            display_frame = frame_buffer.get_frame()
            if display_frame is not None:
                cv2.imshow("PC Stream", display_frame)
                frame_buffer.last_display_time = time.time()
                # 检测ESC退出
                if cv2.waitKey(1) & 0xFF == 27:
                    return False
    except Exception as e:
        print(f"❌ 解码错误: {e}")
    return True

# 主循环
try:
    print(f"等待PC视频流（端口{VIDEO_PORT}）...（按ESC退出）")
    frame_buffers = {}  # 缓存分片数据包 {帧序号: {分片: 数据, ...}}
    last_clean_time = time.time()
    running = True

    while running:
        try:
            # 接收视频数据包
            data, addr = video_sock.recvfrom(MAX_PACKET_SIZE + 5)
            if len(data) < 5:
                continue  # 数据包不完整

            # 解析包头：1B关键帧标记 + 2B帧序号 + 1B当前分片 + 1B总分片
            keyframe_flag = data[0]
            frame_seq = int.from_bytes(data[1:3], byteorder='big')
            packet_idx = data[3]
            total_packets = data[4]
            payload = data[5:]

            # 缓存分片
            if frame_seq not in frame_buffers:
                frame_buffers[frame_seq] = {
                    'fragments': {}, 'timestamp': time.time(),
                    'total_packets': total_packets
                }
            if 0 <= packet_idx < total_packets:
                frame_buffers[frame_seq]['fragments'][packet_idx] = payload

            # 所有分片接收完成后拼接
            if len(frame_buffers[frame_seq]['fragments']) == total_packets:
                # 按分片序号排序并拼接
                sorted_fragments = [frame_buffers[frame_seq]['fragments'][i] 
                                   for i in sorted(frame_buffers[frame_seq]['fragments'])]
                frame_data = b''.join(sorted_fragments)
                running = decode_and_display(frame_data, frame_seq)
                del frame_buffers[frame_seq]  # 处理后删除缓存

            # 定期清理过期帧（防止内存泄漏）
            if time.time() - last_clean_time > 1.0:
                current_time = time.time()
                # 删除超过1秒未接收完整的帧
                to_delete = [seq for seq in frame_buffers 
                           if current_time - frame_buffers[seq]['timestamp'] > 1.0]
                for seq in to_delete:
                    del frame_buffers[seq]
                last_clean_time = current_time

        except socket.timeout:
            # 超时处理：刷新最后一帧（避免画面冻结）
            if frame_buffer.should_display() and frame_buffer.get_frame() is not None:
                cv2.imshow("PC Stream", frame_buffer.get_frame())
                frame_buffer.last_display_time = time.time()
                if cv2.waitKey(1) & 0xFF == 27:
                    running = False
            continue
        except Exception as e:
            print(f"❌ 接收错误: {e}")
            continue

except KeyboardInterrupt:
    print("\n🛑 接收端已关闭")
finally:
    video_sock.close()
    control_sock.close()
    cv2.destroyAllWindows()