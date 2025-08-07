#WiFi接收端
import socket
import cv2
import numpy as np
import time

PORT = 5000

# 创建UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.settimeout(0.5)  # 缩短超时时间
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)  # 增大缓冲区


# 优化缓存结构
class FrameBuffer:
    def __init__(self):
        self.current_frame = None
        self.last_valid_frame = None

    def update(self, frame):
        self.last_valid_frame = self.current_frame
        self.current_frame = frame

    def get_frame(self):
        return self.current_frame if self.current_frame is not None else self.last_valid_frame


frame_buffer = FrameBuffer()

# 显示窗口设置
cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)


def decode_and_display(data):
    try:
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            frame_buffer.update(frame)
        display_frame = frame_buffer.get_frame()
        if display_frame is not None:
            cv2.imshow("Stream", display_frame)
            cv2.waitKey(1)
    except:
        pass


try:
    print(f"等待视频流（端口{PORT}）...（按ESC退出）")
    buffer = {}
    last_clean_time = time.time()

    while True:
        try:
            # 接收数据
            data, addr = sock.recvfrom(2048)

            # 解析头部
            if len(data) < 3:
                continue
            packet_idx = data[0]
            total_packets = int.from_bytes(data[1:3], byteorder='big')
            payload = data[3:]

            # 初始化分片组
            if total_packets not in buffer:
                buffer[total_packets] = {
                    'fragments': {},
                    'timestamp': time.time()
                }

            # 存储分片
            if 0 <= packet_idx < total_packets:
                buffer[total_packets]['fragments'][packet_idx] = payload

            # 检查完整帧
            if len(buffer[total_packets]['fragments']) == total_packets:
                # 按序号排序并拼接
                sorted_packets = [buffer[total_packets]['fragments'][i] for i in
                                  sorted(buffer[total_packets]['fragments'])]
                frame_data = b''.join(sorted_packets)
                decode_and_display(frame_data)
                del buffer[total_packets]

            # 定期清理（每秒一次）
            if time.time() - last_clean_time > 1.0:
                current_time = time.time()
                to_delete = []
                for key in buffer:
                    if current_time - buffer[key]['timestamp'] > 1.0:  # 1秒超时
                        to_delete.append(key)
                for key in to_delete:
                    del buffer[key]
                last_clean_time = time.time()

        except socket.timeout:
            # 显示最后有效帧
            display_frame = frame_buffer.get_frame()
            if display_frame is not None:
                cv2.imshow("Stream", display_frame)
                cv2.waitKey(1)
            continue

except KeyboardInterrupt:
    print("接收端关闭")
finally:
    sock.close()
    cv2.destroyAllWindows()