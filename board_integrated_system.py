#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发板集成医疗诊断系统 - JupyterLab版本
功能集成：
1. 摄像头图像捕获与实时预览
2. 图像传输到PC端进行AI诊断
3. 语音交互（录音、语音识别、TTS播放）
4. 触摸屏控制界面
5. 与PC端医疗诊断系统完整交互
6. 支持JupyterLab单文件运行
"""

import socket
import cv2
import numpy as np
import time
import json
import os
import threading
import queue
import base64
import io
import wave
from datetime import datetime

# 检查可选依赖
try:
    import pyaudio
    HAS_AUDIO = True
    print("[INFO] 音频功能可用")
except ImportError:
    HAS_AUDIO = False
    print("[WARN] PyAudio未安装，音频功能禁用")

try:
    import pygame
    HAS_PYGAME = True
    print("[INFO] 图形界面功能可用")
except ImportError:
    HAS_PYGAME = False
    print("[WARN] Pygame未安装，图形界面功能禁用")

# ===== 导入统一配置 =====
try:
    from system_config import (
        PC_IP, CAMERA_PORT, DIAGNOSIS_PORT, COMMAND_PORT,
        VOICE_SEND_PORT, VOICE_RECEIVE_PORT, VOICE_COMMAND_PORT,
        TOUCH_CONTROL_PORT, CAMERA_CONFIG, AUDIO_CONFIG,
        SYSTEM_CONFIG, connection_manager
    )
    
    # 从配置中提取具体值
    CAMERA_WIDTH = CAMERA_CONFIG['WIDTH']
    CAMERA_HEIGHT = CAMERA_CONFIG['HEIGHT']
    CAMERA_FPS = CAMERA_CONFIG['FPS']
    JPEG_QUALITY = CAMERA_CONFIG['JPEG_QUALITY']
    MAX_PACKET_SIZE = CAMERA_CONFIG['MAX_PACKET_SIZE']
    
    # 音频配置
    if HAS_AUDIO:
        AUDIO_FORMAT = pyaudio.paInt16
        CHANNELS = AUDIO_CONFIG['CHANNELS']
        RATE = AUDIO_CONFIG['RATE']
        CHUNK = AUDIO_CONFIG['CHUNK']
    
    # 目录配置
    SAVE_DIR = SYSTEM_CONFIG['SAVE_DIR']
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("[INFO] 使用统一配置文件")
    
except ImportError:
    # 如果配置文件不存在，使用默认配置
    print("[WARN] 未找到统一配置文件，使用默认配置")
    
    # ===== 默认配置 =====
    PC_IP = "172.20.10.3"  # PC端IP地址（固定配置）
    print(f"[配置] PC IP设置为: {PC_IP}")

    # 网络端口配置
    CAMERA_PORT = 5002          # 摄像头数据传输
    DIAGNOSIS_PORT = 5003       # 诊断结果接收
    COMMAND_PORT = 5004         # 命令控制
    VOICE_SEND_PORT = 5005      # 语音数据发送
    VOICE_RECEIVE_PORT = 5006   # 语音合成接收
    VOICE_COMMAND_PORT = 5007   # 语音命令控制
    TOUCH_CONTROL_PORT = 5001   # 触摸屏控制

    # 摄像头配置
    CAMERA_WIDTH = 512
    CAMERA_HEIGHT = 512
    CAMERA_FPS = 30
    JPEG_QUALITY = 85
    MAX_PACKET_SIZE = 1400

    # 音频配置
    if HAS_AUDIO:
        AUDIO_FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024

    # 目录配置
    SAVE_DIR = "medical_images"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    connection_manager = None

# ===== 摄像头管理器 =====
class CameraThread(threading.Thread):
    """摄像头线程管理器"""
    
    def __init__(self, callback=None):
        super(CameraThread, self).__init__()
        self.working = True
        self.running = False
        self.frame_callback = callback
        self.last_frame = None
        self.cap = None
        self.init_camera()

    def init_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("[错误] 无法打开摄像头")
                return False
            else:
                print('[摄像头] 摄像头已打开')
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print(f"[摄像头] 初始化成功 - 分辨率: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
            return True
            
        except Exception as e:
            print(f"[错误] 摄像头初始化失败: {e}")
            return False

    def run(self):
        """摄像头线程主循环"""
        self.running = True
        print("[摄像头] 视频流线程启动")
        
        while self.working:
            try:
                if not self.cap or not self.cap.isOpened():
                    print("[警告] 摄像头未打开，尝试重新初始化...")
                    if not self.init_camera():
                        time.sleep(1)
                        continue
                
                ret, frame = self.cap.read()
                if not ret:
                    print("[警告] 图像获取失败")
                    time.sleep(0.1)
                    continue
                
                self.last_frame = frame
                
                if self.frame_callback:
                    self.frame_callback(frame)
                
                time.sleep(1.0 / CAMERA_FPS)
                
            except Exception as e:
                print(f"[错误] 摄像头线程错误: {e}")
                break
        
        self.running = False
        print("[摄像头] 视频流线程停止")

    def stop(self):
        """停止摄像头"""
        print("[摄像头] 正在停止视频流...")
        self.working = False
        
        while self.running:
            time.sleep(0.1)
        
        if self.cap:
            self.cap.release()
            print("[摄像头] VideoCapture已释放")

    def capture_frame(self):
        """获取当前帧"""
        return self.last_frame

    def is_camera_opened(self):
        """检查摄像头状态"""
        return self.cap is not None and self.cap.isOpened()

class CameraManager:
    """摄像头管理器"""
    
    def __init__(self):
        self.camera_thread = None
        self.is_previewing = False
        self.is_streaming = False
        self.stream_callback = None
        self.preview_window_name = "Medical Camera Preview"
        
    def initialize(self, stream_callback=None):
        """初始化摄像头"""
        try:
            self.stream_callback = stream_callback
            self.camera_thread = CameraThread(callback=self._frame_handler)
            self.camera_thread.start()
            time.sleep(1)  # 等待初始化完成
            
            if self.camera_thread.is_camera_opened():
                print("[成功] 摄像头管理器初始化完成")
                return True
            else:
                print("[错误] 摄像头初始化失败")
                return False
                
        except Exception as e:
            print(f"[错误] 摄像头管理器初始化失败: {e}")
            return False
    
    def _frame_handler(self, frame):
        """处理每一帧图像"""
        if self.is_streaming and self.stream_callback:
            self.stream_callback(frame)
    
    def start_streaming(self):
        """开始视频流传输"""
        if not self.camera_thread or not self.camera_thread.is_camera_opened():
            print("[错误] 摄像头未初始化，无法开始流传输")
            return False
        
        self.is_streaming = True
        print("[流传输] 摄像头流传输已启动")
        return True
    
    def stop_streaming(self):
        """停止视频流传输"""
        self.is_streaming = False
        print("[流传输] 摄像头流传输已停止")
    
    def start_preview(self):
        """启动预览"""
        if not self.camera_thread or not self.camera_thread.is_camera_opened():
            print("[错误] 摄像头未初始化")
            return False
        
        if self.is_previewing:
            print("[警告] 预览已在运行")
            return True
        
        try:
            self.is_previewing = True
            threading.Thread(target=self._preview_worker, daemon=True).start()
            print("[成功] 摄像头预览已启动")
            return True
            
        except Exception as e:
            print(f"[错误] 预览启动失败: {e}")
            self.is_previewing = False
            return False
    
    def _preview_worker(self):
        """预览工作线程"""
        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.preview_window_name, 640, 480)
        
        print("\n[预览] 摄像头预览操作:")
        print("  空格键 - 拍照并发送诊断")
        print("  's' - 仅保存照片")
        print("  'r' - 开始录音对话")
        print("  'q' - 退出预览")
        
        while self.is_previewing:
            try:
                frame = self.camera_thread.capture_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # 添加操作提示到画面
                frame_with_text = frame.copy()
                cv2.putText(frame_with_text, "Space: Capture & Diagnose", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_with_text, "S: Save Photo", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_with_text, "R: Voice Chat", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_with_text, "Q: Quit", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow(self.preview_window_name, frame_with_text)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # 空格键 - 拍照诊断
                    self.capture_and_diagnose(frame)
                elif key == ord('s'):  # S键 - 保存照片
                    self.save_photo(frame)
                elif key == ord('r'):  # R键 - 语音对话
                    self.start_voice_chat()
                elif key == ord('q'):  # Q键 - 退出
                    self.is_previewing = False
                    break
                    
            except Exception as e:
                print(f"[错误] 预览错误: {e}")
                break
        
        cv2.destroyWindow(self.preview_window_name)
    
    def capture_and_diagnose(self, frame):
        """拍照并发送诊断"""
        try:
            print("[拍照] 正在拍摄并发送诊断...")
            
            # 图像增强
            enhanced_frame = self.enhance_image(frame)
            
            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnosis_capture_{timestamp}.jpg"
            filepath = self.save_image(enhanced_frame, filename)
            
            if filepath:
                self.show_capture_feedback()
                
                # 发送诊断请求（这里会被网络管理器处理）
                print("[发送] 图像已准备发送诊断")
                return enhanced_frame, filepath
            
        except Exception as e:
            print(f"[错误] 拍照诊断失败: {e}")
            return None, None
    
    def save_photo(self, frame):
        """保存照片"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
            filepath = self.save_image(frame, filename)
            
            if filepath:
                print(f"[保存] 照片已保存: {filename}")
                self.show_capture_feedback()
            
        except Exception as e:
            print(f"[错误] 照片保存失败: {e}")
    
    def start_voice_chat(self):
        """启动语音对话"""
        if HAS_AUDIO:
            print("[语音] 启动语音对话功能...")
            # 这里会被语音管理器处理
        else:
            print("[警告] 音频功能不可用")
    
    def enhance_image(self, frame):
        """图像增强处理"""
        try:
            # 1. 去噪
            denoised = cv2.bilateralFilter(frame, 9, 75, 75)
            
            # 2. 锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # 3. 色彩增强
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            print(f"[警告] 图像增强失败，使用原图: {e}")
            return frame
    
    def save_image(self, frame, filename):
        """保存图像"""
        try:
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[图像] 图像已保存: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"[错误] 图像保存失败: {e}")
            return None
    
    def show_capture_feedback(self):
        """显示拍摄反馈"""
        try:
            feedback = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(feedback, "Photo Captured!", (80, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(feedback, "Processing...", (120, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Capture Feedback", feedback)
            cv2.waitKey(1000)
            cv2.destroyWindow("Capture Feedback")
            
        except Exception as e:
            print(f"[错误] 反馈显示失败: {e}")
    
    def get_camera_status(self):
        """获取摄像头状态"""
        if self.camera_thread:
            return {
                "initialized": True,
                "running": self.camera_thread.running,
                "camera_opened": self.camera_thread.is_camera_opened(),
                "preview_active": self.is_previewing,
                "resolution": f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
                "fps": CAMERA_FPS
            }
        else:
            return {"initialized": False}
    
    def release(self):
        """释放摄像头资源"""
        self.is_previewing = False
        if self.camera_thread:
            self.camera_thread.stop()
        cv2.destroyAllWindows()
        print("[摄像头] 摄像头管理器已释放")

# ===== 音频管理器 =====
class AudioManager:
    """音频管理器"""
    
    def __init__(self):
        self.audio = None
        self.is_recording = False
        self.is_playing = False
        self.playback_queue = queue.Queue()
        
        if HAS_AUDIO:
            self.audio = pyaudio.PyAudio()
            self.init_audio_devices()
        
    def init_audio_devices(self):
        """初始化音频设备"""
        if not HAS_AUDIO:
            return
        
        try:
            print("[音频] 可用音频设备:")
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                print(f"  设备 {i}: {info['name']} - 输入:{info['maxInputChannels']}, 输出:{info['maxOutputChannels']}")
            
            self.input_device = None
            self.output_device = None
            print("[成功] 音频设备初始化完成")
            
        except Exception as e:
            print(f"[错误] 音频设备初始化失败: {e}")
    
    def start_recording(self, duration=5):
        """开始录音"""
        if not HAS_AUDIO:
            print("[警告] 音频功能不可用")
            return None
            
        if self.is_recording:
            print("[警告] 正在录音中...")
            return None
            
        try:
            print(f"[录音] 开始录音 {duration} 秒...")
            self.is_recording = True
            
            stream = self.audio.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=CHUNK
            )
            
            frames = []
            for _ in range(0, int(RATE / CHUNK * duration)):
                if not self.is_recording:
                    break
                data = stream.read(CHUNK)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            self.is_recording = False
            
            # 生成WAV数据
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(AUDIO_FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            
            wav_data = wav_buffer.getvalue()
            print(f"[成功] 录音完成，数据大小: {len(wav_data)} 字节")
            return wav_data
            
        except Exception as e:
            print(f"[错误] 录音失败: {e}")
            self.is_recording = False
            return None
    
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        print("[停止] 录音已停止")
    
    def play_audio(self, audio_data):
        """播放音频"""
        if not HAS_AUDIO:
            print("[警告] 音频功能不可用")
            return
            
        try:
            self.playback_queue.put(audio_data)
            
            if not self.is_playing:
                threading.Thread(target=self._playback_worker, daemon=True).start()
                
        except Exception as e:
            print(f"[错误] 音频播放失败: {e}")
    
    def _playback_worker(self):
        """音频播放工作线程"""
        self.is_playing = True
        
        try:
            while not self.playback_queue.empty():
                audio_data = self.playback_queue.get()
                
                wav_buffer = io.BytesIO(audio_data)
                with wave.open(wav_buffer, 'rb') as wf:
                    stream = self.audio.open(
                        format=self.audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        output_device_index=self.output_device
                    )
                    
                    print("[播放] 开始播放语音...")
                    data = wf.readframes(CHUNK)
                    while data:
                        stream.write(data)
                        data = wf.readframes(CHUNK)
                    
                    stream.stop_stream()
                    stream.close()
                    
                print("[完成] 语音播放完成")
                
        except Exception as e:
            print(f"[错误] 播放失败: {e}")
        finally:
            self.is_playing = False
    
    def cleanup(self):
        """清理音频资源"""
        self.is_recording = False
        self.is_playing = False
        if self.audio:
            self.audio.terminate()

# ===== 网络管理器 =====
class NetworkManager:
    """网络通信管理器"""
    
    def __init__(self):
        self.sockets = {}
        self.is_running = True
        self.connection_status = False
        self.last_heartbeat = 0
        self.heartbeat_interval = 5.0  # 心跳间隔5秒
        self.packet_sequence = 0
        self.received_packets = {}  # 用于重组分片数据
        self.is_streaming = False
        self.stream_fps = 15  # 流传输帧率
        self.last_stream_time = 0
        self.init_sockets()
        
    def init_sockets(self):
        """初始化网络套接字"""
        try:
            # 摄像头数据发送
            self.sockets['camera'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sockets['camera'].setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
            
            # 诊断结果接收
            self.sockets['diagnosis'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sockets['diagnosis'].bind(("0.0.0.0", DIAGNOSIS_PORT))
            self.sockets['diagnosis'].settimeout(10.0)
            
            # 命令控制
            self.sockets['command'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # 语音相关套接字
            if HAS_AUDIO:
                self.sockets['voice_send'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets['voice_receive'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sockets['voice_receive'].bind(("0.0.0.0", VOICE_RECEIVE_PORT))
                self.sockets['voice_receive'].settimeout(1.0)
                self.sockets['voice_command'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # 触摸屏控制
            self.sockets['touch'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # 启动心跳检测
            threading.Thread(target=self._heartbeat_worker, daemon=True).start()
            
            print("[网络] 网络套接字初始化完成")
            return True
            
        except Exception as e:
            print(f"[错误] 网络初始化失败: {e}")
            return False
    
    def _heartbeat_worker(self):
        """心跳检测工作线程"""
        while self.is_running:
            try:
                if time.time() - self.last_heartbeat > self.heartbeat_interval:
                    self._send_heartbeat()
                    self.last_heartbeat = time.time()
                time.sleep(1)
            except Exception as e:
                print(f"[心跳] 心跳检测错误: {e}")
    
    def _send_heartbeat(self):
        """发送心跳包"""
        try:
            heartbeat = {
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
                "board_id": "medical_board_001",
                "status": "running"
            }
            heartbeat_data = json.dumps(heartbeat).encode('utf-8')
            self.sockets['command'].sendto(heartbeat_data, (PC_IP, COMMAND_PORT))
        except Exception as e:
            print(f"[心跳] 心跳发送失败: {e}")
    
    def test_connection(self):
        """测试网络连接"""
        print(f"[网络] 正在测试PC端连接 {PC_IP}:{COMMAND_PORT}...")
        
        # 使用统一的连接管理器（如果可用）
        if connection_manager:
            return connection_manager.test_connection(PC_IP, COMMAND_PORT)
        
        # 否则使用本地连接测试
        try:
            test_data = {
                "type": "connection_test",
                "timestamp": datetime.now().isoformat(),
                "board_id": "medical_board_001",
                "version": "2.0"
            }
            test_bytes = json.dumps(test_data).encode('utf-8')
            
            # 发送测试包
            self.sockets['command'].sendto(test_bytes, (PC_IP, COMMAND_PORT))
            print(f"[网络] 已发送连接测试包到 {PC_IP}:{COMMAND_PORT}")
            
            # 等待响应
            self.sockets['command'].settimeout(5.0)  # 增加超时时间
            try:
                response, addr = self.sockets['command'].recvfrom(1024)
                response_data = json.loads(response.decode('utf-8'))
                if response_data.get('type') == 'connection_test_response':
                    self.connection_status = True
                    latency = response_data.get('latency', 'N/A')
                    print(f"✅ [网络] 连接测试成功，延迟: {latency}ms")
                    return True
                else:
                    print(f"⚠️ [网络] 收到意外响应: {response_data}")
                    return False
            except socket.timeout:
                print("❌ [网络] 连接测试超时 - PC端可能未启动或网络不通")
                self.connection_status = False
                return False
                
        except Exception as e:
            print(f"❌ [网络] 连接测试失败: {e}")
            print("💡 请检查:")
            print("   1. PC端是否已启动")
            print("   2. 网络连接是否正常")
            print(f"   3. PC IP地址是否正确: {PC_IP}")
            self.connection_status = False
            return False
        finally:
            self.sockets['command'].settimeout(None)
    
    def send_stream_frame(self, frame):
        """发送视频流帧"""
        if not self.connection_status:
            return False
        
        # 控制帧率
        current_time = time.time()
        if current_time - self.last_stream_time < (1.0 / self.stream_fps):
            return True  # 跳过此帧
        
        self.last_stream_time = current_time
        
        try:
            # 压缩图像
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            img_data = img_encoded.tobytes()
            
            # 创建流传输包
            timestamp = int(time.time() * 1000)
            self.packet_sequence += 1
            stream_id = f"stream_{timestamp}_{self.packet_sequence}"
            
            # 分片发送
            total_packets = (len(img_data) + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
            
            for i in range(total_packets):
                start = i * MAX_PACKET_SIZE
                end = min(start + MAX_PACKET_SIZE, len(img_data))
                packet_data = img_data[start:end]
                
                # 流传输包头：[4字节流ID哈希][2字节包索引][2字节总包数][1字节标志位][图像数据]
                packet_id = hash(stream_id) & 0xFFFFFFFF
                packet_header = (
                    packet_id.to_bytes(4, 'big') +
                    i.to_bytes(2, 'big') +
                    total_packets.to_bytes(2, 'big') +
                    (1 if i == total_packets - 1 else 0).to_bytes(1, 'big')  # 移除0x80标记，使用标准格式
                )
                packet = packet_header + packet_data
                
                self.sockets['camera'].sendto(packet, (PC_IP, CAMERA_PORT))
                
                # 减少流传输的延迟
                if i < total_packets - 1:
                    time.sleep(0.0005)  # 0.5ms间隔
            
            return True
            
        except Exception as e:
            print(f"[错误] 视频流发送失败: {e}")
            return False
    
    def start_streaming(self):
        """开始视频流传输"""
        self.is_streaming = True
        print("[流传输] 网络视频流传输已启动")
    
    def stop_streaming(self):
        """停止视频流传输"""
        self.is_streaming = False
        print("[流传输] 网络视频流传输已停止")
    
    def save_image_to_pc(self, image, filename=None, save_to_pc=True):
        """直接保存图像到PC端指定目录"""
        try:
            if not save_to_pc:
                print("[保存] 跳过PC端保存")
                return None
            
            # 生成文件名
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"board_capture_{timestamp}.jpg"
            
            # 编码图像
            _, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            img_data = img_encoded.tobytes()
            
            # 创建保存请求
            timestamp = int(time.time() * 1000)
            self.packet_sequence += 1
            request_id = f"save_{timestamp}_{self.packet_sequence}"
            
            # 简化的保存请求头
            save_header = {
                "type": "image_save_request",
                "request_id": request_id,
                "timestamp": timestamp,
                "filename": filename,
                "image_size": len(img_data),
                "width": image.shape[1],
                "height": image.shape[0],
                "pc_save_path": r"C:\Users\47449\Desktop\yolo\Intelligent_diagnosis_system\ultralytics-main\datasets\test"
            }
            
            print(f"[保存] 发送图像保存请求，文件名: {filename}")
            print(f"[保存] 目标目录: {save_header['pc_save_path']}")
            
            # 发送保存请求头
            header_data = json.dumps(save_header).encode('utf-8')
            self.sockets['command'].sendto(header_data, (PC_IP, COMMAND_PORT))
            
            # 等待PC端处理请求头
            time.sleep(0.1)
            
            # 发送图像数据（简化版本，不分片）
            if len(img_data) <= MAX_PACKET_SIZE:
                # 小图像直接发送
                packet = img_data
                self.sockets['camera'].sendto(packet, (PC_IP, CAMERA_PORT))
                print(f"[保存] 图像数据已发送 ({len(img_data)} 字节)")
            else:
                # 大图像分片发送
                total_packets = (len(img_data) + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
                print(f"[保存] 图像较大，分片发送: {total_packets} 包")
                
                for i in range(total_packets):
                    start = i * MAX_PACKET_SIZE
                    end = min(start + MAX_PACKET_SIZE, len(img_data))
                    packet_data = img_data[start:end]
                    
                    # 简化的包头：包索引 + 总包数 + 数据
                    packet_header = (
                        i.to_bytes(2, 'big') +
                        total_packets.to_bytes(2, 'big') +
                        packet_data
                    )
                    
                    self.sockets['camera'].sendto(packet_header, (PC_IP, CAMERA_PORT))
                    
                    if i < total_packets - 1:
                        time.sleep(0.001)
            
            print(f"[保存] 图像保存请求完成，request_id: {request_id}")
            return request_id
            
        except Exception as e:
            print(f"[错误] 图像保存失败: {e}")
            return None
    
    def send_image_for_diagnosis(self, image, save_to_pc=True):
        """发送图像到PC端进行诊断（保留原有功能）"""
        try:
            # 编码图像
            _, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            img_data = img_encoded.tobytes()
            
            # 创建传输包
            timestamp = int(time.time() * 1000)
            self.packet_sequence += 1
            request_id = f"req_{timestamp}_{self.packet_sequence}"
            
            header = {
                "type": "diagnosis_request",
                "request_id": request_id,
                "timestamp": timestamp,
                "image_size": len(img_data),
                "width": image.shape[1],
                "height": image.shape[0],
                "total_packets": 0,  # 将在分片时计算
                "compression_quality": JPEG_QUALITY,
                "save_to_pc": save_to_pc,  # 是否保存到PC端
                "pc_save_path": r"C:\Users\47449\Desktop\yolo\Intelligent_diagnosis_system\ultralytics_main\datasets\test"  # PC端保存路径
            }
            
            # 🔥 关键修复：先发送诊断请求头到PC端
            print(f"[发送] 发送诊断请求头，request_id: {request_id}")
            header_data = json.dumps(header).encode('utf-8')
            self.sockets['command'].sendto(header_data, (PC_IP, COMMAND_PORT))
            
            # 等待PC端处理请求头
            time.sleep(0.1)  # 100ms延迟，确保PC端先处理请求头
            
            # 分片发送图像数据
            total_packets = (len(img_data) + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
            header["total_packets"] = total_packets
            
            print(f"[发送] 开始发送图像，大小: {len(img_data)} 字节，分片: {total_packets}")
            if save_to_pc:
                print(f"[保存] 图像将保存到PC端: {header['pc_save_path']}")
            
            for i in range(total_packets):
                start = i * MAX_PACKET_SIZE
                end = min(start + MAX_PACKET_SIZE, len(img_data))
                packet_data = img_data[start:end]
                
                # 改进的包头格式：[4字节请求ID哈希][2字节包索引][2字节总包数][1字节标志位][图像数据]
                packet_id = hash(request_id) & 0xFFFFFFFF
                packet_header = (
                    packet_id.to_bytes(4, 'big') +
                    i.to_bytes(2, 'big') +
                    total_packets.to_bytes(2, 'big') +
                    (1 if i == total_packets - 1 else 0).to_bytes(1, 'big')
                )
                packet = packet_header + packet_data
                
                # 在第一个包中包含request_id信息，便于PC端匹配
                if i == 0:
                    request_info = request_id.encode('utf-8')
                    packet = packet_header + len(request_info).to_bytes(2, 'big') + request_info + packet_data
                
                self.sockets['camera'].sendto(packet, (PC_IP, CAMERA_PORT))
                
                # 动态调整发送间隔
                if i < total_packets - 1:
                    time.sleep(0.001)  # 1ms间隔
            
            print(f"[发送] 图像发送完成，请求ID: {request_id}")
            return request_id
            
        except Exception as e:
            print(f"[错误] 图像发送失败: {e}")
            return None
    
    def wait_for_diagnosis_result(self, request_id=None, timeout=30):
        """等待诊断结果"""
        try:
            print("[等待] 等待PC端诊断结果...")
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    data, addr = self.sockets['diagnosis'].recvfrom(4096)
                    result = json.loads(data.decode('utf-8'))
                    
                    # 检查是否是匹配的请求结果
                    if request_id and result.get('request_id') != request_id:
                        continue
                    
                    if result.get('type') == 'diagnosis_result':
                        print("[成功] 收到诊断结果")
                        self.connection_status = True
                        return result
                    elif result.get('type') == 'diagnosis_error':
                        print(f"[错误] 诊断失败: {result.get('error', '未知错误')}")
                        return result
                        
                except socket.timeout:
                    continue
                except json.JSONDecodeError:
                    print("[警告] 诊断结果格式错误")
                    continue
            
            print("[超时] 诊断结果等待超时")
            return None
            
        except Exception as e:
            print(f"[错误] 接收诊断结果失败: {e}")
            return None
    
    def send_voice_data(self, audio_data, metadata=None):
        """发送语音数据到PC端"""
        if not HAS_AUDIO:
            return False
            
        try:
            # 生成语音包ID
            voice_id = f"voice_{int(time.time() * 1000)}"
            
            # 分片发送语音数据
            max_packet_size = 8192
            total_packets = (len(audio_data) + max_packet_size - 1) // max_packet_size
            
            # 发送语音头部信息
            header = {
                "type": "voice_data_start",
                "voice_id": voice_id,
                "timestamp": datetime.now().isoformat(),
                "format": "wav",
                "sample_rate": RATE,
                "channels": CHANNELS,
                "total_packets": total_packets,
                "total_size": len(audio_data),
                "metadata": metadata or {}
            }
            
            header_data = json.dumps(header).encode('utf-8')
            self.sockets['voice_send'].sendto(header_data, (PC_IP, VOICE_SEND_PORT))
            
            # 分片发送语音数据
            for i in range(total_packets):
                start = i * max_packet_size
                end = min(start + max_packet_size, len(audio_data))
                
                packet_header = (
                    voice_id.encode('utf-8')[:8].ljust(8, b'\x00') +  # 8字节语音ID
                    i.to_bytes(2, 'big') +                            # 2字节包索引
                    total_packets.to_bytes(2, 'big') +                # 2字节总包数
                    (1 if i == total_packets - 1 else 0).to_bytes(1, 'big')  # 1字节标志位
                )
                packet_chunk = packet_header + audio_data[start:end]
                
                self.sockets['voice_send'].sendto(packet_chunk, (PC_IP, VOICE_SEND_PORT))
                time.sleep(0.01)
            
            # 发送语音结束标记
            end_marker = {
                "type": "voice_data_end",
                "voice_id": voice_id,
                "timestamp": datetime.now().isoformat()
            }
            end_data = json.dumps(end_marker).encode('utf-8')
            self.sockets['voice_send'].sendto(end_data, (PC_IP, VOICE_SEND_PORT))
            
            print(f"[发送] 语音数据已发送到PC端 ({len(audio_data)} 字节, {total_packets} 包)")
            return voice_id
            
        except Exception as e:
            print(f"[错误] 语音发送失败: {e}")
            return None
    
    def send_voice_command(self, command, params=None):
        """发送语音控制命令"""
        if not HAS_AUDIO:
            return
            
        try:
            cmd_packet = {
                "type": "voice_command",
                "command": command,
                "params": params or {},
                "timestamp": datetime.now().isoformat()
            }
            
            cmd_data = json.dumps(cmd_packet).encode('utf-8')
            self.sockets['voice_command'].sendto(cmd_data, (PC_IP, VOICE_COMMAND_PORT))
            
            print(f"[命令] 已发送语音命令: {command}")
            
        except Exception as e:
            print(f"[错误] 语音命令发送失败: {e}")
    
    def send_touch_control(self, event_type, x, y):
        """发送触摸屏控制指令"""
        try:
            if event_type == "move":
                cmd = b'\x00' + x.to_bytes(2, 'big') + y.to_bytes(2, 'big')
            elif event_type == "press":
                cmd = b'\x01' + x.to_bytes(2, 'big') + y.to_bytes(2, 'big')
            elif event_type == "release":
                cmd = b'\x02' + x.to_bytes(2, 'big') + y.to_bytes(2, 'big')
            else:
                return
            
            self.sockets['touch'].sendto(cmd, (PC_IP, TOUCH_CONTROL_PORT))
            
        except Exception as e:
            print(f"[错误] 触摸控制发送失败: {e}")
    
    def get_connection_status(self):
        """获取连接状态"""
        return {
            "connected": self.connection_status,
            "last_heartbeat": self.last_heartbeat,
            "pc_ip": PC_IP,
            "ports": {
                "camera": CAMERA_PORT,
                "diagnosis": DIAGNOSIS_PORT,
                "command": COMMAND_PORT,
                "voice_send": VOICE_SEND_PORT,
                "voice_receive": VOICE_RECEIVE_PORT
            }
        }
    
    def close(self):
        """关闭网络连接"""
        self.is_running = False
        for name, sock in self.sockets.items():
            if sock:
                try:
                    sock.close()
                except:
                    pass
        print("[网络] 网络连接已关闭")

# ===== 触摸界面管理器 =====
class TouchInterface:
    """触摸屏界面管理器"""
    
    def __init__(self, camera_manager, audio_manager, network_manager):
        self.camera_manager = camera_manager
        self.audio_manager = audio_manager
        self.network_manager = network_manager
        self.is_running = True
        
        if HAS_PYGAME:
            self.init_pygame()
    
    def init_pygame(self):
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 480))
            pygame.display.set_caption("医疗诊断系统 - 开发板")
            self.clock = pygame.time.Clock()

            font_path = "/home/nle/notebook/Intelligent_diagnosis_system/ultralytics-main/1611458310630572.ttf"
            if os.path.exists(font_path):
                self.font = pygame.font.Font(font_path, 18)
                print("[界面] 中文字体加载成功")
            else:
                self.font = pygame.font.Font(None, 36)
                print("[警告] 找不到中文字体，使用默认字体")

            print("[界面] Pygame界面初始化完成")
        except Exception as e:
            print(f"[错误] Pygame初始化失败: {e}")
        
        print("[界面] 启动触摸界面...")
        
        while self.is_running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.is_running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self.handle_touch(event.pos, "press")
                    elif event.type == pygame.MOUSEBUTTONUP:
                        self.handle_touch(event.pos, "release")
                    elif event.type == pygame.MOUSEMOTION:
                        if pygame.mouse.get_pressed()[0]:
                            self.handle_touch(event.pos, "move")
                
                self.draw_interface()
                self.clock.tick(30)
                
            except Exception as e:
                print(f"[错误] 界面运行错误: {e}")
                break
        
        if HAS_PYGAME:
            pygame.quit()
    
    def run_interface(self):
        """运行触摸界面"""
        if not HAS_PYGAME:
            print("[界面] 图形界面不可用，使用控制台模式")
            self.run_console_interface()
            return
        
        print("[界面] 启动触摸界面...")
        print("[提示] 在JupyterLab环境中，建议使用控制台模式进行测试")
        
        # 在JupyterLab中，优先使用控制台模式
        if self._is_jupyter_environment():
            print("[Jupyter] 检测到Jupyter环境，启动控制台模式")
            self.run_console_interface()
            return
        
        # 正常的Pygame界面
        while self.is_running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.is_running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self.handle_touch(event.pos, "press")
                    elif event.type == pygame.MOUSEBUTTONUP:
                        self.handle_touch(event.pos, "release")
                    elif event.type == pygame.MOUSEMOTION:
                        if pygame.mouse.get_pressed()[0]:
                            self.handle_touch(event.pos, "move")
                
                self.draw_interface()
                self.clock.tick(30)
                
            except Exception as e:
                print(f"[错误] 界面运行错误: {e}")
                break
        
        if HAS_PYGAME:
            pygame.quit()
    
    def _is_jupyter_environment(self):
        """检测是否在Jupyter环境中运行"""
        try:
            # 检查是否在Jupyter环境中
            import IPython
            return IPython.get_ipython() is not None
        except:
            return False
    
    def run_console_interface(self):
        """运行控制台界面"""
        print("\n[控制台] 医疗诊断系统控制台模式")
        print("可用命令:")
        print("  'c' - 启动摄像头预览")
        print("  'p' - 拍照并诊断")
        print("  'r' - 开始录音")
        print("  'v' - 切换视频流传输")
        print("  't' - 测试网络连接")
        print("  's' - 系统状态")
        print("  'q' - 退出")
        print("  'h' - 显示帮助信息")
        
        while self.is_running:
            try:
                cmd = input("\n请输入命令: ").strip().lower()
                
                if cmd == 'c':
                    self.camera_manager.start_preview()
                elif cmd == 'p':
                    self.capture_and_diagnose()
                elif cmd == 'r':
                    self.start_voice_recording()
                elif cmd == 'v':
                    self.toggle_video_streaming()
                elif cmd == 't':
                    self.test_network_connection()
                elif cmd == 's':
                    self.show_system_info()
                elif cmd == 'h':
                    self.show_help()
                elif cmd == 'q':
                    self.is_running = False
                else:
                    print("[提示] 无效命令，输入 'h' 查看帮助")
                    
            except KeyboardInterrupt:
                self.is_running = False
                break
    
    def capture_and_diagnose(self):
        """拍照并诊断"""
        print("[操作] 开始拍照诊断...")
        
        # 首先测试网络连接
        if not self.network_manager.test_connection():
            print("[错误] 网络连接失败，无法发送诊断请求")
            self.show_network_error("网络连接失败", "请检查PC端是否启动，或网络配置是否正确")
            return
        
        if not self.camera_manager.camera_thread:
            print("[错误] 摄像头未初始化")
            return
        
        # 获取当前帧
        frame = self.camera_manager.camera_thread.capture_frame()
        if frame is None:
            print("[错误] 无法获取图像")
            return
        
        # 图像增强
        enhanced_frame = self.camera_manager.enhance_image(frame)
        
        # 保存图像到本地
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnosis_{timestamp}.jpg"
        filepath = self.camera_manager.save_image(enhanced_frame, filename)
        
        if filepath:
            print(f"[保存] 图像已保存到本地: {filepath}")
            
            # 选择操作模式
            print("\n[选择] 请选择操作模式:")
            print("  1. 仅保存到PC端（推荐）")
            print("  2. 保存到PC端并进行AI诊断")
            print("  3. 仅本地保存")
            
            try:
                choice = input("请输入选择 (1/2/3，默认1): ").strip()
                if not choice:
                    choice = "1"
                
                if choice == "1":
                    # 仅保存到PC端
                    self.show_sending_status("正在保存图像到PC端...")
                    request_id = self.network_manager.save_image_to_pc(enhanced_frame, filename)
                    
                    if request_id:
                        self.hide_sending_status()
                        print(f"✅ [成功] 图像已保存到PC端: {filename}")
                        self.show_connection_success("保存成功", f"图像已保存到PC端: {filename}")
                    else:
                        self.hide_sending_status()
                        print("[失败] 图像保存到PC端失败")
                        
                elif choice == "2":
                    # 保存到PC端并进行AI诊断
                    self.show_sending_status("正在发送图像到PC端进行诊断...")
                    request_id = self.network_manager.send_image_for_diagnosis(enhanced_frame, save_to_pc=True)
                    
                    if request_id:
                        self.show_sending_status("等待AI诊断结果...")
                        result = self.network_manager.wait_for_diagnosis_result(request_id)
                        
                        if result:
                            self.hide_sending_status()
                            self.display_diagnosis_result(result)
                        else:
                            self.hide_sending_status()
                            self.show_network_error("诊断超时", "PC端未在预期时间内返回诊断结果")
                    else:
                        self.hide_sending_status()
                        print("[失败] 图像发送失败")
                        
                elif choice == "3":
                    # 仅本地保存
                    print("✅ [完成] 图像仅保存在本地")
                    
                else:
                    print("[提示] 无效选择，默认仅保存到PC端")
                    self.show_sending_status("正在保存图像到PC端...")
                    request_id = self.network_manager.save_image_to_pc(enhanced_frame, filename)
                    
                    if request_id:
                        self.hide_sending_status()
                        print(f"✅ [成功] 图像已保存到PC端: {filename}")
                    else:
                        self.hide_sending_status()
                        print("[失败] 图像保存到PC端失败")
                        
            except (EOFError, KeyboardInterrupt):
                print("\n[提示] 使用默认模式：仅保存到PC端")
                self.show_sending_status("正在保存图像到PC端...")
                request_id = self.network_manager.save_image_to_pc(enhanced_frame, filename)
                
                if request_id:
                    self.hide_sending_status()
                    print(f"✅ [成功] 图像已保存到PC端: {filename}")
                else:
                    self.hide_sending_status()
                    print("[失败] 图像保存到PC端失败")
        else:
            print("[失败] 图像保存失败")
    
    def show_network_error(self, title, message):
        """显示网络错误信息"""
        if HAS_PYGAME:
            # 在Pygame界面显示错误
            error_surface = pygame.Surface((600, 200))
            error_surface.fill((200, 100, 100))
            
            title_font = pygame.font.Font(None, 48)
            msg_font = pygame.font.Font(None, 32)
            
            title_text = title_font.render(title, True, (255, 255, 255))
            msg_text = msg_font.render(message, True, (255, 255, 255))
            
            error_surface.blit(title_text, (20, 20))
            error_surface.blit(msg_text, (20, 80))
            
            self.screen.blit(error_surface, (100, 140))
            pygame.display.flip()
            
            # 3秒后自动清除
            pygame.time.wait(3000)
        else:
            # 控制台模式显示错误
            print(f"\n[网络错误] {title}: {message}")
    
    def show_sending_status(self, message):
        """显示发送状态"""
        if HAS_PYGAME:
            self.sending_message = message
            self.sending_start_time = time.time()
        else:
            print(f"[状态] {message}")
    
    def hide_sending_status(self):
        """隐藏发送状态"""
        if HAS_PYGAME:
            self.sending_message = None
        else:
            print("[状态] 操作完成")
    
    def draw_interface(self):
        """绘制界面"""
        if not HAS_PYGAME:
            return
            
        # 清空屏幕
        self.screen.fill((30, 30, 30))
        
        # 绘制按钮
        pygame.draw.rect(self.screen, (100, 150, 100), (50, 50, 150, 50))
        pygame.draw.rect(self.screen, (150, 100, 100), (50, 120, 150, 50))
        pygame.draw.rect(self.screen, (100, 100, 150), (50, 190, 150, 50))
        pygame.draw.rect(self.screen, (150, 150, 100), (50, 260, 150, 50))  # 连接测试按钮
        pygame.draw.rect(self.screen, (100, 150, 150), (50, 330, 150, 50))  # 视频流按钮
        
        # 绘制按钮文字
        text1 = self.font.render("拍照诊断", True, (255, 255, 255))
        text2 = self.font.render("语音对话", True, (255, 255, 255))
        text3 = self.font.render("摄像头预览", True, (255, 255, 255))
        text4 = self.font.render("连接测试", True, (255, 255, 255))
        
        # 视频流按钮文字根据状态变化
        stream_status = "停止流传输" if self.network_manager.is_streaming else "开始流传输"
        text5 = self.font.render(stream_status, True, (255, 255, 255))
        
        self.screen.blit(text1, (60, 65))
        self.screen.blit(text2, (60, 135))
        self.screen.blit(text3, (60, 205))
        self.screen.blit(text4, (60, 275))
        self.screen.blit(text5, (60, 345))
        
        # 绘制状态信息
        self.draw_status()
        
        # 绘制发送状态
        if hasattr(self, 'sending_message') and self.sending_message:
            self.draw_sending_status()
        
        pygame.display.flip()
    
    def draw_sending_status(self):
        """绘制发送状态"""
        if not HAS_PYGAME:
            return
            
        # 创建半透明状态条
        status_surface = pygame.Surface((600, 60))
        status_surface.set_alpha(200)
        status_surface.fill((50, 100, 200))
        
        # 绘制状态文字
        status_font = pygame.font.Font(None, 28)
        status_text = status_font.render(self.sending_message, True, (255, 255, 255))
        
        # 计算文字位置（居中）
        text_rect = status_text.get_rect()
        text_rect.center = (300, 30)
        
        status_surface.blit(status_text, text_rect)
        self.screen.blit(status_surface, (100, 400))
        
        # 绘制进度指示器
        if hasattr(self, 'sending_start_time'):
            elapsed = time.time() - self.sending_start_time
            progress = min(elapsed / 10.0, 1.0)  # 10秒内显示进度
            
            progress_width = int(500 * progress)
            pygame.draw.rect(self.screen, (100, 200, 100), (100, 470, progress_width, 8))
            pygame.draw.rect(self.screen, (100, 100, 100), (100, 470, 500, 8), 2)
    
    def draw_status(self):
        """绘制状态信息"""
        if not HAS_PYGAME:
            return
            
        status_text = []
        
        # 摄像头状态
        camera_status = self.camera_manager.get_camera_status()
        if camera_status["initialized"]:
            status_text.append(f"摄像头: {'运行中' if camera_status['running'] else '已停止'}")
        else:
            status_text.append("摄像头: 未初始化")
        
        # 音频状态
        if HAS_AUDIO:
            status_text.append("音频: 可用")
        else:
            status_text.append("音频: 不可用")
        
        # 网络状态
        network_status = self.network_manager.get_connection_status()
        if network_status["connected"]:
            status_text.append(f"PC连接: {network_status['pc_ip']} ✅")
        else:
            status_text.append(f"PC连接: {network_status['pc_ip']} ❌")
        
        # 绘制状态文字
        for i, text in enumerate(status_text):
            color = (200, 200, 200) if "✅" not in text else (100, 255, 100)
            rendered = self.font.render(text, True, color)
            self.screen.blit(rendered, (250, 50 + i * 30))
    
    def handle_touch(self, pos, event_type):
        """处理触摸事件"""
        x, y = pos
        
        # 将触摸坐标发送到PC端
        self.network_manager.send_touch_control(event_type, x, y)
        
        # 处理界面按钮
        if event_type == "press":
            if 50 <= x <= 200 and 50 <= y <= 100:  # 拍照按钮
                self.capture_and_diagnose()
            elif 50 <= x <= 200 and 120 <= y <= 170:  # 录音按钮
                if HAS_AUDIO:
                    self.start_voice_recording()
            elif 50 <= x <= 200 and 190 <= y <= 240:  # 预览按钮
                self.camera_manager.start_preview()
            elif 50 <= x <= 200 and 260 <= y <= 310:  # 连接测试按钮
                self.test_network_connection()
            elif 50 <= x <= 200 and 330 <= y <= 380:  # 视频流按钮
                self.toggle_video_streaming()
    
    def test_network_connection(self):
        """测试网络连接"""
        print("[网络] 开始测试网络连接...")
        
        if HAS_PYGAME:
            self.show_sending_status("正在测试网络连接...")
        
        # 执行连接测试
        success = self.network_manager.test_connection()
        
        if HAS_PYGAME:
            self.hide_sending_status()
        
        if success:
            print("[网络] 连接测试成功")
            if HAS_PYGAME:
                self.show_connection_success("网络连接正常", "PC端通信正常")
        else:
            print("[网络] 连接测试失败")
            if HAS_PYGAME:
                self.show_network_error("连接测试失败", "无法连接到PC端，请检查网络配置")
    
    def show_connection_success(self, title, message):
        """显示连接成功信息"""
        if HAS_PYGAME:
            success_surface = pygame.Surface((600, 200))
            success_surface.fill((100, 200, 100))
            
            title_font = pygame.font.Font(None, 48)
            msg_font = pygame.font.Font(None, 32)
            
            title_text = title_font.render(title, True, (255, 255, 255))
            msg_text = msg_font.render(message, True, (255, 255, 255))
            
            success_surface.blit(title_text, (20, 20))
            success_surface.blit(msg_text, (20, 80))
            
            self.screen.blit(success_surface, (100, 140))
            pygame.display.flip()
            
            # 2秒后自动清除
            pygame.time.wait(2000)
        else:
            print(f"\n[连接成功] {title}: {message}")
    
    def toggle_video_streaming(self):
        """切换视频流传输状态"""
        if self.network_manager.is_streaming:
            # 停止视频流
            self.camera_manager.stop_streaming()
            self.network_manager.stop_streaming()
            print("[流传输] 视频流传输已停止")
            if HAS_PYGAME:
                self.show_connection_success("流传输已停止", "视频流传输已关闭")
        else:
            # 启动视频流
            if not self.network_manager.connection_status:
                # 先测试连接
                if not self.network_manager.test_connection():
                    if HAS_PYGAME:
                        self.show_network_error("连接失败", "无法连接到PC端，请检查网络")
                    return
            
            self.camera_manager.start_streaming()
            self.network_manager.start_streaming()
            print("[流传输] 视频流传输已启动")
            if HAS_PYGAME:
                self.show_connection_success("流传输已启动", "视频流正在发送到PC端")
    
    def start_voice_recording(self):
        """开始语音录音"""
        if not HAS_AUDIO:
            print("[警告] 音频功能不可用")
            return
        
        print("[语音] 开始录音...")
        
        # 发送录音开始命令
        self.network_manager.send_voice_command("start_recording")
        
        # 录音
        audio_data = self.audio_manager.start_recording(duration=5)
        
        if audio_data:
            # 发送语音数据
            metadata = {
                "source": "board_microphone",
                "purpose": "voice_chat"
            }
            
            success = self.network_manager.send_voice_data(audio_data, metadata)
            
            if success:
                print("[成功] 语音已发送到PC端，等待回复...")
                self.network_manager.send_voice_command("process_voice", {
                    "action": "chat",
                    "expect_response": True
                })
            else:
                print("[失败] 语音发送失败")
        else:
            print("[失败] 录音失败")
    
    def display_diagnosis_result(self, result):
        """显示诊断结果"""
        print("\n" + "="*50)
        print("[医疗AI] 诊断结果")
        print("="*50)
        
        if 'disease_name' in result:
            print(f"[诊断] 疾病名称: {result['disease_name']}")
        
        if 'confidence' in result:
            confidence = result['confidence']
            print(f"[置信度] {confidence:.2%}")
            
            if confidence > 0.8:
                print("[评级] 高置信度诊断")
            elif confidence > 0.6:
                print("[评级] 中等置信度诊断")
            else:
                print("[评级] 低置信度诊断，建议专业医生确认")
        
        if 'advice' in result:
            print(f"[建议] {result['advice']}")
        
        if 'emergency' in result and result['emergency']:
            print("[⚠️ 紧急] 紧急情况！建议立即就医")
        
        print("="*50)
    
    def show_help(self):
        """显示帮助信息"""
        print("\n[帮助] 医疗诊断系统使用说明")
        print("=" * 50)
        print("📹 摄像头功能:")
        print("  'c' - 启动摄像头预览（OpenCV窗口）")
        print("  'p' - 拍照并发送到PC端进行AI诊断")
        print("  'v' - 切换视频流传输状态")
        print()
        print("🎤 语音功能:")
        print("  'r' - 开始录音并发送到PC端")
        print()
        print("🌐 网络功能:")
        print("  't' - 测试PC端网络连接")
        print("  's' - 显示系统状态信息")
        print()
        print("💡 使用建议:")
        print("  1. 先使用 't' 测试网络连接")
        print("  2. 使用 'v' 启动视频流传输")
        print("  3. 使用 'c' 启动摄像头预览")
        print("  4. 在预览窗口中使用键盘操作:")
        print("     - 空格键: 拍照诊断")
        print("     - 's'键: 保存照片")
        print("     - 'r'键: 语音对话")
        print("     - 'q'键: 退出预览")
        print("=" * 50)
    
    def show_system_info(self):
        """显示系统信息"""
        print("\n[系统信息] 开发板医疗诊断系统状态")
        print("-" * 40)
        
        # 摄像头状态
        camera_status = self.camera_manager.get_camera_status()
        print(f"摄像头状态: {camera_status}")
        
        # 音频状态
        print(f"音频功能: {'可用' if HAS_AUDIO else '不可用'}")
        
        # 图形界面状态
        print(f"图形界面: {'可用' if HAS_PYGAME else '不可用'}")
        
        # 网络配置
        print(f"PC端IP: {PC_IP}")
        print(f"摄像头端口: {CAMERA_PORT}")
        print(f"诊断端口: {DIAGNOSIS_PORT}")
        
        if HAS_AUDIO:
            print(f"语音发送端口: {VOICE_SEND_PORT}")
            print(f"语音接收端口: {VOICE_RECEIVE_PORT}")
        
        print("-" * 40)

# ===== 主系统类 =====
class BoardMedicalSystem:
    """开发板医疗诊断主系统"""
    
    def __init__(self):
        self.camera_manager = CameraManager()
        self.audio_manager = AudioManager()
        self.network_manager = NetworkManager()
        self.touch_interface = None
        self.is_running = False
        
    def initialize(self):
        """初始化系统"""
        print("[启动] 开发板医疗诊断系统初始化...")
        print(f"[环境] 音频功能: {'✅ 可用' if HAS_AUDIO else '❌ 不可用'}")
        print(f"[环境] 图形界面: {'✅ 可用' if HAS_PYGAME else '❌ 不可用'}")
        
        # 初始化摄像头（带流传输回调）
        if not self.camera_manager.initialize(stream_callback=self._on_stream_frame):
            print("[错误] 摄像头初始化失败")
            return False
        
        # 初始化网络
        if not self.network_manager.init_sockets():
            print("[错误] 网络初始化失败")
            return False
        
        # 测试网络连接
        print("[网络] 正在测试PC端连接...")
        if self.network_manager.test_connection():
            print("✅ [网络] PC端连接成功")
            # 自动启动视频流传输
            self.start_video_streaming()
        else:
            print("⚠️ [网络] PC端连接失败，稍后可手动重试")
        
        # 初始化触摸界面
        self.touch_interface = TouchInterface(
            self.camera_manager, 
            self.audio_manager, 
            self.network_manager
        )
        
        self.is_running = True
        print("[成功] 系统初始化完成")
        print(f"[配置] PC端IP: {PC_IP}")
        print("[提示] 请确保PC端医疗诊断系统已启动")
        
        return True
    
    def _on_stream_frame(self, frame):
        """处理流传输帧"""
        if self.network_manager.is_streaming:
            self.network_manager.send_stream_frame(frame)
    
    def start_video_streaming(self):
        """启动视频流传输"""
        self.camera_manager.start_streaming()
        self.network_manager.start_streaming()
        print("📹 [流传输] 视频流传输已启动")
    
    def stop_video_streaming(self):
        """停止视频流传输"""
        self.camera_manager.stop_streaming()
        self.network_manager.stop_streaming()
        print("📹 [流传输] 视频流传输已停止")
    
    def run(self):
        """运行系统"""
        if not self.is_running:
            print("[错误] 系统未初始化")
            return
        
        print("\n[启动] 开发板医疗诊断系统运行中...")
        
        try:
            # 启动触摸界面（主循环）
            self.touch_interface.run_interface()
            
        except KeyboardInterrupt:
            print("\n[中断] 用户中断程序")
        except Exception as e:
            print(f"[错误] 系统运行错误: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """关闭系统"""
        print("\n[关闭] 正在关闭开发板医疗诊断系统...")
        
        self.is_running = False
        
        if self.touch_interface:
            self.touch_interface.is_running = False
        
        self.camera_manager.release()
        self.audio_manager.cleanup()
        self.network_manager.close()
        
        print("[完成] 系统已关闭")

# ===== 主函数 =====
def main():
    """主函数"""
    print("🏥 开发板医疗诊断系统")
    print("=" * 50)
    print("功能特性:")
    print("  ✅ 摄像头图像捕获与实时预览")
    print("  ✅ AI医疗诊断（与PC端交互）")
    print(f"  {'✅' if HAS_AUDIO else '❌'} 语音交互功能")
    print(f"  {'✅' if HAS_PYGAME else '❌'} 触摸屏图形界面")
    print("  ✅ 网络通信与数据传输")
    print("  ✅ 持续视频流传输")
    print("  ✅ JupyterLab单文件运行支持")
    print("=" * 50)
    print(f"📡 网络配置: PC端IP = {PC_IP}")
    print(f"📹 摄像头端口: {CAMERA_PORT}")
    print(f"🔍 诊断端口: {DIAGNOSIS_PORT}")
    print(f"⚙️ 命令端口: {COMMAND_PORT}")
    if HAS_AUDIO:
        print(f"🎤 语音发送端口: {VOICE_SEND_PORT}")
        print(f"🔊 语音接收端口: {VOICE_RECEIVE_PORT}")
    print("=" * 50)
    
    # 创建系统实例
    system = BoardMedicalSystem()
    
    try:
        # 初始化系统
        if not system.initialize():
            print("[失败] 系统初始化失败")
            return 1
        
        # 运行系统
        system.run()
        
        return 0
        
    except Exception as e:
        print(f"[错误] 系统错误: {e}")
        return 1

# ===== 程序入口 =====
if __name__ == "__main__":
    import sys
    sys.exit(main())
