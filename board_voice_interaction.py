#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发板语音交互模块
功能：
1. 开发板麦克风录音 -> PC端语音识别
2. PC端语音合成 -> 开发板扬声器播放
3. 支持实时语音对话
4. 与摄像头功能协同工作
"""

import socket
import threading
import time
import json
import queue
import numpy as np
import pyaudio
import wave
import io
import base64
from datetime import datetime

# 音频配置
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5

# 网络配置  
PC_IP = "172.20.10.3"
VOICE_SEND_PORT = 5005      # 发送语音数据到PC
VOICE_RECEIVE_PORT = 5006   # 接收PC语音合成
COMMAND_PORT = 5007         # 语音命令控制

class AudioManager:
    """音频管理器"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.is_playing = False
        self.recording_thread = None
        self.playback_queue = queue.Queue()
        
        # 初始化音频设备
        self.init_audio_devices()
        
    def init_audio_devices(self):
        """初始化音频设备"""
        try:
            # 查找可用的音频设备
            print("[音频] 可用音频设备:")
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                print(f"  设备 {i}: {info['name']} - 输入:{info['maxInputChannels']}, 输出:{info['maxOutputChannels']}")
            
            # 使用默认设备
            self.input_device = None  # 默认输入设备
            self.output_device = None # 默认输出设备
            
            print("[成功] 音频设备初始化完成")
            
        except Exception as e:
            print(f"[错误] 音频设备初始化失败: {e}")
    
    def start_recording(self, duration=5):
        """开始录音"""
        if self.is_recording:
            print("[警告] 正在录音中...")
            return None
            
        try:
            print(f"[录音] 开始录音 {duration} 秒...")
            self.is_recording = True
            
            # 打开音频流
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
        """播放音频数据"""
        try:
            # 将音频数据放入播放队列
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
                
                # 解析WAV数据
                wav_buffer = io.BytesIO(audio_data)
                with wave.open(wav_buffer, 'rb') as wf:
                    # 打开播放流
                    stream = self.audio.open(
                        format=self.audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        output_device_index=self.output_device
                    )
                    
                    # 播放音频
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
        """清理资源"""
        self.is_recording = False
        self.is_playing = False
        if self.audio:
            self.audio.terminate()

class VoiceNetworkManager:
    """语音网络管理器"""
    
    def __init__(self, audio_manager):
        self.audio_manager = audio_manager
        self.is_running = True
        
        # 网络套接字
        self.voice_send_sock = None
        self.voice_receive_sock = None
        self.command_sock = None
        
        self.init_network()
        
    def init_network(self):
        """初始化网络连接"""
        try:
            # 发送语音数据的套接字
            self.voice_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # 接收语音合成的套接字
            self.voice_receive_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.voice_receive_sock.bind(("0.0.0.0", VOICE_RECEIVE_PORT))
            self.voice_receive_sock.settimeout(1.0)
            
            # 命令控制套接字
            self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            print("[网络] 语音网络连接初始化完成")
            
            # 启动接收线程
            threading.Thread(target=self.voice_receive_worker, daemon=True).start()
            
        except Exception as e:
            print(f"[错误] 语音网络初始化失败: {e}")
    
    def send_voice_to_pc(self, audio_data, metadata=None):
        """发送语音数据到PC端"""
        try:
            if not audio_data:
                return False
                
            # 编码音频数据
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 构造数据包
            packet = {
                "type": "voice_data",
                "timestamp": datetime.now().isoformat(),
                "audio_data": audio_b64,
                "format": "wav",
                "sample_rate": RATE,
                "channels": CHANNELS,
                "duration": len(audio_data) / (RATE * CHANNELS * 2),  # 估算时长
                "metadata": metadata or {}
            }
            
            # 分片发送（音频数据可能较大）
            packet_json = json.dumps(packet)
            packet_bytes = packet_json.encode('utf-8')
            
            max_packet_size = 8192  # 8KB分片
            total_packets = (len(packet_bytes) + max_packet_size - 1) // max_packet_size
            
            for i in range(total_packets):
                start = i * max_packet_size
                end = min(start + max_packet_size, len(packet_bytes))
                
                # 包头：4字节包序号 + 4字节总包数 + 数据
                header = i.to_bytes(4, 'big') + total_packets.to_bytes(4, 'big')
                packet_chunk = header + packet_bytes[start:end]
                
                self.voice_send_sock.sendto(packet_chunk, (PC_IP, VOICE_SEND_PORT))
                time.sleep(0.01)  # 避免网络拥塞
            
            print(f"[发送] 语音数据已发送到PC端 ({len(audio_data)} 字节)")
            return True
            
        except Exception as e:
            print(f"[错误] 语音发送失败: {e}")
            return False
    
    def voice_receive_worker(self):
        """接收PC端语音合成的工作线程"""
        print("[接收] 语音接收线程启动")
        packet_buffer = {}
        
        while self.is_running:
            try:
                data, addr = self.voice_receive_sock.recvfrom(8192 + 8)
                
                if len(data) < 8:
                    continue
                
                # 解析包头
                packet_id = int.from_bytes(data[0:4], 'big')
                total_packets = int.from_bytes(data[4:8], 'big')
                payload = data[8:]
                
                # 缓存分片
                if total_packets not in packet_buffer:
                    packet_buffer[total_packets] = {}
                
                packet_buffer[total_packets][packet_id] = payload
                
                # 检查是否接收完整
                if len(packet_buffer[total_packets]) == total_packets:
                    # 重组数据
                    complete_data = b''.join([
                        packet_buffer[total_packets][i] 
                        for i in range(total_packets)
                    ])
                    
                    # 处理完整的语音数据
                    self.handle_received_voice(complete_data)
                    
                    # 清理缓存
                    del packet_buffer[total_packets]
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"[错误] 语音接收错误: {e}")
    
    def handle_received_voice(self, data):
        """处理接收到的语音数据"""
        try:
            # 解析JSON数据
            packet = json.loads(data.decode('utf-8'))
            
            if packet.get("type") == "tts_audio":
                # 解码音频数据
                audio_data = base64.b64decode(packet["audio_data"])
                
                print(f"[接收] 收到PC端语音合成 ({len(audio_data)} 字节)")
                
                # 播放音频
                self.audio_manager.play_audio(audio_data)
                
        except Exception as e:
            print(f"[错误] 语音数据处理失败: {e}")
    
    def send_voice_command(self, command, params=None):
        """发送语音控制命令"""
        try:
            cmd_packet = {
                "type": "voice_command",
                "command": command,
                "params": params or {},
                "timestamp": datetime.now().isoformat()
            }
            
            cmd_data = json.dumps(cmd_packet).encode('utf-8')
            self.command_sock.sendto(cmd_data, (PC_IP, COMMAND_PORT))
            
            print(f"[命令] 已发送语音命令: {command}")
            
        except Exception as e:
            print(f"[错误] 命令发送失败: {e}")
    
    def cleanup(self):
        """清理网络资源"""
        self.is_running = False
        
        for sock in [self.voice_send_sock, self.voice_receive_sock, self.command_sock]:
            if sock:
                try:
                    sock.close()
                except:
                    pass

class BoardVoiceInterface:
    """开发板语音交互界面"""
    
    def __init__(self):
        self.audio_manager = AudioManager()
        self.network_manager = VoiceNetworkManager(self.audio_manager)
        self.is_running = True
        
    def start_voice_chat(self):
        """开始语音对话"""
        print("\n[语音对话] 开发板语音交互启动")
        print("命令说明:")
        print("  'r' - 开始录音并发送到PC")
        print("  's' - 停止当前录音")
        print("  't' - 发送文本转语音请求")
        print("  'q' - 退出")
        
        while self.is_running:
            try:
                cmd = input("\n请输入命令: ").strip().lower()
                
                if cmd == 'r':
                    self.record_and_send()
                elif cmd == 's':
                    self.audio_manager.stop_recording()
                elif cmd == 't':
                    self.text_to_speech()
                elif cmd == 'q':
                    break
                else:
                    print("[提示] 无效命令")
                    
            except KeyboardInterrupt:
                break
        
        self.cleanup()
    
    def record_and_send(self):
        """录音并发送到PC端"""
        print("[开始] 录音中，请说话...")
        
        # 发送开始录音命令
        self.network_manager.send_voice_command("start_recording")
        
        # 录音
        audio_data = self.audio_manager.start_recording(duration=5)
        
        if audio_data:
            # 发送音频数据到PC端
            metadata = {
                "source": "board_microphone",
                "purpose": "voice_chat"
            }
            
            success = self.network_manager.send_voice_to_pc(audio_data, metadata)
            
            if success:
                print("[成功] 语音已发送到PC端，等待回复...")
                # 发送处理命令
                self.network_manager.send_voice_command("process_voice", {
                    "action": "chat",
                    "expect_response": True
                })
            else:
                print("[失败] 语音发送失败")
        else:
            print("[失败] 录音失败")
    
    def text_to_speech(self):
        """文本转语音测试"""
        text = input("请输入要转换为语音的文本: ").strip()
        
        if text:
            self.network_manager.send_voice_command("text_to_speech", {
                "text": text,
                "language": "zh",
                "voice": "default"
            })
            print("[发送] TTS请求已发送")
        else:
            print("[取消] 文本为空")
    
    def cleanup(self):
        """清理资源"""
        print("\n[清理] 正在关闭语音交互...")
        self.is_running = False
        self.network_manager.cleanup()
        self.audio_manager.cleanup()
        print("[完成] 语音交互已关闭")

def main():
    """主函数"""
    print("开发板语音交互系统")
    print("=" * 40)
    
    try:
        # 创建语音交互界面
        voice_interface = BoardVoiceInterface()
        
        # 启动语音对话
        voice_interface.start_voice_chat()
        
    except Exception as e:
        print(f"[错误] 系统启动失败: {e}")
    
    print("[退出] 系统已退出")

if __name__ == "__main__":
    main()
