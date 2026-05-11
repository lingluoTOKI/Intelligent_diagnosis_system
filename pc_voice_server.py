#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC端语音处理服务器
功能：
1. 接收开发板语音数据进行识别
2. 调用AI进行语音对话
3. 语音合成并发送到开发板
4. 与主诊断系统集成
"""

import socket
import threading
import time
import json
import queue
import base64
import io
import wave
import tempfile
import os
from datetime import datetime

# 语音识别和合成
try:
    import speech_recognition as sr
    import pyttsx3
    HAS_SPEECH = True
except ImportError:
    print("[警告] 语音识别库未安装，使用模拟模式")
    HAS_SPEECH = False

# VOSK离线识别
try:
    import vosk
    HAS_VOSK = True
except ImportError:
    print("[警告] VOSK库未安装，使用在线识别")
    HAS_VOSK = False

# 网络配置
VOICE_RECEIVE_PORT = 5005   # 接收开发板语音
VOICE_SEND_PORT = 5006      # 发送语音到开发板
COMMAND_PORT = 5007         # 语音命令控制

class VoiceRecognitionEngine:
    """语音识别引擎"""
    
    def __init__(self):
        self.recognizer = None
        self.vosk_model = None
        self.init_engines()
    
    def init_engines(self):
        """初始化语音识别引擎"""
        try:
            if HAS_SPEECH:
                self.recognizer = sr.Recognizer()
                print("[语音] SpeechRecognition引擎初始化完成")
            
            if HAS_VOSK:
                # 尝试加载VOSK模型
                model_paths = [
                    "vosk-model-cn-0.22",
                    "models/vosk-model-cn-0.22",
                    "../vosk-model-cn-0.22"
                ]
                
                for path in model_paths:
                    if os.path.exists(path):
                        try:
                            self.vosk_model = vosk.Model(path)
                            print(f"[语音] VOSK中文模型加载成功: {path}")
                            break
                        except Exception as e:
                            print(f"[警告] VOSK模型加载失败: {e}")
                
        except Exception as e:
            print(f"[错误] 语音引擎初始化失败: {e}")
    
    def recognize_audio(self, audio_data):
        """识别语音数据"""
        try:
            # 保存为临时WAV文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            try:
                # 优先使用VOSK离线识别
                if self.vosk_model:
                    text = self._recognize_with_vosk(tmp_path)
                    if text:
                        return text, "vosk"
                
                # 备用在线识别
                if self.recognizer:
                    text = self._recognize_with_sr(tmp_path)
                    if text:
                        return text, "online"
                
                return None, "failed"
                
            finally:
                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"[错误] 语音识别失败: {e}")
            return None, "error"
    
    def _recognize_with_vosk(self, audio_path):
        """使用VOSK进行离线识别"""
        try:
            import json as json_lib
            
            # 读取音频文件
            with wave.open(audio_path, 'rb') as wf:
                # 创建识别器
                rec = vosk.KaldiRecognizer(self.vosk_model, wf.getframerate())
                
                # 处理音频数据
                results = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    
                    if rec.AcceptWaveform(data):
                        result = json_lib.loads(rec.Result())
                        if result.get('text'):
                            results.append(result['text'])
                
                # 获取最终结果
                final_result = json_lib.loads(rec.FinalResult())
                if final_result.get('text'):
                    results.append(final_result['text'])
                
                # 合并结果
                text = ' '.join(results).strip()
                return text if text else None
                
        except Exception as e:
            print(f"[错误] VOSK识别失败: {e}")
            return None
    
    def _recognize_with_sr(self, audio_path):
        """使用SpeechRecognition进行在线识别"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            
            # 尝试多种识别引擎
            engines = [
                ("百度", lambda: self.recognizer.recognize_baidu(audio, 
                    key="your_baidu_key", secret="your_baidu_secret")),
                ("Google", lambda: self.recognizer.recognize_google(audio, language='zh-CN')),
                ("Sphinx", lambda: self.recognizer.recognize_sphinx(audio, language='zh-CN'))
            ]
            
            for engine_name, recognize_func in engines:
                try:
                    text = recognize_func()
                    if text:
                        print(f"[成功] {engine_name}识别: {text}")
                        return text
                except Exception as e:
                    print(f"[失败] {engine_name}识别失败: {e}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"[错误] 在线识别失败: {e}")
            return None

class TextToSpeechEngine:
    """文本转语音引擎"""
    
    def __init__(self):
        self.tts_engine = None
        self.init_tts()
    
    def init_tts(self):
        """初始化TTS引擎"""
        try:
            if HAS_SPEECH:
                self.tts_engine = pyttsx3.init()
                if self.tts_engine:
                    # 设置语音参数
                    self.tts_engine.setProperty('rate', 180)  # 语速
                    self.tts_engine.setProperty('volume', 0.8)  # 音量
                    
                    # 尝试设置中文语音
                    voices = self.tts_engine.getProperty('voices')
                    for voice in voices:
                        if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                    
                    print("[语音] TTS引擎初始化完成")
                
        except Exception as e:
            print(f"[错误] TTS引擎初始化失败: {e}")
    
    def text_to_speech(self, text):
        """文本转语音"""
        try:
            if not self.tts_engine:
                return None
            
            # 生成临时音频文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # 保存到文件
                self.tts_engine.save_to_file(text, tmp_path)
                self.tts_engine.runAndWait()
                
                # 读取生成的音频文件
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()
                
                print(f"[TTS] 语音合成完成: {len(audio_data)} 字节")
                return audio_data
                
            finally:
                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"[错误] 语音合成失败: {e}")
            return None

class AIDialogManager:
    """AI对话管理器"""
    
    def __init__(self):
        self.deepseek_api = None
        self.init_ai()
    
    def init_ai(self):
        """初始化AI对话"""
        try:
            # 尝试导入DeepSeek API
            import sys
            sys.path.append('.')
            from visualization_test2 import DeepSeekAPI
            
            # 读取API密钥
            try:
                with open('saved_api_key.txt', 'r') as f:
                    api_key = f.read().strip()
                if api_key:
                    self.deepseek_api = DeepSeekAPI(api_key)
                    print("[AI] DeepSeek API初始化完成")
                else:
                    print("[警告] API密钥为空")
            except FileNotFoundError:
                print("[警告] 未找到API密钥文件")
                
        except Exception as e:
            print(f"[警告] AI对话初始化失败: {e}")
    
    def get_ai_response(self, user_text):
        """获取AI回复"""
        try:
            if self.deepseek_api:
                # 构造医疗对话提示
                prompt = f"""
作为一个专业的AI医疗助手，请回答用户的问题。
请提供准确、专业但易懂的医疗建议。
如果涉及严重疾病，请建议用户就医。

用户问题：{user_text}

请用简洁、温和的语气回答，控制在100字以内。
"""
                
                response = self.deepseek_api.get_custom_advice(prompt)
                
                if response and "error" not in response.lower():
                    return response
                else:
                    return self._get_fallback_response(user_text)
            else:
                return self._get_fallback_response(user_text)
                
        except Exception as e:
            print(f"[错误] AI回复获取失败: {e}")
            return self._get_fallback_response(user_text)
    
    def _get_fallback_response(self, user_text):
        """备用回复"""
        responses = {
            "你好": "您好！我是AI医疗助手，很高兴为您服务。请问有什么可以帮助您的吗？",
            "谢谢": "不客气！如果您还有其他问题，请随时询问。",
            "头疼": "头疼可能有多种原因，如疲劳、压力或其他疾病。建议您注意休息，如果持续不适请及时就医。",
            "发烧": "发烧是身体的免疫反应。请多喝水、注意休息，体温超过38.5度建议就医。",
            "咳嗽": "咳嗽可能是感冒或其他呼吸道疾病的症状。建议多喝温水，如果症状加重请及时就医。"
        }
        
        # 简单关键词匹配
        for keyword, response in responses.items():
            if keyword in user_text:
                return response
        
        return "抱歉，我暂时无法理解您的问题。建议您咨询专业医生或使用主系统的图像诊断功能。"

class VoiceServer:
    """语音服务器主控制器"""
    
    def __init__(self):
        self.recognition_engine = VoiceRecognitionEngine()
        self.tts_engine = TextToSpeechEngine()
        self.ai_manager = AIDialogManager()
        
        # 网络套接字
        self.voice_receive_sock = None
        self.voice_send_sock = None
        self.command_sock = None
        
        # 处理队列
        self.voice_queue = queue.Queue()
        self.is_running = True
        
        self.init_network()
    
    def init_network(self):
        """初始化网络"""
        try:
            # 接收语音数据
            self.voice_receive_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.voice_receive_sock.bind(("0.0.0.0", VOICE_RECEIVE_PORT))
            self.voice_receive_sock.settimeout(1.0)
            
            # 发送语音数据
            self.voice_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # 命令控制
            self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.command_sock.bind(("0.0.0.0", COMMAND_PORT))
            self.command_sock.settimeout(1.0)
            
            print("[网络] 语音服务器网络初始化完成")
            print(f"   语音接收端口: {VOICE_RECEIVE_PORT}")
            print(f"   语音发送端口: {VOICE_SEND_PORT}")
            print(f"   命令控制端口: {COMMAND_PORT}")
            
        except Exception as e:
            print(f"[错误] 网络初始化失败: {e}")
    
    def start_server(self):
        """启动服务器"""
        print("[启动] PC端语音服务器启动中...")
        
        # 启动处理线程
        threads = [
            threading.Thread(target=self.voice_receive_worker, daemon=True),
            threading.Thread(target=self.command_handler, daemon=True),
            threading.Thread(target=self.voice_processor, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        print("[成功] 语音服务器启动完成")
        return True
    
    def voice_receive_worker(self):
        """语音接收工作线程"""
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
                    
                    # 解析语音数据
                    self.handle_voice_data(complete_data, addr)
                    
                    # 清理缓存
                    del packet_buffer[total_packets]
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"[错误] 语音接收错误: {e}")
    
    def handle_voice_data(self, data, client_addr):
        """处理接收到的语音数据"""
        try:
            # 解析JSON数据包
            packet = json.loads(data.decode('utf-8'))
            
            if packet.get("type") == "voice_data":
                # 解码音频数据
                audio_data = base64.b64decode(packet["audio_data"])
                
                print(f"[接收] 收到语音数据 ({len(audio_data)} 字节) 来自 {client_addr[0]}")
                
                # 添加到处理队列
                self.voice_queue.put({
                    "audio_data": audio_data,
                    "client_addr": client_addr,
                    "metadata": packet.get("metadata", {})
                })
                
        except Exception as e:
            print(f"[错误] 语音数据处理失败: {e}")
    
    def command_handler(self):
        """命令处理线程"""
        print("[命令] 语音命令处理线程启动")
        
        while self.is_running:
            try:
                data, addr = self.command_sock.recvfrom(4096)
                
                try:
                    command = json.loads(data.decode('utf-8'))
                    self.handle_voice_command(command, addr)
                    
                except json.JSONDecodeError:
                    print(f"[警告] 无效命令格式来自 {addr[0]}")
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"[错误] 命令处理错误: {e}")
    
    def handle_voice_command(self, command, client_addr):
        """处理语音命令"""
        try:
            cmd_type = command.get("command")
            params = command.get("params", {})
            
            print(f"[命令] 收到语音命令: {cmd_type}")
            
            if cmd_type == "text_to_speech":
                text = params.get("text", "")
                if text:
                    self.process_tts_request(text, client_addr)
            
            elif cmd_type == "start_recording":
                print("[命令] 开发板开始录音")
            
            elif cmd_type == "process_voice":
                print("[命令] 处理语音识别请求")
            
        except Exception as e:
            print(f"[错误] 命令处理失败: {e}")
    
    def voice_processor(self):
        """语音处理工作线程"""
        print("[处理] 语音处理线程启动")
        
        while self.is_running:
            try:
                if not self.voice_queue.empty():
                    voice_task = self.voice_queue.get()
                    self.process_voice_recognition(voice_task)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"[错误] 语音处理错误: {e}")
    
    def process_voice_recognition(self, voice_task):
        """处理语音识别"""
        try:
            audio_data = voice_task["audio_data"]
            client_addr = voice_task["client_addr"]
            
            print("[识别] 开始语音识别...")
            
            # 语音识别
            text, engine = self.recognition_engine.recognize_audio(audio_data)
            
            if text:
                print(f"[识别] 识别结果({engine}): {text}")
                
                # 获取AI回复
                ai_response = self.ai_manager.get_ai_response(text)
                print(f"[AI] AI回复: {ai_response}")
                
                # 语音合成
                tts_audio = self.tts_engine.text_to_speech(ai_response)
                
                if tts_audio:
                    # 发送合成语音到开发板
                    self.send_tts_to_board(tts_audio, client_addr)
                else:
                    print("[警告] 语音合成失败")
            else:
                print("[失败] 语音识别失败")
                # 发送错误提示
                error_msg = "抱歉，我没有听清楚，请重新说一遍。"
                error_audio = self.tts_engine.text_to_speech(error_msg)
                if error_audio:
                    self.send_tts_to_board(error_audio, client_addr)
                    
        except Exception as e:
            print(f"[错误] 语音识别处理失败: {e}")
    
    def process_tts_request(self, text, client_addr):
        """处理TTS请求"""
        try:
            print(f"[TTS] 处理文本转语音: {text}")
            
            audio_data = self.tts_engine.text_to_speech(text)
            
            if audio_data:
                self.send_tts_to_board(audio_data, client_addr)
            else:
                print("[失败] 语音合成失败")
                
        except Exception as e:
            print(f"[错误] TTS处理失败: {e}")
    
    def send_tts_to_board(self, audio_data, client_addr):
        """发送TTS音频到开发板"""
        try:
            # 编码音频数据
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 构造数据包
            packet = {
                "type": "tts_audio",
                "timestamp": datetime.now().isoformat(),
                "audio_data": audio_b64,
                "format": "wav"
            }
            
            # 分片发送
            packet_json = json.dumps(packet)
            packet_bytes = packet_json.encode('utf-8')
            
            max_packet_size = 8192
            total_packets = (len(packet_bytes) + max_packet_size - 1) // max_packet_size
            
            for i in range(total_packets):
                start = i * max_packet_size
                end = min(start + max_packet_size, len(packet_bytes))
                
                header = i.to_bytes(4, 'big') + total_packets.to_bytes(4, 'big')
                packet_chunk = header + packet_bytes[start:end]
                
                self.voice_send_sock.sendto(packet_chunk, (client_addr[0], VOICE_SEND_PORT))
                time.sleep(0.01)
            
            print(f"[发送] TTS音频已发送到开发板 ({len(audio_data)} 字节)")
            
        except Exception as e:
            print(f"[错误] TTS发送失败: {e}")
    
    def stop_server(self):
        """停止服务器"""
        print("[停止] 正在关闭语音服务器...")
        self.is_running = False
        
        # 关闭套接字
        for sock in [self.voice_receive_sock, self.voice_send_sock, self.command_sock]:
            if sock:
                try:
                    sock.close()
                except:
                    pass
        
        print("[完成] 语音服务器已关闭")

def main():
    """主函数"""
    print("[语音] PC端语音处理服务器")
    print("=" * 50)
    
    server = VoiceServer()
    
    try:
        if server.start_server():
            print("[运行] 语音服务器运行中，按 Ctrl+C 停止")
            
            # 保持服务器运行
            while server.is_running:
                time.sleep(1)
        else:
            print("[错误] 语音服务器启动失败")
            
    except KeyboardInterrupt:
        print("\n[停止] 收到停止信号")
    except Exception as e:
        print(f"[错误] 服务器错误: {e}")
    finally:
        server.stop_server()

if __name__ == "__main__":
    main()
