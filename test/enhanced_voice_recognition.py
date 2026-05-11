#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强语音识别模块
提供更完善的语音识别和合成功能
"""

import speech_recognition as sr
import pyttsx3
import threading
import time
import json
import requests
from PyQt5.QtCore import QEvent, QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

class VoiceRecognitionEvent(QEvent):
    """语音识别事件"""
    def __init__(self, event_type, data=None):
        super().__init__(QEvent.User + 1)
        self.event_type = event_type
        self.data = data

class VoiceManager(QObject):
    """语音管理器"""
    
    # 信号定义
    voice_recognized = pyqtSignal(str)  # 语音识别完成
    voice_error = pyqtSignal(str)       # 语音识别错误
    voice_timeout = pyqtSignal()        # 语音识别超时
    voice_unknown = pyqtSignal()        # 语音无法识别
    tts_started = pyqtSignal()          # TTS开始
    tts_finished = pyqtSignal()         # TTS完成
    tts_error = pyqtSignal(str)         # TTS错误
    
    def __init__(self):
        super().__init__()
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        self.is_listening = False
        self.is_speaking = False
        
        # 初始化语音组件
        self.init_components()
    
    def init_components(self):
        """初始化语音组件"""
        try:
            # 初始化语音识别器
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 4000  # 调整能量阈值
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            
            # 获取麦克风
            self.microphone = self.get_best_microphone()
            
            # 初始化TTS引擎
            self.init_tts_engine()
            
            print("[INFO] 语音组件初始化成功")
            
        except Exception as e:
            print(f"[ERROR] 语音组件初始化失败: {e}")
    
    def get_best_microphone(self):
        """获取最佳麦克风"""
        try:
            # 列出所有音频设备
            import pyaudio
            p = pyaudio.PyAudio()
            
            print("[DEBUG] 可用音频设备:")
            input_devices = []
            
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_devices.append((i, info['name'], info['defaultSampleRate']))
                    print(f"[DEBUG]   设备 {i}: {info['name']} (采样率: {info['defaultSampleRate']})")
            
            p.terminate()
            
            # 尝试使用默认麦克风
            mic = sr.Microphone()
            
            # 测试麦克风是否可用
            with mic as source:
                print("[DEBUG] 测试麦克风访问...")
                pass
            
            print("[DEBUG] 默认麦克风测试成功")
            return mic
            
        except Exception as e:
            print(f"[WARNING] 麦克风测试失败: {e}")
            return sr.Microphone()
    
    def init_tts_engine(self):
        """初始化TTS引擎"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # 设置TTS参数
            voices = self.tts_engine.getProperty('voices')
            print(f"[DEBUG] 找到 {len(voices) if voices else 0} 个TTS语音")
            
            # 尝试设置中文语音
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    print(f"[DEBUG] 设置中文语音: {voice.name}")
                    break
            
            # 设置语音参数
            self.tts_engine.setProperty('rate', 180)    # 语音速度
            self.tts_engine.setProperty('volume', 0.8)  # 音量
            
            print("[DEBUG] TTS引擎初始化成功")
            
        except Exception as e:
            print(f"[ERROR] TTS引擎初始化失败: {e}")
            self.tts_engine = None
    
    def start_voice_recognition(self):
        """开始语音识别"""
        if self.is_listening:
            return
        
        self.is_listening = True
        threading.Thread(target=self._perform_recognition, daemon=True).start()
    
    def _perform_recognition(self):
        """执行语音识别（后台线程）"""
        try:
            print("[DEBUG] 开始语音识别...")
            
            with self.microphone as source:
                print("[DEBUG] 调整环境噪音...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("[DEBUG] 开始录音...")
                
                # 录音
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                print(f"[DEBUG] 录音完成，音频长度: {len(audio.frame_data)} bytes")
            
            # 识别语音
            text = self._recognize_audio(audio)
            
            if text:
                self.voice_recognized.emit(text)
            else:
                self.voice_unknown.emit()
                
        except sr.WaitTimeoutError:
            print("[DEBUG] 录音超时")
            self.voice_timeout.emit()
        except sr.UnknownValueError:
            print("[DEBUG] 无法识别语音内容")
            self.voice_unknown.emit()
        except Exception as e:
            print(f"[ERROR] 语音识别异常: {e}")
            self.voice_error.emit(str(e))
        finally:
            self.is_listening = False
    
    def _recognize_audio(self, audio):
        """识别音频内容"""
        # 尝试多种识别方法
        recognition_methods = [
            ('Google API (中文)', lambda: self.recognizer.recognize_google(audio, language='zh-CN')),
            ('Google API (英文)', lambda: self.recognizer.recognize_google(audio, language='en-US')),
            ('Google API (默认)', lambda: self.recognizer.recognize_google(audio)),
        ]
        
        for method_name, method_func in recognition_methods:
            try:
                text = method_func()
                print(f"[DEBUG] {method_name}识别成功: {text}")
                return text
            except Exception as e:
                print(f"[DEBUG] {method_name}识别失败: {e}")
                continue
        
        return None
    
    def speak_text(self, text):
        """播放文本语音"""
        if not self.tts_engine or self.is_speaking:
            return
        
        self.is_speaking = True
        self.tts_started.emit()
        
        def speak():
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                self.tts_finished.emit()
            except Exception as e:
                print(f"[ERROR] TTS播放失败: {e}")
                self.tts_error.emit(str(e))
            finally:
                self.is_speaking = False
        
        threading.Thread(target=speak, daemon=True).start()
    
    def stop_speaking(self):
        """停止语音播放"""
        if self.tts_engine and self.is_speaking:
            self.tts_engine.stop()
            self.is_speaking = False
    
    def test_microphone(self):
        """测试麦克风"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            return True, "麦克风测试通过"
        except Exception as e:
            return False, f"麦克风测试失败: {e}"
    
    def get_voice_status(self):
        """获取语音功能状态"""
        status = {
            'recognizer_available': self.recognizer is not None,
            'microphone_available': self.microphone is not None,
            'tts_available': self.tts_engine is not None,
            'is_listening': self.is_listening,
            'is_speaking': self.is_speaking
        }
        return status

class BaiduSpeechAPI:
    """百度语音识别API（备用方案）"""
    
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.access_token = None
        self.token_url = "https://aip.baidubce.com/oauth/2.0/token"
        self.asr_url = "https://vop.baidu.com/server_api"
    
    def get_access_token(self):
        """获取访问令牌"""
        try:
            params = {
                'grant_type': 'client_credentials',
                'client_id': self.api_key,
                'client_secret': self.secret_key
            }
            
            response = requests.post(self.token_url, params=params)
            result = response.json()
            
            if 'access_token' in result:
                self.access_token = result['access_token']
                return True
            else:
                print(f"获取百度访问令牌失败: {result}")
                return False
                
        except Exception as e:
            print(f"获取百度访问令牌异常: {e}")
            return False
    
    def recognize_speech(self, audio_data):
        """识别语音"""
        if not self.access_token:
            if not self.get_access_token():
                return None
        
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            
            params = {
                'cuid': 'python_client',
                'token': self.access_token,
                'dev_pid': 1537  # 普通话识别
            }
            
            # 将音频数据转换为base64
            import base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            data = {
                'format': 'pcm',
                'rate': 16000,
                'channel': 1,
                'token': self.access_token,
                'speech': audio_base64,
                'len': len(audio_data)
            }
            
            response = requests.post(self.asr_url, headers=headers, json=data)
            result = response.json()
            
            if result.get('err_no') == 0:
                return result['result'][0]
            else:
                print(f"百度语音识别失败: {result.get('err_msg', '未知错误')}")
                return None
                
        except Exception as e:
            print(f"百度语音识别异常: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    # 测试语音管理器
    voice_manager = VoiceManager()
    
    # 连接信号
    voice_manager.voice_recognized.connect(lambda text: print(f"识别结果: {text}"))
    voice_manager.voice_error.connect(lambda error: print(f"识别错误: {error}"))
    voice_manager.voice_timeout.connect(lambda: print("识别超时"))
    voice_manager.voice_unknown.connect(lambda: print("无法识别"))
    
    # 测试TTS
    voice_manager.tts_started.connect(lambda: print("开始播放语音"))
    voice_manager.tts_finished.connect(lambda: print("语音播放完成"))
    voice_manager.tts_error.connect(lambda error: print(f"TTS错误: {error}"))
    
    print("语音管理器测试完成") 