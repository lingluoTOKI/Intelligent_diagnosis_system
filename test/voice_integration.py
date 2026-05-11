#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语音功能集成模块
提供简单的语音识别和合成功能接口
"""

import speech_recognition as sr
import pyttsx3
import threading
import time
from PyQt5.QtCore import QObject, pyqtSignal

class VoiceIntegration(QObject):
    """语音功能集成类"""
    
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
        self.init_voice_components()
    
    def init_voice_components(self):
        """初始化语音组件"""
        try:
            # 初始化语音识别器
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            
            # 获取麦克风
            self.microphone = sr.Microphone()
            
            # 初始化TTS引擎
            self.init_tts_engine()
            
            print("[INFO] 语音组件初始化成功")
            return True
            
        except Exception as e:
            print(f"[ERROR] 语音组件初始化失败: {e}")
            return False
    
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
        if self.is_listening or not self.recognizer:
            return False
        
        self.is_listening = True
        threading.Thread(target=self._perform_recognition, daemon=True).start()
        return True
    
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
        if not self.tts_engine or self.is_speaking or not text:
            return False
        
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
        return True
    
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

# 简单的语音功能测试
def test_voice_integration():
    """测试语音功能集成"""
    print("=" * 50)
    print("语音功能集成测试")
    print("=" * 50)
    
    # 创建语音集成对象
    voice = VoiceIntegration()
    
    # 检查状态
    status = voice.get_voice_status()
    print(f"语音识别器: {'✅ 可用' if status['recognizer_available'] else '❌ 不可用'}")
    print(f"麦克风: {'✅ 可用' if status['microphone_available'] else '❌ 不可用'}")
    print(f"TTS引擎: {'✅ 可用' if status['tts_available'] else '❌ 不可用'}")
    
    # 测试麦克风
    success, message = voice.test_microphone()
    print(f"麦克风测试: {'✅ ' if success else '❌ '}{message}")
    
    # 测试TTS
    if voice.tts_engine:
        print("测试TTS功能...")
        voice.speak_text("语音功能测试完成")
        time.sleep(2)  # 等待语音播放完成
    
    print("=" * 50)
    print("测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    test_voice_integration() 