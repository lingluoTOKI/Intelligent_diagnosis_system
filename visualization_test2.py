# ------------------------------------------------------------
#  AI眼科疾病智诊系统
# ------------------------------------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import cv2
import json
import base64

import requests
from datetime import datetime, timedelta
from PyQt5.QtCore import Qt, QTimer, QSize, QEvent, QObject, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QPushButton, QHBoxLayout, QMessageBox,
                             QFileDialog, QStatusBar, QGroupBox, QSplitter,
                             QTextEdit, QTabWidget, QScrollArea, QProgressDialog,
                             QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QGridLayout, QSizePolicy, QLineEdit, QProgressBar, QCheckBox, QShortcut,
                             QSlider, QComboBox, QFormLayout)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPalette, QColor, QFont, QCursor, QBrush, QKeySequence
import numpy as np
import io
import re
from contextlib import redirect_stdout
def load_plotting():
    """Only statistics/report views need the plotting stack."""
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 120, 'font.size': 12,
                         'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False})
    return plt, FigureCanvasQTAgg

import uuid
import functools
try:
    import speech_recognition as sr
except ImportError:
    sr = None
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
import threading
import socket
import time

from configs.runtime_config import PROJECT_ROOT, load_app_config, save_app_config
from configs.network_config import get_ipv4_interfaces, read_system_config, save_network_config
from src.core.classification import EyeDiseaseDetector, ClassificationError, valid_prediction
from src.core.llm import DeepSeekAPI, MedicalAIService, AdviceResult
from src.core.history import HistoryDB, get_history_db
from src.visualization.workbench_theme import apply_workbench_theme, style_dialog
from src.visualization.workbench_layout import build_workbench, update_probability_panel, ImageCanvas
from src.visualization.workbench_jobs import WorkbenchJobs
APP_CONFIG = load_app_config()

# ===== 导入统一配置 =====
try:
    from configs.system_config import (
        PC_IP, NETWORK_PORTS, CAMERA_CONFIG, AUDIO_CONFIG, 
        SYSTEM_CONFIG, connection_manager, get_local_ip
    )
    print("✅ PC端使用统一配置文件")
    print(f"📡 本机IP: {get_local_ip()}")
except ImportError:
    print("⚠️ 未找到统一配置文件,使用默认配置")
    NETWORK_PORTS = {
        "CAMERA_PORT": 5002,
        "DIAGNOSIS_PORT": 5003,
        "COMMAND_PORT": 5004,
        "VOICE_SEND_PORT": 5005,
        "VOICE_RECEIVE_PORT": 5006,
    }
    connection_manager = None


# 简化的语音组件
class NetworkDetector:
    """网络状态检测器"""
    def __init__(self):
        self.is_online = True
    
    def get_network_status(self):
        """获取网络状态"""
        return self.is_online

class SmartVoiceManager(QObject):
    """简化的语音管理器（本地离线语音识别优先）"""

    # 语音相关信号
    voice_recognized = pyqtSignal(str)
    voice_error = pyqtSignal(str)
    voice_timeout = pyqtSignal()
    voice_unknown = pyqtSignal()

    # 网络与TTS相关信号（占位,便于主界面连接,不强依赖）
    network_status_changed = pyqtSignal(bool, str)
    tts_started = pyqtSignal()
    tts_finished = pyqtSignal()
    tts_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_recording = False
        self.local_recognizer = None
        self.tts_engine = None
        self.vosk_model = None
        self._vosk_json = None
        self.recognition_duration = 10  # 默认识别时长为10秒
        self.init_voice_components()
    
    def init_voice_components(self):
        """初始化本地识别器与离线模型（不创建嵌套管理器）"""
        try:
            # 本地识别器
            self.local_recognizer = sr.Recognizer()
            # 可选加载本地Vosk中文模型
            self._init_vosk()
            print("[DEBUG] 语音组件初始化完成")
        except Exception as e:
            print(f"[ERROR] 语音组件初始化失败: {e}")

    def _init_vosk(self):
        """延迟搜索Vosk模型路径,但不立即加载模型"""
        try:
            import vosk  # noqa: F401
            import json as _json
            
            # 只搜索模型路径,不立即加载模型
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.getenv("VOSK_MODEL_PATH", ""),
                os.path.join(script_dir, "vosk-model-small-cn-0.22"),
                os.path.join(script_dir, "vosk-model-cn-0.22"),
                os.path.join(os.getcwd(), "vosk-model-small-cn-0.22"),
                os.path.join(os.getcwd(), "vosk-model-cn-0.22"),
            ]
            
            candidates = [p for p in candidates if p and os.path.isdir(p)]
            
            if candidates:
                self.vosk_model_dir = candidates[0]
                self.vosk_model = None  # 不在这里加载模型
                self._vosk_json = _json
                print(f"[DEBUG] 找到Vosk模型目录: {self.vosk_model_dir} (将延迟加载)")
            else:
                self.vosk_model = None
                self.vosk_model_dir = None
                self._vosk_json = None
                print("[DEBUG] 未找到Vosk中文模型,将使用在线识别")
                
        except Exception as e:
            self.vosk_model = None
            self.vosk_model_dir = None
            self._vosk_json = None
            print(f"[DEBUG] Vosk不可用: {e}")
            
    def _load_vosk_model_async(self):
        """异步加载Vosk模型（只在需要时调用）"""
        if not self.vosk_model_dir or self.vosk_model is not None:
            return
            
        try:
            from vosk import Model
            print(f"[DEBUG] 开始异步加载Vosk模型: {self.vosk_model_dir}")
            self.vosk_model = Model(self.vosk_model_dir)
            print("[DEBUG] ✅ Vosk中文离线模型异步加载成功")
        except Exception as e:
            print(f"[DEBUG] ❌ Vosk模型异步加载失败: {e}")
            self.vosk_model = None
    
    def start_voice_recognition(self, duration=None):
        """开始语音识别
        
        Args:
            duration: 可选,指定识别时长（秒）,None表示使用默认值
        """
        if self.is_recording:
            # 如果正在录音,则取消录音
            self.cancel_recording()
            return
        
        # 如果指定了时长,则临时更新识别时长
        if duration is not None and duration > 0:
            self.recognition_duration = duration
            print(f"[DEBUG] 🕒 设置语音识别时长为 {duration} 秒")
        
        self.is_recording = True
        self._recording_thread = threading.Thread(target=self._perform_recognition, daemon=True)
        self._recording_thread.start()
        
    def cancel_recording(self):
        """取消录音"""
        if self.is_recording:
            print("[DEBUG] 🛑 用户取消录音")
            self.is_recording = False
            # 发送取消信号
            self.voice_error.emit("录音已取消")
    
    def start_network_monitoring(self):
        """启动网络状态监控（本地离线优先,这里仅做友好提示）"""
        # 本地离线识别不依赖网络,直接提示为离线可用
        try:
            # 立即发一次状态
            self.network_status_changed.emit(False, "本地离线语音识别已启用")
        except Exception:
            pass
    
    def _perform_recognition(self):
        """执行语音识别（改进的中文识别策略）"""
        try:
            print("[DEBUG] 🎤 开始录音...")
            
            # 发送开始录音信号
            self.voice_recognized.emit("__RECORDING_START__")
            
            # 录音
            with sr.Microphone() as source:
                print("[DEBUG] 🔧 调整环境噪音...")
                self.local_recognizer.adjust_for_ambient_noise(source, duration=0.8)
                print("[DEBUG] 🗣️ 请说话（点击按钮停止录音）...")
                
                # 发送录音中信号
                self.voice_recognized.emit("__RECORDING__")
                
                # 用户自定义录音时长,无时间限制
                audio = self.local_recognizer.listen(source, timeout=30, phrase_time_limit=self.recognition_duration)
                print(f"[DEBUG] ✅ 录音完成,音频长度: {len(audio.frame_data)} bytes")

            # 发送处理中信号
            self.voice_recognized.emit("__PROCESSING__")
            
            text = None
            recognition_method = ""

            # 优先策略：本地Vosk识别（如果可用）- 中文效果更好
            if self.vosk_model_dir and self.vosk_model is None:
                # 如果有模型路径但模型未加载,先尝试异步加载
                self._load_vosk_model_async()
                
            if self.vosk_model is not None:
                try:
                    print("[DEBUG] 🏠 使用本地Vosk中文识别...")
                    from vosk import KaldiRecognizer
                    rec = KaldiRecognizer(self.vosk_model, audio.sample_rate or 16000)
                    frame = audio.frame_data
                    chunk_size = 4000
                    for i in range(0, len(frame), chunk_size):
                        rec.AcceptWaveform(frame[i:i+chunk_size])
                    result_json = rec.FinalResult()
                    if self._vosk_json is not None:
                        result_obj = self._vosk_json.loads(result_json)
                        text = (result_obj.get("text") or "").strip()
                        recognition_method = "Vosk 本地识别"
                        print(f"[DEBUG] ✅ Vosk识别结果: '{text}'")
                except Exception as e:
                    print(f"[DEBUG] ❌ Vosk识别失败: {e}")

            # 备用策略：Google API（优化参数）
            if not text or len(text) < 2:
                try:
                    print("[DEBUG] 🌐 尝试Google API中文识别（优化参数）...")
                    # 尝试不同的Google识别参数
                    text = self.local_recognizer.recognize_google(audio, language='zh-CN', show_all=False)
                    recognition_method = "Google API (中文优化)"
                    print(f"[DEBUG] ✅ Google API识别成功: '{text}'")
                except sr.RequestError as e:
                    print(f"[DEBUG] ❌ Google API网络错误: {e}")
                except sr.UnknownValueError:
                    print("[DEBUG] ❌ Google API无法识别语音内容")
                    # 尝试英文识别
                    try:
                        print("[DEBUG] 🌐 尝试Google API英文识别...")
                        text = self.local_recognizer.recognize_google(audio, language='en-US')
                        recognition_method = "Google API (英文)"
                        print(f"[DEBUG] ✅ Google API英文识别: '{text}'")
                    except Exception as e2:
                        print(f"[DEBUG] ❌ Google API英文识别失败: {e2}")
                except Exception as e:
                    print(f"[DEBUG] ❌ Google API其他错误: {e}")

            # 结果处理
            if text and text.strip() and len(text.strip()) > 0:
                final_text = text.strip()
                print(f"[INFO] 🎉 语音识别成功 ({recognition_method}): '{final_text}'")
                self.voice_recognized.emit(final_text)
                return
            else:
                print("[DEBUG] ❌ 所有识别方法都未获得有效结果")
                self.voice_unknown.emit()
                
        except sr.WaitTimeoutError:
            print("[DEBUG] ⏱️ 录音超时")
            self.voice_timeout.emit()
        except sr.UnknownValueError:
            print("[DEBUG] ❓ 无法识别语音内容")
            self.voice_unknown.emit()
        except Exception as e:
            print(f"[DEBUG] ❌ 语音识别异常: {e}")
            self.voice_error.emit(f"语音识别错误: {str(e)}")
        finally:
            self.is_recording = False
    
    def test_microphone(self):
        """测试麦克风"""
        try:
            with sr.Microphone() as source:
                self.local_recognizer.adjust_for_ambient_noise(source, duration=1)
            return True, "麦克风测试成功"
        except Exception as e:
            return False, f"麦克风测试失败: {e}"
    
    def get_voice_status(self):
        """获取语音状态"""
        return {
            "is_recording": self.is_recording,
            "is_online": True,
            "has_microphone": True,
            "recognition_duration": self.recognition_duration
        }
        
    def set_recognition_duration(self, duration):
        """设置语音识别时长
        
        Args:
            duration: 识别时长（秒）,必须大于0
        """
        if duration > 0:
            self.recognition_duration = duration
            print(f"[DEBUG] 🕒 语音识别时长已更新为 {duration} 秒")
            return True
        return False

# 注释掉全局异常处理器,避免无限递归
# def global_exception_handler(exctype, value, traceback):
#     """全局异常处理器"""
#     if exctype != SystemExit:
#         print(f"未处理的异常: {exctype.__name__}: {value}")
# 
# # 设置全局异常处理器
# sys.excepthook = global_exception_handler

# =================  追加到 import 区域之后即可  =================

# 语音识别相关的事件类
class VoiceRecognitionEvent(QEvent):
    """语音识别事件"""
    def __init__(self, event_type, data=None):
        super().__init__(QEvent.User + 1)
        self.event_type = event_type
        self.data = data


class AIResponseEvent(QEvent):
    """AI回复事件"""
    def __init__(self, event_type, data=None, progress=0, tts=None):
        super().__init__(QEvent.User + 2)
        self.event_type = event_type
        self.data = data
        self.progress = progress
        self.tts = tts  # 简短摘要用于语音播报


class ResultProcessor:
    """检测结果处理工具类,负责解析、展示和格式化结果"""
    def __init__(self, detector: EyeDiseaseDetector):
        self.detector = detector  # 疾病检测器实例（包含映射关系）
        self.current_disease = None  # 当前检测到的疾病
        self.current_confidence = 0.0  # 当前检测的置信度

    def display_annotated_image(self, annotated_image, label: QLabel):
        """
        在QLabel上显示带标注的检测结果图像
        :param annotated_image: 带标注的图像(numpy数组)
        :param label: 要显示图像的QLabel控件
        """
        try:
            # 确保图像有效
            if annotated_image is None or not isinstance(annotated_image, np.ndarray):
                raise ValueError("无效的标注图像数据")
                
            # 转换颜色空间
            if len(annotated_image.shape) == 2:  # 灰度图
                image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB)
            else:  # 彩色图
                image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
            # 转换为QImage
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_img)
            if isinstance(label, ImageCanvas):
                label.setPixmap(pixmap)
                return
            label.setPixmap(
                pixmap.scaled(
                    label.width(), label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            
        except Exception as e:
            print(f"显示标注图像时出错: {e}")
            label.setText("无法显示检测结果")
            label.setAlignment(Qt.AlignCenter)

    def parse_prediction_output(self, prediction_output):
        # Human-readable console output is not a reliable classification interface.
        return False

    def parse_model_results(self, model_results):
        self.current_disease = None
        self.current_confidence = 0.0
        outcome = self.detector.parse_result(model_results)
        if not outcome.success:
            return False
        self.current_disease = outcome.disease_name
        self.current_confidence = outcome.confidence
        return True

    def get_fallback_result(self):
        self.current_disease = None
        self.current_confidence = 0.0
        raise ClassificationError("分类解析失败，不能生成默认疾病或置信度")

    def show_disease_result_dialog(self, parent, background_color, text_color, highlight_color):
        """
        在对话框中展示检测结果
        """
        if not valid_prediction(self.current_disease, self.current_confidence):
            QMessageBox.warning(parent, "没有有效结果", "分类解析失败，请重新加载图像并分类")
            return
        result_text = f"""
        <div style='text-align:center; padding:15px;'>
            <h2 style='color:{highlight_color}; margin-bottom:20px;'>疾病分类结果</h2>
            <p style='font-size:16pt; margin:15px 0;'>检测到的疾病: <b style='color:{highlight_color};'>{self.current_disease}</b></p>
            <p style='font-size:16pt; margin:15px 0;'>置信度: <b style='color:{highlight_color};'>{self.current_confidence:.2f}</b></p>
            <p style='margin-top:25px; color:#a0aec0; font-size:13pt;'>可点击「AI治疗建议」获取详细方案</p>
        </div>
        """
        msg_box = QMessageBox(parent)
        msg_box.setWindowTitle("分类结果")
        msg_box.setText(result_text)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStyleSheet(f"""
            QMessageBox {{
                background-color: {background_color};
                color: {text_color};
                min-width: 520px;
            }}
            QPushButton {{
                background-color: #00B5D8;
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #0097B2;
            }}
        """)
        msg_box.exec_()



# ============================================================
#  摄像头接收器
# ============================================================
class CommandListener(QObject):
    """开发板命令监听器"""
    command_received = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.socket = None
        self.is_listening = False
        self.listening_thread = None
        
    def start_listening(self, port=5004):
        """启动命令监听"""
        if self.is_listening:
            return
            
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(("0.0.0.0", port))
            self.socket.settimeout(1.0)
            
            self.is_listening = True
            self.listening_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listening_thread.start()
            
            print(f"[开发板] 命令监听器已启动,监听端口 {port}")
            
        except Exception as e:
            print(f"[开发板] 启动命令监听器失败: {e}")
    
    def stop_listening(self):
        """停止命令监听"""
        self.is_listening = False
        if self.socket:
            self.socket.close()
            self.socket = None
        
        if self.listening_thread:
            self.listening_thread.join(timeout=1.0)
            self.listening_thread = None
        
        print("[开发板] 命令监听器已停止")
    
    def _listen_loop(self):
        """监听循环"""
        while self.is_listening:
            try:
                data, addr = self.socket.recvfrom(4096)
                self._process_command(data, addr)
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[开发板] 命令监听错误: {e}")
                continue
    
    def _process_command(self, data, addr):
        """处理接收到的命令"""
        try:
            command_json = data.decode('utf-8')
            command_data = json.loads(command_json)
            
            # 添加源地址信息
            command_data['source_addr'] = addr
            
            # 发送命令信号
            self.command_received.emit(command_data)
            
        except Exception as e:
            print(f"[开发板] 命令解析错误: {e}")

class BoardCameraReceiver(QObject):
    """开发板摄像头数据接收器"""
    frame_received = pyqtSignal(np.ndarray)
    connection_status_changed = pyqtSignal(bool)
    diagnosis_request_received = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.socket = None
        self.is_receiving = False
        self.receiving_thread = None
        self.packet_buffer = {}  # 用于重组分片数据
        self.request_headers = {}  # 存储请求头信息
        self.packet_to_request = {}  # 存储packet_id到request_id的映射
        self.last_heartbeat = 0
        self.connection_active = False
        
    def start_receiving(self, port=5002):
        """启动数据接收"""
        if self.is_receiving:
            return
            
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(("0.0.0.0", port))
            self.socket.settimeout(1.0)
            
            self.is_receiving = True
            self.receiving_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receiving_thread.start()
            
            print(f"[开发板] 摄像头数据接收器已启动,监听端口 {port}")
            self.connection_status_changed.emit(True)
            
        except Exception as e:
            print(f"[开发板] 启动接收器失败: {e}")
            self.connection_status_changed.emit(False)
    
    def stop_receiving(self):
        """停止数据接收"""
        self.is_receiving = False
        if self.socket:
            self.socket.close()
            self.socket = None
        
        if self.receiving_thread:
            self.receiving_thread.join(timeout=1.0)
            self.receiving_thread = None
        
        print("[开发板] 摄像头数据接收器已停止")
        self.connection_status_changed.emit(False)
    
    def _receive_loop(self):
        """接收循环"""
        while self.is_receiving:
            try:
                data, addr = self.socket.recvfrom(4096)
                self._process_received_data(data, addr)
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[开发板] 接收数据错误: {e}")
                continue
    
    def _process_received_data(self, data, addr):
        """处理接收到的数据"""
        try:
            print(f"[调试] 收到数据包,总长度: {len(data)} 字节")
            print(f"[调试] 数据包前20字节: {data[:20].hex() if len(data) >= 20 else data.hex()}")
            
            # 检查是否是心跳包
            if len(data) >= 9:
                packet_id = int.from_bytes(data[0:4], 'big')
                packet_index = int.from_bytes(data[4:6], 'big')
                total_packets = int.from_bytes(data[6:8], 'big')
                
                if packet_id == 0 and packet_index == 0 and total_packets == 0:
                    self._handle_heartbeat(data, addr)
                    return
            
            # 检查是否是简化的图像保存数据包
            if len(data) >= 4:  # 至少包含包索引和总包数
                try:
                    # 尝试解析为简化格式：[2字节包索引][2字节总包数][图像数据]
                    packet_index = int.from_bytes(data[0:2], 'big')
                    total_packets = int.from_bytes(data[2:4], 'big')
                    packet_data = data[4:]
                    
                    print(f"[调试] 简化图像包 - 包索引: {packet_index}, 总包数: {total_packets}")
                    
                    # 生成一个临时的packet_id用于重组
                    temp_packet_id = hash(f"temp_save_{time.time()}") & 0xFFFFFFFF
                    
                    # 重组分片数据
                    if temp_packet_id not in self.packet_buffer:
                        self.packet_buffer[temp_packet_id] = [None] * total_packets
                    
                    self.packet_buffer[temp_packet_id][packet_index] = packet_data
                    
                    # 检查是否所有包都已接收
                    if all(packet is not None for packet in self.packet_buffer[temp_packet_id]):
                        self._process_saved_image(temp_packet_id, addr)
                    
                    return
                    
                except Exception as e:
                    print(f"[调试] 简化格式解析失败: {e}")
            
            # 原有的复杂格式处理（保留兼容性）
            if len(data) < 9:  # 最小包头长度
                print(f"[调试] 数据包太短,长度: {len(data)},需要至少9字节")
                return
            
            # 解析包头：[4字节请求ID哈希][2字节包索引][2字节总包数][1字节标志位][数据]
            packet_id = int.from_bytes(data[0:4], 'big')
            packet_index = int.from_bytes(data[4:6], 'big')
            total_packets = int.from_bytes(data[6:8], 'big')
            is_last = bool(data[8])
            
            print(f"[调试] 标准包头解析 - packet_id: {packet_id}, packet_index: {packet_index}, total_packets: {total_packets}, is_last: {is_last}")
            
            # 处理第一个包中的request_id信息
            if packet_index == 0:
                print(f"[调试] 处理第一个包,数据总长度: {len(data)}")
                
                if len(data) >= 11:
                    # 第一个包包含request_id信息：[包头][2字节长度][request_id][图像数据]
                    request_id_len = int.from_bytes(data[9:11], 'big')
                    print(f"[调试] request_id长度: {request_id_len}")
                    print(f"[调试] 需要的总长度: {11 + request_id_len}, 实际长度: {len(data)}")
                    
                    if len(data) >= 11 + request_id_len:
                        try:
                            request_id = data[11:11+request_id_len].decode('utf-8')
                            packet_data = data[11+request_id_len:]
                            
                            # 存储request_id到packet_id的映射
                            self.packet_to_request[packet_id] = request_id
                            print(f"[开发板] 收到图像包,packet_id: {packet_id}, request_id: {request_id}")
                            print(f"[开发板] 映射关系已存储,当前映射数量: {len(self.packet_to_request)}")
                            print(f"[开发板] 当前请求头数量: {len(self.request_headers)}")
                        except UnicodeDecodeError as e:
                            print(f"[调试] request_id解码失败: {e}")
                            print(f"[调试] 原始字节: {data[11:11+request_id_len].hex()}")
                            packet_data = data[9:]
                    else:
                        packet_data = data[9:]
                        print(f"[调试] 第一个包数据长度不足,无法提取request_id")
                        print(f"[调试] 需要: {11 + request_id_len}, 实际: {len(data)}")
                else:
                    packet_data = data[9:]
                    print(f"[调试] 第一个包长度不足11字节,无法读取request_id长度")
            else:
                packet_data = data[9:]
            
            # 重组分片数据
            if packet_id not in self.packet_buffer:
                self.packet_buffer[packet_id] = [None] * total_packets
            
            self.packet_buffer[packet_id][packet_index] = packet_data
            
            # 检查是否所有包都已接收
            if is_last and all(packet is not None for packet in self.packet_buffer[packet_id]):
                self._reconstruct_and_process_image(packet_id, addr)
                
        except Exception as e:
            print(f"[开发板] 数据处理错误: {e}")
    
    def _handle_heartbeat(self, data, addr):
        """处理心跳包"""
        try:
            # 解析心跳数据
            heartbeat_data = data[9:].decode('utf-8')
            heartbeat = json.loads(heartbeat_data)
            
            if heartbeat.get('type') == 'heartbeat':
                self.last_heartbeat = time.time()
                self.connection_active = True
                
                # 发送心跳响应
                response = {
                    "type": "heartbeat_response",
                    "timestamp": datetime.now().isoformat(),
                    "server_status": "running",
                    "latency": int((time.time() - datetime.fromisoformat(heartbeat['timestamp']).timestamp()) * 1000)
                }
                
                response_data = json.dumps(response).encode('utf-8')
                self.socket.sendto(response_data, addr)
                
                print(f"[开发板] 收到心跳包,延迟: {response['latency']}ms")
                
        except Exception as e:
            print(f"[开发板] 心跳处理错误: {e}")
    
    def _reconstruct_and_process_image(self, packet_id, addr):
        """重组并处理图像"""
        try:
            # 重组图像数据
            image_data = b''.join(self.packet_buffer[packet_id])
            
            # 解码图像
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is not None:
                print(f"[开发板] 图像重组成功,大小: {image.shape}")
                
                # 查找对应的request_id
                request_id = self.packet_to_request.get(packet_id)
                print(f"[开发板] 查找request_id映射,packet_id: {packet_id}, found: {request_id is not None}")
                print(f"[开发板] 当前映射关系: {list(self.packet_to_request.keys())}")
                print(f"[开发板] 当前请求头: {list(self.request_headers.keys())}")
                
                if request_id:
                    # 查找对应的请求头信息
                    request_header = self.request_headers.get(request_id)
                    print(f"[开发板] 查找请求头信息,request_id: {request_id}, found: {request_header is not None}")
                    
                    if request_header:
                        # 发送诊断请求信号
                        self.diagnosis_request_received.emit({
                            "image": image,
                            "header": request_header,
                            "source": "board_camera",
                            "addr": addr
                        })
                        
                        # 清理缓存
                        del self.packet_buffer[packet_id]
                        del self.request_headers[request_id]
                        del self.packet_to_request[packet_id]
                        print(f"[开发板] 图像处理完成,request_id: {request_id}")
                    else:
                        print(f"[开发板] 未找到请求头信息,request_id: {request_id}")
                        # 清理包缓存,但保留映射关系以便调试
                        if packet_id in self.packet_buffer:
                            del self.packet_buffer[packet_id]
                else:
                    print(f"[开发板] 未找到request_id映射,packet_id: {packet_id}")
                    # 清理包缓存
                    if packet_id in self.packet_buffer:
                        del self.packet_buffer[packet_id]
            else:
                print(f"[开发板] 图像解码失败")
                
        except Exception as e:
            print(f"[开发板] 图像重组处理错误: {e}")
    
    def _process_saved_image(self, packet_id, addr):
        """处理保存的图像（简化格式）"""
        try:
            # 重组图像数据
            image_data = b''.join(self.packet_buffer[packet_id])
            
            # 解码图像
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is not None:
                print(f"[保存] 图像重组成功,大小: {image.shape}")
                
                # 查找对应的保存请求头
                save_request_header = None
                save_request_id = None
                
                # 遍历所有请求头,找到保存请求
                for req_id, header in self.request_headers.items():
                    if header.get('type') == 'image_save_request':
                        save_request_header = header
                        save_request_id = req_id
                        break
                
                if save_request_header:
                    # 保存图像到PC端
                    pc_save_path = save_request_header.get('pc_save_path', '')
                    filename = save_request_header.get('filename', f'saved_image_{int(time.time())}.jpg')
                    
                    print(f"[保存] 找到保存请求,文件名: {filename}")
                    print(f"[保存] 保存路径: {pc_save_path}")
                    
                    if pc_save_path:
                        success = self._save_image_to_pc_direct(image, filename, pc_save_path)
                        
                        if success:
                            print(f"✅ [保存] 图像已保存到PC端: {filename}")
                            
                            # 发送保存成功响应到开发板
                            self._send_save_response(save_request_id, True, filename, addr)
                        else:
                            print(f"❌ [保存] 图像保存失败")
                            self._send_save_response(save_request_id, False, filename, addr)
                        
                        # 清理缓存
                        del self.packet_buffer[packet_id]
                        if save_request_id in self.request_headers:
                            del self.request_headers[save_request_id]
                    else:
                        print("[保存] 未找到PC端保存路径")
                        self._send_save_response(save_request_id, False, filename, addr)
                else:
                    print("[保存] 未找到保存请求头,使用默认路径保存")
                    
                    # 使用默认路径保存
                    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medical_images")
                    filename = f'auto_saved_{int(time.time())}.jpg'
                    
                    success = self._save_image_to_pc_direct(image, filename, default_path)
                    if success:
                        print(f"✅ [保存] 图像已自动保存: {filename}")
                    
                    # 清理缓存
                    del self.packet_buffer[packet_id]
                    
            else:
                print(f"[保存] 图像解码失败")
                
        except Exception as e:
            print(f"[保存] 图像保存处理错误: {e}")
    
    def _save_image_to_pc_direct(self, image, filename, pc_save_path):
        """直接保存图像到PC端"""
        try:
            import os
            
            # 确保目录存在
            os.makedirs(pc_save_path, exist_ok=True)
            
            # 保存图像
            image_path = os.path.join(pc_save_path, filename)
            success = cv2.imwrite(image_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success and os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                print(f"✅ [保存] 图像已保存到PC端:")
                print(f"   文件路径: {image_path}")
                print(f"   文件大小: {file_size} 字节")
                return True
            else:
                print(f"❌ [保存] 图像保存失败,文件未创建")
                return False
            
        except Exception as e:
            print(f"❌ [保存] 保存到PC端失败: {e}")
            return False
    
    def _send_save_response(self, request_id, success, filename, addr):
        """发送保存响应到开发板"""
        try:
            response = {
                "type": "save_response",
                "request_id": request_id,
                "success": success,
                "filename": filename,
                "timestamp": datetime.now().isoformat()
            }
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            response_json = json.dumps(response, ensure_ascii=False)
            response_bytes = response_json.encode('utf-8')
            
            # 发送到开发板诊断端口
            sock.sendto(response_bytes, (addr[0], 5003))
            sock.close()
            
            print(f"[响应] 保存结果已发送到开发板: {'成功' if success else '失败'}")
            
        except Exception as e:
            print(f"[响应] 发送保存响应失败: {e}")
    
    def store_request_header(self, request_id, header):
        """存储请求头信息"""
        self.request_headers[request_id] = header
        print(f"[开发板] 存储请求头,request_id: {request_id}")
    
    def get_connection_status(self):
        """获取连接状态"""
        current_time = time.time()
        return {
            "active": self.connection_active,
            "last_heartbeat": self.last_heartbeat,
            "time_since_heartbeat": current_time - self.last_heartbeat if self.last_heartbeat > 0 else float('inf'),
            "buffered_packets": len(self.packet_buffer),
            "pending_requests": len(self.request_headers)
        }

# ============================================================
#  主窗口
# ============================================================
class MainWindow(WorkbenchJobs, QMainWindow):
    report_ready = pyqtSignal(object)
    chat_reply = pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.report_ready.connect(self._apply_report_reply)
        from src.visualization.conversation import finish
        self.chat_reply.connect(lambda payload: finish(self,payload))
        
        # 设置窗口属性
        self.setWindowTitle("眼科图像分类辅助诊断系统")
        self.setGeometry(100, 100, 1400, 800)
        
        # 初始化摄像头接收器
        self.camera_receiver = BoardCameraReceiver()
        self.camera_receiver.frame_received.connect(self.update_camera_preview)
        self.camera_receiver.connection_status_changed.connect(self.update_camera_status)
        
        # 自动启动命令监听器（关键修复）
        self.command_listener = CommandListener()
        self.command_listener.command_received.connect(self.handle_board_command)
        if APP_CONFIG.enable_board:
            self.command_listener.start_listening(NETWORK_PORTS["COMMAND_PORT"])
        else:
            print("PC模式：开发板通信未启用")
        
        self.setMinimumSize(1100, 750)

        # 启动时自动全屏显示
        self.showMaximized()
        
        # 医疗风颜色主题 - 临床暗黑风格
        self.background_color = "#171C24"    # 更深沉的背景
        self.primary_color = "#242C37"       # 主面板色
        self.secondary_bg = "#242C37"        # 次级面板色
        self.accent_color = "#51C5E3"        # 医疗青色 (主色调)
        self.highlight_color = "#51C5E3"     # 沉稳紫 (替代刺眼的粉色)
        self.text_color = "#E5E9F0"          # 柔和的灰白文字
        self.success_color = "#38A169"       # 成功/连接状态绿
        self.danger_color = "#E53E3E"        # 警告/断开状态红

        # 设置全局样式
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {self.background_color};
                color: {self.text_color};
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            }}
            QLabel {{
                color: {self.text_color};
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            }}
            QGroupBox {{
                font-weight: bold;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            }}
            QTextEdit {{
                background-color: {self.secondary_bg};
                color: {self.text_color};
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 13pt;
                selection-background-color: {self.accent_color};
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            }}
            QScrollBar:vertical {{
                background-color: {self.secondary_bg};
                width: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {self.accent_color};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QTabWidget::pane {{
                border: 1px solid {self.accent_color};
                border-radius: 5px;
                top: -1px;
            }}
            QTabBar::tab {{
                background-color: {self.secondary_bg};
                color: {self.text_color};
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 12px;
                margin-right: 2px;
                font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            }}
            QTabBar::tab:selected {{
                background-color: {self.accent_color};
            }}
            QTabBar::tab:hover {{
                background-color: #5a6478;
            }}
        """)

        # 延迟初始化组件（提高启动速度）
        self.detector = None
        self.result_processor = None
        self.deepseek_api = None
        self.voice_manager = None
        
        # 状态变量
        self.current_image_path = None
        self.current_image = None
        self.detection_results = None
        self.history_records = []
        
        # 初始化UI (只调用一次)
        self.init_ui()
        self.init_status_bar()
        build_workbench(self, APP_CONFIG)
        apply_workbench_theme(self)
        from src.visualization.preferences import restore
        restore(self)
        self.image_button.setEnabled(True)
        self.menuBar().hide()
        
        # AI对话历史上下文
        self.chat_history = []
        
        # 延迟加载保存的API密钥
        QTimer.singleShot(100, self.load_saved_api_key)
        
        # 延迟初始化重量级组件,提高启动速度
        QTimer.singleShot(300, self.lazy_load_components)
        QTimer.singleShot(600, self._load_configured_model)
        
        # 显示加载状态
        self.status_bar.showMessage("正在初始化系统组件...")
        
        # 异步加载历史记录
        # History is queried on demand when the user opens it.

        # 启动时清理过期临时图像 (7天前)
        # Keep image files referenced by history; no automatic deletion at startup.

        # 设置键盘快捷键,提升用户体验
        self.setup_shortcuts()


    def _clear_prediction_state(self):
        if hasattr(self,"chat_display"):
            from src.visualization.conversation import reset
            reset(self)
        self._report_generation = getattr(self, "_report_generation", 0) + 1
        self.current_results = None
        self.current_disease = None
        self.current_confidence = 0.0
        if hasattr(self, "classification_summary_label"):
            self.classification_summary_label.setText("尚未分类")
            self.classification_score_label.setText("—")
            if hasattr(self, "probability_rows"):
                update_probability_panel(self)
                self.inference_time_label.setText("等待分析")
        self.prediction_output = ""
        if hasattr(self, "advice_text"):
            self.advice_text.setPlainText("等待有效分类结果；尚未生成当前图像的辅助建议")
        self.all_classes_confidence = {}
        if self.result_processor is not None:
            self.result_processor.current_disease = None
            self.result_processor.current_confidence = 0.0
        for name in ("results_button", "advice_button"):
            button = getattr(self, name, None)
            if button is not None:
                button.setEnabled(False)
        if hasattr(self, "detected_image_label"):
            self.detected_image_label.clear()
            self.detected_image_label.setText("等待有效分类结果")

    def _update_classification_summary(self):
        names = {"AMD": "年龄相关性黄斑变性", "Cataract": "白内障",
                 "Diabetic Retinopathy": "糖尿病视网膜病变", "Glaucoma": "青光眼",
                 "Hypertensive Retinopathy": "高血压性视网膜病变",
                 "Myopia": "近视性黄斑病变", "Normal": "正常", "Other": "其他"}
        name = self.current_disease
        self.classification_summary_label.setText(names.get(name, name or "尚未分类"))
        self.classification_score_label.setText(f"{self.current_confidence:.1%}")

    def _classification_failed(self, message):
        self._clear_prediction_state()
        self.status_bar.showMessage("分类失败：" + str(message))
        if hasattr(self,"task_status_label"): self.task_status_label.setText("分类失败，请检查图像或模型")
        self.show_message_box("分类失败", str(message), QMessageBox.Warning)

    def _load_configured_model(self):
        for name in ("voice_input_button", "board_voice_button", "board_interaction_button", "connect_camera_button"):
            button = getattr(self, name, None)
            is_voice = name == "voice_input_button"
            enabled = APP_CONFIG.enable_voice if is_voice else APP_CONFIG.enable_board
            if button is not None and not enabled:
                button.setEnabled(False)
                button.setToolTip("可选功能未启用；请修改configs/app_config.json")
        if not APP_CONFIG.auto_load_model:
            return
        if not APP_CONFIG.model_path.is_file():
            self.status_bar.showMessage(f"默认权重不存在：{APP_CONFIG.model_path}；可手动选择")
            return
        self.load_model(str(APP_CONFIG.model_path))


    def create_runtime_settings_dialog(self, embedded=False):
        dialog = QWidget(self) if embedded else QDialog(self)
        dialog.setWindowTitle("运行与网络设置")
        dialog.resize(780, 500)
        dialog.setStyleSheet("""
            QDialog { background-color: #1e222a; }
            QLabel, QCheckBox { color: #e5e9f0; }
            QLineEdit, QComboBox { background-color: #292e38; color: #e5e9f0;
                border: 1px solid #4c566a; border-radius: 4px; padding: 5px; }
            QLineEdit:disabled { color: #a8b3c7; }
            QPushButton { background-color: #394355; color: white; padding: 7px 14px;
                border: 1px solid #5d6b82; border-radius: 4px; }
        """)
        layout = QVBoxLayout(dialog)
        advanced_content = QWidget()
        advanced_form = QFormLayout(advanced_content)
        form = QFormLayout()
        model_input = QLineEdit(str(APP_CONFIG.model_path))
        framework_input = QLineEdit(str(APP_CONFIG.framework_path))
        model_input.setToolTip(str(APP_CONFIG.model_path))
        framework_input.setToolTip(str(APP_CONFIG.framework_path))
        device_input = QComboBox()
        device_input.addItems(["auto", "cpu", "0"])
        device_input.setCurrentText(APP_CONFIG.device)
        advanced_form.addRow("默认8类分类权重", model_input)
        advanced_form.addRow("Ultralytics源码目录", framework_input)
        advanced_form.addRow("推理设备（0为独立显卡）", device_input)
        voice = QCheckBox("启用可选语音功能（需要语音依赖与麦克风）")
        board = QCheckBox("启用可选开发板通信")
        voice.setChecked(APP_CONFIG.enable_voice)
        board.setChecked(APP_CONFIG.enable_board)
        advanced_form.addRow(voice)
        advanced_form.addRow(board)
        network = read_system_config()
        board_address = QLineEdit(network.get('network', {}).get('board_ip', '172.20.10.8'))
        board_address.setObjectName("board_ip")
        auto = QCheckBox("自动识别活动局域网网卡，排除代理与虚拟网卡")
        auto.setObjectName("network_auto")
        auto.setChecked(network.get('startup', {}).get('auto_detect_ip', True))
        interfaces = QComboBox()
        interfaces.setObjectName("network_interfaces")
        interfaces.addItem("自动选择", "")
        for item in get_ipv4_interfaces():
            interfaces.addItem(item['interface'] + "  " + item['ip'], item['interface'])
        preferred = network.get('network', {}).get('preferred_interface', '')
        found = interfaces.findData(preferred)
        if found >= 0:
            interfaces.setCurrentIndex(found)
        address = QLineEdit(get_local_ip())
        address.setObjectName("network_ip")
        address.setEnabled(not auto.isChecked())
        auto.toggled.connect(lambda checked: address.setEnabled(not checked))
        detect = QPushButton("重新识别")
        def detect_address():
            current = get_ipv4_interfaces()
            preferred_name = interfaces.currentData()
            chosen = next((item for item in current if item['interface'] == preferred_name), current[0] if current else None)
            address.setText(chosen['ip'] if chosen else '127.0.0.1')
        detect.clicked.connect(detect_address)
        interfaces.currentIndexChanged.connect(lambda _: detect_address() if auto.isChecked() else None)
        auto.toggled.connect(lambda checked: detect_address() if checked else None)
        form.addRow(auto)
        form.addRow("优先局域网网卡", interfaces)
        form.addRow("PC局域网IPv4地址", address)
        form.addRow(detect)
        form.addRow("开发板 IPv4 地址", board_address)
        port_info = QLabel("摄像头 {camera_port} · 分类 {diagnosis_port} · 命令 {command_port}".format(**{
            key: network.get('network', {}).get(key, default)
            for key, default in [('camera_port',5002),('diagnosis_port',5003),('command_port',5004)]}))
        port_info.setWordWrap(True)
        form.addRow("当前通信端口", port_info)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.addLayout(form)
        advanced = QPushButton("展开模型与语音 / 开发板开关")
        advanced.setCheckable(True)
        advanced.toggled.connect(advanced_content.setVisible)
        advanced.toggled.connect(lambda checked: advanced.setText("收起高级配置" if checked else "展开模型与语音 / 开发板开关"))
        advanced_content.setVisible(False)
        advanced_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.addWidget(advanced)
        layout.addWidget(advanced_content)
        note = QLabel("这些设置保存到项目配置文件，重启后生效。仅在PC上分类不需要开发板或固定IP。\n环境变量EYE_MODEL_PATH等优先于文件配置。")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QHBoxLayout()
        save = QPushButton("保存设置")
        cancel = QPushButton("取消")
        def save_settings():
            try:
                import ipaddress
                from pathlib import Path
                if auto.isChecked():
                    detect_address()
                ipaddress.IPv4Address(address.text().strip())
                ipaddress.IPv4Address(board_address.text().strip())
                weight = Path(model_input.text()).expanduser()
                source = Path(framework_input.text()).expanduser()
                if not weight.is_file() or weight.suffix.lower() != '.pt':
                    raise ValueError("请选择存在的.pt分类权重")
                if not (source / 'ultralytics' / '__init__.py').is_file():
                    raise ValueError("源码目录必须包含ultralytics包")
                save_app_config(dict(model_path=str(weight.resolve()),framework_path=str(source.resolve()),
                                     device=device_input.currentText(),enable_voice=voice.isChecked(),enable_board=board.isChecked()))
                save_network_config(address.text().strip(), auto.isChecked(), interfaces.currentData(), board_ip=board_address.text().strip())
                self.status_bar.showMessage("运行与网络设置已保存；重启后生效")
                note.setText("设置已保存。请重启应用，使模型、语音和网络服务统一采用新配置。")
                if not embedded:
                    QMessageBox.information(dialog, "已保存", "请重启应用加载新设置。")
                    dialog.accept()
            except (ValueError, OSError, TypeError) as error:
                QMessageBox.warning(dialog, "设置无效", str(error))
        save.clicked.connect(save_settings)
        if not embedded:
            cancel.clicked.connect(dialog.reject)
        buttons.addStretch()
        buttons.addWidget(save)
        if not embedded:
            buttons.addWidget(cancel)
        layout.addLayout(buttons)
        return dialog

    def ensure_connection_settings(self, index):
        if index != 2 or getattr(self, "connection_settings", None) is not None:
            return
        group = QGroupBox("运行与局域网连接")
        layout = QVBoxLayout(group)
        self.connection_settings = self.create_runtime_settings_dialog(embedded=True)
        layout.addWidget(self.connection_settings)
        self.connection_page.layout().insertWidget(0, group)
        apply_workbench_theme(self)

    def show_runtime_settings(self):
        from PyQt5.QtWidgets import QSizePolicy
        self.ensure_connection_settings(2)
        dialog = QDialog(self)
        dialog.setWindowTitle("系统设置")
        dialog.setSizeGripEnabled(True)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18,16,18,14)
        layout.setSpacing(12)
        tabs=QTabWidget()
        appearance=QWidget();appearance_layout=QVBoxLayout(appearance)
        appearance_layout.setContentsMargins(20,20,20,20)
        appearance_note=QLabel('调整界面字体与控件字号，立即生效并自动保存。正文阅读字号在阅读区单独调整。')
        appearance_note.setWordWrap(True);appearance_layout.addWidget(appearance_note)
        form=QFormLayout();appearance_layout.addLayout(form)
        for title,source in [('界面字体',self.font_family_combo),('界面字号',self.font_size_combo)]:
            choice=QComboBox()
            for i in range(source.count()):choice.addItem(source.itemText(i),source.itemData(i))
            choice.setCurrentIndex(source.currentIndex())
            def changed(index,source=source):
                source.setCurrentIndex(index)
                style_dialog(dialog,self)
            choice.currentIndexChanged.connect(changed)
            form.addRow(title,choice)
        from src.visualization.image_directory import read as read_image_preferences,save as save_image_preferences,initial as image_initial
        directory_row=QHBoxLayout()
        directory_label=QLabel(read_image_preferences(self).get('image_directory') or '自动记住上次选图目录')
        directory_label.setWordWrap(True);directory_row.addWidget(directory_label,1)
        choose_directory=QPushButton('选择目录');directory_row.addWidget(choose_directory)
        reset_directory=QPushButton('恢复自动');directory_row.addWidget(reset_directory)
        def update_directory(value):
            try:
                save_image_preferences(self,image_directory=value)
                directory_label.setText(value or '自动记住上次选图目录')
                self.status_bar.showMessage('图像目录已保存，下次加载图像时生效')
            except OSError as error:QMessageBox.warning(dialog,'保存失败',str(error))
        def choose_image_directory():
            folder=QFileDialog.getExistingDirectory(dialog,'选择固定图像目录',image_initial(self))
            if folder:update_directory(folder)
        choose_directory.clicked.connect(choose_image_directory)
        reset_directory.clicked.connect(lambda:update_directory(''))
        form.addRow('固定图像目录',directory_row)
        restore_defaults=QPushButton("恢复界面默认设置")
        def reset_ui():
            from src.visualization.preferences import reset
            reset(self)
            dialog.accept()
        restore_defaults.clicked.connect(reset_ui)
        appearance_layout.addWidget(restore_defaults)
        appearance_layout.addStretch()
        tabs.addTab(appearance,'界面显示')
        groups=self.connection_page.findChildren(QGroupBox,options=Qt.FindDirectChildrenOnly)
        scrolls=[]
        for group in groups:
            title=group.title()
            content=QWidget();body=QVBoxLayout(content)
            body.setContentsMargins(12,14,12,12);body.setSpacing(14)
            descriptions={
                'DeepSeek 引擎配置':'配置云端辅助问答。保存密钥后即时生效，连接测试不会修改配置。',
                '外设测试':'测试本机麦克风并调整识别时长。语音功能开关位于运行与网络设置中。',
                '运行与局域网连接':'PC 单机分类无需开发板。修改运行与网络配置后，请保存并重启系统。'}
            note=QLabel(descriptions.get(title,''));note.setWordWrap(True);note.setObjectName('muted');body.addWidget(note)
            group.setTitle('')
            group.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Maximum)
            group.layout().setSpacing(16)
            group.layout().setAlignment(Qt.AlignTop)
            body.addWidget(group);body.addStretch(1)
            scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(content)
            tabs.addTab(scroll,{'DeepSeek 引擎配置':'云端 AI','外设测试':'语音与外设','运行与局域网连接':'运行与网络'}.get(title,title))
            scrolls.append((scroll,group,title))
        layout.addWidget(tabs,1)
        footer=QHBoxLayout()
        hint=QLabel('密钥与运行配置需分别保存；语音时长调整即时生效。');hint.setWordWrap(True);hint.setObjectName('muted')
        footer.addWidget(hint,1)
        close=QPushButton('完成 / 关闭');close.clicked.connect(dialog.accept);footer.addWidget(close)
        layout.addLayout(footer)
        style_dialog(dialog,self)
        # Content stays top aligned; the scroll area absorbs smaller screens and larger fonts.
        screen=self.screen().availableGeometry()
        size=self.font_size_combo.currentData() or 14
        dialog.resize(min(round(860*size/14),screen.width()-60),min(round(560*size/14),screen.height()-70))
        dialog.move(screen.center()-dialog.rect().center())
        try:
            dialog.exec_()
        finally:
            self.api_key_input.setEchoMode(QLineEdit.Password)
            self.toggle_password_button.setText('显示')
            for scroll,group,title in scrolls:
                group.setTitle(title)
                self.connection_page.layout().insertWidget(self.connection_page.layout().count()-1,group)
            self.connection_page.setParent(self._optional_controls)
            self.connection_page.hide()
            dialog.deleteLater()

    def setup_shortcuts(self):
        """设置键盘快捷键"""
        try:
            # Ctrl+O 打开图像
            open_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
            open_shortcut.activated.connect(self.load_image)
            
            # Ctrl+D 开始检测
            detect_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
            detect_shortcut.activated.connect(self.detect_button.click)
            
            # F11 切换全屏
            fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
            fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
            
            # Space 语音输入
            voice_shortcut = QShortcut(QKeySequence("Space"), self)
            voice_shortcut.activated.connect(self.toggle_voice_input)
            
            print("[DEBUG] 键盘快捷键设置完成")
        except Exception as e:
            print(f"[WARNING] 键盘快捷键设置失败: {e}")

    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def toggle_batch_report_fullscreen(self, dialog):
        """切换批量检测报告的全屏模式"""
        if dialog.isMaximized():
            dialog.showNormal()
        else:
            dialog.showMaximized()

    def toggle_voice_input(self):
        if not APP_CONFIG.enable_voice or sr is None:
            self.status_bar.showMessage("语音未启用；安装可选语音依赖并修改配置后可用")
            return
        """切换语音输入状态"""
        if hasattr(self, 'voice_input_button') and self.voice_input_button.isEnabled():
            self.start_voice_input()

    def lazy_load_components(self):
        """延迟加载重量级组件,提高启动响应速度"""
        try:
            # 初始化DeepSeek API组件
            if self.deepseek_api is None:
                self.deepseek_api = DeepSeekAPI()
                print("[DEBUG] DeepSeek API已初始化")
            
            # 初始化语音组件（现在不会阻塞,因为Vosk模型延迟加载）
            if APP_CONFIG.enable_voice and sr is not None and self.voice_manager is None:
                self.voice_manager = SmartVoiceManager()
                self.connect_smart_voice_signals()
                
                # 设置初始识别时长（如果UI已初始化）
                if hasattr(self, 'duration_slider'):
                    initial_duration = self.duration_slider.value()
                    self.voice_manager.set_recognition_duration(initial_duration)
                    print(f"[DEBUG] 🕒 设置初始识别时长为 {initial_duration} 秒")
                
                print("[DEBUG] 语音管理器已初始化")
            
            # 后台异步初始化其他语音组件（TTS等）
            if APP_CONFIG.enable_voice and sr is not None and pyttsx3 is not None:
                threading.Thread(target=self.init_speech_components_async, daemon=True).start()
            
            # 更新状态
            self.status_bar.showMessage("PC组件就绪；语音和开发板按配置启用")
            
        except Exception as e:
            print(f"[ERROR] 延迟加载组件失败: {e}")
            self.status_bar.showMessage("部分组件加载失败,基本功能可用")

    def init_speech_components_async(self):
        """异步初始化语音组件"""
        try:
            print("[DEBUG] 后台异步初始化语音组件...")
            
            # 初始化传统语音组件作为备用
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()  # 快速初始化,不做耗时测试
            
            # 初始化TTS引擎
            try:
                self.tts_engine = pyttsx3.init()
                if self.tts_engine:
                    self.tts_engine.setProperty('rate', 180)
                    self.tts_engine.setProperty('volume', 0.8)
                print("[DEBUG] TTS引擎初始化成功")
            except:
                self.tts_engine = None
                print("[WARNING] TTS引擎初始化失败,语音播放功能不可用")
            
            print("[DEBUG] 语音组件异步初始化完成")
            
        except Exception as e:
            print(f"[WARNING] 语音组件异步初始化失败: {e}")

    def init_ui(self):
        # 主布局 - 使用QSplitter实现可调整的两部分布局
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)
        main_splitter.setStretchFactor(0, 6)  # 左侧视觉区占比 6
        main_splitter.setStretchFactor(1, 4)  # 右侧分析区占比 4
        main_splitter.setChildrenCollapsible(False)  # 防止一侧完全折叠
        main_splitter.setHandleWidth(3)  # 拖拽手柄宽度
        main_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: #3b4252;
            }}
            QSplitter::handle:hover {{
                background-color: {self.accent_color};
            }}
        """)

        # ========================================================
        # 左侧容器 - 视觉与操作区
        # ========================================================
        left_widget = QWidget()
        left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(15, 15, 10, 15)

        # 标题
        title_label = QLabel("AI 眼科疾病智诊系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setStyleSheet(f"color: {self.accent_color}; letter-spacing: 2px; margin-bottom: 6px;")
        left_layout.addWidget(title_label)

        # 图像展示区 (采用 QTabWidget 节省空间)
        self.image_tab_widget = QTabWidget()
        self.image_tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_tab_widget.tabBar().setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.image_tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid #3b4252; border-radius: 8px; background-color: {self.secondary_bg}; }}
            QTabBar::tab {{ background-color: {self.primary_color}; color: #81A1C1; padding: 10px 20px; border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: bold; font-size: 12pt; margin-right: 2px; min-height: 34px; }}
            QTabBar::tab:selected {{ background-color: {self.accent_color}; color: white; }}
        """)
        self.image_tab_widget.tabBar().setElideMode(Qt.ElideNone)
        self.image_tab_widget.tabBar().setUsesScrollButtons(True)  # 标签过多时可滚动

        # 标签页 A：本地图像分析
        local_tab = QWidget()
        local_layout = QHBoxLayout(local_tab)
        local_layout.setSpacing(15)

        # 原始图像卡片
        self.original_image_label = QLabel("等待加载图像...")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(250, 250)
        self.original_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_image_label.setWordWrap(True)
        self.original_image_label.setScaledContents(False)
        self.original_image_label.setStyleSheet(f"background-color: #1a1e24; border-radius: 8px; border: 2px dashed #3b4252;")
        self.original_image_label.setText(
            f"<div style='text-align: center; color: #616E88;'>"
            f"<div style='font-size: 32pt; margin-bottom: 15px;'>📸</div>"
            f"<div style='font-size: 15pt; font-weight: bold; color: #E5E9F0;'>等待加载原片影像</div>"
            f"<div style='font-size: 13pt; margin-top: 8px;'>请点击下方「加载图像」按钮导入照片</div>"
            f"</div>"
        )
        local_layout.addWidget(self.original_image_label)

        # 检测结果卡片
        self.detected_image_label = QLabel()
        self.detected_image_label.setAlignment(Qt.AlignCenter)
        self.detected_image_label.setMinimumSize(250, 250)
        self.detected_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.detected_image_label.setWordWrap(True)
        self.detected_image_label.setScaledContents(False)
        self.detected_image_label.setStyleSheet(f"background-color: #1a1e24; border-radius: 8px; border: 2px dashed #3b4252;")
        self.detected_image_label.setText(
            f"<div style='text-align: center; color: #616E88;'>"
            f"<div style='font-size: 32pt; margin-bottom: 15px;'>🧠</div>"
            f"<div style='font-size: 15pt; font-weight: bold; color: #E5E9F0;'>等待 AI 模型诊断</div>"
            f"<div style='font-size: 13pt; margin-top: 8px;'>图像就绪后，点击「开始检测」进行特征分析</div>"
            f"</div>"
        )
        local_layout.addWidget(self.detected_image_label)
        self.image_tab_widget.addTab(local_tab, "🖼️ 本地图像分析")

        # 标签页 B：开发板流媒体
        board_tab = QWidget()
        board_layout = QVBoxLayout(board_tab)
        board_layout.setSpacing(12)

        # 摄像头预览区 — 不再限制最大尺寸，自适应空间
        self.camera_preview_label = QLabel("📱 摄像头未连接\n\n点击下方「连接开发板」开始实时预览")
        self.camera_preview_label.setAlignment(Qt.AlignCenter)
        self.camera_preview_label.setMinimumSize(200, 150)
        self.camera_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_preview_label.setWordWrap(True)
        self.camera_preview_label.setScaledContents(False)
        self.camera_preview_label.setStyleSheet(f"""
            background-color: #1a1e24;
            border-radius: 8px;
            border: 2px dashed #3b4252;
            color: #616E88;
            font-size: 13pt;
            font-family: 'Microsoft YaHei', sans-serif;
        """)
        board_layout.addWidget(self.camera_preview_label, stretch=1)

        # 状态信息栏
        info_layout = QHBoxLayout()
        info_layout.setSpacing(15)

        self.board_camera_status = QLabel("🔴 未连接")
        self.board_camera_status.setStyleSheet(f"""
            font-weight: bold;
            color: #E53E3E;
            font-size: 13pt;
            padding: 6px 12px;
            background-color: {self.secondary_bg};
            border-radius: 4px;
        """)

        self.board_resolution_label = QLabel("分辨率: --")
        self.board_resolution_label.setStyleSheet(f"color: #81A1C1; font-size: 11pt;")

        self.board_fps_label = QLabel("帧率: --")
        self.board_fps_label.setStyleSheet(f"color: #81A1C1; font-size: 11pt;")

        info_layout.addWidget(self.board_camera_status)
        info_layout.addWidget(self.board_resolution_label)
        info_layout.addWidget(self.board_fps_label)
        info_layout.addStretch()
        board_layout.addLayout(info_layout)

        # 控制按钮栏
        cam_ctrl_layout = QHBoxLayout()
        self.connect_camera_button = QPushButton("🔗 连接开发板")
        self.capture_from_camera_button = QPushButton("📸 截取并诊断")
        self.capture_from_camera_button.setEnabled(False)

        cam_btn_style = f"QPushButton {{ background-color: {self.success_color}; color: white; padding: 12px 24px; border-radius: 6px; font-weight: bold; font-size: 13pt; }} QPushButton:hover {{ background-color: #2F855A; }} QPushButton:disabled {{ background-color: #4A5568; color: #A0AEC0; }}"
        for btn in [self.connect_camera_button, self.capture_from_camera_button]:
            btn.setStyleSheet(cam_btn_style)
            btn.setCursor(QCursor(Qt.PointingHandCursor))

        self.connect_camera_button.clicked.connect(self.toggle_camera_connection)
        self.capture_from_camera_button.clicked.connect(self.capture_from_board_camera)

        cam_ctrl_layout.addStretch()
        cam_ctrl_layout.addWidget(self.connect_camera_button)
        cam_ctrl_layout.addWidget(self.capture_from_camera_button)
        cam_ctrl_layout.addStretch()
        board_layout.addLayout(cam_ctrl_layout)

        # 提示标签
        hint_label = QLabel("💡 提示：请确保 PC 与开发板在同一局域网，并已启动开发板端摄像头服务")
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setStyleSheet(f"color: #616E88; font-size: 10pt; padding: 4px;")
        hint_label.setWordWrap(True)
        board_layout.addWidget(hint_label)

        self.image_tab_widget.addTab(board_tab, "📱 硬件视窗")

        left_layout.addWidget(self.image_tab_widget, stretch=1)  # 图像区占据弹性空间

        # --- 2. 按钮控制面板 (逻辑分组) ---
        btn_panel = QWidget()
        btn_panel.setStyleSheet(f"background-color: {self.secondary_bg}; border-radius: 8px; border-top: 3px solid {self.accent_color};")
        btn_layout_v = QVBoxLayout(btn_panel)
        btn_layout_v.setContentsMargins(15, 15, 15, 15)
        btn_layout_v.setSpacing(15)

        # 核心工作流按钮
        main_flow_layout = QHBoxLayout()
        main_flow_layout.setSpacing(10)
        self.model_button = QPushButton("1. 🔁 加载模型")
        self.image_button = QPushButton("2. 🖼️ 加载图像")
        self.detect_button = QPushButton("3. 🔍 开始检测")
        self.results_button = QPushButton("4. 📊 查看报告")

        main_btn_style = f"QPushButton {{ background-color: {self.accent_color}; color: white; padding: 14px 20px; border-radius: 6px; font-weight: bold; font-size: 13pt; }} QPushButton:hover {{ background-color: #0097B2; }} QPushButton:disabled {{ background-color: #3b4252; color: #7b88a1; }}"
        for btn in [self.model_button, self.image_button, self.detect_button, self.results_button]:
            btn.setStyleSheet(main_btn_style)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.setMinimumHeight(55)
            main_flow_layout.addWidget(btn)

        self.model_button.clicked.connect(lambda: self.load_model(None))
        self.image_button.clicked.connect(self.load_image)
        self.image_button.setEnabled(False)
        self.detect_button.clicked.connect(self.detect_image)
        self.detect_button.setEnabled(False)
        self.results_button.clicked.connect(self.show_results)
        self.results_button.setEnabled(False)

        # 扩展工具按钮
        tools_layout = QHBoxLayout()
        tools_layout.setSpacing(10)
        self.batch_button = QPushButton("📁 批量处理")
        self.history_button = QPushButton("📜 历史记录")
        self.board_interaction_button = QPushButton("📱 开发板交互")
        self.board_voice_button = QPushButton("📱 唤醒板端语音")

        tool_style = f"""
            QPushButton {{
                background-color: #3B4252;
                border: 1px solid #4C566A;
                color: {self.text_color};
                padding: 12px 18px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13pt;
            }}
            QPushButton:hover {{
                background-color: #4C566A;
                border: 1px solid {self.accent_color};
            }}
            QPushButton:disabled {{
                background-color: {self.primary_color};
                color: #616E88;
                border: 1px solid #3b4252;
            }}
        """
        TOOL_BTN_STYLE = tool_style  # 保存一份引用供 toggle_voice_server 恢复样式
        for btn in [self.batch_button, self.history_button, self.board_interaction_button, self.board_voice_button]:
            btn.setStyleSheet(tool_style)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            tools_layout.addWidget(btn)

        self.batch_button.clicked.connect(self.batch_process)
        self.batch_button.setEnabled(False)
        self.history_button.clicked.connect(self.show_history)
        self.board_interaction_button.clicked.connect(self.show_board_interaction)
        self.board_voice_button.clicked.connect(self.trigger_board_voice)

        btn_layout_v.addLayout(main_flow_layout)
        btn_layout_v.addLayout(tools_layout)
        left_layout.addWidget(btn_panel, stretch=0)  # 保持按钮区原始高度

        # 右侧容器 - AI 分析与设置区 (同样采用 Tab 化)
        # ========================================================
        right_widget = QWidget()
        right_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 15, 15, 15)

        self.right_tab_widget = QTabWidget()
        self.right_tab_widget.tabBar().setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.right_tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid #3b4252; border-radius: 8px; background-color: {self.secondary_bg}; }}
            QTabBar::tab {{ background-color: {self.primary_color}; color: #81A1C1; padding: 10px 20px; font-weight: bold; font-size: 12pt; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; min-height: 34px; }}
            QTabBar::tab:selected {{ background-color: {self.highlight_color}; color: white; }}
        """)
        self.right_tab_widget.tabBar().setElideMode(Qt.ElideNone)
        self.right_tab_widget.tabBar().setUsesScrollButtons(True)  # 标签过多时可滚动

        # --- Tab 1: AI 诊疗建议 (主视图) ---
        advice_tab = QWidget()
        advice_layout = QVBoxLayout(advice_tab)

        # 全屏/操作工具栏
        advice_tool_layout = QHBoxLayout()
        self.advice_button = QPushButton("🔄 生成当前报告")
        self.advice_button.setStyleSheet(f"QPushButton {{ background-color: {self.highlight_color}; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; }} QPushButton:disabled {{ background-color: #3b4252; }}")
        self.advice_button.clicked.connect(self.show_ai_advice)
        self.advice_button.setEnabled(False)
        self.advice_button.setCursor(QCursor(Qt.PointingHandCursor))

        self.fullscreen_advice_btn = QPushButton("🗖 全屏阅览")
        self.fullscreen_advice_btn.setStyleSheet(f"QPushButton {{ background-color: transparent; border: 1px solid {self.highlight_color}; color: {self.highlight_color}; padding: 6px 12px; border-radius: 4px; }} QPushButton:hover {{ background-color: rgba(128, 90, 213, 0.2); }}")
        self.fullscreen_advice_btn.clicked.connect(self.show_fullscreen_advice)
        self.fullscreen_advice_btn.setCursor(QCursor(Qt.PointingHandCursor))

        advice_tool_layout.addWidget(self.advice_button)
        advice_tool_layout.addStretch()
        advice_tool_layout.addWidget(self.fullscreen_advice_btn)

        # 建议展示文本框
        self.advice_text = QTextEdit()
        self.advice_text.setReadOnly(True)
        self.advice_text.setMinimumHeight(150)
        self.advice_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.advice_text.setFont(QFont("Microsoft YaHei", 14))
        self.advice_text.setStyleSheet(f"background-color: {self.primary_color}; border: none; padding: 10px; color: {self.text_color}; font-size: 13pt;")
        self.advice_text.setHtml(f"""
        <div style='font-family: "Microsoft YaHei", sans-serif; text-align: center; padding: 40px 20px;'>
            <h2 style='color: {self.highlight_color}; font-size: 19pt; margin-bottom: 30px;'>DeepSeek AI 诊疗引擎</h2>
            <div style='background-color: #21252B; border-radius: 10px; padding: 25px; border: 1px solid #2c323c;
                text-align: left; display: inline-block;'>
                <h3 style='color: {self.accent_color}; margin-top: 0; font-size: 14pt;'>🚀 快速上手</h3>
                <p style='margin: 12px 0;'><b style='color: {self.accent_color};'>1.</b> 加载模型 — 选择 YOLO 权重文件 (.pt)</p>
                <p style='margin: 12px 0;'><b style='color: {self.accent_color};'>2.</b> 加载图像 — 选择本地照片或连接开发板</p>
                <p style='margin: 12px 0;'><b style='color: {self.accent_color};'>3.</b> 开始检测 — AI 自动分析并输出分类结果</p>
                <p style='margin: 12px 0;'><b style='color: {self.accent_color};'>4.</b> 点击🔄 生成当前报告获取诊疗建议</p>
            </div>
            <p style='color: #4C566A; font-size: 11pt; margin-top: 20px;'>💡 请前往「系统设置」校验 API Key 状态</p>
        </div>
        """)

        advice_layout.addLayout(advice_tool_layout)
        advice_layout.addWidget(self.advice_text)
        self.right_tab_widget.addTab(advice_tab, "🩺 诊疗建议")

        # --- Tab 2: 自定义对话 ---
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)
        chat_layout.setSpacing(10)

        # 对话历史显示区
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.primary_color};
                border: 1px solid #4C566A;
                border-radius: 6px;
                padding: 15px;
                color: {self.text_color};
                font-size: 13pt;
                line-height: 1.6;
            }}
        """)
        chat_layout.addWidget(self.chat_display, stretch=1)

        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("在此描述其他症状或向AI提问...")
        self.chat_input.setMaximumHeight(100)
        self.chat_input.setStyleSheet(f"""
            QTextEdit {{
                background-color: #1a202c;
                border: 2px solid #3b4252;
                border-radius: 8px;
                padding: 12px;
                color: {self.text_color};
                font-size: 13pt;
            }}
            QTextEdit:focus {{
                border: 2px solid {self.highlight_color};
                background-color: {self.primary_color};
            }}
        """)

        voice_ctrl_layout = QHBoxLayout()
        self.voice_chat_enabled = QCheckBox("🔊 自动朗读 AI 回复")
        self.voice_chat_enabled.setStyleSheet(f"color: {self.text_color};")
        self.voice_chat_enabled.setChecked(APP_CONFIG.enable_voice)
        self.voice_chat_enabled.stateChanged.connect(self.toggle_voice_chat)
        voice_ctrl_layout.addWidget(self.voice_chat_enabled)

        self.voice_input_button = QPushButton("🎤 语音输入")
        self.voice_input_button.setStyleSheet(f"background-color: {self.accent_color}; color: white; padding: 6px 15px; border-radius: 15px;")
        self.voice_input_button.clicked.connect(self.start_voice_input)
        self.voice_input_button.setEnabled(True)
        self.voice_input_button.setCursor(QCursor(Qt.PointingHandCursor))
        voice_ctrl_layout.addWidget(self.voice_input_button)
        voice_ctrl_layout.addStretch()

        chat_btns_layout = QHBoxLayout()
        self.send_chat_button = QPushButton("发送提问")
        self.send_chat_button.setStyleSheet(f"background-color: {self.highlight_color}; color: white; padding: 8px 20px; border-radius: 4px; font-weight: bold;")
        self.send_chat_button.clicked.connect(self.send_chat_message)
        self.send_chat_button.setCursor(QCursor(Qt.PointingHandCursor))

        self.clear_chat_button = QPushButton("清除记录")
        self.clear_chat_button.setStyleSheet(f"background-color: {self.primary_color}; border: 1px solid #4C566A; color: {self.text_color}; padding: 8px 20px; border-radius: 4px;")
        self.clear_chat_button.clicked.connect(self.clear_chat_history)
        self.clear_chat_button.setCursor(QCursor(Qt.PointingHandCursor))

        chat_btns_layout.addStretch()
        chat_btns_layout.addWidget(self.clear_chat_button)
        chat_btns_layout.addWidget(self.send_chat_button)

        self.ai_progress_bar = QProgressBar()
        self.ai_progress_bar.setVisible(False)
        self.ai_progress_bar.setTextVisible(True)
        self.ai_progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                background-color: #1E222A;
                height: 6px;
                border-radius: 3px;
                text-align: center;
                font-size: 10pt;
                color: {self.accent_color};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.accent_color}, stop:1 {self.highlight_color});
                border-radius: 3px;
            }}
        """)

        chat_layout.addWidget(self.chat_input)
        chat_layout.addLayout(voice_ctrl_layout)
        chat_layout.addWidget(self.ai_progress_bar)
        chat_layout.addLayout(chat_btns_layout)
        chat_layout.addStretch()
        self.right_tab_widget.addTab(chat_tab, "💬 医疗问答")

        # --- Tab 3: 系统设置 (收纳低频配置) ---
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setSpacing(15)

        group_style = f"QGroupBox {{ border: 1px solid #4C566A; border-radius: 6px; margin-top: 15px; padding: 15px; }} QGroupBox::title {{ color: {self.accent_color}; top: -10px; left: 10px; }}"

        api_group = QGroupBox("DeepSeek 引擎配置")
        api_group.setStyleSheet(group_style)
        api_layout_v = QVBoxLayout(api_group)

        self.use_api_checkbox = QCheckBox("启用云端 AI 推理")
        self.use_api_checkbox.setChecked(True)
        self.use_api_checkbox.setStyleSheet(f"color: {self.text_color}; font-weight: bold;")
        self.use_api_checkbox.toggled.connect(self.toggle_api_usage)

        key_input_layout = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("输入 DeepSeek API Key")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setStyleSheet(f"background-color: {self.primary_color}; border: 1px solid #4C566A; padding: 8px; border-radius: 4px; color: {self.text_color};")

        self.toggle_password_button = QPushButton("👁")
        self.toggle_password_button.setMinimumSize(42, 42)
        self.toggle_password_button.setStyleSheet("background: transparent; border: none; font-size: 15pt;")
        self.toggle_password_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.toggle_password_button.clicked.connect(self.toggle_password_visibility)

        self.save_api_key_button = QPushButton("保存配置")
        self.save_api_key_button.setStyleSheet(f"background-color: {self.success_color}; color: white; padding: 8px 15px; border-radius: 4px;")
        self.save_api_key_button.clicked.connect(self.save_api_key)
        self.save_api_key_button.setCursor(QCursor(Qt.PointingHandCursor))

        key_input_layout.addWidget(self.api_key_input)
        key_input_layout.addWidget(self.toggle_password_button)
        key_input_layout.addWidget(self.save_api_key_button)

        self.network_test_button = QPushButton("测试 API 连通性")
        self.network_test_button.setStyleSheet(f"background-color: {self.primary_color}; border: 1px solid #4C566A; color: {self.text_color}; padding: 8px 15px; border-radius: 4px;")
        self.network_test_button.clicked.connect(self.test_network_and_show_result)
        self.network_test_button.setCursor(QCursor(Qt.PointingHandCursor))

        api_layout_v.addWidget(self.use_api_checkbox)
        api_layout_v.addLayout(key_input_layout)
        api_layout_v.addWidget(self.network_test_button)

        # 语音辅助配置
        mic_group = QGroupBox("外设测试")
        mic_group.setStyleSheet(group_style)
        mic_layout = QVBoxLayout(mic_group)
        self.mic_test_button = QPushButton("测试麦克风")
        self.mic_test_button.setStyleSheet(f"background-color: {self.primary_color}; border: 1px solid #4C566A; color: {self.text_color}; padding: 8px 15px; border-radius: 4px;")
        self.mic_test_button.clicked.connect(self.test_microphone)
        self.mic_test_button.setCursor(QCursor(Qt.PointingHandCursor))
        mic_layout.addWidget(self.mic_test_button)

        # 语音时长滑块
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(3, 30)
        self.duration_slider.setValue(10)
        self.duration_value_label = QLabel("10秒")
        self.duration_value_label.setStyleSheet(f"color: {self.text_color};")
        self.duration_slider.valueChanged.connect(self.update_duration_value)

        dur_layout = QHBoxLayout()
        dur_label = QLabel("语音最长识别时间:")
        dur_label.setStyleSheet(f"color: {self.text_color};")
        dur_layout.addWidget(dur_label)
        dur_layout.addWidget(self.duration_slider)
        dur_layout.addWidget(self.duration_value_label)
        mic_layout.addLayout(dur_layout)

        settings_layout.addWidget(api_group)
        settings_layout.addWidget(mic_group)
        settings_layout.addStretch()
        self.right_tab_widget.addTab(settings_tab, "⚙️ 系统设置")

        right_layout.addWidget(self.right_tab_widget)

        # 添加到主分离器
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([900, 700])  # 调整初始大小比例

        # 语音管理器将在延迟加载中初始化, 这里不做任何操作
        # 避免阻塞主线程的Vosk模型加载
    
    def connect_smart_voice_signals(self):
        """连接智能语音信号"""
        if hasattr(self, 'voice_manager'):
            # 语音识别信号
            self.voice_manager.voice_recognized.connect(self.on_smart_voice_recognized)
            self.voice_manager.voice_error.connect(self.on_smart_voice_error)
            self.voice_manager.voice_timeout.connect(self.on_smart_voice_timeout)
            self.voice_manager.voice_unknown.connect(self.on_smart_voice_unknown)
            
            # 网络状态信号
            self.voice_manager.network_status_changed.connect(self.on_network_status_changed)
            
            # TTS信号
            self.voice_manager.tts_started.connect(self.on_smart_tts_started)
            self.voice_manager.tts_finished.connect(self.on_smart_tts_finished)
            self.voice_manager.tts_error.connect(self.on_smart_tts_error)
    
    def on_smart_voice_recognized(self, text):
        """智能语音识别成功"""
        # 过滤掉状态信号,只处理实际识别的文本
        if text and not text.startswith('__') and hasattr(self, 'chat_input'):
            self.chat_input.setPlainText(text)
            self.status_bar.showMessage(f"智能语音识别成功: {text}")
            # 恢复按钮状态
            if hasattr(self, 'voice_input_button'):
                self.voice_input_button.setText("🎤 语音输入")
                self.voice_input_button.setEnabled(True)
                self.voice_input_button.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {self.highlight_color};
                        color: white;
                        border: none;
                        border-radius: 8px;
                        padding: 10px 20px;
                        font-size: 13pt;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: #2c5aa0;
                        transform: translateY(-2px);
                    }}
                """)
        elif text == "__RECORDING_START__":
            self.status_bar.showMessage("🎤 开始录音,请说话...")
            if hasattr(self, 'voice_input_button'):
                self.voice_input_button.setText("🛑 点击停止")
                self.voice_input_button.setEnabled(True)
        elif text == "__RECORDING__":
            self.status_bar.showMessage("🎤 正在录音中...")
            if hasattr(self, 'voice_input_button'):
                self.voice_input_button.setText("🛑 点击停止")
                self.voice_input_button.setEnabled(True)
        elif text == "__PROCESSING__":
            self.status_bar.showMessage("🔄 正在识别语音内容...")
            if hasattr(self, 'voice_input_button'):
                self.voice_input_button.setText("⏳ 识别中...")
                self.voice_input_button.setEnabled(False)
    
    def on_smart_voice_error(self, error):
        """智能语音识别错误"""
        self.status_bar.showMessage(f"智能语音识别错误: {error}")
        # 恢复按钮状态
        if hasattr(self, 'voice_input_button'):
            self.voice_input_button.setText("🎤 语音输入")
            self.voice_input_button.setEnabled(True)
        if "录音已取消" not in error:  # 如果不是用户主动取消,则显示错误对话框
            self.show_message_box("错误", f"智能语音识别失败：{error}", QMessageBox.Critical)
    
    def on_smart_voice_timeout(self):
        """智能语音识别超时"""
        self.status_bar.showMessage("智能语音识别超时")
        # 恢复按钮状态
        if hasattr(self, 'voice_input_button'):
            self.voice_input_button.setText("🎤 语音输入")
            self.voice_input_button.setEnabled(True)
        self.show_message_box("提示", "录音超时,请重试", QMessageBox.Information)
    
    def on_smart_voice_unknown(self):
        """智能语音无法识别"""
        self.status_bar.showMessage("无法识别语音")
        # 恢复按钮状态
        if hasattr(self, 'voice_input_button'):
            self.voice_input_button.setText("🎤 语音输入")
            self.voice_input_button.setEnabled(True)
        self.show_message_box("提示", "无法识别您的语音,请清晰地说话后重试", QMessageBox.Information)
    
    def on_network_status_changed(self, is_online, message):
        """网络状态变化"""
        if is_online:
            self.status_bar.showMessage(f"🌐 {message}")
        else:
            self.status_bar.showMessage(f"💻 {message}")
    
    def on_smart_tts_started(self):
        """智能TTS开始"""
        self.status_bar.showMessage("开始播放语音...")
    
    def on_smart_tts_finished(self):
        """智能TTS完成"""
        self.status_bar.showMessage("语音播放完成")
    
    def on_smart_tts_error(self, error):
        """智能TTS错误"""
        self.status_bar.showMessage(f"语音播放错误: {error}")
        self.show_message_box("错误", f"语音播放失败：{error}", QMessageBox.Critical)
    
    def start_smart_voice_input(self):
        """开始智能语音输入"""
        if hasattr(self, 'voice_manager'):
            self.voice_manager.start_voice_recognition()
    
    def test_smart_microphone(self):
        """测试智能麦克风"""
        if hasattr(self, 'voice_manager'):
            success, message = self.voice_manager.test_microphone()
            if success:
                self.show_message_box("成功", message, QMessageBox.Information)
            else:
                self.show_message_box("错误", message, QMessageBox.Critical)
    
    def get_smart_voice_status(self):
        """获取智能语音状态"""
        if hasattr(self, 'voice_manager'):
            return self.voice_manager.get_voice_status()
        return {}

    def init_status_bar(self):
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {self.primary_color};
                color: {self.text_color};
                border-top: 1px solid #3b4252;
                padding: 8px 15px;
                font-size: 13pt;
                font-weight: bold;
            }}
        """)
        self.setStatusBar(self.status_bar)
        self.model_info_label = QLabel("模型未加载 | 八类分类")
        self.status_bar.addPermanentWidget(self.model_info_label)
        self.network_info_label = QLabel("局域网：" + get_local_ip())
        self.status_bar.addPermanentWidget(self.network_info_label)
        self.status_bar.showMessage("系统就绪,请点击「加载模型」按钮选择模型文件")

    # ------------------------------------------------------------------
    #  以下为功能实现（保持原实现不动,仅修正明显错误）
    # ------------------------------------------------------------------
    def toggle_password_visibility(self):
        """切换密码可见性"""
        if self.api_key_input.echoMode() == QLineEdit.Password:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
            self.toggle_password_button.setText("隐藏")
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)
            self.toggle_password_button.setText("显示")

    def send_chat_message(self):
        """发送自定义医疗对话消息"""
        # 使用带进度条的发送方法
        self.send_chat_message_with_progress()

    def display_chat_with_context(self, current_response):
        """显示带有上下文的完整对话历史"""
        try:
            # 构建完整的对话历史HTML
            chat_html = """
            <div style='font-family: "Microsoft YaHei", "SimHei", sans-serif; line-height: 1.6; padding: 10px;'>
            """
            
            # 显示历史对话（最近5轮）
            if len(self.chat_history) > 1:
                chat_html += """
                <div style='background-color: #2a3441; border-left: 4px solid #00B5D8; padding: 15px; margin-bottom: 20px; border-radius: 8px;'>
                    <h3 style='color: #00B5D8; margin-top: 0;'>📋 对话历史</h3>
                """
                
                for i, chat in enumerate(self.chat_history[-6:-1], 1):  # 显示最近5轮历史
                    chat_html += f"""
                    <div style='margin-bottom: 15px; padding: 10px; background-color: #1a202c; border-radius: 6px;'>
                        <div style='color: #805AD5; font-weight: bold; margin-bottom: 5px;'>
                            👤 第{i}轮问题 ({chat['timestamp']}):
                        </div>
                        <div style='color: #e2e8f0; margin-bottom: 8px; padding-left: 15px;'>
                            {chat['question'][:200]}{'...' if len(chat['question']) > 200 else ''}
                        </div>
                        <div style='color: #00B5D8; font-weight: bold; margin-bottom: 5px;'>
                            🩺 医生回复:
                        </div>
                        <div style='color: #e2e8f0; padding-left: 15px; font-size: 13pt;'>
                            {chat['answer'][:300]}{'...' if len(chat['answer']) > 300 else ''}
                        </div>
                    </div>
                    """
                
                chat_html += "</div>"
            
            # 显示当前问题和回复
            if not self.chat_history:
                # 如果没有对话历史,直接显示当前回复
                self.chat_display.setHtml(self.format_advice_html(current_response))
                return
            
            current_chat = self.chat_history[-1]
            chat_html += f"""
            <div style='background-color: #2a3441; border-left: 4px solid #805AD5; padding: 15px; margin-bottom: 20px; border-radius: 8px;'>
                <h3 style='color: #805AD5; margin-top: 0;'>💬 当前对话</h3>
                <div style='margin-bottom: 15px;'>
                    <div style='color: #805AD5; font-weight: bold; margin-bottom: 5px;'>
                        👤 您的问题 ({current_chat['timestamp']}):
                    </div>
                    <div style='color: #e2e8f0; margin-bottom: 15px; padding: 10px; background-color: #1a202c; border-radius: 6px;'>
                        {current_chat['question']}
                    </div>
                    <div style='color: #00B5D8; font-weight: bold; margin-bottom: 10px;'>
                        🩺 AI医生回复:
                    </div>
                    <div style='background-color: #1a202c; padding: 15px; border-radius: 6px;'>
                        {self.format_advice_content(current_response)}
                    </div>
                </div>
            </div>
            """
            
            # 添加清除历史按钮提示
            if len(self.chat_history) > 3:
                chat_html += """
                <div style='text-align: center; margin-top: 20px; padding: 10px; background-color: #2a3441; border-radius: 6px;'>
                    <p style='color: #00B5D8; margin: 0; font-size: 11pt;'>
                        💡 提示: 对话历史已保留,AI会根据上下文提供更准确的建议
                    </p>
                </div>
                """
            
            chat_html += "</div>"
            
            # 设置到chat_display
            self.chat_display.setHtml(chat_html)
            
        except Exception as e:
            print(f"显示对话上下文时出错: {e}")
            # 如果出错,至少显示当前回复
            self.chat_display.setHtml(self.format_advice_html(current_response))

    def format_advice_content(self, content):
        """格式化建议内容为HTML（统一使用 format_advice_html 解析器）"""
        return self.format_advice_html(content)

    def clear_chat_history(self):
        from src.visualization.conversation import reset
        reset(self)
        self.status_bar.showMessage("会话与模型上下文已清除，辅助建议保留")

    def show_network_error_result(self, title, error_message):
        """显示网络错误结果"""
        result_html = f"""
        <div style='text-align: center; padding: 40px; font-family: "Microsoft YaHei", "SimHei", sans-serif;'>
            <h2 style='color: #f56565; margin-bottom: 20px;'>❌ {title}</h2>
            <div style='background-color: #2a3441; padding: 20px; border-radius: 8px; margin: 20px 0;'>
                <h3 style='color: #00B5D8; margin-top: 0;'>错误信息</h3>
                <p style='color: #f56565; margin: 10px 0; font-size: 13pt;'>{error_message}</p>
            </div>
            <div style='margin-top: 20px; padding: 15px; background-color: #2a3441; border-radius: 6px;'>
                <h4 style='color: #805AD5; margin-top: 0;'>解决方案:</h4>
                <ul style='color: #e2e8f0; text-align: left; font-size: 13pt;'>
                    <li>请检查网络连接是否正常</li>
                    <li>确认API密钥是否正确设置</li>
                    <li>稍后重试或联系技术支持</li>
                </ul>
            </div>
        </div>
        """
        self.advice_text.setHtml(result_html)
        self.status_bar.showMessage("网络测试失败")


    def show_unified_response(self, question, answer):
        """显示统一的回复对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("AI医疗建议")
        dialog.resize(800, 600)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {self.background_color};
                color: {self.text_color};
            }}
        """)
        
        # 主布局
        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 顶部栏：添加"全屏/退出全屏"切换与快捷键
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        dialog_fullscreen_btn = QPushButton("全屏")
        dialog_fullscreen_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.highlight_color};
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 13pt;
            }}
            QPushButton:hover {{
                background-color: #d53f8c;
            }}
        """)

        def toggle_dialog_fullscreen():
            if dialog.isFullScreen():
                dialog.showNormal()
                dialog_fullscreen_btn.setText("全屏")
            else:
                dialog.showFullScreen()
                dialog_fullscreen_btn.setText("退出全屏")

        dialog_fullscreen_btn.clicked.connect(toggle_dialog_fullscreen)
        QShortcut(QKeySequence("F11"), dialog, activated=toggle_dialog_fullscreen)
        QShortcut(QKeySequence("Esc"), dialog, activated=lambda: (dialog.showNormal(), dialog_fullscreen_btn.setText("全屏")))
        top_bar.addWidget(dialog_fullscreen_btn)
        main_layout.addLayout(top_bar)
        
        # 创建滚动区域来容纳所有内容
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {self.background_color};
            }}
            QScrollBar:vertical {{
                background-color: {self.secondary_bg};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {self.accent_color};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {self.highlight_color};
            }}
        """)
        
        # 滚动区域的内容widget
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(15)
        
        # 问题区域
        question_group = QGroupBox("您的问题")
        question_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {self.accent_color};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                color: {self.accent_color};
                font-weight: bold;
                font-size: 13pt;
            }}
        """)
        question_layout = QVBoxLayout()
        question_label = QLabel(question)
        question_label.setWordWrap(True)
        question_label.setStyleSheet(f"padding: 10px; font-size: 11pt; line-height: 1.4;")
        question_layout.addWidget(question_label)
        question_group.setLayout(question_layout)
        
        # 回答区域
        answer_group = QGroupBox("AI医疗建议")
        answer_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {self.highlight_color};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                color: {self.highlight_color};
                font-weight: bold;
                font-size: 13pt;
            }}
        """)
        answer_layout = QVBoxLayout()
        
        # 使用QTextEdit显示格式化的回答,设置适当的最小高度
        answer_text = QTextEdit()
        answer_text.setPlainText(answer)
        answer_text.setReadOnly(True)
        answer_text.setMinimumHeight(200)  # 设置最小高度
        answer_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.secondary_bg};
                color: {self.text_color};
                border: none;
                padding: 15px;
                font-size: 13pt;
                line-height: 1.5;
                border-radius: 6px;
            }}
            QTextEdit QScrollBar:vertical {{
                background-color: {self.secondary_bg};
                width: 10px;
                border-radius: 5px;
            }}
            QTextEdit QScrollBar::handle:vertical {{
                background-color: {self.accent_color};
                border-radius: 5px;
                min-height: 20px;
            }}
        """)
        answer_layout.addWidget(answer_text)
        answer_group.setLayout(answer_layout)
        
        # 添加到滚动布局
        scroll_layout.addWidget(question_group)
        scroll_layout.addWidget(answer_group)
        scroll_layout.addStretch()  # 添加弹性空间
        
        # 设置滚动区域内容
        scroll_area.setWidget(scroll_content)
        
        # 关闭按钮
        close_button = QPushButton("关闭")
        close_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.accent_color};
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13pt;
            }}
            QPushButton:hover {{
                background-color: #0097B2;
            }}
        """)
        close_button.clicked.connect(dialog.accept)
        
        # 添加到主布局
        main_layout.addWidget(scroll_area, 1)  # stretch=1,让滚动区域占据大部分空间
        main_layout.addWidget(close_button)
        
        style_dialog(dialog, self)
        dialog.exec_()

    def save_api_key(self):
        """保存API密钥"""
        try:
            api_key = self.api_key_input.text().strip()
            
            if api_key:
                # 验证API密钥格式
                if not api_key.startswith('sk-'):
                    reply = self.show_message_box("警告", 
                        "API密钥格式可能不正确,通常以'sk-'开头。\n\n是否仍要保存此密钥？", 
                        QMessageBox.Question)
                    if reply != QMessageBox.Yes:
                        return
                
                # 验证API密钥长度（DeepSeek API密钥通常较长）
                if len(api_key) < 20:
                    reply = self.show_message_box("警告", 
                        "API密钥长度似乎过短,可能不是有效的DeepSeek API密钥。\n\n是否仍要保存此密钥？", 
                        QMessageBox.Question)
                    if reply != QMessageBox.Yes:
                        return
                
                # 确保deepseek_api已初始化
                if self.deepseek_api is None:
                    self.deepseek_api = DeepSeekAPI()
                self.deepseek_api.set_api_key(api_key)
                
                # 持久化保存API密钥到文件 (Base64混淆)
                try:
                    from src.core.credentials import encode_key
                    encoded = encode_key(api_key)
                    with open(PROJECT_ROOT / "saved_api_key.txt", 'w', encoding='utf-8') as f:
                        f.write(encoded)
                    print("API密钥已加密保存到文件")
                except Exception as e:
                    self.show_message_box("保存失败", "密钥未能写入本机安全存储，请检查目录权限后重试。", QMessageBox.Warning)
                    return
                
                self.status_bar.showMessage("API密钥已保存")
                self.show_message_box("成功", "DeepSeek API密钥已加密保存，即时生效。可测试连接后使用辅助问答。", QMessageBox.Information)
            else:
                self.status_bar.showMessage("API密钥为空")
                self.show_message_box("提示", "API密钥为空,将使用内置的治疗建议。", QMessageBox.Warning)
                
        except Exception as e:
            print(f"保存API密钥时出错: {e}")
            self.show_message_box("错误", f"保存API密钥时发生错误: {str(e)}", QMessageBox.Critical)

    def _cleanup_temp_images(self):
        """启动时自动清理 medical_images/ 中 7 天前的临时图像"""
        try:
            img_dir = str(APP_CONFIG.image_dir)
            if not os.path.isdir(img_dir):
                return
            cutoff = time.time() - 7 * 24 * 3600
            cleaned = 0
            for f in os.listdir(img_dir):
                if f.startswith("temp_image_") and f.endswith(".png"):
                    fpath = os.path.join(img_dir, f)
                    if os.path.getmtime(fpath) < cutoff:
                        try:
                            os.remove(fpath)
                            cleaned += 1
                        except OSError:
                            pass
            if cleaned > 0:
                print(f"[Cleanup] 已清理 {cleaned} 个过期临时图像")
        except Exception:
            pass  # 清理失败不阻塞启动

    def show_message_box(self, title, message, icon=QMessageBox.Information):
        """显示消息框"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(icon)
        
        # 如果是Question类型的消息框,添加Yes/No按钮
        if icon == QMessageBox.Question:
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
        
        msg_box.setStyleSheet(f"""
            QMessageBox {{
                background-color: {self.background_color};
                color: {self.text_color};
            }}
            QPushButton {{
                background-color: {self.accent_color};
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #0097B2;
            }}
        """)
        style_dialog(msg_box, self)
        return msg_box.exec_()







    def export_analysis(self, prefix, figures, headers, rows, messages=None):
        """Export the exact displayed snapshot, never query a different data scope."""
        import csv
        from pathlib import Path
        directory=QFileDialog.getExistingDirectory(self,"选择导出目录")
        if not directory:return
        target=Path(directory)/(prefix+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        try:
            target.mkdir()
            with (target/"statistics.csv").open("w",encoding="utf-8-sig",newline="") as stream:
                writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)
            if messages is not None:
                with (target/"details.csv").open("w",encoding="utf-8-sig",newline="") as stream:
                    writer=csv.writer(stream);writer.writerow(["任务明细"]);writer.writerows([[m] for m in messages])
            for index,figure in enumerate(figures,1):
                for extension in ("png","pdf"):
                    figure.savefig(str(target/f"chart_{index}.{extension}"),dpi=200,bbox_inches="tight",facecolor="#242C37",transparent=False)
            self.show_message_box("导出完成",str(target))
        except Exception as error:
            self.show_message_box("导出失败",f"{error}；已生成的部分文件位于 {target}")

    def show_batch_report(self, disease_counter, results_summary):
        plt, FigureCanvas = load_plotting()
        fig = None
        dialog = QDialog(self)
        dialog.setWindowTitle("批量检测统计报告")
        dialog.resize(1150, 850)
        dialog.setStyleSheet(f"""
            QDialog {{ background-color: {self.background_color}; color: {self.text_color}; }}
        """)

        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # ===== 1. KPI 数据卡片 =====
        summary_group = QGroupBox("📊 总体检测数据")
        summary_group.setStyleSheet(f"""
            QGroupBox {{ border: 1px solid #3b4252; border-top: 3px solid {self.accent_color};
                border-radius: 8px; padding-top: 25px; font-weight: bold; color: white; }}
            QGroupBox::title {{ color: {self.accent_color}; top: -10px; left: 15px; font-size: 13pt; }}
        """)
        cards_layout = QHBoxLayout(summary_group)
        cards_layout.setSpacing(15)

        total = sum(disease_counter.values())
        total_card = QLabel(f"<div style='text-align:center;'><div style='font-size:12pt;color:#81A1C1;margin-bottom:5px;'>已保存成功数</div><div style='font-size:21pt;color:#00E5FF;font-weight:bold;'>{total}</div></div>")
        total_card.setStyleSheet("background-color: #21252B; border-radius: 8px; padding: 15px; border: 1px solid #2c323c;")
        cards_layout.addWidget(total_card)

        cards_layout.addWidget(QLabel(f"已处理 {len(results_summary)} 张 · 未成功保存 {len(results_summary)-total} 张"))
        cards_layout.addWidget(QLabel("范围：本次任务已完成部分，不包含历史记录"))

        cards_layout.addStretch()
        main_layout.addWidget(summary_group, stretch=0)

        # ===== 2. 图表区 =====
        if disease_counter:
            filtered = {k: v for k, v in disease_counter.items() if v > 0}
            theme_colors = ['#4DD6B4', '#51C5E3', '#F1B761', '#A7C98B', '#B7A1D8', '#E58C89', '#8BC4CB', '#CBB394']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.set_facecolor('#1a1e24'); fig.patch.set_alpha(0.0)

            wedges, texts, autotexts = ax1.pie(
                list(filtered.values()), labels=list(filtered.keys()), autopct='%1.1f%%',
                startangle=140, colors=theme_colors[:len(filtered)],
                textprops={'color': 'white', 'fontweight': 'bold'},
                wedgeprops={'edgecolor': '#242C37', 'linewidth': 1.5})
            ax1.set_title('疾病分布比例', color='white', pad=20, fontsize=14)
            for t in texts: t.set_color('white')
            for t in autotexts: t.set_color('white')

            bars = ax2.bar(list(filtered.keys()), list(filtered.values()),
                           color=theme_colors[:len(filtered)])
            ax2.set_ylabel('数量', color='white')
            ax2.set_title('疾病分布柱状图', color='white', pad=20, fontsize=14)
            ax2.tick_params(axis='x', rotation=45, colors='white')
            ax2.tick_params(axis='y', colors='white')
            ax2.set_facecolor('#242C37')
            ax2.grid(color='#4a5568', linestyle='--', linewidth=0.5, axis='y')
            for bar in bars:
                h = bar.get_height()
                ax2.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='white')
            for ax in [ax1, ax2]:
                ax.set_facecolor('#1a1e24'); ax.patch.set_alpha(0.0)
                for spine in ax.spines.values(): spine.set_edgecolor('#4a5568')

            fig.subplots_adjust(top=0.85, bottom=0.2, wspace=0.25)
            fig.tight_layout(pad=2.0)
            canvas = FigureCanvas(fig)
            chart_container = QWidget()
            chart_container.setStyleSheet("background-color: #2d3748; border-radius: 8px;")
            chart_layout = QVBoxLayout(chart_container)
            chart_layout.setContentsMargins(5, 5, 5, 5)
            chart_layout.addWidget(canvas)
            stats_tabs=QTabWidget()
            stats_tabs.addTab(chart_container,"分布图")
            table=QTableWidget(len(filtered),3)
            table.setHorizontalHeaderLabels(["分类类别","数量","占成功结果比例"])
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            for row,(disease,count) in enumerate(filtered.items()):
                for col,value in enumerate([disease,str(count),f"{count/total:.1%}"]):
                    table.setItem(row,col,QTableWidgetItem(value))
            stats_tabs.addTab(table,"统计表")
            main_layout.addWidget(stats_tabs, stretch=4)

        # ===== 3. 暗黑终端日志 =====
        log_group = QWidget()
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(5)

        log_label = QLabel("📋 详细检测日志")
        log_label.setStyleSheet("color: #81A1C1; font-weight: bold; font-size: 13pt;")
        log_layout.addWidget(log_label)

        result_box = QTextEdit()
        result_box.setReadOnly(True)
        formatted_logs = []
        for line in results_summary:
            if "✅" in line: formatted_logs.append(f"<span style='color: #00E676;'>{line}</span>")
            elif "❌" in line: formatted_logs.append(f"<span style='color: #FF5252;'>{line}</span>")
            else: formatted_logs.append(f"<span style='color: #A0AEC0;'>{line}</span>")
        result_box.setHtml("<br>".join(formatted_logs))
        result_box.setStyleSheet(f"""
            QTextEdit {{ background-color: #11151c; color: #A0AEC0; border: 1px solid #2c323c;
                border-radius: 6px; padding: 12px; font-family: Consolas, "Courier New", monospace;
                font-size: 13pt; line-height: 1.5; }}
            QScrollBar:vertical {{ background-color: #11151c; width: 10px; }}
            QScrollBar::handle:vertical {{ background-color: #4C566A; border-radius: 5px; }}
        """)
        log_layout.addWidget(result_box)
        main_layout.addWidget(log_group, stretch=2)

        # ===== 4. 按钮区 =====
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        btn_style = f"QPushButton {{ background-color: {self.accent_color}; color: white; padding: 10px 24px; border-radius: 6px; font-weight: bold; font-size: 13pt; }} QPushButton:hover {{ background-color: #0097B2; }}"

        fullscreen_btn = QPushButton("🖥️ 全屏")
        fullscreen_btn.setStyleSheet(f"""
            QPushButton {{ background-color: transparent; color: {self.accent_color};
                border: 1px solid {self.accent_color}; padding: 8px 30px; border-radius: 6px;
                font-weight: bold; font-size: 13pt; }}
            QPushButton:hover {{ background-color: rgba(0, 181, 216, 0.1); }}
        """)
        fullscreen_btn.clicked.connect(lambda: self.toggle_batch_report_fullscreen(dialog))

        close_btn = QPushButton("❌ 关闭")
        close_btn.setStyleSheet(f"""
            QPushButton {{ background-color: #3b4252; color: #E5E9F0; border: none;
                padding: 8px 30px; border-radius: 6px; font-weight: bold; font-size: 13pt; }}
            QPushButton:hover {{ background-color: #4C566A; }}
        """)
        close_btn.clicked.connect(dialog.accept)

        button_layout.addWidget(fullscreen_btn)
        button_layout.addSpacing(15)
        button_layout.addWidget(close_btn)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        export_button=QPushButton("导出统计图与数据")
        export_button.clicked.connect(lambda: self.export_analysis(
            "batch", [fig] if fig is not None else [],
            ["类别", "成功数量"], list(disease_counter.items()), results_summary))
        main_layout.addWidget(export_button)
        style_dialog(dialog, self)
        try:
            dialog.exec_()
        finally:
            if fig is not None:plt.close(fig)

    def save_to_history(self, image_path, disease_name, confidence):
        """保存检测结果到历史记录 (SQLite)，仅保存有效疾病名"""
        try:
            # 过滤掉以 [对话] 开头的聊天记录
            if not valid_prediction(disease_name, confidence):
                return False
            db = get_history_db()
            db.add(
                record_id=str(uuid.uuid4()),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                image_path=image_path,
                disease_name=disease_name,
                confidence=confidence
            )
            return True
        except Exception as e:
            print(f"保存历史记录失败: {e}")
            return False

    def load_history_records(self):
        """加载历史记录 (SQLite, 首次自动迁移旧 JSON)"""
        try:
            db = get_history_db()
            # 首次使用自动迁移旧数据
            history_dir = str(APP_CONFIG.history_db.parent)
            json_path = os.path.join(history_dir, "history.json")
            if os.path.exists(json_path):
                db.migrate_from_json(json_path)
            return db.get_all()
        except Exception as e:
            print(f"加载历史记录失败: {e}")
            return []

    def show_history(self, embedded=False):
        """History page reuses the existing detail, delete and statistics actions."""
        if not embedded:
            self.workspace_pages.setCurrentIndex(2)
            return
        # 创建对话框
        history_dialog = QWidget(self)
        history_dialog.setWindowTitle("检测历史记录")
        history_dialog.resize(1200, 800)
        history_dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {self.background_color};
                color: {self.text_color};
            }}
        """)

        # 创建布局
        main_layout = QVBoxLayout(history_dialog)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(15)

        # 创建标题
        title_label = QLabel("检测历史记录")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setStyleSheet(f"""
            padding: 15px;
            color: {self.accent_color};
            border-bottom: 2px solid {self.accent_color};
            margin-bottom: 20px;
        """)
        main_layout.addWidget(title_label)

        # 创建表格视图
        self.history_table = QTableWidget()
        self.history_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {self.secondary_bg};
                color: {self.text_color};
                border: 1px solid #3b4252;
                border-radius: 6px;
                gridline-color: #2c323c;
                outline: none;
            }}
            QTableWidget::item:focus {{
                outline: none;
            }}
            QHeaderView::section {{
                background-color: {self.primary_color};
                color: #81A1C1;
                padding: 12px 15px;
                border: none;
                border-bottom: 2px solid {self.accent_color};
                border-right: 1px solid #3b4252;
                font-weight: bold;
                font-size: 13pt;
            }}
            QTableWidget::item {{
                padding: 8px;
                border-bottom: 1px solid #3b4252;
            }}
            QTableWidget::item:selected {{
                background-color: rgba(0, 181, 216, 0.35);
                color: white;
            }}
            QTableWidget::item:selected:!active {{
                background-color: rgba(0, 181, 216, 0.25);
                color: #E5E9F0;
            }}
            QTableWidget::item:alternate {{
                background-color: #262B33;
            }}
            QTableWidget::item:alternate:selected {{
                background-color: rgba(0, 181, 216, 0.35);
                color: white;
            }}
        """)
        # 整行选择样式
        self.history_table.setStyleSheet(self.history_table.styleSheet() + f"""
            QTableWidget {{
                selection-background-color: rgba(0, 181, 216, 0.35);
                selection-color: white;
            }}
        """)

        # 隐藏行号列,设置行高
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)  # 根据字体自动调整行高
        self.history_table.verticalHeader().setMinimumSectionSize(45)  # 设置最小行高

        # 设置表格列
        columns = ["时间戳", "图像名称", "检测结果", "置信度", "操作"]
        self.history_table.setColumnCount(len(columns))
        self.history_table.setHorizontalHeaderLabels(columns)

        legacy_history = APP_CONFIG.history_db.parent / "history.json"
        if legacy_history.is_file():
            get_history_db().migrate_from_json(str(legacy_history))
        self._history_page = 0
        pager = QHBoxLayout()
        self.history_previous = QPushButton("上一页")
        self.history_next = QPushButton("下一页")
        self.history_page_label = QLabel()
        self.history_previous.clicked.connect(lambda: self._change_history_page(-1))
        self.history_next.clicked.connect(lambda: self._change_history_page(1))
        pager.addWidget(self.history_previous)
        pager.addWidget(self.history_page_label, 1)
        pager.addWidget(self.history_next)
        main_layout.addLayout(pager)
        self._populate_history_table()

        main_layout.addWidget(self.history_table)
        
        # 优化表格显示
        self.history_table.setAlternatingRowColors(True)  # 交替行颜色
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)  # 整行选择
        self.history_table.setSelectionMode(QTableWidget.ExtendedSelection)  # 多行选择

        # 单元格点击 → 查看详情（替代逐个创建 QPushButton）
        def on_cell_clicked(row, col):
            if col == 4 and hasattr(self, '_history_records_cache'):
                if 0 <= row < len(self._history_records_cache):
                    self.view_history_record(self._history_records_cache[row])
        self.history_table.cellClicked.connect(on_cell_clicked)

        # 创建按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        # 删除选中记录按钮
        delete_button = QPushButton("删除选中记录")
        delete_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #e53e3e;
                color: white;
                padding: 10px 22px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #c53030;
            }}
            QPushButton:disabled {{
                background-color: #718096;
                color: #a0aec0;
            }}
        """)
        delete_button.clicked.connect(self.delete_selected_history)

        # 清空历史按钮
        clear_button = QPushButton("清空历史记录")
        clear_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #ed8936;
                color: white;
                padding: 10px 22px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #dd6b20;
            }}
            QPushButton:disabled {{
                background-color: #718096;
                color: #a0aec0;
            }}
        """)
        clear_button.clicked.connect(self.clear_history)

        # 关闭按钮
        close_button = QPushButton("关闭")
        close_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.accent_color};
                color: white;
                padding: 10px 22px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #0097B2;
            }}
        """)
        close_button.setText("刷新记录")
        close_button.clicked.connect(self._populate_history_table)

        # 趋势分析按钮
        trend_btn = QPushButton("分类记录统计")
        trend_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.accent_color};
                color: white;
                font-weight: bold;
                padding: 10px 22px;
                border-radius: 6px;
            }}
            QPushButton:hover {{ background-color: #0097B2; }}
        """)
        trend_btn.clicked.connect(self.show_trend_analysis)

        button_layout.addStretch()
        button_layout.addWidget(trend_btn)
        button_layout.addWidget(delete_button)
        button_layout.addWidget(clear_button)
        button_layout.addWidget(close_button)
        main_layout.addLayout(button_layout)

        # 显示对话框
        style_dialog(history_dialog, self)
        return history_dialog

    def view_history_record(self, record):
        """Read-only historical detail; opening a record never generates new advice."""
        from src.visualization.workbench_layout import ImageCanvas, NAMES
        from pathlib import Path
        dialog=QDialog(self);dialog.setWindowTitle('历史记录详情');dialog.setSizeGripEnabled(True)
        layout=QVBoxLayout(dialog);layout.setContentsMargins(20,18,20,18);layout.setSpacing(16)
        header=QHBoxLayout();title=QLabel('历史分析详情');title.setObjectName('pageTitle');header.addWidget(title);header.addStretch()
        fullscreen=QPushButton('全屏查看');header.addWidget(fullscreen);layout.addLayout(header)
        def toggle():
            if dialog.isFullScreen():dialog.showNormal();fullscreen.setText('全屏查看')
            else:dialog.showFullScreen();fullscreen.setText('退出全屏')
        fullscreen.clicked.connect(toggle);QShortcut(QKeySequence('F11'),dialog,activated=toggle)
        split=QSplitter(Qt.Horizontal);layout.addWidget(split,1)
        left=QWidget();ll=QVBoxLayout(left);ll.setContentsMargins(0,0,12,0)
        label=QLabel('原始图像');label.setObjectName('sectionTitle');ll.addWidget(label)
        path=str(record.get('image_path') or record.get('path') or '')
        canvas=ImageCanvas();canvas.setAcceptDrops(False);canvas.setCursor(Qt.ArrowCursor);canvas.setToolTip('原始图像 · 等比例显示')
        image=None
        try:image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        except (OSError,ValueError):pass
        if image is not None:
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);h,w,_=rgb.shape
            canvas.setPixmap(QPixmap.fromImage(QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()))
            ll.addWidget(canvas,1)
        else:
            missing=QLabel('原始图像已移动或无法读取');missing.setAlignment(Qt.AlignCenter);missing.setWordWrap(True);ll.addWidget(missing,1)
        filename=QLabel(Path(path).name);filename.setWordWrap(True);filename.setToolTip(path);filename.setTextInteractionFlags(Qt.TextSelectableByMouse);ll.addWidget(filename)
        split.addWidget(left)
        right=QWidget();rl=QVBoxLayout(right);rl.setContentsMargins(12,0,0,0);rl.setSpacing(12)
        disease=str(record.get('disease_name','未知'));name=NAMES.get(disease.replace('[演示]',''),disease)
        result=QLabel(name);result.setWordWrap(True);result.setObjectName('prediction');rl.addWidget(result)
        confidence=record.get('confidence');score=QLabel(f'模型概率  {confidence:.2%}' if isinstance(confidence,(float,int)) else '模型概率未记录');score.setObjectName('sectionTitle');rl.addWidget(score)
        stamp=QLabel('分析时间  '+str(record.get('timestamp','未记录')));stamp.setWordWrap(True);stamp.setObjectName('muted');rl.addWidget(stamp)
        note=QLabel('模型概率未经校准，分类结果仅供辅助参考。');note.setWordWrap(True);note.setObjectName('notice');rl.addWidget(note)
        heading=QLabel('已保存的辅助说明');heading.setObjectName('sectionTitle');rl.addWidget(heading)
        advice=QTextEdit();advice.setReadOnly(True);rl.addWidget(advice,1)
        saved=record.get('advice')
        if saved and str(saved).strip():advice.setHtml(self.format_advice_html(str(saved)))
        else:advice.setPlainText('此记录尚未保存辅助说明。历史详情只展示已保存内容。')
        split.addWidget(right);split.setChildrenCollapsible(False);split.setSizes([380,620])
        footer=QHBoxLayout();hint=QLabel('拖动中间分隔线可调整图像与说明比例');hint.setObjectName('muted');hint.setWordWrap(True);footer.addWidget(hint,1)
        close=QPushButton('关闭');close.clicked.connect(dialog.accept);footer.addWidget(close);layout.addLayout(footer)
        style_dialog(dialog,self)
        advice.document().setDefaultFont(QFont(self.font_family_combo.currentData(),self.font_size_combo.currentData() or 14))
        screen=self.screen().availableGeometry();dialog.resize(min(1080,screen.width()-60),min(720,screen.height()-70))
        dialog.move(screen.center()-dialog.rect().center());dialog.exec_()

    def show_history_advice(self, disease_name, confidence):
        """显示历史记录的AI建议"""
        self.status_bar.showMessage("正在生成AI治疗建议,请稍候...")
        QApplication.processEvents()

        # 获取治疗建议原始文本
        raw_advice = self.deepseek_api.get_treatment_advice(disease_name, confidence)

        # 创建对话框显示建议
        advice_dialog = QDialog(self)
        advice_dialog.setWindowTitle(f"{disease_name}的治疗建议")
        advice_dialog.resize(800, 600)
        advice_dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {self.background_color};
                color: {self.text_color};
            }}
        """)

        # 创建布局
        main_layout = QVBoxLayout(advice_dialog)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {self.accent_color};
                border-radius: 8px;
                background-color: {self.secondary_bg};
            }}
        """)

        # 创建文本编辑区域用于显示建议
        advice_text = QTextEdit()
        advice_text.setReadOnly(True)
        advice_text.setMinimumHeight(600)
        advice_text.setFont(QFont("Microsoft YaHei", 12))
        advice_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.secondary_bg};
                color: {self.text_color};
                border: none;
                padding: 15px;
            }}
        """)

        # 格式化建议文本
        formatted_advice = self.format_advice_html(raw_advice)
        advice_text.setHtml(formatted_advice)

        scroll_area.setWidget(advice_text)
        main_layout.addWidget(scroll_area)

        # 添加关闭按钮
        close_button = QPushButton("关闭")
        close_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.accent_color};
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #0097B2;
            }}
        """)
        close_button.clicked.connect(advice_dialog.accept)

        main_layout.addWidget(close_button)

        # 显示对话框
        style_dialog(advice_dialog, self)
        advice_dialog.exec_()
        self.status_bar.showMessage("就绪")

    def _change_history_page(self, delta):
        self._history_page += delta
        self._populate_history_table()

    def _populate_history_table(self):
        """Render at most 100 records and preserve explicit demo labels."""
        history, total, page = get_history_db().get_page(getattr(self, "_history_page", 0))
        self._history_page = page
        self._history_records_cache = history
        self.history_table.setUpdatesEnabled(False)
        try:
            self.history_table.clearSelection()
            self.history_table.setRowCount(len(history))
            self.history_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
            for row, record in enumerate(history):
                values = [record["timestamp"], os.path.basename(record["image_path"] or ""), record["disease_name"], f"{record['confidence']:.1%}", "查看详情"]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if col == 0:
                        item.setData(Qt.UserRole, record["record_id"])
                    self.history_table.setItem(row, col, item)
            self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        finally:
            self.history_table.setUpdatesEnabled(True)
        if hasattr(self, "history_page_label"):
            self.history_page_label.setText(f"第 {page+1} / {max(1,(total+99)//100)} 页 · 共 {total} 条 · 最新在前")
            self.history_previous.setEnabled(page > 0)
            self.history_next.setEnabled((page+1)*100 < total)
        self.status_bar.showMessage(f"历史记录：本页 {len(history)} 条，共 {total} 条")

    def delete_selected_history(self):
        """删除选中的历史记录 (SQLite)"""
        # 直接从表格行中取出 record_id，避免索引映射错误
        record_ids_to_delete = set()
        for item in self.history_table.selectedItems():
            if item.column() == 0:  # 只处理第一列（时间戳列，存了 record_id）
                rid = item.data(Qt.UserRole)
                if rid:
                    record_ids_to_delete.add(rid)

        if not record_ids_to_delete:
            self.show_message_box("提示", "请先选择要删除的记录！")
            return

        reply = QMessageBox.question(
            self, "确认删除", f"确定要删除选中的{len(record_ids_to_delete)}条记录吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                db = get_history_db()
                deleted_count = 0
                for record_id in record_ids_to_delete:
                    try:
                        db.delete_by_record_id(record_id)
                        deleted_count += 1
                    except Exception as e:
                        print(f"[ERROR] 删除记录失败 (id={record_id}): {e}")
                self._populate_history_table()
                self.show_message_box("成功", f"已成功删除 {deleted_count} 条记录！")
            except Exception as e:
                self.show_message_box("错误", f"删除记录失败: {str(e)}")

    def clear_history(self):
        """清空所有历史记录"""
        # 确认清空
        reply = QMessageBox.question(
            self, "确认清空", "确定要清空所有历史记录吗？此操作不可恢复！",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                db = get_history_db()
                db.delete_all()
                self._populate_history_table()
                self.show_message_box("成功", "所有历史记录已清空！")
            except Exception as e:
                self.show_message_box("错误", f"清空历史记录失败: {str(e)}")

    def _generate_demo_data(self):
        return get_history_db().generate_demo_data()

    def show_trend_analysis(self):
        plt, FigureCanvas = load_plotting()
        """显示病情趋势分析"""
        date_count, disease_count, date_disease = get_history_db().trend_counts()

        if not date_count:
            self.show_message_box("提示", "没有有效的历史记录数据。")
            return

        # 创建对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("分类记录统计（排除演示数据，不代表病情变化）")
        dialog.resize(1000, 800)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {self.background_color};
                color: {self.text_color};
            }}
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        # 顶部栏：添加"全屏/退出全屏"切换与快捷键
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        
        # 全屏/退出全屏按钮
        dialog_fullscreen_btn = QPushButton("🔍 全屏显示")
        dialog_fullscreen_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.highlight_color};
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13pt;
                font-weight: bold;
                margin-right: 10px;
            }}
            QPushButton:hover {{
                background-color: #d53f8c;
            }}
        """)

        def toggle_dialog_fullscreen():
            if dialog.isFullScreen():
                dialog.showNormal()
                dialog_fullscreen_btn.setText("🔍 全屏显示")
                dialog_fullscreen_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {self.highlight_color};
                        color: white;
                        padding: 8px 16px;
                        border-radius: 6px;
                        font-size: 13pt;
                        font-weight: bold;
                        margin-right: 10px;
                    }}
                    QPushButton:hover {{
                        background-color: #d53f8c;
                    }}
                """)
            else:
                dialog.showFullScreen()
                dialog_fullscreen_btn.setText("📱 退出全屏")
                dialog_fullscreen_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: #e53e3e;
                        color: white;
                        padding: 8px 16px;
                        border-radius: 6px;
                        font-size: 13pt;
                        font-weight: bold;
                        margin-right: 10px;
                    }}
                    QPushButton:hover {{
                        background-color: #c53030;
                    }}
                """)

        dialog_fullscreen_btn.clicked.connect(toggle_dialog_fullscreen)
        
        # 快捷键支持
        QShortcut(QKeySequence("F11"), dialog, activated=toggle_dialog_fullscreen)
        QShortcut(QKeySequence("Esc"), dialog, activated=lambda: (dialog.showNormal(), dialog_fullscreen_btn.setText("🔍 全屏显示")))
        
        top_bar.addWidget(dialog_fullscreen_btn)
        layout.addLayout(top_bar)

        # 创建标签页
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {self.accent_color};
                border-radius: 5px;
            }}
            QTabBar::tab {{
                background-color: {self.secondary_bg};
                color: {self.text_color};
                padding: 8px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.accent_color};
            }}
            QTabBar::tab:hover {{
                background-color: #5a6478;
            }}
        """)

        # 每日检测数量趋势标签页
        daily_tab = QWidget()
        daily_layout = QVBoxLayout(daily_tab)

        # 日期趋势折线图
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        fig1.set_facecolor('#242C37')
        ax1.set_facecolor('#242C37')

        dates = sorted(date_count.keys())
        values = [date_count[d] for d in dates]

        ax1.plot(dates, values, marker='o', color='#4DD6B4', linewidth=3.5, markersize=10,
                 markeredgecolor='white', markeredgewidth=1.5)
        ax1.fill_between(range(len(dates)), values, alpha=0.1, color='#4DD6B4')
        ax1.set_title('每日检测数量趋势', color='white', pad=20, fontsize=14, fontweight='bold')
        ax1.set_xlabel('日期', color='white')
        ax1.set_ylabel('检测数量', color='white')
        ax1.tick_params(axis='x', rotation=45, colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.grid(color='#4a5568', linestyle='--', linewidth=0.5, alpha=0.7)

        for spine in ax1.spines.values():
            spine.set_edgecolor('#4a5568')

        fig1.tight_layout(pad=1.0)
        canvas1 = FigureCanvas(fig1)
        canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        daily_layout.addWidget(canvas1, 1)
        canvas1.draw()

        tab_widget.addTab(daily_tab, "每日趋势")

        # 疾病分布标签页
        disease_tab = QWidget()
        disease_layout = QVBoxLayout(disease_tab)

        # 疾病分布饼图
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 5))
        fig2.set_facecolor('#242C37')

        # 饼图
        ax2.set_facecolor('#242C37')
        theme_colors = ['#4DD6B4', '#51C5E3', '#F1B761', '#A7C98B', '#B7A1D8', '#E58C89', '#8BC4CB', '#CBB394']
        wedges, texts, autotexts = ax2.pie(
            list(disease_count.values()),
            labels=list(disease_count.keys()),
            autopct='%1.1f%%',
            startangle=140,
            colors=theme_colors[:len(disease_count)],
            textprops={'color': 'white', 'fontweight': 'bold'},
            wedgeprops={'edgecolor': '#242C37', 'linewidth': 1.5}
        )
        ax2.set_title('疾病分布比例', color='white', pad=20, fontsize=14, fontweight='bold')

        # 设置文本颜色
        for text in texts:
            text.set_color('white')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # 柱状图
        ax3.set_facecolor('#242C37')
        colors_bar = theme_colors[:len(disease_count)]
        bars = ax3.bar(list(disease_count.keys()), list(disease_count.values()), color=colors_bar)
        for bar in bars:
            bar.set_edgecolor('white')
            bar.set_linewidth(0.5)
        ax3.set_title('疾病分布数量', color='white', pad=20, fontsize=14, fontweight='bold')
        ax3.set_ylabel('数量', color='white')
        ax3.tick_params(axis='x', rotation=45, colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.grid(color='#4a5568', linestyle='--', linewidth=0.5, axis='y', alpha=0.7)

        for spine in ax3.spines.values():
            spine.set_edgecolor('#4a5568')

        fig2.tight_layout(pad=1.0)
        canvas2 = FigureCanvas(fig2)
        canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        disease_layout.addWidget(canvas2, 1)
        canvas2.draw_idle()

        tab_widget.addTab(disease_tab, "疾病分布")

        # 疾病趋势标签页
        trend_tab = QWidget()
        trend_layout = QVBoxLayout(trend_tab)

        # 疾病趋势折线图
        fig3, ax4 = plt.subplots(figsize=(10, 5))
        fig3.set_facecolor('#242C37')
        ax4.set_facecolor('#242C37')

        # 准备数据
        diseases = sorted(list(disease_count.keys()))
        dates_sorted = sorted(date_disease.keys())

        # 每种疾病一个线条
        theme_colors = ['#4DD6B4', '#51C5E3', '#F1B761', '#A7C98B', '#B7A1D8', '#E58C89', '#8BC4CB', '#CBB394']

        for i, disease in enumerate(diseases):
            values = []
            for date in dates_sorted:
                values.append(date_disease[date].get(disease, 0))

            ax4.plot(dates_sorted, values, marker='o', color=theme_colors[i % len(theme_colors)],
                     label=disease, linewidth=3.5, markersize=8,
                     markeredgecolor='white', markeredgewidth=1.5)

        ax4.set_title('各疾病每日趋势', color='white', pad=20)
        ax4.set_xlabel('日期', color='white')
        ax4.set_ylabel('检测数量', color='white')
        ax4.tick_params(axis='x', rotation=45, colors='white')
        ax4.tick_params(axis='y', colors='white')
        ax4.grid(color='#4a5568', linestyle='--', linewidth=0.5)
        ax4.legend(facecolor='#242C37', edgecolor='#242C37', labelcolor='white')

        for spine in ax4.spines.values():
            spine.set_edgecolor('#4a5568')

        fig3.tight_layout(pad=1.0)
        canvas3 = FigureCanvas(fig3)
        canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        trend_layout.addWidget(canvas3, 1)
        canvas3.draw_idle()

        tab_widget.addTab(trend_tab, "疾病趋势")

        layout.addWidget(tab_widget)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.accent_color};
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                margin-top: 10px;
            }}
            QPushButton:hover {{
                background-color: #0097B2;
            }}
        """)
        close_btn.clicked.connect(dialog.accept)

        # 生成演示数据按钮
        demo_btn = QPushButton("📊 生成30天演示数据")
        demo_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.highlight_color};
                color: white;
                padding: 8px 16px;
                border-radius: 6px;
                margin-top: 6px;
            }}
            QPushButton:hover {{ background-color: #d69e2e; }}
        """)
        def generate_demo():
            reply = QMessageBox.question(
                dialog, "生成演示数据",
                "将在数据库中生成约 5000 条模拟的 30 天检测记录。\n"
                "已有的演示数据会被清除并重新生成。\n\n确定继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    count = self._generate_demo_data()
                    QMessageBox.information(dialog, "成功",
                        f"已生成 {count} 条演示数据！\n弹窗将自动关闭，演示记录可在历史页查看，不计入真实分类统计。")
                    dialog.accept()
                except Exception as ex:
                    QMessageBox.critical(dialog, "生成失败", f"错误: {ex}")
        demo_btn.clicked.connect(generate_demo)

        btn_row = QHBoxLayout()
        export_button=QPushButton("导出统计图与数据")
        export_button.clicked.connect(lambda: self.export_analysis(
            "history", [fig1,fig2,fig3], ["日期","类别","数量"],
            [(day,disease,count) for day,values in sorted(date_disease.items()) for disease,count in sorted(values.items())]))
        btn_row.addWidget(export_button)
        btn_row.addWidget(demo_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        style_dialog(dialog, self)
        try:
            dialog.exec_()
        finally:
            for figure in (fig1, fig2, fig3):
                plt.close(figure)

    # ------------------------------------------------------------------
    #  结果展示 & AI 建议
    # ------------------------------------------------------------------
    def parse_and_show_results(self, results):
        if self.detector is None or results is None or len(results) != 1:
            self._classification_failed("没有有效的单图分类结果")
            return False
        outcome = self.detector.parse_result(results[0])
        if not outcome.success:
            self._classification_failed(outcome.error)
            return False
        self.current_results = results
        self.current_disease = outcome.disease_name
        self.current_confidence = outcome.confidence
        self._update_classification_summary()
        self.all_classes_confidence = outcome.probabilities
        self.result_processor.current_disease = outcome.disease_name
        self.result_processor.current_confidence = outcome.confidence
        update_probability_panel(self, outcome.probabilities)
        return True

    def display_results(self):
        update_probability_panel(self, getattr(self, "all_classes_confidence", {}))

    def show_disease_result(self, disease_name, confidence, image_path=None, image=None):
        """显示疾病检测结果,采用全新现代医疗UI"""
        if not valid_prediction(disease_name, confidence):
            self._classification_failed("没有有效分类结果")
            return

        if image is None:
            image = self.current_image

        dialog = QDialog(self)
        dialog.setWindowTitle("疾病分类结果报告")
        dialog.setMinimumSize(750, 600)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {self.background_color};
                color: {self.text_color};
            }}
            QPushButton {{
                background-color: {self.accent_color};
                color: white;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13pt;
            }}
            QPushButton:hover {{
                background-color: #0097B2;
            }}
            QTableWidget {{
                background-color: {self.secondary_bg};
                color: {self.text_color};
                border: 1px solid #3b4252;
                border-radius: 6px;
                gridline-color: #2c323c;
                outline: none;
            }}
            QTableWidget::item:focus {{
                outline: none;
            }}
            QTableWidget::item {{
                padding: 8px;
                border-bottom: 1px solid #2c323c;
            }}
            QTableWidget::item:selected {{
                background-color: rgba(128, 90, 213, 0.35);
                color: white;
            }}
            QTableWidget::item:selected:!active {{
                background-color: rgba(128, 90, 213, 0.25);
                color: #E5E9F0;
            }}
            QTableWidget::item:alternate {{
                background-color: #262B33;
            }}
            QTableWidget::item:alternate:selected {{
                background-color: rgba(128, 90, 213, 0.35);
                color: white;
            }}
            QHeaderView::section {{
                background-color: {self.primary_color};
                color: #81A1C1;
                padding: 10px;
                border: none;
                border-bottom: 2px solid {self.highlight_color};
                font-weight: bold;
            }}
        """)

        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(25, 25, 25, 25)

        # 顶部标题栏
        title_layout = QHBoxLayout()
        title_label = QLabel("📊 影像学分类结果")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setStyleSheet(f"color: {self.accent_color}; margin-bottom: 10px;")
        title_layout.addWidget(title_label, 1)

        # 全屏按钮
        result_fullscreen_btn = QPushButton("🔍 全屏")
        result_fullscreen_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.highlight_color};
                color: white;
                padding: 6px 14px;
                border-radius: 4px;
                font-size: 12pt;
            }}
            QPushButton:hover {{ background-color: #d69e2e; }}
        """)
        def toggle_result_fullscreen():
            if dialog.isFullScreen():
                dialog.showNormal()
                result_fullscreen_btn.setText("🔍 全屏")
            else:
                dialog.showFullScreen()
                result_fullscreen_btn.setText("📱 退出全屏")
        result_fullscreen_btn.clicked.connect(toggle_result_fullscreen)
        QShortcut(QKeySequence("F11"), dialog, activated=toggle_result_fullscreen)
        title_layout.addWidget(result_fullscreen_btn)
        main_layout.addLayout(title_layout)

        content_layout = QHBoxLayout()

        # 左侧：影像展示
        image_group = QGroupBox("原片影像")
        image_group.setStyleSheet(f"QGroupBox {{ border: 1px solid #3b4252; border-top: 3px solid {self.accent_color}; border-radius: 6px; padding-top: 20px; }} QGroupBox::title {{ color: {self.accent_color}; top: -5px; left: 10px; }}")
        image_layout = QVBoxLayout(image_group)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(350, 350)
        image_label.setStyleSheet(f"background-color: #1a1e24; border-radius: 4px;")

        if image is not None:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            q_img = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            image_label.setPixmap(pixmap.scaled(image_label.width(), image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            image_label.setText("影像加载失败")

        image_layout.addWidget(image_label)
        content_layout.addWidget(image_group, 1)

        # 右侧：数据与置信度
        info_group = QGroupBox("AI 诊断指标")
        info_group.setStyleSheet(f"QGroupBox {{ border: 1px solid #3b4252; border-top: 3px solid {self.highlight_color}; border-radius: 6px; padding-top: 20px; }} QGroupBox::title {{ color: {self.highlight_color}; top: -5px; left: 10px; }}")
        info_layout = QVBoxLayout(info_group)

        result_text = f"""
        <div style='text-align:center; padding:10px;'>
            <p style='font-size:15pt; color:#A0AEC0; margin-bottom:5px;'>首选诊断 (Top-1)</p>
            <p style='font-size:21pt; font-weight:bold; color:{self.highlight_color}; margin:0;'>{disease_name}</p>
            <p style='font-size:13pt; margin-top:10px;'>模型置信度: <span style='color:{self.highlight_color}; font-weight:bold;'>{confidence:.2%}</span></p>
        </div>
        """
        text_label = QLabel(result_text)
        info_layout.addWidget(text_label)

        if hasattr(self, 'all_classes_confidence') and self.all_classes_confidence:
            classes_table = QTableWidget()
            classes_table.setRowCount(len(self.all_classes_confidence))
            classes_table.setColumnCount(2)
            classes_table.setHorizontalHeaderLabels(["病种分类", "置信度"])
            classes_table.verticalHeader().setVisible(False)
            classes_table.setAlternatingRowColors(True)

            for row, (class_name, conf) in enumerate(self.all_classes_confidence.items()):
                name_item = QTableWidgetItem(class_name)
                conf_item = QTableWidgetItem(f"{conf:.4f}")
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                conf_item.setFlags(conf_item.flags() & ~Qt.ItemIsEditable)

                if class_name == disease_name:
                    name_item.setForeground(QBrush(QColor(self.highlight_color)))
                    conf_item.setForeground(QBrush(QColor(self.highlight_color)))
                    font = name_item.font()
                    font.setBold(True)
                    name_item.setFont(font)
                    conf_item.setFont(font)

                classes_table.setItem(row, 0, name_item)
                classes_table.setItem(row, 1, conf_item)

            classes_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            classes_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            info_layout.addWidget(classes_table)

        content_layout.addWidget(info_group, 1)
        main_layout.addLayout(content_layout)

        # 底部操作区
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("完成阅片 (Esc)")
        close_btn.setStyleSheet(f"background-color: transparent; border: 1px solid #4C566A; color: {self.text_color};")
        close_btn.clicked.connect(dialog.accept)

        report_btn = QPushButton("生成 DeepSeek 报告")
        report_btn.setStyleSheet(f"background-color: {self.highlight_color}; color: white;")
        report_btn.clicked.connect(lambda: [dialog.accept(), self.advice_button.click()])

        button_layout.addWidget(close_btn)
        button_layout.addWidget(report_btn)
        main_layout.addLayout(button_layout)

        def esc_result():
            if dialog.isFullScreen():
                dialog.showNormal()
                result_fullscreen_btn.setText("🔍 全屏")
            else:
                dialog.accept()
        QShortcut(QKeySequence("Esc"), dialog, activated=esc_result)
        style_dialog(dialog, self)
        dialog.exec_()
    

    def show_results(self):
        if valid_prediction(getattr(self,"current_disease",None),getattr(self,"current_confidence",0)):
            self.right_tab_widget.setCurrentIndex(0)
            self.result_card.setFocus()
            self.status_bar.showMessage("八类概率已在右侧完整展示")
        else:
            self.show_message_box("提示", "请先完成分类")

    def display_image(self, image, label):
        """显示图像到指定的QLabel"""
        if image is not None:
            # 转换颜色空间从BGR到RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(
                pixmap.scaled(
                    label.width(), label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        else:
            label.setText("无法显示图像")

    def load_saved_api_key(self):
        """从文件加载保存的API密钥"""
        try:
            api_key_file = str(PROJECT_ROOT / "saved_api_key.txt")
            if os.path.exists(api_key_file):
                with open(api_key_file, 'r', encoding='utf-8') as f:
                    saved_data = f.read().strip()
                    # 尝试Base64解码, 兼容旧版明文格式
                    from src.core.credentials import decode_key
                    saved_key = decode_key(saved_data)
                    if saved_key and saved_key.startswith('sk-'):
                        # 确保deepseek_api已初始化
                        if self.deepseek_api is None:
                            self.deepseek_api = DeepSeekAPI()
                        self.deepseek_api.set_api_key(saved_key)

                        # 在输入框中显示密钥
                        if hasattr(self, 'api_key_input'):
                            self.api_key_input.setText(saved_key)

                        print("已加载保存的API密钥")
                        self.status_bar.showMessage("已自动加载保存的API密钥")
        except Exception as e:
            print(f"加载API密钥时出错: {e}")

    def toggle_api_usage(self):
        """切换API使用状态"""
        if self.use_api_checkbox.isChecked():
            self.use_api_checkbox.setText("启用DeepSeek API（推荐）")
            self.status_bar.showMessage("已启用DeepSeek API")
        else:
            self.use_api_checkbox.setText("禁用DeepSeek API")
            self.status_bar.showMessage("已禁用DeepSeek API,将使用默认建议")

    def show_ai_advice(self):
        disease, confidence = getattr(self, 'current_disease', None), getattr(self, 'current_confidence', None)
        if not valid_prediction(disease, confidence):
            self.show_message_box("没有有效分类", "请先完成有效图像分类")
            return
        if self.deepseek_api is None:
            self.deepseek_api = DeepSeekAPI(config=APP_CONFIG)
        client = self.deepseek_api
        use_online = self.use_api_checkbox.isChecked() and bool(client.api_key)
        self._report_generation = getattr(self, '_report_generation', 0) + 1
        generation = self._report_generation
        signature = (disease, confidence, id(self.current_results))
        self.advice_text.setPlainText(f"正在为分类结果{disease}生成辅助参考说明...")
        self.advice_button.setEnabled(False)
        self.status_bar.showMessage("正在生成辅助说明，界面仍可操作")
        def fetch():
            try:
                result = client.request_treatment_advice(disease, confidence) if use_online else AdviceResult(
                    False, client._get_default_advice(disease), 'local_reference', 'online_disabled_or_missing_key')
            except Exception as error:
                result = AdviceResult(False, f"辅助建议请求失败：{error}", 'error', str(error))
            self.report_ready.emit({'generation': generation, 'signature': signature, 'advice': result})
        threading.Thread(target=fetch, daemon=True).start()

    def _apply_report_reply(self, payload):
        signature = (getattr(self, 'current_disease', None), getattr(self, 'current_confidence', None), id(self.current_results))
        if payload['generation'] != getattr(self, '_report_generation', None) or payload['signature'] != signature:
            return False
        if not valid_prediction(signature[0], signature[1]):
            return False
        result = payload['advice']
        self.advice_text.setHtml(self.format_advice_html(result.content))
        self.advice_button.setEnabled(True)
        status = 'DeepSeek辅助说明已返回' if result.success else '在线服务未使用或失败；请查看标明来源的参考说明'
        self.status_bar.showMessage(status)
        return True

    def show_fullscreen_advice(self):
        """全屏显示DeepSeek AI诊疗建议"""
        dialog = QDialog(self)
        dialog.setWindowTitle("DeepSeek AI 智能诊疗建议 - 全屏模式")
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {self.background_color};
                color: {self.text_color};
            }}
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(30, 30, 30, 30)

        # 复制当前的文本内容
        fullscreen_text = QTextEdit()
        fullscreen_text.setReadOnly(True)
        fullscreen_text.setHtml(self.advice_text.toHtml())
        fullscreen_text.setStyleSheet(self.advice_text.styleSheet())

        reading_bar=QHBoxLayout();reading_bar.addStretch();reading_bar.addWidget(QLabel('阅读字号'))
        reading_size=QComboBox()
        for value in (12,14,16,18,20,22,24,28):reading_size.addItem(str(value),value)
        initial=self.reading_size_combo.currentData() if hasattr(self,'reading_size_combo') else 14
        reading_size.setCurrentIndex(reading_size.findData(initial))
        def resize_reading():
            from src.visualization.rich_text import apply_reading_font
            apply_reading_font(fullscreen_text,reading_size.currentData())
        reading_size.currentIndexChanged.connect(resize_reading)
        reading_bar.addWidget(reading_size);layout.addLayout(reading_bar)
        resize_reading()
        layout.addWidget(fullscreen_text)

        # 退出全屏按钮
        close_btn = QPushButton("退出全屏 (Esc)")
        close_btn.setMinimumSize(160, 50)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.accent_color};
                color: white;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13pt;
            }}
            QPushButton:hover {{
                background-color: #0097B2;
            }}
        """)
        close_btn.clicked.connect(dialog.accept)
        close_btn.setCursor(QCursor(Qt.PointingHandCursor))

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        # 绑定 Esc 快捷键直接退出
        QShortcut(QKeySequence("Esc"), dialog, activated=dialog.accept)

        # 以最大化模式显示对话框
        dialog.showMaximized()
        style_dialog(dialog, self)
        dialog.exec_()

    def format_advice_html(self, markdown_text):
        from src.visualization.rich_text import render_text
        return render_text(markdown_text)

    def init_speech_components(self):
        """初始化智能语音识别组件"""
        try:
            print("[DEBUG] 开始初始化智能语音组件...")
            
            # 连接智能语音信号
            self.connect_smart_voice_signals()
            print("[DEBUG] 智能语音信号连接成功")
            
            # 启动网络监控（若已创建）
            if self.voice_manager is not None:
                self.voice_manager.start_network_monitoring()
                print("[DEBUG] 网络监控启动成功")
            
            # 初始化传统语音组件作为备用
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = self.get_best_microphone()
                print("[DEBUG] 传统语音组件初始化成功")
            except Exception as e:
                print(f"[WARNING] 传统语音组件初始化失败: {e}")
            
            # 初始化TTS引擎
            try:
                self.tts_engine = pyttsx3.init()
                voices = self.tts_engine.getProperty('voices')
                
                # 尝试设置中文语音
                for voice in voices:
                    if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('volume', 0.8)
                print("[DEBUG] TTS引擎初始化成功")
                
            except ImportError:
                print("[WARNING] pyttsx3未安装,TTS功能将不可用")
                self.tts_engine = None
            
            # 后台校准麦克风
            threading.Thread(target=self.calibrate_microphone, daemon=True).start()
            
            print("[DEBUG] 智能语音组件初始化完成")
            
        except Exception as e:
            print(f"[ERROR] 智能语音组件初始化失败: {e}")
            if hasattr(self, 'voice_chat_enabled'):
                self.voice_chat_enabled.setEnabled(False)
                self.voice_chat_enabled.setToolTip("语音功能不可用,请检查网络连接和API配置")

    def get_best_microphone(self):
        """获取最佳可用麦克风"""
        try:
            # 列出所有音频设备
            import pyaudio
            p = pyaudio.PyAudio()
            
            print("[DEBUG] 可用音频设备:")
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"[DEBUG]   设备 {i}: {info['name']} (采样率: {info['defaultSampleRate']})")
            
            p.terminate()
            
            # 尝试使用默认麦克风
            mic = sr.Microphone()
            
            # 测试麦克风是否可用
            with mic as source:
                print("[DEBUG] 测试麦克风访问...")
                # 简单测试,不调整噪音以节省时间
                pass
            
            print("[DEBUG] 默认麦克风测试成功")
            return mic
            
        except Exception as e:
            print(f"[WARNING] 麦克风测试失败: {e}")
            # 即使测试失败,也返回默认麦克风,让用户自己尝试
            return sr.Microphone()

    def test_microphone(self):
        """Open/calibrate the microphone off the GUI thread."""
        if not APP_CONFIG.enable_voice or sr is None:
            self.show_message_box("语音未启用", "请在系统设置中启用语音，并安装语音依赖后重启。")
            return
        if self.voice_manager is not None and self.voice_manager.is_recording:
            self.show_message_box("正在录音", "请先结束语音输入，再测试麦克风。")
            return
        def probe(cancel, progress):
            try:
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                return True, "麦克风访问与环境噪声采样成功。语音识别效果需要实际录音验证。"
            except Exception as error:
                return False, f"麦克风测试失败：{error}。请检查设备、权限与占用情况。"
        def finished(result):
            success, message = result
            self.task_status_label.setText("麦克风测试完成" if success else "麦克风不可用")
            self.show_message_box("麦克风测试", message,
                                  QMessageBox.Information if success else QMessageBox.Warning)
        self._start_job(probe, finished, "正在后台测试麦克风…")

    def calibrate_microphone(self):
        """校准麦克风（后台线程）"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            print(f"麦克风校准失败: {e}")

    def toggle_voice_chat(self, state):
        """切换语音播报开关：仅控制是否朗读AI回复，不影响麦克风输入"""
        enabled = state == Qt.Checked
        if enabled:
            self.status_bar.showMessage("🔊 AI 语音朗读已开启（不影响麦克风输入）")
        else:
            self.status_bar.showMessage("🔇 AI 语音朗读已关闭（仍可使用麦克风提问）")

    def update_duration_value(self, value):
        """更新语音识别时长值显示"""
        self.duration_value_label.setText(f"{value}秒")
        # 如果语音管理器已初始化,则更新其时长设置
        if self.voice_manager:
            self.voice_manager.set_recognition_duration(value)
            print(f"[DEBUG] 💾 已更新SmartVoiceManager的识别时长为 {value} 秒")
        else:
            print(f"[DEBUG] ⚠️ SmartVoiceManager尚未初始化,时长设置将在初始化后应用")

    def start_voice_input(self):
        """开始智能语音输入或取消录音"""
        # 检查语音组件是否已初始化
        if self.voice_manager is None:
            self.show_message_box("提示", "语音组件正在初始化中,请稍后再试", QMessageBox.Information)
            return
            
        # 检查是否正在录音,如果是则取消
        if self.voice_manager.is_recording:
            print("[DEBUG] 🛑 取消当前录音")
            self.voice_manager.cancel_recording()
            return
            
        # 获取当前设置的时长
        duration = self.duration_slider.value()
        print(f"[DEBUG] 🎤 开始智能语音识别,时长设置: {duration}秒")
        
        # 确保语音管理器使用最新的时长设置
        self.voice_manager.set_recognition_duration(duration)
        print(f"[DEBUG] 🔧 已确保SmartVoiceManager使用最新时长: {duration}秒")
        
        # 启动语音识别（传入时长参数以确保生效）
        self.voice_manager.start_voice_recognition(duration)

    def perform_voice_recognition(self):
        """执行语音识别（备用方法,与主识别器策略一致）"""
        try:
            print("[DEBUG] 🎤 开始备用语音识别...")
            
            # 发送状态信号
            QApplication.postEvent(self, VoiceRecognitionEvent("completed", "__RECORDING_START__"))
            
            # 录音
            with self.microphone as source:
                print("[DEBUG] 🔧 调整环境噪音...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                print("[DEBUG] 🗣️ 请说话（点击按钮停止录音）...")
                
                QApplication.postEvent(self, VoiceRecognitionEvent("completed", "__RECORDING__"))
                
                # 用户自定义录音时长,获取当前滑块设置的时长
                current_duration = self.duration_slider.value() if hasattr(self, 'duration_slider') else 10
                recognition_duration = self.voice_manager.recognition_duration if self.voice_manager else current_duration
                print(f"[DEBUG] 🕒 备用识别器使用时长: {recognition_duration}秒")
                audio = self.recognizer.listen(source, timeout=30, phrase_time_limit=recognition_duration)
                print(f"[DEBUG] ✅ 录音完成,音频长度: {len(audio.frame_data)} bytes")
                
            # 发送处理中状态
            QApplication.postEvent(self, VoiceRecognitionEvent("completed", "__PROCESSING__"))
            
            text = None
            recognition_method = ""

            # 优先策略：本地Vosk识别（如果可用）
            if hasattr(self, 'voice_manager') and hasattr(self.voice_manager, 'vosk_model') and self.voice_manager.vosk_model:
                try:
                    print("[DEBUG] 🏠 使用本地Vosk中文识别...")
                    from vosk import KaldiRecognizer
                    rec = KaldiRecognizer(self.voice_manager.vosk_model, audio.sample_rate or 16000)
                    frame = audio.frame_data
                    chunk_size = 4000
                    for i in range(0, len(frame), chunk_size):
                        rec.AcceptWaveform(frame[i:i+chunk_size])
                    result_json = rec.FinalResult()
                    result_obj = self.voice_manager._vosk_json.loads(result_json)
                    text = (result_obj.get("text") or "").strip()
                    recognition_method = "Vosk 本地识别"
                    print(f"[DEBUG] ✅ Vosk识别结果: '{text}'")
                except Exception as e:
                    print(f"[DEBUG] ❌ Vosk识别失败: {e}")

            # 备用策略：Google API（优化参数）
            if not text or len(text) < 2:
                try:
                    print("[DEBUG] 🌐 尝试Google API中文识别（优化参数）...")
                    text = self.recognizer.recognize_google(audio, language='zh-CN', show_all=False)
                    recognition_method = "Google API (中文优化)"
                    print(f"[DEBUG] ✅ Google API识别成功: '{text}'")
                except sr.RequestError as e:
                    print(f"[DEBUG] ❌ Google API网络错误: {e}")
                except sr.UnknownValueError:
                    print("[DEBUG] ❌ Google API无法识别语音内容")
                    # 尝试英文识别
                    try:
                        print("[DEBUG] 🌐 尝试Google API英文识别...")
                        text = self.recognizer.recognize_google(audio, language='en-US')
                        recognition_method = "Google API (英文)"
                        print(f"[DEBUG] ✅ Google API英文识别: '{text}'")
                    except Exception as e2:
                        print(f"[DEBUG] ❌ Google API英文识别失败: {e2}")
                except Exception as e:
                    print(f"[DEBUG] ❌ Google API其他错误: {e}")

            # 清理和发送结果
            if text and text.strip() and len(text.strip()) > 0:
                final_text = text.strip()
                print(f"[INFO] 🎉 备用语音识别成功 ({recognition_method}): '{final_text}'")
                QApplication.postEvent(self, VoiceRecognitionEvent("completed", final_text))
                return
            else:
                print("[DEBUG] ❌ 所有识别方法都未获得有效结果")
                QApplication.postEvent(self, VoiceRecognitionEvent("unknown"))
                
        except sr.WaitTimeoutError:
            print("[DEBUG] ⏱️ 录音超时")
            QApplication.postEvent(self, VoiceRecognitionEvent("timeout"))
        except sr.UnknownValueError:
            print("[DEBUG] ❓ 无法识别语音内容")
            QApplication.postEvent(self, VoiceRecognitionEvent("unknown"))
        except sr.RequestError as e:
            print(f"[DEBUG] ❌ 语音服务请求错误: {e}")
            QApplication.postEvent(self, VoiceRecognitionEvent("error", f"语音服务错误: {str(e)}"))
        except Exception as e:
            print(f"[DEBUG] ❌ 语音识别异常: {e}")
            QApplication.postEvent(self, VoiceRecognitionEvent("error", f"识别异常: {str(e)}"))

    def speak_text(self, text):
        try:
            from src.visualization.speech_player import SpeechPlayer
            if not hasattr(self,'_speech_player'):self._speech_player=SpeechPlayer(self)
            self._speech_player.speak(text)
        except ImportError:
            self.status_bar.showMessage('当前环境缺少 Qt 多媒体组件，无法在应用内播放语音')

    def _summarize_for_tts(self, markdown_text):
        """从 AI 回复中提取简短摘要（≤100字），用于语音播报"""
        import re
        # 先提取纯文本
        text = markdown_text
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'[-*•`]\s*', '', text)
        text = re.sub(r'\d+\.\s+', '', text)
        text = re.sub(r'\n+', '，', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # 取前100个字符作为摘要
        if len(text) > 100:
            # 尝试在句号处截断
            cutoff = text[:100].rfind('。')
            if cutoff == -1:
                cutoff = text[:100].rfind('，')
            if cutoff == -1 or cutoff < 40:
                cutoff = 99
            text = text[:cutoff+1] + "具体建议请查看界面文字。"
        return text

    def _extract_plain_text(self, markdown_text):
        """从 Markdown/HTML 中提取纯文本（用于 TTS 播报）"""
        import re
        text = markdown_text
        # 去除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除 Markdown 标记符号
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'[-*•]\s+', '', text)
        text = re.sub(r'\d+\.\s+', '', text)
        text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
        text = re.sub(r'---+', '', text)
        text = re.sub(r'\n+', '\n', text)
        text = text.strip()
        # 限制长度避免播放过久
        if len(text) > 500:
            text = text[:500] + "..."
        return text

    def _clean_text_for_tts(self, text):
        """清理文本,使其适合TTS播放"""
        import re
        
        # 去除HTML标签
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        # 去除Markdown格式
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)  # 加粗
        clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)      # 斜体
        clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)        # 代码
        clean_text = re.sub(r'#+\s*', '', clean_text)               # 标题
        clean_text = re.sub(r'-\s+', '', clean_text)                # 列表项
        clean_text = re.sub(r'\d+\.\s+', '', clean_text)            # 数字列表
        
        # 去除多余的空白字符
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = clean_text.strip()
        
        # 限制长度,避免过长的语音播放
        if len(clean_text) > 300:
            clean_text = clean_text[:300] + "..."
            
        return clean_text

    def update_ai_progress(self, value, text=""):
        """更新AI回复进度条"""
        # 确保value是整数类型
        try:
            progress_value = int(value)
        except (ValueError, TypeError):
            progress_value = 0
            
        self.ai_progress_bar.setValue(progress_value)
        if text:
            self.ai_progress_bar.setFormat(f"{text} - {progress_value}%")
        else:
            self.ai_progress_bar.setFormat(f"{progress_value}%")

    def show_ai_progress(self, show=True):
        """显示或隐藏AI进度条"""
        self.ai_progress_bar.setVisible(show)
        if show:
            self.ai_progress_bar.setValue(0)


    def send_chat_message_with_progress(self):
        from src.visualization.conversation import send
        send(self, APP_CONFIG)

    def process_ai_response(self):
        try:
            message = getattr(self, '_last_user_question', '')
            if not message:
                QApplication.postEvent(self, AIResponseEvent("error", "请输入有效问题"))
                return
            if self.deepseek_api is None:
                self.deepseek_api = DeepSeekAPI(config=APP_CONFIG)
            key = getattr(self, '_chat_api_key', '')
            if key and key != self.deepseek_api.api_key:
                self.deepseek_api.set_api_key(key)
            disease = getattr(self, 'current_disease', None)
            confidence = getattr(self, 'current_confidence', None)
            if valid_prediction(disease, confidence):
                message += f"\n[图像分类上下文，非临床诊断] {disease}，概率{confidence:.4f}"
            QApplication.postEvent(self, AIResponseEvent("progress", "正在请求DeepSeek...", 50))
            advice_result = self.deepseek_api.request_custom_advice(message)
            response = advice_result.content
            if not advice_result.success:
                QApplication.postEvent(self, AIResponseEvent("error", response))
                return
            summary = self._summarize_for_tts(response)
            QApplication.postEvent(self, AIResponseEvent("completed", response, tts=summary))
        except Exception as error:
            QApplication.postEvent(self, AIResponseEvent("error", f"AI调用失败：{error}"))

    def customEvent(self, event):
        """处理自定义事件"""
        if isinstance(event, VoiceRecognitionEvent):
            self.handle_voice_recognition_event(event)
        elif isinstance(event, AIResponseEvent):
            if event.event_type == "tts_ready":
                self._play_next_tts()
            else:
                self.handle_ai_response_event(event)
        else:
            super().customEvent(event)

    def handle_voice_recognition_event(self, event):
        """处理语音识别事件"""
        if event.event_type == "listening":
            self.voice_input_button.setText("🎤 正在录音...")
            self.voice_input_button.setEnabled(False)
            self.status_bar.showMessage("正在录音,请说话（点击按钮停止）...")
            
        elif event.event_type == "processing":
            self.voice_input_button.setText("🔄 处理中...")
            self.status_bar.showMessage("正在识别语音内容...")
            
        elif event.event_type == "completed":
            # 检查是否是状态信号
            if event.data == "__RECORDING_START__":
                self.voice_input_button.setText("🎙️ 准备录音...")
                self.voice_input_button.setEnabled(False)
                self.status_bar.showMessage("🔧 正在调整麦克风...")
                return
                
            elif event.data == "__RECORDING__":
                self.voice_input_button.setText("🛑 点击取消")
                self.voice_input_button.setEnabled(True)  # 启用按钮以允许取消
                self.voice_input_button.setStyleSheet("""
                    QPushButton {
                        background-color: #e53e3e;
                        color: white;
                        border: 2px solid #c53030;
                        border-radius: 6px;
                        padding: 8px 16px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #c53030;
                    }
                """)
                self.status_bar.showMessage("🎤 正在录音中,请清晰说话（点击按钮停止录音）...")
                return
                
            elif event.data == "__PROCESSING__":
                self.voice_input_button.setText("⏳ 识别中...")
                self.voice_input_button.setStyleSheet("""
                    QPushButton {
                        background-color: #0097B2;
                        color: white;
                        border: 2px solid #2c5aa0;
                        border-radius: 6px;
                        padding: 8px 16px;
                        font-weight: bold;
                    }
                """)
                self.status_bar.showMessage("🔍 正在识别语音内容...")
                return
            
            # 正常识别结果
            self.voice_input_button.setText("🎤 语音输入")
            self.voice_input_button.setEnabled(True)
            # 恢复按钮样式
            self.voice_input_button.setStyleSheet("""
                QPushButton {
                    background-color: #38a169;
                    color: white;
                    border: 2px solid #2f855a;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2f855a;
                }
                QPushButton:disabled {
                    background-color: #666;
                    color: #999;
                }
            """)
            
            # 将识别结果填入聊天框
            if hasattr(self, 'chat_input') and event.data:
                self.chat_input.setPlainText(event.data)
                self.status_bar.showMessage(f"✅ 语音识别成功：{event.data}")
                
                # 播放成功提示音（可选）
                try:
                    import winsound
                    winsound.MessageBeep(winsound.MB_OK)
                except:
                    pass
                
                # 自动发送给AI进行对话
                QTimer.singleShot(800, self.send_chat_message_with_progress)
                self.status_bar.showMessage(f"✅ 识别成功,即将自动发送给AI: {event.data}")
            
        elif event.event_type == "timeout":
            self.voice_input_button.setText("🎤 语音输入")
            self.voice_input_button.setEnabled(True)
            self._reset_button_style()
            self.status_bar.showMessage("⏱️ 录音超时,请重试（或点击按钮停止录音）")
            # 播放超时提示音
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except:
                pass
            
        elif event.event_type == "unknown":
            self.voice_input_button.setText("🎤 语音输入")
            self.voice_input_button.setEnabled(True)
            self._reset_button_style()
            self.status_bar.showMessage("❓ 无法识别语音内容,请重试")
            # 播放失败提示音
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONQUESTION)
            except:
                pass
            
        elif event.event_type == "error":
            self.voice_input_button.setText("🎤 语音输入")
            self.voice_input_button.setEnabled(True)
            self._reset_button_style()
            
            # 处理取消录音
            if "录音已取消" in str(event.data):
                self.status_bar.showMessage("🛑 录音已取消")
                # 播放取消提示音
                try:
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                except:
                    pass
            else:
                self.status_bar.showMessage("❌ 语音识别失败,请重试")
                
                # 只有在严重错误时才显示弹窗
                if "网络" in str(event.data) or "麦克风" in str(event.data):
                    self.show_message_box("语音识别错误", f"语音识别失败：{event.data}", QMessageBox.Warning)
    
    def _reset_button_style(self):
        """重置语音按钮样式"""
        self.voice_input_button.setStyleSheet("""
            QPushButton {
                background-color: #38a169;
                color: white;
                border: 2px solid #2f855a;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2f855a;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)

    def handle_ai_response_event(self, event):
        """处理AI回复事件——一问一答追加模式"""
        if event.event_type == "progress":
            self.update_ai_progress(event.progress, event.data)

        elif event.event_type == "completed":
            self.show_ai_progress(False)

            ai_msg = event.data

            self.chat_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "question": getattr(self, '_last_user_question', ''),
                "answer": ai_msg[:1000]
            })
            if len(self.chat_history) > 50:
                self.chat_history = self.chat_history[-50:]

            # 构建 AI 回复块
            ai_block = f"""
            <div style='margin-bottom: 20px; padding: 12px; background-color: #1E222A; border-radius: 8px; border-left: 3px solid #00B5D8;'>
                <div style='color: #00B5D8; font-weight: bold; margin-bottom: 6px;'>🩺 AI 回复 ({self.chat_history[-1]['timestamp']}):</div>
                <div style='color: #E5E9F0; padding-left: 10px; line-height: 1.6;'>
                    {self.format_advice_html(ai_msg)}
                </div>
            </div>
            """

            current_html = self.chat_display.toHtml()
            # 移除之前追加的 thinking 占位块
            marker = 'AI 正在回复...'
            clean = current_html
            if marker in current_html:
                idx = current_html.find(marker)
                if idx > 0:
                    start = current_html.rfind("<div style='margin-bottom:", 0, idx)
                    if start >= 0 and start < idx:
                        end1 = current_html.find("</div>", idx)
                        if end1 >= 0:
                            end2 = current_html.find("</div>", end1 + 6)
                            if end2 >= 0:
                                clean = current_html[:start] + current_html[end2 + 6:]

            if len(clean) < 500:
                wrapper = f"<html><body style='color:{self.text_color}; background:{self.primary_color}; font-family:Microsoft YaHei;'>{ai_block}</body></html>"
                self.chat_display.setHtml(wrapper)
            else:
                self.chat_display.setHtml(clean.replace("</body>", ai_block + "</body>"))

            self.chat_input.clear()
            self.status_bar.showMessage("对话完成")

            if self.voice_chat_enabled.isChecked():
                # AI 生成的简短摘要优先，否则用本地截取
                tts_text = event.tts if event.tts else self._summarize_for_tts(ai_msg)
                self.speak_text(tts_text)
            else:
                # 用户取消勾选 → 停止正在播放的 TTS
                self.stop_speaking()

        elif event.event_type == "error":
            self.show_ai_progress(False)
            self.status_bar.showMessage("AI回复失败")
            self.show_message_box("错误", f"AI回复失败：{event.data}", QMessageBox.Critical)

    def show_board_interaction(self):
        """开发板交互：自动切换到硬件视窗，仅在未连接时发起连接（避免误触断开）"""
        # 1. 自动切换到硬件视窗 Tab
        if hasattr(self, 'image_tab_widget'):
            for i in range(self.image_tab_widget.count()):
                if "硬件" in self.image_tab_widget.tabText(i):
                    self.image_tab_widget.setCurrentIndex(i)
                    break

        # 2. 如果已经连接，不再重复连接（防止误触断开）
        if hasattr(self, 'camera_receiver') and self.camera_receiver.is_receiving:
            self.status_bar.showMessage("✅ 已切换到硬件视窗，开发板摄像头已连接")
            return

        # 3. 发起连接
        self.toggle_camera_connection()
        self.status_bar.showMessage("🔄 已切换到硬件实时视窗，正在连接开发板...")

    def trigger_board_voice(self):
        """远程唤醒开发板语音服务，并自动重定向 AI 音频输出"""
        try:
            # 1. 智能路由：自动将 AI 语音回复输出设置为开发板
            if hasattr(self, 'set_audio_device'):
                self.set_audio_device('board')

            # 2. 向开发板发送 UDP 唤醒指令
            board_ip = read_system_config()["network"]["board_ip"]
            target_port = NETWORK_PORTS.get("VOICE_RECEIVE_PORT", 5006)

            command = {
                "type": "pc_control",
                "command": "start_recording",
                "timestamp": datetime.now().isoformat()
            }

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(json.dumps(command).encode('utf-8'), (board_ip, target_port))
            sock.close()

            # 3. UI 反馈
            self.status_bar.showMessage(f"📡 已向开发板 ({board_ip}) 发送语音唤醒指令")
            self.board_voice_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #38A169;
                    border: 1px solid #2F855A;
                    color: white;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 13pt;
                }}
            """)
            self.board_voice_button.setText("📡 监听中...")

            # 3秒后恢复按钮样式
            QTimer.singleShot(3000, lambda: (
                self.board_voice_button.setText("📱 唤醒板端语音"),
                self.board_voice_button.setStyleSheet(self.history_button.styleSheet())
            ))

        except Exception as e:
            self.show_message_box("网络错误",
                f"无法连接到开发板语音服务: {e}\n\n请检查开发板 IP 和局域网连接。",
                QMessageBox.Critical)
    
    def start_board_camera(self):
        """启动开发板摄像头功能"""
        try:
            if not hasattr(self, 'camera_receiver'):
                self.camera_receiver = BoardCameraReceiver()
                self.camera_receiver.frame_received.connect(self.update_camera_preview)
                self.camera_receiver.connection_status_changed.connect(self.update_camera_status)
                self.camera_receiver.diagnosis_request_received.connect(self.handle_board_diagnosis_request)
            
            self.camera_receiver.start_receiving()
            # 更新开发板摄像头连接状态
            if hasattr(self, 'board_camera_status'):
                self.board_camera_status.setText("🟢 已连接")
            
            # 同时启动命令监听器
            if not hasattr(self, 'command_listener'):
                self.start_command_listener()
                
            print("✅ 开发板摄像头和命令监听器已启动")
            self.status_bar.showMessage("开发板连接已建立")
            
        except Exception as e:
            self.show_message_box("错误", f"开发板摄像头启动失败: {e}")
            print(f"❌ 开发板摄像头启动失败: {e}")
    
    def handle_board_diagnosis_request(self, request_data):
        """处理开发板发来的诊断请求"""
        try:
            image = request_data['image']
            header = request_data['header']
            source = request_data['source']
            addr = request_data['addr']
            
            print(f"[开发板] 收到诊断请求,来源: {source}, 地址: {addr}")
            print(f"[开发板] 图像大小: {image.shape}, 请求ID: {header.get('request_id', 'N/A')}")
            
            # 检查是否需要保存到PC端
            save_to_pc = header.get('save_to_pc', False)
            pc_save_path = header.get('pc_save_path', '')
            
            if save_to_pc and pc_save_path:
                print(f"[保存] 准备保存图像到PC端: {pc_save_path}")
                self._save_image_to_pc(image, header, pc_save_path)
            
            # 更新摄像头预览
            self.update_camera_preview(image)
            
            # 执行AI诊断
            if hasattr(self, 'detector') and self.detector:
                try:
                    # 进行预测
                    results = self.detector.predict(image)
                    
                    # 解析结果
                    if results:
                        # 获取疾病名称和置信度
                        disease_name, confidence = self.parse_detection_results(results)
                        
                        # 生成建议
                        advice = self.generate_medical_advice(disease_name, confidence)
                        
                        # 构造诊断结果
                        diagnosis_result = {
                            "type": "diagnosis_result",
                            "request_id": header.get('request_id'),
                            "timestamp": datetime.now().isoformat(),
                            "disease_name": disease_name,
                            "confidence": confidence,
                            "advice": advice,
                            "image_size": image.shape,
                            "processing_time": time.time() - float(header.get('timestamp', time.time() * 1000)) / 1000
                        }
                        
                        # 发送诊断结果回开发板
                        self.send_diagnosis_result_to_board(diagnosis_result, addr)
                        
                        print(f"[开发板] 诊断完成: {disease_name} (置信度: {confidence:.2%})")
                        
                    else:
                        # 诊断失败
                        error_result = {
                            "type": "diagnosis_error",
                            "request_id": header.get('request_id'),
                            "timestamp": datetime.now().isoformat(),
                            "error": "AI模型预测失败",
                            "advice": "请重新拍摄或检查图像质量"
                        }
                        self.send_diagnosis_result_to_board(error_result, addr)
                        
                except Exception as e:
                    print(f"[开发板] AI诊断错误: {e}")
                    error_result = {
                        "type": "diagnosis_error",
                        "request_id": header.get('request_id'),
                        "timestamp": datetime.now().isoformat(),
                        "error": f"诊断过程出错: {str(e)}",
                        "advice": "请稍后重试或联系技术支持"
                    }
                    self.send_diagnosis_result_to_board(error_result, addr)
            else:
                print("[开发板] AI检测器未初始化")
                error_result = {
                    "type": "diagnosis_error",
                    "request_id": header.get('request_id'),
                    "timestamp": datetime.now().isoformat(),
                    "error": "AI检测器未就绪",
                    "advice": "请等待系统初始化完成"
                }
                self.send_diagnosis_result_to_board(error_result, addr)
                
        except Exception as e:
            print(f"[开发板] 处理诊断请求失败: {e}")
    
    def _save_image_to_pc(self, image, header, pc_save_path):
        """保存图像到PC端指定目录"""
        try:
            import os
            from datetime import datetime
            
            # 确保目录存在
            os.makedirs(pc_save_path, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            request_id = header.get('request_id', 'unknown')
            
            # 保存原始图像
            image_filename = f"board_image_{timestamp}_{request_id}.jpg"
            image_path = os.path.join(pc_save_path, image_filename)
            cv2.imwrite(image_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 保存请求信息
            info_filename = f"board_info_{timestamp}_{request_id}.json"
            info_path = os.path.join(pc_save_path, info_filename)
            
            info_data = {
                "request_id": request_id,
                "timestamp": header.get('timestamp'),
                "image_size": image.shape,
                "image_path": image_path,
                "source": "board_camera",
                "pc_save_time": datetime.now().isoformat()
            }
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ [保存] 图像已保存到PC端:")
            print(f"   图像文件: {image_path}")
            print(f"   信息文件: {info_path}")
            
        except Exception as e:
            print(f"❌ [保存] 保存到PC端失败: {e}")
    
    def send_diagnosis_result_to_board(self, result, addr):
        """发送诊断结果到开发板"""
        try:
            # 创建UDP套接字发送结果
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # 将结果转换为JSON并发送
            result_json = json.dumps(result, ensure_ascii=False)
            result_bytes = result_json.encode('utf-8')
            
            # 发送到开发板的诊断结果端口
            sock.sendto(result_bytes, (addr[0], 5003))  # 5003是开发板诊断结果接收端口
            sock.close()
            
            print(f"[开发板] 诊断结果已发送到 {addr[0]}:5003")
            
        except Exception as e:
            print(f"[开发板] 发送诊断结果失败: {e}")
    
    def parse_detection_results(self, results):
        """解析检测结果,返回疾病名称和置信度"""
        try:
            # 这里需要根据实际的AI模型输出格式进行解析
            # 假设results是字符串格式的结果
            if isinstance(results, str):
                # 简单的字符串解析
                if "正常" in results or "normal" in results.lower():
                    return "正常", 0.95
                elif "白内障" in results or "cataract" in results.lower():
                    return "白内障", 0.85
                elif "青光眼" in results or "glaucoma" in results.lower():
                    return "青光眼", 0.80
                else:
                    return "眼部异常", 0.70
            else:
                # 如果是其他格式,返回默认值
                return "眼部检查", 0.75
                
        except Exception as e:
            print(f"解析检测结果失败: {e}")
            return "未知", 0.50
    
    def generate_medical_advice(self, disease_name, confidence):
        """根据疾病名称和置信度生成医疗建议"""
        try:
            if disease_name == "正常":
                advice = "眼部检查结果正常,建议定期进行眼部健康检查,保持良好的用眼习惯。"
            elif disease_name == "白内障":
                if confidence > 0.8:
                    advice = "检测到白内障症状,建议及时就医进行专业检查,可能需要手术治疗。"
                else:
                    advice = "可能存在白内障风险,建议到医院进行详细检查确认。"
            elif disease_name == "青光眼":
                if confidence > 0.8:
                    advice = "检测到青光眼症状,这是严重的眼部疾病,建议立即就医治疗。"
                else:
                    advice = "可能存在青光眼风险,建议尽快到医院进行专业检查。"
            else:
                advice = "检测到眼部异常,建议到医院进行专业检查,确定具体病情。"
            
            # 播放建议语音（根据设置选择播放设备）
            self.play_advice_audio(advice)
            return advice
                
        except Exception as e:
            print(f"生成医疗建议失败: {e}")
            return "建议到医院进行专业检查。"
    
    def play_advice_audio(self, advice_text):
        """播放AI建议语音,支持双端选择"""
        try:
            # 获取音频播放设置
            audio_device = getattr(self, 'audio_output_device', 'pc')  # 默认PC端播放
            
            if audio_device == 'pc':
                # 在PC端播放
                self.speak_text(advice_text)
                print(f"[音频] PC端播放建议: {advice_text[:50]}...")
            elif audio_device == 'board':
                # 发送到开发板播放
                self.send_audio_to_board(advice_text)
                print(f"[音频] 开发板播放建议: {advice_text[:50]}...")
            elif audio_device == 'both':
                # 双端同时播放
                self.speak_text(advice_text)
                self.send_audio_to_board(advice_text)
                print(f"[音频] 双端播放建议: {advice_text[:50]}...")
                
        except Exception as e:
            print(f"[音频] 播放建议失败: {e}")
    
    def send_audio_to_board(self, text):
        """发送音频文本到开发板进行TTS播放"""
        try:
            # 构造音频播放命令
            audio_command = {
                "type": "tts_play",
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "source": "pc_ai_advice"
            }
            
            # 发送到开发板
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            command_json = json.dumps(audio_command, ensure_ascii=False)
            command_bytes = command_json.encode('utf-8')
            
            # 发送到开发板音频端口
            board_ip = read_system_config()["network"]["board_ip"]  # 默认开发板IP
            sock.sendto(command_bytes, (board_ip, NETWORK_PORTS.get("VOICE_RECEIVE_PORT", 5006)))
            sock.close()
            
            print(f"[音频] 音频命令已发送到开发板 {board_ip}:5006")
            
        except Exception as e:
            print(f"[音频] 发送到开发板失败: {e}")
    
    def set_audio_device(self, device):
        """设置音频输出设备"""
        try:
            self.audio_output_device = device
            print(f"[设置] 音频输出设备已设置为: {device}")
            
            # 更新界面提示
            device_names = {
                'pc': 'PC端播放',
                'board': '开发板播放', 
                'both': '双端播放'
            }
            
            self.show_message_box(
                "设置成功", 
                f"音频播放设备已设置为: {device_names.get(device, device)}",
                QMessageBox.Information
            )
            
        except Exception as e:
            print(f"[设置] 设置音频设备失败: {e}")
    
    def test_audio_output(self):
        """测试音频输出"""
        try:
            test_text = "这是音频播放测试,如果您能听到这条消息,说明音频设备工作正常。"
            current_device = getattr(self, 'audio_output_device', 'pc')
            
            print(f"[测试] 测试音频设备: {current_device}")
            self.play_advice_audio(test_text)
            
        except Exception as e:
            print(f"[测试] 音频测试失败: {e}")
            self.show_message_box("测试失败", f"音频测试失败: {e}", QMessageBox.Warning)
    
    def start_board_voice(self):
        """启动开发板语音指导"""
        # 先确保语音服务器运行
        if not hasattr(self, 'voice_server_process') or not self.voice_server_process:
            self.toggle_voice_server()
            
        instructions = """
开发板语音对话启动步骤：

1. 确保PC端语音服务已启动 ✓

2. 在开发板终端运行：
   python board_voice_interaction.py

3. 语音对话操作：
   - 'r': 录音并发送
   - 't': 文本转语音
   - 'q': 退出

4. 或在摄像头预览中按 'r'

注意：需要开发板有麦克风和扬声器
        """
        self.show_message_box("语音对话启动", instructions, QMessageBox.Information)
    
    def test_board_connection(self):
        """测试开发板连接"""
        try:
            import socket
            
            # 启动命令监听器
            if not hasattr(self, 'command_listener'):
                self.start_command_listener()
            
            # 简单的端口测试
            test_results = []
            test_ports = [5000, 5002, 5005]
            
            for port in test_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.bind(("0.0.0.0", port))
                    sock.close()
                    test_results.append(f"端口 {port}: [可用]")
                except:
                    test_results.append(f"端口 {port}: [占用]")
            
            result_text = "网络端口测试结果：\n\n" + "\n".join(test_results)
            result_text += "\n\n建议：\n- 确保防火墙允许这些端口\n- 检查是否有其他程序占用端口"
            
            self.show_message_box("连接测试", result_text, QMessageBox.Information)
            
        except Exception as e:
            self.show_message_box("测试失败", "连接测试失败: {}".format(e), QMessageBox.Critical)
    
    def start_command_listener(self):
        """启动命令监听器"""
        try:
            if not hasattr(self, 'command_listener'):
                self.command_listener = CommandListener()
                self.command_listener.command_received.connect(self.handle_board_command)
                self.command_listener.start_listening()
                print("[开发板] 命令监听器已启动")
        except Exception as e:
            print(f"[开发板] 启动命令监听器失败: {e}")
    
    def handle_board_command(self, command_data):
        """处理开发板发来的命令"""
        try:
            command_type = command_data.get('type')
            print(f"[开发板] 收到命令: {command_type}")
            
            if command_type == 'connection_test':
                # 处理连接测试
                self.handle_connection_test(command_data)
            elif command_type == 'diagnosis_request':
                # 处理诊断请求（图像数据会通过摄像头接收器处理）
                self.handle_diagnosis_request(command_data)
            elif command_type == 'image_save_request':
                # 处理图像保存请求
                self.handle_image_save_request(command_data)
            elif command_type == 'voice_command':
                # 处理语音命令
                self.handle_voice_command(command_data)
            elif command_type == 'heartbeat':
                # 处理心跳包
                self.handle_heartbeat(command_data)
            else:
                print(f"[开发板] 未知命令类型: {command_type}")
                
        except Exception as e:
            print(f"[开发板] 处理命令失败: {e}")
    
    def handle_connection_test(self, command_data):
        """处理连接测试命令"""
        try:
            # 发送连接测试响应
            response = {
                "type": "connection_test_response",
                "timestamp": datetime.now().isoformat(),
                "server_status": "running",
                "latency": 0,  # 可以计算实际延迟
                "services": {
                    "ai_diagnosis": "available",
                    "voice_processing": "available",
                    "image_processing": "available"
                }
            }
            
            # 发送响应
            self.send_command_response(response, command_data.get('source_addr'))
            print("[开发板] 连接测试响应已发送")
            
        except Exception as e:
            print(f"[开发板] 处理连接测试失败: {e}")
    
    def handle_diagnosis_request(self, command_data):
        """处理诊断请求命令"""
        try:
            # 存储请求头信息,等待图像数据
            request_id = command_data.get('request_id')
            print(f"[开发板] 处理诊断请求,request_id: {request_id}")
            print(f"[开发板] 请求数据: {command_data}")
            
            if request_id and hasattr(self, 'camera_receiver'):
                # 将请求头信息存储到摄像头接收器
                self.camera_receiver.store_request_header(request_id, command_data)
                print(f"[开发板] 诊断请求头已存储: {request_id}")
                print(f"[开发板] 当前存储的请求头数量: {len(self.camera_receiver.request_headers)}")
            else:
                print(f"[开发板] 无法存储诊断请求头,request_id: {request_id}, has_camera_receiver: {hasattr(self, 'camera_receiver')}")
                
        except Exception as e:
            print(f"[开发板] 处理诊断请求失败: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_image_save_request(self, command_data):
        """处理图像保存请求命令"""
        try:
            request_id = command_data.get('request_id')
            filename = command_data.get('filename', 'unknown.jpg')
            pc_save_path = command_data.get('pc_save_path', '')
            
            print(f"[开发板] 处理图像保存请求,request_id: {request_id}")
            print(f"[开发板] 文件名: {filename}")
            print(f"[开发板] 保存路径: {pc_save_path}")
            
            if request_id and hasattr(self, 'camera_receiver'):
                # 将保存请求头信息存储到摄像头接收器
                self.camera_receiver.store_request_header(request_id, command_data)
                print(f"[开发板] 图像保存请求头已存储: {request_id}")
                print(f"[开发板] 当前存储的请求头数量: {len(self.camera_receiver.request_headers)}")
            else:
                print(f"[开发板] 无法存储图像保存请求头,request_id: {request_id}, has_camera_receiver: {hasattr(self, 'camera_receiver')}")
                
        except Exception as e:
            print(f"[开发板] 处理图像保存请求失败: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_voice_command(self, command_data):
        """处理语音命令"""
        try:
            command = command_data.get('command')
            params = command_data.get('params', {})
            addr = command_data.get('source_addr')
            
            if command == 'start_recording':
                print("[开发板] 开发板开始录音")
                # 可以在这里更新UI状态
                
            elif command == 'process_voice':
                print("[开发板] 开发板请求处理语音")
                # 处理语音识别请求
                self.process_board_voice_data(command_data, addr)
                
            elif command == 'voice_text':
                # 处理开发板发来的语音识别文本
                text = params.get('text', '')
                if text:
                    print(f"[语音] 收到开发板语音文本: {text}")
                    self.handle_board_voice_text(text, addr)
                    
            else:
                print(f"[开发板] 未知语音命令: {command}")
                
        except Exception as e:
            print(f"[开发板] 处理语音命令失败: {e}")
    
    def process_board_voice_data(self, command_data, addr):
        """处理开发板发来的语音数据"""
        try:
            # 这里可以接收和处理语音数据
            # 实际的语音数据可能通过其他端口传输
            print("[语音] 准备处理开发板语音数据")
            
            # 发送处理状态响应
            response = {
                "type": "voice_processing_status",
                "status": "processing",
                "timestamp": datetime.now().isoformat()
            }
            self.send_command_response(response, addr)
            
        except Exception as e:
            print(f"[语音] 处理开发板语音数据失败: {e}")
    
    def handle_board_voice_text(self, text, addr):
        """处理开发板发来的语音识别文本"""
        try:
            print(f"[AI对话] 开发板语音输入: {text}")
            
            # 使用DeepSeek API处理语音文本
            if hasattr(self, 'deepseek_api') and self.deepseek_api:
                # 异步处理AI响应
                def process_ai_response():
                    try:
                        # 获取AI回复
                        ai_response = self.deepseek_api.get_custom_advice(text)
                        
                        # 发送AI回复到开发板
                        self.send_ai_response_to_board(ai_response, addr)
                        
                        # 同时在PC端显示对话
                        self.display_board_conversation(text, ai_response)
                        
                    except Exception as e:
                        print(f"[AI对话] AI处理失败: {e}")
                        error_response = "抱歉,AI处理出现问题,请稍后重试。"
                        self.send_ai_response_to_board(error_response, addr)
                
                # 在后台线程处理
                import threading
                threading.Thread(target=process_ai_response, daemon=True).start()
                
            else:
                print("[AI对话] DeepSeek API未初始化")
                fallback_response = "AI服务暂时不可用,请稍后重试。"
                self.send_ai_response_to_board(fallback_response, addr)
                
        except Exception as e:
            print(f"[AI对话] 处理开发板语音文本失败: {e}")
    
    def send_ai_response_to_board(self, response_text, addr):
        """发送AI回复到开发板"""
        try:
            # 根据音频设备设置决定是否发送到开发板
            audio_device = getattr(self, 'audio_output_device', 'pc')
            
            if audio_device in ['board', 'both']:
                # 发送TTS播放命令到开发板
                self.send_audio_to_board(response_text)
            
            if audio_device in ['pc', 'both']:
                # 在PC端也播放
                self.speak_text(response_text)
            
            # 发送文本响应
            response = {
                "type": "ai_response",
                "text": response_text,
                "timestamp": datetime.now().isoformat(),
                "audio_device": audio_device
            }
            
            self.send_command_response(response, addr)
            print(f"[AI对话] AI回复已发送到开发板: {response_text[:50]}...")
            
        except Exception as e:
            print(f"[AI对话] 发送AI回复失败: {e}")
    
    def display_board_conversation(self, user_text, ai_response):
        """在PC端显示开发板对话"""
        try:
            # 在聊天界面显示对话
            conversation = f"[开发板用户] {user_text}\n\n[AI助手] {ai_response}"
            
            # 如果有聊天界面,显示对话
            if hasattr(self, 'chat_display') and self.chat_display:
                current_content = self.chat_display.toPlainText()
                new_content = current_content + "\n\n" + "="*50 + "\n" + conversation + "\n" + "="*50
                self.chat_display.setPlainText(new_content)
                
                # 滚动到底部
                cursor = self.chat_display.textCursor()
                cursor.movePosition(cursor.End)
                self.chat_display.setTextCursor(cursor)
            
            print(f"[界面] 开发板对话已显示在PC端")
            
        except Exception as e:
            print(f"[界面] 显示开发板对话失败: {e}")
    
    def handle_heartbeat(self, command_data):
        """处理心跳包"""
        try:
            # 发送心跳响应
            response = {
                "type": "heartbeat_response",
                "timestamp": datetime.now().isoformat(),
                "server_status": "running",
                "services_available": True
            }
            
            # 发送响应
            self.send_command_response(response, command_data.get('source_addr'))
            
        except Exception as e:
            print(f"[开发板] 处理心跳失败: {e}")
    
    def send_command_response(self, response, addr):
        """发送命令响应"""
        try:
            if not addr:
                return
                
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            response_json = json.dumps(response, ensure_ascii=False)
            response_bytes = response_json.encode('utf-8')
            
            # 根据响应类型选择正确的端口
            if response.get('type') == 'connection_test_response':
                # 连接测试响应直接回复到发送者的地址
                sock.sendto(response_bytes, addr)
                print(f"✅ [开发板] 连接测试响应已发送到 {addr}")
            else:
                # 其他响应发送到开发板的诊断端口
                sock.sendto(response_bytes, (addr[0], 5003))  # 5003是开发板诊断接收端口
                print(f"✅ [开发板] 响应已发送到 {addr[0]}:5003")
            
            sock.close()
            
        except Exception as e:
            print(f"❌ [开发板] 发送命令响应失败: {e}")
    
    # ============================================================
    #  摄像头相关方法
    # ============================================================
    
    def toggle_camera_connection(self):
        """切换摄像头连接状态"""
        if not self.camera_receiver.is_receiving:
            self.camera_receiver.start_receiving()
            self.connect_camera_button.setText("🔌 断开")
        else:
            self.camera_receiver.stop_receiving()
            self.connect_camera_button.setText("🔗 连接")
            self.camera_preview_label.setText("摄像头未连接\\n点击连接开始预览")
            self.capture_from_camera_button.setEnabled(False)
    
    def update_camera_status(self, connected):
        """更新摄像头连接状态"""
        if connected:
            self.connect_camera_button.setText("🔌 断开")
            self.capture_from_camera_button.setEnabled(True)
            self.camera_preview_label.setText("正在接收视频流...")
            # 更新开发板摄像头连接状态
            if hasattr(self, 'board_camera_status'):
                self.board_camera_status.setText("🟢 已连接")
        else:
            self.connect_camera_button.setText("🔗 连接")
            self.capture_from_camera_button.setEnabled(False)
            # 更新开发板摄像头连接状态
            if hasattr(self, 'board_camera_status'):
                self.board_camera_status.setText("🔴 未连接")
    
    def update_camera_preview(self, frame):
        """更新摄像头预览"""
        try:
            # 调整帧大小以适应预览窗口
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            
            # 转换为QImage
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # 缩放到适合的大小
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.camera_preview_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.camera_preview_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print("摄像头预览更新失败: {}".format(e))
    
    def capture_from_board_camera(self):
        """从开发板摄像头拍照并诊断"""
        try:
            # 获取当前预览帧
            pixmap = self.camera_preview_label.pixmap()
            if pixmap is None:
                QMessageBox.warning(self, "警告", "无法获取摄像头图像,请确保摄像头已连接")
                return
            
            # 转换为numpy数组进行诊断
            qimage = pixmap.toImage()
            qimage = qimage.convertToFormat(QImage.Format_RGB888)
            
            width = qimage.width()
            height = qimage.height()
            ptr = qimage.constBits()
            ptr.setsize(qimage.byteCount())
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
            
            # 转换为OpenCV格式 (BGR)
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            
            # 设置为当前图像并进行检测
            self.current_image = frame_bgr
            self.original_image_label.setPixmap(pixmap)
            
            # 启用检测按钮并自动开始检测
            self.detect_button.setEnabled(True)
            self.detect_image()
            
            QMessageBox.information(self, "提示", "已从开发板摄像头拍照并开始诊断！")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", "拍照诊断失败：{}".format(str(e)))



# ============================================================
#  程序入口
# ============================================================
        


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 11)
    app.setFont(font)
    # 全局统一字体：QSS 中未指定 font-family 时回退到微软雅黑
    app.setStyleSheet("* { font-family: 'Microsoft YaHei', 'SimHei', sans-serif; }")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
