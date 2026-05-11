#!/usr/bin/env python3
"""
PC端医疗诊断服务器
功能：
1. 接收来自开发板的图像数据
2. 调用医疗诊断系统进行AI分析
3. 返回诊断结果到开发板
4. 支持多开发板同时连接
"""

import socket
import cv2
import numpy as np
import time
import json
import threading
import sys
import os
from datetime import datetime
import queue

# 添加主系统路径以导入诊断模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ===== 配置参数 =====
SERVER_IP = "0.0.0.0"      # 服务器绑定IP
CAMERA_PORT = 5002         # 接收摄像头数据端口
DIAGNOSIS_PORT = 5003      # 发送诊断结果端口
COMMAND_PORT = 5004        # 接收命令控制端口

MAX_PACKET_SIZE = 1400     # 最大数据包大小
DIAGNOSIS_TIMEOUT = 30     # 诊断超时时间(秒)
MAX_CLIENTS = 5            # 最大同时连接客户端数
# ===================

class ImageBuffer:
    """图像缓存管理器"""
    
    def __init__(self):
        self.buffers = {}  # {client_id: {packet_id: data, ...}}
        self.metadata = {}  # {client_id: header_info}
        self.lock = threading.Lock()
        
    def add_packet(self, client_addr, packet_id, total_packets, data):
        """添加图像数据包"""
        with self.lock:
            client_id = f"{client_addr[0]}:{client_addr[1]}"
            
            if client_id not in self.buffers:
                self.buffers[client_id] = {}
                
            self.buffers[client_id][packet_id] = data
            
            # 检查是否接收完整
            if len(self.buffers[client_id]) == total_packets:
                # 重组图像
                sorted_packets = [self.buffers[client_id][i] for i in range(total_packets)]
                image_data = b''.join(sorted_packets)
                
                # 清理缓存
                del self.buffers[client_id]
                
                return image_data
            
            return None
    
    def set_metadata(self, client_addr, metadata):
        """设置图像元数据"""
        with self.lock:
            client_id = f"{client_addr[0]}:{client_addr[1]}"
            self.metadata[client_id] = metadata
    
    def get_metadata(self, client_addr):
        """获取图像元数据"""
        with self.lock:
            client_id = f"{client_addr[0]}:{client_addr[1]}"
            return self.metadata.get(client_id, {})
    
    def cleanup_client(self, client_addr):
        """清理客户端数据"""
        with self.lock:
            client_id = f"{client_addr[0]}:{client_addr[1]}"
            self.buffers.pop(client_id, None)
            self.metadata.pop(client_id, None)

class DiagnosisEngine:
    """诊断引擎适配器"""
    
    def __init__(self):
        self.detector = None
        self.processor = None
        self.init_diagnosis_components()
    
    def init_diagnosis_components(self):
        """初始化诊断组件"""
        try:
            # 尝试导入主系统的诊断组件
            from visualization_test2 import EyeDiseaseDetector, ResultProcessor
            
            self.detector = EyeDiseaseDetector()
            self.processor = ResultProcessor(self.detector)
            
            # 尝试加载默认模型
            default_model_paths = [
                "models/eye_disease_model.pth",
                "models/best.pt",
                "../models/eye_disease_model.pth",
                "../models/best.pt"
            ]
            
            for model_path in default_model_paths:
                if os.path.exists(model_path):
                    print(f"[检查] 加载诊断模型: {model_path}")
                    if self.detector.load_model(model_path):
                        print("[成功] 诊断模型加载成功")
                        break
            else:
                print("[警告] 未找到诊断模型，请手动指定模型路径")
                
        except ImportError as e:
            print(f"[警告] 无法导入诊断组件: {e}")
            print("[配置] 使用模拟诊断模式")
            self.detector = None
            self.processor = None
        except Exception as e:
            print(f"[错误] 诊断组件初始化失败: {e}")
            self.detector = None
            self.processor = None
    
    def diagnose_image(self, image):
        """诊断图像"""
        try:
            if self.detector and self.processor:
                # 使用真实的诊断系统
                results = self.detector.predict(image)
                parsed_results = self.processor.parse_model_results(results)
                
                if parsed_results:
                    disease_name = parsed_results.get('disease_name', '未知')
                    confidence = parsed_results.get('confidence', 0.0)
                    
                    # 生成医疗建议
                    advice = self.generate_medical_advice(disease_name, confidence)
                    
                    return {
                        'success': True,
                        'disease_name': disease_name,
                        'confidence': float(confidence),
                        'advice': advice,
                        'emergency': confidence > 0.85 and disease_name != '正常',
                        'timestamp': datetime.now().isoformat(),
                        'model_info': '眼部疾病检测模型'
                    }
                else:
                    return self.get_fallback_result()
            else:
                # 模拟诊断结果
                return self.simulate_diagnosis(image)
                
        except Exception as e:
            print(f"[错误] 诊断过程出错: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def simulate_diagnosis(self, image):
        """模拟诊断结果（当真实模型不可用时）"""
        # 基于图像特征的简单分析
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # 模拟不同的诊断结果
        if mean_intensity < 80:
            disease_name = "可能的视网膜病变"
            confidence = 0.65
        elif std_intensity < 30:
            disease_name = "图像质量不佳"
            confidence = 0.45
        else:
            disease_name = "未发现明显异常"
            confidence = 0.75
        
        advice = self.generate_medical_advice(disease_name, confidence)
        
        return {
            'success': True,
            'disease_name': disease_name,
            'confidence': confidence,
            'advice': advice,
            'emergency': False,
            'timestamp': datetime.now().isoformat(),
            'model_info': '模拟诊断系统',
            'note': '这是模拟结果，请使用真实的AI模型进行诊断'
        }
    
    def generate_medical_advice(self, disease_name, confidence):
        """生成医疗建议"""
        advice_map = {
            '正常': '目前未发现明显异常，建议定期检查',
            '白内障': '建议尽快咨询眼科医生，可能需要手术治疗',
            '青光眼': '这是一种严重的眼病，请立即就医进行详细检查',
            '糖尿病视网膜病变': '需要立即就医，同时控制血糖水平',
            '黄斑变性': '建议咨询视网膜专科医生，可能需要特殊治疗',
            '未发现明显异常': '目前看起来正常，建议定期体检',
            '图像质量不佳': '请在光线充足的环境下重新拍摄',
            '可能的视网膜病变': '建议进一步检查，咨询专科医生'
        }
        
        base_advice = advice_map.get(disease_name, '建议咨询专业医生进行详细检查')
        
        if confidence < 0.6:
            base_advice += '\n[警告] 由于检测置信度较低，强烈建议人工复查'
        elif confidence > 0.8:
            base_advice += '\n[成功] 检测置信度较高，结果相对可靠'
        
        return base_advice
    
    def get_fallback_result(self):
        """获取备用结果"""
        return {
            'success': True,
            'disease_name': '无法确定',
            'confidence': 0.0,
            'advice': '检测失败，请重新拍摄或咨询医生',
            'emergency': False,
            'timestamp': datetime.now().isoformat(),
            'model_info': '检测失败'
        }

class DiagnosisServer:
    """诊断服务器主控制器"""
    
    def __init__(self):
        self.image_buffer = ImageBuffer()
        self.diagnosis_engine = DiagnosisEngine()
        self.diagnosis_queue = queue.Queue()
        self.is_running = False
        
        # 网络套接字
        self.camera_sock = None
        self.diagnosis_sock = None
        self.command_sock = None
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_diagnoses': 0,
            'failed_diagnoses': 0,
            'connected_clients': set(),
            'start_time': time.time()
        }
    
    def init_sockets(self):
        """初始化网络套接字"""
        try:
            # 摄像头数据接收套接字
            self.camera_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.camera_sock.bind((SERVER_IP, CAMERA_PORT))
            self.camera_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
            
            # 诊断结果发送套接字
            self.diagnosis_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.diagnosis_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
            
            # 命令控制接收套接字
            self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.command_sock.bind((SERVER_IP, COMMAND_PORT))
            
            print(f"[网络] 服务器套接字初始化完成")
            print(f"   摄像头数据端口: {CAMERA_PORT}")
            print(f"   诊断结果端口: {DIAGNOSIS_PORT}")
            print(f"   命令控制端口: {COMMAND_PORT}")
            return True
            
        except Exception as e:
            print(f"[错误] 网络初始化失败: {e}")
            return False
    
    def start_server(self):
        """启动服务器"""
        print("[启动] 启动PC端医疗诊断服务器...")
        
        if not self.init_sockets():
            return False
        
        self.is_running = True
        
        # 启动各个处理线程
        threads = [
            threading.Thread(target=self.camera_data_handler, daemon=True),
            threading.Thread(target=self.command_handler, daemon=True),
            threading.Thread(target=self.diagnosis_worker, daemon=True),
            threading.Thread(target=self.stats_reporter, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        print("[成功] 服务器启动完成，等待开发板连接...")
        return True
    
    def camera_data_handler(self):
        """处理摄像头数据"""
        print("[摄像头] 摄像头数据处理线程启动")
        
        while self.is_running:
            try:
                data, addr = self.camera_sock.recvfrom(MAX_PACKET_SIZE + 10)
                
                if len(data) < 8:
                    continue
                
                # 解析包头
                packet_id = int.from_bytes(data[0:4], 'big')
                total_packets = int.from_bytes(data[4:8], 'big')
                payload = data[8:]
                
                # 重组图像
                image_data = self.image_buffer.add_packet(addr, packet_id, total_packets, payload)
                
                if image_data:
                    # 图像接收完成，加入诊断队列
                    metadata = self.image_buffer.get_metadata(addr)
                    
                    task = {
                        'client_addr': addr,
                        'image_data': image_data,
                        'metadata': metadata,
                        'timestamp': time.time()
                    }
                    
                    self.diagnosis_queue.put(task)
                    self.stats['connected_clients'].add(addr[0])
                    print(f"[图像] 收到来自 {addr[0]} 的图像 ({len(image_data)} 字节)")
                
            except Exception as e:
                if self.is_running:
                    print(f"[错误] 摄像头数据处理错误: {e}")
    
    def command_handler(self):
        """处理命令控制"""
        print("[命令] 命令控制处理线程启动")
        
        while self.is_running:
            try:
                data, addr = self.command_sock.recvfrom(4096)
                
                try:
                    command = json.loads(data.decode('utf-8'))
                    
                    if command.get('type') == 'diagnosis_request':
                        # 保存图像元数据
                        self.image_buffer.set_metadata(addr, command)
                        self.stats['total_requests'] += 1
                        print(f"[请求] 收到诊断请求 - 客户端: {addr[0]}")
                    
                except json.JSONDecodeError:
                    print(f"[警告] 无效的命令格式来自 {addr[0]}")
                
            except Exception as e:
                if self.is_running:
                    print(f"[错误] 命令处理错误: {e}")
    
    def diagnosis_worker(self):
        """诊断工作线程"""
        print("[诊断] 诊断工作线程启动")
        
        while self.is_running:
            try:
                # 从队列获取诊断任务
                task = self.diagnosis_queue.get(timeout=1.0)
                
                client_addr = task['client_addr']
                image_data = task['image_data']
                
                print(f"[检查] 开始诊断图像 - 客户端: {client_addr[0]}")
                
                # 解码图像
                img_array = np.frombuffer(image_data, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if image is not None:
                    # 执行诊断
                    start_time = time.time()
                    diagnosis_result = self.diagnosis_engine.diagnose_image(image)
                    diagnosis_time = time.time() - start_time
                    
                    # 添加诊断时间信息
                    diagnosis_result['diagnosis_time'] = diagnosis_time
                    diagnosis_result['server_info'] = {
                        'server_ip': socket.gethostname(),
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    # 发送结果
                    self.send_diagnosis_result(client_addr, diagnosis_result)
                    
                    # 更新统计
                    if diagnosis_result.get('success', False):
                        self.stats['successful_diagnoses'] += 1
                        print(f"[成功] 诊断完成 - {client_addr[0]} - 用时: {diagnosis_time:.2f}s")
                    else:
                        self.stats['failed_diagnoses'] += 1
                        print(f"[错误] 诊断失败 - {client_addr[0]}")
                
                else:
                    # 图像解码失败
                    error_result = {
                        'success': False,
                        'error': '图像解码失败',
                        'timestamp': datetime.now().isoformat()
                    }
                    self.send_diagnosis_result(client_addr, error_result)
                    self.stats['failed_diagnoses'] += 1
                    print(f"[错误] 图像解码失败 - {client_addr[0]}")
                
                self.diagnosis_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[错误] 诊断工作线程错误: {e}")
    
    def send_diagnosis_result(self, client_addr, result):
        """发送诊断结果"""
        try:
            result_data = json.dumps(result, ensure_ascii=False).encode('utf-8')
            self.diagnosis_sock.sendto(result_data, (client_addr[0], DIAGNOSIS_PORT))
            print(f"[发送] 诊断结果已发送到 {client_addr[0]}")
            
        except Exception as e:
            print(f"[错误] 发送诊断结果失败: {e}")
    
    def stats_reporter(self):
        """统计信息报告线程"""
        while self.is_running:
            try:
                time.sleep(30)  # 每30秒报告一次
                
                uptime = time.time() - self.stats['start_time']
                hours = int(uptime // 3600)
                minutes = int((uptime % 3600) // 60)
                
                print("\n" + "="*50)
                print("[统计] 服务器运行状态")
                print("="*50)
                print(f"[时间]  运行时间: {hours:02d}:{minutes:02d}")
                print(f"[请求] 总请求数: {self.stats['total_requests']}")
                print(f"[成功] 成功诊断: {self.stats['successful_diagnoses']}")
                print(f"[错误] 失败诊断: {self.stats['failed_diagnoses']}")
                print(f"[连接] 连接设备: {len(self.stats['connected_clients'])}")
                print(f"[等待] 待处理队列: {self.diagnosis_queue.qsize()}")
                
                if self.stats['total_requests'] > 0:
                    success_rate = (self.stats['successful_diagnoses'] / self.stats['total_requests']) * 100
                    print(f"[状态] 成功率: {success_rate:.1f}%")
                
                print("="*50 + "\n")
                
            except Exception as e:
                print(f"[错误] 统计报告错误: {e}")
    
    def shutdown(self):
        """关闭服务器"""
        print("\n[更新] 正在关闭服务器...")
        self.is_running = False
        
        # 关闭套接字
        for sock in [self.camera_sock, self.diagnosis_sock, self.command_sock]:
            if sock:
                sock.close()
        
        print("[成功] 服务器已关闭")

def main():
    """主函数"""
    # 设置控制台编码以支持Unicode
    import sys
    if sys.platform == "win32":
        import locale
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            # Python < 3.7
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    
    print("[医院] PC端医疗诊断服务器")
    print("=" * 50)
    
    server = DiagnosisServer()
    
    try:
        if server.start_server():
            print("[提示] 服务器运行中，按 Ctrl+C 停止")
            
            # 保持主线程运行
            while server.is_running:
                time.sleep(1)
        else:
            print("[错误] 服务器启动失败")
            
    except KeyboardInterrupt:
        print("\n[取消] 收到停止信号")
    except Exception as e:
        print(f"[错误] 服务器错误: {e}")
    finally:
        server.shutdown()

if __name__ == "__main__":
    main()