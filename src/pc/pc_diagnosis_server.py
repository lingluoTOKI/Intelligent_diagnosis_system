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
from pathlib import Path

# 添加主系统路径以导入诊断模块
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.runtime_config import load_app_config
from configs.system_config import NETWORK_PORTS
from src.core.classification import EyeDiseaseDetector
from src.core.llm import DeepSeekAPI

# ===== 配置参数 =====
SERVER_IP = "0.0.0.0"      # 服务器绑定IP
CAMERA_PORT = NETWORK_PORTS["CAMERA_PORT"]         # 接收摄像头数据端口
DIAGNOSIS_PORT = NETWORK_PORTS["DIAGNOSIS_PORT"]      # 发送诊断结果端口
COMMAND_PORT = NETWORK_PORTS["COMMAND_PORT"]        # 接收命令控制端口

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
    """Headless adapter over the same validated classification service as the GUI."""
    def __init__(self, config=None, detector=None, api=None):
        self.config = config or load_app_config()
        self.detector = detector or EyeDiseaseDetector(self.config)
        self.api = api or DeepSeekAPI(config=self.config)
        self.initialization_error = None
        if detector is None and not self.detector.load_model():
            self.initialization_error = self.detector.last_error
            print("分类模型未就绪:", self.initialization_error)

    def diagnose_image(self, image):
        result = self.detector.classify(image)
        payload = result.to_dict()
        payload['timestamp'] = datetime.now().isoformat()
        if not result.success:
            payload['error'] = result.error or self.initialization_error or '分类失败'
            return payload
        advice = self.api.request_treatment_advice(result.disease_name, result.confidence)
        payload.update(advice=advice.content, advice_source=advice.source,
                       advice_success=advice.success, emergency=False)
        return payload


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