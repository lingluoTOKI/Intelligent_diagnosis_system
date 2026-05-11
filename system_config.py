# -*- coding: utf-8 -*-
"""
医疗诊断系统统一配置文件
用于确保PC端和开发板端配置一致
"""
import socket
import os

# ===== 网络配置 =====
def get_local_ip():
    """自动获取本机IP地址"""
    try:
        # 连接到远程地址来获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

# PC端IP配置（开发板需要连接的PC IP）
PC_IP = "172.20.10.3"  # 固定配置，不需要自动检测

# 网络端口配置（确保两端一致）
NETWORK_PORTS = {
    "CAMERA_PORT": 5002,        # 摄像头数据传输
    "DIAGNOSIS_PORT": 5003,     # 诊断结果接收
    "COMMAND_PORT": 5004,       # 命令控制
    "VOICE_SEND_PORT": 5005,    # 语音数据发送
    "VOICE_RECEIVE_PORT": 5006, # 语音合成接收
    "VOICE_COMMAND_PORT": 5007, # 语音命令控制
    "TOUCH_CONTROL_PORT": 5001, # 触摸屏控制
    "MOUSE_CONTROL_PORT": 5008, # 鼠标控制（屏幕共享）
}

# 将端口配置导出为变量以便兼容现有代码
CAMERA_PORT = NETWORK_PORTS["CAMERA_PORT"]
DIAGNOSIS_PORT = NETWORK_PORTS["DIAGNOSIS_PORT"]
COMMAND_PORT = NETWORK_PORTS["COMMAND_PORT"]
VOICE_SEND_PORT = NETWORK_PORTS["VOICE_SEND_PORT"]
VOICE_RECEIVE_PORT = NETWORK_PORTS["VOICE_RECEIVE_PORT"]
VOICE_COMMAND_PORT = NETWORK_PORTS["VOICE_COMMAND_PORT"]
TOUCH_CONTROL_PORT = NETWORK_PORTS["TOUCH_CONTROL_PORT"]
MOUSE_CONTROL_PORT = NETWORK_PORTS["MOUSE_CONTROL_PORT"]

# ===== 摄像头配置 =====
CAMERA_CONFIG = {
    "WIDTH": 640,
    "HEIGHT": 480,
    "FPS": 30,
    "JPEG_QUALITY": 85,
    "MAX_PACKET_SIZE": 1400,
}

# ===== 音频配置 =====
AUDIO_CONFIG = {
    "RATE": 16000,
    "CHANNELS": 1,
    "CHUNK": 1024,
    "RECORD_SECONDS": 5,
}

# ===== 系统配置 =====
SYSTEM_CONFIG = {
    "SAVE_DIR": "medical_images",
    "HEARTBEAT_INTERVAL": 5.0,  # 心跳间隔（秒）
    "CONNECTION_TIMEOUT": 10.0,  # 连接超时（秒）
    "RETRY_ATTEMPTS": 3,        # 重试次数
}

# ===== AI模型配置 =====
AI_CONFIG = {
    "MODEL_PATH": "models/best.pt",  # YOLO模型路径
    "CONFIDENCE_THRESHOLD": 0.5,     # 置信度阈值
    "API_TIMEOUT": 30,               # API请求超时
}

# ===== 连接状态检测 =====
class ConnectionManager:
    """统一的连接状态管理器"""
    
    def __init__(self):
        self.connection_status = {}
        self.last_heartbeat = {}
    
    def test_connection(self, target_ip, port):
        """测试到目标的网络连接"""
        import time
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(3.0)
            
            # 发送测试包
            test_data = {
                "type": "connection_test",
                "timestamp": time.time(),
                "source": "connection_manager"
            }
            
            import json
            test_bytes = json.dumps(test_data).encode('utf-8')
            sock.sendto(test_bytes, (target_ip, port))
            
            # 等待响应
            start_time = time.time()
            try:
                response, addr = sock.recvfrom(1024)
                response_data = json.loads(response.decode('utf-8'))
                latency = (time.time() - start_time) * 1000
                
                self.connection_status[f"{target_ip}:{port}"] = {
                    "connected": True,
                    "latency": latency,
                    "last_test": time.time()
                }
                
                print(f"连接测试成功 {target_ip}:{port} 延迟: {latency:.1f}ms")
                return True
                
            except socket.timeout:
                self.connection_status[f"{target_ip}:{port}"] = {
                    "connected": False,
                    "error": "timeout",
                    "last_test": time.time()
                }
                print(f"连接测试超时 {target_ip}:{port}")
                return False
                
        except Exception as e:
            self.connection_status[f"{target_ip}:{port}"] = {
                "connected": False,
                "error": str(e),
                "last_test": time.time()
            }
            print(f"连接测试失败 {target_ip}:{port} - {e}")
            return False
        finally:
            sock.close()
    
    def get_status(self, target_ip=None, port=None):
        """获取连接状态"""
        if target_ip and port:
            return self.connection_status.get(f"{target_ip}:{port}", {"connected": False})
        return self.connection_status
    
    def is_connected(self, target_ip, port):
        """检查是否连接正常"""
        status = self.get_status(target_ip, port)
        return status.get("connected", False)

# 创建全局连接管理器实例
connection_manager = ConnectionManager()

# ===== 配置验证和打印 =====
def print_config():
    """打印当前配置信息"""
    print("=" * 50)
    print("医疗诊断系统配置信息")
    print("=" * 50)
    print(f"PC端IP地址: {PC_IP}")
    print(f"本机IP地址: {get_local_ip()}")
    print("\n网络端口配置:")
    for name, port in NETWORK_PORTS.items():
        print(f"  {name}: {port}")
    
    print(f"\n摄像头配置: {CAMERA_CONFIG['WIDTH']}x{CAMERA_CONFIG['HEIGHT']} @ {CAMERA_CONFIG['FPS']}fps")
    print(f"音频配置: {AUDIO_CONFIG['RATE']}Hz, {AUDIO_CONFIG['CHANNELS']}声道")
    print(f"保存目录: {SYSTEM_CONFIG['SAVE_DIR']}")
    print("=" * 50)

def validate_config():
    """验证配置有效性"""
    errors = []
    
    # 验证IP地址格式
    try:
        socket.inet_aton(PC_IP)
    except socket.error:
        errors.append(f"无效的PC IP地址: {PC_IP}")
    
    # 验证端口范围
    for name, port in NETWORK_PORTS.items():
        if not (1024 <= port <= 65535):
            errors.append(f"端口 {name}={port} 超出有效范围 (1024-65535)")
    
    # 验证目录
    save_dir = SYSTEM_CONFIG['SAVE_DIR']
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
            print(f"创建保存目录: {save_dir}")
        except Exception as e:
            errors.append(f"无法创建保存目录 {save_dir}: {e}")
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("配置验证通过")
        return True

if __name__ == "__main__":
    print_config()
    validate_config()