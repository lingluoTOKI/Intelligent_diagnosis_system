# 开发板医疗诊断系统集成指南

## 📋 概述

本指南介绍如何将开发板摄像头功能集成到现有的医疗诊断系统中，并优化触摸屏显示延迟。

## 🏗️ 系统架构

```
开发板端                     PC端
┌─────────────────┐         ┌─────────────────┐
│ 摄像头捕获      │         │ 图像接收        │
│ ↓               │         │ ↓               │
│ 图像增强        │ ──WiFi─→ │ AI诊断分析      │
│ ↓               │         │ ↓               │
│ 网络传输        │         │ 结果生成        │
│ ↓               │         │ ↓               │
│ 结果显示        │ ←─WiFi── │ 结果发送        │
└─────────────────┘         └─────────────────┘

触摸屏控制 ←─────── 延迟优化 ─────────→ 屏幕投影
```

## 📁 文件结构

```
medical_diagnosis_system/
├── visualization_test2.py          # 主医疗诊断系统
├── link/
│   ├── wifi_pc_sender_2.0.py      # PC端屏幕发送
│   └── wifi_pc_receiver_2.0.py    # 开发板端接收
├── board_camera_integration.py     # 开发板摄像头集成
├── pc_diagnosis_server.py          # PC端诊断服务器
├── latency_optimizer.py            # 延迟优化模块
└── integration_guide.md            # 本文档
```

## 🚀 快速开始

### 1. 环境准备

**PC端依赖:**
```bash
pip install opencv-python numpy PyQt5 torch torchvision
pip install vosk pyttsx3 speech_recognition
```

**开发板端依赖:**
```bash
pip install opencv-python numpy requests
```

### 2. 网络配置

确保开发板和PC在同一网络中，修改以下配置：

**开发板端配置** (`board_camera_integration.py`):
```python
PC_IP = "172.20.10.3"      # 修改为您PC的实际IP
CAMERA_PORT = 5002         # 摄像头数据端口
DIAGNOSIS_PORT = 5003      # 诊断结果端口
COMMAND_PORT = 5004        # 命令控制端口
```

**PC端配置** (`pc_diagnosis_server.py`):
```python
SERVER_IP = "0.0.0.0"      # 监听所有接口
CAMERA_PORT = 5002         # 与开发板保持一致
DIAGNOSIS_PORT = 5003      # 与开发板保持一致
COMMAND_PORT = 5004        # 与开发板保持一致
```

### 3. 启动顺序

#### 第一步：启动PC端服务

```bash
# 1. 启动主医疗诊断系统
python visualization_test2.py

# 2. 启动诊断服务器（新终端）
python pc_diagnosis_server.py

# 3. 启动屏幕发送服务（新终端，用于触摸屏显示）
cd link
python wifi_pc_sender_2.0.py
```

#### 第二步：启动开发板端服务

```bash
# 1. 启动屏幕接收（用于触摸控制）
cd link
python wifi_pc_receiver_2.0.py &

# 2. 启动摄像头诊断服务
python board_camera_integration.py
```

## 🔧 功能详解

### 1. 摄像头集成功能

#### 主要特性：
- **高质量图像捕获**: 支持图像增强处理
- **实时诊断**: 将图像发送到PC端进行AI分析
- **结果显示**: 在开发板上显示诊断结果
- **历史记录**: 自动保存诊断记录和图像

#### 使用方法：
```python
from board_camera_integration import MedicalDiagnosisBoard

# 创建诊断系统
board = MedicalDiagnosisBoard()
board.initialize()

# 单次诊断
result = board.capture_and_diagnose()

# 连续监控
board.continuous_monitoring(interval=5)

# 交互模式
board.interactive_mode()
```

### 2. PC端诊断服务

#### 主要特性：
- **多客户端支持**: 可同时为多个开发板提供服务
- **AI模型集成**: 直接调用现有的医疗诊断模型
- **性能监控**: 实时统计服务器运行状态
- **错误处理**: 完善的异常处理和重试机制

#### 服务状态监控：
```
📊 服务器运行状态
==================================================
⏱️  运行时间: 02:15
📋 总请求数: 45
✅ 成功诊断: 42
❌ 失败诊断: 3
🔗 连接设备: 2
⏳ 待处理队列: 0
📈 成功率: 93.3%
==================================================
```

### 3. 延迟优化系统

#### 优化策略：

**运动预测补偿:**
```python
# 预测未来位置减少延迟感知
predictor = MotionPredictor()
predictor.add_position(x, y, timestamp)
future_pos = predictor.predict_position(0.03)  # 预测30ms后位置
```

**自适应质量控制:**
```python
# 根据性能动态调整图像质量
controller = AdaptiveQualityController()
controller.update_metrics(fps=30, latency_ms=45)
optimal_quality = controller.get_optimal_quality()
```

**智能区域更新:**
```python
# 只传输变化的区域减少数据量
preprocessor = FramePreprocessor()
motion_areas = preprocessor.detect_motion_areas(frame)
optimized_data = preprocessor.create_optimized_frame(frame, controller)
```

## 🔨 集成到现有系统

### 1. 修改主系统 (visualization_test2.py)

在主窗口类中添加摄像头诊断功能：

```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... 现有初始化代码 ...
        
        # 添加开发板连接状态
        self.board_connected = False
        self.diagnosis_server = None
        
        # 启动诊断服务器
        self.start_diagnosis_server()
    
    def start_diagnosis_server(self):
        """启动诊断服务器"""
        try:
            import threading
            from pc_diagnosis_server import DiagnosisServer
            
            self.diagnosis_server = DiagnosisServer()
            server_thread = threading.Thread(
                target=self.diagnosis_server.start_server, 
                daemon=True
            )
            server_thread.start()
            
            print("✅ 诊断服务器已启动")
            
        except Exception as e:
            print(f"❌ 诊断服务器启动失败: {e}")
    
    def init_ui(self):
        # ... 现有UI代码 ...
        
        # 添加开发板状态指示
        self.board_status_label = QLabel("开发板: 未连接")
        self.board_status_label.setStyleSheet(f"color: {self.text_color};")
        self.statusBar().addPermanentWidget(self.board_status_label)
```

### 2. 添加菜单项

```python
def init_ui(self):
    # ... 现有代码 ...
    
    # 添加开发板菜单
    board_menu = self.menuBar().addMenu("开发板")
    
    board_status_action = QAction("连接状态", self)
    board_status_action.triggered.connect(self.show_board_status)
    board_menu.addAction(board_status_action)
    
    board_config_action = QAction("配置开发板", self)
    board_config_action.triggered.connect(self.configure_board)
    board_menu.addAction(board_config_action)

def show_board_status(self):
    """显示开发板状态"""
    if self.diagnosis_server:
        stats = self.diagnosis_server.stats
        status_text = f"""
        开发板连接状态:
        
        📱 连接设备数: {len(stats['connected_clients'])}
        📋 总诊断请求: {stats['total_requests']}
        ✅ 成功诊断: {stats['successful_diagnoses']}
        ❌ 失败诊断: {stats['failed_diagnoses']}
        ⏰ 运行时间: {time.time() - stats['start_time']:.0f} 秒
        """
        
        self.show_message_box("开发板状态", status_text)
```

### 3. 延迟优化集成

修改现有的屏幕发送程序：

```python
# 在 wifi_pc_sender_2.0.py 中集成优化
from latency_optimizer import LatencyOptimizer

class VideoEncoder:
    def __init__(self, width, height):
        # ... 现有代码 ...
        
        # 添加延迟优化器
        self.latency_optimizer = LatencyOptimizer()
        
    def _encode_software(self, frame):
        # 使用优化器处理帧
        optimized_data, frame_type = self.latency_optimizer.optimize_frame_for_transmission(frame)
        
        # ... 其余编码逻辑 ...
```

修改开发板接收程序：

```python
# 在 wifi_pc_receiver_2.0.py 中集成优化
from latency_optimizer import OptimizedReceiver

# 替换原有的帧缓存
frame_buffer = OptimizedReceiver((1920, 1080))

def decode_and_display(data, frame_seq):
    # 使用优化的接收器
    frame_buffer.add_frame_data(data, 'full_frame')
    # ... 其余显示逻辑 ...
```

## 📊 性能监控

### 1. 延迟测量

```python
def measure_latency():
    """测量端到端延迟"""
    start_time = time.time()
    
    # 发送测试数据包
    test_packet = b'LATENCY_TEST_' + str(start_time).encode()
    sock.sendto(test_packet, (target_ip, target_port))
    
    # 等待响应
    response, addr = sock.recvfrom(1024)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    print(f"延迟: {latency_ms:.2f}ms")
    return latency_ms
```

### 2. 性能统计

```python
class PerformanceMonitor:
    def __init__(self):
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.latency_samples = []
        
    def update_fps(self):
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:
            fps = self.fps_counter / elapsed
            print(f"FPS: {fps:.1f}")
            
            self.fps_counter = 0
            self.fps_start_time = time.time()
            return fps
        return None
    
    def add_latency_sample(self, latency_ms):
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > 100:
            self.latency_samples = self.latency_samples[-100:]
        
        avg_latency = sum(self.latency_samples) / len(self.latency_samples)
        print(f"平均延迟: {avg_latency:.1f}ms")
```

## 🔍 故障排除

### 常见问题及解决方案

**1. 网络连接问题**
```bash
# 检查网络连通性
ping 172.20.10.3

# 检查端口是否开放
netstat -an | grep 5002
```

**2. 摄像头无法打开**
```python
# 检查摄像头设备
import cv2
cap = cv2.VideoCapture(0)
print(f"摄像头可用: {cap.isOpened()}")
cap.release()
```

**3. 诊断服务器无响应**
```bash
# 检查服务器进程
ps aux | grep pc_diagnosis_server

# 检查服务器日志
tail -f diagnosis_server.log
```

**4. 延迟过高**
- 降低图像分辨率
- 减少JPEG压缩质量
- 启用运动预测
- 优化网络配置

## 📈 性能优化建议

### 1. 网络优化
- 使用5GHz WiFi频段
- 减少网络设备间距离
- 避免网络拥塞时段
- 考虑有线连接

### 2. 系统优化
- 增加系统内存
- 使用SSD存储
- 关闭不必要的后台程序
- 调整系统优先级

### 3. 代码优化
- 使用多线程处理
- 启用硬件加速
- 优化图像处理算法
- 减少内存拷贝

## 🔧 配置文件示例

创建 `config.json` 配置文件：

```json
{
    "network": {
        "pc_ip": "172.20.10.3",
        "camera_port": 5002,
        "diagnosis_port": 5003,
        "command_port": 5004
    },
    "camera": {
        "width": 640,
        "height": 480,
        "fps": 30,
        "quality": 85
    },
    "optimization": {
        "target_fps": 60,
        "motion_prediction": true,
        "adaptive_quality": true,
        "max_latency_ms": 50
    },
    "diagnosis": {
        "model_path": "models/eye_disease_model.pth",
        "confidence_threshold": 0.6,
        "timeout_seconds": 30
    }
}
```

## 📞 技术支持

如果在集成过程中遇到问题，请检查：

1. **网络配置**: 确保IP地址和端口配置正确
2. **依赖库**: 验证所有必需的Python包已安装
3. **权限设置**: 确保摄像头和网络访问权限
4. **防火墙**: 检查防火墙是否阻止了相应端口
5. **系统资源**: 确保有足够的CPU和内存资源

---

**注意**: 本系统仅用于辅助诊断，不能替代专业医疗诊断。所有诊断结果应由专业医护人员确认。
