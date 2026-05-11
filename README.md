# 智能医疗诊断系统

基于深度学习的眼部疾病智能诊断系统，支持开发板-PC端协同工作，提供完整的AI医疗诊断解决方案。

## 🎯 系统特性

### 核心功能
- **AI智能诊断**: 基于深度学习的眼部疾病检测
- **实时语音识别**: 支持中文语音输入和AI对话
- **开发板协同**: 支持开发板摄像头拍摄和远程诊断
- **屏幕共享**: 开发板可实时查看和控制PC端
- **历史记录**: 完整的诊断历史和数据分析

### 技术特性
- **多平台支持**: Windows/Linux/Mac兼容
- **低延迟优化**: 触摸屏控制延迟优化
- **网络传输**: WiFi无线传输，支持实时数据交换
- **智能缓存**: 图像处理和网络传输优化
- **容错机制**: 网络断线重连和错误恢复

## 🏗️ 系统架构

```
PC端                          开发板端
┌─────────────────────┐      ┌─────────────────────┐
│ 主诊断系统           │      │ 摄像头采集           │
│ - AI模型推理        │      │ - 图像增强          │
│ - 语音识别          │      │ - 实时预览          │
│ - 历史管理          │ <----│ - 触摸控制          │
│                     │ WiFi │                     │
│ 诊断服务器           │      │ 屏幕接收             │
│ - 图像接收          │      │ - 实时显示          │
│ - 结果返回          │ ---->│ - 鼠标控制          │
│                     │      │ - 全屏显示          │
│ 屏幕发送             │      │                     │
│ - 画面传输          │      │ 摄像头客户端         │
│ - 鼠标接收          │ <----│ - 拍照上传          │
└─────────────────────┘      └─────────────────────┘
```

## 📁 项目结构

```
intelligent_diagnosis_system/
├── 📋 主程序文件
│   ├── visualization_test2.py          # 主诊断系统界面
│   ├── pc_diagnosis_server.py          # PC端诊断服务器
│   ├── board_camera_integration.py     # 开发板摄像头集成
│   └── latency_optimizer.py            # 延迟优化模块
│
├── 🌐 网络通信模块
│   └── link/
│       ├── wifi_pc_sender_with_mouse.py    # PC端屏幕发送(含鼠标控制)
│       ├── wifi_pc_sender_2.0.py           # PC端屏幕发送(基础版)
│       └── wifi_pc_receiver_2.0.py         # 开发板端屏幕接收
│
├── 🚀 启动脚本
│   ├── start_system.py                # 主启动脚本
│   ├── start_system_safe.py           # Windows兼容启动脚本
│   └── start_board.sh                 # 开发板启动脚本
│
├── ⚙️ 配置文件
│   ├── system_config.json             # 系统配置
│   └── saved_api_key.txt              # API密钥存储
│
├── 🧪 测试工具
│   └── test/
│       ├── fix_unicode.py             # Unicode编码修复工具
│       ├── touchscreen_troubleshoot.py # 触摸屏问题诊断
│       ├── test_api.py                # API测试工具
│       └── [其他测试文件...]
│
├── 📚 文档资料
│   └── markdown/
│       ├── integration_guide.md       # 集成指南
│       ├── 语音对话使用说明.md        # 语音功能说明
│       └── [其他文档...]
│
└── 🤖 AI模型 (需要单独下载)
    ├── ultralytics-main/              # YOLO模型
    └── vosk-model-cn-0.22/           # 中文语音识别模型
```

## 🚀 快速开始

### 环境要求

**PC端 (Windows/Linux/Mac):**
- Python 3.9+
- 8GB+ RAM
- 支持CUDA的显卡 (可选，用于AI加速)

**开发板端 (推荐树莓派4B+):**
- Python 3.7+
- 4GB+ RAM
- 摄像头模块
- 触摸屏显示器

### 依赖安装

**PC端依赖:**
```bash
pip install torch torchvision
pip install opencv-python numpy PyQt5
pip install vosk pyttsx3 speech_recognition
pip install ultralytics requests
pip install mss pyautogui
```

**开发板端依赖:**
```bash
pip install opencv-python numpy requests
pip install socket threading
```

### 网络配置

1. **确保PC和开发板在同一网络**
2. **配置IP地址** (修改 `system_config.json`):
```json
{
    "network": {
        "pc_ip": "192.168.1.100",      # PC的实际IP
        "board_ip": "192.168.1.101",   # 开发板的实际IP
        "camera_port": 5002,
        "diagnosis_port": 5003,
        "command_port": 5004,
        "screen_port": 5000,
        "control_port": 5001
    }
}
```

### 模型下载

1. **AI诊断模型**: 下载训练好的眼部疾病检测模型
2. **语音识别模型**: 下载VOSK中文模型到 `vosk-model-cn-0.22/`

## 🎮 使用方法

### 方式一：一键启动 (推荐)

**PC端:**
```bash
# 启动所有PC端服务
python start_system.py
```

**开发板端:**
```bash
# 执行生成的启动脚本
./start_board.sh

# 或手动启动
cd link
python wifi_pc_receiver_2.0.py &
python ../board_camera_integration.py
```

### 方式二：手动启动

**PC端 (需要3个终端):**
```bash
# 终端1: 主诊断系统
python visualization_test2.py

# 终端2: 诊断服务器
python pc_diagnosis_server.py

# 终端3: 屏幕共享
cd link
python wifi_pc_sender_with_mouse.py
```

**开发板端:**
```bash
# 终端1: 屏幕接收
cd link
python wifi_pc_receiver_2.0.py

# 终端2: 摄像头诊断
python board_camera_integration.py
```

## 📱 功能使用

### 主诊断系统
1. **图像诊断**: 上传眼部图像，获取AI诊断结果
2. **语音对话**: 按住语音按钮，语音咨询医疗问题
3. **历史查看**: 查看历史诊断记录和趋势分析
4. **批量处理**: 批量处理多张图像

### 开发板功能
1. **实时拍摄**: 使用摄像头拍摄眼部图像
2. **远程诊断**: 将图像发送到PC端进行AI分析
3. **触摸控制**: 通过触摸屏控制PC端界面
4. **结果查看**: 在开发板上查看诊断结果

### 屏幕共享
1. **实时显示**: 开发板实时显示PC端画面
2. **触摸控制**: 触摸屏幕控制PC端鼠标
3. **全屏模式**: 支持全屏显示和窗口模式
4. **低延迟**: 优化的网络传输，减少延迟

## ⚙️ 配置说明

### 系统配置 (`system_config.json`)
```json
{
    "network": {
        "pc_ip": "PC的IP地址",
        "board_ip": "开发板的IP地址",
        "各端口配置": "5000-5004"
    },
    "startup": {
        "auto_detect_ip": true,        # 自动检测IP
        "start_delay": 2,              # 启动延迟
        "max_retries": 3               # 最大重试次数
    },
    "services": {
        "main_system": true,           # 启用主系统
        "diagnosis_server": true,      # 启用诊断服务器
        "screen_sharing": true,        # 启用屏幕共享
        "camera_integration": true     # 启用摄像头集成
    }
}
```

### 网络端口说明
- **5000**: 屏幕数据传输
- **5001**: 鼠标控制指令
- **5002**: 摄像头图像数据
- **5003**: 诊断结果传输
- **5004**: 命令控制

## 🔧 故障排除

### 常见问题

**1. 触摸屏无法控制PC**
```bash
# 检查网络连接
ping PC的IP地址

# 检查PC端是否运行了带鼠标控制的程序
python link/wifi_pc_sender_with_mouse.py

# 运行诊断工具
python test/touchscreen_troubleshoot.py
```

**2. 语音识别不工作**
- 检查麦克风权限
- 确认VOSK模型已下载
- 检查语音设备配置

**3. AI诊断失败**
- 确认AI模型文件完整
- 检查GPU/CPU配置
- 查看错误日志

**4. 网络连接问题**
- 确认防火墙设置
- 检查IP地址配置
- 验证端口是否被占用

### 调试模式
```bash
# 启用详细日志
python start_system.py --debug

# 检查系统状态
python start_system.py --check

# 仅生成开发板命令
python start_system.py --board-only
```

## 📊 性能优化

### 延迟优化
- **运动预测**: 预测触摸位置减少延迟感知
- **自适应质量**: 根据网络状况自动调整图像质量
- **智能缓存**: 多线程处理和帧缓存优化

### 网络优化
- **数据压缩**: JPEG压缩和区域更新
- **错误恢复**: 自动重连和丢包恢复
- **带宽控制**: 动态调整传输参数

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- **YOLOv8/YOLOv10**: AI目标检测框架
- **VOSK**: 离线语音识别引擎
- **OpenCV**: 计算机视觉库
- **PyQt5**: GUI框架
- **DeepSeek**: AI对话API

## 📞 技术支持

如遇问题，请：

1. 查看 [故障排除](#🔧-故障排除) 部分
2. 运行诊断工具: `python test/touchscreen_troubleshoot.py`
3. 查看项目文档: `markdown/` 目录
4. 提交 Issue 或联系开发者

---

**注意**: 本系统仅用于辅助诊断，不能替代专业医疗诊断。所有诊断结果应由专业医护人员确认。