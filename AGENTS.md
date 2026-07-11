# AGENTS.md

本文件为 Codex（Codex.ai/code）在此仓库中工作时提供指导。

## 项目概述

AI 眼科辅助诊断系统，采用 **PC + 开发板** 分离式架构。PC 端运行 PyQt5 图形界面，集成了基于 YOLO 的眼部疾病分类、语音交互和屏幕共享功能。开发板端负责摄像头采集，并通过 WiFi 接收 PC 画面和控制指令。

## 系统架构

```
visualization_test2.py          ← PC 端主入口（PyQt5 GUI 单体应用）
    ├── YOLO 推理（ultralytics + 自定义 AKConv 模型）
    ├── 语音：SmartVoiceManager（Vosk 离线 → Google 在线回退）
    ├── DeepSeek API 提供 AI 治疗建议
    ├── 网络：屏幕共享、诊断服务、语音服务
    └── configs/system_config.py（硬编码 IP/端口）

src/pc/                          ← PC 端服务
    ├── pc_diagnosis_server.py   ← 接收开发板图像，执行诊断
    └── pc_voice_server.py       ← 语音处理服务

src/network/                     ← WiFi 通信
    ├── wifi_pc_sender_with_mouse.py  ← PC 屏幕投屏 + 远程鼠标控制
    ├── wifi_pc_sender_2.0.py         ← 简化版发送端
    └── wifi_pc_receiver_2.0.py       ← 开发板端屏幕接收

src/board/                       ← 开发板代码（树莓派）
    ├── board_camera_integration.py   ← 摄像头采集 + 发送至 PC
    ├── board_integrated_system.py    ← 开发板主入口
    ├── board_local_model.py          ← 开发板本地模型推理
    └── board_voice_interaction.py    ← 开发板端语音交互
```

PC 端 IP 硬编码为 `172.20.10.3`，开发板为 `172.20.10.8`。所有端口在 `configs/system_config.py` 中定义（范围 5000–5008）。实时数据（屏幕、摄像头）使用 UDP 通信，命令控制使用 TCP。

## 模型

基于 **YOLO11 分类** 模型，集成自定义 AKConv 模块。权重文件位置：

- `models/custom/AKConv_best_moudle/best.pt` — 主要权重文件
- `models/custom/AKConv_best_moudle/best.onnx` — 用于开发板推理的 ONNX 导出
- `models/custom/common/best.pt` — 备用权重

AKConv 模块代码位于 `ultralytics-main/ultralytics/nn/modules/akconv.py`。`ultralytics-main/` 目录是 YOLO 框架的本地副本（尽管 `requirements.txt` 中列出了 ultralytics，但实际使用的是该本地版本）。

**8 种疾病类别：** A（正常）、C（白内障）、D（糖尿病视网膜病变）、G（青光眼）、H（高血压性视网膜病变）、M（黄斑病变）、N（视神经病变）、O（其他）。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动主界面（主要入口）
python visualization_test2.py

# 启动系统编排器（启动所有 PC 端服务）
python scripts/start_system.py

# 快速启动助手
python scripts/quick_start.py

# 单独启动 PC 端服务
python src/pc/pc_diagnosis_server.py
python src/pc/pc_voice_server.py
python src/network/wifi_pc_sender_with_mouse.py

# 开发板端
python src/network/wifi_pc_receiver_2.0.py
python src/board/board_camera_integration.py

# 训练（在 ultralytics-main/ 目录下执行）
cd ultralytics-main && python eyes_train.py

# YOLO 命令行
yolo predict model=models/custom/AKConv_best_moudle/best.pt source=<image>
yolo val model=models/custom/AKConv_best_moudle/best.pt data=data/eyes_dataset.yaml
```

## 关键配置

`configs/system_config.py` 包含所有网络配置。`configs/system_config.json` 是启动脚本使用的简化 JSON 版本。系统启动时会验证配置 —— PC 与开发板之间的 IP/端口不匹配会导致通信失败。

- `PC_IP`：硬编码为 `172.20.10.3`
- `BOARD_IP`：`172.20.10.8`（在发送端脚本中）
- 接收图像的保存目录：`medical_images/` 或 `SYSTEM_CONFIG["SAVE_DIR"]`

## 语音系统

`SmartVoiceManager`（定义在 `visualization_test2.py` 中）处理：
1. **Vosk** 离线中文模型（`vosk-model-cn-0.22/`）—— 延迟加载以避免启动延迟
2. **Google 语音识别** —— 在线回退方案
3. **pyttsx3** —— 语音合成，用于朗读 AI 回复

语音依赖：`speechrecognition`、`pyttsx3`、`vosk`、`pyaudio`（系统级）。

## 数据集

位于 `data/eyes_val/`，子目录为 A/C/D/G/H/M/N/O（每类一个目录）。配置文件为 `data/eyes_dataset.yaml`。训练使用标准 YOLO 分类格式。

## 重要约定

- 主入口是 `visualization_test2.py`，**不是** `visualization_test1.py` 或 `ultralytics-main/` 下的任何文件
- `src/visualization/` 包含历史版本（1.0–5.0）—— 仅供参考，不用于生产
- `src/utils/` 存放的是一次性测试/调试脚本，而非工具库
- `services/` 目录只包含过期的 `.pyc` 字节码文件 —— 可忽略
- README 中引用的 `models/custom/self_model/` 实际路径为 `models/custom/AKConv_best_moudle/`
- API 密钥（DeepSeek）存储在 `saved_api_key.txt` —— 切勿提交此文件
- 根目录下的 `temp_image_*.png` 是运行时产物，可以安全删除
