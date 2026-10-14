# AGENTS.md

本文件为 Codex（Codex.ai/code）在此仓库中工作时提供指导。

## 项目概述

AI 眼科辅助诊断系统，采用 **PC + 开发板** 分离式架构。PC 端运行 PyQt5 图形界面，集成了基于 YOLO 的眼部疾病分类、语音交互和屏幕共享功能。开发板端负责摄像头采集，并通过 WiFi 接收 PC 画面和控制指令。

## 当前业务模块

- `src/core/classification.py`：权重加载、八类验证、分类与明确失败状态。
- `src/core/llm.py`：DeepSeek、受限会话记忆、响应来源和超时。
- `src/core/history.py`：SQLite与显式演示记录；GUI不直接执行SQL。
- `configs/app_config.json`：权重、框架、设备及可选功能。
- GUI顶部“系统设置 → 运行与网络”可编辑配置，重启生效。
- 示例配置默认关闭语音和开发板通信；本机配置可能开启；硬件功能没有实机验收证据。

## 系统架构

```
visualization_test2.py          ← PC 端主入口（PyQt5 GUI 单体应用）
    ├── YOLO 推理（ultralytics，当前权重为原生 YOLO11-cls）
    ├── 语音：SmartVoiceManager（Vosk 离线 → Google 在线回退）
    ├── DeepSeek API 提供辅助说明与科普问答
    ├── 网络：屏幕共享、诊断服务、语音服务
    └── configs/（运行与网络配置）

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

PC局域网IP通过活动物理网卡识别或面板手动设置，网络配置保存在`configs/system_config.json`，由Python配置模块读取。PC实时数据及命令使用UDP；不要把代理/虚拟网卡地址当作物理局域网地址。

## 模型

基于 **YOLO11-cls 分类** 模型。权重文件位置：

- `models/custom/AKConv_best_moudle/best.pt` — 主要权重文件（实为原生 YOLO11-cls 结构）
- `models/custom/AKConv_best_moudle/best.onnx` — 用于开发板推理的 ONNX 导出
- `models/custom/common/best.pt` — 备用权重

> **关于 AKConv：** 模块代码位于 `ultralytics-main/ultralytics/nn/modules/akconv.py`，并已完成框架注册（`__init__.py` 导入 + `tasks.py` 的 `parse_model()`）。默认部署权重不含AKConv；YOLO11与YOLO26框架的AKConv配置及实验权重需要分别核实，不能据文件夹名推断结构。AKConv 原始工作见 Xin Zhang et al., *AKConv: Convolutional Kernel with Arbitrary Sampled Shapes and Arbitrary Number of Parameters*, arXiv:2311.11587v2。

`ultralytics-main/` 目录是 YOLO 框架的本地副本（尽管 `requirements.txt` 中列出了 ultralytics，但实际使用的是该本地版本）。

**8 种疾病类别：** A（AMD，年龄相关性黄斑变性）、C（白内障）、D（糖尿病视网膜病变）、G（青光眼）、H（高血压性视网膜病变）、M（近视性黄斑病变）、N（正常）、O（其他）。以 `visualization_test2.py` 的 `EyeDiseaseDetector.class_names` 为准。

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

# 训练（eyes_train.py 未提交到仓库，需自行编写）

# YOLO 命令行
yolo predict model=models/custom/AKConv_best_moudle/best.pt source=<image>
yolo val model=models/custom/AKConv_best_moudle/best.pt data=data
```

## 关键配置

`configs/system_config.py` 包含所有网络配置。`configs/system_config.json` 是启动脚本使用的简化 JSON 版本。系统启动时会验证配置 —— PC 与开发板之间的 IP/端口不匹配会导致通信失败。

- `PC_IP`：从活动局域网网卡自动识别，或读取保存的手动 IPv4。
- `BOARD_IP`：以系统设置保存的地址为准；模板为空，实机使用前配置。
- 接收图像的保存目录：`medical_images/` 或 `SYSTEM_CONFIG["SAVE_DIR"]`

## 语音系统

`SmartVoiceManager`（定义在 `visualization_test2.py` 中）处理：
1. **Vosk** 离线中文模型（`vosk-model-cn-0.22/`）—— 延迟加载以避免启动延迟
2. **Google 语音识别** —— 在线回退方案
3. **Edge TTS MP3** —— 自然音色与应用内播放；缓存相同文本；失败时按当前播放器实现回退。

语音依赖见 `requirements-voice.txt`，包括 `edge-tts`、`speechrecognition`、`pyttsx3`、`vosk`、`pyaudio`。

## 数据集

分类数据位于`data/train/`和`data/val/`，类别为A/C/D/G/H/M/N/O。分类data参数传目录，`eyes_dataset.yaml`仅用于说明。

## 重要约定

- 主入口是 `visualization_test2.py`，**不是** `visualization_test1.py` 或 `ultralytics-main/` 下的任何文件
- `src/visualization/visualization*.py` 是历史版本；同目录workbench、conversation、analysis_pages等为当前生产模块
- `src/utils/` 存放的是一次性测试/调试脚本，而非工具库
- 原 `services/` 的过期 `.pyc` 已归档至 `runtime/local_install_cache/2026-10-14/services_legacy/`，当前入口不导入它们。
- README 中引用的 `models/custom/self_model/` 实际路径为 `models/custom/AKConv_best_moudle/`
- API 密钥（DeepSeek）存储在 `saved_api_key.txt` —— 切勿提交此文件
- 根目录下的 `temp_image_*.png` 是运行时产物，可以安全删除

## PC验证

使用`D:/Anaconda3/envs/yolov10/python.exe`运行GUI与系统测试，实验训练环境独立。运行步骤见`docs/guide/PC运行与测试说明.md`。任何分类失败都不得返回默认疾病、保存成功历史或生成当前图像报告。
