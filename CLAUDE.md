# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AI 眼科辅助诊断系统，采用 **PC + 开发板** 分离式架构。PC 端运行 PyQt5 图形界面，集成了基于 YOLO 的眼部疾病分类、DeepSeek API 诊疗对话、语音交互和屏幕共享功能。开发板端负责摄像头采集，并通过 WiFi 接收 PC 画面和控制指令。

## 系统架构

```
visualization_test2.py          ← PC 端主入口（PyQt5 GUI 单体应用，~6500 行）
    ├── YOLO 推理（ultralytics + 自定义 AKConv 模块）
    ├── SmartVoiceManager（Vosk 离线 → Google 在线回退）
    ├── DeepSeekAPI（AI 治疗建议 + 对话式问诊）
    ├── HistoryDB（SQLite，存储诊断历史）
    ├── 网络服务：屏幕共享、诊断服务、语音服务
    └── 从 configs/system_config.py 导入配置（可选，失败则用默认值）

src/pc/                          ← PC 端独立服务
    ├── pc_diagnosis_server.py   ← UDP 接收开发板图像 → 调用主程序模型 → 返回诊断结果
    └── pc_voice_server.py       ← 语音处理服务

src/network/                     ← WiFi 屏幕共享 + 远程控制（全部 UDP）
    ├── wifi_pc_sender_with_mouse.py  ← PC 端屏幕捕获 + 接收鼠标控制（主用版本）
    ├── wifi_pc_sender_2.0.py         ← 简化版发送端（无鼠标控制）
    ├── wifi_pc_receiver_2.0.py       ← 开发板端屏幕接收 + 触摸→鼠标转发
    └── wifi1.0/                      ← 旧版归档（不再使用）

src/board/                       ← 开发板代码（树莓派）
    ├── board_camera_integration.py   ← 摄像头采集 + UDP 分包发送至 PC
    ├── board_integrated_system.py    ← 开发板主入口
    ├── board_local_model.py          ← 开发板本地模型推理（ONNX）
    └── board_voice_interaction.py    ← 开发板端语音交互
```

**通信协议：** 所有实时数据（屏幕、摄像头、命令、诊断结果）均使用 **UDP** 通信（非 TCP）。
屏幕/摄像头数据分包传输（每包最大 1400 字节，包头 8 字节：4 字节 packet_id + 4 字节 total_packets）。
PC 端 IP 硬编码为 `172.20.10.3`，开发板为 `172.20.10.8`。端口范围 5000–5008。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动主界面（主要入口）
python visualization_test2.py

# 一键启动所有 PC 端服务（主界面 + 诊断服务器 + 屏幕共享）
python scripts/start_system.py

# 交互式启动助手（自动检测环境）
python scripts/quick_start.py

# 单独启动 PC 端服务
python src/pc/pc_diagnosis_server.py
python src/pc/pc_voice_server.py
python src/network/wifi_pc_sender_with_mouse.py

# 开发板端
python src/network/wifi_pc_receiver_2.0.py
python src/board/board_camera_integration.py

# 训练（在 ultralytics-main/ 目录下，注意硬编码路径需修改）
cd ultralytics-main && python eyes_train.py

# YOLO 命令行（使用本地 ultralytics-main/，非 pip 包）
yolo predict model=models/custom/AKConv_best_moudle/best.pt source=<image>
yolo val model=models/custom/AKConv_best_moudle/best.pt data=data/eyes_dataset.yaml
```

## 模型

基于 **YOLO11 分类** 模型，集成自定义 **AKConv**（可变形卷积）模块。

- `models/custom/AKConv_best_moudle/best.pt` — 主要权重（注意：README 中误写为 `self_model/AKConv_best_moudle/`）
- `models/custom/AKConv_best_moudle/best.onnx` — ONNX 导出（开发板推理用）
- `models/custom/common/best.pt` — 备用权重

**AKConv 注册机制：** 自定义模块定义在 `ultralytics-main/ultralytics/nn/modules/akconv.py`，通过以下两步注册到 YOLO：
1. `ultralytics-main/ultralytics/nn/modules/__init__.py` 第 19 行：`from .akconv import *`
2. `ultralytics-main/ultralytics/nn/tasks.py` 将 `AKConv` 和 `C3k2_AKConv` 加入 `parse_model()` 的可用模块字典

这意味着模型 YAML 配置中可以直接引用 `C3k2_AKConv` 模块，训练/推理时会自动解析。

`ultralytics-main/` 是整个 YOLO 框架的本地副本（包含 AKConv 修改），**不是** pip 安装的 `ultralytics` 包。运行时需要确保 `ultralytics-main/` 在 `sys.path` 中（或在项目根目录运行）。

**8 种疾病类别：** A（正常）、C（白内障）、D（糖尿病视网膜病变）、G（青光眼）、H（高血压性视网膜病变）、M（黄斑病变）、N（视神经病变）、O（其他）。

## 配置系统

存在两套并行的配置：

| 文件 | 用途 | 使用者 |
|------|------|--------|
| `configs/system_config.py` | Python 配置 + `ConnectionManager` 类 | `visualization_test2.py`（通过 `from system_config import ...` 导入） |
| `configs/system_config.json` | JSON 配置（仅 6 个端口） | `scripts/start_system.py` |

**重要：** `visualization_test2.py` 第 43 行 `from system_config import ...` 直接导入（无 `configs.` 前缀），因此需要 `configs/` 目录在 `PYTHONPATH` 中，或从 `configs/` 目录运行。导入失败时程序会使用硬编码默认值继续运行。

端口定义在 `configs/system_config.py` 中有完整 8 个端口（5000–5008），但各独立脚本（`pc_diagnosis_server.py`、`wifi_pc_sender_with_mouse.py` 等）各自硬编码端口号，修改端口需同步多处。

## DeepSeek API 集成

`visualization_test2.py` 中的 `DeepSeekAPI` 类（第 528 行）提供：
- **治疗建议**：根据诊断结果（疾病名 + 置信度）调用 DeepSeek API 生成专业建议
- **AI 对话**：自由文本问答，支持语音输入 → DeepSeek 回复 → TTS 朗读
- Endpoint：`https://api.deepseek.com/v1/chat/completions`，模型：`deepseek-chat`
- API Key 存储：base64 编码后保存在 `saved_api_key.txt`（已在 `.gitignore` 中）

## 语音系统

`SmartVoiceManager`（`visualization_test2.py` 第 158 行）：
1. **Vosk** 离线中文模型 — 延迟加载（首次使用时才加载模型文件），搜索路径优先级：
   - `$VOSK_MODEL_PATH` 环境变量
   - `./vosk-model-small-cn-0.22/`
   - `./vosk-model-cn-0.22/`
2. **Google Speech Recognition** — 在线回退
3. **pyttsx3** — 语音合成朗读 AI 回复

## 数据存储

- **SQLite 历史记录**：`~/EyeDiseaseDetectorHistory/history.db`（用户主目录下，非项目目录）
- 自动从旧版 JSON 文件迁移（迁移后重命名为 `.bak`）
- **图像保存目录**：`medical_images/`（由 `SYSTEM_CONFIG["SAVE_DIR"]` 配置）

## 关键注意事项

- 主入口是 `visualization_test2.py`，**不是** `visualization_test1.py` 或 `src/visualization/` 下的历史版本（1.0–5.0，仅供参考）
- `src/utils/` 存放的是一次性测试/调试脚本，非通用工具库
- `services/` 目录仅包含过期的 `.pyc` 字节码文件，可忽略
- API 密钥文件 `saved_api_key.txt` 已在 `.gitignore` 中，切勿提交
- 根目录下的 `temp_image_*.png` 是运行时产物，可安全删除
- `os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'` 必须在所有导入之前设置（macOS OpenMP 兼容性）
- matplotlib 后端强制设为 `Qt5Agg`，修改 GUI 相关代码时不要更改此后端
- `pyautogui.FAILSAFE = False` 被显式禁用（屏幕共享鼠标控制需要）
- 各脚本中的 IP 地址硬编码分散在多处，修改 IP 需要全局搜索替换
