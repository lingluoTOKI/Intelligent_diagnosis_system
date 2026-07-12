# Intelligent Diagnosis System

<div align="center">

**AI 眼科辅助诊断系统**

基于 YOLO11 + AKConv 的眼部疾病智能诊断系统，集成 DeepSeek AI 诊疗对话、离线语音识别、PC + 开发板协同与 WiFi 屏幕共享。

[快速开始](#-快速开始) · [系统架构](#-系统架构) · [主界面功能](#-主界面功能) · [模型说明](#-模型说明) · [通信协议](#-通信协议)

</div>

---

## ✨ 项目简介

本项目聚焦于「**眼部图像智能辅助诊断**」，结合 YOLO 视觉推理、中文离线语音识别、DeepSeek API 诊疗对话、局域网通信与开发板联动，构建了一个可演示、可部署的综合系统。

**适用场景：**

- 课程设计 / 毕业设计
- AI 医疗方向原型系统
- PC + 开发板协同教学项目
- 视觉识别 + 语音交互综合实验平台

> ⚠️ 说明：本系统仅用于辅助诊断和教学展示，不能替代专业医生的临床诊断。

---

## 🧭 系统架构

```
visualization_test2.py          ← PC 端主入口（PyQt5 单体 GUI，~6500 行）
    ├── EyeDiseaseDetector      ← YOLO11 + 自定义 AKConv 推理引擎
    ├── ResultProcessor         ← 检测结果解析与可视化
    ├── SmartVoiceManager       ← Vosk 离线优先 → Google 在线回退
    ├── DeepSeekAPI             ← AI 诊疗建议 + 对话式问诊
    ├── MedicalAIService        ← 自由文本医疗咨询
    ├── HistoryDB               ← SQLite 诊断历史存储
    ├── BoardCameraReceiver     ← UDP 接收开发板摄像头数据
    ├── CommandListener         ← UDP 命令控制监听
    └── configs/system_config   ← 端口/网络/摄像头配置

src/pc/                          ← PC 端独立服务
    ├── pc_diagnosis_server.py   ← 接收开发板图像 → YOLO 推理 → 返回结果
    └── pc_voice_server.py       ← 语音识别/合成服务（外部独立进程）

src/network/                     ← WiFi 屏幕共享 + 远程控制（全 UDP）
    ├── wifi_pc_sender_with_mouse.py  ← PC 屏幕捕获 + 接收鼠标控制
    ├── wifi_pc_receiver_2.0.py       ← 开发板屏幕接收 + 触摸→鼠标转发
    └── wifi1.0/                      ← 旧版归档（不再使用）

src/board/                       ← 开发板端（树莓派）
    ├── board_camera_integration.py   ← 摄像头采集 + UDP 分包发送
    ├── board_integrated_system.py    ← 开发板集成入口
    ├── board_local_model.py          ← 本地 ONNX 推理
    └── board_voice_interaction.py    ← 开发板语音交互
```

---

## 🖥️ 主界面功能

程序启动后界面分为 **左侧视觉操作区** 和 **右侧 AI 分析区**：

### 左侧 —— 图像检测与硬件视窗

| 区域 | 说明 |
|------|------|
| 🖼️ 本地图像分析 | 加载本地眼底图像 → YOLO 推理 → 并排显示原图与标注结果 |
| 📱 硬件视窗 | 实时接收开发板摄像头画面、连接管理、截取诊断 |
| 工作流按钮 | `加载模型` → `加载图像` → `开始检测` → `查看报告` 四步操作 |
| 扩展工具 | 批量处理、历史记录、开发板交互、语音服务 |

### 右侧 —— AI 分析与对话

| Tab | 说明 |
|-----|------|
| 🩺 诊疗建议 | 检测完成后生成 DeepSeek AI 专业治疗报告，支持全屏阅览 |
| 💬 医疗问答 | 多轮对话式 AI 问诊，自动附带当前检测结果作为上下文 |
| ⚙️ 系统设置 | DeepSeek API Key 配置、录音时长调节、云端 AI 开关 |

### 其他功能

- **历史记录**：SQLite 存储，支持查看详情、批量删除、趋势分析
- **批量处理**：选中多张眼底图像 → 一键推理
- **语音交互**：点击语音按钮说话 → 自动识别 → AI 回复 → **AI 生成100字摘要**用 edge-tts 语音朗读，详细回复保留在 UI（不弹第三方播放器）

---

## 💾 数据存储

### 诊断历史（SQLite）

检测结果自动保存到 SQLite 数据库，位于用户主目录下：

```
~/EyeDiseaseDetectorHistory/history.db
```

**records 表结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 自增主键 |
| record_id | TEXT | UUID 唯一标识 |
| timestamp | TEXT | 检测时间（ISO 格式） |
| image_path | TEXT | 原始图像路径 |
| disease_name | TEXT | 检测结果（疾病名 或 `[对话] 问题摘要`） |
| confidence | REAL | 置信度（0~1，对话类记录为 0.0） |

**触发保存的场景：**
1. 图像检测完成后自动保存（含疾病名 + 置信度 + 图像路径）
2. 医疗问答每轮对话完成后自动保存（disease_name 为 `[对话] 问题前80字`，confidence 为 0.0）
3. 旧版 JSON 文件（`history.json`）首次启动时自动迁移到 SQLite，迁移后原文件重命名为 `.bak`

**API 封装：** `visualization_test2.py` → `HistoryDB` 类，提供 `add / get_all / delete_by_record_id / delete_all / count / migrate_from_json` 方法。全局单例通过 `get_history_db()` 获取。

### 音频缓存

TTS 合成的中间 MP3 文件为临时文件，播放后系统自动清理。

### API 密钥

DeepSeek API Key 经 base64 编码后存储于 `saved_api_key.txt`（已加入 `.gitignore`），启动时自动加载。

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- Windows / Linux / macOS
- 8GB+ 内存推荐

```bash
# 安装依赖
pip install -r requirements.txt

# 启动主界面
python visualization_test2.py

# 一键启动所有服务
python scripts/start_system.py

# 交互式启动
python scripts/quick_start.py
```

---

## 🤖 模型说明

基于 **YOLO11-cls** 分类模型，集成自定义 **AKConv**（可变形卷积）模块。

**模型文件：**

| 文件 | 用途 |
|------|------|
| `models/custom/AKConv_best_moudle/best.pt` | 主要权重（PyTorch） |
| `models/custom/AKConv_best_moudle/best.onnx` | ONNX 导出（开发板推理） |
| `models/custom/common/best.pt` | 备用权重 |

**AKConv 注册：** 自定义模块定义在 `ultralytics-main/ultralytics/nn/modules/akconv.py`，通过 `__init__.py` 的 `from .akconv import *` 和 `tasks.py` 的模块字典注册，使模型 YAML 可直接引用 `C3k2_AKConv`。

**8 种疾病类别：**

| 代码 | 疾病 | 中文 |
|------|------|------|
| A | AMD | 年龄相关性黄斑变性 |
| C | Cataract | 白内障 |
| D | Diabetic Retinopathy | 糖尿病视网膜病变 |
| G | Glaucoma | 青光眼 |
| H | Hypertensive Retinopathy | 高血压性视网膜病变 |
| M | Myopia | 近视性黄斑病变 |
| N | Normal | 正常眼 |
| O | Other | 其他眼部疾病 |

---

## 📡 通信协议

所有实时数据（屏幕、摄像头、命令、诊断结果）均使用 **UDP** 通信。

**数据分包格式：** 每包最大 1400 字节，包头 8 字节（4 字节 packet_id + 4 字节 total_packets），开发板 → PC 端通过多端口并行传输。

**默认端口：**

| 端口 | 用途 |
|------|------|
| 5000 | 屏幕共享视频 |
| 5001 | 触摸/鼠标控制 |
| 5002 | 摄像头数据传输 |
| 5003 | 诊断结果回传 |
| 5004 | 命令控制 |
| 5005 | 语音数据发送 |
| 5006 | 语音合成接收 |
| 5007 | 语音命令控制 |
| 5008 | 鼠标控制 |

**默认 IP：** PC `172.20.10.3`，开发板 `172.20.10.8`

---

## ⚙️ 配置

两套配置并存：

| 文件 | 用途 |
|------|------|
| `configs/system_config.py` | Python 配置 + ConnectionManager（主界面导入） |
| `configs/system_config.json` | JSON 配置（启动脚本使用） |

> 注意：`visualization_test2.py` 第 43 行 `from system_config import ...` 直接导入（无 `configs.` 前缀），需 `configs/` 在 PYTHONPATH 中。导入失败时使用硬编码默认值。

---

## 💬 DeepSeek API

`visualization_test2.py` 中的 `DeepSeekAPI` 类提供：
- **治疗建议**：根据检测疾病 + 置信度调用 API 生成专业报告
- **AI 对话**：自由文本问答，自动附带当前检测结果上下文，**每轮对话调两次 API**——一次生成详细回复显示在 UI，一次生成 ≤100 字摘要用于语音播报
- **语音集成**：语音输入 → API 回复 → TTS 朗读（摘要版）
- Endpoint：`https://api.deepseek.com/v1/chat/completions`，模型：`deepseek-chat`
- API Key 存储：base64 编码 → `saved_api_key.txt`（已加入 .gitignore）

---

## 🎙️ 语音系统

### 语音输入（语音转文字）

`SmartVoiceManager` 识别策略：

1. **Vosk** 离线中文模型（延迟加载，搜索路径：`$VOSK_MODEL_PATH` → `./vosk-model-small-cn-0.22/` → `./vosk-model-cn-0.22/`）
2. **Google Speech Recognition**（在线回退）

识别完成后自动填入聊天框，稍后自动发送给 AI。

### 语音输出（文字转语音 TTS）

采用 **两级回退** 机制，确保在各种网络环境下都能播报：

| 优先级 | 方案 | 说明 |
|--------|------|------|
| 1 | **edge-tts** | 微软神经网络语音 `zh-CN-XiaoxiaoNeural`，发音自然流畅，需联网 |
| 2 | **pyttsx3** | Windows SAPI5 离线引擎，无需网络，作为兜底 |

**TTS 流程：**

1. AI 生成详细回复 → 显示在 UI 聊天框
2. AI 同时生成 **100 字以内简短摘要**（调 DeepSeek API 专门总结）
3. 摘要文本 → edge-tts 合成 → wmplayer.exe 以绝对路径播报（不弹网易云等第三方播放器）
4. edge-tts 网络不可用时自动回退 pyttsx3 本地引擎

**音频控制：**

- 勾选 `🔊 自动朗读 AI 回复` → 每轮对话结束后自动播报摘要
- 取消勾选 → 不朗读（不影响语音输入）
- 新对话生成时会打断上一轮未播完的音频

### 语音按钮

| 按钮 | 位置 | 功能 |
|------|------|------|
| `🎤 语音输入` | 医疗问答 Tab 底部 | 点击开始录音，自动识别中文并发送给 AI |
| `📱 唤醒板端语音` | 左侧扩展工具 | 通过 UDP 向开发板（172.20.10.8:5006）发送语音唤醒指令 |



---

## 🧪 训练

```bash
# 在 ultralytics-main/ 目录下（需修改 eyes_train.py 中的硬编码路径）
cd ultralytics-main && python eyes_train.py

# YOLO 命令行
yolo predict model=models/custom/AKConv_best_moudle/best.pt source=<image>
yolo val model=models/custom/AKConv_best_moudle/best.pt data=data/eyes_dataset.yaml
```

---

## 🔒 安全提醒

- 请勿将 API Key、密码等敏感信息提交到仓库
- `saved_api_key.txt` 已加入 `.gitignore`
- 如曾提交敏感信息，请轮换密钥并清理 Git 历史

---

## 📄 许可证

MIT License，详见 `LICENSE`。
