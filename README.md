# AI 眼科辅助诊断系统

<div align="center">

**基于 YOLO11 + AKConv 的眼部疾病智能诊断系统**

集成 DeepSeek AI 诊疗对话 · 离线语音识别 · PC + 开发板协同 · WiFi 屏幕共享

[快速开始](#-快速开始) · [系统架构](#-系统架构) · [主界面](#-主界面) · [模型说明](#-模型说明) · [通信协议](#-通信协议)

</div>

---

## 项目简介

本项目聚焦于「**眼部图像智能辅助诊断**」，结合 YOLO 视觉推理、中文离线语音识别、DeepSeek API 诊疗对话、局域网通信与开发板联动，构建了一个可演示、可部署的综合系统。

**适用场景：**
- 课程设计 / 毕业设计
- AI 医疗方向原型系统
- PC + 开发板协同教学项目
- 视觉识别 + 语音交互综合实验平台

> ⚠️ 本系统仅用于辅助诊断和教学展示，不能替代专业医生的临床诊断。

---

## 快速开始

### 环境要求

- Python 3.9+
- Windows / Linux / macOS
- 8GB+ 内存推荐
- CUDA 可选（GPU 推理加速）

### 安装与运行

```bash
# 1. 克隆仓库
git clone <repo-url>
cd Intelligent_diagnosis_system

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载 Vosk 离线语音模型（可选，用于离线语音识别）
#    将模型解压到项目根目录，命名为 vosk-model-cn-0.22/

# 4. 启动主界面（主要入口）
python visualization_test2.py

# 5. 一键启动所有 PC 端服务
python scripts/start_system.py

# 6. 交互式启动助手
python scripts/quick_start.py
```

### 首次使用

1. 启动后进入 `⚙️ 系统设置` → 输入 DeepSeek API Key → 点击保存
2. 点击 `🔄 加载模型` → 等待模型加载完成
3. 点击 `🖼️ 加载图像` → 选择一张眼底图像
4. 点击 `🔍 开始检测` → 等待推理完成 → 自动弹出检测报告
5. 切换到 `🩺 诊疗建议` 查看 AI 生成的专业治疗建议

---

## 系统架构

```
visualization_test2.py          ← PC 端主入口（PyQt5 单体 GUI，~7000 行）
    │
    ├── EyeDiseaseDetector      ← YOLO11 + 自定义 AKConv 推理引擎
    ├── ResultProcessor         ← 检测结果解析、可视化、备用结果
    ├── SmartVoiceManager       ← Vosk 离线优先 → Google 在线回退
    ├── DeepSeekAPI             ← AI 诊疗建议 + 对话式问诊 + 默认医学知识库
    ├── MedicalAIService        ← 自由文本医疗咨询（含网络检测）
    ├── HistoryDB               ← SQLite 诊断历史存储（含 AI 建议缓存）
    ├── BoardCameraReceiver     ← UDP 接收开发板摄像头数据
    ├── CommandListener         ← UDP 命令控制监听
    └── configs/system_config   ← 端口 / 网络 / 摄像头配置（可选，失败用默认值）

src/pc/                          ← PC 端独立服务
    ├── pc_diagnosis_server.py   ← 接收开发板图像 → YOLO 推理 → 返回诊断结果
    └── pc_voice_server.py       ← 语音识别 / 合成服务（外部独立进程）

src/network/                     ← WiFi 屏幕共享 + 远程控制（全部 UDP）
    ├── wifi_pc_sender_with_mouse.py  ← PC 屏幕捕获 + 接收鼠标控制（主用版本）
    ├── wifi_pc_sender_2.0.py         ← 简化版发送端（无鼠标控制）
    ├── wifi_pc_receiver_2.0.py       ← 开发板屏幕接收 + 触摸 → 鼠标转发
    └── wifi1.0/                      ← 旧版归档（不再使用）

src/board/                       ← 开发板端（树莓派）
    ├── board_camera_integration.py   ← 摄像头采集 + UDP 分包发送至 PC
    ├── board_integrated_system.py    ← 开发板集成主入口
    ├── board_local_model.py          ← 本地 ONNX 模型推理
    └── board_voice_interaction.py    ← 开发板端语音交互

src/utils/                       ← 一次性测试 / 调试脚本（非通用工具库）
```

### 通信方式

所有实时数据（屏幕、摄像头、命令、诊断结果）均使用 **UDP** 通信（非 TCP）。

**数据分包格式：**
- 每包最大 1400 字节
- 包头 8 字节：4 字节 `packet_id` + 4 字节 `total_packets`
- 开发板 → PC 端通过多端口并行传输

**默认 IP：**

| 设备 | IP |
|------|-----|
| PC 端 | `172.20.10.3` |
| 开发板 | `172.20.10.8` |

**端口分配：**

| 端口 | 用途 |
|------|------|
| 5000 | 屏幕共享视频流 |
| 5001 | 触摸 / 鼠标控制转发 |
| 5002 | 摄像头数据传输 |
| 5003 | 诊断结果回传 |
| 5004 | 命令控制 |
| 5005 | 语音数据发送 |
| 5006 | 语音合成接收 |
| 5007 | 语音命令控制 |
| 5008 | 鼠标控制 |

---

## 主界面

程序启动后界面通过 **QSplitter** 分为左右两个区域，支持拖拽调整比例（默认左 6 : 右 4）。

### 左侧 —— 视觉操作区

#### 🖼️ 本地图像分析（Tab）

- 左右并排显示**原始图像**和**检测结果图像**
- 空状态时显示占位提示（📸 等待加载 / 🧠 等待诊断）
- 图像自动缩放适配窗口，保持宽高比

**工作流按钮（四步操作）：**

| 步骤 | 按钮 | 说明 |
|------|------|------|
| 1 | 🔁 加载模型 | 加载 YOLO11 + AKConv 模型权重 |
| 2 | 🖼️ 加载图像 | 选择本地眼底图像文件 |
| 3 | 🔍 开始检测 | YOLO 推理 → 自动展示结果 + 保存历史 |
| 4 | 📊 查看报告 | 重新打开最近一次检测的报告弹窗 |

**扩展工具按钮：**

| 按钮 | 功能 |
|------|------|
| 📁 批量处理 | 选中多张图像 → 一键批量推理 → 统计面板 + 图表 |
| 📜 历史记录 | SQLite 历史管理：查看详情、多选删除、清空、趋势分析 |
| 📱 开发板交互 | 向开发板发送指令 / 接收状态 |
| 📱 唤醒板端语音 | 通过 UDP 唤醒开发板端语音交互功能 |

#### 📱 硬件视窗（Tab）

- 实时显示开发板摄像头画面
- 状态指示：🔴 未连接 / 🟢 已连接，分辨率、帧率
- 控制按钮：🔗 连接开发板 / 📸 截取并诊断

### 右侧 —— AI 分析区

#### 🩺 诊疗建议（Tab）

- 检测完成后显示 DeepSeek AI 生成的专业治疗报告
- **Markdown 渲染引擎**：支持标题（h1-h5）、粗体、有序/无序列表、引用块、表格、水平线
- **默认医学知识库**：内置 8 种眼部疾病的完整治疗建议（疾病简介 + 治疗方案 + 日常护理 + 随访建议），API 不可用时自动降级
- **全屏阅览**：F11 全屏 / Esc 退出，大屏展示报告
- 按钮：🔄 生成当前报告 / 📋 复制 / 🔍 全屏

#### 💬 医疗问答（Tab）

- 多轮对话式 AI 问诊，自动附带当前检测结果作为上下文
- 输入框 + 🎤 语音输入 + 发送 / 清除按钮
- 对话历史保留（最多 50 轮），AI 根据上下文提供连贯建议
- **每轮对话调两次 API**：一次生成详细回复显示在 UI，一次生成 ≤100 字摘要用于 TTS 播报
- 聊天记录自动保存到 SQLite（disease_name 标记为 `[对话]...`）

#### ⚙️ 系统设置（Tab）

- **DeepSeek API Key** 配置：输入框 + 👁 显示/隐藏 + 💾 保存 + 🌐 网络连接测试
- **语音识别时长**：滑块调节（5-30 秒），默认 10 秒
- **云端 AI 开关**：☑️ 使用 DeepSeek API 生成治疗建议（关闭时使用本地默认建议）
- **自动朗读开关**：☑️ 自动朗读 AI 回复（控制 TTS 播报）
- API Key 经 base64 编码存储于 `saved_api_key.txt`（已加入 `.gitignore`）

---

## 模型说明

### 模型架构

基于 **YOLO11-cls** 分类模型，集成自定义 **AKConv**（可变形卷积）模块以提升特征提取能力。

**模型文件：**

| 文件 | 格式 | 用途 |
|------|------|------|
| `models/custom/AKConv_best_moudle/best.pt` | PyTorch | PC 端主要推理权重 |
| `models/custom/AKConv_best_moudle/best.onnx` | ONNX | 开发板端推理 |
| `models/custom/common/best.pt` | PyTorch | 备用权重 |

**AKConv 注册机制：**

```
ultralytics-main/ultralytics/nn/modules/akconv.py   ← 自定义模块定义
    ↓ from .akconv import * (__init__.py L19)
    ↓ parse_model() 添加 AKConv / C3k2_AKConv (tasks.py)
    ↓
模型 YAML 可直接引用 C3k2_AKConv 模块
```

> 注意：`ultralytics-main/` 是整个 YOLO 框架的本地副本（含 AKConv 修改），不是 pip 安装的 `ultralytics` 包。运行前需确保项目根目录在 `sys.path` 中。

### 疾病类别（8 分类）

| 代码 | 英文名称 | 中文名称 |
|------|----------|----------|
| A | AMD | 年龄相关性黄斑变性 |
| C | Cataract | 白内障 |
| D | Diabetic Retinopathy | 糖尿病视网膜病变 |
| G | Glaucoma | 青光眼 |
| H | Hypertensive Retinopathy | 高血压性视网膜病变 |
| M | Myopia | 近视性黄斑病变 |
| N | Normal | 正常眼底 |
| O | Other | 其他眼部疾病 |

### 类别映射

`EyeDiseaseDetector.class_names` 字典将 YOLO 输出的类别索引（0-7）映射到英文疾病名称。`ResultProcessor.letter_to_disease` 字典将字母代码（A-H, M, N, O）映射到英文名称，用于解析文本输出。

```python
class_names = {
    0: 'AMD', 1: 'Cataract', 2: 'Diabetic Retinopathy',
    3: 'Glaucoma', 4: 'Hypertensive Retinopathy',
    5: 'Myopia', 6: 'Normal', 7: 'Other'
}
```

### 训练

```bash
# 在 ultralytics-main/ 目录下（需修改 eyes_train.py 中的硬编码路径）
cd ultralytics-main && python eyes_train.py

# YOLO 命令行（使用本地 ultralytics-main，非 pip 包）
yolo predict model=models/custom/AKConv_best_moudle/best.pt source=<image>
yolo val model=models/custom/AKConv_best_moudle/best.pt data=data/eyes_dataset.yaml
```

---

## 数据存储

### 诊断历史（SQLite）

检测结果和对话记录自动保存到 SQLite 数据库：

```
~/EyeDiseaseDetectorHistory/history.db
```

**records 表结构：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER | 自增主键 |
| `record_id` | TEXT UNIQUE | UUID 唯一标识 |
| `timestamp` | TEXT | 检测时间（`YYYY-MM-DD HH:MM:SS`） |
| `image_path` | TEXT | 原始图像路径 |
| `disease_name` | TEXT | 疾病名称（或 `[对话]...` 标记的对话记录，或 `[演示]...` 标记的演示数据） |
| `confidence` | REAL | 置信度（0~1，对话记录为 0.0） |
| `advice` | TEXT | AI 治疗建议缓存（首次生成后持久化，下次秒开免 API 调用） |

**触发保存的场景：**

1. 图像检测完成 → 自动保存（疾病名 + 置信度 + 图像路径）
2. 医疗问答每轮对话 → 自动保存（`[对话]问题前80字`，confidence=0.0）
3. 旧版 JSON 文件（`history.json`）→ 首次启动自动迁移到 SQLite，原文件重命名为 `.bak`

**`advice` 字段缓存机制：**

```
打开历史详情
    ↓
立即展示本地默认医学建议（0ms，8 种疾病各有完整方案）
    ↓
后台检查 advice 字段
    ├── 有缓存 → 替换为 AI 详细建议（秒开，0 API 消耗）
    └── 无缓存 → 调 DeepSeek API 生成 → 写入 advice 字段 → 替换显示
```

**API 封装：** `HistoryDB` 类（`visualization_test2.py:63`）提供：

| 方法 | 说明 |
|------|------|
| `add(record_id, timestamp, image_path, disease_name, confidence)` | 新增记录 |
| `get_all(limit=10000)` | 获取所有记录（时间倒序，默认上限 10000） |
| `delete_by_record_id(record_id)` | 按 UUID 删除单条 |
| `delete_all()` | 清空全部 |
| `update_advice(record_id, advice)` | 写入 AI 建议缓存 |
| `count()` | 记录总数 |
| `migrate_from_json(json_path)` | 从旧版 JSON 迁移 |

全局单例通过 `get_history_db()` 获取。

### 音频缓存

TTS 合成的中间 MP3 文件为临时文件，播放后系统自动清理。

---

## Markdown 渲染引擎

诊疗建议和医疗问答中的 AI 回复使用统一的 `format_advice_html()` 方法进行 Markdown → HTML 转换，适配 Qt QTextEdit 的富文本渲染。

### 支持的语法

| Markdown | HTML 输出 | 样式 |
|----------|-----------|------|
| `# 标题` | `<h2>` | 蓝色左边框 + 浅蓝背景 |
| `## 标题` | `<h3>` | 紫色底部边框 + 浅紫背景 |
| `### 标题` | `<h4>` | 蓝色左边框 |
| `#### 标题` | `<h5>` | 紫色文字 |
| `**粗体**` | `<b>` | 蓝色高亮 |
| `*text**` / `**text*` | `<b>` | 兼容 DeepSeek 偶发的星号缺失 |
| `1. 列表项` | `<ol>` + `<li>` | 自动编号 |
| `- / * / • 列表项` | 子弹列表 | 蓝色子弹符 |
| `> 引用` | `<blockquote>` | 左侧蓝色边框 + 斜体 |
| `---` | `<hr>` | 灰色分隔线 |
| `\| 列1 \| 列2 \|` | 表格行 | 底部边框分隔 |

### 设计要点

- 所有 HTML 元素均带显式 `color` 属性，确保深色背景下文字可见
- 标题和列表块级元素处理前自动关闭已打开的 HTML 标签，防止结构嵌套错乱
- `font-size` 使用 `pt` 单位，跟随系统 DPI 自动缩放
- 全局 QSS `* { font-family: 'Microsoft YaHei' }` 确保字体统一

---

## 语音系统

### 语音输入（STT）

`SmartVoiceManager` 采用两级回退策略：

| 优先级 | 引擎 | 说明 |
|--------|------|------|
| 1 | **Vosk** 离线 | 中文模型，延迟加载，搜索路径：`$VOSK_MODEL_PATH` → `./vosk-model-small-cn-0.22/` → `./vosk-model-cn-0.22/` |
| 2 | **Google Speech** 在线 | 网络可用时的回退方案 |

识别完成后自动填入聊天输入框。

### 语音输出（TTS）

| 优先级 | 引擎 | 说明 |
|--------|------|------|
| 1 | **edge-tts** | 微软神经网络语音 `zh-CN-XiaoxiaoNeural`，需联网 |
| 2 | **pyttsx3** | Windows SAPI5 离线引擎，无需网络 |

**TTS 流程：**

1. AI 生成详细回复 → 显示在 UI 聊天框
2. AI 同时生成 ≤100 字简短摘要（调 DeepSeek API 专门总结）
3. 摘要 → edge-tts 合成 MP3 → `wmplayer.exe` 以绝对路径播报
4. edge-tts 不可用时自动回退 pyttsx3

### 语音按钮

| 按钮 | 位置 | 功能 |
|------|------|------|
| 🎤 语音输入 | 医疗问答 Tab 底部 | 点击开始录音 → 自动识别 → 发送给 AI |
| 📱 唤醒板端语音 | 左侧扩展工具 | UDP 向开发板（172.20.10.8:5006）发送唤醒指令 |
| 🔊 自动朗读 | 系统设置 | 勾选后每轮对话自动播报摘要 |

---

## 趋势分析与演示数据

### 趋势分析（📈 病情趋势分析）

基于历史记录生成三个可视化标签页：

| Tab | 图表类型 | 内容 |
|-----|----------|------|
| 每日趋势 | 折线图 + 面积填充 | 每日检测总量随时间变化 |
| 疾病分布 | 饼图 + 柱状图 | 各疾病占比和绝对数量 |
| 疾病趋势 | 多线折线图 | 每种疾病每日检测量变化趋势 |

- 图表使用 Matplotlib 渲染，暗色主题（#2d3748 背景 + 彩色数据线）
- 支持 F11 全屏 / Esc 退出
- 自动过滤 `[演示]` 前缀和 `[对话]` 记录

### 演示数据生成器

当数据库记录不足时，可通过趋势分析弹窗底部的 `📊 生成30天演示数据` 按钮批量生成测试数据：

| 参数 | 值 |
|------|------|
| 时间跨度 | 30 天 |
| 总数据量 | ~5000 条 |
| 疾病分布 | Diabetic Retinopathy(50%)、Normal(20%)、Myopia(15%)、AMD(5%)、Cataract(4%)、Glaucoma(3%)、HTN(3%)、Other(2%) |
| 工作日 | 150-250 条/天 |
| 周末 | 50-100 条/天 |
| 标记方式 | disease_name 前缀 `[演示]`，可批量删除 |

- 再次点击会先清除旧演示数据再重新生成
- 演示数据在历史表格和趋势分析中自动去掉 `[演示]` 前缀显示
- 直接 SQL `DELETE WHERE disease_name LIKE '[演示]%'` 清理，不受 `get_all()` 的 LIMIT 限制

---

## 历史记录性能优化

针对大数据量（5000+ 条记录）场景的专项优化：

| 优化项 | 方案 | 效果 |
|--------|------|------|
| 操作列按钮 | `QTableWidgetItem` + `cellClicked` 信号替代逐行 `QPushButton` | 避免创建 5000 个 widget |
| 界面重绘 | `setUpdatesEnabled(False)` 批量写入后一次性渲染 | 减少重绘开销 |
| 进度反馈 | ≤500 条直接秒开；>500 条弹 `QProgressDialog` 进度条 | 小数据无感，大数据有反馈 |
| 进度刷新 | 每 200 条更新一次进度条 + `processEvents()` | 平衡速度与视觉反馈 |
| 取消加载 | 红色 `⏹ 取消加载` 按钮 | 中途可停止，保留已加载行 |
| 数据上限 | `get_all(limit=10000)` | 5000+ 条演示数据不被截断 |
| 删除操作 | `record_id` 通过 `Qt.UserRole` 绑定在单元格上 | 删除时直接取 ID，不依赖索引映射 |
| 删除刷新 | `_populate_history_table()` 原地刷新表格 | 不再弹第二个窗 |

---

## UI 适配说明

### 高分屏（HiDPI）优化

- 程序入口启用 `AA_EnableHighDpiScaling` + `AA_UseHighDpiPixmaps`
- 全局基础字体：`QFont("Microsoft YaHei", 11)`
- 所有 QSS `font-size` 使用 **`pt`** 单位（非 `px`），跟随系统 DPI 自动缩放
- 全局 QSS：`* { font-family: 'Microsoft YaHei', 'SimHei', sans-serif; }` 确保字体统一

### 控件自适应

- 移除所有 `setFixedSize` / `setFixedWidth` / `setFixedHeight` 硬约束
- 移除 QSS 中的 `min-width` / `width` / `height` 约束，改用 `padding` 驱动
- 图像卡片移除 `setMaximumSize(500,500)` 限制
- 表格行高使用 `QHeaderView.ResizeToContents` 自动适配
- Tab 栏通过 `tabBar().setFont()` 设置字体确保高度正确计算
- Tab 文字永不截断（`setElideMode(Qt.ElideNone)` + `setUsesScrollButtons(True)`）

### 多弹窗全屏支持

| 弹窗 | 全屏按钮 | F11 | Esc 行为 |
|------|----------|-----|----------|
| 检测报告 (`show_disease_result`) | 🔍 全屏 | 切换 | 全屏→退出→关闭 |
| 历史详情 (`view_history_record`) | 🔍 全屏 | 切换 | 全屏→退出→关闭 |
| 趋势分析 (`show_trend_analysis`) | 🔍 全屏显示 | 切换 | Esc 退出全屏 |

---

## 配置系统

两套配置并存，端口号分散在各独立脚本中硬编码，修改需同步多处：

| 文件 | 用途 | 导入方式 |
|------|------|----------|
| `configs/system_config.py` | Python 配置 + `ConnectionManager` 类 | `visualization_test2.py` 直接 `from system_config import ...` |
| `configs/system_config.json` | JSON 配置（仅 6 个端口） | `scripts/start_system.py` 读取 |

> 注意：`from system_config import ...` 无 `configs.` 前缀，需 `configs/` 在 `PYTHONPATH` 中。导入失败时程序使用硬编码默认值继续运行。

---

## DeepSeek API 集成

`visualization_test2.py` → `DeepSeekAPI` 类（第 528 行）提供：

| 功能 | 方法 | 说明 |
|------|------|------|
| 治疗建议 | `get_treatment_advice(disease, confidence)` | 根据检测结果生成专业报告 |
| AI 对话 | `chat(user_message)` | 自由文本医疗问答 |
| 默认知识库 | `_get_default_advice(disease_name)` | 内置 8 种疾病的完整治疗建议 |
| 网络诊断 | `_get_enhanced_fallback_advice()` | API 不可用时的故障排查指南 |

**API 参数：**
- Endpoint：`https://api.deepseek.com/v1/chat/completions`
- 模型：`deepseek-chat`
- API Key：base64 编码存储于 `saved_api_key.txt`（已在 `.gitignore` 中）

---

## 关键注意事项

- 主入口是 `visualization_test2.py`，**不是** `visualization_test1.py` 或 `src/visualization/` 下的历史版本（1.0-5.0，仅供参考）
- `os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'` 必须在所有导入之前设置（macOS OpenMP 兼容性）
- matplotlib 后端强制设为 `Qt5Agg`，修改 GUI 相关代码时不要更改此后端
- `pyautogui.FAILSAFE = False` 被显式禁用（屏幕共享鼠标控制需要）
- 各脚本中的 IP 地址硬编码分散在多处，修改 IP 需要全局搜索替换
- API 密钥文件 `saved_api_key.txt` 已在 `.gitignore` 中，切勿提交
- 根目录下的 `temp_image_*.png` 是运行时产物，可安全删除
- `services/` 目录仅包含过期的 `.pyc` 字节码文件，可忽略

---

## 安全提醒

- 请勿将 API Key、密码等敏感信息提交到仓库
- `saved_api_key.txt` 已加入 `.gitignore`
- 如曾提交敏感信息，请轮换密钥并清理 Git 历史

---

## 许可证

MIT License，详见 `LICENSE`。
