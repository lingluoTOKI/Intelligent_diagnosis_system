# 智能诊断系统项目目录结构

## 项目概述
基于YOLO的智能诊断系统，包含开发板端和PC端功能模块。

## 目录结构

```
Intelligent_diagnosis_system/
├── configs/                    # 配置文件目录
│   ├── system_config.json      # 系统配置JSON文件
│   └── system_config.py        # 系统配置Python模块
├── data/                       # 数据集目录（预留）
├── docs/                       # 项目文档
│   ├── guide/                  # 使用指南
│   ├── notes/                  # 开发笔记
│   ├── reports/                # 项目报告
│   └── setup/                  # 环境配置说明
├── markdown/                   # Markdown文档
├── models/                     # 模型文件
│   └── custom/                 # 自定义模型
│       └── self_model/         # 自训练模型
├── scripts/                    # 启动脚本
├── src/                        # 源代码目录
│   ├── board/                  # 开发板端代码
│   ├── network/                # 网络通信模块
│   ├── pc/                     # PC端代码
│   ├── utils/                  # 工具函数
│   └── visualization/          # 可视化模块
├── tests/                      # 测试目录
│   ├── network/                # 网络测试
│   ├── notebooks/              # Jupyter笔记本
│   ├── reports/                # 测试报告
│   ├── scripts/                # 测试脚本
│   └── voice/                  # 语音测试
├── ultralytics-main/           # YOLO源码（第三方库）
├── .gitignore                  # Git忽略配置
├── LICENSE                     # 许可证
├── README.md                   # 项目说明
└── PROJECT_STRUCTURE.md        # 本文件
```

## 目录说明

| 目录 | 说明 | 状态 |
|------|------|------|
| `configs/` | 系统配置文件，包含JSON和Python配置 | ✅ |
| `data/` | 数据集存储目录（预留） | ⏳ |
| `docs/` | 项目文档，包含指南、笔记、报告 | ✅ |
| `models/` | 训练好的模型文件 | ✅ |
| `scripts/` | 快速启动脚本 | ✅ |
| `src/board/` | 开发板端核心代码 | ✅ |
| `src/network/` | WiFi网络通信模块 | ✅ |
| `src/pc/` | PC端服务代码 | ✅ |
| `src/utils/` | 通用工具函数 | ✅ |
| `src/visualization/` | 可视化模块（遗留代码） | ✅ |
| `tests/` | 测试代码和报告 | ✅ |

## 主要文件说明

### src/board/ - 开发板端
- `board_camera_integration.py` - 摄像头集成
- `board_integrated_system.py` - 集成系统
- `board_local_model.py` - 本地模型推理
- `board_voice_interaction.py` - 语音交互

### src/pc/ - PC端
- `pc_diagnosis_server.py` - 诊断服务
- `pc_voice_server.py` - 语音服务

### scripts/ - 启动脚本
- `quick_start.py` - 快速启动
- `start_system.py` - 启动系统
- `system_launcher.py` - 系统启动器

## 使用建议

1. **开发板端运行**: 使用 `scripts/quick_start.py` 或 `src/board/start_board.sh`
2. **PC端运行**: 使用 `pc_diagnosis_server.py` 和 `pc_voice_server.py`
3. **模型管理**: 新训练的模型放入 `models/custom/` 目录
4. **配置修改**: 修改 `configs/system_config.json` 或 `configs/system_config.py`