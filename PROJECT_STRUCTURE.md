# 项目结构

主入口是 `visualization_test2.py`。它处理Qt界面和任务调度，分类、DeepSeek及SQLite业务位于 `src/core`。

```text
Intelligent_diagnosis_system/
├─ 启动系统.cmd                本机双击启动（检查环境并记录日志）
├─ visualization_test2.py      当前PC GUI
├─ visualization_test1.py      历史GUI
├─ src/core/                  分类、对话、历史及本地参考说明
├─ src/pc/                    可选PC服务，复用业务模块
├─ src/board/                 树莓派扩展，未完成实机验收
├─ src/network/               投屏及通信扩展
├─ src/visualization/          当前桌面模块及历史界面版本
├─ configs/                   应用、网络、启动配置
├─ models/                    默认部署权重及备用资源
├─ data/                      train/val数据，不随提交包分发
├─ ultralytics-main/          本地YOLO11框架和AKConv注册
├─ ablation_fast_workers8.ipynb 四组实验工作台
├─ runs/                      实验、CSV、权重和报告图
├─ runtime/                   缓存与运行检查，Git忽略
├─ scripts/                   启动、运行检查、实验记录和绘图
├─ tests/                     隔离业务和offscreen GUI测试
├─ docs/                      运行说明与历史文档
└─ requirements*.txt          核心、PC、语音、硬件、测试依赖
```

YOLO26框架来自同级 `../yolo26/ultralytics-main`，供实验或显式配置使用；PC默认框架来源在app_config.json中指定。

应用与网络设置在顶部“系统设置”编辑；运行网络需重启，界面偏好即时生效。最新运行流程见 `docs/guide/PC运行与测试说明.md`；历史文档的完成状态不能替代实际验收。

## 当前界面模块
- `workbench_layout.py` / `workbench_theme.py`：页面布局、统一主题、独立阅读字号。
- `workbench_jobs.py` / `background_jobs.py`：后台分类、任务与退出协调。
- `analysis_pages.py` / `result_export.py`：批量与历史筛选、统计、统一导出。
- `conversation.py` / `rich_text.py`：会话归属、消息操作、分层富文本。
- `speech_player.py` / `local_camera.py`：应用内语音播放、本机相机。
- `preferences.py` / `image_directory.py`：窗口和控件偏好、选图目录。
- `scripts/launch_desktop.py`：启动环境检查与独立日志。

本机配置 `configs/app_config.json`、`ui_preferences.json`、备份、数据集、密钥、运行截图及临时导出不随Git提交。应用配置模板为 `configs/app_config.example.json`。

## 整理原则与保留区域
本次仅补充说明和Git忽略规则，不删除或搬移文件。

| 区域 | 用途 | 当前处理 |
|---|---|---|
| visualization_test2.py、src/core、当前界面模块 | 运行主线 | 继续维护 |
| 启动系统.cmd、scripts/launch_desktop.py | 日常启动 | 推荐入口 |
| ablation_fast_workers8.ipynb、实验脚本 | 训练和实验分析 | 保留；提交版本清除输出，本地输出保留 |
| visualization_test1.py、src/visualization/visualization*.py | 旧界面参考 | 保留，非默认入口 |
| backups/ | 历次改动备份 | 本机保留，不提交 |
| runtime/、tests/text/、tests/voice/batch_results.csv | 验证截图、日志、手工导出 | 本机保留，不提交 |
| runs/ | 真实实验产物 | 保留，不自动清理 |
| docs/guide/ | 当前操作说明与验收记录 | 以最新日期文档为准 |
| 其他历史报告 | 历史材料 | 保留，不作为当前实验真实性依据 |

旧 `ablation.ipynb` 的本地缺失是此前工作区状态，本次同步不提交该删除；GitHub保留旧文件。
