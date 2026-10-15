# 眼科图像分类辅助诊断系统

《智能软件系统》课程设计项目。PC端结合八类图像分类、DeepSeek辅助问答和SQLite历史记录，实验Notebook记录两套框架环境的基线与AKConv共四组训练。检查点核对发现成对结构与权重相同，不能写成四种不同架构比较。语音、树莓派采集和投屏是可选扩展，示例配置默认关闭；本机开关以系统设置为准。

课程材料的事实基线、验证结果与已知限制见 [工程与证据核对结果](docs/guide/工程与证据核对结果.md)。四份课程设计报告（文献分析、技术方案设计、模型分析、实验分析）位于 `deliverables/course_design/`，均已按任务书要求完成正文扩充、设计图绘制与参考文献著录。

## 运行

本机可直接双击 **启动系统.cmd**。启动器优先选择项目 `.venv` 或 `venv`，也支持通过 `EYE_PYTHON` 明确指定解释器；保留原开发机环境作为兼容路径。启动前检查 GUI 依赖，日志写入 `runtime/startup_logs/`。首次没有本地网络配置时读取 `configs/system_config.example.json`。详细说明见 [桌面启动与验收记录](docs/guide/桌面启动与验收记录.md)。

首次无本地 `configs/app_config.json` 时读取可移植的 `configs/app_config.example.json`；在系统设置保存后生成本地配置。

本机GUI和系统测试环境为 `D:/Anaconda3/envs/yolov10/python.exe`，实验训练环境为 `D:/Anaconda3/envs/yolo26/python.exe`。

```powershell
Set-Location 'C:/Users/47449/Desktop/yolo/Intelligent_diagnosis_system'
& 'D:/Anaconda3/envs/yolov10/python.exe' visualization_test2.py
```

新PC环境在项目根目录执行 `python -m pip install -r requirements.txt`，然后运行 `python visualization_test2.py`。安装文件分为核心、PC、可选语音、可选硬件和测试依赖；当前版本基于已验证的PyTorch 2.5.1/TorchVision 0.20.1组合。

详细步骤见 [PC运行与测试说明](docs/guide/PC运行与测试说明.md)。分类与历史查询不需要任何在线账号；在线问答需要个人 DeepSeek API 密钥。服务端口令校验为可选能力，由环境变量 `EYE_ACCESS_CODE` 控制：未设置时完全放行（当前线上部署即为此状态），设置后前端通过 `/api/access` 判断是否需要口令，并经 `/api/login` 校验。

## 访问入口

| 入口 | 地址 | 说明 |
| --- | --- | --- |
| 线上服务（IP 直连） | http://101.132.104.41:8000 | 已部署，实测 `/api/health` 正常返回 |
| 线上服务（域名） | http://game.lingluotoki.dpdns.org:8000 | DDNS 域名，指向同一服务 |
| 本机桌面端 | 双击 `启动系统.cmd` | 或执行 `python visualization_test2.py` |
| 本机服务端 | http://127.0.0.1:8000 | 执行 `python -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000` |

健康检查接口 `/api/health` 返回模型加载状态、权重路径、输入尺寸与八类映射，可用于确认服务是否就绪。分类与历史查询免登录；使用在线问答前需在「系统设置」中填入个人 DeepSeek API 密钥（密钥仅保存在服务端内存，重启后需重新填写）。

## 设置和操作

顶部“系统设置”集中管理界面显示、云端AI、语音外设和运行网络。界面字体与阅读字号独立；阅读字号、自动朗读和正常退出时的窗口布局会保存。加载图像支持记住上次目录或固定目录。运行与网络配置保存后重启生效。

默认自动加载 `models/custom/AKConv_best_moudle/best.pt`。该默认部署权重为原生YOLO11分类结构，不含AKConv；不能依据文件夹名称推断结构。状态栏及运行检查输出实际框架、设备、类别与AKConv层数。

加载图像后执行分类，再查看结果、历史或辅助说明。分类失败会清空当前结果，不生成固定的AMD/0.98，不保存失败为成功历史，也不生成当前报告。API故障显示明确失败；本地静态参考与在线回答分开标注。

## 业务架构

```text
visualization_test2.py             PyQt5交互、展示和后台请求调度
src/core/classification.py         模型加载、类别与概率验证、单图/批量分类
src/core/llm.py                    DeepSeek、超时与失败状态、受限会话记忆
src/core/history.py                SQLite存储、查询、建议缓存和显式演示记录
configs/app_config.json            权重、框架、设备和可选功能
configs/system_config.json         网络与启动服务配置
configs/network_config.py          局域网识别与保存
src/pc/pc_diagnosis_server.py       调用相同业务服务的可选UDP分类服务
```

`visualization_test1.py`及`src/visualization/visualization*.py`是历史版本；`src/visualization/`下的workbench、conversation等模块是当前界面的组成部分。PC独立服务不再为了访问分类或DeepSeek而导入整套GUI。

## 类别和数据

A=AMD，C=白内障，D=糖尿病视网膜病变，G=青光眼，H=高血压性视网膜病变，M=近视性黄斑病变，N=正常，O=其他。

分类训练使用 `data/train` 和 `data/val`，分别有5110和1163张图像。`data`参数传目录；`eyes_dataset.yaml`只是说明文件。数据来源、授权、患者级划分和标签转换仍需补充，提交源码不包含完整数据集。

## 实验和图表

打开 `ablation_fast_workers8.ipynb`，先执行配置与检查，再选择所需训练单元。正式默认为50轮、224输入、batch32、8进程、seed0、FP32、随机初始化及AdamW。训练产生数据指纹、版本、模型结构和完成状态记录；不要用小样本测试替代正式实验。

`runs/ablation`保留历史AMP对比；`runs/comparison_notebook`存正式Notebook运行；`runs/report_figures`存报告图表。历史AMP和新FP32结果不能直接当作同条件对比。跨框架实现差异也不能全部归因于骨干结构，因为本地两版分类YAML配置相同。

AKConv是引用Lv等人 *AKConv Convolutional Kernel with Arbitrary Sampled Shapes and Arbitrary Number of Parameters* 的集成探索，不能宣称自行提出。保留Ultralytics及其他开源组件的许可证与引用；框架许可证见 `ultralytics-main/LICENSE`。

## 检查和测试

```powershell
python scripts/check_pc_runtime.py
python scripts/check_pc_runtime.py --image '实际图像.jpg' --device cpu
python -m pytest tests/test_application_services.py tests/test_gui_classification.py -q
```

检查脚本不调用DeepSeek、不写用户历史。测试使用隔离数据库和API测试客户端，覆盖分类失败、批量错误、API故障、记忆、历史及GUI和网络设置。真实在线API、麦克风、树莓派和多设备通信需要单独实测。

运行产物在 `runtime/`，用户历史默认位于 `~/EyeDiseaseDetectorHistory/history.db`。密钥文件不进入Git或提交包。四份课程报告、正式四组实验和教师要求的部署验收需要按真实完成情况交付，不因代码存在而自动视为完成。

## 当前桌面功能与验收边界（2026-10-13）
- 单图、批量、历史独立分页；八类概率、分类分布、趋势及按筛选范围导出。
- 本机摄像头与开发板摄像头分离；本机支持设备枚举、预览、拍照载入。
- 聊天支持 Enter 发送、Shift+Enter 换行，消息复制、朗读/停止、文本导出。
- Edge TTS 晓晓音色生成 MP3，应用内播放；生成失败明确提示，不自动改换离线音色。
- 本轮52项自动测试通过；隔离数据库的真实GPU加载及单图分类通过。硬件拍摄、麦克风、实际出声、开发板及干净环境安装尚未全部实测。
- 课程交付仍需补齐四份真实报告、数据来源与患者级划分说明、实验可追溯材料以及部署验收。仓库中的历史文档不等于当前验收证明。
