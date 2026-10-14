# PROJECT_HANDOFF.md — 眼科图像分类辅助诊断系统（最终阶段）

生成时间：2026-10-14 16:4x
本文件由本轮审计生成，**只记录已实际核查的事实**，未核实项明确标注。

## 1. 项目基本信息

- 项目：基于 YOLO 与大语言模型的眼科图像分类辅助诊断系统
- 成员：欧阳宸（230722105）／汪晓蕾（231613212）
- 根目录：`C:\Users\47449\Desktop\yolo\Intelligent_diagnosis_system`
- 课程：智能软件系统课程设计

## 2. Git 当前提交（已核实）

- 分支：`main`
- **HEAD：`1346f61`** —「fix: 修复课程交付包并核对模型实验与报告证据」
- 远程：`https://github.com/lingluoTOKI/Intelligent_diagnosis_system.git`
- **与 `origin/main` 完全同步（0 ahead / 0 behind）**，1346f61 已推送。
- ⚠️ 交接词给出的基线 `89fb11c` **已过时**（落后 4 个提交），不要再以它为基线。
- 工作区未提交（按约定属本地配置/实验，**不入库**）：
  - `ablation_fast_workers8.ipynb`
  - `configs/system_config.json`

## 3. 四份报告实际状态（已核实）

位置：`deliverables\course_design\`（主）与
`deliverables\final_submission\ZY0302308_...\课程设计报告\`（交付副本）。
原始备份已存在：`runtime\report_backups\20261014_151834\`、
`runtime\submission_backups\20261014_151432_749544\`、`..._153956_431653\`。

| 报告 | 正文字符 | 表格 | 图片 | 封面 | 页码 | 标题样式 |
|---|---|---|---|---|---|---|
| 文献分析报告 | 19,646 | 1 | 0 | ✅ | ✅ PAGE 域 | ✅ Heading1 ×4 |
| 技术方案设计报告 | 35,575 | 3 | 0 | ✅ | ✅ PAGE 域 | ✅ Heading1 ×5 |
| 模型分析报告 | 21,112 | 2 | 0 | ✅ | ✅ PAGE 域 | ✅ Heading1 ×5 |
| 实验分析报告 | 36,791 | 2 | 3 | ✅ | ✅ PAGE 域 | ✅ Heading1 ×5 |

**已修复缺陷（2026-10-14 16:57 完成）**
1. ~~四份报告均无页码~~ → 已在 `word/footer1.xml` 插入居中 `PAGE` 域（保留原有页脚静态文字，新增一个居中页码段落）。
2. ~~四份报告均未使用段落样式~~ → 已为 19 个一级标题插入 `<w:pStyle w:val="Heading1"/>`，同时把 `styles.xml`
   中 Heading1/2/3 的字体改为「宋体 / Times New Roman」、颜色改 `000000`（防止套用样式后变蓝色主题字体）。
   **段落原有直接格式（标题 sz=32 加粗、正文 sz=24 / w:line=360）全部保留未动**，视觉零变化。

- 修复备份：`runtime/report_backups/20261014T165645/`
- 脚本：`C:\tmp\fix_reports.py`（修复）、`C:\tmp\repack.py`（同步与重新打包）
- 复验结果：四份 XML 均合法、正文小四+1.5 倍行距段数不变、页脚 PAGE 域齐全。

**已确认无问题（重要红线项）**
- ✅ **AKConv 未虚构**：模型分析报告原文写明"当前实际部署权重经结构核对 AKConv 层数为 0，因而系统不能宣称在线部署使用了 AKConv"；文献报告强调"不能仅依据代码中存在模块或目录名称作结论"。符合要求。
- ✅ **参考文献真实**：每份 10 条（≥8 满足）。含 ODIR-2019、Ting/BJO 2019、Gulshan/JAMA 2016、Ting/JAMA 2017、Litjens/MedIA 2017、AKConv arXiv:2311.11587v2、Ultralytics 文档、Liu/Lancet DH、DeepSeek-V3 arXiv:2412.19437、Zhou/Nature 2023。且 [1] 已诚实标注"数据集说明页面，不计作核心研究论文"。
- ✅ 报告篇幅充足（2 万～3.7 万字符），非"过短"。

## 3.1 课程任务书符合性核对（任务书由用户 2026-10-14 提供）

**重要判断：本项目属「（一）深度学习与智能感知类」+ LLM 应用，不是多智能体选题。**
→ 任务书中"多智能体选题须补充"的条款（角色职责、交互模式、冲突处理、单 LLM vs 多智能体对比、
消融中的多智能体对比、失效场景根因等）**一律不适用，不得补写**。

| 任务书要求 | 实际 | 结论 |
|---|---|---|
| 代码不含数据集，附获取途径说明 | 包内无 `data/` `medical_images/`；含 `docs/guide/数据来源与划分审计.md` | ✅ |
| 依赖环境文件 requirements.txt | 含 7 个 requirements-*.txt | ✅ |
| 运行说明（启动步骤/访问入口/操作指引） | 含 `PC运行与测试说明.md`、`服务器部署说明.md`、`启动系统.cmd`、`提交说明.txt` | ✅ |
| A4 | 四份均 21.0×29.7 cm | ✅ |
| 宋体 | eastAsia 字体全部"宋体" | ✅ |
| 小四 | **正文（非表格）段落 sz=24 = 12pt = 小四** | ✅ |
| 1.5 倍行距 | **正文段落 w:line=360 = 1.5 倍** | ✅ |
| 标准页边距 | 四周均 2.54 cm | ✅ |
| 封面：课程名/选题/组员信息/提交日期 | 四份均有"智能软件系统课程设计"+报告名+题目+组员(学号)+提交日期 | ✅ |
| 文献 ≥8 篇核心文献并对比评述 | 每份 10 条，含对比评述 | ✅ |
| 命名 `ZY0302308_题目_姓名学号.zip` | `ZY0302308_基于YOLO与大语言模型的眼科图像分类辅助诊断系统_欧阳宸230722105_汪晓蕾231613212.zip` | ✅ |
| 标准版式隐含的页码 | 四份页脚均含居中 `PAGE` 域 | ✅ 已修复 |
| 统一标题层级 | 四份共 19 个 `Heading1`，导航窗格/自动目录可用 | ✅ 已修复 |

> 注：表格内为 1.25 倍行距 / 五号字，属表格通行做法，任务书约束对象为"正文"，视为可接受。
> 早期审计曾误判"正文字号偏小"，经段落级复核（按是否在 `<w:tbl>` 内区分）已修正为**达标**。

## 4. 已验证的模型与实验信息

- 线上权重：`/opt/eye-diagnosis/models/server/best512.onnx`
- 线上 `/api/health` 实测（2026-10-14 16:4x）：`status=ok`、`image_size=512`、ONNX Runtime 1.23.2、`CPUExecutionProvider`、8 类（AMD/Cataract/Diabetic Retinopathy/Glaucoma/Hypertensive Retinopathy/Myopia/Normal/Other）、`llm_enabled=true`、`history_ready=true`。
- 训练/对比日志存在于 `runs\comparison_notebook\`、`runs\integration_checks\`（baseline / akconv / yolo26n / yolo26n_akconv）。
- ⚠️ 实验数字是否都能追溯到日志：**尚未逐项核对**（待做）。
- ⚠️ 部署权重 AKConv 层数 = 0 已在报告中披露，与线上一致。

## 5. 自动化测试结果

- 上一位 AI 声称 `52 passed`。
- **本轮未能复现验证**：项目自带 `runtime/clean_env` 的 venv 已损坏（`pyvenv.cfg` 不可读），
  系统 Anaconda Python 无 pytest，沙箱 Python 也无 pytest。**未安装依赖强跑，避免得出误导性失败数**。
- **重要：四份报告正文中均未出现"52 passed"或任何测试通过率声明**（已逐份全文检索确认）。
  → 因此该数字**不构成报告造假风险**，它只是内部记录。
- 如需自行验证，在你可用的环境执行：
  `python -m pytest tests -q` 并记录 命令／Python 版本／通过数／失败数／耗时，回填本文件。

## 5.1 实验数字溯源（已验证 ✅）

| 报告中的数字 | 溯源结果 |
|---|---|
| top1 = 63.285% | ✅ `runs/comparison_summary.csv`、`docs/evidence/course_evidence_audit.json`（`best_top1_pct`） |
| top1 = 63.199% | ✅ 同上（AKConv 变体） |
| 基线参数量 1,541,352 | ✅ comparison_summary.csv 参数列 |
| AKConv 候选参数量 1,487,772 | ✅ 同上 |
| AKConv 层数 基线=0 / 候选=6 | ✅ 同上，与"部署权重 AKConv=0"的披露一致 |

结论：**实验报告与模型分析报告中的关键数字均可追溯到真实日志，未发现编造。**
另注意：baseline 与 yolo26n 数值与权重哈希完全相同，报告已诚实披露"不能作为四种不同架构的性能比较"。

## 6. 服务器状态（已核实）

- 公网：`http://101.132.104.41:8000`（阿里云 Ubuntu 22.04，2 核 / ~1.6GB，无 GPU，已加 2G swap）
- systemd 单元 `eye-diagnosis`，uvicorn 直听 `:8000`
- 前端版本：**v2026-10-14.23**（已确认 131,626 字节，`disease` 上下文字段存在）
- ⚠️ `EYE_ACCESS_CODE` 未设置 → **服务免口令开放**（安全风险，待用户决定是否加口令）

## 7. 交付包检查（已核实）

`deliverables\final_submission\ZY0302308_基于YOLO与大语言模型的眼科图像分类辅助诊断系统_欧阳宸230722105_汪晓蕾231613212.zip`

- 545 条目，未压缩 27.0 MB，压缩包 21.1 MB
- ✅ 四份报告齐全（`课程设计报告\` 下）
- ✅ **未发现** API Key / Token / 密码（文件名 + 文本内容双重扫描）
- ✅ 不含数据集原始文件（无 `data/` `medical_images/`）
- ✅ 不含 `runtime/` 缓存与备份
- ✅ 含模型权重：`models/server/best512.onnx`、`models/custom/...`
- ✅ **无缓存垃圾**：`__pycache__`／`.pyc`／`.ipynb_checkpoints`／`.pytest_cache`／`.git` 等扫描结果为 **0 条**
- ✅ `ultralytics-main` **判定：保留**（296 条 / 2.6 MB，占包约 10%）。
  理由：它不是纯第三方依赖，含本项目自定义核心代码——
  `ultralytics/nn/modules/akconv.py`、`ultralytics/cfg/models/11/yolo11-cls-AKConv.yaml`、
  且 `requirements-core.txt` 以 `-e ./ultralytics-main` 本地可编辑安装。
  删除会丢失自定义模块、导致消融实验不可复现。任务书"不含依赖库"针对的是未经修改的第三方库，此项属改造后的框架源码。
- ⚠️ 含 `models/custom/AKConv_best_moudle/`（目录名带 AKConv，但报告已声明部署未使用；目录名可能引起评审误解，建议保留但依赖报告中的明确披露）

## 8. 本轮已完成的修改

1. 新建 `PROJECT_HANDOFF.md`（本文件）。
2. **四份报告补页码**：页脚插入居中 `PAGE` 域（4 份）。
3. **四份报告补标题层级**：19 个一级标题挂 `Heading1`；同时修正 Heading1/2/3 样式字体为宋体、颜色为黑。
4. **交付副本同步 + 重新打包**：
   - 副本已更新至 `deliverables/final_submission/ZY0302308_.../课程设计报告/`
   - 旧压缩包保留为 `..._v1备份_20261014T165749.zip`（未删除）
   - 新包重建：`ZY0302308_...zip`，460 条目、20.9 MB（已剔除 `__pycache__` / `.pyc` 等缓存）
   - 新包复验：四份报告标题数与页码域均正确

> 全程未重新训练模型、未改动线上服务器、未重写任何报告正文。

## 9. 未完成事项

| 项 | 状态 | 说明 |
|---|---|---|
| 报告补页码 | ✅ 已完成 | 四份均插入 PAGE 域并复验 |
| 报告补标题层级（Heading 样式） | ✅ 已完成 | 19 个 Heading1，导航/目录可用 |
| 实验数字溯源核对 | ✅ 已完成 | 63.285% / 63.199%、参数量、AKConv 层数均能溯源至 `runs/comparison_summary.csv` 与 `docs/evidence/course_evidence_audit.json` |
| 重新运行 pytest 验证 52 passed | ⚠️ 未验证 | 项目 venv `runtime/clean_env` 的 `pyvenv.cfg` 损坏；报告正文**未声称** 52 passed，故不构成造假风险 |
| 交付包清理 ultralytics-main | ✅ 已决策保留 | 内含自定义 `akconv.py` 与 `yolo11-cls-AKConv.yaml`，删除将丢失复现能力；任务书"不含依赖库"指未改动的第三方库 |
| 服务器加访问口令 | 待用户决定 | 当前免口令开放 |
| 学校任务书核对 | ✅ 已完成 | 用户已提供任务书，符合性矩阵见 §3.1，全部达标 |

## 10. 需要用户现场验收的功能（不得伪造）

以下均**未实机验证**，报告中不得写成"已通过"：
- 摄像头稳定采集、重复开关与拍照
- 麦克风语音识别
- 自然音色 MP3 播放、音频缓存与停止播放
- 服务器浏览器端完整操作
- 开发板实机验收
- 桌面端干净 Windows 环境从零安装复核

## 11. 下一位 AI 应执行的步骤

1. 先读本文件，勿以 `89fb11c` 为基线（当前 HEAD 为 `1346f61`）。
2. 修改报告前确认 `runtime/report_backups/` 已有原始备份（已有）。
3. ~~优先修 P0：补页码 → 补 Heading 层级~~ 已于 2026-10-14 完成；剩余可选项仅"重跑 pytest 留证"（需先修复 `runtime/clean_env` venv）。
4. 任何实验数字必须能追溯到 `runs/` 下日志或结果文件，无法追溯的一律标注"待核实"，**禁止编造**。
5. 涉及线上服务器重启/覆盖、重新训练、重新划分数据集：必须先取得用户同意。
