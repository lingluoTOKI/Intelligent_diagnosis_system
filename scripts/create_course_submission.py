"""Build a non-destructive course-submission folder and ZIP (no dataset or secrets)."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "final_submission"
PACKAGE = "ZY0302308_基于YOLO与大语言模型的眼科图像分类辅助诊断系统_欧阳宸230722105_汪晓蕾231613212"
TARGET = OUT / PACKAGE

FILES = [
    "README.md", "CHANGELOG.md", "PROJECT_STRUCTURE.md", "LICENSE", "启动系统.cmd",
    "visualization_test2.py", "requirements-dev.txt", "ablation.ipynb",
    "requirements.txt", "requirements-core.txt", "requirements-pc.txt",
    "requirements-server.txt", "requirements-voice.txt", "requirements-board.txt",
]
DIRS = ["src", "assets", "configs", "deploy", "docs", "tests", "models", "scripts"]
SKIP_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git", "text", "voice_output"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".key", ".pem", ".lnk"}
SKIP_FILES = {"saved_api_key.txt", "app_config.json", "ui_preferences.json", "system_config.json"}
REPORTS = ("文献分析报告.docx", "技术方案设计报告.docx", "模型分析报告.docx", "实验分析报告.docx")
REQUIRED = (*FILES, "ultralytics-main/pyproject.toml", "ultralytics-main/ultralytics/__init__.py",
            "ultralytics-main/ultralytics/data/augment.py", "configs/app_config.example.json", "configs/system_config.example.json", "scripts/launch_desktop.py",
            "models/custom/AKConv_best_moudle/best.pt", "models/server/best512.onnx",
            *("课程设计报告/" + name for name in REPORTS))


def validate_package(target: Path) -> None:
    missing = [name for name in REQUIRED if not (target / name).is_file()]
    if missing:
        raise RuntimeError("提交包缺少必需文件: " + "; ".join(missing))
    forbidden = []
    for file in target.rglob("*"):
        if not file.is_file():
            continue
        rel = file.relative_to(target)
        if rel.parts[0] in {"data", "runtime", "deliverables"} or any(part in SKIP_NAMES for part in rel.parts) or file.name in SKIP_FILES or file.name == ".env" or file.name.startswith(".env.") or file.suffix.lower() in SKIP_SUFFIXES:
            forbidden.append(str(rel))
    if forbidden:
        raise RuntimeError("提交包包含本机或私有文件: " + "; ".join(forbidden))
    values = json.loads((target / "configs/app_config.example.json").read_text(encoding="utf-8"))
    for name in ("model_path", "framework_path"):
        if not (target / values[name]).exists():
            raise RuntimeError("默认配置引用的路径不存在: " + values[name])


def ignore(path: str, names: list[str]) -> set[str]:
    path = Path(path)
    excluded = set()
    for name in names:
        candidate = path / name
        if name in SKIP_NAMES or name in SKIP_FILES or name == ".env" or name.startswith(".env.") or candidate.suffix.lower() in SKIP_SUFFIXES:
            excluded.add(name)
    return excluded


def copy_file(rel: str) -> None:
    source = ROOT / rel
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = TARGET / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep previous generated folders for comparison; never recursively delete them.
    if TARGET.exists():
        previous = ROOT / "runtime" / "submission_backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if not TARGET.resolve().is_relative_to(OUT.resolve()) or not previous.resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError("拒绝移动项目交付区域之外的目录")
        previous.parent.mkdir(parents=True, exist_ok=True)
        TARGET.rename(previous)
    TARGET.mkdir()
    for item in FILES:
        copy_file(item)
    if (ROOT / "眼科图像分类辅助诊断系统.exe").is_file():
        copy_file("眼科图像分类辅助诊断系统.exe")
    for name in DIRS:
        source = ROOT / name
        if source.exists():
            shutil.copytree(source, TARGET / name, ignore=ignore)
    # The editable requirement and default runtime configuration use this local framework.
    framework = TARGET / "ultralytics-main"
    framework.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        copy_file("ultralytics-main/" + name)
    shutil.copytree(ROOT / "ultralytics-main/ultralytics", framework / "ultralytics", ignore=ignore)
    # Preserve training code without distributing local cell outputs or private file paths.
    notebook = json.loads((ROOT / "ablation_fast_workers8.ipynb").read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    (TARGET / "ablation_fast_workers8.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    reports = ROOT / "deliverables" / "course_design"
    report_target = TARGET / "课程设计报告"
    report_target.mkdir()
    for name in REPORTS:
        report = reports / name
        shutil.copy2(report, report_target / report.name)
    note = TARGET / "提交说明.txt"
    note.write_text(
        "本压缩包不含数据集、个人密钥、本地运行时状态及临时文件。\n"
        "四份报告为事实纠错基础稿，后续须补充逐篇文献评述、设计图和实验深入分析；本包不是报告终稿。\n"
        "数据来源、重复样本审计和已知限制见 docs/guide/数据来源与划分审计.md。\n"
        "实机验收未完成项见 docs/guide/实机验收记录.md。\n",
        encoding="utf-8",
    )
    archive = OUT / PACKAGE
    validate_package(TARGET)
    files = {p.relative_to(TARGET).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in TARGET.rglob("*") if p.is_file()}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8").splitlines()
    (TARGET / "交付文件校验清单.json").write_text(json.dumps({"commit": commit, "working_tree_changes": status,
        "built_at": datetime.now().isoformat(), "files": files}, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.make_archive(str(archive), "zip", OUT, PACKAGE)
    print(TARGET)
    print(archive.with_suffix(".zip"))


if __name__ == "__main__":
    main()
