"""Build a non-destructive course-submission folder and ZIP (no dataset or secrets)."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "final_submission"
PACKAGE = "ZY0302308_基于YOLO与大语言模型的眼科图像分类辅助诊断系统_欧阳宸230722105_汪晓蕾231613212"
TARGET = OUT / PACKAGE

FILES = [
    "README.md", "CHANGELOG.md", "PROJECT_STRUCTURE.md", "启动系统.cmd",
    "requirements.txt", "requirements-core.txt", "requirements-pc.txt",
    "requirements-server.txt", "requirements-voice.txt", "requirements-board.txt",
]
DIRS = ["src", "assets", "configs", "deploy", "docs", "tests", "models", "scripts"]
SKIP_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "runtime", "data", ".git", "deliverables"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".key", ".pem", ".lnk"}
SKIP_FILES = {"saved_api_key.txt", "app_config.json", "ui_preferences.json", "system_config.json"}


def ignore(path: str, names: list[str]) -> set[str]:
    path = Path(path)
    excluded = set()
    for name in names:
        candidate = path / name
        if name in SKIP_NAMES or name in SKIP_FILES or candidate.suffix.lower() in SKIP_SUFFIXES:
            excluded.add(name)
    return excluded


def copy_file(rel: str) -> None:
    source = ROOT / rel
    if source.exists():
        destination = TARGET / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Replace only the generated submission folder; project files are never deleted.
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir()
    for item in FILES:
        copy_file(item)
    for name in DIRS:
        source = ROOT / name
        if source.exists():
            shutil.copytree(source, TARGET / name, ignore=ignore)
    reports = ROOT / "deliverables" / "course_design"
    report_target = TARGET / "课程设计报告"
    report_target.mkdir()
    for report in reports.glob("*.docx"):
        shutil.copy2(report, report_target / report.name)
    note = TARGET / "提交说明.txt"
    note.write_text(
        "本压缩包不含数据集、个人密钥、本地运行时状态及临时文件。\n"
        "数据来源、重复样本审计和已知限制见 docs/guide/数据来源与划分审计.md。\n"
        "实机验收未完成项见 docs/guide/实机验收记录.md。\n",
        encoding="utf-8",
    )
    archive = OUT / PACKAGE
    shutil.make_archive(str(archive), "zip", OUT, PACKAGE)
    print(TARGET)
    print(archive.with_suffix(".zip"))


if __name__ == "__main__":
    main()
