"""生成服务器部署包：只含运行必需文件，排除数据集、桌面端、实验产物与本机配置。

用法：
    python scripts/make_server_package.py
    python scripts/make_server_package.py --output dist/eye-server --zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 运行服务端必需的文件，路径相对项目根
REQUIRED_FILES = (
    "src/__init__.py",
    "src/core/__init__.py",
    "src/core/classification.py",
    "src/core/history.py",
    "src/core/llm.py",
    "src/core/reference_advice.py",
    "src/server/__init__.py",
    "src/server/app.py",
    "src/server/onnx_classifier.py",
    "configs/__init__.py",
    "configs/runtime_config.py",
    "configs/app_config.example.json",
    "requirements-server.txt",
)

REQUIRED_DIRS = (
    "src/server/static",
)

OPTIONAL_FILES = (
    "models/server/best512.onnx",
    "models/custom/AKConv_best_moudle/best.onnx",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成服务器部署包")
    parser.add_argument("--output", default="dist/eye-server")
    parser.add_argument("--zip", action="store_true", help="额外生成 zip 压缩包")
    args = parser.parse_args()

    target = (PROJECT_ROOT / args.output).resolve()
    dist_root = (PROJECT_ROOT / "dist").resolve()
    if target == dist_root or dist_root not in target.parents:
        raise SystemExit("输出目录必须位于项目 dist 目录内，且不能直接使用 dist 根目录")
    if target.exists():
        raise SystemExit(f"输出目录已存在，为避免覆盖请换一个版本化目录: {target}")
    target.mkdir(parents=True)

    missing = []
    for relative in REQUIRED_FILES:
        source = PROJECT_ROOT / relative
        if not source.is_file():
            missing.append(relative)
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for relative in REQUIRED_DIRS:
        source = PROJECT_ROOT / relative
        if not source.is_dir():
            missing.append(relative)
            continue
        shutil.copytree(source, target / relative,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 权重：优先部署副本，缺失时回退到仓库内原始 ONNX
    weight_copied = None
    for relative in OPTIONAL_FILES:
        source = PROJECT_ROOT / relative
        if source.is_file():
            destination = target / "models" / "server" / "best512.onnx"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            weight_copied = relative
            break
    if weight_copied is None:
        missing.append("ONNX 权重（先运行 scripts/export_onnx.py 或准备 best.onnx）")

    shutil.copy2(PROJECT_ROOT / "deploy" / "server" / "deploy.sh", target / "deploy.sh")
    shutil.copy2(PROJECT_ROOT / "deploy" / "server" / "eye-diagnosis.service.example",
                 target / "eye-diagnosis.service.example")

    print(f"部署包已生成: {target}")
    print(f"权重来源: {weight_copied}")
    if missing:
        print("缺失文件: " + "；".join(missing))
    else:
        print("必需文件齐全")

    if args.zip:
        archive = shutil.make_archive(str(target), "zip", root_dir=target)
        print(f"压缩包: {archive}")


if __name__ == "__main__":
    sys.exit(main())
