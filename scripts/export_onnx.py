"""按桌面端配置导出 ONNX 权重。

仓库里现存的 best.onnx 是 512 输入导出，桌面端实际按 224 推理，两者不可混用。
本脚本使用与桌面端相同的本地框架和 image_size 配置重新导出，供服务器部署使用。

用法（GUI/测试环境解释器，项目根目录执行）：
    D:/Anaconda3/envs/yolov10/python.exe scripts/export_onnx.py --output models/server/best224.onnx
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def declared_size(model) -> int:
    """权重自带的 transforms 优先于 imgsz 参数，导出必须沿用它的尺寸。"""
    transforms = getattr(getattr(model, "model", None), "transforms", None)
    for transform in getattr(transforms, "transforms", []) or []:
        size = getattr(transform, "size", None)
        if isinstance(size, int):
            return size
        if isinstance(size, (tuple, list)) and size:
            return int(size[0])
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="导出与桌面端一致的 ONNX 权重")
    parser.add_argument("--weights", help="源 .pt 权重，默认取应用配置 model_path")
    parser.add_argument("--imgsz", type=int, default=0, help="输入尺寸，0 表示取权重自带尺寸")
    parser.add_argument("--output", default="models/server/best512.onnx")
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    from configs.runtime_config import load_app_config  # noqa: E402
    from src.core.classification import configured_yolo  # noqa: E402

    config = load_app_config()
    source = Path(args.weights) if args.weights else config.model_path
    if not Path(source).is_file():
        raise SystemExit(f"源权重不存在: {source}")
    factory, framework = configured_yolo(config)
    print(f"框架: {getattr(framework, '__version__', 'unknown')}  来源: {config.framework_path}")
    print(f"源权重: {source}  配置 image_size: {config.image_size}")

    model = factory(str(source))
    if getattr(model, "task", None) != "classify":
        raise SystemExit(f"仅支持分类权重，当前任务: {getattr(model, 'task', None)}")

    # 桌面端实际使用权重自带的 transforms，imgsz 参数会被忽略
    inherent = declared_size(model)
    if inherent and args.imgsz and args.imgsz != inherent:
        print(f"[警告] 权重自带预处理尺寸为 {inherent}，指定 {args.imgsz} 会与桌面端不一致")
    imgsz = args.imgsz or inherent
    if inherent and imgsz != config.image_size:
        print(f"[注意] 配置 image_size={config.image_size} 与权重自带 {inherent} 不一致，"
              f"按权重自带的 {imgsz} 导出以保持与桌面端一致")
    print(f"导出输入尺寸: {imgsz}")

    exported = model.export(format="onnx", imgsz=imgsz, opset=args.opset,
                            dynamic=False, simplify=False)
    exported = Path(exported)
    target = PROJECT_ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    # export 会写在源权重同目录，移动到目标位置，避免覆盖原有 best.onnx
    shutil.move(str(exported), str(target))
    print(f"已导出: {target}  ({target.stat().st_size / 1024 / 1024:.2f} MB)")
    print("下一步: 用 scripts/compare_prob_outputs.py 与桌面端结果比对")


if __name__ == "__main__":
    main()
