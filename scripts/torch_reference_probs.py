"""导出桌面端（PyTorch + Ultralytics）分类概率，作为服务端 ONNX 的一致性基准。

用法（GUI/测试环境解释器，项目根目录执行）：
    D:/Anaconda3/envs/yolov10/python.exe scripts/torch_reference_probs.py ^
        --weights models/custom/AKConv_best_moudle/best.pt ^
        --image-dir data/val --limit 20 --output runtime/parity/torch.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="导出桌面端分类概率")
    parser.add_argument("--weights", help="best.pt 路径，默认取应用配置")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from src.core.classification import DISEASES, EyeDiseaseDetector  # noqa: E402

    detector = EyeDiseaseDetector()
    if not detector.load_model(args.weights):
        raise SystemExit(f"桌面端模型加载失败: {detector.last_error}")

    directory = Path(args.image_dir)
    # data/val 按类别分子目录，需要递归遍历
    paths = sorted(p for p in directory.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"目录内没有可用图片: {directory}")

    results = {}
    for path in paths:
        # 与 GUI 保持一致：fromfile + imdecode，得到 BGR uint8
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            results[str(path)] = None
            print(f"[跳过] {path.name}: 无法解码")
            continue
        result = detector.classify(image)
        if not result.success:
            results[str(path)] = None
            print(f"[失败] {path.name}: {result.error}")
            continue
        results[str(path)] = [result.probabilities[name] for name in DISEASES]

    payload = {"model": "torch", "classes": list(DISEASES), "results": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已导出 {len(results)} 条桌面端结果到 {out_path}")
    print("model_info:", json.dumps(detector.model_info, ensure_ascii=False))


if __name__ == "__main__":
    main()
