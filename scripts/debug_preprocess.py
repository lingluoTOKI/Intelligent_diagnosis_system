"""定位桌面端与服务端预处理差异：对比两侧送入模型的张量。

用法（GUI/测试环境解释器，项目根目录执行）：
    D:/Anaconda3/envs/yolov10/python.exe scripts/debug_preprocess.py --image <图片路径>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="对比预处理张量")
    parser.add_argument("--image", required=True)
    parser.add_argument("--size", type=int, default=512,
                        help="输入尺寸，默认 512（权重自带预处理尺寸）")
    args = parser.parse_args()

    from configs.runtime_config import load_app_config  # noqa: E402

    config = load_app_config()
    if str(config.framework_path) not in sys.path:
        sys.path.insert(0, str(config.framework_path))

    from ultralytics.data.augment import classify_transforms  # noqa: E402

    from src.server.onnx_classifier import preprocess  # noqa: E402

    image_bgr = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit("图像解码失败")
    rgb_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    transform = classify_transforms(args.size, crop_fraction=1.0)
    reference = transform(rgb_image).numpy().astype(np.float32)
    candidate = preprocess(rgb_image, args.size)[0]

    print(f"原图尺寸(BGR): {image_bgr.shape[1]}x{image_bgr.shape[0]}")
    print(f"桌面端张量: shape={reference.shape} range=[{reference.min():.4f}, {reference.max():.4f}] "
          f"mean={reference.mean():.4f}")
    print(f"服务端张量: shape={candidate.shape} range=[{candidate.min():.4f}, {candidate.max():.4f}] "
          f"mean={candidate.mean():.4f}")
    if reference.shape != candidate.shape:
        print("形状不一致，预处理链路不同")
        return
    diff = np.abs(reference - candidate)
    print(f"最大绝对差: {diff.max():.6f}  平均绝对差: {diff.mean():.6f}")
    if diff.max() < 1e-4:
        print("结论: 预处理一致，问题在推理环节")
    else:
        print("结论: 预处理不一致")
        # 逐通道看是否是通道顺序或缩放问题
        for index, name in enumerate("RGB"):
            print(f"  通道 {name}: 桌面端均值 {reference[index].mean():.4f} "
                  f"服务端均值 {candidate[index].mean():.4f}")


if __name__ == "__main__":
    main()
