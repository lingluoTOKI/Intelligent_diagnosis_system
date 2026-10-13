"""导出同一输入张量下桌面端 PyTorch 的原始 logits，用于与 ONNX 逐位比对。

用法（GUI/测试环境解释器）：
    D:/Anaconda3/envs/yolov10/python.exe scripts/debug_logits.py --image <图片路径>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 PyTorch logits")
    parser.add_argument("--image", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--input-out", default="runtime/parity/input.npy")
    parser.add_argument("--logits-out", default="runtime/parity/torch_logits.json")
    args = parser.parse_args()

    from configs.runtime_config import load_app_config  # noqa: E402
    from src.core.classification import configured_yolo  # noqa: E402

    from src.server.onnx_classifier import preprocess  # noqa: E402

    from PIL import Image  # noqa: E402
    import cv2  # noqa: E402

    config = load_app_config()
    factory, _ = configured_yolo(config)
    model = factory(str(config.model_path))

    image_bgr = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    batch = preprocess(rgb_image, args.size)  # 与服务端完全相同的输入

    input_path = Path(args.input_out)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(input_path), batch)

    with torch.no_grad():
        raw = model.model(torch.from_numpy(batch))
        while isinstance(raw, (tuple, list)):
            raw = raw[0]
        logits = np.asarray(raw.detach().cpu().numpy(), dtype=np.float64).reshape(-1)

    Path(args.logits_out).write_text(json.dumps({"logits": [float(v) for v in logits]}),
                                     encoding="utf-8")
    print(f"输入张量已保存: {input_path}  shape={batch.shape}")
    print("PyTorch logits:", [round(float(v), 5) for v in logits])
    print("softmax 后:", [round(float(v), 5) for v in
                          (np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum())])


if __name__ == "__main__":
    main()
