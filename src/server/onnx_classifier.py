"""服务端 ONNX 推理模块。

不依赖 PyTorch、Ultralytics 与 PyQt5，可在 2GB 内存的服务器上运行。
复用 src/core/classification.py 的八类定义，保证与桌面端结果口径一致。

预处理对齐 ultralytics-main/ultralytics/data/augment.py::classify_transforms：
  1. RGB 输入（上传图片解码后即为 RGB，桌面端是 cv2 BGR 转 RGB，此处等价）
  2. Resize：较短边缩放到模型声明的输入尺寸（当前权重为 512）
  3. 按同一尺寸 CenterCrop
  4. /255 后转 CHW
  5. Normalize(mean=0, std=1)，本项目配置为恒等变换
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_CODES = ("A", "C", "D", "G", "H", "M", "N", "O")
DISEASES = ("AMD", "Cataract", "Diabetic Retinopathy", "Glaucoma",
            "Hypertensive Retinopathy", "Myopia", "Normal", "Other")
IMAGE_SIZE = 512
INPUT_NAME_HINT = "images"


class OnnxClassifierError(RuntimeError):
    pass


@dataclass
class Prediction:
    success: bool
    disease: str = ""
    confidence: float = 0.0
    probabilities: dict = field(default_factory=dict)
    error: str = ""


def load_image_bytes(data: bytes) -> Image.Image:
    """用字节解码，避免中文路径与不同平台的文件读取差异。"""
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as error:  # noqa: BLE001 - 需向调用方暴露明确原因
        raise OnnxClassifierError(f"图像解码失败: {error}") from error


def preprocess(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """复刻 classify_transforms：短边缩放 + 中心裁剪 + 归一化到 0-1。"""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise OnnxClassifierError("图像尺寸无效")
    # torchvision 的 Resize(int) 语义：较短边变为 size，另一边按 int 截断等比缩放
    if width < height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)
    if (new_width, new_height) != (width, height):
        image = image.resize((new_width, new_height), Image.BILINEAR)
    left = max((new_width - size) // 2, 0)
    top = max((new_height - size) // 2, 0)
    image = image.crop((left, top, left + size, top + size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))  # HWC -> CHW
    return np.ascontiguousarray(array[None, ...])  # 增加 batch 维


class OnnxClassifier:
    """加载 best.onnx 并执行八类分类。"""

    def __init__(self, model_path: str | Path, providers=None, image_size: int | None = None):
        try:
            import onnxruntime as ort
        except ImportError as error:  # noqa: BLE001
            raise OnnxClassifierError("缺少 onnxruntime，请先安装 requirements-server.txt") from error

        path = Path(model_path)
        if not path.is_file():
            raise OnnxClassifierError(f"ONNX 权重不存在: {path}")

        self.model_path = str(path)
        self._ort = ort
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 2GB 内存服务器：关闭 CPU 内存池可显著降低峰值 RSS，避免 worker 被 OOM kill
        #（worker 死后 nginx 会对所有 /api/ 请求返回 502）
        options.enable_cpu_mem_arena = False
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(path), options, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        in_shape = self.session.get_inputs()[0].shape
        if len(in_shape) != 4 or in_shape[1] != 3:
            raise OnnxClassifierError(f"ONNX 输入形状应为 NCHW 三通道，实际: {in_shape}")
        # 以权重自身声明的尺寸为准；当前桌面端与服务端都使用 512 输入。
        declared = in_shape[2]
        if isinstance(declared, int) and declared > 0:
            self.input_size = declared
        else:
            self.input_size = image_size or IMAGE_SIZE
        if image_size and image_size != self.input_size:
            print(f"[警告] 指定输入尺寸 {image_size} 与权重声明的 {self.input_size} 不一致，"
                  f"以权重声明尺寸为准")

        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) == 2 and isinstance(out_shape[1], int) and out_shape[1] != 8:
            raise OnnxClassifierError(f"ONNX 输出应为 8 类，实际: {out_shape[1]}")

        self._needs_softmax = None  # 首次推理时根据输出和判定
        self.class_names = dict(enumerate(DISEASES))

    def model_info(self) -> dict:
        return {
            "weight_path": self.model_path,
            "runtime": self._ort.__version__,
            "providers": self.session.get_providers(),
            "input_name": self.input_name,
            "output_logits": self._needs_softmax,
            "classes": self.class_names,
            "image_size": self.input_size,
        }

    def _softmax_if_needed(self, values: np.ndarray) -> np.ndarray:
        if self._needs_softmax is None:
            total = float(values.sum())
            self._needs_softmax = not math.isclose(total, 1.0, abs_tol=1e-3)
        if not self._needs_softmax:
            return values
        shifted = values - values.max()
        exp = np.exp(shifted)
        return exp / exp.sum()

    def predict_array(self, batch: np.ndarray) -> Prediction:
        try:
            outputs = self.session.run([self.output_name], {self.input_name: batch})
        except Exception as error:  # noqa: BLE001
            return Prediction(False, error=f"ONNX 推理失败: {error}")
        values = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        if values.shape != (8,):
            return Prediction(False, error=f"输出类别数不是 8，实际: {values.shape}")
        if not np.isfinite(values).all():
            return Prediction(False, error="输出包含非有限数值")
        values = self._softmax_if_needed(values)
        if (values < 0).any() or (values > 1).any():
            return Prediction(False, error="概率超出 0 到 1 范围")
        index = int(np.argmax(values))
        return Prediction(True, DISEASES[index], float(values[index]),
                          {DISEASES[i]: float(values[i]) for i in range(8)})

    def predict_image(self, image: Image.Image) -> Prediction:
        try:
            return self.predict_array(preprocess(image, self.input_size))
        except OnnxClassifierError as error:
            return Prediction(False, error=str(error))

    def predict_bytes(self, data: bytes) -> Prediction:
        try:
            return self.predict_image(load_image_bytes(data))
        except OnnxClassifierError as error:
            return Prediction(False, error=str(error))

    def predict_file(self, path: str | Path) -> Prediction:
        # fromfile 避免 Windows 下中文路径读取失败
        try:
            data = np.fromfile(str(path), dtype=np.uint8).tobytes()
        except OSError as error:
            return Prediction(False, error=f"图像读取失败: {error}")
        if not data:
            return Prediction(False, error="图像为空")
        return self.predict_bytes(data)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="ONNX 分类推理与概率导出")
    parser.add_argument("--weights", required=True, help="best.onnx 路径")
    parser.add_argument("--image-dir", help="批量推理目录")
    parser.add_argument("--image", help="单张图片路径")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少张，0 表示全部")
    parser.add_argument("--imgsz", type=int, default=0, help="覆盖输入尺寸，0 表示以权重声明为准")
    parser.add_argument("--output", help="导出 JSON 路径，用于与桌面端对比")
    args = parser.parse_args()

    classifier = OnnxClassifier(args.weights, image_size=args.imgsz or None)
    targets = []
    if args.image:
        targets.append(Path(args.image))
    if args.image_dir:
        directory = Path(args.image_dir)
        # data/val 按类别分子目录，需要递归遍历
        targets.extend(sorted(p for p in directory.rglob("*")
                              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}))
    if args.limit > 0:
        targets = targets[:args.limit]

    results = {}
    for path in targets:
        prediction = classifier.predict_file(path)
        results[str(path)] = ([prediction.probabilities[name] for name in DISEASES]
                              if prediction.success else None)
        if not prediction.success:
            print(f"[失败] {path.name}: {prediction.error}")

    print(json.dumps({"model_info": classifier.model_info(), "classes": list(DISEASES)},
                     ensure_ascii=False, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps({"model": "onnx", "classes": list(DISEASES), "results": results},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"已导出 {len(results)} 条结果到 {args.output}")


if __name__ == "__main__":
    _cli()
