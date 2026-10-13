"""Eight-class inference with validated outputs and explicit failure states."""
from dataclasses import dataclass, field
from pathlib import Path
import importlib
import math
import os
import sys
import threading
import numpy as np

from configs.runtime_config import PROJECT_ROOT, load_app_config, project_path

CLASS_CODES = ("A", "C", "D", "G", "H", "M", "N", "O")
DISEASES = ("AMD", "Cataract", "Diabetic Retinopathy", "Glaucoma",
            "Hypertensive Retinopathy", "Myopia", "Normal", "Other")
LETTER_TO_DISEASE = dict(zip(CLASS_CODES, DISEASES))


class ClassificationError(RuntimeError):
    pass


def valid_prediction(disease, confidence):
    try:
        value = float(confidence)
        return disease in DISEASES and math.isfinite(value) and 0 < value <= 1
    except (TypeError, ValueError):
        return False


def canonical_name(name):
    if name in LETTER_TO_DISEASE:
        return LETTER_TO_DISEASE[name]
    normalized = str(name).replace("_", " ").strip().casefold()
    aliases = {value.casefold(): value for value in DISEASES}
    aliases.update({"age related macular degeneration": "AMD",
                    "age-related macular degeneration": "AMD",
                    "myopic maculopathy": "Myopia"})
    return aliases.get(normalized)


def configured_yolo(config):
    package_dir = (config.framework_path / "ultralytics").resolve()
    if not (package_dir / "__init__.py").is_file():
        raise ClassificationError(f"配置的框架不存在: {config.framework_path}")
    cached = sys.modules.get("ultralytics")
    if cached is not None and not Path(cached.__file__).resolve().is_relative_to(package_dir):
        raise ClassificationError("当前进程已加载其他Ultralytics框架，请重启并检查framework_path")
    runtime = PROJECT_ROOT / "runtime" / "ultralytics"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime))
    source = str(config.framework_path)
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module("ultralytics")
    if not Path(module.__file__).resolve().is_relative_to(package_dir):
        raise ClassificationError("Ultralytics来源与配置不一致")
    return module.YOLO, module


@dataclass
class ClassificationResult:
    success: bool
    disease_name: str = None
    confidence: float = 0.0
    probabilities: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)
    error: str = None
    raw_results: object = field(default=None, repr=False)

    def to_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "raw_results"}


class EyeDiseaseDetector:
    letter_to_disease = LETTER_TO_DISEASE

    def __init__(self, config=None, model_factory=None):
        self.config = config or load_app_config()
        self._factory = model_factory
        self._lock = threading.RLock()
        self.model = None
        self.current_model_path = None
        self.class_names = dict(enumerate(DISEASES))
        self.model_info = {}
        self.last_error = None

    def load_model(self, model_path=None):
        with self._lock:
            # A failed switch must not leave the previous model usable as the new one.
            self.model = None
            self.current_model_path = None
            self.model_info = {}
            self.class_names = dict(enumerate(DISEASES))
            try:
                path = project_path(model_path or self.config.model_path)
                if not path.is_file() or path.suffix.lower() != ".pt":
                    raise ClassificationError(f"分类权重不存在或格式错误: {path}")
                if self._factory is None:
                    factory, framework = configured_yolo(self.config)
                    import torch
                    device = self.config.device
                    if device == "auto":
                        device = "0" if torch.cuda.is_available() else "cpu"
                    if device != "cpu" and not torch.cuda.is_available():
                        raise ClassificationError("配置要求GPU，但CUDA不可用")
                else:
                    factory, framework, device = self._factory, None, "cpu"
                model = factory(str(path))
                if getattr(model, "task", None) != "classify":
                    raise ClassificationError("仅支持8类图像分类权重")
                names = getattr(model, "names", {})
                if not isinstance(names, dict):
                    names = dict(enumerate(names))
                mapped = {int(index): canonical_name(name) for index, name in names.items()}
                if set(mapped) != set(range(8)) or set(mapped.values()) != set(DISEASES):
                    raise ClassificationError("权重类别与A/C/D/G/H/M/N/O八类不一致")
                device_name = "CPU"
                if self._factory is None:
                    target_device = "cpu" if device == "cpu" else "cuda:" + str(device).replace("cuda:", "").split(",")[0]
                    model.to(target_device)
                    if target_device != "cpu":
                        device_name = torch.cuda.get_device_name(int(target_device.split(":")[1]))
                self.model = model
                self.current_model_path = str(path)
                self.class_names = mapped
                self.model_info = {
                    "weight_path": str(path), "task": "classify", "device": device,
                    "framework_path": str(self.config.framework_path),
                    "framework_version": getattr(framework, "__version__", "injected-backend"),
                    "device_name": device_name,
                    "classes": mapped,
                    "akconv_layers": sum(type(layer).__name__ == "AKConv"
                        for layer in model.model.modules()) if getattr(model, "model", None) is not None else 0,
                }
                self.last_error = None
                return True
            except Exception as error:
                self.last_error = str(error)
                return False

    def predict(self, image):
        if self.model is None:
            raise ClassificationError(self.last_error or "请先加载分类模型")
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ClassificationError("图像为空或格式不正确")
        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
            raise ClassificationError("图像通道数不支持")
        if image.dtype != np.uint8:
            raise ClassificationError("图像必须为uint8像素数据")
        with self._lock:
            try:
                return self.model.predict(image, verbose=False, imgsz=self.config.image_size,
                                          device=self.model_info["device"])
            except Exception as error:
                raise ClassificationError(f"分类推理失败: {error}") from error

    def parse_result(self, raw):
        try:
            probs = getattr(raw, "probs", None)
            if probs is None:
                raise ClassificationError("结果缺少分类概率")
            values = probs.data
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            values = np.asarray(values, dtype=float)
            if values.shape != (8,) or not np.isfinite(values).all():
                raise ClassificationError("分类概率数量或数值无效")
            if (values < 0).any() or (values > 1).any() or not np.isclose(values.sum(), 1, atol=1e-3):
                raise ClassificationError("分类概率必须在0到1之间且总和为1")
            index = int(np.argmax(values))
            disease, confidence = self.class_names[index], float(values[index])
            if not valid_prediction(disease, confidence):
                raise ClassificationError("类别或置信度无效")
            return ClassificationResult(True, disease, confidence,
                {self.class_names[i]: float(value) for i, value in enumerate(values)},
                dict(self.model_info))
        except Exception as error:
            return ClassificationResult(False, error=str(error), model_info=dict(self.model_info))

    def classify(self, image):
        try:
            results = self.predict(image)
            if results is None or len(results) != 1:
                raise ClassificationError("单图分类没有返回唯一结果")
            result = self.parse_result(results[0])
            if result.success:
                result.raw_results = results
            self.last_error = result.error
            return result
        except Exception as error:
            self.last_error = str(error)
            return ClassificationResult(False, error=str(error), model_info=dict(self.model_info))

    def classify_files(self, image_paths):
        """Keep input/output correspondence and continue past unreadable batch files."""
        import cv2
        for image_path in image_paths:
            try:
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                result = self.classify(image)
            except (OSError, ValueError, cv2.error) as error:
                result = ClassificationResult(False, error=f"图像读取失败: {error}",
                                              model_info=dict(self.model_info))
            yield str(image_path), result
