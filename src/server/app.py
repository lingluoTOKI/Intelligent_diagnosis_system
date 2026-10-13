"""眼科图像分类 HTTP 服务。

服务端不加载 PyTorch、Ultralytics 与 PyQt5：
  - 分类走 ONNX Runtime（src/server/onnx_classifier.py）
  - 历史与辅助问答复用桌面端的 src/core/history.py 与 src/core/llm.py

启动：
    python -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000

环境变量：
    EYE_SERVER_WEIGHTS   ONNX 权重路径，默认 models/custom/AKConv_best_moudle/best.onnx
    EYE_HISTORY_DB       历史库路径，默认 ~/EyeDiseaseDetectorHistory/server_history.db
    EYE_MAX_UPLOAD_MB    单张上传上限，默认 12
    DEEPSEEK_API_KEY     可选，未配置时问答接口返回本地静态参考
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List  # noqa: E402

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.core.history import HistoryDB  # noqa: E402
from src.core.llm import DeepSeekAPI  # noqa: E402
from src.core.reference_advice import get_reference_advice  # noqa: E402
from src.server.onnx_classifier import DISEASES, OnnxClassifier  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
# 桌面端权重自带 Resize(512)+CenterCrop(512)，推理实际按 512 进行，
# config.image_size=224 不会被 ultralytics 采用，服务端必须沿用 512 权重。
WEIGHT_CANDIDATES = (
    PROJECT_ROOT / "models" / "server" / "best512.onnx",
    PROJECT_ROOT / "models" / "custom" / "AKConv_best_moudle" / "best.onnx",
)


def resolve_weights() -> Path:
    """优先环境变量，其次部署副本，最后回退到仓库内的原始权重。"""
    configured = os.getenv("EYE_SERVER_WEIGHTS", "").strip()
    if configured:
        return Path(configured)
    for candidate in WEIGHT_CANDIDATES:
        if candidate.is_file():
            return candidate
    return WEIGHT_CANDIDATES[0]
MAX_UPLOAD_BYTES = int(float(os.getenv("EYE_MAX_UPLOAD_MB", "12")) * 1024 * 1024)
MAX_BATCH_FILES = int(os.getenv("EYE_MAX_BATCH_FILES", "100"))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/jpg"}


class ChatRequest(BaseModel):
    question: str = Field(default="", min_length=0, max_length=2000)
    disease: str = ""
    confidence: float = 0.0


class ServerState:
    """持有分类器、历史库与问答客户端，启动失败时保持明确的错误状态。"""

    def __init__(self):
        self.classifier = None
        self.model_error = ""
        self.history = None
        self.history_error = ""
        self.llm = None

    def startup(self):
        weights = resolve_weights()
        try:
            self.classifier = OnnxClassifier(weights)
        except Exception as error:  # noqa: BLE001 - 启动失败要保留原因而不是崩溃
            self.model_error = str(error)
            self.classifier = None

        db_path = os.getenv("EYE_HISTORY_DB") or str(
            Path.home() / "EyeDiseaseDetectorHistory" / "server_history.db")
        try:
            self.history = HistoryDB(db_path)
        except Exception as error:  # noqa: BLE001
            self.history_error = str(error)
            self.history = None

        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.llm = DeepSeekAPI(api_key=api_key) if api_key else None


state = ServerState()


def require_model():
    if state.classifier is None:
        raise HTTPException(status_code=503, detail=state.model_error or "分类模型未就绪")


app = FastAPI(title="眼科图像分类辅助诊断系统 · 服务端",
              description="Course project deployment endpoint. Not for clinical use.",
              version="1.0.0")


@app.on_event("startup")
def on_startup():
    state.startup()


@app.get("/api/health")
def health():
    return {
        "status": "ok" if state.classifier is not None else "degraded",
        "model": state.classifier.model_info() if state.classifier else None,
        "model_error": state.model_error or None,
        "history_ready": state.history is not None,
        "history_error": state.history_error or None,
        "history_db": str(state.history.db_path) if state.history else None,
        "llm_enabled": state.llm is not None,
        "classes": list(DISEASES),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    require_model()
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        return JSONResponse({"success": False, "error": f"不支持的文件类型: {file.content_type}"},
                            status_code=415)
    data = await file.read()
    if not data:
        return JSONResponse({"success": False, "error": "上传内容为空"}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"success": False,
             "error": f"图片超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限"}, status_code=413)

    started = time.perf_counter()
    prediction = state.classifier.predict_bytes(data)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    if not prediction.success:
        # 失败如实返回，不生成占位结果，也不写入历史
        return JSONResponse({"success": False, "error": prediction.error, "latency_ms": latency_ms},
                            status_code=422)

    record_id = uuid.uuid4().hex
    saved = False
    if state.history is not None:
        try:
            state.history.add(record_id, datetime.now().isoformat(timespec="seconds"),
                              file.filename or "upload", prediction.disease,
                              prediction.confidence, prediction.probabilities,
                              state.classifier.model_info())
            saved = True
        except Exception as error:  # noqa: BLE001 - 历史写入失败不应让诊断失败
            saved = False
            print(f"[警告] 历史写入失败: {error}")

    return {
        "success": True,
        "record_id": record_id,
        "disease": prediction.disease,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "latency_ms": latency_ms,
        "history_saved": saved,
        "model": state.classifier.model_info(),
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    disease = request.disease.strip()
    if disease and disease in DISEASES and state.llm is not None:
        advice = state.llm.request_treatment_advice(disease, request.confidence)
        return {"success": advice.success, "content": advice.content,
                "source": advice.source, "error": advice.error}

    question = request.question.strip()
    if question and state.llm is not None:
        advice = state.llm.request_custom_advice(question)
        return {"success": advice.success, "content": advice.content,
                "source": advice.source, "error": advice.error}

    # 未配置 API 密钥时只给本地静态参考，并明确标注来源
    if disease in DISEASES:
        return {"success": True, "content": get_reference_advice(disease),
                "source": "local-reference",
                "error": "未配置 DEEPSEEK_API_KEY，返回本地静态参考"}
    if question:
        return {"success": False, "content": "", "source": "unavailable",
                "error": "未配置 DEEPSEEK_API_KEY，在线问答不可用"}
    return {"success": False, "content": "", "source": "unavailable",
            "error": "缺少疾病名称或提问内容"}


def _classify_one(data: bytes, name: str) -> dict:
    """对单张图像执行推理并写库，失败项如实标注，不生成占位结果。"""
    item = {"image_path": name, "status": "pending", "save_status": "pending", "error": ""}
    try:
        if not data:
            raise ValueError("上传内容为空")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限")
        prediction = state.classifier.predict_bytes(data)
        if not prediction.success:
            raise ValueError(prediction.error)
        item.update(disease_name=prediction.disease, confidence=prediction.confidence,
                    probabilities=prediction.probabilities, status="success")
    except Exception as error:  # noqa: BLE001 - 单张失败不应中断整批
        item["status"] = "failed"
        item["save_status"] = "failed"
        item["error"] = str(error)
        return item

    saved = False
    if state.history is not None:
        try:
            record_id = uuid.uuid4().hex
            state.history.add(record_id, datetime.now().isoformat(timespec="seconds"), name,
                              prediction.disease, prediction.confidence, prediction.probabilities,
                              state.classifier.model_info())
            item["record_id"] = record_id
            saved = True
        except Exception as error:  # noqa: BLE001
            item["error"] = str(error)
    item["save_status"] = "saved" if saved else "failed"
    return item


@app.post("/api/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """批量分类：逐张推理并入库，返回每条的状态，失败项不写历史。"""
    require_model()
    if len(files) > MAX_BATCH_FILES:
        return JSONResponse({"success": False, "error": f"一次最多提交 {MAX_BATCH_FILES} 张"},
                            status_code=413)
    started = time.perf_counter()
    results = []
    for upload in files:
        if upload.content_type and upload.content_type not in ALLOWED_CONTENT_TYPES:
            results.append({"image_path": upload.filename or "upload", "status": "failed",
                            "save_status": "failed", "error": f"不支持的文件类型: {upload.content_type}"})
            continue
        results.append(_classify_one(await upload.read(), upload.filename or "upload"))
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    done = sum(1 for item in results if item["status"] == "success")
    return {"success": True, "count": len(results), "done": done,
            "failed": len(results) - done, "results": results, "latency_ms": latency_ms}


@app.get("/api/history")
def history(limit: int = 100, page: int = 0, start: str = "", end: str = "",
            disease: str = "", search: str = "", batch: str = ""):
    """历史查询：支持日期区间、类别、批次与文件名搜索，并附带图表统计。"""
    if state.history is None:
        return JSONResponse({"success": False, "error": state.history_error or "历史库不可用"},
                            status_code=503)
    page_size = max(1, min(int(limit), 200))
    filters = {"search": search.strip(), "disease": disease.strip(),
               "batch": batch.strip(), "real_only": True}
    for key, value in (("start", start), ("end", end)):
        value = value.strip()
        if not value:
            continue
        try:
            day = datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return JSONResponse({"success": False, "error": f"{key} 需为 YYYY-MM-DD 格式"},
                                status_code=400)
        # 结束日期为闭区间，内部按半开区间过滤
        if key == "end":
            day = day + timedelta(days=1)
        filters[key] = day.strftime("%Y-%m-%d")
    if filters.get("start") and filters.get("end") and filters["start"] > filters["end"]:
        return JSONResponse({"success": False, "error": "开始日期不能晚于结束日期"}, status_code=400)

    rows, total, page_index, counts, daily = state.history.filtered(
        filters, page=max(0, int(page)), page_size=page_size)
    return {"success": True, "rows": rows, "records": rows, "count": len(rows), "total": total,
            "page": page_index, "page_size": page_size, "counts": counts, "daily": daily}


@app.delete("/api/history/{record_id}")
def delete_history(record_id: str):
    if state.history is None:
        return JSONResponse({"success": False, "error": state.history_error or "历史库不可用"},
                            status_code=503)
    state.history.delete_by_record_id(record_id)
    return {"success": True, "record_id": record_id}


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
