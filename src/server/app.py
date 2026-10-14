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
    EYE_ACCESS_CODE      访问口令，留空则关闭口令校验（本机开发默认关闭，部署时建议设置）
    DEEPSEEK_API_KEY     可选，未配置时问答接口返回本地静态参考
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List  # noqa: E402

from fastapi import FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
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
# 2GB 内存服务器上 ONNX Runtime 峰值内存有限，单批上限默认收紧到 10 张，
# 防止超大批次把 worker 拖到 OOM（worker 一旦被 kill，nginx 会对所有请求返回 502）
MAX_BATCH_FILES = int(os.getenv("EYE_MAX_BATCH_FILES", "10"))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/jpg"}

# 历史回看原图：分类时把上传图像另存到 runtime/uploads，历史详情可直接显示原图。
# 注意：眼底图像属于医疗数据，若不想在服务器留存，设置 EYE_SAVE_UPLOADS=0 关闭。
SAVE_UPLOADS = os.getenv("EYE_SAVE_UPLOADS", "1").strip().lower() not in {"0", "false", "no", "off"}
UPLOAD_DIR = Path(os.getenv("EYE_UPLOAD_DIR", PROJECT_ROOT / "runtime" / "uploads"))
if SAVE_UPLOADS:
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as error:  # noqa: BLE001 - 目录创建失败不应阻断启动
        print(f"[警告] 上传目录不可用，将不保存原图: {error}")
        SAVE_UPLOADS = False


def _save_upload(data: bytes, record_id: str, name: str) -> str:
    """把上传图像另存到 UPLOAD_DIR，返回落盘文件名；失败返回空串，不影响分类结果。"""
    if not SAVE_UPLOADS or not data:
        return ""
    try:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "upload").name)[:60] or "upload"
        filename = record_id + "_" + safe
        (UPLOAD_DIR / filename).write_bytes(data)
        return filename
    except Exception as error:  # noqa: BLE001 - 另存失败只影响回看，不影响诊断
        print(f"[警告] 图像另存失败: {error}")
        return ""


class ChatRequest(BaseModel):
    question: str = Field(default="", min_length=0, max_length=2000)
    disease: str = ""
    confidence: float = 0.0


class KeyRequest(BaseModel):
    """运行时下发到服务端的 DeepSeek 密钥；为空字符串表示清除。"""
    key: str = Field(default="", max_length=200)


class LoginRequest(BaseModel):
    code: str = Field(default="", min_length=0, max_length=128)


class TTSRequest(BaseModel):
    """自然语音合成请求，语速与前端阅读设置保持一致。"""
    text: str = Field(min_length=1, max_length=5000)
    lang: str = Field(default="zh-CN", max_length=16)
    rate: float = Field(default=1.0, ge=0.7, le=1.3)


# ---------------- 访问口令（可选，留空即关闭） ----------------
ACCESS_CODE = os.getenv("EYE_ACCESS_CODE", "").strip()
COOKIE_NAME = "eye_access"
COOKIE_MAX_AGE = 7 * 24 * 3600
LOGIN_PAGE = STATIC_DIR / "login.html"
# 简单的失败限流：同一来源 60 秒内连续错 10 次即冷却，防暴力试口令
_FAILED_ATTEMPTS: dict[str, list] = {}
_MAX_FAILS = 10
_FAIL_WINDOW = 60.0


def access_token(code: str) -> str:
    """由口令派生 Cookie 值，Cookie 中不保存明文口令。"""
    return hmac.new(code.encode("utf-8"), b"eye-diagnosis-access-v1", hashlib.sha256).hexdigest()


def is_authorized(request: Request) -> bool:
    if not ACCESS_CODE:
        return True
    token = request.cookies.get(COOKIE_NAME, "")
    return hmac.compare_digest(token, access_token(ACCESS_CODE))


def client_key(request: Request) -> str:
    return (request.client.host if request.client else "unknown")


def throttled(key: str) -> bool:
    now = time.time()
    hits = [t for t in _FAILED_ATTEMPTS.get(key, []) if now - t < _FAIL_WINDOW]
    _FAILED_ATTEMPTS[key] = hits
    return len(hits) >= _MAX_FAILS


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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """兜底：任何未捕获异常都返回 JSON 500，避免前端拿到空响应/纯文本报
    SyntaxError: Unexpected end of JSON input。"""
    print(f"[错误] 未处理异常 {request.url.path}: {exc}")
    return JSONResponse({"success": False, "error": f"服务端内部错误: {exc}"},
                        status_code=500)


@app.middleware("http")
async def access_gate(request: Request, call_next):
    """口令门：未设置 EYE_ACCESS_CODE 时完全放行，保持本机开发体验不变。"""
    path = request.url.path
    if is_authorized(request) or path in ("/api/login", "/api/logout"):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"success": False, "error": "未授权，请先输入访问口令"}, status_code=401)
    if LOGIN_PAGE.is_file():
        return HTMLResponse(LOGIN_PAGE.read_text(encoding="utf-8"), status_code=200)
    return JSONResponse({"error": "未授权"}, status_code=401)


@app.post("/api/login")
def login(payload: LoginRequest, request: Request):
    if not ACCESS_CODE:
        return {"success": True, "required": False}
    key = client_key(request)
    if throttled(key):
        return JSONResponse({"success": False, "error": "尝试过于频繁，请稍后再试"}, status_code=429)
    if not hmac.compare_digest(payload.code, ACCESS_CODE):
        _FAILED_ATTEMPTS.setdefault(key, []).append(time.time())
        return JSONResponse({"success": False, "error": "口令不正确"}, status_code=401)
    _FAILED_ATTEMPTS.pop(key, None)
    response = JSONResponse({"success": True})
    response.set_cookie(COOKIE_NAME, access_token(ACCESS_CODE), max_age=COOKIE_MAX_AGE,
                        httponly=True, samesite="lax", path="/")
    return response


@app.post("/api/logout")
def logout():
    response = JSONResponse({"success": True})
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


@app.get("/api/access")
def access_state():
    """前端用于判断是否需要口令；不泄露口令本身。"""
    return {"required": bool(ACCESS_CODE)}


@app.get("/api/health")
def health():
    return {
        "status": "ok" if state.classifier is not None else "degraded",
        "model": state.classifier.model_info() if state.classifier else None,
        "model_error": state.model_error or None,
        "history_ready": state.history is not None,
        "history_error": state.history_error or None,
        "history_db": str(state.history.db_path) if state.history else None,
        "llm_enabled": bool(state.llm is not None and state.llm.api_key.strip()),
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
                              state.classifier.model_info(),
                              image_file=_save_upload(data, record_id, file.filename or "upload"))
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
    question = request.question.strip()

    # 同时携带分类结果与追问时，把分类结果作为上下文拼进多轮问答，
    # 让 AI 知道"当前图片被识别为什么病"，避免给出通用回复。
    if disease and disease in DISEASES and question and state.llm is not None:
        prompt = (
            f"【当前图像分类结果】模型预测为：{disease}"
            f"（置信度 {float(request.confidence) * 100:.1f}%）。"
            f"该结果仅供参考，不能替代医生诊断。在此前提下，请回答我的问题：{question}"
        )
        advice = state.llm.request_custom_advice(prompt)
        return {"success": advice.success, "content": advice.content,
                "source": advice.source, "error": advice.error}

    if disease and disease in DISEASES and state.llm is not None:
        advice = state.llm.request_treatment_advice(disease, request.confidence)
        return {"success": advice.success, "content": advice.content,
                "source": advice.source, "error": advice.error}

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


@app.post("/api/tts")
async def text_to_speech(payload: TTSRequest):
    """用 Edge Neural Voice 生成 MP3，并按文本、语言和语速缓存。"""
    try:
        import edge_tts
    except ImportError as error:
        raise HTTPException(status_code=503, detail="服务端未安装 edge-tts") from error

    clean = " ".join(payload.text.split()).strip()
    if not clean:
        raise HTTPException(status_code=400, detail="没有可朗读的文字")
    voice = "en-US-JennyNeural" if payload.lang.lower().startswith("en") \
        else "zh-CN-XiaoxiaoNeural"
    rate_percent = max(-30, min(30, round((payload.rate - 1.0) * 100)))
    cache_key = hashlib.sha256(
        f"{voice}|{rate_percent}|{clean}".encode("utf-8")
    ).hexdigest()
    cache_dir = Path(os.getenv("EYE_TTS_CACHE", PROJECT_ROOT / "runtime" / "tts_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{cache_key}.mp3"
    if not target.is_file() or target.stat().st_size == 0:
        temporary = cache_dir / f".{cache_key}.{uuid.uuid4().hex}.tmp.mp3"
        try:
            communicator = edge_tts.Communicate(
                clean, voice=voice, rate=f"{rate_percent:+d}%"
            )
            await communicator.save(str(temporary))
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("语音服务未返回有效音频")
            os.replace(temporary, target)
        except Exception as error:  # noqa: BLE001 - 供前端明确降级
            temporary.unlink(missing_ok=True)
            raise HTTPException(status_code=502, detail=f"语音生成失败: {error}") from error
    return FileResponse(target, media_type="audio/mpeg", filename="speech.mp3",
                        headers={"Cache-Control": "private, max-age=86400",
                                 "X-TTS-Voice": voice})


@app.post("/api/set-key")
def set_key(request: KeyRequest):
    """允许已登录用户将 DeepSeek 密钥下发到服务端运行内存。

    密钥不落服务器磁盘，只保存在调用方的本机浏览器（localStorage）；
    服务端重启后若未配置环境变量，需要浏览器重新提交一次。
    """
    key = (request.key or "").strip()
    if state.llm is None:
        state.llm = DeepSeekAPI(api_key=key)
    else:
        state.llm.set_api_key(key)
    enabled = bool(state.llm and state.llm.api_key.strip())
    masked = ("sk-****" + state.llm.api_key[-4:]) if enabled else ""
    return {"success": True, "llm_enabled": enabled, "masked": masked}


def new_batch_id() -> str:
    """可读批次号：B + 日期时间 + 4 位随机后缀，例如 B20261013-194930-7f3a。

    一次「开始分类」对应一个批次号，写入 records.batch_id 并登记到 batches 表，
    便于在历史记录页按批次回看整批结果。
    """
    return "B" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def _classify_one(data: bytes, name: str, batch_id: str = "") -> dict:
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
                              state.classifier.model_info(), batch_id=batch_id or None,
                              image_file=_save_upload(data, record_id, name))
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
    batch_id = new_batch_id()
    model_info = state.classifier.model_info()
    if state.history is not None:
        try:
            state.history.create_batch(batch_id, [(upload.filename or "upload") for upload in files],
                                       model_info)
        except Exception as error:  # noqa: BLE001 - 登记失败不应中断分类
            print(f"[警告] 批次登记失败: {error}")

    started = time.perf_counter()
    results = []
    for index, upload in enumerate(files):
        name = upload.filename or "upload"
        if upload.content_type and upload.content_type not in ALLOWED_CONTENT_TYPES:
            item = {"image_path": name, "status": "failed", "save_status": "failed",
                    "error": f"不支持的文件类型: {upload.content_type}"}
        else:
            item = _classify_one(await upload.read(), name, batch_id)
        item["batch_id"] = batch_id
        results.append(item)
        if state.history is not None:
            try:
                state.history.save_batch_item(batch_id, index, item)
            except Exception as error:  # noqa: BLE001
                print(f"[警告] 批次项写入失败: {error}")

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    done = sum(1 for item in results if item["status"] == "success")
    failed = len(results) - done
    if state.history is not None:
        try:
            state.history.finish_batch(batch_id, "done" if failed == 0 else "done_with_errors")
        except Exception as error:  # noqa: BLE001
            print(f"[警告] 批次收尾失败: {error}")
    return {"success": True, "batch_id": batch_id, "count": len(results), "done": done,
            "failed": failed, "results": results, "latency_ms": latency_ms}


@app.get("/api/batches")
def list_batches():
    """近期批次列表（含登记张数与入库张数），供前端下拉回看。"""
    if state.history is None:
        return JSONResponse({"success": False, "error": state.history_error or "历史库不可用"},
                            status_code=503)
    try:
        return {"success": True, "batches": state.history.batch_stats()}
    except Exception as error:  # noqa: BLE001
        return JSONResponse({"success": False, "error": str(error)}, status_code=500)


@app.get("/api/batches/{batch_id}")
def batch_detail(batch_id: str):
    """单个批次的逐项结果，含失败项的错误原因。"""
    if state.history is None:
        return JSONResponse({"success": False, "error": state.history_error or "历史库不可用"},
                            status_code=503)
    try:
        items = state.history.batch_items(batch_id)
    except Exception as error:  # noqa: BLE001
        return JSONResponse({"success": False, "error": str(error)}, status_code=500)
    return {"success": True, "batch_id": batch_id, "count": len(items), "items": items}


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


@app.get("/api/history/{record_id}/image")
def history_image(record_id: str):
    """历史原图回看：返回该记录分类时另存的原始图像。

    老记录（开启此功能之前产生的）没有留图，返回 404，前端会自动隐藏图片区域。
    """
    if state.history is None:
        return JSONResponse({"success": False, "error": state.history_error or "历史库不可用"},
                            status_code=503)
    row = state.history.get_by_record_id(record_id)
    if not row:
        return JSONResponse({"success": False, "error": "记录不存在"}, status_code=404)
    stored = (row.get("image_file") or "").strip()
    if not stored:
        return JSONResponse({"success": False, "error": "该记录未保存原图"}, status_code=404)
    # 只取文件名，杜绝路径穿越
    filename = Path(stored).name
    target = UPLOAD_DIR / filename
    if not target.is_file():
        return JSONResponse({"success": False, "error": "原图文件已不存在"}, status_code=404)
    return FileResponse(str(target))


@app.delete("/api/history/{record_id}")
def delete_history(record_id: str):
    if state.history is None:
        return JSONResponse({"success": False, "error": state.history_error or "历史库不可用"},
                            status_code=503)
    row = state.history.get_by_record_id(record_id)
    state.history.delete_by_record_id(record_id)
    # 连带清理已留存的原图，避免孤儿文件持续占用磁盘
    stored = (row or {}).get("image_file") or ""
    if stored:
        try:
            target = UPLOAD_DIR / Path(stored).name
            if target.is_file():
                target.unlink()
        except Exception as error:  # noqa: BLE001 - 清理失败不影响记录删除结果
            print(f"[警告] 原图清理失败: {error}")
    return {"success": True, "record_id": record_id}


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
