"""PC runtime configuration; all relative paths resolve against the project root."""
from dataclasses import dataclass
from pathlib import Path
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    framework_path: Path
    history_db: Path
    image_dir: Path
    device: str = "auto"
    image_size: int = 224
    api_endpoint: str = "https://api.deepseek.com/v1/chat/completions"
    api_model: str = "deepseek-chat"
    api_connect_timeout: float = 5
    api_read_timeout: float = 30
    api_attempts: int = 2
    chat_turns: int = 6
    auto_load_model: bool = True
    enable_voice: bool = False
    enable_board: bool = False


def load_app_config(path=None):
    path = project_path(path or os.getenv("EYE_APP_CONFIG", "configs/app_config.json"))
    source = path if path.exists() or path.name != "app_config.json" else PROJECT_ROOT / "configs/app_config.example.json"
    values = json.loads(source.read_text(encoding="utf-8-sig"))
    return config_from_values(values)


def config_from_values(values, use_environment=True):
    def setting(name, default):
        return os.getenv(name, default) if use_environment else default
    config = AppConfig(
        model_path=project_path(setting("EYE_MODEL_PATH", values["model_path"])),
        framework_path=project_path(setting("EYE_FRAMEWORK_PATH", values["framework_path"])),
        history_db=project_path(setting("EYE_HISTORY_DB", values["history_db"])),
        image_dir=project_path(values["image_dir"]),
        **{key: value for key, value in values.items()
           if key not in {"model_path", "framework_path", "history_db", "image_dir", "device"}},
        device=setting("EYE_DEVICE", values.get("device", "auto")),
    )
    if config.image_size <= 0 or config.api_attempts < 1 or config.chat_turns < 1:
        raise ValueError("输入尺寸、API尝试次数和会话轮数必须大于0")
    if min(config.api_connect_timeout, config.api_read_timeout) <= 0:
        raise ValueError("API超时必须大于0")
    return config


def save_app_config(updates, path=None):
    target = project_path(path or os.getenv("EYE_APP_CONFIG", "configs/app_config.json"))
    source = target if target.exists() else PROJECT_ROOT / "configs/app_config.example.json"
    values = json.loads(source.read_text(encoding="utf-8-sig"))
    values.update(updates)
    config = config_from_values(values, use_environment=False)
    target.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    return config
