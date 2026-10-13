"""Persist image picker locations separately from captured-image paths."""
import json
from pathlib import Path

def read(window):
    try:return json.loads(window._ui_preferences_path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}

def save(window,**values):
    settings=read(window);settings.update(values)
    path=window._ui_preferences_path;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')

def initial(window):
    settings=read(window)
    for key in ('image_directory','last_image_directory'):
        value=settings.get(key,'')
        if value and Path(value).is_dir():return value
    return str(Path(__file__).resolve().parents[2]/'data')

def remember(window,path):
    try:save(window,last_image_directory=str(Path(path).resolve().parent))
    except OSError:window.status_bar.showMessage('本次已选图，但未能保存最近图像目录')
