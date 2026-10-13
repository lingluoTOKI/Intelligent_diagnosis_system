import os
from pathlib import Path
ROOT = Path(r'C:/Users/47449/Desktop/yolo/Intelligent_diagnosis_system')
os.environ['YOLO_CONFIG_DIR'] = r'C:/Users/47449/Desktop/yolo/experiment_config'
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r'C:/Users/47449/Desktop/yolo/yolo26/ultralytics-main/ultralytics/cfg/models/26/yolo26-cls.yaml', task='classify')
    model.train(data=str(ROOT / 'data'), epochs=50, imgsz=224, batch=32,
                device=0, workers=8, seed=0, deterministic=True,
                optimizer='AdamW', lr0=0.000833, momentum=0.9,
                weight_decay=0.0005, amp=True, half=False, pretrained=False,
                project=str(ROOT / 'runs' / 'comparison'), name='yolo26n_scratch_224_50',
                exist_ok=False, plots=True)
