from ultralytics import YOLO
import os

# 检查 YAML 文件是否存在
yaml_path = "C:/Users/47449/Desktop/yolo/yolov11/ultralytics-main/eyes_disease.yaml"
if not os.path.exists(yaml_path):
    print(f"错误：文件 {yaml_path} 不存在！")
else:
    print(f"文件 {yaml_path} 存在，继续训练...")

    # 检查数据集目录是否存在
    dataset_path = "C:/Users/47449/Desktop/yolo/yolov11/ultralytics-main/eyes_split"
    if not os.path.exists(dataset_path):
        print(f"错误：数据集目录 {dataset_path} 不存在！")
    else:
        print(f"数据集目录 {dataset_path} 存在，继续训练...")

        # 加载模型
        model = YOLO("yolo11n-cls.pt")   # 使用 YOLOv11 分类模型

        # 配置训练参数
        results = model.train(
            data=r"C:/Users/47449/Desktop/yolo/yolov11/ultralytics-main/eyes_disease.yaml",
            epochs=200,
            batch=32,      # 增大批次大小，充分利用 GPU 内存
            imgsz=512,     # 适当缩小图像尺寸，减少计算量
            lr0=0.001,
            lrf=0.01,
            cos_lr=True,   # 使用余弦退火学习率调度
            workers=16,    # 增加数据加载线程数，提高数据加载效率
            device=0,      # 指定使用的 GPU 设备
            verbose=True,
            amp=True       # 启用混合精度训练
        )

        # 验证最佳模型
        model.val()

        # 导出模型（可选）
        model.export(format="onnx")