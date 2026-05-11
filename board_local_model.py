#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发板本地AI模型部署方案
支持在开发板上直接运行轻量级眼科疾病检测模型
"""

import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
import threading

# 尝试导入不同的推理框架
try:
    import onnxruntime as ort
    HAS_ONNX = True
    print("✅ ONNX Runtime 可用")
except ImportError:
    HAS_ONNX = False
    print("❌ ONNX Runtime 不可用")

try:
    import tensorflow as tf
    HAS_TF = True
    print("✅ TensorFlow 可用")
except ImportError:
    HAS_TF = False
    print("❌ TensorFlow 不可用")

try:
    import torch
    HAS_TORCH = True
    print("✅ PyTorch 可用")
except ImportError:
    HAS_TORCH = False
    print("❌ PyTorch 不可用")

class LocalEyeDiseaseModel:
    """本地眼科疾病检测模型"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.is_loaded = False
        self.input_size = (224, 224)  # 默认输入尺寸
        
        # 疾病类别映射（简化版）
        self.disease_classes = {
            0: "正常",
            1: "白内障",
            2: "青光眼", 
            3: "糖尿病视网膜病变",
            4: "年龄相关性黄斑变性",
            5: "其他异常"
        }
        
        # 治疗建议映射
        self.treatment_advice = {
            "正常": "眼部健康状况良好，建议定期检查。",
            "白内障": "建议及时就医，可能需要手术治疗。避免强光刺激。",
            "青光眼": "紧急情况！请立即就医。青光眼可能导致失明。",
            "糖尿病视网膜病变": "请控制血糖，定期眼底检查，必要时激光治疗。",
            "年龄相关性黄斑变性": "建议补充叶黄素，避免强光，定期复查。",
            "其他异常": "检测到异常，建议专业医生进一步检查。"
        }
    
    def load_model(self, model_path=None):
        """加载模型"""
        print("🔍 搜索可用的模型文件...")
        
        # 搜索模型文件
        model_files = self._find_model_files()
        
        if model_path and os.path.exists(model_path):
            selected_model = model_path
        elif model_files:
            selected_model = model_files[0]  # 使用第一个找到的模型
            print(f"📁 找到模型: {selected_model}")
        else:
            print("❌ 未找到模型文件，使用模拟模型")
            return self._load_mock_model()
        
        # 根据文件扩展名选择加载方式
        if selected_model.endswith('.onnx') and HAS_ONNX:
            return self._load_onnx_model(selected_model)
        elif selected_model.endswith('.tflite') and HAS_TF:
            return self._load_tflite_model(selected_model)
        elif selected_model.endswith('.pt') and HAS_TORCH:
            return self._load_pytorch_model(selected_model)
        else:
            print("⚠️ 不支持的模型格式或缺少运行环境，使用模拟模型")
            return self._load_mock_model()
    
    def _find_model_files(self):
        """查找模型文件"""
        model_extensions = ['.onnx', '.tflite', '.pt', '.pb']
        model_files = []
        
        # 搜索当前目录和子目录
        for root, dirs, files in os.walk('.'):
            for file in files:
                if any(file.endswith(ext) for ext in model_extensions):
                    model_files.append(os.path.join(root, file))
        
        return model_files
    
    def _load_onnx_model(self, model_path):
        """加载ONNX模型"""
        try:
            print(f"🔄 加载ONNX模型: {model_path}")
            self.model = ort.InferenceSession(model_path)
            self.model_type = "onnx"
            self.is_loaded = True
            
            # 获取输入形状
            input_shape = self.model.get_inputs()[0].shape
            if len(input_shape) >= 3:
                self.input_size = (input_shape[-2], input_shape[-1])
            
            print(f"✅ ONNX模型加载成功，输入尺寸: {self.input_size}")
            return True
            
        except Exception as e:
            print(f"❌ ONNX模型加载失败: {e}")
            return False
    
    def _load_tflite_model(self, model_path):
        """加载TensorFlow Lite模型"""
        try:
            print(f"🔄 加载TFLite模型: {model_path}")
            self.model = tf.lite.Interpreter(model_path=model_path)
            self.model.allocate_tensors()
            self.model_type = "tflite"
            self.is_loaded = True
            
            # 获取输入形状
            input_details = self.model.get_input_details()
            input_shape = input_details[0]['shape']
            if len(input_shape) >= 3:
                self.input_size = (input_shape[1], input_shape[2])
            
            print(f"✅ TFLite模型加载成功，输入尺寸: {self.input_size}")
            return True
            
        except Exception as e:
            print(f"❌ TFLite模型加载失败: {e}")
            return False
    
    def _load_pytorch_model(self, model_path):
        """加载PyTorch模型"""
        try:
            print(f"🔄 加载PyTorch模型: {model_path}")
            self.model = torch.jit.load(model_path, map_location='cpu')
            self.model.eval()
            self.model_type = "pytorch"
            self.is_loaded = True
            
            print(f"✅ PyTorch模型加载成功，输入尺寸: {self.input_size}")
            return True
            
        except Exception as e:
            print(f"❌ PyTorch模型加载失败: {e}")
            return False
    
    def _load_mock_model(self):
        """加载模拟模型（用于演示）"""
        print("🎭 加载模拟模型（仅用于演示）")
        self.model_type = "mock"
        self.is_loaded = True
        return True
    
    def preprocess_image(self, image):
        """图像预处理"""
        try:
            # 调整尺寸
            if image.shape[:2] != self.input_size:
                image = cv2.resize(image, self.input_size)
            
            # 归一化
            image = image.astype(np.float32) / 255.0
            
            # 根据模型类型调整维度
            if self.model_type == "onnx":
                # ONNX: (1, 3, H, W)
                image = np.transpose(image, (2, 0, 1))
                image = np.expand_dims(image, axis=0)
            elif self.model_type == "tflite":
                # TFLite: (1, H, W, 3)
                image = np.expand_dims(image, axis=0)
            elif self.model_type == "pytorch":
                # PyTorch: (1, 3, H, W)
                image = np.transpose(image, (2, 0, 1))
                image = torch.FloatTensor(image).unsqueeze(0)
            else:
                # Mock模型
                image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"❌ 图像预处理失败: {e}")
            return None
    
    def predict(self, image):
        """执行预测"""
        if not self.is_loaded:
            return self._mock_prediction()
        
        try:
            # 预处理
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return self._mock_prediction()
            
            # 推理
            start_time = time.time()
            
            if self.model_type == "onnx":
                input_name = self.model.get_inputs()[0].name
                outputs = self.model.run(None, {input_name: processed_image})
                predictions = outputs[0]
                
            elif self.model_type == "tflite":
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()
                
                self.model.set_tensor(input_details[0]['index'], processed_image)
                self.model.invoke()
                predictions = self.model.get_tensor(output_details[0]['index'])
                
            elif self.model_type == "pytorch":
                with torch.no_grad():
                    predictions = self.model(processed_image)
                    predictions = predictions.numpy()
            else:
                return self._mock_prediction()
            
            inference_time = time.time() - start_time
            
            # 解析结果
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            disease_name = self.disease_classes.get(predicted_class, "未知疾病")
            advice = self.treatment_advice.get(disease_name, "建议咨询专业医生")
            
            result = {
                "disease_name": disease_name,
                "confidence": confidence,
                "advice": advice,
                "inference_time": f"{inference_time:.3f}s",
                "model_type": self.model_type,
                "timestamp": datetime.now().isoformat(),
                "emergency": disease_name == "青光眼" and confidence > 0.7
            }
            
            print(f"🔍 本地诊断结果: {disease_name} (置信度: {confidence:.2%})")
            return result
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return self._mock_prediction()
    
    def _mock_prediction(self):
        """模拟预测（用于演示）"""
        import random
        
        # 模拟结果
        diseases = list(self.disease_classes.values())
        disease_name = random.choice(diseases)
        confidence = random.uniform(0.6, 0.95)
        
        result = {
            "disease_name": disease_name,
            "confidence": confidence,
            "advice": self.treatment_advice.get(disease_name, "建议咨询专业医生"),
            "inference_time": "0.050s",
            "model_type": "mock",
            "timestamp": datetime.now().isoformat(),
            "emergency": disease_name == "青光眼" and confidence > 0.7
        }
        
        print(f"🎭 模拟诊断结果: {disease_name} (置信度: {confidence:.2%})")
        return result
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            "loaded": self.is_loaded,
            "model_type": self.model_type,
            "input_size": self.input_size,
            "classes": list(self.disease_classes.values()),
            "runtime_available": {
                "onnx": HAS_ONNX,
                "tensorflow": HAS_TF,
                "pytorch": HAS_TORCH
            }
        }

class LocalMedicalSystem:
    """本地医疗诊断系统"""
    
    def __init__(self):
        self.model = LocalEyeDiseaseModel()
        self.is_running = False
        
    def initialize(self):
        """初始化系统"""
        print("🏥 初始化本地医疗诊断系统...")
        
        # 加载模型
        if self.model.load_model():
            print("✅ 本地AI模型已就绪")
            return True
        else:
            print("❌ 模型加载失败")
            return False
    
    def diagnose_image(self, image_path_or_array):
        """诊断图像"""
        try:
            # 加载图像
            if isinstance(image_path_or_array, str):
                image = cv2.imread(image_path_or_array)
                if image is None:
                    raise ValueError(f"无法加载图像: {image_path_or_array}")
            else:
                image = image_path_or_array
            
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 执行预测
            result = self.model.predict(image)
            
            # 添加本地标识
            result["source"] = "local_board"
            result["requires_network"] = False
            
            return result
            
        except Exception as e:
            print(f"❌ 诊断失败: {e}")
            return {
                "error": str(e),
                "source": "local_board",
                "timestamp": datetime.now().isoformat()
            }
    
    def run_demo(self):
        """运行演示"""
        print("\n🎯 本地模型演示模式")
        print("=" * 50)
        
        # 显示模型信息
        info = self.model.get_model_info()
        print(f"模型状态: {'✅ 已加载' if info['loaded'] else '❌ 未加载'}")
        print(f"模型类型: {info['model_type']}")
        print(f"输入尺寸: {info['input_size']}")
        print(f"支持疾病: {', '.join(info['classes'])}")
        
        print("\n🔧 运行环境:")
        for runtime, available in info['runtime_available'].items():
            status = "✅" if available else "❌"
            print(f"  {runtime}: {status}")
        
        # 如果有摄像头，使用实时预测
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("\n📷 开始实时诊断 (按 'q' 退出, 空格键拍照诊断)")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    cv2.imshow('Local Medical AI - Press SPACE to diagnose', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        print("\n🔍 执行诊断...")
                        result = self.diagnose_image(frame)
                        self._display_result(result)
                
                cap.release()
                cv2.destroyAllWindows()
            else:
                print("❌ 无法打开摄像头，使用模拟图像")
                self._demo_with_mock_image()
                
        except Exception as e:
            print(f"⚠️ 摄像头演示失败: {e}")
            self._demo_with_mock_image()
    
    def _demo_with_mock_image(self):
        """使用模拟图像演示"""
        print("\n🎭 使用模拟图像进行演示...")
        
        # 创建模拟图像
        mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 执行诊断
        result = self.diagnose_image(mock_image)
        self._display_result(result)
    
    def _display_result(self, result):
        """显示诊断结果"""
        print("\n" + "="*50)
        print("🏥 本地AI诊断结果")
        print("="*50)
        
        if 'error' in result:
            print(f"❌ 诊断失败: {result['error']}")
            return
        
        print(f"🔍 疾病名称: {result['disease_name']}")
        print(f"📊 置信度: {result['confidence']:.2%}")
        print(f"⏱️ 推理时间: {result['inference_time']}")
        print(f"🤖 模型类型: {result['model_type']}")
        
        if result.get('emergency'):
            print("🚨 紧急情况！建议立即就医！")
        
        print(f"\n💡 建议: {result['advice']}")
        print("="*50)

def main():
    """主函数"""
    print("🏥 开发板本地AI医疗诊断系统")
    print("支持ONNX、TensorFlow Lite、PyTorch模型")
    
    system = LocalMedicalSystem()
    
    if system.initialize():
        system.run_demo()
    else:
        print("❌ 系统初始化失败")

if __name__ == "__main__":
    main()
