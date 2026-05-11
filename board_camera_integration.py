#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发板摄像头与PC端医疗诊断系统集成模块
功能：
1. 从开发板摄像头捕获图像
2. 将图像传输到PC端进行AI诊断
3. 接收诊断结果并在开发板显示
4. 支持图像保存和历史记录
"""

import socket
import cv2
import numpy as np
import time
import json
import os
import threading
from datetime import datetime
import base64

# ===== 配置参数 =====
PC_IP = "172.20.10.3"  # PC端IP地址
CAMERA_PORT = 5002      # 摄像头数据传输端口
DIAGNOSIS_PORT = 5003   # 诊断结果接收端口
COMMAND_PORT = 5004     # 命令控制端口

# 摄像头设置
CAMERA_WIDTH = 640      # 摄像头分辨率宽度
CAMERA_HEIGHT = 480     # 摄像头分辨率高度
CAMERA_FPS = 30         # 摄像头帧率
JPEG_QUALITY = 85       # JPEG压缩质量

# 图像保存设置
SAVE_DIR = "/home/pi/medical_images"  # 图像保存目录
MAX_PACKET_SIZE = 1400  # 最大数据包大小
# ===================

class CameraManager:
    """摄像头管理器"""
    
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.capture_thread = None
        self.frame_counter = 0
        self.last_frame = None
        self.preview_window = None
        self.is_previewing = False
        
        # 创建保存目录
        os.makedirs(SAVE_DIR, exist_ok=True)
        
    def init_camera(self):
        """初始化摄像头"""
        try:
            self.camera = cv2.VideoCapture(0)  # 使用默认摄像头
            if not self.camera.isOpened():
                print("[错误] 无法打开摄像头")
                return False
                
            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟
            
            print("[成功] 摄像头初始化成功 - 分辨率: {}x{}".format(CAMERA_WIDTH, CAMERA_HEIGHT))
            return True
            
        except Exception as e:
            print("[错误] 摄像头初始化失败: {}".format(e))
            return False
    
    def capture_frame(self):
        """捕获单帧图像"""
        if not self.camera or not self.camera.isOpened():
            return None
            
        ret, frame = self.camera.read()
        if ret:
            self.last_frame = frame
            return frame
        return None
    
    def start_preview(self):
        """开始视频预览"""
        if not self.camera or not self.camera.isOpened():
            print("[错误] 摄像头未初始化")
            return False
            
        if self.is_previewing:
            print("[警告] 预览已在运行")
            return True
            
        try:
            self.is_previewing = True
            self.preview_thread = threading.Thread(target=self._preview_worker, daemon=True)
            self.preview_thread.start()
            print("[成功] 摄像头预览已启动")
            return True
            
        except Exception as e:
            print("[错误] 预览启动失败: {}".format(e))
            self.is_previewing = False
            return False
    
    def stop_preview(self):
        """停止视频预览"""
        if self.is_previewing:
            self.is_previewing = False
            if hasattr(self, 'preview_thread'):
                self.preview_thread.join(timeout=2)
            cv2.destroyAllWindows()
            print("[停止] 摄像头预览已停止")
    
    def _preview_worker(self):
        """预览工作线程"""
        cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Preview", 640, 480)
        
        # 添加操作提示
        print("\n[预览] 摄像头预览操作:")
        print("  空格键 - 拍照并发送诊断")
        print("  's' - 仅保存照片")
        print("  'r' - 开始录音对话")
        print("  'q' - 退出预览")
        
        while self.is_previewing:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # 添加操作提示文字到画面
                frame_with_text = frame.copy()
                cv2.putText(frame_with_text, "Space: Capture & Diagnose", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_with_text, "S: Save Photo", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_with_text, "R: Voice Chat", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_with_text, "Q: Quit", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("Camera Preview", frame_with_text)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # 空格键 - 拍照诊断
                    self.capture_and_diagnose_from_preview(frame)
                elif key == ord('s'):  # S键 - 保存照片
                    self.save_photo_from_preview(frame)
                elif key == ord('r'):  # R键 - 语音对话
                    self.start_voice_chat()
                elif key == ord('q'):  # Q键 - 退出
                    self.is_previewing = False
                    break
                    
            except Exception as e:
                print("[错误] 预览错误: {}".format(e))
                break
        
        cv2.destroyWindow("Camera Preview")
    
    def capture_and_diagnose_from_preview(self, frame):
        """从预览中拍照并诊断"""
        try:
            print("[拍照] 正在拍摄并发送诊断...")
            
            # 图像增强
            enhanced_frame = self.enhance_image(frame)
            
            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "preview_capture_{}.jpg".format(timestamp)
            filepath = self.save_image(enhanced_frame, filename)
            
            if filepath:
                # 显示拍摄提示
                self.show_capture_feedback()
                
                # 发送诊断（这里需要集成网络管理器）
                print("[发送] 图像已准备发送诊断")
                return filepath
            
        except Exception as e:
            print("[错误] 拍照诊断失败: {}".format(e))
            return None
    
    def save_photo_from_preview(self, frame):
        """从预览中保存照片"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "preview_photo_{}.jpg".format(timestamp)
            filepath = self.save_image(frame, filename)
            
            if filepath:
                print("[保存] 照片已保存: {}".format(filename))
                self.show_capture_feedback()
            
        except Exception as e:
            print("[错误] 照片保存失败: {}".format(e))
    
    def show_capture_feedback(self):
        """显示拍摄反馈"""
        try:
            # 创建拍摄反馈窗口
            feedback = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(feedback, "Photo Captured!", (80, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(feedback, "Processing...", (120, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Capture Feedback", feedback)
            cv2.waitKey(1000)  # 显示1秒
            cv2.destroyWindow("Capture Feedback")
            
        except Exception as e:
            print("[错误] 反馈显示失败: {}".format(e))
    
    def start_voice_chat(self):
        """启动语音对话"""
        try:
            print("[语音] 启动语音对话功能...")
            
            # 这里需要集成语音交互模块
            import subprocess
            import sys
            
            # 启动语音交互程序
            subprocess.Popen([sys.executable, "board_voice_interaction.py"], 
                           cwd=os.getcwd())
            
            print("[启动] 语音对话程序已启动")
            
        except Exception as e:
            print("[错误] 语音对话启动失败: {}".format(e))
    
    def capture_for_diagnosis(self):
        """为诊断捕获高质量图像"""
        frame = self.capture_frame()
        if frame is not None:
            # 应用图像增强
            enhanced_frame = self.enhance_image(frame)
            return enhanced_frame
        return None
    
    def enhance_image(self, frame):
        """图像增强处理"""
        try:
            # 1. 去噪
            denoised = cv2.bilateralFilter(frame, 9, 75, 75)
            
            # 2. 锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # 3. 色彩增强
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            print("[警告] 图像增强失败，使用原图: {}".format(e))
            return frame
    
    def save_image(self, frame, filename=None):
        """保存图像到本地"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = "medical_image_{}.jpg".format(timestamp)
            
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print("[图像] 图像已保存: {}".format(filepath))
            return filepath
            
        except Exception as e:
            print("[错误] 图像保存失败: {}".format(e))
            return None
    
    def release(self):
        """释放摄像头资源"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            print("[摄像头] 摄像头已释放")

class NetworkManager:
    """网络通信管理器"""
    
    def __init__(self):
        self.camera_sock = None
        self.diagnosis_sock = None
        self.command_sock = None
        self.init_sockets()
        
    def init_sockets(self):
        """初始化网络套接字"""
        try:
            # 摄像头数据发送套接字
            self.camera_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.camera_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
            
            # 诊断结果接收套接字
            self.diagnosis_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.diagnosis_sock.bind(("0.0.0.0", DIAGNOSIS_PORT))
            self.diagnosis_sock.settimeout(10.0)  # 10秒超时
            
            # 命令控制套接字
            self.command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            print("[网络] 网络套接字初始化完成")
            return True
            
        except Exception as e:
            print("[错误] 网络初始化失败: {}".format(e))
            return False
    
    def send_image_for_diagnosis(self, image):
        """发送图像到PC端进行诊断"""
        try:
            # 编码图像
            _, img_encoded = cv2.imencode('.jpg', image, 
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            img_data = img_encoded.tobytes()
            
            # 创建传输包
            timestamp = int(time.time() * 1000)
            header = {
                "type": "diagnosis_request",
                "timestamp": timestamp,
                "image_size": len(img_data),
                "width": image.shape[1],
                "height": image.shape[0]
            }
            
            # 发送头部信息
            header_data = json.dumps(header).encode('utf-8')
            self.command_sock.sendto(header_data, (PC_IP, COMMAND_PORT))
            
            # 分片发送图像数据
            total_packets = (len(img_data) + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
            
            for i in range(total_packets):
                start = i * MAX_PACKET_SIZE
                end = min(start + MAX_PACKET_SIZE, len(img_data))
                packet_data = img_data[start:end]
                
                # 包头：4字节包序号 + 4字节总包数 + 数据
                packet_header = i.to_bytes(4, 'big') + total_packets.to_bytes(4, 'big')
                packet = packet_header + packet_data
                
                self.camera_sock.sendto(packet, (PC_IP, CAMERA_PORT))
                time.sleep(0.001)  # 小延迟避免网络拥塞
            
            print("[发送] 图像已发送到PC端 (大小: {} 字节, 分片: {})".format(len(img_data), total_packets))
            return timestamp
            
        except Exception as e:
            print("[错误] 图像发送失败: {}".format(e))
            return None
    
    def wait_for_diagnosis_result(self, timeout=30):
        """等待诊断结果"""
        try:
            print("[等待] 等待PC端诊断结果...")
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    data, addr = self.diagnosis_sock.recvfrom(4096)
                    result = json.loads(data.decode('utf-8'))
                    print("[成功] 收到诊断结果")
                    return result
                    
                except socket.timeout:
                    print("[时间] 等待诊断结果超时")
                    continue
                except json.JSONDecodeError:
                    print("[警告] 诊断结果格式错误")
                    continue
            
            print("[错误] 诊断超时")
            return None
            
        except Exception as e:
            print("[错误] 接收诊断结果失败: {}".format(e))
            return None
    
    def close(self):
        """关闭网络连接"""
        for sock in [self.camera_sock, self.diagnosis_sock, self.command_sock]:
            if sock:
                sock.close()
        print("[网络] 网络连接已关闭")

class MedicalDiagnosisBoard:
    """开发板医疗诊断主控制器"""
    
    def __init__(self):
        self.camera_manager = CameraManager()
        self.network_manager = NetworkManager()
        self.is_running = False
        self.diagnosis_history = []
        
    def initialize(self):
        """初始化系统"""
        print("[启动] 初始化开发板医疗诊断系统...")
        
        if not self.camera_manager.init_camera():
            return False
            
        if not self.network_manager.init_sockets():
            return False
            
        self.is_running = True
        print("[成功] 系统初始化完成")
        return True
    
    def capture_and_diagnose(self):
        """捕获图像并进行诊断"""
        print("\n[图像] 开始图像捕获和诊断...")
        
        # 捕获图像
        frame = self.camera_manager.capture_for_diagnosis()
        if frame is None:
            print("[错误] 图像捕获失败")
            return None
        
        # 保存原始图像
        saved_path = self.camera_manager.save_image(frame)
        
        # 发送到PC端诊断
        request_id = self.network_manager.send_image_for_diagnosis(frame)
        if request_id is None:
            return None
        
        # 等待诊断结果
        result = self.network_manager.wait_for_diagnosis_result()
        
        if result:
            # 保存诊断记录
            record = {
                "timestamp": datetime.now().isoformat(),
                "image_path": saved_path,
                "diagnosis": result,
                "request_id": request_id
            }
            self.diagnosis_history.append(record)
            
            # 显示结果
            self.display_diagnosis_result(result)
            
            # 保存诊断记录到文件
            self.save_diagnosis_record(record)
            
            return record
        
        return None
    
    def display_diagnosis_result(self, result):
        """显示诊断结果"""
        print("\n" + "="*50)
        print("[医院] AI医疗诊断结果")
        print("="*50)
        
        if 'disease_name' in result:
            print("[请求] 诊断结果: {}".format(result['disease_name']))
        
        if 'confidence' in result:
            confidence = result['confidence']
            print("[目标] 置信度: {:.2%}".format(confidence))
            
            # 置信度颜色提示
            if confidence > 0.8:
                print("[高] 高置信度诊断")
            elif confidence > 0.6:
                print("[中] 中等置信度诊断")
            else:
                print("[低] 低置信度诊断，建议专业医生确认")
        
        if 'advice' in result:
            print("[提示] 建议: {}".format(result['advice']))
        
        if 'emergency' in result and result['emergency']:
            print("[紧急] 紧急情况！建议立即就医")
        
        print("="*50)
    
    def save_diagnosis_record(self, record):
        """保存诊断记录"""
        try:
            record_file = os.path.join(SAVE_DIR, "diagnosis_history.json")
            
            # 读取现有记录
            if os.path.exists(record_file):
                with open(record_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
            
            # 添加新记录
            history.append(record)
            
            # 保持最近100条记录
            if len(history) > 100:
                history = history[-100:]
            
            # 保存到文件
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            print("[文档] 诊断记录已保存")
            
        except Exception as e:
            print("[错误] 诊断记录保存失败: {}".format(e))
    
    def continuous_monitoring(self, interval=5):
        """连续监控模式"""
        print("[更新] 开始连续监控模式，间隔 {} 秒".format(interval))
        
        try:
            while self.is_running:
                self.capture_and_diagnose()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n[取消] 停止连续监控")
    
    def interactive_mode(self):
        """交互模式"""
        print("\n[命令] 进入交互模式")
        print("命令:")
        print("  'c' - 捕获并诊断")
        print("  'h' - 查看历史记录")
        print("  'm' - 连续监控模式")
        print("  'q' - 退出")
        
        while self.is_running:
            try:
                cmd = input("\n请输入命令: ").strip().lower()
                
                if cmd == 'c':
                    self.capture_and_diagnose()
                elif cmd == 'h':
                    self.show_history()
                elif cmd == 'm':
                    interval = input("请输入监控间隔(秒，默认5): ").strip()
                    try:
                        interval = int(interval) if interval else 5
                    except:
                        interval = 5
                    self.continuous_monitoring(interval)
                elif cmd == 'q':
                    break
                else:
                    print("[错误] 无效命令")
                    
            except KeyboardInterrupt:
                break
    
    def show_history(self):
        """显示历史记录"""
        if not self.diagnosis_history:
            print("[空] 暂无诊断历史")
            return
        
        print("\n[历史] 最近 {} 条诊断记录:".format(len(self.diagnosis_history)))
        print("-" * 60)
        
        for i, record in enumerate(self.diagnosis_history[-10:], 1):
            timestamp = record['timestamp']
            diagnosis = record['diagnosis']
            print("{}. {}".format(i, timestamp))
            if 'disease_name' in diagnosis:
                print("   诊断: {}".format(diagnosis['disease_name']))
            if 'confidence' in diagnosis:
                print("   置信度: {:.2%}".format(diagnosis['confidence']))
            print("-" * 60)
    
    def shutdown(self):
        """关闭系统"""
        print("\n[更新] 正在关闭系统...")
        self.is_running = False
        self.camera_manager.release()
        self.network_manager.close()
        print("[成功] 系统已关闭")

def main():
    """主函数"""
    print("[医院] 开发板医疗诊断系统启动")
    
    # 创建系统实例
    diagnosis_board = MedicalDiagnosisBoard()
    
    try:
        # 初始化系统
        if not diagnosis_board.initialize():
            print("[错误] 系统初始化失败")
            return
        
        # 启动交互模式
        diagnosis_board.interactive_mode()
        
    except KeyboardInterrupt:
        print("\n[取消] 程序中断")
    except Exception as e:
        print("[错误] 系统错误: {}".format(e))
    finally:
        diagnosis_board.shutdown()

if __name__ == "__main__":
    main()
