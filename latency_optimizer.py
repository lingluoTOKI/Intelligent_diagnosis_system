#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
延迟优化模块
专门用于优化开发板触摸屏显示延迟问题
包含多种优化策略和技术
"""

import cv2
import numpy as np
import time
import threading
import queue
import socket
from collections import deque
import math

# ===== 优化配置 =====
# 帧率优化
TARGET_FPS = 60              # 目标帧率
MIN_FPS = 30                 # 最低帧率
ADAPTIVE_QUALITY = True      # 自适应质量调整

# 预测优化
MOTION_PREDICTION = True     # 运动预测
PREDICTION_FRAMES = 2        # 预测帧数
PREDICTION_STRENGTH = 0.3    # 预测强度

# 缓存优化
FRAME_BUFFER_SIZE = 3        # 帧缓存大小
DECODE_THREADS = 2           # 解码线程数

# 网络优化
PACKET_PRIORITIZATION = True # 数据包优先级
ADAPTIVE_COMPRESSION = True  # 自适应压缩
# ===================

class MotionPredictor:
    """运动预测器 - 预测下一帧的显示位置"""
    
    def __init__(self, history_size=5):
        self.history_size = history_size
        self.position_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        
    def add_position(self, x, y, timestamp=None):
        """添加位置历史"""
        if timestamp is None:
            timestamp = time.time()
            
        self.position_history.append((x, y))
        self.time_history.append(timestamp)
    
    def predict_position(self, future_time_offset):
        """预测未来位置"""
        if len(self.position_history) < 2:
            return None
            
        # 计算平均速度
        positions = list(self.position_history)
        times = list(self.time_history)
        
        # 线性回归预测
        if len(positions) >= 3:
            # 使用最近3个点进行预测
            recent_positions = positions[-3:]
            recent_times = times[-3:]
            
            # 计算速度向量
            vx = (recent_positions[-1][0] - recent_positions[0][0]) / (recent_times[-1] - recent_times[0])
            vy = (recent_positions[-1][1] - recent_positions[0][1]) / (recent_times[-1] - recent_times[0])
            
            # 预测位置
            last_pos = positions[-1]
            predicted_x = last_pos[0] + vx * future_time_offset
            predicted_y = last_pos[1] + vy * future_time_offset
            
            return (int(predicted_x), int(predicted_y))
        
        return None

class AdaptiveQualityController:
    """自适应质量控制器"""
    
    def __init__(self):
        self.current_quality = 75
        self.min_quality = 30
        self.max_quality = 95
        self.fps_history = deque(maxlen=10)
        self.latency_history = deque(maxlen=10)
        
    def update_metrics(self, fps, latency_ms):
        """更新性能指标"""
        self.fps_history.append(fps)
        self.latency_history.append(latency_ms)
        
    def get_optimal_quality(self):
        """获取最优质量设置"""
        if not self.fps_history or not self.latency_history:
            return self.current_quality
            
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        
        # 根据FPS调整质量
        if avg_fps < MIN_FPS:
            # 帧率过低，降低质量
            self.current_quality = max(self.min_quality, self.current_quality - 5)
        elif avg_fps > TARGET_FPS * 0.9 and avg_latency < 50:
            # 性能良好，可以提高质量
            self.current_quality = min(self.max_quality, self.current_quality + 2)
        
        # 根据延迟调整质量
        if avg_latency > 100:  # 延迟超过100ms
            self.current_quality = max(self.min_quality, self.current_quality - 8)
        elif avg_latency < 30:  # 延迟很低
            self.current_quality = min(self.max_quality, self.current_quality + 3)
            
        return self.current_quality

class FramePreprocessor:
    """帧预处理器 - 优化帧数据减少传输延迟"""
    
    def __init__(self):
        self.prev_frame = None
        self.motion_areas = []
        
    def detect_motion_areas(self, current_frame):
        """检测运动区域"""
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return [(0, 0, current_frame.shape[1], current_frame.shape[0])]  # 全屏
        
        # 计算帧差
        diff = cv2.absdiff(current_frame, self.prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 阈值处理
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # 形态学操作连接相近区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_areas = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # 过滤小区域
                x, y, w, h = cv2.boundingRect(contour)
                # 扩展边界以包含更多上下文
                x = max(0, x - 10)
                y = max(0, y - 10)
                w = min(current_frame.shape[1] - x, w + 20)
                h = min(current_frame.shape[0] - y, h + 20)
                motion_areas.append((x, y, w, h))
        
        # 如果没有检测到运动，返回小的中心区域
        if not motion_areas:
            h, w = current_frame.shape[:2]
            motion_areas = [(w//4, h//4, w//2, h//2)]
        
        self.prev_frame = current_frame.copy()
        self.motion_areas = motion_areas
        return motion_areas
    
    def create_optimized_frame(self, frame, quality_controller):
        """创建优化的帧数据"""
        motion_areas = self.detect_motion_areas(frame)
        quality = quality_controller.get_optimal_quality()
        
        if len(motion_areas) == 1 and motion_areas[0] == (0, 0, frame.shape[1], frame.shape[0]):
            # 全屏更新，直接压缩
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, encoded = cv2.imencode('.jpg', frame, encode_params)
            return encoded.tobytes(), 'full_frame'
        else:
            # 仅传输运动区域
            motion_data = []
            for x, y, w, h in motion_areas:
                roi = frame[y:y+h, x:x+w]
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, min(quality + 10, 95)]  # 运动区域高质量
                _, encoded = cv2.imencode('.jpg', roi, encode_params)
                motion_data.append({
                    'region': (x, y, w, h),
                    'data': encoded.tobytes()
                })
            
            return motion_data, 'motion_regions'

class OptimizedReceiver:
    """优化的接收器 - 在开发板端使用"""
    
    def __init__(self, display_size):
        self.display_size = display_size
        self.frame_buffer = None
        self.motion_predictor = MotionPredictor()
        self.decode_queue = queue.Queue(maxsize=FRAME_BUFFER_SIZE)
        self.display_queue = queue.Queue(maxsize=2)
        
        # 启动解码线程
        for i in range(DECODE_THREADS):
            threading.Thread(target=self.decode_worker, daemon=True).start()
        
        # 启动显示线程
        threading.Thread(target=self.display_worker, daemon=True).start()
        
    def decode_worker(self):
        """解码工作线程"""
        while True:
            try:
                frame_data, frame_type, timestamp = self.decode_queue.get(timeout=1.0)
                
                if frame_type == 'full_frame':
                    # 完整帧解码
                    img_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # 调整到显示尺寸
                        if frame.shape[:2] != self.display_size:
                            frame = cv2.resize(frame, self.display_size)
                        self.frame_buffer = frame.copy()
                        
                elif frame_type == 'motion_regions':
                    # 运动区域更新
                    if self.frame_buffer is not None:
                        for region_data in frame_data:
                            x, y, w, h = region_data['region']
                            img_array = np.frombuffer(region_data['data'], dtype=np.uint8)
                            roi = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            
                            if roi is not None:
                                # 调整ROI尺寸
                                roi_h, roi_w = roi.shape[:2]
                                if (roi_w, roi_h) != (w, h):
                                    roi = cv2.resize(roi, (w, h))
                                
                                # 更新缓冲区
                                self.frame_buffer[y:y+h, x:x+w] = roi
                
                # 将处理好的帧放入显示队列
                if self.frame_buffer is not None:
                    self.display_queue.put((self.frame_buffer.copy(), timestamp))
                
                self.decode_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[错误] 解码错误: {e}")
    
    def display_worker(self):
        """显示工作线程"""
        last_display_time = 0
        frame_interval = 1.0 / TARGET_FPS
        
        while True:
            try:
                frame, timestamp = self.display_queue.get(timeout=1.0)
                
                current_time = time.time()
                
                # 帧率控制
                if current_time - last_display_time < frame_interval:
                    time.sleep(frame_interval - (current_time - last_display_time))
                
                # 应用运动预测
                if MOTION_PREDICTION:
                    frame = self.apply_motion_prediction(frame, current_time)
                
                # 显示帧
                cv2.imshow("Optimized Display", frame)
                cv2.waitKey(1)
                
                last_display_time = time.time()
                self.display_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[错误] 显示错误: {e}")
    
    def apply_motion_prediction(self, frame, current_time):
        """应用运动预测"""
        try:
            # 预测未来位置（假设延迟为30ms）
            future_offset = 0.03
            predicted_pos = self.motion_predictor.predict_position(future_offset)
            
            if predicted_pos:
                # 基于预测调整帧（简单的位移补偿）
                px, py = predicted_pos
                h, w = frame.shape[:2]
                
                # 计算位移量（限制在合理范围内）
                offset_x = max(-10, min(10, px - w//2))
                offset_y = max(-10, min(10, py - h//2))
                
                # 应用微小的位移补偿
                if abs(offset_x) > 1 or abs(offset_y) > 1:
                    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                    frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            return frame
            
        except Exception as e:
            print(f"[警告] 运动预测失败: {e}")
            return frame
    
    def add_frame_data(self, frame_data, frame_type):
        """添加帧数据到处理队列"""
        try:
            timestamp = time.time()
            self.decode_queue.put((frame_data, frame_type, timestamp), block=False)
        except queue.Full:
            # 队列满时丢弃最老的帧
            try:
                self.decode_queue.get(block=False)
                self.decode_queue.put((frame_data, frame_type, timestamp), block=False)
            except queue.Empty:
                pass

class NetworkOptimizer:
    """网络优化器"""
    
    def __init__(self):
        self.packet_loss_rate = 0.0
        self.rtt = 0.0
        self.bandwidth_mbps = 0.0
        
    def optimize_transmission_params(self):
        """优化传输参数"""
        params = {
            'packet_size': MAX_PACKET_SIZE,
            'send_interval': 0.001,  # 发送间隔
            'compression_level': 75,
            'redundancy': False
        }
        
        # 根据网络状况调整参数
        if self.packet_loss_rate > 0.05:  # 丢包率超过5%
            params['redundancy'] = True  # 启用冗余传输
            params['packet_size'] = int(MAX_PACKET_SIZE * 0.8)  # 减小包大小
        
        if self.rtt > 100:  # RTT超过100ms
            params['send_interval'] = 0.0005  # 减少发送间隔
            params['compression_level'] = 60  # 降低压缩质量换取速度
        
        return params
    
    def update_network_stats(self, packet_loss, rtt, bandwidth):
        """更新网络统计"""
        self.packet_loss_rate = packet_loss
        self.rtt = rtt
        self.bandwidth_mbps = bandwidth

class LatencyOptimizer:
    """主延迟优化器"""
    
    def __init__(self):
        self.quality_controller = AdaptiveQualityController()
        self.frame_preprocessor = FramePreprocessor()
        self.network_optimizer = NetworkOptimizer()
        
        # 性能统计
        self.stats = {
            'frames_processed': 0,
            'avg_processing_time': 0.0,
            'avg_compression_ratio': 0.0,
            'start_time': time.time()
        }
    
    def optimize_frame_for_transmission(self, frame):
        """为传输优化帧"""
        start_time = time.time()
        
        # 获取优化的帧数据
        frame_data, frame_type = self.frame_preprocessor.create_optimized_frame(
            frame, self.quality_controller)
        
        # 更新统计
        processing_time = time.time() - start_time
        self.stats['frames_processed'] += 1
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * (self.stats['frames_processed'] - 1) + processing_time) /
            self.stats['frames_processed']
        )
        
        return frame_data, frame_type
    
    def update_performance_metrics(self, fps, latency_ms, packet_loss=0.0, rtt=0.0, bandwidth=0.0):
        """更新性能指标"""
        self.quality_controller.update_metrics(fps, latency_ms)
        self.network_optimizer.update_network_stats(packet_loss, rtt, bandwidth)
    
    def get_optimization_report(self):
        """获取优化报告"""
        uptime = time.time() - self.stats['start_time']
        
        return {
            'uptime_seconds': uptime,
            'frames_processed': self.stats['frames_processed'],
            'avg_processing_time_ms': self.stats['avg_processing_time'] * 1000,
            'current_quality': self.quality_controller.current_quality,
            'fps_history': list(self.quality_controller.fps_history),
            'latency_history': list(self.quality_controller.latency_history)
        }

# 使用示例函数
def create_optimized_sender(target_ip, target_port):
    """创建优化的发送端"""
    optimizer = LatencyOptimizer()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 配置socket缓冲区
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
    
    return optimizer, sock

def create_optimized_receiver(display_width, display_height):
    """创建优化的接收端"""
    receiver = OptimizedReceiver((display_width, display_height))
    return receiver

if __name__ == "__main__":
    print("[启动] 延迟优化模块")
    print("=" * 40)
    print("功能:")
    print("- 运动预测补偿")
    print("- 自适应质量控制")
    print("- 智能区域更新")
    print("- 多线程解码显示")
    print("- 网络参数优化")
    print("=" * 40)
    
    # 创建优化器实例
    optimizer = LatencyOptimizer()
    
    # 模拟性能数据
    import random
    for i in range(10):
        fps = random.uniform(25, 35)
        latency = random.uniform(40, 120)
        optimizer.update_performance_metrics(fps, latency)
        
        print(f"帧 {i+1}: FPS={fps:.1f}, 延迟={latency:.1f}ms, 质量={optimizer.quality_controller.get_optimal_quality()}")
    
    # 显示优化报告
    report = optimizer.get_optimization_report()
    print(f"\n[统计] 优化报告:")
    print(f"处理帧数: {report['frames_processed']}")
    print(f"平均处理时间: {report['avg_processing_time_ms']:.2f}ms")
    print(f"当前质量: {report['current_quality']}")
