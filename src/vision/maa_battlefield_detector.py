"""
干员血条位置检测器 - 基于MAA的BattlefieldDetector移植
使用YOLOv8模型检测战场上干员的血条位置
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict, Optional
import os


class MaaBattlefieldDetector:
    """干员血条位置检测器 - 基于MAA的YOLOv8模型"""
    
    def __init__(self, onnx_model_path: str = None):
        """
        初始化干员血条检测器
        
        Args:
            onnx_model_path: ONNX模型文件路径
        """
        if onnx_model_path is None:
            onnx_model_path = r"d:\BiShe\MaaAssistantArknights-dev\resource\onnx\operators_det.onnx"
        
        self.onnx_model_path = onnx_model_path
        self.session = None
        self.input_size = (640, 640)  # YOLOv8输入尺寸
        self.confidence_threshold = 0.3  # 置信度阈值
        self.nms_threshold = 0.7  # 非极大值抑制阈值
        
        # 初始化ONNX推理会话
        self._init_onnx_session()
    
    def _init_onnx_session(self):
        """初始化ONNX推理会话"""
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {self.onnx_model_path}")
        
        # 创建推理会话
        self.session = ort.InferenceSession(
            self.onnx_model_path, 
            providers=['CPUExecutionProvider']
        )
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"✅ 加载ONNX模型: {os.path.basename(self.onnx_model_path)}")
        print(f"📊 输入名称: {self.input_name}")
        print(f"📊 输出名称: {self.output_name}")
    
    def detect_operators(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的干员血条位置
        
        Args:
            image: 输入图像(BGR格式)
            
        Returns:
            检测到的干员信息列表
        """
        if self.session is None:
            raise RuntimeError("ONNX会话未初始化")
        
        # 1. 图像预处理
        processed_image, scale_x, scale_y = self._preprocess_image(image)
        
        # 2. ONNX推理
        input_tensor = processed_image.astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # 3. 解析YOLOv8输出
        raw_output = outputs[0]
        detections = self._parse_yolov8_output(raw_output, scale_x, scale_y)
        
        # 4. 非极大值抑制
        filtered_detections = self._non_max_suppression(detections)
        
        return filtered_detections
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        图像预处理
        
        Args:
            image: 原始图像
            
        Returns:
            processed_image: 预处理后的图像
            scale_x: X轴缩放比例
            scale_y: Y轴缩放比例
        """
        # 计算缩放比例
        scale_x = self.input_size[0] / image.shape[1]
        scale_y = self.input_size[1] / image.shape[0]
        
        # 缩放图像
        resized_image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
        
        # 转换为RGB并归一化
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        normalized_image = rgb_image.astype(np.float32) / 255.0
        
        # 调整维度顺序 (H, W, C) -> (C, H, W)
        channel_first = np.transpose(normalized_image, (2, 0, 1))
        
        # 添加批次维度
        batch_image = np.expand_dims(channel_first, axis=0)
        
        return batch_image, scale_x, scale_y
    
    def _parse_yolov8_output(self, raw_output: np.ndarray, scale_x: float, scale_y: float) -> List[Dict]:
        """
        解析YOLOv8输出
        
        Args:
            raw_output: 原始输出
            scale_x: X轴缩放比例
            scale_y: Y轴缩放比例
            
        Returns:
            解析后的检测结果
        """
        # YOLOv8输出形状: (1, 5, 8400)
        # 解析逻辑参考MAA的实现
        output_shape = raw_output.shape
        
        # 重新组织输出数据
        output_data = []
        for i in range(output_shape[1]):
            output_data.append(raw_output[0, i, :])
        
        detections = []
        
        # 遍历所有检测框
        for i in range(output_shape[2]):
            score = output_data[4][i]  # 置信度
            
            # 过滤低置信度检测
            if score < self.confidence_threshold:
                continue
            
            # 解析边界框坐标
            center_x = int(output_data[0][i] / scale_x)
            center_y = int(output_data[1][i] / scale_y)
            width = int(output_data[2][i] / scale_x)
            height = int(output_data[3][i] / scale_y)
            
            # 计算边界框坐标
            x = center_x - width // 2
            y = center_y - height // 2
            
            detections.append({
                'bbox': (x, y, width, height),
                'score': float(score),
                'class': 'operator',
                'center': (center_x, center_y)
            })
        
        return detections
    
    def _non_max_suppression(self, detections: List[Dict]) -> List[Dict]:
        """
        非极大值抑制 - 参考MAA的实现
        
        Args:
            detections: 原始检测结果
            
        Returns:
            过滤后的检测结果
        """
        if not detections:
            return []
        
        # 按置信度排序
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        filtered_detections = []
        
        while detections:
            current = detections.pop(0)
            filtered_detections.append(current)
            
            remaining = []
            for detection in detections:
                # 计算IoU
                iou = self._calculate_iou(current['bbox'], detection['bbox'])
                
                # 如果IoU小于阈值，保留检测框
                if iou < self.nms_threshold:
                    remaining.append(detection)
            
            detections = remaining
        
        return filtered_detections
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集区域
        xx1 = max(x1, x2)
        yy1 = max(y1, y2)
        xx2 = min(x1 + w1, x2 + w2)
        yy2 = min(y1 + h1, y2 + h2)
        
        # 计算交集面积
        inter_width = max(0, xx2 - xx1)
        inter_height = max(0, yy2 - yy1)
        inter_area = inter_width * inter_height
        
        # 计算并集面积
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        # 计算IoU
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def visualize_detection(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果
            
        Returns:
            可视化后的图像
        """
        vis_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            score = detection['score']
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # 添加置信度标签
            label = f"Operator: {score:.2f}"
            cv2.putText(vis_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return vis_image


def test_battlefield_detector():
    """测试干员血条检测器"""
    detector = MaaBattlefieldDetector()
    
    print("🎯 MAA干员血条检测器初始化完成！")
    print("📊 模型信息:")
    print(f"   - 输入尺寸: {detector.input_size}")
    print(f"   - 置信度阈值: {detector.confidence_threshold}")
    print(f"   - NMS阈值: {detector.nms_threshold}")
    
    print("\n📝 使用方法:")
    print("   1. 准备游戏截图图像")
    print("   2. 调用 detector.detect_operators(image)")
    print("   3. 获取干员血条位置信息")


if __name__ == "__main__":
    test_battlefield_detector()