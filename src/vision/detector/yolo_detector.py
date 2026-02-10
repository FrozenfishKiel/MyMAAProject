"""
YOLO目标检测器模块
基于Ultralytics YOLO实现游戏界面中的目标检测功能
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging

# 尝试导入Ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("Ultralytics YOLO未安装，目标检测功能将不可用")


# 全局单例实例，避免重复初始化
_yolo_detector_instance = None

class YOLODetector:
    """YOLO目标检测器"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', use_gpu: bool = False, 
                 confidence_threshold: float = 0.5, iou_threshold: float = 0.5):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型路径，可以是预训练模型名称或本地模型文件路径
            use_gpu: 是否使用GPU加速
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        global _yolo_detector_instance
        
        # 防止重复初始化 - 如果单例实例已存在，抛出异常
        if _yolo_detector_instance is not None:
            raise RuntimeError("YOLODetector实例已存在，请使用get_instance()方法获取单例实例")
        
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.is_initialized = False
        self.class_names = []
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 设置全局单例实例
        _yolo_detector_instance = self
        
        # 初始化YOLO模型
        self._initialize_yolo()
    
    @classmethod
    def get_instance(cls, model_path: str = 'yolov8n.pt', use_gpu: bool = False, 
                    confidence_threshold: float = 0.5, iou_threshold: float = 0.5):
        """获取YOLO检测器的单例实例
        
        Args:
            model_path: YOLO模型路径，可以是预训练模型名称或本地模型文件路径
            use_gpu: 是否使用GPU加速
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值
            
        Returns:
            YOLODetector实例
        """
        global _yolo_detector_instance
        
        if _yolo_detector_instance is None:
            _yolo_detector_instance = cls(model_path, use_gpu, confidence_threshold, iou_threshold)
        
        return _yolo_detector_instance
    
    def _initialize_yolo(self) -> bool:
        """初始化YOLO模型"""
        if YOLO is None:
            self.logger.error("Ultralytics YOLO未安装，无法初始化检测器")
            return False
        
        try:
            # 添加调试信息
            self.logger.info(f"正在加载YOLO模型: {self.model_path}")
            
            # 检查模型文件是否存在
            if os.path.exists(self.model_path):
                self.logger.info(f"[OK] 模型文件存在: {self.model_path}")
            else:
                self.logger.warning(f"[WARNING] 模型文件不存在: {self.model_path}")
            
            # 加载YOLO模型
            self.model = YOLO(self.model_path)
            
            # 获取类别名称
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                self.logger.info(f"[DEBUG] 模型类别名称: {self.class_names}")
                self.logger.info(f"[DEBUG] 类别数量: {len(self.class_names)}")
            else:
                # 如果是自定义模型，可能需要从配置中获取类别名称
                self.class_names = {}
                self.logger.warning("[WARNING] 模型没有names属性，使用空类别字典")
            
            # 设置设备
            device = 'cuda' if self.use_gpu else 'cpu'
            self.model.to(device)
            
            # 验证设备设置是否成功
            actual_device = str(next(self.model.model.parameters()).device)
            self.logger.info(f"[DEBUG] 模型实际运行在设备: {actual_device}")
            
            self.is_initialized = True
            self.logger.info(f"[OK] YOLO检测器初始化成功，使用模型: {self.model_path}")
            self.logger.info(f"[DEBUG] 检测器配置运行在设备: {device}")
            self.logger.info(f"[DEBUG] 模型实际运行在设备: {actual_device}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] YOLO检测器初始化失败: {e}")
            self.is_initialized = False
            return False
    
    def detect_objects(self, image: np.ndarray, classes: List[int] = None, 
                      regions: List[Tuple[int, int, int, int]] = None) -> List[Dict]:
        """
        检测图像中的目标物体
        
        Args:
            image: 输入图像，BGR格式
            classes: 可选，指定检测的类别ID列表
            regions: 可选，指定检测区域列表，每个区域为(x, y, w, h)
            
        Returns:
            目标检测结果列表，每个元素包含类别、置信度、边界框等信息
        """
        if not self.is_initialized or self.model is None:
            self.logger.warning("YOLO检测器未初始化")
            return []
        
        try:
            # 如果指定了检测区域，则只在这些区域进行检测
            if regions:
                results = []
                for region in regions:
                    x, y, w, h = region
                    roi = image[y:y+h, x:x+w]
                    
                    # 对每个区域进行目标检测
                    roi_results = self.model.predict(
                        roi, 
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        classes=classes,
                        verbose=False
                    )
                    
                    # 处理检测结果
                    for result in roi_results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                # 获取检测信息
                                cls_id = int(box.cls.item()) if box.cls is not None else 0
                                conf = box.conf.item() if box.conf is not None else 0.0
                                bbox = box.xyxy[0].tolist() if box.xyxy is not None else [0, 0, 0, 0]
                                
                                # 调整坐标到原图坐标系
                                adjusted_bbox = [
                                    bbox[0] + x,  # x1
                                    bbox[1] + y,  # y1
                                    bbox[2] + x,  # x2
                                    bbox[3] + y   # y2
                                ]
                                
                                results.append({
                                    'class_id': cls_id,
                                    'class_name': self.class_names.get(cls_id, f'class_{cls_id}'),
                                    'confidence': conf,
                                    'bbox': adjusted_bbox,
                                    'region': region
                                })
                return results
            else:
                # 全图检测
                results = self.model.predict(
                    image,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    classes=classes,
                    verbose=False
                )
                
                # 格式化结果
                formatted_results = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls_id = int(box.cls.item()) if box.cls is not None else 0
                            conf = box.conf.item() if box.conf is not None else 0.0
                            bbox = box.xyxy[0].tolist() if box.xyxy is not None else [0, 0, 0, 0]
                            
                            formatted_results.append({
                                'class_id': cls_id,
                                'class_name': self.class_names.get(cls_id, f'class_{cls_id}'),
                                'confidence': conf,
                                'bbox': bbox,
                                'region': None
                            })
                
                return formatted_results
                
        except Exception as e:
            self.logger.error(f"目标检测失败: {e}")
            return []
    
    def detect_specific_classes(self, image: np.ndarray, target_classes: List[str], 
                              confidence_threshold: float = None) -> List[Dict]:
        """
        检测特定类别的目标
        
        Args:
            image: 输入图像
            target_classes: 目标类别名称列表
            confidence_threshold: 置信度阈值，如果为None则使用默认阈值
            
        Returns:
            匹配到的目标信息列表
        """
        # 将类别名称转换为类别ID
        class_ids = []
        for class_name in target_classes:
            for cls_id, name in self.class_names.items():
                if name == class_name:
                    class_ids.append(cls_id)
                    break
        
        # 如果没有找到对应的类别ID，返回空列表
        if not class_ids:
            self.logger.warning(f"未找到目标类别: {target_classes}")
            return []
        
        # 使用指定的置信度阈值
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        # 检测指定类别的目标
        results = self.detect_objects(image, classes=class_ids)
        
        # 过滤置信度
        filtered_results = [
            result for result in results 
            if result['confidence'] >= conf_thresh
        ]
        
        return filtered_results
    
    def get_bbox_center(self, bbox: List[float]) -> Tuple[int, int]:
        """
        计算边界框的中心点坐标
        
        Args:
            bbox: 边界框，格式为[x1, y1, x2, y2]
            
        Returns:
            中心点坐标(x, y)
        """
        if not bbox or len(bbox) < 4:
            return (0, 0)
        
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        
        return (center_x, center_y)
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       show_labels: bool = True, show_confidences: bool = True) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            show_labels: 是否显示标签
            show_confidences: 是否显示置信度
            
        Returns:
            绘制了检测结果的图像
        """
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 准备标签文本
            label_parts = []
            if show_labels:
                label_parts.append(class_name)
            if show_confidences:
                label_parts.append(f'{confidence:.2f}')
            
            if label_parts:
                label = ' '.join(label_parts)
                
                # 计算标签背景大小
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # 绘制标签背景
                cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # 绘制标签文本
                cv2.putText(result_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image
    
    def is_available(self) -> bool:
        """检查YOLO检测功能是否可用"""
        return self.is_initialized and self.model is not None
    
    def get_class_names(self) -> Dict[int, str]:
        """获取类别名称映射"""
        return self.class_names.copy()
    
    def update_confidence_threshold(self, threshold: float):
        """更新置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"置信度阈值更新为: {self.confidence_threshold}")
    
    def update_iou_threshold(self, threshold: float):
        """更新IOU阈值"""
        self.iou_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"IOU阈值更新为: {self.iou_threshold}")