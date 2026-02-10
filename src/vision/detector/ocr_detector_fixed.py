"""
修复版OCR文字识别检测器模块
专门解决"could not create a primitive"错误
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import traceback

# 强制设置环境变量，解决PaddlePaddle底层问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # 限制线程数，避免并发问题
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# 尝试导入PaddleOCR
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None
    logging.warning("PaddleOCR未安装，OCR功能将不可用")


class OCRDetectorFixed:
    """修复版OCR文字识别检测器"""
    
    def __init__(self, lang: str = 'ch', use_gpu: bool = False, model_dir: str = None):
        """
        初始化OCR检测器（修复版）
        
        Args:
            lang: 语言类型，'ch'表示中文，'en'表示英文
            use_gpu: 是否使用GPU加速
            model_dir: 模型文件目录，如果为None则使用默认路径
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.model_dir = model_dir
        self.ocr_engine = None
        self.is_initialized = False
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化OCR引擎（修复版初始化方法）
        self._initialize_ocr_fixed()
    
    def _initialize_ocr_fixed(self) -> bool:
        """修复版OCR引擎初始化方法"""
        if PaddleOCR is None:
            self.logger.error("PaddleOCR未安装，无法初始化OCR引擎")
            return False
        
        try:
            # 简化初始化参数，避免复杂配置导致的问题
            ocr_kwargs = {
                'use_angle_cls': False,      # 关闭角度分类，减少复杂度
                'lang': self.lang,           # 语言类型
                'use_gpu': False,            # 强制使用CPU，避免GPU问题
                'show_log': False,           # 不显示详细日志
                'det_limit_side_len': 960,   # 降低检测尺寸限制，减少内存使用
                'det_db_thresh': 0.3,        # 使用标准检测阈值
                'det_db_box_thresh': 0.5,    # 使用标准框检测阈值
                'rec_batch_num': 1,          # 减少批次大小，避免内存问题
                'drop_score': 0.5,           # 使用标准过滤阈值
            }
            
            # 分步初始化，避免一次性加载过多资源
            self.logger.info("开始分步初始化OCR引擎...")
            
            # 第一步：尝试最简单的初始化
            try:
                self.ocr_engine = PaddleOCR(**ocr_kwargs)
                self.logger.info("✓ 第一步初始化成功")
            except Exception as e:
                self.logger.warning(f"第一步初始化失败: {e}")
                
                # 第二步：尝试更简化的配置
                try:
                    ocr_kwargs_simple = {
                        'use_angle_cls': False,
                        'lang': self.lang,
                        'use_gpu': False,
                        'show_log': False
                    }
                    self.ocr_engine = PaddleOCR(**ocr_kwargs_simple)
                    self.logger.info("✓ 简化配置初始化成功")
                except Exception as e2:
                    self.logger.error(f"简化配置初始化也失败: {e2}")
                    self.logger.error(f"详细错误信息: {traceback.format_exc()}")
                    return False
            
            # 测试引擎是否可用
            self.logger.info("测试OCR引擎可用性...")
            try:
                # 创建一个简单的测试图像
                test_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
                result = self.ocr_engine.ocr(test_image, cls=False)
                self.logger.info("✓ OCR引擎测试通过")
            except Exception as e:
                self.logger.warning(f"OCR引擎测试失败，但继续使用: {e}")
            
            self.is_initialized = True
            self.logger.info("OCR引擎初始化成功（修复版）")
            return True
            
        except Exception as e:
            self.logger.error(f"OCR引擎初始化失败（修复版）: {e}")
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            self.is_initialized = False
            return False
    
    def detect_text_safe(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """
        安全版文字检测，包含错误处理和重试机制
        
        Args:
            image: 输入图像
            confidence_threshold: 置信度阈值
            
        Returns:
            文字检测结果列表
        """
        if not self.is_initialized:
            self.logger.error("OCR引擎未初始化")
            return []
        
        try:
            # 第一步：尝试直接识别
            result = self.ocr_engine.ocr(image, cls=False)
            
            # 处理识别结果
            text_detections = []
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        bbox = line[0]
                        text_info = line[1]
                        
                        if len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            
                            if confidence >= confidence_threshold:
                                detection = {
                                    'text': text,
                                    'confidence': confidence,
                                    'bbox': bbox,
                                    'center_x': np.mean([point[0] for point in bbox]),
                                    'center_y': np.mean([point[1] for point in bbox])
                                }
                                text_detections.append(detection)
            
            self.logger.info(f"检测到 {len(text_detections)} 个文字区域")
            return text_detections
            
        except Exception as e:
            self.logger.error(f"OCR检测失败: {e}")
            
            # 尝试重新初始化引擎
            self.logger.info("尝试重新初始化OCR引擎...")
            self._initialize_ocr_fixed()
            
            if self.is_initialized:
                # 重试一次
                try:
                    result = self.ocr_engine.ocr(image, cls=False)
                    
                    text_detections = []
                    if result and result[0]:
                        for line in result[0]:
                            if len(line) >= 2:
                                bbox = line[0]
                                text_info = line[1]
                                
                                if len(text_info) >= 2:
                                    text = text_info[0]
                                    confidence = text_info[1]
                                    
                                    if confidence >= confidence_threshold:
                                        detection = {
                                            'text': text,
                                            'confidence': confidence,
                                            'bbox': bbox,
                                            'center_x': np.mean([point[0] for point in bbox]),
                                            'center_y': np.mean([point[1] for point in bbox])
                                        }
                                        text_detections.append(detection)
                    
                    self.logger.info(f"重试后检测到 {len(text_detections)} 个文字区域")
                    return text_detections
                    
                except Exception as e2:
                    self.logger.error(f"重试也失败: {e2}")
                    return []
            else:
                return []
    
    def detect_text(self, image: np.ndarray, confidence_threshold: float = 0.3, 
                   preprocess: bool = True, use_multi_scale: bool = False) -> List[Dict]:
        """
        文字检测方法（兼容原接口）
        
        Args:
            image: 输入图像
            confidence_threshold: 置信度阈值
            preprocess: 是否进行图像预处理（修复版忽略此参数）
            use_multi_scale: 是否使用多尺度检测（修复版忽略此参数）
            
        Returns:
            文字检测结果列表
        """
        return self.detect_text_safe(image, confidence_threshold)


# 兼容性包装器
def create_ocr_detector(lang='ch', use_gpu=False, model_dir=None):
    """创建OCR检测器（修复版）"""
    return OCRDetectorFixed(lang=lang, use_gpu=use_gpu, model_dir=model_dir)