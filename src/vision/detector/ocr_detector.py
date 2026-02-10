"""
OCR文字识别检测器模块
基于PaddleOCR实现游戏界面中的文字识别功能
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


# 全局单例实例，避免重复初始化
_ocr_detector_instance = None

class OCRDetector:
    """OCR文字识别检测器"""
    
    def __init__(self, lang: str = 'ch', use_gpu: bool = False, model_dir: str = None):
        """
        初始化OCR检测器
        
        Args:
            lang: 语言类型，'ch'表示中文，'en'表示英文
            use_gpu: 是否使用GPU加速
            model_dir: 模型文件目录，如果为None则使用默认路径
        """
        global _ocr_detector_instance
        
        # 防止重复初始化 - 如果单例实例已存在，抛出异常
        if _ocr_detector_instance is not None:
            raise RuntimeError("OCRDetector实例已存在，请使用get_instance()方法获取单例实例")
        
        self.lang = lang
        self.use_gpu = use_gpu
        self.model_dir = model_dir
        self.ocr_engine = None
        self.is_initialized = False
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 设置全局单例实例
        _ocr_detector_instance = self
        
        # 初始化OCR引擎
        self._initialize_ocr()
    
    @classmethod
    def get_instance(cls, lang: str = 'ch', use_gpu: bool = False, model_dir: str = None):
        """获取OCR检测器的单例实例
        
        Args:
            lang: 语言类型
            use_gpu: 是否使用GPU加速
            model_dir: 模型文件目录
            
        Returns:
            OCRDetector实例
        """
        global _ocr_detector_instance
        
        if _ocr_detector_instance is None:
            _ocr_detector_instance = cls(lang, use_gpu, model_dir)
        
        return _ocr_detector_instance
    
    def _initialize_ocr(self) -> bool:
        """初始化PaddleOCR引擎 - 修复版"""
        if PaddleOCR is None:
            self.logger.error("PaddleOCR未安装，无法初始化OCR引擎")
            return False
        
        try:
            # 优化初始化参数，启用GPU并提高检测率
            ocr_kwargs = {
                'use_angle_cls': True,       # 启用角度分类，提高识别准确率
                'lang': self.lang,           # 语言类型
                'use_gpu': True,             # 启用GPU加速，提高性能
                'show_log': False,           # 不显示详细日志
                'det_limit_side_len': 2560,  # 提高检测尺寸限制，检测更多文字
                'det_db_thresh': 0.01,       # 极低检测阈值，最大化检测率
                'det_db_box_thresh': 0.05,   # 极低框检测阈值，几乎不筛选
                'rec_batch_num': 8,          # 增加批次大小，提高识别效率
                'drop_score': 0.1,           # 极低过滤阈值，保留更多结果
                'rec_image_shape': '3, 32, 480', # 优化识别图像尺寸
                'use_space_char': True,      # 使用空格字符
                'max_text_length': 100,      # 增加最大文本长度
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
                self.logger.info("✓ OCR引擎测试成功")
                self.is_initialized = True
                return True
            except Exception as e:
                self.logger.error(f"OCR引擎测试失败: {e}")
                self.logger.error(f"详细错误信息: {traceback.format_exc()}")
                return False
                
        except Exception as e:
            self.logger.error(f"OCR引擎初始化失败: {e}")
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理，提高OCR识别率
        
        Args:
            image: 输入图像，BGR格式
            
        Returns:
            预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 直方图均衡化，增强对比度
        gray = cv2.equalizeHist(gray)
        
        # 高斯模糊去噪
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 形态学操作，去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 转换回BGR格式
        processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return processed

    def detect_text(self, image: np.ndarray, confidence_threshold: float = 0.3, 
                   preprocess: bool = True, use_multi_scale: bool = True) -> List[Dict]:
        """
        检测图像中的文字
        
        Args:
            image: 输入图像
            confidence_threshold: 置信度阈值
            preprocess: 是否进行图像预处理
            use_multi_scale: 是否使用多尺度检测（解决识别不完整问题）
            
        Returns:
            文字检测结果列表
        """
        if use_multi_scale:
            return self.detect_text_multi_scale(image, confidence_threshold=confidence_threshold)
        
        if preprocess:
            image = self._preprocess_image(image)
        
        try:
            # 使用PaddleOCR进行文字检测和识别
            result = self.ocr_engine.ocr(image, cls=False)
            
            if result is None:
                self.logger.warning("OCR检测结果为空")
                return []
            
            # 解析结果
            text_results = []
            for line in result:
                if line is None:
                    continue
                
                for word_info in line:
                    if word_info is None:
                        continue
                    
                    bbox = word_info[0]
                    text, confidence = word_info[1]
                    
                    if confidence >= confidence_threshold:
                        text_results.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
            
            # 按置信度排序
            text_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.logger.info(f"检测到 {len(text_results)} 个文字区域")
            return text_results
            
        except Exception as e:
            self.logger.error(f"OCR检测失败: {e}")
            return []
    
    def detect_text_multi_scale(self, image: np.ndarray, scales: List[float] = None, 
                               confidence_threshold: float = 0.3) -> List[Dict]:
        """
        多尺度文字检测，解决密集文字识别不完整问题
        
        Args:
            image: 输入图像
            scales: 缩放比例列表，默认为[1.0, 0.8, 1.2]
            confidence_threshold: 置信度阈值
            
        Returns:
            合并后的文字检测结果
        """
        if scales is None:
            scales = [1.0, 0.8, 1.2]  # 原始尺寸、缩小、放大
        
        all_results = []
        
        for scale in scales:
            # 缩放图像
            if scale != 1.0:
                h, w = image.shape[:2]
                new_w = int(w * scale)
                new_h = int(h * scale)
                scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_image = image
            
            # 在当前尺度下检测文字（避免递归调用，设置use_multi_scale=False）
            scale_results = self.detect_text(scaled_image, confidence_threshold=confidence_threshold, use_multi_scale=False)
            
            # 调整坐标到原图坐标系
            if scale != 1.0:
                for result in scale_results:
                    bbox = result['bbox']
                    adjusted_bbox = []
                    for point in bbox:
                        adjusted_bbox.append([
                            int(point[0] / scale),
                            int(point[1] / scale)
                        ])
                    result['bbox'] = adjusted_bbox
            
            all_results.extend(scale_results)
            self.logger.debug(f"尺度 {scale} 检测到 {len(scale_results)} 个文字")
        
        # 去重：合并相同位置的检测结果
        merged_results = self._merge_duplicate_results(all_results)
        self.logger.info(f"多尺度检测完成，总计检测到 {len(merged_results)} 个文字区域")
        
        return merged_results
    
    def _merge_duplicate_results(self, results: List[Dict], iou_threshold: float = 0.8) -> List[Dict]:
        """
        合并重复的检测结果 - 极低要求版本，降低去重要求
        
        Args:
            results: 检测结果列表
            iou_threshold: IoU阈值，高于此值认为重复（提高阈值，减少去重）
            
        Returns:
            去重后的结果列表
        """
        if not results:
            return []
        
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        used_indices = set()
        
        for i, result in enumerate(results):
            if i in used_indices:
                continue
            
            # 计算当前结果的边界框
            bbox_i = result['bbox']
            x_coords = [p[0] for p in bbox_i]
            y_coords = [p[1] for p in bbox_i]
            x1_i, x2_i = min(x_coords), max(x_coords)
            y1_i, y2_i = min(y_coords), max(y_coords)
            
            # 查找重叠的结果
            similar_indices = [i]
            for j in range(i+1, len(results)):
                if j in used_indices:
                    continue
                
                bbox_j = results[j]['bbox']
                x_coords_j = [p[0] for p in bbox_j]
                y_coords_j = [p[1] for p in bbox_j]
                x1_j, x2_j = min(x_coords_j), max(x_coords_j)
                y1_j, y2_j = min(y_coords_j), max(y_coords_j)
                
                # 计算IoU
                inter_x1 = max(x1_i, x1_j)
                inter_y1 = max(y1_i, y1_j)
                inter_x2 = min(x2_i, x2_j)
                inter_y2 = min(y2_i, y2_j)
                
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area_i = (x2_i - x1_i) * (y2_i - y1_i)
                area_j = (x2_j - x1_j) * (y2_j - y1_j)
                
                iou = inter_area / (area_i + area_j - inter_area) if (area_i + area_j - inter_area) > 0 else 0
                
                if iou > iou_threshold:
                    similar_indices.append(j)
            
            # 合并相似结果，保留置信度最高的
            if len(similar_indices) > 1:
                best_result = max([results[idx] for idx in similar_indices], 
                                 key=lambda x: x['confidence'])
                merged.append(best_result)
                used_indices.update(similar_indices)
            else:
                merged.append(result)
                used_indices.add(i)
        
        return merged

    def detect_specific_text(self, image: np.ndarray, target_texts: List[str], 
                           confidence_threshold: float = 0.7, use_multi_scale: bool = True) -> List[Dict]:
        """
        检测特定文字内容
        
        Args:
            image: 输入图像
            target_texts: 目标文字列表
            confidence_threshold: 置信度阈值
            use_multi_scale: 是否使用多尺度检测
            
        Returns:
            匹配到的文字信息列表
        """
        if use_multi_scale:
            all_results = self.detect_text_multi_scale(image, confidence_threshold=0.3)
        else:
            all_results = self.detect_text(image, confidence_threshold=0.3)
        
        matched_results = []
        for result in all_results:
            text = result['text']
            confidence = result['confidence']
            
            # 检查是否匹配目标文字
            for target in target_texts:
                if target in text and confidence >= confidence_threshold:
                    matched_results.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': result['bbox'],
                        'target': target,
                        'exact_match': text == target
                    })
        
        return matched_results
    
    def get_text_center(self, bbox: List[List[float]]) -> Tuple[int, int]:
        """
        计算文字区域的中心点坐标
        
        Args:
            bbox: 文字边界框，格式为[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            
        Returns:
            中心点坐标(x, y)
        """
        if not bbox or len(bbox) < 4:
            return (0, 0)
        
        # 计算边界框的中心点
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        center_x = int(sum(x_coords) / len(x_coords))
        center_y = int(sum(y_coords) / len(y_coords))
        
        return (center_x, center_y)
    
    def is_available(self) -> bool:
        """检查OCR功能是否可用"""
        return self.is_initialized and self.ocr_engine is not None
    
    def __del__(self):
        """析构函数，清理资源"""
        if self.ocr_engine:
            # PaddleOCR会自动清理资源
            pass


# 测试函数
def test_ocr_detector():
    """测试OCR检测器"""
    # 创建一个简单的测试图像
    test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    
    # 添加一些测试文字（这里只是模拟，实际需要真实文字图像）
    cv2.putText(test_image, "测试文字", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 创建OCR检测器 - 使用单例模式
    detector = OCRDetector.get_instance(lang='ch', use_gpu=False)
    
    if detector.is_available():
        print("OCR检测器可用")
        
        # 测试文字检测
        results = detector.detect_text(test_image)
        print(f"检测到 {len(results)} 个文字区域")
        
        for i, result in enumerate(results):
            print(f"文字 {i+1}: {result['text']}, 置信度: {result['confidence']:.2f}")
    else:
        print("OCR检测器不可用")


if __name__ == "__main__":
    test_ocr_detector()