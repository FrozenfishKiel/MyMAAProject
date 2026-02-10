"""
图像处理器模块 - 负责游戏图像的预处理和增强

目的：
1. 实现专业的图像预处理功能
2. 支持多种图像增强技术
3. 提供图像质量评估和优化
4. 支持模板匹配和特征提取

包含：
- 图像预处理（灰度化、去噪、增强等）
- 图像质量评估
- 模板匹配功能
- 特征提取和边缘检测
- 图像对比度增强
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Union
from enum import Enum


class ImageProcessingMode(Enum):
    """图像处理模式枚举"""
    BASIC = "basic"        # 基础预处理
    OCR_OPTIMIZED = "ocr"  # OCR优化处理
    OBJECT_DETECTION = "object"  # 目标检测优化
    TEMPLATE_MATCHING = "template"  # 模板匹配优化


class ImageProcessor:
    """图像处理器类 - 负责游戏图像的预处理和增强"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化图像处理器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化OpenCV参数
        self._setup_opencv_params()
        
        self.logger.info("图像处理器初始化完成")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            # 基础预处理参数
            'resize_width': 1920,
            'resize_height': 1080,
            'normalize_range': [0, 255],
            
            # 去噪参数
            'gaussian_kernel_size': (5, 5),
            'gaussian_sigma': 0,
            'median_kernel_size': 3,
            
            # 增强参数
            'clahe_clip_limit': 2.0,
            'clahe_tile_grid_size': (8, 8),
            'gamma_correction': 1.2,
            
            # 二值化参数
            'binary_threshold': 127,
            'binary_max_value': 255,
            'adaptive_method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            'adaptive_block_size': 11,
            'adaptive_c': 2,
            
            # 边缘检测参数
            'canny_threshold1': 50,
            'canny_threshold2': 150,
            'canny_aperture_size': 3,
            
            # 形态学操作参数
            'morph_kernel_size': (3, 3),
            'morph_iterations': 1,
            
            # 模板匹配参数
            'template_match_method': cv2.TM_CCOEFF_NORMED,
            'template_match_threshold': 0.8,
        }
    
    def _setup_opencv_params(self):
        """设置OpenCV参数"""
        # 确保OpenCV后端可用
        try:
            cv2.setUseOptimized(True)
            self.logger.debug("OpenCV优化模式已启用")
        except Exception as e:
            self.logger.warning(f"OpenCV优化设置失败: {e}")
    
    def preprocess_image(self, image: np.ndarray, 
                        mode: ImageProcessingMode = ImageProcessingMode.BASIC) -> np.ndarray:
        """
        图像预处理主方法
        
        Args:
            image: 原始图像
            mode: 处理模式
            
        Returns:
            预处理后的图像
        """
        if image is None:
            self.logger.error("输入图像为空")
            return None
        
        try:
            # 根据模式选择处理流程
            if mode == ImageProcessingMode.BASIC:
                return self._basic_preprocessing(image)
            elif mode == ImageProcessingMode.OCR_OPTIMIZED:
                return self._ocr_optimized_preprocessing(image)
            elif mode == ImageProcessingMode.OBJECT_DETECTION:
                return self._object_detection_preprocessing(image)
            elif mode == ImageProcessingMode.TEMPLATE_MATCHING:
                return self._template_matching_preprocessing(image)
            else:
                self.logger.warning(f"未知处理模式: {mode}, 使用基础模式")
                return self._basic_preprocessing(image)
                
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            return image
    
    def _basic_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """基础图像预处理"""
        processed = image.copy()
        
        # 1. 图像尺寸调整
        processed = self._resize_image(processed)
        
        # 2. 转换为灰度图
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # 3. 直方图均衡化
        processed = cv2.equalizeHist(processed)
        
        # 4. 高斯模糊去噪
        processed = cv2.GaussianBlur(
            processed, 
            self.config['gaussian_kernel_size'], 
            self.config['gaussian_sigma']
        )
        
        return processed
    
    def _ocr_optimized_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """OCR优化预处理"""
        processed = image.copy()
        
        # 1. 图像尺寸调整
        processed = self._resize_image(processed)
        
        # 2. 转换为灰度图
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # 3. CLAHE对比度增强
        clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe_clip_limit'],
            tileGridSize=self.config['clahe_tile_grid_size']
        )
        processed = clahe.apply(processed)
        
        # 4. 中值滤波去噪
        processed = cv2.medianBlur(processed, self.config['median_kernel_size'])
        
        # 5. 伽马校正
        processed = self._gamma_correction(processed)
        
        return processed
    
    def _object_detection_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """目标检测优化预处理"""
        processed = image.copy()
        
        # 1. 图像尺寸调整
        processed = self._resize_image(processed)
        
        # 2. 保持彩色（目标检测通常需要颜色信息）
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # 3. 图像归一化
        processed = processed.astype(np.float32) / 255.0
        
        # 4. 高斯模糊去噪
        processed = cv2.GaussianBlur(
            processed, 
            self.config['gaussian_kernel_size'], 
            self.config['gaussian_sigma']
        )
        
        return processed
    
    def _template_matching_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """模板匹配优化预处理"""
        processed = image.copy()
        
        # 1. 图像尺寸调整
        processed = self._resize_image(processed)
        
        # 2. 转换为灰度图
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # 3. 边缘检测
        processed = cv2.Canny(
            processed, 
            self.config['canny_threshold1'], 
            self.config['canny_threshold2'],
            apertureSize=self.config['canny_aperture_size']
        )
        
        # 4. 形态学闭操作（连接边缘）
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            self.config['morph_kernel_size']
        )
        processed = cv2.morphologyEx(
            processed, 
            cv2.MORPH_CLOSE, 
            kernel, 
            iterations=self.config['morph_iterations']
        )
        
        return processed
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """图像尺寸调整"""
        height, width = image.shape[:2]
        target_width = self.config['resize_width']
        target_height = self.config['resize_height']
        
        # 如果图像尺寸与目标尺寸不同，进行缩放
        if width != target_width or height != target_height:
            return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def _gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """伽马校正"""
        gamma = self.config['gamma_correction']
        
        # 构建伽马查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        
        # 应用伽马校正
        return cv2.LUT(image, table)
    
    def binary_threshold(self, image: np.ndarray, 
                        method: str = "global") -> np.ndarray:
        """
        图像二值化
        
        Args:
            image: 输入图像
            method: 二值化方法（"global"或"adaptive"）
            
        Returns:
            二值化图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == "global":
            # 全局阈值二值化
            _, binary = cv2.threshold(
                gray, 
                self.config['binary_threshold'], 
                self.config['binary_max_value'], 
                cv2.THRESH_BINARY
            )
        elif method == "adaptive":
            # 自适应阈值二值化
            binary = cv2.adaptiveThreshold(
                gray,
                self.config['binary_max_value'],
                self.config['adaptive_method'],
                cv2.THRESH_BINARY,
                self.config['adaptive_block_size'],
                self.config['adaptive_c']
            )
        else:
            self.logger.warning(f"未知二值化方法: {method}, 使用全局阈值")
            _, binary = cv2.threshold(
                gray, 
                self.config['binary_threshold'], 
                self.config['binary_max_value'], 
                cv2.THRESH_BINARY
            )
        
        return binary
    
    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """边缘检测"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Canny边缘检测
        edges = cv2.Canny(
            gray, 
            self.config['canny_threshold1'], 
            self.config['canny_threshold2'],
            apertureSize=self.config['canny_aperture_size']
        )
        
        return edges
    
    def find_contours(self, image: np.ndarray) -> List:
        """查找轮廓"""
        # 二值化图像
        binary = self.binary_threshold(image, "global")
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
    
    def template_matching(self, source_image: np.ndarray, 
                         template_image: np.ndarray) -> Dict:
        """
        模板匹配
        
        Args:
            source_image: 源图像
            template_image: 模板图像
            
        Returns:
            匹配结果字典
        """
        # 预处理源图像和模板图像
        source_processed = self.preprocess_image(
            source_image, 
            ImageProcessingMode.TEMPLATE_MATCHING
        )
        template_processed = self.preprocess_image(
            template_image, 
            ImageProcessingMode.TEMPLATE_MATCHING
        )
        
        # 执行模板匹配
        result = cv2.matchTemplate(
            source_processed, 
            template_processed, 
            self.config['template_match_method']
        )
        
        # 查找最佳匹配位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 根据匹配方法确定最佳匹配
        if self.config['template_match_method'] in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            best_match_val = min_val
            best_match_loc = min_loc
        else:
            best_match_val = max_val
            best_match_loc = max_loc
        
        # 检查是否超过阈值
        is_matched = best_match_val >= self.config['template_match_threshold']
        
        return {
            'matched': is_matched,
            'confidence': float(best_match_val),
            'location': best_match_loc,
            'template_size': template_processed.shape[:2],
            'source_size': source_processed.shape[:2]
        }
    
    def color_space_conversion(self, image: np.ndarray, 
                              conversion: str) -> np.ndarray:
        """
        颜色空间转换
        
        Args:
            image: 输入图像
            conversion: 转换类型（"BGR2RGB", "BGR2HSV", "BGR2GRAY"等）
            
        Returns:
            转换后的图像
        """
        conversion_map = {
            "BGR2RGB": cv2.COLOR_BGR2RGB,
            "BGR2HSV": cv2.COLOR_BGR2HSV,
            "BGR2GRAY": cv2.COLOR_BGR2GRAY,
            "RGB2BGR": cv2.COLOR_RGB2BGR,
            "RGB2HSV": cv2.COLOR_RGB2HSV,
            "RGB2GRAY": cv2.COLOR_RGB2GRAY,
            "HSV2BGR": cv2.COLOR_HSV2BGR,
            "HSV2RGB": cv2.COLOR_HSV2RGB,
        }
        
        if conversion not in conversion_map:
            self.logger.warning(f"未知颜色空间转换: {conversion}")
            return image
        
        return cv2.cvtColor(image, conversion_map[conversion])
    
    def get_image_quality_metrics(self, image: np.ndarray) -> Dict:
        """
        获取图像质量指标
        
        Args:
            image: 输入图像
            
        Returns:
            质量指标字典
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 计算图像质量指标
        metrics = {
            'brightness': float(np.mean(gray)),  # 亮度
            'contrast': float(np.std(gray)),     # 对比度
            'sharpness': self._calculate_sharpness(gray),  # 锐度
            'noise_level': self._estimate_noise_level(gray),  # 噪声水平
            'entropy': self._calculate_entropy(gray),  # 信息熵
        }
        
        return metrics
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """计算图像锐度（使用拉普拉斯方差）"""
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return float(laplacian_var)
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """估计图像噪声水平"""
        # 使用中值滤波估计噪声
        median_filtered = cv2.medianBlur(image, 3)
        noise = image - median_filtered
        return float(np.std(noise))
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """计算图像信息熵"""
        # 计算直方图
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()  # 归一化
        
        # 计算信息熵
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy)


def test_image_processor():
    """图像处理器测试函数"""
    import os
    import sys
    
    # 添加项目根目录到路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 创建图像处理器
    processor = ImageProcessor()
    
    # 测试基础预处理
    basic_result = processor.preprocess_image(test_image, ImageProcessingMode.BASIC)
    print(f"基础预处理结果形状: {basic_result.shape}")
    
    # 测试OCR优化预处理
    ocr_result = processor.preprocess_image(test_image, ImageProcessingMode.OCR_OPTIMIZED)
    print(f"OCR优化预处理结果形状: {ocr_result.shape}")
    
    # 测试二值化
    binary_result = processor.binary_threshold(test_image)
    print(f"二值化结果形状: {binary_result.shape}")
    
    # 测试边缘检测
    edges_result = processor.edge_detection(test_image)
    print(f"边缘检测结果形状: {edges_result.shape}")
    
    # 测试图像质量评估
    quality_metrics = processor.get_image_quality_metrics(test_image)
    print(f"图像质量指标: {quality_metrics}")
    
    print("✅ 图像处理器测试完成")


if __name__ == "__main__":
    """图像处理器模块测试代码"""
    test_image_processor()