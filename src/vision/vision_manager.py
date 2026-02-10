"""
视觉识别管理器模块
整合OCR文字识别、图像处理等功能，为游戏自动化提供视觉分析能力
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# 导入自定义模块
from .detector.ocr_detector import OCRDetector
from .detector.yolo_detector import YOLODetector
from .processor.ocr_processor import OCRProcessor
from .processor.image_processor import ImageProcessor, ImageProcessingMode


class VisionManager:
    """视觉识别管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化视觉识别管理器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.ocr_detector = None
        self.yolo_detector = None
        self.ocr_processor = None
        self.image_processor = None
        
        # 状态标志
        self.is_initialized = False
        
        # 初始化视觉组件
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化各个视觉组件"""
        try:
            # 获取配置参数
            ocr_config = self.config.get('ocr', {})
            use_gpu = ocr_config.get('use_gpu', False)
            lang = ocr_config.get('lang', 'ch')
            model_dir = ocr_config.get('model_dir')
            
            # 初始化OCR检测器 - 使用单例模式
            self.ocr_detector = OCRDetector.get_instance(
                lang=lang,
                use_gpu=use_gpu,
                model_dir=model_dir
            )
            
            # 初始化OCR处理器
            processor_config = self.config.get('processor', {})
            confidence_threshold = processor_config.get('confidence_threshold', 0.6)
            similarity_threshold = processor_config.get('similarity_threshold', 0.8)
            
            self.ocr_processor = OCRProcessor(
                confidence_threshold=confidence_threshold,
                similarity_threshold=similarity_threshold
            )
            
            # 初始化YOLO检测器
            yolo_config = self.config.get('yolo', {})
            model_path = yolo_config.get('model_path', 'yolov8n.pt')
            use_gpu = yolo_config.get('use_gpu', True)  # 默认启用GPU
            conf_threshold = yolo_config.get('confidence_threshold', 0.5)
            iou_threshold = yolo_config.get('iou_threshold', 0.5)
            
            self.logger.info(f"[DEBUG] 正在初始化YOLO检测器...")
            self.logger.info(f"[DEBUG] YOLO配置 - 模型路径: {model_path}")
            self.logger.info(f"[DEBUG] YOLO配置 - 使用GPU: {use_gpu}")
            self.logger.info(f"[DEBUG] YOLO配置 - 置信度阈值: {conf_threshold}")
            self.logger.info(f"[DEBUG] YOLO配置 - IOU阈值: {iou_threshold}")
            
            # 初始化YOLO检测器 - 使用单例模式
            self.yolo_detector = YOLODetector.get_instance(
                model_path=model_path,
                use_gpu=use_gpu,
                confidence_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
            # 检查YOLO检测器是否初始化成功
            if self.yolo_detector.is_initialized:
                self.logger.info("[OK] YOLO检测器初始化成功")
            else:
                self.logger.warning("[WARNING] YOLO检测器初始化失败")
            
            # 初始化图像处理器
            image_config = self.config.get('image_processing', {})
            self.image_processor = ImageProcessor(image_config)
            
            # 添加自定义关键词（可根据具体游戏扩展）
            self._setup_custom_keywords()
            
            self.is_initialized = True
            self.logger.info("视觉识别管理器初始化成功")
            
        except Exception as e:
            self.logger.error(f"视觉识别管理器初始化失败: {e}")
            self.is_initialized = False
    
    def _setup_custom_keywords(self):
        """设置自定义关键词"""
        if self.ocr_processor:
            # 这里可以添加特定游戏的关键词
            custom_keywords = self.config.get('custom_keywords', {})
            
            for category, keywords in custom_keywords.items():
                self.ocr_processor.add_custom_keywords(category, keywords)
    
    def analyze_screen(self, image: np.ndarray, 
                      analysis_type: str = 'all') -> Dict[str, Any]:
        """
        分析屏幕图像
        
        Args:
            image: 输入图像，BGR格式
            analysis_type: 分析类型，可选 'all', 'text', 'numbers', 'keywords'
            
        Returns:
            分析结果字典
        """
        if not self.is_initialized:
            self.logger.warning("视觉识别管理器未初始化")
            return {}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_type,
            'image_size': image.shape[:2] if image is not None else (0, 0)
        }
        
        try:
            # 图像预处理
            processed_image = self.image_processor.preprocess_image(
                image, 
                mode=ImageProcessingMode.OCR_OPTIMIZED
            )
            results['preprocessing_mode'] = 'OCR_OPTIMIZED'
            results['preprocessing_applied'] = True
            
            # OCR文字识别分析
            if analysis_type in ['all', 'text', 'numbers', 'keywords']:
                text_results = self._analyze_text(processed_image, analysis_type)
                results.update(text_results)
            
            # YOLO目标检测分析
            if analysis_type in ['all', 'objects', 'detection']:
                object_results = self._analyze_objects(processed_image, analysis_type)
                results.update(object_results)
            
            # 模板匹配分析
            if analysis_type in ['all', 'template', 'match']:
                template_results = self._analyze_templates(processed_image, analysis_type)
                results.update(template_results)
            
            self.logger.info(f"屏幕分析完成，类型: {analysis_type}")
            
        except Exception as e:
            self.logger.error(f"屏幕分析失败: {e}")
            results['error'] = str(e)
        
        return results
    
    def _analyze_text(self, image: np.ndarray, analysis_type: str) -> Dict[str, Any]:
        """分析图像中的文字"""
        text_results = {}
        
        # 检测文字
        ocr_detections = self.ocr_detector.detect_text(image)
        
        # 过滤结果
        filtered_results = self.ocr_processor.filter_results(ocr_detections)
        
        text_results['text_detections'] = filtered_results
        text_results['text_count'] = len(filtered_results)
        
        # 根据分析类型进行进一步处理
        if analysis_type in ['all', 'numbers']:
            # 提取数字
            number_results = self.ocr_processor.extract_numbers(filtered_results)
            text_results['numbers'] = number_results
            text_results['number_count'] = len(number_results)
        
        if analysis_type in ['all', 'keywords']:
            # 查找关键词
            keyword_results = self.ocr_processor.find_keywords(filtered_results)
            text_results['keywords'] = keyword_results
            
            # 统计关键词数量
            keyword_count = sum(len(results) for results in keyword_results.values())
            text_results['keyword_count'] = keyword_count
        
        # 按位置排序
        sorted_results = self.ocr_processor.sort_by_position(filtered_results, 'reading-order')
        text_results['sorted_text'] = [r['text'] for r in sorted_results]
        
        return text_results
    
    def _analyze_objects(self, image: np.ndarray, analysis_type: str) -> Dict[str, Any]:
        """分析图像中的目标物体"""
        object_results = {}
        
        if not self.yolo_detector or not self.yolo_detector.is_available():
            self.logger.warning("YOLO检测器不可用")
            return object_results
        
        try:
            # 确保图像是3通道RGB格式（YOLO需要）
            if len(image.shape) == 2:
                # 灰度图像转RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 1:
                # 单通道图像转RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                # 已经是多通道图像，确保是RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 检测目标物体
            detections = self.yolo_detector.detect_objects(image_rgb)
            object_results['object_detections'] = detections
            object_results['object_count'] = len(detections)
            
            # 按类别统计
            class_counts = {}
            for detection in detections:
                class_name = detection.get('class_name', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            object_results['class_counts'] = class_counts
            
            # 按置信度排序
            sorted_detections = sorted(detections, 
                                     key=lambda x: x.get('confidence', 0), 
                                     reverse=True)
            object_results['sorted_objects'] = sorted_detections
            
            # 提取特定类别的物体
            if analysis_type in ['all', 'specific']:
                # 可以在这里添加特定类别的检测逻辑
                pass
                
        except Exception as e:
            self.logger.error(f"目标物体分析失败: {e}")
            object_results['error'] = str(e)
        
        return object_results
    
    def _analyze_templates(self, image: np.ndarray, analysis_type: str) -> Dict[str, Any]:
        """分析图像中的模板匹配"""
        template_results = {}
        
        # 获取模板配置
        template_config = self.config.get('templates', {})
        templates = template_config.get('templates', [])
        
        if not templates:
            self.logger.info("未配置模板，跳过模板匹配分析")
            return template_results
        
        try:
            template_matches = []
            
            for template_info in templates:
                template_name = template_info.get('name', 'unknown')
                template_path = template_info.get('path')
                
                if not template_path or not os.path.exists(template_path):
                    self.logger.warning(f"模板文件不存在: {template_path}")
                    continue
                
                # 加载模板图像
                template_image = cv2.imread(template_path)
                if template_image is None:
                    self.logger.warning(f"无法加载模板图像: {template_path}")
                    continue
                
                # 执行模板匹配
                match_result = self.image_processor.template_matching(image, template_image)
                
                if match_result['matched']:
                    template_matches.append({
                        'template_name': template_name,
                        'template_path': template_path,
                        'confidence': match_result['confidence'],
                        'location': match_result['location'],
                        'template_size': match_result['template_size'],
                        'source_size': match_result['source_size']
                    })
            
            template_results['template_matches'] = template_matches
            template_results['template_count'] = len(template_matches)
            
            # 按置信度排序
            sorted_matches = sorted(template_matches, 
                                  key=lambda x: x['confidence'], 
                                  reverse=True)
            template_results['sorted_templates'] = sorted_matches
            
            self.logger.info(f"模板匹配分析完成，找到 {len(template_matches)} 个匹配")
            
        except Exception as e:
            self.logger.error(f"模板匹配分析失败: {e}")
            template_results['error'] = str(e)
        
        return template_results
    
    def find_ui_element(self, image: np.ndarray, 
                       element_type: str, 
                       element_params: Dict[str, Any]) -> List[Dict]:
        """
        查找UI元素
        
        Args:
            image: 输入图像
            element_type: 元素类型，如 'button', 'text', 'icon'
            element_params: 元素参数，如文字内容、图标特征等
            
        Returns:
            找到的元素信息列表
        """
        if not self.is_initialized:
            return []
        
        try:
            # 图像预处理
            processed_image = self.image_processor.preprocess_image(
                image, 
                mode=ImageProcessingMode.OCR_OPTIMIZED
            )
            
            if element_type == 'text':
                # 查找文字元素
                target_texts = element_params.get('texts', [])
                if not target_texts:
                    return []
                
                # 检测文字
                ocr_detections = self.ocr_detector.detect_text(processed_image)
                
                # 查找特定文字
                matched_results = self.ocr_processor.find_specific_text(
                    ocr_detections, target_texts)
                
                # 转换为元素格式
                elements = []
                for result in matched_results:
                    bbox = result.get('bbox', [])
                    if len(bbox) >= 4:
                        # 计算中心点
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        center_x = int(sum(x_coords) / len(x_coords))
                        center_y = int(sum(y_coords) / len(y_coords))
                        
                        elements.append({
                            'type': 'text',
                            'text': result['text'],
                            'confidence': result['confidence'],
                            'bbox': bbox,
                            'center': (center_x, center_y),
                            'target_text': result.get('target_text'),
                            'match_type': result.get('match_type')
                        })
                
                return elements
            
            elif element_type in ['icon', 'button', 'object']:
                # 使用YOLO检测器查找特定类别的UI元素
                if not self.yolo_detector or not self.yolo_detector.is_available():
                    self.logger.warning("YOLO检测器不可用")
                    return []
                
                # 获取目标类别
                target_classes = element_params.get('classes', [])
                confidence_threshold = element_params.get('confidence_threshold', 0.5)
                
                # 确保图像是3通道RGB格式（YOLO需要）
                if len(processed_image.shape) == 2 or processed_image.shape[2] == 1:
                    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                else:
                    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                # 检测目标物体
                detections = self.yolo_detector.detect_specific_classes(
                    processed_image_rgb, target_classes, confidence_threshold)
                
                # 转换为元素格式
                elements = []
                for detection in detections:
                    bbox = detection.get('bbox', [])
                    if len(bbox) >= 4:
                        # 计算中心点
                        center_x, center_y = self.yolo_detector.get_bbox_center(bbox)
                        
                        elements.append({
                            'type': element_type,
                            'class_name': detection['class_name'],
                            'confidence': detection['confidence'],
                            'bbox': bbox,
                            'center': (center_x, center_y),
                            'class_id': detection['class_id']
                        })
                
                return elements
            
            elif element_type == 'template':
                # 使用模板匹配查找UI元素
                template_name = element_params.get('template_name')
                template_path = element_params.get('template_path')
                
                if not template_path or not os.path.exists(template_path):
                    self.logger.warning(f"模板文件不存在: {template_path}")
                    return []
                
                # 加载模板图像
                template_image = cv2.imread(template_path)
                if template_image is None:
                    self.logger.warning(f"无法加载模板图像: {template_path}")
                    return []
                
                # 执行模板匹配
                match_result = self.image_processor.template_matching(processed_image, template_image)
                
                if not match_result['matched']:
                    return []
                
                # 转换为元素格式
                location = match_result['location']
                template_size = match_result['template_size']
                
                # 计算边界框
                x, y = location
                w, h = template_size
                bbox = [x, y, x + w, y + h]
                center_x = x + w // 2
                center_y = y + h // 2
                
                return [{
                    'type': 'template',
                    'template_name': template_name or 'unknown',
                    'template_path': template_path,
                    'confidence': match_result['confidence'],
                    'bbox': bbox,
                    'center': (center_x, center_y),
                    'location': location,
                    'template_size': template_size
                }]
            
            else:
                self.logger.warning(f"不支持的UI元素类型: {element_type}")
                return []
                
        except Exception as e:
            self.logger.error(f"UI元素查找失败: {e}")
            return []
    
    def check_game_state(self, image: np.ndarray, 
                        state_indicators: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        检查游戏状态
        
        Args:
            image: 输入图像
            state_indicators: 状态指示器配置，如果为None则使用默认配置
            
        Returns:
            游戏状态信息
        """
        if not self.is_initialized:
            return {'state': 'unknown', 'confidence': 0.0}
        
        try:
            # 使用默认状态指示器配置
            if state_indicators is None:
                state_indicators = self._get_default_state_indicators()
            
            # 图像预处理
            processed_image = self.image_processor.preprocess_image(
                image, 
                mode=ImageProcessingMode.OCR_OPTIMIZED
            )
            
            state_info = {
                'state': 'unknown',
                'confidence': 0.0,
                'indicators': {}
            }
            
            # 分析屏幕文字
            text_analysis = self._analyze_text(processed_image, 'keywords')
            
            # 检查各个状态指示器
            for state_name, indicators in state_indicators.items():
                confidence = 0.0
                matched_indicators = []
                
                # 检查文字指示器
                text_indicators = indicators.get('text', [])
                if text_indicators:
                    keyword_results = text_analysis.get('keywords', {})
                    
                    for indicator in text_indicators:
                        for category, results in keyword_results.items():
                            for result in results:
                                if indicator in result.get('text', ''):
                                    confidence += 0.3  # 每个匹配的文字增加置信度
                                    matched_indicators.append({
                                        'type': 'text',
                                        'indicator': indicator,
                                        'text': result['text']
                                    })
                                    break
                
                # 检查图像特征指示器（YOLO检测）
                image_indicators = indicators.get('image_features', [])
                if image_indicators and self.yolo_detector:
                    yolo_results = self.yolo_detector.detect(processed_image)
                    for feature in image_indicators:
                        for detection in yolo_results:
                            if feature in detection.get('class_name', ''):
                                confidence += 0.4  # 图像特征匹配权重更高
                                matched_indicators.append({
                                    'type': 'image',
                                    'indicator': feature,
                                    'class_name': detection['class_name'],
                                    'confidence': detection['confidence']
                                })
                
                # 更新状态信息
                if confidence > state_info['confidence']:
                    state_info['state'] = state_name
                    state_info['confidence'] = min(confidence, 1.0)
                    state_info['indicators'] = matched_indicators
            
            return state_info
            
        except Exception as e:
            self.logger.error(f"游戏状态检查失败: {e}")
            return {'state': 'error', 'confidence': 0.0, 'error': str(e)}
    
    def _get_default_state_indicators(self) -> Dict[str, Any]:
        """
        获取默认的游戏状态指示器配置
        
        Returns:
            默认状态指示器配置字典
        """
        return {
            'main_menu': {
                'text': ['开始游戏', '主界面', '登录', '设置', '退出'],
                'image_features': ['logo', 'menu_button', 'start_button']
            },
            'battle_prepare': {
                'text': ['编队', '选择关卡', '开始作战', '准备'],
                'image_features': ['team_selection', 'level_selection', 'start_battle']
            },
            'battle_deploy': {
                'text': ['部署', '干员', '开始行动', '撤退'],
                'image_features': ['deploy_area', 'operator_icon', 'action_button']
            },
            'battle_fighting': {
                'text': ['战斗', '敌人', '生命值', '技能'],
                'image_features': ['enemy', 'health_bar', 'skill_icon']
            },
            'battle_result': {
                'text': ['胜利', '失败', '结算', '奖励'],
                'image_features': ['victory', 'defeat', 'reward_screen']
            },
            'base_management': {
                'text': ['基建', '制造站', '贸易站', '宿舍'],
                'image_features': ['base_icon', 'manufacturing', 'trading']
            },
            'recruitment': {
                'text': ['公开招募', '标签', '时间', '确认'],
                'image_features': ['recruitment_icon', 'tag_selection']
            },
            'shop': {
                'text': ['商店', '购买', '货币', '商品'],
                'image_features': ['shop_icon', 'item_list']
            }
        }
    
    def save_debug_image(self, image: np.ndarray, 
                        analysis_results: Dict[str, Any], 
                        save_path: str):
        """
        保存调试图像，标注检测结果
        
        Args:
            image: 原始图像
            analysis_results: 分析结果
            save_path: 保存路径
        """
        if image is None:
            return
        
        try:
            # 创建副本
            debug_image = image.copy()
            
            # 标注文字检测结果
            text_detections = analysis_results.get('text_detections', [])
            for detection in text_detections:
                bbox = detection.get('bbox', [])
                text = detection.get('text', '')
                confidence = detection.get('confidence', 0)
                
                if len(bbox) >= 4:
                    # 绘制边界框
                    points = np.array(bbox, dtype=np.int32)
                    cv2.polylines(debug_image, [points], True, (0, 255, 0), 2)
                    
                    # 添加文字标签
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    center_x = int(sum(x_coords) / len(x_coords))
                    center_y = int(sum(y_coords) / len(y_coords))
                    
                    label = f"{text} ({confidence:.2f})"
                    cv2.putText(debug_image, label, (center_x, center_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 标注目标检测结果
            object_detections = analysis_results.get('object_detections', [])
            for detection in object_detections:
                bbox = detection.get('bbox', [])
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0)
                
                if len(bbox) >= 4:
                    # 绘制边界框（使用不同颜色）
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # 添加类别标签
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    label = f"{class_name} ({confidence:.2f})"
                    cv2.putText(debug_image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 保存图像
            cv2.imwrite(save_path, debug_image)
            self.logger.info(f"调试图像已保存: {save_path}")
            
        except Exception as e:
            self.logger.error(f"保存调试图像失败: {e}")
    
    def is_ready(self) -> bool:
        """检查视觉识别系统是否就绪"""
        return self.is_initialized and (
            self.ocr_detector is not None and self.ocr_detector.is_available()
        )
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态信息"""
        status = {
            'initialized': self.is_initialized,
            'ocr_available': False,
            'components': {}
        }
        
        if self.ocr_detector:
            status['ocr_available'] = self.ocr_detector.is_available()
            status['components']['ocr_detector'] = 'available'
        else:
            status['components']['ocr_detector'] = 'not_initialized'
        
        if self.ocr_processor:
            status['components']['ocr_processor'] = 'available'
        else:
            status['components']['ocr_processor'] = 'not_initialized'
        
        return status

    def load_models(self) -> bool:
        """
        加载模型（兼容性方法）
        
        注意：VisionManager在初始化时已经自动加载了所有模型
        此方法仅用于兼容性，总是返回True
        
        Returns:
            bool: 总是返回True
        """
        self.logger.info("模型加载方法被调用（兼容性模式）")
        return self.is_initialized


# 测试函数
def test_vision_manager():
    """测试视觉识别管理器"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试配置
    config = {
        'ocr': {
            'use_gpu': False,
            'lang': 'ch'
        },
        'processor': {
            'confidence_threshold': 0.6
        },
        'custom_keywords': {
            'test': ['测试', 'TEST']
        }
    }
    
    # 创建视觉管理器
    vision_manager = VisionManager(config)
    
    # 检查状态
    status = vision_manager.get_status()
    print(f"系统状态: {status}")
    
    if vision_manager.is_ready():
        print("视觉识别系统就绪")
        
        # 创建一个简单的测试图像
        test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "测试文字", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "100", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 测试屏幕分析
        results = vision_manager.analyze_screen(test_image, 'all')
        print(f"分析结果: {results}")
        
        # 测试UI元素查找
        elements = vision_manager.find_ui_element(test_image, 'text', {'texts': ['测试文字']})
        print(f"找到的UI元素: {elements}")
        
    else:
        print("视觉识别系统未就绪")


if __name__ == "__main__":
    test_vision_manager()