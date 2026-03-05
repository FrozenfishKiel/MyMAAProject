from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class TemplateMatchResult:
    label: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]


class TemplateMatcher:
    def __init__(self, controller: Any) -> None:
        """
        模板匹配识别器
        
        Args:
            controller: MaaFramework控制器
        """
        self._controller = controller
        self._client: Optional[Any] = None
    
    def load(self) -> None:
        """
        加载MaaFramework
        """
        import importlib
        
        maa = importlib.import_module("maa")
        try:
            version = maa.library.Library.version()
        except Exception as e:
            raise RuntimeError(f"MaaFramework Python runtime not available: {e}")
        
        self._client = maa
    
    def match(
        self,
        bgr_image: Any,
        template_path: str,
        threshold: float = 0.7,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[TemplateMatchResult]:
        """
        模板匹配
        
        Args:
            bgr_image: BGR图像
            template_path: 模板图像路径
            threshold: 匹配阈值（0-1），默认0.7
            roi: 感兴趣区域（x1, y1, x2, y2），默认None（全图）
        
        Returns:
            模板匹配结果，如果未匹配到则返回None
        """
        if self._client is None:
            raise RuntimeError("MaaFramework not loaded")
        
        # 使用OpenCV进行模板匹配
        import cv2
        import numpy as np
        
        # 读取模板图像
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None:
            raise RuntimeError(f"Failed to load template image: {template_path}")
        
        # 如果指定了ROI，裁剪图像
        if roi is not None:
            x1, y1, x2, y2 = roi
            image = bgr_image[y1:y2, x1:x2]
        else:
            image = bgr_image
        
        # 转换为灰度图像
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # 模板匹配
        result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # 找到最佳匹配位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 如果匹配值小于阈值，返回None
        if max_val < threshold:
            return None
        
        # 计算匹配框的位置
        h, w = template_gray.shape
        top_left = max_loc
        
        # 如果指定了ROI，需要加上ROI的偏移
        if roi is not None:
            x1, y1, x2, y2 = roi
            top_left = (top_left[0] + x1, top_left[1] + y1)
        
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        return TemplateMatchResult(
            label=template_path,
            confidence=float(max_val),
            box_xyxy=(top_left[0], top_left[1], bottom_right[0], bottom_right[1])
        )
    
    def match_all(
        self,
        bgr_image: Any,
        template_paths: List[str],
        threshold: float = 0.7,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[TemplateMatchResult]:
        """
        多模板匹配
        
        Args:
            bgr_image: BGR图像
            template_paths: 模板图像路径列表
            threshold: 匹配阈值（0-1），默认0.7
            roi: 感兴趣区域（x1, y1, x2, y2），默认None（全图）
        
        Returns:
            模板匹配结果列表
        """
        results: List[TemplateMatchResult] = []
        
        for template_path in template_paths:
            result = self.match(bgr_image, template_path, threshold, roi)
            if result is not None:
                results.append(result)
        
        # 按置信度从高到低排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
