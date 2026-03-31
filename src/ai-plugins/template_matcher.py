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
        threshold: float = 0.4,
        roi: Optional[Tuple[int, int, int, int]] = None,
        silent: bool = False,
        exact_scale: bool = False
    ) -> Optional[TemplateMatchResult]:
        """
        模板匹配

        Args:
            bgr_image: BGR图像
            template_path: 模板图像路径
            threshold: 匹配阈值（0-1），默认0.4
            roi: 感兴趣区域（x1, y1, x2, y2），默认None（全图）
            silent: 如果为 True，则不打印任何调试日志
            exact_scale: 如果为 True，则不进行缩放搜索，只按 1.0 原图比例匹配（速度更快且精准）
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

        # === 新增：保存裁剪后的图像供调试（覆盖式） ===
        try:
            from pathlib import Path
            import os
            # logs 目录在项目根目录下
            debug_dir = Path(__file__).resolve().parent.parent.parent / "logs" / "roi_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            # 提取模板名字作为文件名
            template_name_only = str(template_path).replace('\\', '/').split('/')[-1].split('.')[0]
            debug_img_path = str(debug_dir / f"{template_name_only}_roi.png")

            # 保存这张将被用于实际匹配的图
            cv2.imwrite(debug_img_path, image)
        except Exception as e:
            print(f"[TemplateMatcher] 保存 debug 图片失败: {e}")
        # =============================================

        # 转换前检查：如果目标图像比模板图像还小，直接匹配失败
        if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
            print(f"[TemplateMatcher] Warning: Template {template_path} is larger than the target image region. Match skipped.")
            return None

        # 转换为灰度图像
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # === 算法升级：多尺度特征金字塔匹配 (Multi-Scale Pyramids) ===
        best_max_val = -1.0
        best_max_loc = (0, 0)
        best_scale = 1.0
        best_h, best_w = template_gray.shape

        # 如果要求精确比例，只扫 1.0；否则进行多尺度扫描
        scales = [1.0] if exact_scale else np.arange(0.8, 1.25, 0.05)

        for scale in scales:
            # 缩放模板图
            resized_w = int(template_gray.shape[1] * scale)
            resized_h = int(template_gray.shape[0] * scale)

            if resized_w > image_gray.shape[1] or resized_h > image_gray.shape[0]:
                continue

            resized_template = cv2.resize(template_gray, (resized_w, resized_h))

            # 使用缩放后的模板进行匹配
            res = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # 记录得分最高的尺度
            if max_val > best_max_val:
                best_max_val = max_val
                best_max_loc = max_loc
                best_scale = scale
                best_h, best_w = resized_h, resized_w

        max_val = best_max_val
        max_loc = best_max_loc

        # 静默模式则跳过控制台输出，但写入日志文件（如果有专门的 log 对象的话）
        # 这里通过检查 sys.stdout 是否被重定向到 Logger 来判断
        import sys
        if not silent or hasattr(sys.stdout, 'log'):
            template_name = str(template_path).replace('\\', '/').split('/')[-1]
            log_msg = f"[TemplateMatcher] {template_name} -> max conf: {max_val:.3f} (needs {threshold}, optimal scale: {best_scale:.2f}x)\n"

            if silent and hasattr(sys.stdout, 'log'):
                # 如果是静默模式，且存在文件日志系统，只写文件，不输出到屏幕
                sys.stdout.log.write(log_msg)
                sys.stdout.log.flush()
            else:
                # 正常模式，调用 print (会被双重日志捕捉)
                print(log_msg, end='')

        # 如果最高匹配得分依然小于阈值，返回None
        if max_val < threshold:
            return None

        # 计算最终匹配框的位置
        h, w = best_h, best_w
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
        threshold: float = 0.4,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[TemplateMatchResult]:
        """
        多模板匹配
        
        Args:
            bgr_image: BGR图像
            template_paths: 模板图像路径列表
            threshold: 匹配阈值（0-1），默认0.4
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
