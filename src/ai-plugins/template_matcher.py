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
        
        # 读取模板图像 (尝试保留 Alpha 透明通道作为 MAA 掩码)
        template_raw = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        if template_raw is None:
            raise RuntimeError(f"Failed to load template image: {template_path}")

        # 分离 BGR 和 Alpha(Mask)
        if len(template_raw.shape) == 3 and template_raw.shape[2] == 4:
            template = template_raw[:, :, :3]
            template_mask = template_raw[:, :, 3]
        else:
            template = template_raw
            template_mask = None
        
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

        # 判断是否是需要区分颜色的模板，如果是，强制使用彩色图像(BGR)进行匹配以区分颜色
        # 目前：start_action（开始行动按钮）
        filename = str(template_path).lower()
        is_color_sensitive = "start_action" in filename
        is_star_template = "star" in filename

        if is_color_sensitive:
            # 彩色匹配
            image_to_match = image
            template_to_match = template
            best_h, best_w = template_to_match.shape[:2]
            match_method = cv2.TM_CCOEFF_NORMED
        elif is_star_template:
            # ============================================================
            # MAA 混合视觉架构：第一阶段【形状提取】
            # 使用灰度图进行稳定的轮廓匹配，无视颜色干扰，只要找到"三颗星星"的物理坐标框即可
            # ============================================================
            image_to_match = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            template_to_match = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            best_h, best_w = template_to_match.shape
            match_method = cv2.TM_CCOEFF_NORMED
        else:
            # 常规灰度匹配
            image_to_match = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            template_to_match = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            best_h, best_w = template_to_match.shape
            match_method = cv2.TM_CCOEFF_NORMED

        # === 算法升级：多尺度特征金字塔匹配 (Multi-Scale Pyramids) ===
        best_max_val = -1.0
        best_max_loc = (0, 0)
        best_scale = 1.0

        # 如果要求精确比例，只扫 1.0；否则进行多尺度扫描
        scales = [1.0] if exact_scale else np.arange(0.8, 1.25, 0.05)

        for scale in scales:
            # 缩放模板图
            if is_color_sensitive:
                resized_w = int(template_to_match.shape[1] * scale)
                resized_h = int(template_to_match.shape[0] * scale)
            elif is_star_template:
                resized_w = int(template_to_match.shape[1] * scale)
                resized_h = int(template_to_match.shape[0] * scale)
            else:
                resized_w = int(template_to_match.shape[1] * scale)
                resized_h = int(template_to_match.shape[0] * scale)

            if resized_w > image_to_match.shape[1] or resized_h > image_to_match.shape[0]:
                continue

            resized_template = cv2.resize(template_to_match, (resized_w, resized_h))

            # 使用缩放后的模板进行匹配
            res = cv2.matchTemplate(image_to_match, resized_template, match_method)

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
                # 同时加个特殊前缀，方便在日志里直接搜 "SILENT_LOG"
                sys.stdout.log.write(f"[SILENT_LOG] {log_msg}")
                sys.stdout.log.flush()
            else:
                # 正常模式，调用 print (会被双重日志捕捉)
                print(log_msg, end='')

        # 计算最终匹配框的位置
        h, w = best_h, best_w
        top_left = max_loc

        # ============================================================
        # MAA 混合视觉架构：第二阶段【二次颜色特征校验 (HSVCount)】
        # ============================================================
        if is_star_template and max_val >= 0.2: # 只要形状有 20% 像，就认为找到了目标框，开始数颜色
            # 把当前画面中，定位到的星星框切下来（使用彩色原图 bgr_image）
            # 注意需要加上 ROI 偏移以获取真实全图坐标
            global_x1 = top_left[0] + (roi[0] if roi else 0)
            global_y1 = top_left[1] + (roi[1] if roi else 0)
            target_crop = bgr_image[global_y1:global_y1+h, global_x1:global_x1+w]

            # 提取明日方舟特有的 "蓝/青色" 像素
            hsv_crop = cv2.cvtColor(target_crop, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([90, 40, 150])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv_crop, lower_blue, upper_blue)

            # 统计纯蓝像素点的数量 (每个像素点为 255)
            blue_pixel_count = cv2.countNonZero(blue_mask)

            # 我们需要对比的目标是我们要找的那个模板图 (template)
            # 同样统计模板图里的蓝像素数量作为标准参照物 (TP)
            hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            template_blue_mask = cv2.inRange(hsv_template, lower_blue, upper_blue)
            template_blue_count = cv2.countNonZero(template_blue_mask)

            if template_blue_count > 0:
                # MAA 核心逻辑：计算 F1-Score 或者重叠比例 (当前画面的蓝色量 / 模板规定的蓝色量)
                color_ratio = blue_pixel_count / template_blue_count

                # 对于 3星 和 2星，颜色比例必须严丝合缝
                if "3star" in filename or "2star" in filename:
                    # 如果蓝光像素达标 (大于 70%)，说明确实是这个星级！强行把形状得分和颜色比例综合
                    if color_ratio > 0.70:
                        max_val = 0.95 # 确信无疑，强行拉高置信度
                    else:
                        max_val = 0.1  # 蓝光不够，绝对是误判，强行压低置信度
                # 对于 0星，它不能有蓝色像素
                elif "0star" in filename:
                    if blue_pixel_count < 100: # 几乎没有蓝光
                        max_val = 0.95
                    else:
                        max_val = 0.1
            else:
                # 理论上不会走到这里，除非模板是纯黑图
                pass

        # 如果最高匹配得分依然小于阈值，返回None
        if max_val < threshold:
            return None
        
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
