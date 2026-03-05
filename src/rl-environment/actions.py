from __future__ import annotations

import random
from typing import Optional, Tuple, Dict, Any

from src.ai_plugins.yolo_recognizer import YoloRecognizer, Detection
from src.ai_plugins.template_matcher import TemplateMatcher
from src.rl_environment.green_highlight import find_green_highlight


class DeployOperatorActions:
    """
    部署干员的动作实现
    
    包含4个动作：
    1. 点击干员头像
    2. 拖拽到放置区域
    3. 调整方向
    4. 松手完成部署
    """
    
    def __init__(self, controller: Any, yolo_recognizer: YoloRecognizer, template_matcher: TemplateMatcher) -> None:
        """
        初始化部署干员动作
        
        Args:
            controller: MaaFramework控制器
            yolo_recognizer: YOLO识别器
            template_matcher: 模板匹配识别器
        """
        self._controller = controller
        self._yolo_recognizer = yolo_recognizer
        self._template_matcher = template_matcher
        
        # 部署区范围：999 x 193
        # 左上角坐标：(0, 806)
        # 右下角坐标：(999, 999)
        self._deployment_area = {
            "x1": 0,
            "y1": 806,
            "x2": 999,
            "y2": 999
        }
    
    def action1_click_operator_avatar(self) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        动作1：点击干员头像
        
        在部署区范围内随机点击，通过模板匹配识别干员信息血条
        
        Returns:
            (success, position): 是否成功点击干员头像，干员位置
        """
        # 在部署区范围内随机点击
        x = random.randint(self._deployment_area["x1"], self._deployment_area["x2"])
        y = random.randint(self._deployment_area["y1"], self._deployment_area["y2"])
        
        # 点击干员头像
        self._controller.post_click(x, y).wait()
        
        # 截图
        image = self._controller.post_screencap().wait()
        
        # 使用模板匹配识别干员信息血条
        # 干员信息血条的位置是固定的，所以可以使用模板匹配
        hp_bar_template_path = "data/templates/hp_bar.png"
        hp_bar_result = self._template_matcher.match(image, hp_bar_template_path, threshold=0.7)
        
        if hp_bar_result is not None:
            # 成功点击干员头像
            return (True, (x, y))
        else:
            # 失败点击干员头像
            return (False, None)
    
    def action2_drag_to_deployment_area(self, start_position: Tuple[int, int]) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """
        动作2：拖拽到放置区域
        
        从干员位置拖拽到绿色高亮区域的中心点
        
        Args:
            start_position: 干员位置 (x, y)
        
        Returns:
            (success, end_position): 是否成功拖拽到放置区域，放置区域中心点
        """
        # 点击干员头像后，游戏自动标亮可放置的地点
        # 截图
        image = self._controller.post_screencap().wait()
        
        # 找到绿色高亮区域的中心点
        green_center = find_green_highlight(image)
        
        if green_center:
            # 拖拽到绿色高亮区域中心点
            start_x, start_y = start_position
            end_x, end_y = green_center
            self._controller.post_swipe(start_x, start_y, end_x, end_y, 500).wait()
            
            # 使用模板匹配识别点击取消UI
            # 点击取消UI的位置是固定的，所以可以使用模板匹配
            cancel_ui_template_path = "data/templates/cancel_ui.png"
            cancel_ui_result = self._template_matcher.match(image, cancel_ui_template_path, threshold=0.7)
            
            if cancel_ui_result is not None:
                # 成功拖拽到放置区域
                return (True, green_center)
            else:
                # 失败拖拽到放置区域
                return (False, None)
        else:
            # 未找到绿色高亮区域
            return (False, None)
    
    def action3_adjust_direction(self, center_position: Tuple[int, int]) -> Tuple[int, Tuple[int, int]]:
        """
        动作3：调整方向
        
        随机选择一个方向（上/下/左/右），滑动选择方向
        
        Args:
            center_position: 放置区域中心点 (x, y)
        
        Returns:
            (direction, end_position): 方向（0-3），滑动结束位置
                0: 上
                1: 下
                2: 左
                3: 右
        """
        # 随机选择一个方向
        direction = random.choice([0, 1, 2, 3])
        
        # 滑动距离
        distance = 50
        
        # 计算滑动结束位置
        x, y = center_position
        if direction == 0:  # 上
            end_position = (x, y - distance)
        elif direction == 1:  # 下
            end_position = (x, y + distance)
        elif direction == 2:  # 左
            end_position = (x - distance, y)
        else:  # 右
            end_position = (x + distance, y)
        
        # 滑动选择方向
        self._controller.post_swipe(x, y, end_position[0], end_position[1], 300).wait()
        
        return (direction, end_position)
    
    def action4_release_to_deploy(self, center_position: Tuple[int, int]) -> bool:
        """
        动作4：松手完成部署
        
        松手完成部署，通过干员血条判断是否成功部署
        
        Args:
            center_position: 放置区域中心点 (x, y)
        
        Returns:
            success: 是否成功部署
        """
        # 简单松手（不需要额外操作）
        # 等待一小段时间让游戏响应
        import time
        time.sleep(0.5)
        
        # 截图
        image = self._controller.post_screencap().wait()
        
        # 使用YOLO识别干员血条
        # 干员血条的位置在部署位置稍微偏移一点点
        detections = self._yolo_recognizer.detect(image, conf=0.25)
        operator_hp_bar_detected = any(d.label == "operator_hp_bar" for d in detections)
        
        if operator_hp_bar_detected:
            # 成功部署
            return True
        else:
            # 失败部署
            return False
    
    def deploy_operator(self) -> Dict[str, Any]:
        """
        完整的部署干员流程
        
        Returns:
            result: 部署结果
                - success: 是否成功部署
                - action1_success: 动作1是否成功
                - action2_success: 动作2是否成功
                - action3_direction: 动作3选择的方向
                - action4_success: 动作4是否成功
                - operator_position: 干员位置
                - deployment_position: 放置区域中心点
        """
        result = {
            "success": False,
            "action1_success": False,
            "action2_success": False,
            "action3_direction": None,
            "action4_success": False,
            "operator_position": None,
            "deployment_position": None
        }
        
        # 动作1：点击干员头像
        action1_success, operator_position = self.action1_click_operator_avatar()
        result["action1_success"] = action1_success
        result["operator_position"] = operator_position
        
        if not action1_success:
            return result
        
        # 动作2：拖拽到放置区域
        action2_success, deployment_position = self.action2_drag_to_deployment_area(operator_position)
        result["action2_success"] = action2_success
        result["deployment_position"] = deployment_position
        
        if not action2_success:
            return result
        
        # 动作3：调整方向
        action3_direction, _ = self.action3_adjust_direction(deployment_position)
        result["action3_direction"] = action3_direction
        
        # 动作4：松手完成部署
        action4_success = self.action4_release_to_deploy(deployment_position)
        result["action4_success"] = action4_success
        result["success"] = action4_success
        
        return result
