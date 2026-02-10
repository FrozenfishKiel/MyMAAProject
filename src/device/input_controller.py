"""
输入控制模块 - 负责复杂的输入操作和手势识别

目的：
1. 封装复杂的输入操作（长按、双击、多点触控等）
2. 提供手势识别和轨迹模拟功能
3. 实现输入序列和组合操作
4. 支持输入延迟和随机化处理
5. 提供输入操作的验证和反馈

包含：
- 复杂输入操作（长按、双击、多点触控）
- 手势识别和轨迹模拟
- 输入序列管理
- 输入验证和反馈
- 性能优化和错误处理
"""

import time
import random
import logging
import threading
from typing import Optional, Tuple, List, Dict, Any, Callable
from pathlib import Path
from enum import Enum

from .config import DeviceConfig, ErrorType, default_config
from .adb_controller import ADBController, KeyCodes


class InputType(Enum):
    """输入类型枚举"""
    TAP = "tap"                    # 点击
    LONG_PRESS = "long_press"      # 长按
    DOUBLE_TAP = "double_tap"      # 双击
    SWIPE = "swipe"                # 滑动
    MULTI_TOUCH = "multi_touch"   # 多点触控
    KEY_EVENT = "key_event"        # 按键事件


class GestureType(Enum):
    """手势类型枚举"""
    UP = "up"                      # 上滑
    DOWN = "down"                  # 下滑
    LEFT = "left"                  # 左滑
    RIGHT = "right"                # 右滑
    CIRCLE = "circle"              # 画圈
    ZIGZAG = "zigzag"              # Z字形


class InputController:
    """输入控制器类 - 管理复杂的输入操作和手势识别"""
    
    def __init__(self, adb_controller: ADBController, config: DeviceConfig = default_config):
        """
        初始化输入控制器
        
        Args:
            adb_controller: ADB控制器实例
            config: 设备配置对象
        """
        self.adb = adb_controller
        self.config = config
        self.logger = self._setup_logger()
        
        # 输入状态
        self.is_input_active = False
        self.input_queue = []
        self.input_thread: Optional[threading.Thread] = None
        
        # 性能统计
        self.input_count = 0
        self.total_input_time = 0.0
        self.last_input_time = 0.0
        
        # 输入验证回调
        self.verification_callbacks: Dict[str, Callable] = {}
        
        self.logger.info("输入控制器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("input_controller")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # 避免重复添加处理器
        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 控制台处理器
            if self.config.log_to_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            # 文件处理器
            if self.config.log_to_file:
                log_path = Path(self.config.log_file_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(
                    self.config.log_file_path,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def _apply_input_delay(self) -> None:
        """应用输入延迟"""
        if self.config.input_delay > 0:
            delay = self.config.input_delay + random.uniform(-0.1, 0.1)  # 添加随机性
            time.sleep(delay)
    
    def tap(self, x: int, y: int, verify: bool = False) -> bool:
        """
        点击屏幕指定位置
        
        Args:
            x: X坐标
            y: Y坐标
            verify: 是否验证点击结果
            
        Returns:
            点击是否成功
        """
        start_time = time.time()
        
        if not self.adb.is_connected():
            self.logger.error("设备未连接，无法执行点击操作")
            return False
        
        # 应用输入延迟
        self._apply_input_delay()
        
        success = self.adb.tap(x, y)
        
        # 更新性能统计
        input_time = time.time() - start_time
        self.input_count += 1
        self.total_input_time += input_time
        self.last_input_time = input_time
        
        if success:
            self.logger.info(f"点击成功: ({x}, {y}), 耗时: {input_time:.3f}秒")
            
            # 验证点击结果
            if verify:
                return self._verify_tap_result(x, y)
            
            return True
        else:
            self.logger.error(f"点击失败: ({x}, {y})")
            return False

    def click(self, x: int, y: int, verify: bool = False) -> bool:
        """
        点击屏幕指定位置（tap方法的别名）
        
        Args:
            x: X坐标
            y: Y坐标
            verify: 是否验证点击结果
            
        Returns:
            点击是否成功
        """
        return self.tap(x, y, verify)
    
    def long_press(self, x: int, y: int, duration: int = 1000) -> bool:
        """
        长按屏幕指定位置
        
        Args:
            x: X坐标
            y: Y坐标
            duration: 长按持续时间（毫秒）
            
        Returns:
            长按是否成功
        """
        if not self.adb.is_connected():
            self.logger.error("设备未连接，无法执行长按操作")
            return False
        
        # 应用输入延迟
        self._apply_input_delay()
        
        # 使用滑动操作模拟长按（起点和终点相同）
        success = self.adb.swipe(x, y, x, y, duration)
        
        if success:
            self.logger.info(f"长按成功: ({x}, {y}), 时长: {duration}ms")
        else:
            self.logger.error(f"长按失败: ({x}, {y})")
        
        return success
    
    def double_tap(self, x: int, y: int, interval: float = 0.2) -> bool:
        """
        双击屏幕指定位置
        
        Args:
            x: X坐标
            y: Y坐标
            interval: 两次点击的时间间隔（秒）
            
        Returns:
            双击是否成功
        """
        # 第一次点击
        if not self.tap(x, y):
            return False
        
        # 等待间隔
        time.sleep(interval)
        
        # 第二次点击
        return self.tap(x, y)
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, 
              duration: Optional[int] = None, steps: int = 10) -> bool:
        """
        从起点滑动到终点（支持多段滑动）
        
        Args:
            x1: 起点X坐标
            y1: 起点Y坐标
            x2: 终点X坐标
            y2: 终点Y坐标
            duration: 滑动持续时间（毫秒）
            steps: 滑动分段数
            
        Returns:
            滑动是否成功
        """
        if not self.adb.is_connected():
            self.logger.error("设备未连接，无法执行滑动操作")
            return False
        
        # 应用输入延迟
        self._apply_input_delay()
        
        if duration is None:
            duration = self.config.swipe_duration
        
        # 计算每段滑动的距离
        dx = (x2 - x1) / steps
        dy = (y2 - y1) / steps
        
        # 执行多段滑动（更平滑）
        current_x, current_y = x1, y1
        
        for i in range(steps):
            next_x = int(current_x + dx)
            next_y = int(current_y + dy)
            
            # 最后一段直接到终点
            if i == steps - 1:
                next_x, next_y = x2, y2
            
            segment_duration = duration // steps
            
            success = self.adb.swipe(int(current_x), int(current_y), next_x, next_y, segment_duration)
            
            if not success:
                self.logger.error(f"滑动分段失败: 第{i+1}段")
                return False
            
            current_x, current_y = next_x, next_y
            
            # 段间短暂延迟
            if i < steps - 1:
                time.sleep(0.01)
        
        self.logger.info(f"滑动成功: ({x1}, {y1}) -> ({x2}, {y2}), 分段: {steps}段")
        return True
    
    def multi_touch(self, points: List[Tuple[int, int, int]], duration: int = 500) -> bool:
        """
        多点触控操作
        
        Args:
            points: 触点列表 [(x, y, 持续时间), ...]
            duration: 总持续时间（毫秒）
            
        Returns:
            多点触控是否成功
        """
        if not self.adb.is_connected():
            self.logger.error("设备未连接，无法执行多点触控操作")
            return False
        
        # 应用输入延迟
        self._apply_input_delay()
        
        # 多点触控需要特殊处理，这里使用简化实现
        # 实际实现可能需要更复杂的ADB命令或第三方工具
        self.logger.warning("多点触控功能为简化实现，实际效果可能有限")
        
        # 依次执行每个触点的长按操作
        for i, (x, y, point_duration) in enumerate(points):
            success = self.long_press(x, y, point_duration)
            
            if not success:
                self.logger.error(f"多点触控第{i+1}个触点失败: ({x}, {y})")
                return False
            
            # 触点间短暂延迟
            if i < len(points) - 1:
                time.sleep(0.1)
        
        self.logger.info(f"多点触控成功: {len(points)}个触点")
        return True
    
    def key_event(self, keycode: int) -> bool:
        """
        发送按键事件
        
        Args:
            keycode: 按键代码
            
        Returns:
            按键是否成功
        """
        if not self.adb.is_connected():
            self.logger.error("设备未连接，无法执行按键操作")
            return False
        
        # 应用输入延迟
        self._apply_input_delay()
        
        success = self.adb.key_event(keycode)
        
        if success:
            self.logger.info(f"按键成功: {keycode}")
        else:
            self.logger.error(f"按键失败: {keycode}")
        
        return success
    
    def gesture_swipe(self, gesture: GestureType, start_x: int, start_y: int, 
                      distance: int = 200, duration: int = 500) -> bool:
        """
        执行手势滑动
        
        Args:
            gesture: 手势类型
            start_x: 起点X坐标
            start_y: 起点Y坐标
            distance: 滑动距离
            duration: 滑动持续时间
            
        Returns:
            手势执行是否成功
        """
        end_x, end_y = start_x, start_y
        
        # 根据手势类型计算终点坐标
        if gesture == GestureType.UP:
            end_y = start_y - distance
        elif gesture == GestureType.DOWN:
            end_y = start_y + distance
        elif gesture == GestureType.LEFT:
            end_x = start_x - distance
        elif gesture == GestureType.RIGHT:
            end_x = start_x + distance
        elif gesture == GestureType.CIRCLE:
            # 画圈手势（简化实现）
            return self._execute_circle_gesture(start_x, start_y, distance, duration)
        elif gesture == GestureType.ZIGZAG:
            # Z字形手势（简化实现）
            return self._execute_zigzag_gesture(start_x, start_y, distance, duration)
        
        # 确保坐标在屏幕范围内
        resolution = self.adb.get_screen_resolution()
        if resolution:
            width, height = resolution
            end_x = max(0, min(end_x, width - 1))
            end_y = max(0, min(end_y, height - 1))
        
        return self.swipe(start_x, start_y, end_x, end_y, duration)
    
    def _execute_circle_gesture(self, center_x: int, center_y: int, 
                               radius: int, duration: int) -> bool:
        """执行画圈手势"""
        # 简化实现：使用8个点模拟圆形
        points = []
        steps = 8
        
        for i in range(steps):
            angle = 2 * 3.14159 * i / steps
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            points.append((x, y))
        
        # 连接起点和终点形成闭环
        points.append(points[0])
        
        # 执行多段滑动
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            segment_duration = duration // len(points)
            
            if not self.swipe(x1, y1, x2, y2, segment_duration, steps=3):
                return False
        
        return True
    
    def _execute_zigzag_gesture(self, start_x: int, start_y: int, 
                               distance: int, duration: int) -> bool:
        """执行Z字形手势"""
        # Z字形手势：右→右下→右→右上→右
        segment_distance = distance // 4
        
        points = [
            (start_x, start_y),
            (start_x + segment_distance, start_y),
            (start_x + segment_distance * 2, start_y + segment_distance),
            (start_x + segment_distance * 3, start_y),
            (start_x + segment_distance * 4, start_y)
        ]
        
        # 执行多段滑动
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            segment_duration = duration // (len(points) - 1)
            
            if not self.swipe(x1, y1, x2, y2, segment_duration, steps=3):
                return False
        
        return True
    
    def execute_input_sequence(self, sequence: List[Dict[str, Any]]) -> bool:
        """
        执行输入序列
        
        Args:
            sequence: 输入序列 [{"type": "tap", "x": 100, "y": 200}, ...]
            
        Returns:
            序列执行是否成功
        """
        self.logger.info(f"开始执行输入序列，共{len(sequence)}个操作")
        
        for i, operation in enumerate(sequence):
            op_type = operation.get("type", "")
            
            try:
                if op_type == "tap":
                    success = self.tap(operation["x"], operation["y"])
                elif op_type == "long_press":
                    success = self.long_press(operation["x"], operation["y"], 
                                            operation.get("duration", 1000))
                elif op_type == "double_tap":
                    success = self.double_tap(operation["x"], operation["y"], 
                                            operation.get("interval", 0.2))
                elif op_type == "swipe":
                    success = self.swipe(operation["x1"], operation["y1"], 
                                       operation["x2"], operation["y2"], 
                                       operation.get("duration"))
                elif op_type == "key_event":
                    success = self.key_event(operation["keycode"])
                else:
                    self.logger.error(f"未知操作类型: {op_type}")
                    return False
                
                if not success:
                    self.logger.error(f"序列第{i+1}个操作失败: {operation}")
                    return False
                
                # 操作间延迟
                if i < len(sequence) - 1:
                    time.sleep(operation.get("delay", 0.5))
                    
            except Exception as e:
                self.logger.error(f"序列第{i+1}个操作执行异常: {e}")
                return False
        
        self.logger.info("输入序列执行完成")
        return True
    
    def _verify_tap_result(self, x: int, y: int) -> bool:
        """
        验证点击结果（简化实现）
        
        Args:
            x: 点击X坐标
            y: 点击Y坐标
            
        Returns:
            验证是否通过
        """
        # 这里可以添加更复杂的验证逻辑
        # 例如：截图分析点击后的界面变化
        # 当前为简化实现，直接返回True
        
        self.logger.debug(f"点击验证: ({x}, {y}) - 简化验证通过")
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        avg_time = self.total_input_time / self.input_count if self.input_count > 0 else 0
        
        return {
            "total_inputs": self.input_count,
            "total_time": self.total_input_time,
            "average_time": avg_time,
            "last_input_time": self.last_input_time
        }
    
    def reset_stats(self) -> None:
        """重置性能统计"""
        self.input_count = 0
        self.total_input_time = 0.0
        self.last_input_time = 0.0


# 导入numpy用于数学计算
import numpy as np


if __name__ == "__main__":
    """输入控制器测试代码"""
    # 创建ADB控制器和输入控制器实例
    adb = ADBController()
    input_ctrl = InputController(adb)
    
    try:
        # 连接设备
        if adb.connect_device():
            print("✅ 设备连接成功")
            
            # 获取屏幕分辨率
            resolution = adb.get_screen_resolution()
            if resolution:
                width, height = resolution
                print(f"屏幕分辨率: {width}x{height}")
                
                # 测试点击操作
                center_x, center_y = width // 2, height // 2
                print(f"测试点击屏幕中心: ({center_x}, {center_y})")
                # input_ctrl.tap(center_x, center_y)  # 注释掉实际操作，避免误操作
                
                # 测试长按操作
                print(f"测试长按屏幕中心: ({center_x}, {center_y})")
                # input_ctrl.long_press(center_x, center_y, 500)  # 注释掉实际操作
                
                # 测试滑动操作
                start_x, start_y = center_x - 100, center_y
                end_x, end_y = center_x + 100, center_y
                print(f"测试水平滑动: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
                # input_ctrl.swipe(start_x, start_y, end_x, end_y)  # 注释掉实际操作
                
                # 测试手势操作
                print("测试上滑手势")
                # input_ctrl.gesture_swipe(GestureType.UP, center_x, center_y)  # 注释掉实际操作
                
                # 测试输入序列
                sequence = [
                    {"type": "tap", "x": center_x - 50, "y": center_y, "delay": 0.5},
                    {"type": "tap", "x": center_x + 50, "y": center_y, "delay": 0.5},
                    {"type": "swipe", "x1": center_x, "y1": center_y - 50, 
                     "x2": center_x, "y2": center_y + 50, "delay": 1.0}
                ]
                print("测试输入序列（3个操作）")
                # input_ctrl.execute_input_sequence(sequence)  # 注释掉实际操作
                
                # 获取性能统计
                stats = input_ctrl.get_performance_stats()
                print(f"性能统计: {stats}")
            
            # 断开连接
            adb.disconnect_device()
            print("✅ 设备断开成功")
        else:
            print("❌ 设备连接失败")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")