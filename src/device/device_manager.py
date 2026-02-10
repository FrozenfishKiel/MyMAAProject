"""
设备管理器模块 - 设备控制模块的总控制器，整合所有子模块功能

目的：
1. 提供统一的设备管理接口
2. 整合ADB控制、屏幕截图、输入控制功能
3. 实现设备状态监控和健康检查
4. 提供任务调度和错误恢复机制
5. 支持多设备管理和配置切换

包含：
- 设备状态管理
- 模块整合和协调
- 任务调度系统
- 错误恢复机制
- 性能监控和统计
"""

import time
import threading
import logging
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from enum import Enum

from .config import DeviceConfig, ErrorType, default_config
from .adb_controller import ADBController, KeyCodes
from .screen_capture import ScreenCapture
from .input_controller import InputController, InputType, GestureType


class DeviceStatus(Enum):
    """设备状态枚举"""
    DISCONNECTED = "disconnected"      # 未连接
    CONNECTING = "connecting"          # 连接中
    CONNECTED = "connected"            # 已连接
    ERROR = "error"                    # 错误状态
    RECONNECTING = "reconnecting"      # 重连中


class TaskType(Enum):
    """任务类型枚举"""
    SCREENSHOT = "screenshot"          # 截图任务
    INPUT_OPERATION = "input_operation" # 输入操作
    GESTURE = "gesture"                # 手势操作
    SEQUENCE = "sequence"              # 操作序列
    HEALTH_CHECK = "health_check"      # 健康检查


class DeviceManager:
    """设备管理器类 - 整合所有设备控制功能"""
    
    def __init__(self, config: DeviceConfig = default_config):
        """
        初始化设备管理器
        
        Args:
            config: 设备配置对象
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # 设备状态
        self.status = DeviceStatus.DISCONNECTED
        self.last_error: Optional[str] = None
        self.connection_time: Optional[float] = None
        
        # 子模块实例
        self.adb_controller: Optional[ADBController] = None
        self.screen_capture: Optional[ScreenCapture] = None
        self.input_controller: Optional[InputController] = None
        
        # 任务管理
        self.task_queue: List[Dict[str, Any]] = []
        self.task_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # 性能统计
        self.operation_count = 0
        self.total_operation_time = 0.0
        self.success_count = 0
        self.failure_count = 0
        
        # 回调函数
        self.status_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        self.logger.info("设备管理器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("device_manager")
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
    
    def connect(self) -> bool:
        """
        连接设备并初始化所有子模块
        
        Returns:
            连接是否成功
        """
        self._update_status(DeviceStatus.CONNECTING)
        
        try:
            # 创建ADB控制器
            self.adb_controller = ADBController(self.config)
            
            # 连接设备
            if not self.adb_controller.connect_device():
                self._update_status(DeviceStatus.ERROR, "设备连接失败")
                return False
            
            # 创建屏幕截图模块
            self.screen_capture = ScreenCapture(self.adb_controller, self.config)
            
            # 创建输入控制器
            self.input_controller = InputController(self.adb_controller, self.config)
            
            # 更新状态
            self._update_status(DeviceStatus.CONNECTED)
            self.connection_time = time.time()
            
            # 启动任务处理线程
            self._start_task_processor()
            
            self.logger.info("✅ 设备管理器连接成功，所有子模块初始化完成")
            return True
            
        except Exception as e:
            self._update_status(DeviceStatus.ERROR, f"连接过程中出现异常: {e}")
            self.logger.error(f"❌ 设备管理器连接失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        断开设备连接并清理资源
        
        Returns:
            断开是否成功
        """
        self._stop_task_processor()
        
        if self.adb_controller:
            success = self.adb_controller.disconnect_device()
            
            if success:
                self._update_status(DeviceStatus.DISCONNECTED)
                self.logger.info("✅ 设备管理器断开成功")
            else:
                self._update_status(DeviceStatus.ERROR, "设备断开失败")
                self.logger.error("❌ 设备管理器断开失败")
            
            # 清理资源
            self.adb_controller = None
            self.screen_capture = None
            self.input_controller = None
            
            return success
        
        self._update_status(DeviceStatus.DISCONNECTED)
        return True
    
    def _update_status(self, status: DeviceStatus, error_msg: Optional[str] = None) -> None:
        """
        更新设备状态
        
        Args:
            status: 新状态
            error_msg: 错误信息（可选）
        """
        old_status = self.status
        self.status = status
        
        if error_msg:
            self.last_error = error_msg
        
        # 触发状态回调
        for callback in self.status_callbacks:
            try:
                callback(old_status, status, error_msg)
            except Exception as e:
                self.logger.warning(f"状态回调执行失败: {e}")
        
        self.logger.debug(f"设备状态更新: {old_status.value} -> {status.value}")
    
    def _start_task_processor(self) -> None:
        """启动任务处理线程"""
        if self.is_running:
            return
        
        self.is_running = True
        self.task_thread = threading.Thread(target=self._task_processor_loop, daemon=True)
        self.task_thread.start()
        self.logger.info("任务处理线程已启动")
    
    def _stop_task_processor(self) -> None:
        """停止任务处理线程"""
        self.is_running = False
        
        if self.task_thread and self.task_thread.is_alive():
            self.task_thread.join(timeout=5.0)
            
        self.task_thread = None
        self.logger.info("任务处理线程已停止")
    
    def _task_processor_loop(self) -> None:
        """任务处理循环"""
        while self.is_running:
            try:
                if self.task_queue:
                    task = self.task_queue.pop(0)
                    self._execute_task(task)
                else:
                    # 队列为空，短暂休眠
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"任务处理循环异常: {e}")
                time.sleep(1.0)
    
    def _execute_task(self, task: Dict[str, Any]) -> None:
        """
        执行单个任务
        
        Args:
            task: 任务数据
        """
        task_type = task.get("type", "")
        task_id = task.get("id", "unknown")
        
        self.logger.debug(f"开始执行任务: {task_id} ({task_type})")
        
        start_time = time.time()
        success = False
        
        try:
            if task_type == TaskType.SCREENSHOT.value:
                success = self._execute_screenshot_task(task)
            elif task_type == TaskType.INPUT_OPERATION.value:
                success = self._execute_input_task(task)
            elif task_type == TaskType.GESTURE.value:
                success = self._execute_gesture_task(task)
            elif task_type == TaskType.SEQUENCE.value:
                success = self._execute_sequence_task(task)
            elif task_type == TaskType.HEALTH_CHECK.value:
                success = self._execute_health_check_task(task)
            else:
                self.logger.error(f"未知任务类型: {task_type}")
                success = False
            
        except Exception as e:
            self.logger.error(f"任务执行异常: {task_id}, 错误: {e}")
            success = False
        
        # 更新统计
        self.operation_count += 1
        execution_time = time.time() - start_time
        self.total_operation_time += execution_time
        
        if success:
            self.success_count += 1
            self.logger.info(f"任务执行成功: {task_id}, 耗时: {execution_time:.3f}秒")
        else:
            self.failure_count += 1
            self.logger.error(f"任务执行失败: {task_id}, 耗时: {execution_time:.3f}秒")
    
    def _execute_screenshot_task(self, task: Dict[str, Any]) -> bool:
        """执行截图任务"""
        if not self.screen_capture:
            return False
        
        # 获取截图参数
        save_path = task.get("save_path")
        compress = task.get("compress", True)
        
        # 执行截图
        screenshot = self.screen_capture.capture_screenshot()
        
        if screenshot is not None:
            # 如果需要保存
            if save_path:
                import cv2
                cv2.imwrite(save_path, screenshot)
                self.logger.debug(f"截图已保存: {save_path}")
            
            # 触发回调
            callback = task.get("callback")
            if callback:
                try:
                    callback(screenshot)
                except Exception as e:
                    self.logger.warning(f"截图回调执行失败: {e}")
            
            return True
        
        return False
    
    def _execute_input_task(self, task: Dict[str, Any]) -> bool:
        """执行输入任务"""
        if not self.input_controller:
            return False
        
        operation = task.get("operation", {})
        op_type = operation.get("type", "")
        
        if op_type == InputType.TAP.value:
            return self.input_controller.tap(operation["x"], operation["y"])
        elif op_type == InputType.LONG_PRESS.value:
            return self.input_controller.long_press(operation["x"], operation["y"], 
                                                   operation.get("duration", 1000))
        elif op_type == InputType.DOUBLE_TAP.value:
            return self.input_controller.double_tap(operation["x"], operation["y"], 
                                                   operation.get("interval", 0.2))
        elif op_type == InputType.SWIPE.value:
            return self.input_controller.swipe(operation["x1"], operation["y1"], 
                                             operation["x2"], operation["y2"], 
                                             operation.get("duration"))
        elif op_type == InputType.KEY_EVENT.value:
            return self.input_controller.key_event(operation["keycode"])
        else:
            self.logger.error(f"未知输入操作类型: {op_type}")
            return False
    
    def _execute_gesture_task(self, task: Dict[str, Any]) -> bool:
        """执行手势任务"""
        if not self.input_controller:
            return False
        
        gesture = task.get("gesture", {})
        gesture_type = gesture.get("type", "")
        
        # 获取手势参数
        start_x = gesture.get("start_x", 0)
        start_y = gesture.get("start_y", 0)
        distance = gesture.get("distance", 200)
        duration = gesture.get("duration", 500)
        
        # 执行手势
        try:
            gesture_enum = GestureType(gesture_type)
            return self.input_controller.gesture_swipe(gesture_enum, start_x, start_y, 
                                                      distance, duration)
        except ValueError:
            self.logger.error(f"未知手势类型: {gesture_type}")
            return False
    
    def _execute_sequence_task(self, task: Dict[str, Any]) -> bool:
        """执行操作序列任务"""
        if not self.input_controller:
            return False
        
        sequence = task.get("sequence", [])
        return self.input_controller.execute_input_sequence(sequence)
    
    def _execute_health_check_task(self, task: Dict[str, Any]) -> bool:
        """执行健康检查任务"""
        # 检查设备连接状态
        if not self.adb_controller or not self.adb_controller.is_connected():
            self.logger.warning("健康检查: 设备未连接")
            return False
        
        # 检查屏幕分辨率
        resolution = self.adb_controller.get_screen_resolution()
        if not resolution:
            self.logger.warning("健康检查: 无法获取屏幕分辨率")
            return False
        
        # 检查截图功能
        if self.screen_capture:
            screenshot = self.screen_capture.capture_screenshot()
            if screenshot is None:
                self.logger.warning("健康检查: 截图功能异常")
                return False
        
        self.logger.debug("健康检查: 所有功能正常")
        return True
    
    def add_task(self, task: Dict[str, Any]) -> str:
        """
        添加任务到队列
        
        Args:
            task: 任务数据
            
        Returns:
            任务ID
        """
        # 生成任务ID
        import uuid
        task_id = str(uuid.uuid4())[:8]
        task["id"] = task_id
        
        # 添加到队列
        self.task_queue.append(task)
        self.logger.debug(f"任务已添加到队列: {task_id}")
        
        return task_id
    
    def capture_screenshot(self, save_path: Optional[str] = None, 
                          callback: Optional[Callable] = None) -> str:
        """
        添加截图任务
        
        Args:
            save_path: 保存路径（可选）
            callback: 回调函数（可选）
            
        Returns:
            任务ID
        """
        task = {
            "type": TaskType.SCREENSHOT.value,
            "save_path": save_path,
            "callback": callback
        }
        
        return self.add_task(task)
    
    def execute_input(self, operation: Dict[str, Any]) -> str:
        """
        添加输入操作任务
        
        Args:
            operation: 输入操作数据
            
        Returns:
            任务ID
        """
        task = {
            "type": TaskType.INPUT_OPERATION.value,
            "operation": operation
        }
        
        return self.add_task(task)
    
    def execute_gesture(self, gesture: Dict[str, Any]) -> str:
        """
        添加手势任务
        
        Args:
            gesture: 手势数据
            
        Returns:
            任务ID
        """
        task = {
            "type": TaskType.GESTURE.value,
            "gesture": gesture
        }
        
        return self.add_task(task)
    
    def execute_sequence(self, sequence: List[Dict[str, Any]]) -> str:
        """
        添加操作序列任务
        
        Args:
            sequence: 操作序列
            
        Returns:
            任务ID
        """
        task = {
            "type": TaskType.SEQUENCE.value,
            "sequence": sequence
        }
        
        return self.add_task(task)
    
    def health_check(self) -> str:
        """
        添加健康检查任务
        
        Returns:
            任务ID
        """
        task = {
            "type": TaskType.HEALTH_CHECK.value
        }
        
        return self.add_task(task)
    
    def get_status(self) -> Dict[str, Any]:
        """获取设备状态信息"""
        return {
            "status": self.status.value,
            "last_error": self.last_error,
            "connection_time": self.connection_time,
            "operation_count": self.operation_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "queue_size": len(self.task_queue),
            "is_running": self.is_running
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        avg_time = self.total_operation_time / self.operation_count if self.operation_count > 0 else 0
        success_rate = self.success_count / self.operation_count if self.operation_count > 0 else 0
        
        return {
            "total_operations": self.operation_count,
            "total_time": self.total_operation_time,
            "average_time": avg_time,
            "success_rate": success_rate,
            "success_count": self.success_count,
            "failure_count": self.failure_count
        }
    
    def register_status_callback(self, callback: Callable) -> None:
        """注册状态回调函数"""
        self.status_callbacks.append(callback)
        self.logger.debug("状态回调函数已注册")
    
    def register_error_callback(self, callback: Callable) -> None:
        """注册错误回调函数"""
        self.error_callbacks.append(callback)
        self.logger.debug("错误回调函数已注册")
    
    def wait_for_tasks(self, timeout: float = 30.0) -> bool:
        """
        等待所有任务完成
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            是否所有任务都完成
        """
        start_time = time.time()
        
        while self.task_queue:
            if time.time() - start_time > timeout:
                self.logger.warning("等待任务完成超时")
                return False
            
            time.sleep(0.1)
        
        return True
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()


if __name__ == "__main__":
    """设备管理器测试代码"""
    # 创建设备管理器实例
    device_manager = DeviceManager()
    
    try:
        # 连接设备
        if device_manager.connect():
            print("✅ 设备管理器连接成功")
            
            # 获取设备状态
            status = device_manager.get_status()
            print(f"设备状态: {status}")
            
            # 添加健康检查任务
            health_task_id = device_manager.health_check()
            print(f"健康检查任务已添加: {health_task_id}")
            
            # 等待任务完成
            if device_manager.wait_for_tasks(timeout=10.0):
                print("✅ 健康检查任务完成")
            
            # 获取性能统计
            stats = device_manager.get_performance_stats()
            print(f"性能统计: {stats}")
            
            # 断开连接
            device_manager.disconnect()
            print("✅ 设备管理器断开成功")
        else:
            print("❌ 设备管理器连接失败")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")