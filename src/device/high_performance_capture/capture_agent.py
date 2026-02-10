"""
智能捕获代理 - 基于MAA技术方案实现

功能：
1. 自动性能测试选择最优捕获方法
2. 支持手动切换捕获方法
3. 提供性能监控和统计信息
4. 实现失败重试和降级机制

基于MAA的ScreencapAgent.cpp实现
"""

import time
import logging
from typing import Optional, Dict, List, Tuple
import threading

import cv2
import numpy as np

from ..adb_controller import ADBController
from ..config import DeviceConfig


class CaptureAgent:
    """智能捕获代理 - 基于MAA的ScreencapAgent实现"""
    
    def __init__(self, adb_controller: ADBController, config: DeviceConfig):
        """
        初始化智能捕获代理
        
        Args:
            adb_controller: ADB控制器实例
            config: 设备配置对象
        """
        self.adb = adb_controller
        self.config = config
        self.logger = logging.getLogger("capture_agent")
        
        # 捕获方法字典
        self.capture_units: Dict[str, object] = {}
        self.active_unit: Optional[object] = None
        self.active_method: str = "unknown"
        
        # 性能统计
        self.performance_stats: Dict[str, Dict] = {}
        self.capture_count = 0
        self.total_capture_time = 0.0
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 初始化捕获方法
        self._init_capture_units()
        
        # 执行性能测试选择最优方法
        self._speed_test()
        
        self.logger.info(f"智能捕获代理初始化完成，当前使用: {self.active_method}")
    
    def _init_capture_units(self) -> None:
        """初始化所有可用的捕获方法"""
        # 优先尝试MUMU SDK（最高性能）
        try:
            from .mumu_sdk import MumuSDKCapture
            self.capture_units['mumu_sdk'] = MumuSDKCapture(self.adb, self.config)
            self.logger.info("MUMU SDK捕获初始化成功")
        except Exception as e:
            self.logger.warning(f"MUMU SDK捕获初始化失败: {e}")
        
        # 尝试MinicapStream（次高性能）
        try:
            from .minicap_stream import MinicapStream
            self.capture_units['minicap_stream'] = MinicapStream(self.adb, self.config)
            self.logger.info("MinicapStream初始化成功")
        except Exception as e:
            self.logger.warning(f"MinicapStream初始化失败: {e}")
        
        # 如果所有高性能方法都失败，使用基础ADB方法作为备用
        if not self.capture_units:
            from .basic_adb_capture import BasicADBCapture
            self.capture_units['basic_adb'] = BasicADBCapture(self.adb, self.config)
            self.logger.warning("所有高性能方法失败，使用基础ADB方法")
    
    def _speed_test(self) -> None:
        """性能测试选择最优捕获方法 - 基于MAA的实现"""
        if not self.capture_units:
            self.logger.error("没有可用的捕获方法")
            return
        
        fastest_method = "unknown"
        fastest_time = float('inf')
        
        self.logger.info("开始性能测试选择最优捕获方法...")
        
        for method_name, capture_unit in self.capture_units.items():
            try:
                # 测试捕获性能
                start_time = time.time()
                screenshot = capture_unit.screencap()
                capture_time = time.time() - start_time
                
                if screenshot is not None:
                    # 记录性能数据
                    self.performance_stats[method_name] = {
                        'last_capture_time': capture_time,
                        'success_rate': 1.0,
                        'last_test_time': time.time()
                    }
                    
                    self.logger.info(f"方法 {method_name}: 耗时 {capture_time:.3f}秒")
                    
                    # 选择最快的方法
                    if capture_time < fastest_time:
                        fastest_time = capture_time
                        fastest_method = method_name
                else:
                    self.logger.warning(f"方法 {method_name}: 捕获失败")
                    self.performance_stats[method_name] = {
                        'last_capture_time': float('inf'),
                        'success_rate': 0.0,
                        'last_test_time': time.time()
                    }
                    
            except Exception as e:
                self.logger.error(f"方法 {method_name} 测试失败: {e}")
                self.performance_stats[method_name] = {
                    'last_capture_time': float('inf'),
                    'success_rate': 0.0,
                    'last_test_time': time.time()
                }
        
        if fastest_method != "unknown":
            self.active_method = fastest_method
            self.active_unit = self.capture_units[fastest_method]
            self.logger.info(f"性能测试完成，选择方法: {fastest_method}, 耗时: {fastest_time:.3f}秒")
        else:
            self.logger.error("所有捕获方法测试失败")
    
    def screencap(self) -> Optional[np.ndarray]:
        """
        执行屏幕捕获 - 使用当前最优方法
        
        Returns:
            屏幕截图图像或None
        """
        with self.lock:
            if self.active_unit is None:
                self.logger.error("没有活动的捕获单元")
                return None
            
            start_time = time.time()
            
            try:
                screenshot = self.active_unit.screencap()
                capture_time = time.time() - start_time
                
                # 更新性能统计
                self.capture_count += 1
                self.total_capture_time += capture_time
                
                if screenshot is not None:
                    # 更新方法性能统计
                    if self.active_method in self.performance_stats:
                        stats = self.performance_stats[self.active_method]
                        stats['last_capture_time'] = capture_time
                        stats['success_rate'] = (stats.get('success_rate', 0.0) * 0.9 + 0.1)
                    
                    return screenshot
                else:
                    # 捕获失败，尝试重新选择方法
                    self.logger.warning(f"方法 {self.active_method} 捕获失败，重新测试...")
                    self._speed_test()
                    
                    # 使用新方法重试
                    if self.active_unit:
                        return self.active_unit.screencap()
                    
            except Exception as e:
                self.logger.error(f"捕获过程中出现异常: {e}")
                # 重新测试方法
                self._speed_test()
        
        return None
    
    def get_performance_info(self) -> Dict:
        """
        获取性能信息
        
        Returns:
            包含性能信息的字典
        """
        with self.lock:
            avg_capture_time = (self.total_capture_time / self.capture_count 
                              if self.capture_count > 0 else 0)
            
            return {
                'active_method': self.active_method,
                'capture_count': self.capture_count,
                'average_capture_time': avg_capture_time,
                'total_capture_time': self.total_capture_time,
                'performance_stats': self.performance_stats.copy(),
                'available_methods': list(self.capture_units.keys())
            }
    
    def switch_method(self, method_name: str) -> bool:
        """
        手动切换捕获方法
        
        Args:
            method_name: 方法名称
            
        Returns:
            切换是否成功
        """
        with self.lock:
            if method_name in self.capture_units:
                self.active_method = method_name
                self.active_unit = self.capture_units[method_name]
                self.logger.info(f"手动切换到方法: {method_name}")
                return True
            else:
                self.logger.error(f"未知的捕获方法: {method_name}")
                return False
    
    def refresh_methods(self) -> bool:
        """
        刷新捕获方法列表并重新测试
        
        Returns:
            刷新是否成功
        """
        with self.lock:
            self.logger.info("刷新捕获方法列表...")
            
            # 重新初始化方法
            self.capture_units.clear()
            self._init_capture_units()
            
            # 重新测试性能
            self._speed_test()
            
            return self.active_unit is not None