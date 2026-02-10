"""
屏幕截图模块 - 负责从MUMU模拟器捕获屏幕截图并进行图像处理

目的：
1. 实现高效的屏幕截图捕获功能
2. 支持PNG格式截图和适当压缩
3. 提供截图队列管理和频率控制
4. 实现图像预处理和优化功能
5. 支持截图缓存和性能优化

包含：
- 屏幕截图捕获
- 图像压缩和格式转换
- 截图队列管理
- 图像预处理功能
- 性能监控和优化
"""

import os
import time
import logging
import threading
from typing import Optional, Tuple, List, Union
from pathlib import Path
from queue import Queue, Empty
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from .config import DeviceConfig, ErrorType, default_config
from .adb_controller import ADBController


class ScreenCapture:
    """屏幕截图类 - 管理屏幕截图捕获和处理"""
    
    def __init__(self, adb_controller: ADBController, config: DeviceConfig = default_config):
        """
        初始化屏幕截图模块
        
        Args:
            adb_controller: ADB控制器实例
            config: 设备配置对象
        """
        self.adb = adb_controller
        self.adb_controller = adb_controller  # 为测试脚本添加别名属性
        self.config = config
        self.logger = self._setup_logger()
        
        # 截图队列
        self.screenshot_queue: Queue = Queue(maxsize=self.config.max_screenshot_queue_size)
        self.capture_thread: Optional[threading.Thread] = None
        self.is_capturing = False
        
        # 性能统计
        self.capture_count = 0
        self.total_capture_time = 0.0
        self.last_capture_time = 0.0
        
        # 图像缓存
        self.last_screenshot: Optional[np.ndarray] = None
        self.cache_timestamp = 0.0
        
        # 高性能捕获模块
        self.high_performance_capture = None
        if hasattr(self.config, 'high_performance_capture_enabled') and self.config.high_performance_capture_enabled:
            self._init_high_performance_capture()
        
        self.logger.info("屏幕截图模块初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("screen_capture")
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
    
    def _init_high_performance_capture(self) -> None:
        """初始化高性能捕获模块"""
        try:
            from .high_performance_capture.capture_agent import CaptureAgent
            self.high_performance_capture = CaptureAgent(self.adb, self.config)
            self.logger.info("高性能捕获模块初始化成功")
        except Exception as e:
            self.logger.warning(f"高性能捕获模块初始化失败，使用普通模式: {e}")
    
    def capture_screenshot_high_performance(self) -> Optional[np.ndarray]:
        """
        高性能截图捕获 - 基于MAA方案
        
        Returns:
            高性能屏幕截图或None
        """
        if self.high_performance_capture:
            return self.high_performance_capture.screencap()
        else:
            # 回退到普通模式
            self.logger.warning("高性能捕获不可用，使用普通模式")
            return self.capture_screenshot()
    
    def get_high_performance_info(self) -> dict:
        """
        获取高性能捕获的性能信息
        
        Returns:
            包含性能信息的字典
        """
        if self.high_performance_capture:
            return self.high_performance_capture.get_performance_info()
        else:
            return {'error': '高性能捕获模块未初始化'}
    
    def switch_capture_method(self, method_name: str) -> bool:
        """
        手动切换捕获方法
        
        Args:
            method_name: 方法名称
            
        Returns:
            切换是否成功
        """
        if self.high_performance_capture:
            return self.high_performance_capture.switch_method(method_name)
        else:
            self.logger.error("高性能捕获模块未初始化，无法切换方法")
            return False

    def capture_screenshot(self) -> Optional[np.ndarray]:
        """
        捕获单张屏幕截图（优化版本）
        
        Returns:
            OpenCV格式的图像数组或None
        """
        if not self.adb.is_connected():
            self.logger.error("设备未连接，无法截图")
            return None
        
        start_time = time.time()
        
        # 检查缓存是否可用（减少重复截图）
        cache_valid = (time.time() - self.cache_timestamp) < self.config.screenshot_cache_timeout
        if cache_valid and self.last_screenshot is not None:
            self.logger.debug("使用缓存的截图")
            return self.last_screenshot.copy()
        
        for attempt in range(self.config.screenshot_retry_count):
            try:
                # 使用ADB命令截图（启用二进制输出模式）
                command = "shell screencap -p"
                success, output = self.adb._execute_adb_command(command, binary_output=True)
                
                if not success:
                    self.logger.warning(f"截图命令执行失败 (尝试 {attempt + 1}/{self.config.screenshot_retry_count})")
                    continue
                
                # 将二进制数据转换为图像
                if output and len(output) > 0:
                    # 修复MUMU模拟器ADB截图数据格式
                    cleaned_output = self._fix_adb_screenshot_data(output)
                    
                    # 基于MAA文档：使用原始数据格式解码
                    screenshot = self._decode_raw_screenshot_data(cleaned_output)
                    
                    # 应用压缩（如果启用）
                    if self.config.screenshot_compression:
                        screenshot = self._compress_image(screenshot)
                    
                    # 更新性能统计
                    capture_time = time.time() - start_time
                    self.capture_count += 1
                    self.total_capture_time += capture_time
                    self.last_capture_time = capture_time
                    
                    # 更新缓存
                    self.last_screenshot = screenshot.copy()
                    self.cache_timestamp = time.time()
                    
                    self.logger.debug(f"截图成功 - 尺寸: {screenshot.shape}, 耗时: {capture_time:.3f}秒")
                    return screenshot
                else:
                    self.logger.warning(f"截图返回空数据 (尝试 {attempt + 1}/{self.config.screenshot_retry_count})")
                
            except Exception as e:
                self.logger.error(f"截图过程中出现异常 (尝试 {attempt + 1}/{self.config.screenshot_retry_count}): {e}")
                
                if attempt == self.config.screenshot_retry_count - 1:
                    self.logger.error("所有截图尝试均失败")
                    return None
                
                time.sleep(self.config.retry_delay)
        
        return None
    
    def _fix_adb_screenshot_data(self, screenshot_data: bytes) -> bytes:
        """
        修复ADB截图数据 - 基于MAA的原始数据格式
        
        Args:
            screenshot_data: 原始截图数据
            
        Returns:
            修复后的截图数据
        """
        try:
            # 基于MAA文档：Android screencap原始格式
            # 格式：4字节宽度 + 4字节高度 + 像素数据
            
            # 检查数据长度是否足够
            if len(screenshot_data) < 8:
                self.logger.warning("截图数据长度不足")
                return screenshot_data
            
            # 解析宽度和高度（小端序）
            width = int.from_bytes(screenshot_data[0:4], 'little')
            height = int.from_bytes(screenshot_data[4:8], 'little')
            
            # 验证数据完整性
            expected_size = 8 + width * height * 4  # 4字节RGBA
            if len(screenshot_data) < expected_size:
                self.logger.warning(f"截图数据不完整: 期望 {expected_size}, 实际 {len(screenshot_data)}")
                return screenshot_data
            
            self.logger.debug(f"原始截图数据: 宽度={width}, 高度={height}, 总长度={len(screenshot_data)}")
            
            # 已经是正确的原始格式，无需修复
            return screenshot_data
            
        except Exception as e:
            self.logger.warning(f"解析截图数据失败: {e}")
            return screenshot_data
    
    def _decode_raw_screenshot_data(self, screenshot_data: bytes) -> Optional[np.ndarray]:
        """
        解码原始截图数据 - 基于MAA的原始数据格式
        
        Args:
            screenshot_data: 原始截图数据
            
        Returns:
            解码后的图像或None
        """
        try:
            # 基于MAA文档：Android screencap原始格式
            # 格式：4字节宽度 + 4字节高度 + 像素数据
            
            # 检查数据长度是否足够
            if len(screenshot_data) < 8:
                self.logger.warning("截图数据长度不足，无法解码")
                return None
            
            # 解析宽度和高度（小端序）
            width = int.from_bytes(screenshot_data[0:4], 'little')
            height = int.from_bytes(screenshot_data[4:8], 'little')
            
            self.logger.debug(f"解码原始数据: 宽度={width}, 高度={height}, 总长度={len(screenshot_data)}")
            
            # 计算像素数据起始位置
            # 基于实际数据：header_size = 总长度 - (宽度 * 高度 * 4)
            pixel_data_size = width * height * 4
            header_size = len(screenshot_data) - pixel_data_size
            
            if header_size < 8 or header_size > len(screenshot_data):
                self.logger.warning(f"无效的header_size: {header_size}, 总长度={len(screenshot_data)}, 像素数据大小={pixel_data_size}")
                return None
            
            # 提取像素数据
            pixel_data = screenshot_data[header_size:]
            
            if len(pixel_data) != 4 * width * height:
                self.logger.warning(f"像素数据长度不匹配: 期望 {4 * width * height}, 实际 {len(pixel_data)}")
                return None
            
            # 创建OpenCV Mat对象（RGBA格式）
            # 基于MAA文档：cv::Mat temp(im_height, im_width, CV_8UC4, const_cast<uint8_t*>(im_data))
            temp = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, 4)
            
            # 转换为BGR格式
            # 基于MAA文档：cv::cvtColor(temp, temp, cv::COLOR_RGBA2BGR)
            bgr_image = cv2.cvtColor(temp, cv2.COLOR_RGBA2BGR)
            
            # 图像翻转（如果需要）
            # 基于MAA文档：cv::flip(bgr, dst, 0)
            final_image = cv2.flip(bgr_image, 0)
            
            self.logger.debug(f"原始数据解码成功: 尺寸={final_image.shape}")
            return final_image
            
        except Exception as e:
            self.logger.error(f"原始数据解码失败: {e}")
            return None
    
    def _compress_image(self, image: np.ndarray) -> np.ndarray:
        """
        压缩图像（保持质量的同时减小尺寸）
        
        Args:
            image: 原始图像
            
        Returns:
            压缩后的图像
        """
        try:
            # 如果图像尺寸较大，进行适当缩放
            height, width = image.shape[:2]
            max_dimension = 1920  # 最大尺寸限制
            
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                self.logger.debug(f"图像缩放: {width}x{height} -> {new_width}x{new_height}")
            
            return image
            
        except Exception as e:
            self.logger.warning(f"图像压缩失败: {e}")
            return image
    
    def start_continuous_capture(self) -> bool:
        """
        开始连续截图捕获
        
        Returns:
            启动是否成功
        """
        if self.is_capturing:
            self.logger.warning("截图捕获已在运行中")
            return True
        
        if not self.adb.is_connected():
            self.logger.error("设备未连接，无法开始连续截图")
            return False
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.logger.info("开始连续截图捕获")
        return True
    
    def capture_continuous(self) -> bool:
        """
        开始连续截图捕获（start_continuous_capture的别名方法）
        
        Returns:
            启动是否成功
        """
        return self.start_continuous_capture()
    
    def stop_continuous_capture(self) -> None:
        """停止连续截图捕获"""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
        
        # 清空队列
        while not self.screenshot_queue.empty():
            try:
                self.screenshot_queue.get_nowait()
            except Empty:
                break
        
        self.logger.info("停止连续截图捕获")
    
    def stop_capture(self) -> None:
        """停止连续截图捕获（stop_continuous_capture的别名方法）"""
        self.stop_continuous_capture()
    
    def _capture_loop(self) -> None:
        """截图循环 - 在后台线程中运行"""
        self.logger.debug("截图循环开始")
        
        while self.is_capturing:
            try:
                # 捕获截图
                screenshot = self.capture_screenshot()
                
                if screenshot is not None:
                    # 放入队列（如果队列已满，丢弃最旧的截图）
                    if self.screenshot_queue.full():
                        try:
                            self.screenshot_queue.get_nowait()  # 丢弃最旧的
                        except Empty:
                            pass
                    
                    self.screenshot_queue.put(screenshot)
                
                # 等待指定的间隔时间
                time.sleep(self.config.screenshot_interval)
                
            except Exception as e:
                self.logger.error(f"截图循环中出现异常: {e}")
                time.sleep(1.0)  # 出错后等待1秒再继续
        
        self.logger.debug("截图循环结束")
    
    def get_latest_screenshot(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        获取最新的截图
        
        Args:
            timeout: 等待超时时间（秒）
            
        Returns:
            最新的截图或None
        """
        try:
            if self.screenshot_queue.empty() and self.last_screenshot is not None:
                # 如果队列为空但有缓存，返回缓存
                return self.last_screenshot.copy()
            
            screenshot = self.screenshot_queue.get(timeout=timeout)
            self.screenshot_queue.task_done()
            return screenshot
            
        except Empty:
            self.logger.debug("截图队列为空")
            return None
    
    def save_screenshot(self, screenshot: np.ndarray, filepath: str) -> bool:
        """
        保存截图到文件
        
        Args:
            screenshot: 截图图像
            filepath: 保存路径
            
        Returns:
            保存是否成功
        """
        try:
            # 确保目录存在
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # 根据配置保存为PNG格式
            if self.config.screenshot_format.upper() == "PNG":
                # 转换为RGB格式（PIL使用RGB）
                rgb_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                # 保存为PNG，使用适当的质量设置
                pil_image.save(filepath, "PNG", optimize=True, compress_level=6)
            else:
                # 使用OpenCV保存
                cv2.imwrite(filepath, screenshot, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            
            self.logger.debug(f"截图保存成功: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"截图保存失败: {e}")
            return False
    
    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        avg_capture_time = (self.total_capture_time / self.capture_count 
                           if self.capture_count > 0 else 0)
        
        return {
            "total_captures": self.capture_count,
            "average_capture_time": avg_capture_time,
            "last_capture_time": self.last_capture_time,
            "queue_size": self.screenshot_queue.qsize(),
            "is_capturing": self.is_capturing
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_continuous_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_continuous_capture()


if __name__ == "__main__":
    """屏幕截图模块测试代码"""
    # 创建ADB控制器
    adb = ADBController()
    
    try:
        # 连接设备
        if adb.connect_device():
            print("✅ 设备连接成功")
            
            # 创建截图模块
            capture = ScreenCapture(adb)
            
            # 测试单次截图
            print("测试单次截图...")
            screenshot = capture.capture_screenshot()
            
            if screenshot is not None:
                print(f"✅ 截图成功 - 尺寸: {screenshot.shape}")
                
                # 保存测试截图
                test_path = "test_screenshot.png"
                if capture.save_screenshot(screenshot, test_path):
                    print(f"✅ 截图保存成功: {test_path}")
                else:
                    print("❌ 截图保存失败")
                
                # 测试连续截图
                print("测试连续截图（3秒）...")
                capture.start_continuous_capture()
                time.sleep(3)
                
                # 获取性能统计
                stats = capture.get_performance_stats()
                print(f"性能统计: {stats}")
                
                # 停止连续截图
                capture.stop_continuous_capture()
                print("✅ 连续截图测试完成")
            else:
                print("❌ 截图失败")
            
            # 断开连接
            adb.disconnect_device()
            print("✅ 设备断开成功")
        else:
            print("❌ 设备连接失败")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")