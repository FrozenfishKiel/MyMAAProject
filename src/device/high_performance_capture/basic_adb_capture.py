"""
基础ADB捕获模块 - 作为高性能捕获的fallback方案

功能：
1. 使用标准ADB screencap命令进行截图
2. 支持原始数据格式解码（基于MAA方案）
3. 提供基本的错误处理和重试机制

基于MAA的Raw格式解码实现
"""

import time
import logging
from typing import Optional

import cv2
import numpy as np

from ..adb_controller import ADBController
from ..config import DeviceConfig


class BasicADBCapture:
    """基础ADB捕获模块 - 基于MAA的Raw格式解码"""
    
    def __init__(self, adb_controller: ADBController, config: DeviceConfig):
        """
        初始化基础ADB捕获模块
        
        Args:
            adb_controller: ADB控制器实例
            config: 设备配置对象
        """
        self.adb = adb_controller
        self.config = config
        self.logger = logging.getLogger("basic_adb_capture")
        
        # 性能统计
        self.capture_count = 0
        self.total_capture_time = 0.0
        
        self.logger.info("基础ADB捕获模块初始化完成")
    
    def screencap(self) -> Optional[np.ndarray]:
        """
        执行基础ADB截图
        
        Returns:
            屏幕截图图像或None
        """
        if not self.adb.is_connected():
            self.logger.error("设备未连接")
            return None
        
        start_time = time.time()
        
        for attempt in range(self.config.screenshot_retry_count):
            try:
                # 使用ADB命令截图（启用二进制输出模式）
                command = "shell screencap -p"
                success, output = self.adb._execute_adb_command(command, binary_output=True)
                
                if not success:
                    self.logger.warning(f"截图命令执行失败 (尝试 {attempt + 1}/{self.config.screenshot_retry_count})")
                    continue
                
                if output and len(output) > 0:
                    # 尝试解码原始数据格式（基于MAA方案）
                    screenshot = self._decode_raw_screenshot_data(output)
                    
                    if screenshot is not None:
                        # 更新性能统计
                        capture_time = time.time() - start_time
                        self.capture_count += 1
                        self.total_capture_time += capture_time
                        
                        self.logger.debug(f"基础ADB截图成功 - 尺寸: {screenshot.shape}, 耗时: {capture_time:.3f}秒")
                        return screenshot
                    else:
                        # 如果原始格式解码失败，尝试PNG解码
                        screenshot = self._decode_png_screenshot_data(output)
                        if screenshot is not None:
                            capture_time = time.time() - start_time
                            self.capture_count += 1
                            self.total_capture_time += capture_time
                            
                            self.logger.debug(f"PNG格式截图成功 - 尺寸: {screenshot.shape}, 耗时: {capture_time:.3f}秒")
                            return screenshot
                
                self.logger.warning(f"截图返回空数据 (尝试 {attempt + 1}/{self.config.screenshot_retry_count})")
                
            except Exception as e:
                self.logger.error(f"截图过程中出现异常 (尝试 {attempt + 1}/{self.config.screenshot_retry_count}): {e}")
                
                if attempt == self.config.screenshot_retry_count - 1:
                    self.logger.error("所有截图尝试均失败")
                    return None
                
                time.sleep(self.config.retry_delay)
        
        return None
    
    def _decode_raw_screenshot_data(self, screenshot_data: bytes) -> Optional[np.ndarray]:
        """
        解码原始截图数据 - 基于MAA的Raw格式解码
        
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
                self.logger.warning("截图数据长度不足，无法解码原始格式")
                return None
            
            # 检查是否为PNG格式（PNG文件头）
            if screenshot_data.startswith(b'\x89PNG\r\n\x1a\n'):
                self.logger.debug("检测到PNG格式数据，跳过原始格式解码")
                return None
            
            # 解析宽度和高度（小端序）
            width = int.from_bytes(screenshot_data[0:4], 'little')
            height = int.from_bytes(screenshot_data[4:8], 'little')
            
            # 验证分辨率合理性
            if width <= 0 or height <= 0 or width > 10000 or height > 10000:
                self.logger.warning(f"无效的分辨率: {width}x{height}")
                return None
            
            self.logger.debug(f"尝试解码原始数据: 宽度={width}, 高度={height}, 总长度={len(screenshot_data)}")
            
            # 计算像素数据起始位置
            # 基于实际数据：header_size = 总长度 - (宽度 * 高度 * 4)
            pixel_data_size = width * height * 4
            
            # 检查数据长度是否足够
            if len(screenshot_data) < pixel_data_size:
                self.logger.warning(f"数据长度不足: 期望至少 {pixel_data_size}, 实际 {len(screenshot_data)}")
                return None
            
            # 提取像素数据（从数据末尾开始）
            pixel_data = screenshot_data[-pixel_data_size:]
            
            if len(pixel_data) != pixel_data_size:
                self.logger.warning(f"像素数据长度不匹配: 期望 {pixel_data_size}, 实际 {len(pixel_data)}")
                return None
            
            # 创建OpenCV图像（RGBA格式）
            img_rgba = np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width, 4))
            
            # 转换为BGR格式
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            
            self.logger.debug(f"原始格式解码成功: {img_bgr.shape}")
            return img_bgr
            
        except Exception as e:
            self.logger.warning(f"原始格式解码失败: {e}")
            return None
    
    def _decode_png_screenshot_data(self, screenshot_data: bytes) -> Optional[np.ndarray]:
        """
        解码PNG格式截图数据
        
        Args:
            screenshot_data: PNG格式截图数据
            
        Returns:
            解码后的图像或None
        """
        try:
            # 检查是否为PNG格式（PNG文件头）
            if screenshot_data.startswith(b'\x89PNG\r\n\x1a\n'):
                # 使用OpenCV解码PNG
                nparr = np.frombuffer(screenshot_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    self.logger.debug(f"PNG格式解码成功: {img.shape}")
                    return img
            
            self.logger.warning("不是有效的PNG格式数据")
            return None
            
        except Exception as e:
            self.logger.warning(f"PNG格式解码失败: {e}")
            return None
    
    def get_performance_info(self) -> dict:
        """获取性能信息"""
        avg_capture_time = (self.total_capture_time / self.capture_count 
                          if self.capture_count > 0 else 0)
        
        return {
            'method': 'basic_adb',
            'capture_count': self.capture_count,
            'average_capture_time': avg_capture_time,
            'total_capture_time': self.total_capture_time
        }