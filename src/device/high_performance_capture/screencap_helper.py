"""
图像解码辅助 - 基于MAA方案的统一图像解码器

技术原理：
1. 处理不同平台的换行符差异
2. 提供多种解码器（JPEG、Raw、GZIP等）
3. 统一图像格式转换（RGBA到BGR）
4. 错误处理和重试机制

基于MAA ScreencapHelper.cpp源码实现，确保技术方案的一致性。
"""

import cv2
import numpy as np
import gzip
from typing import Optional, Callable, Any
import logging


class ScreencapHelper:
    """图像解码辅助 - 基于MAA的实现"""
    
    def __init__(self):
        """初始化ScreencapHelper"""
        self.logger = logging.getLogger("screencap_helper")
        
        # 换行符处理状态
        self.end_of_line = "unknown"
        
        self.logger.info("ScreencapHelper初始化完成")
    
    def process_data(self, buffer: bytes, decoder: Callable[[bytes], Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """
        处理图像数据 - 基于MAA的实现
        
        Args:
            buffer: 原始图像数据
            decoder: 解码器函数
            
        Returns:
            解码后的图像或None
            
        技术原理：
        1. 处理不同平台的换行符差异
        2. 调用具体的解码器
        3. 转换为BGR格式
        """
        try:
            # 处理Windows平台的换行符差异
            buffer_copy = bytearray(buffer)
            
            # 检查是否需要清理回车符
            if self.end_of_line == "unknown":
                # 尝试清理回车符
                cleaned_buffer = self._clean_cr(buffer_copy)
                
                # 尝试解码
                result = decoder(cleaned_buffer)
                if result is not None:
                    self.end_of_line = "crlf"
                    
                    # 转换为BGR格式
                    temp = result
                    if len(temp.shape) == 3 and temp.shape[2] == 4:
                        cv2.cvtColor(temp, temp, cv2.COLOR_RGBA2BGR)
                    
                    return temp
            
            # 直接调用解码器
            result = decoder(buffer_copy)
            
            if result is not None:
                # 转换为BGR格式
                temp = result
                if len(temp.shape) == 3 and temp.shape[2] == 4:
                    cv2.cvtColor(temp, temp, cv2.COLOR_RGBA2BGR)
                
                return temp
            
            return None
            
        except Exception as e:
            self.logger.error(f"处理图像数据异常: {e}")
            return None
    
    def _clean_cr(self, buffer: bytearray) -> bytearray:
        """
        清理回车符 - 处理Windows换行符差异
        
        Args:
            buffer: 原始数据
            
        Returns:
            清理后的数据
        """
        try:
            # 检查是否是有效的PNG文件
            if buffer.startswith(b'\x89PNG\r\n\x1a\n'):
                # 已经是标准PNG格式，无需修复
                return buffer
            
            # 修复回车符问题：将Windows风格的 \r\n 替换为Unix风格的 \n
            cleaned = buffer.replace(b'\r\n', b'\n')
            
            # 如果修复后数据长度变化，记录日志
            if len(cleaned) != len(buffer):
                self.logger.debug(f"修复了回车符：长度 {len(buffer)} -> {len(cleaned)}")
            
            return cleaned
            
        except Exception as e:
            self.logger.warning(f"清理回车符失败: {e}")
            return buffer
    
    @staticmethod
    def decode_jpg(buffer: bytes) -> Optional[np.ndarray]:
        """
        解码JPEG图像
        
        Args:
            buffer: JPEG压缩的图像数据
            
        Returns:
            解码后的图像或None
        """
        try:
            # 将字节数据转换为numpy数组
            jpeg_array = np.frombuffer(buffer, dtype=np.uint8)
            
            # 使用OpenCV解码JPEG
            image = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
            
            return image
            
        except Exception as e:
            logging.getLogger("screencap_helper").error(f"JPEG解码异常: {e}")
            return None
    
    @staticmethod
    def decode_raw(buffer: bytes) -> Optional[np.ndarray]:
        """
        解码Raw格式图像 - 基于MAA的实现
        
        Args:
            buffer: Raw格式的图像数据
            
        Returns:
            解码后的图像或None
            
        技术原理：
        Android screencap原始格式：4字节宽度 + 4字节高度 + 像素数据
        
        基于MAA ScreencapHelper.cpp源码实现。
        """
        try:
            if len(buffer) < 8:
                logging.getLogger("screencap_helper").warning("Raw数据长度不足")
                return None
            
            # Android screencap原始格式：4字节宽度 + 4字节高度 + 像素数据
            data = buffer
            
            # 读取宽度和高度（小端序）
            im_width = int.from_bytes(data[0:4], byteorder='little')
            im_height = int.from_bytes(data[4:8], byteorder='little')
            
            # 计算像素数据起始位置
            header_size = len(buffer) - (4 * im_width * im_height)
            
            if header_size < 8 or header_size >= len(buffer):
                logging.getLogger("screencap_helper").warning("Raw数据格式错误")
                return None
            
            # 提取像素数据
            im_data = data[header_size:]
            
            if len(im_data) < 4 * im_width * im_height:
                logging.getLogger("screencap_helper").warning("像素数据长度不足")
                return None
            
            # 创建OpenCV Mat对象（RGBA格式）
            temp = np.frombuffer(im_data, dtype=np.uint8)
            temp = temp.reshape((im_height, im_width, 4))
            
            return temp
            
        except Exception as e:
            logging.getLogger("screencap_helper").error(f"Raw解码异常: {e}")
            return None
    
    @staticmethod
    def decode_gzip(buffer: bytes) -> Optional[np.ndarray]:
        """
        解码GZIP压缩的图像数据
        
        Args:
            buffer: GZIP压缩的图像数据
            
        Returns:
            解码后的图像或None
        """
        try:
            # 解压GZIP数据
            decompressed = gzip.decompress(buffer)
            
            # 尝试解码为Raw格式
            return ScreencapHelper.decode_raw(decompressed)
            
        except Exception as e:
            logging.getLogger("screencap_helper").error(f"GZIP解码异常: {e}")
            return None
    
    @staticmethod
    def decode_png(buffer: bytes) -> Optional[np.ndarray]:
        """
        解码PNG图像
        
        Args:
            buffer: PNG图像数据
            
        Returns:
            解码后的图像或None
        """
        try:
            # 将字节数据转换为numpy数组
            png_array = np.frombuffer(buffer, dtype=np.uint8)
            
            # 使用OpenCV解码PNG
            image = cv2.imdecode(png_array, cv2.IMREAD_COLOR)
            
            return image
            
        except Exception as e:
            logging.getLogger("screencap_helper").error(f"PNG解码异常: {e}")
            return None
    
    @staticmethod
    def trunc_decode_jpg(buffer: bytes) -> Optional[np.ndarray]:
        """
        截断解码JPEG图像 - 用于处理不完整的JPEG数据
        
        Args:
            buffer: 可能不完整的JPEG数据
            
        Returns:
            解码后的图像或None
        """
        try:
            # 查找JPEG结束标记
            end_marker = buffer.find(b'\xff\xd9')
            
            if end_marker != -1:
                # 截取到有效的JPEG数据
                valid_buffer = buffer[:end_marker + 2]
                return ScreencapHelper.decode_jpg(valid_buffer)
            else:
                # 没有找到结束标记，尝试直接解码
                return ScreencapHelper.decode_jpg(buffer)
                
        except Exception as e:
            logging.getLogger("screencap_helper").error(f"截断JPEG解码异常: {e}")
            return None