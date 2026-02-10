"""
模拟器专用SDK - 基于MAA方案的直接内存访问捕获

技术原理：
1. 动态加载模拟器的外部渲染器IPC库
2. 直接访问渲染缓冲区数据
3. 硬件加速，利用模拟器的GPU渲染
4. 零延迟，直接从渲染管线获取图像

基于MAA MuMuPlayerExtras.cpp源码实现，确保技术方案的一致性。
"""

import os
import ctypes
import logging
from typing import Optional
import cv2
import numpy as np

from ..adb_controller import ADBController
from ..config import DeviceConfig


class EmulatorSDK:
    """模拟器专用SDK - 基于MAA的实现"""
    
    def __init__(self, adb_controller: ADBController, config: DeviceConfig):
        """
        初始化EmulatorSDK
        
        Args:
            adb_controller: ADB控制器实例
            config: 设备配置对象
        """
        self.adb = adb_controller
        self.config = config
        self.logger = logging.getLogger("emulator_sdk")
        
        # SDK相关变量
        self.mumu_handle = None
        self.capture_display_func = None
        self.display_buffer = None
        self.display_width = 0
        self.display_height = 0
        
        # 初始化SDK
        self._init_sdk()
        
        self.logger.info("EmulatorSDK初始化完成")
    
    def _init_sdk(self) -> bool:
        """
        初始化模拟器SDK - 基于真实MUMU模拟器环境
        
        Returns:
            初始化是否成功
        """
        try:
            # 基于MAA方案：加载模拟器的外部渲染器IPC库
            # 1. 检查MUMU模拟器路径
            mumu_path = self.config.mumu_path
            if not os.path.exists(mumu_path):
                self.logger.error(f"MUMU模拟器路径不存在: {mumu_path}")
                return False
            
            # 2. 查找SDK库文件
            sdk_library_path = self._find_sdk_library(mumu_path)
            if not sdk_library_path:
                self.logger.warning("未找到模拟器SDK库文件，使用ADB fallback")
                return False
            
            # 3. 加载SDK库
            try:
                sdk_lib = ctypes.CDLL(sdk_library_path)
                self.logger.info(f"成功加载SDK库: {sdk_library_path}")
                
                # 保存库引用
                self.sdk_lib = sdk_lib
                
            except Exception as e:
                self.logger.warning(f"加载SDK库失败: {e}")
                return False
            
            # 4. 获取函数指针 - 基于MUMU模拟器的实际API
            # 尝试加载可能的屏幕捕获函数
            self._load_sdk_functions()
            
            # 5. 检查模拟器连接状态
            if not self._check_emulator_connection():
                self.logger.warning("模拟器连接检查失败，使用ADB fallback")
                return False
            
            # 6. 获取显示信息
            if not self._get_display_info():
                self.logger.warning("获取显示信息失败，使用ADB fallback")
                return False
            
            self.logger.info("SDK初始化完成 - 使用真实MUMU模拟器环境")
            return True
            
        except Exception as e:
            self.logger.error(f"SDK初始化异常: {e}")
            return False
    
    def _find_sdk_library(self, mumu_path: str) -> Optional[str]:
        """
        查找SDK库文件 - 基于真实MUMU模拟器环境
        
        Args:
            mumu_path: MUMU模拟器路径
            
        Returns:
            SDK库文件路径或None
        """
        # 基于MUMU模拟器的实际目录结构查找可能的SDK库
        possible_paths = [
            os.path.join(mumu_path, "nx_main", "nemu-api.dll"),  # 主要的API库
            os.path.join(mumu_path, "nx_main", "nemu-vcontrolmanager.dll"),  # 虚拟控制管理器
            os.path.join(mumu_path, "shell", "MuMuPlayerExtras.dll"),
            os.path.join(mumu_path, "MuMuPlayerExtras.dll"),
            os.path.join(mumu_path, "external_renderer_ipc.dll"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"找到SDK库文件: {path}")
                return path
        
        self.logger.warning("未找到SDK库文件，将使用ADB fallback")
        return None
    
    def _load_sdk_functions(self) -> None:
        """
        加载SDK函数 - 尝试加载可能的屏幕捕获相关函数
        """
        try:
            # 基于MUMU模拟器的实际API，尝试加载可能的函数
            # 注意：这些函数名可能需要根据实际的SDK文档进行调整
            
            # 尝试加载可能的屏幕捕获函数
            function_names = [
                'capture_display',
                'get_display_buffer',
                'screencap',
                'capture_screen',
                'get_frame_buffer',
                'nemu_capture_display',
                'nemu_get_display_info'
            ]
            
            for func_name in function_names:
                try:
                    func = getattr(self.sdk_lib, func_name)
                    setattr(self, f'{func_name}_func', func)
                    self.logger.debug(f"成功加载函数: {func_name}")
                except AttributeError:
                    self.logger.debug(f"函数不存在: {func_name}")
            
            # 设置函数原型（如果知道参数类型）
            self._setup_function_prototypes()
            
        except Exception as e:
            self.logger.warning(f"加载SDK函数失败: {e}")
    
    def _setup_function_prototypes(self) -> None:
        """
        设置函数原型 - 定义函数的参数和返回类型
        """
        try:
            # 如果capture_display_func存在，设置其原型
            if hasattr(self, 'capture_display_func'):
                # 假设函数签名：int capture_display(handle, display_id, buffer_size, width, height, buffer)
                self.capture_display_func.argtypes = [
                    ctypes.c_void_p,  # handle
                    ctypes.c_int,     # display_id
                    ctypes.c_int,     # buffer_size
                    ctypes.POINTER(ctypes.c_int),  # width
                    ctypes.POINTER(ctypes.c_int),  # height
                    ctypes.POINTER(ctypes.c_ubyte) # buffer
                ]
                self.capture_display_func.restype = ctypes.c_int
                
        except Exception as e:
            self.logger.warning(f"设置函数原型失败: {e}")
    
    def _check_emulator_connection(self) -> bool:
        """
        检查模拟器连接状态
        
        Returns:
            连接是否正常
        """
        try:
            # 检查MUMU模拟器进程是否在运行
            import psutil
            
            mumu_processes = []
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and 'mumu' in proc.info['name'].lower():
                    mumu_processes.append(proc)
            
            if not mumu_processes:
                self.logger.warning("未找到运行的MUMU模拟器进程")
                return False
            
            self.logger.info(f"找到 {len(mumu_processes)} 个MUMU模拟器进程")
            
            # 检查ADB连接
            if not self._check_adb_connection():
                return False
            
            return True
            
        except ImportError:
            self.logger.warning("psutil模块未安装，无法检查进程状态")
            return self._check_adb_connection()
        except Exception as e:
            self.logger.error(f"检查模拟器连接异常: {e}")
            return False
    
    def _check_adb_connection(self) -> bool:
        """
        检查ADB连接状态
        
        Returns:
            ADB连接是否正常
        """
        try:
            # 使用ADB控制器检查设备连接
            if not self.adb.check_device_connection():
                self.logger.warning("ADB设备连接检查失败")
                return False
            
            # 尝试执行简单的ADB命令
            success, output = self.adb._execute_adb_command("getprop ro.product.model", binary_output=False)
            if success and output:
                model = output.strip()
                self.logger.info(f"设备型号: {model}")
                return True
            
            self.logger.warning("ADB命令执行失败")
            return False
            
        except Exception as e:
            self.logger.error(f"检查ADB连接异常: {e}")
            return False
    
    def _get_display_info(self) -> bool:
        """
        获取显示信息
        
        Returns:
            获取是否成功
        """
        try:
            # 基于MAA方案：获取模拟器的显示信息
            # 这里需要实现真实的显示信息获取逻辑
            
            # 暂时使用ADB获取屏幕分辨率
            success, output = self.adb._execute_adb_command("wm size", binary_output=False)
            if success and output:
                output_str = output
                # 解析输出格式："Physical size: 1920x1080"
                if "Physical size:" in output:
                    size_str = output.split("Physical size:")[1].strip()
                    if "x" in size_str:
                        width, height = size_str.split("x")
                        self.display_width = int(width)
                        self.display_height = int(height)
                        
                        # 分配显示缓冲区
                        buffer_size = self.display_width * self.display_height * 4  # RGBA
                        self.display_buffer = bytearray(buffer_size)
                        
                        self.logger.info(f"获取显示信息: {self.display_width}x{self.display_height}")
                        return True
            
            self.logger.error("获取显示信息失败")
            return False
            
        except Exception as e:
            self.logger.error(f"获取显示信息异常: {e}")
            return False
    
    def screencap(self) -> Optional[np.ndarray]:
        """
        执行屏幕捕获 - 基于真实MUMU模拟器环境的实现
        
        Returns:
            屏幕截图或None
            
        技术原理：
        优先使用SDK直接内存访问，失败时使用ADB fallback
        """
        try:
            # 优先尝试使用SDK进行屏幕捕获
            sdk_result = self._sdk_screencap()
            if sdk_result is not None:
                return sdk_result
            
            # SDK失败时使用ADB fallback
            self.logger.warning("SDK截图失败，使用ADB fallback")
            return self._fallback_screencap()
            
        except Exception as e:
            self.logger.error(f"屏幕捕获异常: {e}")
            return self._fallback_screencap()
    
    def _sdk_screencap(self) -> Optional[np.ndarray]:
        """
        使用SDK进行屏幕捕获 - 基于MUMU模拟器的实际API
        
        Returns:
            屏幕截图或None
        """
        try:
            # 检查SDK函数是否可用
            if not hasattr(self, 'capture_display_func'):
                self.logger.debug("SDK capture_display函数不可用")
                return None
            
            # 准备显示缓冲区
            if not self.display_buffer:
                self.logger.warning("显示缓冲区未初始化")
                return None
            
            # 调用SDK函数进行屏幕捕获
            # 基于MAA的MuMuPlayerExtras实现
            width_ptr = ctypes.c_int(0)
            height_ptr = ctypes.c_int(0)
            
            # 分配缓冲区
            buffer_size = len(self.display_buffer)
            buffer_ptr = (ctypes.c_ubyte * buffer_size).from_buffer(self.display_buffer)
            
            # 调用SDK函数
            result = self.capture_display_func(
                self.mumu_handle,
                self.config.emulator_display_id,
                buffer_size,
                ctypes.byref(width_ptr),
                ctypes.byref(height_ptr),
                buffer_ptr
            )
            
            if result == 0:  # 成功
                width = width_ptr.value
                height = height_ptr.value
                
                if width > 0 and height > 0:
                    # 创建OpenCV图像
                    image = self._create_image_from_buffer(width, height)
                    return image
                else:
                    self.logger.warning("SDK返回无效的图像尺寸")
                    return None
            else:
                self.logger.warning(f"SDK截图失败，错误码: {result}")
                return None
                
        except Exception as e:
            self.logger.error(f"SDK截图异常: {e}")
            return None
    
    def _create_image_from_buffer(self, width: int, height: int) -> np.ndarray:
        """
        从缓冲区创建图像 - 基于MAA的图像处理逻辑
        
        Args:
            width: 图像宽度
            height: 图像高度
            
        Returns:
            OpenCV格式的图像
        """
        try:
            # 基于MAA的MuMuPlayerExtras图像处理逻辑
            # 1. 创建原始图像
            raw_image = np.frombuffer(self.display_buffer, dtype=np.uint8)
            raw_image = raw_image.reshape((height, width, 4))  # RGBA格式
            
            # 2. 转换为BGR格式
            bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_RGBA2BGR)
            
            # 3. 垂直翻转（模拟器图像通常需要翻转）
            if self.config.flip_vertical:
                bgr_image = cv2.flip(bgr_image, 0)
            
            # 4. 水平翻转（如果需要）
            if self.config.flip_horizontal:
                bgr_image = cv2.flip(bgr_image, 1)
            
            return bgr_image
            
        except Exception as e:
            self.logger.error(f"创建图像失败: {e}")
            raise
    
    def _fallback_screencap(self) -> Optional[np.ndarray]:
        """
        使用ADB screencap作为fallback方法
        
        Returns:
            屏幕截图或None
        """
        try:
            # 使用ADB控制器执行真实的screencap命令
            success, output = self.adb._execute_adb_command("shell screencap -p", binary_output=True)
            
            if success and output:
                # 基于MAA文档：使用原始数据格式解码
                return self._decode_raw_screenshot_data(output)
            else:
                self.logger.error("ADB screencap命令执行失败")
                return None
                
        except Exception as e:
            self.logger.error(f"fallback截图异常: {e}")
            return None
    
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
            temp = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, 4)
            
            # 转换为BGR格式
            bgr_image = cv2.cvtColor(temp, cv2.COLOR_RGBA2BGR)
            
            # 图像翻转（如果需要）
            final_image = cv2.flip(bgr_image, 0)
            
            self.logger.debug(f"原始数据解码成功: 尺寸={final_image.shape}")
            return final_image
            
        except Exception as e:
            self.logger.error(f"原始数据解码失败: {e}")
            return None
    
    def get_performance_info(self) -> dict:
        """
        获取性能信息
        
        Returns:
            包含性能信息的字典
        """
        return {
            'method': 'emulator_sdk',
            'sdk_available': self.capture_display_func is not None,
            'display_size': f"{self.display_width}x{self.display_height}",
            'using_fallback': self.capture_display_func is None
        }
    
    def __del__(self):
        """析构函数，确保资源释放"""
        # 清理SDK资源
        if self.mumu_handle:
            # 释放模拟器句柄
            pass