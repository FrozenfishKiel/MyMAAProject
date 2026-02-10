"""
MUMU模拟器SDK捕获模块 - 完全按照MAA框架实现

基于MAA的MuMuPlayerExtras.cpp实现，直接使用MUMU模拟器的SDK接口
技术原理：
1. 使用MUMU模拟器的专用SDK库：external_renderer_ipc.dll
2. 直接调用模拟器内部接口：nemu_capture_display等函数
3. 绕过窗口系统，直接获取渲染缓冲区
4. 实现高性能的屏幕捕获

函数接口（基于MAA实现）：
- nemu_connect: 连接模拟器
- nemu_disconnect: 断开连接
- nemu_capture_display: 捕获显示内容
- nemu_get_display_id: 获取显示ID
- nemu_input_text: 输入文本
- nemu_input_event_finger_touch_down: 触摸按下
- nemu_input_event_finger_touch_up: 触摸抬起
- nemu_input_event_key_down: 按键按下
- nemu_input_event_key_up: 按键抬起
"""

import os
import time
import logging
import ctypes
from ctypes import wintypes
from typing import Optional, Tuple, List

import cv2
import numpy as np

from ..adb_controller import ADBController
from ..config import DeviceConfig


class MUMUSDKMAACapture:
    """MUMU模拟器SDK捕获 - 完全按照MAA框架实现"""
    
    # 函数名常量（基于MAA的实现）
    NEMU_CONNECT_FUNC = "nemu_connect"
    NEMU_DISCONNECT_FUNC = "nemu_disconnect"
    NEMU_CAPTURE_DISPLAY_FUNC = "nemu_capture_display"
    NEMU_GET_DISPLAY_ID_FUNC = "nemu_get_display_id"
    NEMU_INPUT_TEXT_FUNC = "nemu_input_text"
    NEMU_INPUT_EVENT_FINGER_TOUCH_DOWN_FUNC = "nemu_input_event_finger_touch_down"
    NEMU_INPUT_EVENT_FINGER_TOUCH_UP_FUNC = "nemu_input_event_finger_touch_up"
    NEMU_INPUT_EVENT_KEY_DOWN_FUNC = "nemu_input_event_key_down"
    NEMU_INPUT_EVENT_KEY_UP_FUNC = "nemu_input_event_key_up"
    
    def __init__(self, adb_controller: ADBController, config: DeviceConfig):
        """
        初始化MUMU模拟器SDK捕获
        
        Args:
            adb_controller: ADB控制器实例
            config: 设备配置对象
        """
        self.adb = adb_controller
        self.config = config
        self.logger = logging.getLogger("mumu_sdk_maa")
        
        # MUMU模拟器路径
        self.mumu_path = config.mumu_path
        self.mumu_index = getattr(config, "mumu_index", 0)
        
        # SDK库路径（基于MAA的实现）
        self.lib_path = self._find_mumu_library()
        
        # SDK组件
        self.sdk_loaded = False
        self.mumu_handle = None
        
        # 函数指针
        self.connect_func = None
        self.disconnect_func = None
        self.capture_display_func = None
        self.get_display_id_func = None
        self.input_text_func = None
        self.input_event_touch_down_func = None
        self.input_event_touch_up_func = None
        self.input_event_key_down_func = None
        self.input_event_key_up_func = None
        
        # 显示缓冲区
        self.display_buffer = None
        self.display_width = 0
        self.display_height = 0
        
        # 应用包名
        self.mumu_app_package = getattr(config, "mumu_app_package", "")
        self.mumu_app_cloned_index = getattr(config, "mumu_app_cloned_index", 0)
        
        # 捕获状态
        self.capture_initialized = False
        self.capture_count = 0
        self.total_capture_time = 0.0
        
        # 初始化捕获
        if not self._initialize_capture():
            raise RuntimeError("MUMU SDK捕获初始化失败")
    
    def _find_mumu_library(self) -> str:
        """查找MUMU模拟器的SDK库"""
        # 基于MAA的实现，查找external_renderer_ipc.dll
        possible_paths = [
            os.path.join(self.mumu_path, "nx_main", "sdk", "external_renderer_ipc.dll"),
            os.path.join(self.mumu_path, "shell", "sdk", "external_renderer_ipc.dll"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"找到MUMU SDK库: {path}")
                return path
        
        # 如果找不到专用库，尝试使用其他DLL
        fallback_paths = [
            os.path.join(self.mumu_path, "shell", "nemu-api.dll"),
            os.path.join(self.mumu_path, "nx_device", "12.0", "device", "sdk", "neac", "NeacInterface.dll"),
            os.path.join(self.mumu_path, "nx_device", "12.0", "device", "libRenderer.dll"),
        ]
        
        for path in fallback_paths:
            if os.path.exists(path):
                self.logger.warning(f"使用备用库: {path}")
                return path
        
        raise FileNotFoundError(f"未找到MUMU SDK库，请检查路径: {self.mumu_path}")
    
    def _initialize_capture(self) -> bool:
        """初始化捕获环境"""
        try:
            self.logger.info("初始化MUMU SDK捕获环境...")
            
            # 1. 加载SDK库
            if not self._load_mumu_library():
                self.logger.error("加载MUMU SDK库失败")
                return False
            
            # 2. 连接MUMU模拟器
            if not self._connect_mumu():
                self.logger.error("连接MUMU模拟器失败")
                return False
            
            # 3. 初始化屏幕捕获
            if not self._init_screencap():
                self.logger.error("初始化屏幕捕获失败")
                return False
            
            self.capture_initialized = True
            self.logger.info("MUMU SDK捕获初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化捕获环境失败: {e}")
            return False
    
    def _load_mumu_library(self) -> bool:
        """加载MUMU SDK库"""
        try:
            self.logger.info(f"加载MUMU SDK库: {self.lib_path}")
            
            # 加载DLL
            self.mumu_lib = ctypes.WinDLL(self.lib_path)
            
            # 获取所有函数指针
            functions_to_load = [
                (self.NEMU_CONNECT_FUNC, "connect_func"),
                (self.NEMU_DISCONNECT_FUNC, "disconnect_func"),
                (self.NEMU_CAPTURE_DISPLAY_FUNC, "capture_display_func"),
                (self.NEMU_GET_DISPLAY_ID_FUNC, "get_display_id_func"),
                (self.NEMU_INPUT_TEXT_FUNC, "input_text_func"),
                (self.NEMU_INPUT_EVENT_FINGER_TOUCH_DOWN_FUNC, "input_event_touch_down_func"),
                (self.NEMU_INPUT_EVENT_FINGER_TOUCH_UP_FUNC, "input_event_touch_up_func"),
                (self.NEMU_INPUT_EVENT_KEY_DOWN_FUNC, "input_event_key_down_func"),
                (self.NEMU_INPUT_EVENT_KEY_UP_FUNC, "input_event_key_up_func"),
            ]
            
            for func_name, attr_name in functions_to_load:
                try:
                    func_ptr = getattr(self.mumu_lib, func_name)
                    setattr(self, attr_name, func_ptr)
                    self.logger.debug(f"成功获取函数: {func_name}")
                except AttributeError:
                    self.logger.warning(f"未找到函数: {func_name}")
                    setattr(self, attr_name, None)
            
            # 设置函数原型
            self._setup_function_prototypes()
            
            self.sdk_loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"加载MUMU SDK库失败: {e}")
            return False
    
    def _setup_function_prototypes(self):
        """设置函数原型"""
        # nemu_connect: 连接模拟器
        if self.connect_func:
            self.connect_func.argtypes = [ctypes.c_wchar_p, ctypes.c_int]
            self.connect_func.restype = ctypes.c_void_p
        
        # nemu_disconnect: 断开连接
        if self.disconnect_func:
            self.disconnect_func.argtypes = [ctypes.c_void_p]
            self.disconnect_func.restype = None
        
        # nemu_capture_display: 捕获显示内容
        if self.capture_display_func:
            self.capture_display_func.argtypes = [
                ctypes.c_void_p,  # handle
                ctypes.c_int,     # display_id
                ctypes.c_int,     # buffer_size
                ctypes.POINTER(ctypes.c_int),  # width
                ctypes.POINTER(ctypes.c_int),  # height
                ctypes.c_void_p   # buffer
            ]
            self.capture_display_func.restype = ctypes.c_int
        
        # nemu_get_display_id: 获取显示ID
        if self.get_display_id_func:
            self.get_display_id_func.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
            self.get_display_id_func.restype = ctypes.c_int
        
        # nemu_input_text: 输入文本
        if self.input_text_func:
            self.input_text_func.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
            self.input_text_func.restype = ctypes.c_int
        
        # nemu_input_event_finger_touch_down: 触摸按下
        if self.input_event_touch_down_func:
            self.input_event_touch_down_func.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            self.input_event_touch_down_func.restype = ctypes.c_int
        
        # nemu_input_event_finger_touch_up: 触摸抬起
        if self.input_event_touch_up_func:
            self.input_event_touch_up_func.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            self.input_event_touch_up_func.restype = ctypes.c_int
        
        # nemu_input_event_key_down: 按键按下
        if self.input_event_key_down_func:
            self.input_event_key_down_func.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            self.input_event_key_down_func.restype = ctypes.c_int
        
        # nemu_input_event_key_up: 按键抬起
        if self.input_event_key_up_func:
            self.input_event_key_up_func.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            self.input_event_key_up_func.restype = ctypes.c_int
    
    def _connect_mumu(self) -> bool:
        """连接MUMU模拟器"""
        try:
            if not self.connect_func:
                self.logger.error("connect_func is null")
                return False
            
            # 转换为宽字符串路径
            mumu_path_w = str(self.mumu_path)
            
            # 调用连接函数
            self.mumu_handle = self.connect_func(mumu_path_w, self.mumu_index)
            
            if self.mumu_handle == 0:
                self.logger.error(f"连接MUMU模拟器失败: 路径={mumu_path_w}, 索引={self.mumu_index}")
                return False
            
            self.logger.info(f"MUMU模拟器连接成功: 句柄={self.mumu_handle}")
            return True
            
        except Exception as e:
            self.logger.error(f"连接MUMU模拟器失败: {e}")
            return False
    
    def _get_display_id(self) -> int:
        """获取显示ID"""
        try:
            if not self.get_display_id_func:
                self.logger.warning("get_display_id_func is null, 使用默认显示ID=0")
                return 0
            
            if self.mumu_app_package:
                display_id = self.get_display_id_func(
                    self.mumu_handle, 
                    self.mumu_app_package.encode('utf-8'), 
                    self.mumu_app_cloned_index
                )
            else:
                # 使用默认包名
                default_pkg = "default"
                display_id = self.get_display_id_func(
                    self.mumu_handle, 
                    default_pkg.encode('utf-8'), 
                    0
                )
            
            self.logger.debug(f"获取显示ID: {display_id}")
            return display_id
            
        except Exception as e:
            self.logger.error(f"获取显示ID失败: {e}")
            return 0
    
    def _init_screencap(self) -> bool:
        """初始化屏幕捕获"""
        try:
            if not self.capture_display_func:
                self.logger.error("capture_display_func is null")
                return False
            
            # 获取显示ID
            display_id = self._get_display_id()
            
            # 调用捕获函数获取分辨率（buffer_size=0, buffer=nullptr）
            width = ctypes.c_int()
            height = ctypes.c_int()
            
            # 基于MAA的实现：这里0才是成功
            ret = self.capture_display_func(
                self.mumu_handle,
                display_id,
                0,  # buffer_size=0
                ctypes.byref(width),
                ctypes.byref(height),
                None  # buffer=nullptr
            )
            
            if ret != 0:
                self.logger.error(f"初始化屏幕捕获失败，返回值: {ret}")
                return False
            
            # 检查分辨率是否有效
            if width.value <= 0 or height.value <= 0:
                self.logger.error(f"无效的分辨率: {width.value}x{height.value}")
                return False
            
            # 初始化显示缓冲区
            buffer_size = width.value * height.value * 4  # RGBA格式
            self.display_buffer = bytearray(buffer_size)
            self.display_width = width.value
            self.display_height = height.value
            
            self.logger.info(f"屏幕捕获初始化成功: 分辨率={self.display_width}x{self.display_height}, 缓冲区大小={buffer_size}字节")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化屏幕捕获失败: {e}")
            return False
    
    def screencap(self) -> Optional[np.ndarray]:
        """执行屏幕捕获"""
        if not self.capture_initialized:
            self.logger.error("捕获未初始化")
            return None
        
        start_time = time.time()
        
        try:
            # 获取显示ID
            display_id = self._get_display_id()
            
            # 准备参数
            width = ctypes.c_int()
            height = ctypes.c_int()
            buffer_size = len(self.display_buffer)
            
            # 创建缓冲区指针
            buffer_ptr = (ctypes.c_ubyte * buffer_size).from_buffer(self.display_buffer)
            
            # 调用捕获函数
            ret = self.capture_display_func(
                self.mumu_handle,
                display_id,
                buffer_size,
                ctypes.byref(width),
                ctypes.byref(height),
                ctypes.cast(buffer_ptr, ctypes.c_void_p)
            )
            
            # 基于MAA的实现：这里0才是成功
            if ret != 0:
                self.logger.error(f"SDK捕获失败，返回值: {ret}")
                return None
            
            # 检查分辨率是否匹配
            if width.value != self.display_width or height.value != self.display_height:
                self.logger.warning(f"分辨率不匹配: 期望{self.display_width}x{self.display_height}, 实际{width.value}x{height.value}")
                # 重新初始化缓冲区
                self._init_screencap()
                return None
            
            # 转换为OpenCV图像
            pixel_count = width.value * height.value
            expected_size = pixel_count * 4
            
            if expected_size > buffer_size:
                self.logger.error(f"缓冲区大小不足: 需要{expected_size}, 实际{buffer_size}")
                return None
            
            # 创建numpy数组
            img_data = np.frombuffer(self.display_buffer[:expected_size], dtype=np.uint8)
            img_rgba = img_data.reshape((height.value, width.value, 4))
            
            # 转换为BGR格式
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            
            # 垂直翻转（基于MAA的实现）
            img_final = cv2.flip(img_bgr, 0)
            
            # 更新统计信息
            capture_time = time.time() - start_time
            self.capture_count += 1
            self.total_capture_time += capture_time
            
            self.logger.debug(f"截图成功: {img_final.shape}, 耗时: {capture_time:.3f}秒")
            
            return img_final
            
        except Exception as e:
            self.logger.error(f"截图过程中出现异常: {e}")
            return None
    
    def get_performance_info(self) -> dict:
        """获取性能信息"""
        avg_time = self.total_capture_time / self.capture_count if self.capture_count > 0 else 0
        
        return {
            'capture_initialized': self.capture_initialized,
            'capture_count': self.capture_count,
            'average_capture_time': avg_time,
            'sdk_loaded': self.sdk_loaded,
            'mumu_handle': bool(self.mumu_handle),
            'display_resolution': f"{self.display_width}x{self.display_height}",
            'display_buffer_size': len(self.display_buffer) if self.display_buffer else 0
        }
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'capture_initialized') and self.capture_initialized:
            self.logger.info("清理MUMU SDK捕获资源")
            
            # 断开连接
            if self.mumu_handle and self.disconnect_func:
                try:
                    self.disconnect_func(self.mumu_handle)
                    self.logger.info("已断开MUMU模拟器连接")
                except Exception as e:
                    self.logger.error(f"断开连接失败: {e}")


def test_mumu_sdk_maa():
    """测试MUMU SDK捕获"""
    import sys
    import os
    
    # 添加src目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..')
    sys.path.insert(0, project_root)
    
    from src.device.config import DeviceConfig
    from src.device.adb_controller import ADBController
    
    # 创建配置
    config = DeviceConfig()
    config.mumu_path = r"e:\\Program Files\\Netease\\MuMu Player 12"
    config.mumu_index = 0
    
    # 创建ADB控制器
    adb = ADBController(config)
    
    # 测试MUMU SDK捕获
    try:
        mumu_sdk = MUMUSDKMAACapture(adb, config)
        
        # 测试截图
        screenshot = mumu_sdk.screencap()
        
        if screenshot is not None:
            print(f"✅ MUMU SDK捕获成功! 图像尺寸: {screenshot.shape}")
            print(f"   像素范围: [{screenshot.min()}, {screenshot.max()}]")
            
            # 显示性能信息
            perf_info = mumu_sdk.get_performance_info()
            print(f"性能信息: {perf_info}")
            
            # 保存测试图像
            cv2.imwrite("mumu_sdk_maa_test.png", screenshot)
            print("测试图像已保存为: mumu_sdk_maa_test.png")
        else:
            print("❌ MUMU SDK捕获失败")
            
    except Exception as e:
        print(f"❌ MUMU SDK捕获测试失败: {e}")


if __name__ == "__main__":
    test_mumu_sdk_maa()