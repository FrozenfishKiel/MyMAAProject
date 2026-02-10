"""
高性能捕获模块 - 基于MAA技术方案实现

包含：
- MinicapStream: 流式屏幕捕获 (10-30FPS)
- EmulatorSDK: 模拟器专用捕获 (30-60FPS) 
- CaptureAgent: 智能选择代理
- ScreencapHelper: 图像解码辅助

基于MAA开源项目的技术方案实现
"""

from .capture_agent import CaptureAgent
from .minicap_stream import MinicapStream
from .mumu_sdk_maa import MUMUSDKMAACapture
from .basic_adb_capture import BasicADBCapture
from .screencap_helper import ScreencapHelper

__all__ = ['CaptureAgent', 'MinicapStream', 'MUMUSDKMAACapture', 'BasicADBCapture', 'ScreencapHelper']