"""
设备控制模块配置文件 - 定义MUMU模拟器设备控制相关的所有配置参数

目的：
1. 集中管理设备控制模块的所有配置参数
2. 提供MUMU模拟器路径、ADB连接参数、截图配置等
3. 支持日志系统配置和错误处理策略
4. 便于后续维护和参数调整

包含：
- MUMU模拟器路径配置
- ADB连接参数设置  
- 屏幕截图配置
- 输入控制参数
- 错误处理策略
- 日志系统配置
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class ErrorType(Enum):
    """错误类型枚举 - 用于区分不同类型的错误"""
    CONNECTION_ERROR = "connection_error"      # 连接错误
    ADB_ERROR = "adb_error"                    # ADB错误
    EXECUTION_ERROR = "execution_error"        # 执行错误
    TIMEOUT_ERROR = "timeout_error"            # 超时错误
    SCREENSHOT_ERROR = "screenshot_error"      # 截图错误
    INPUT_ERROR = "input_error"                # 输入错误


@dataclass
class DeviceConfig:
    """设备配置类 - 存储所有设备相关的配置参数"""
    
    # MUMU模拟器配置
    mumu_path: str = r"E:\Program Files\Netease\MuMu Player 12"
    adb_path: str = os.path.join(mumu_path, "shell", "adb.exe")
    
    # ADB连接配置
    adb_host: str = "127.0.0.1"
    adb_port: int = 7555
    adb_timeout: int = 10  # ADB命令执行超时时间（秒）
    
    # 截图配置
    screenshot_format: str = "PNG"
    screenshot_quality: int = 85  # PNG压缩质量（0-100）
    screenshot_interval: float = 0.5  # 截图间隔（秒）
    screenshot_retry_count: int = 3  # 截图重试次数
    
    # 输入控制配置
    input_delay: float = 0.0  # 输入延迟（秒）
    tap_duration: int = 50  # 点击持续时间（毫秒）
    swipe_duration: int = 500  # 滑动持续时间（毫秒）
    
    # 错误处理配置
    retry_count: int = 3  # 快速重试次数
    retry_delay: float = 0.1  # 重试间隔（秒）
    max_connection_attempts: int = 5  # 最大连接尝试次数
    
    # 日志配置
    log_level: str = "DEBUG"  # 日志级别
    log_to_console: bool = True  # 是否输出到控制台
    log_to_file: bool = True  # 是否写入文件
    log_file_path: str = "logs/device_control.log"  # 日志文件路径
    log_file_max_size: int = 10 * 1024 * 1024  # 日志文件最大大小（10MB）
    log_file_backup_count: int = 5  # 备份文件数量
    
    # 性能配置
    max_screenshot_queue_size: int = 10  # 截图队列最大大小
    screenshot_compression: bool = True  # 是否启用截图压缩
    screenshot_cache_timeout: float = 0.1  # 截图缓存超时时间（秒）
    
    # 高性能捕获配置 - 基于MAA方案
    high_performance_capture_enabled: bool = True  # 是否启用高性能捕获
    minicap_path: str = "tools/minicap"  # minicap工具路径
    emulator_sdk_path: str = "tools/mumu_sdk"  # 模拟器SDK路径
    performance_test_timeout: float = 5.0  # 性能测试超时时间（秒）
    max_fps: int = 30  # 最大帧率限制
    preferred_capture_method: str = "auto"  # 首选捕获方法：auto, minicap_stream, emulator_sdk
    
    # Minicap流式捕获配置
    minicap_stream_enabled: bool = True  # 是否启用minicap流式捕获
    minicap_socket_port: int = 1313  # minicap socket端口
    minicap_frame_timeout: float = 2.0  # 帧等待超时时间（秒）
    minicap_buffer_size: int = 1024 * 1024  # 缓冲区大小（1MB）
    minicap_quality: int = 80  # JPEG压缩质量（0-100）
    
    # 模拟器SDK配置
    emulator_sdk_enabled: bool = True  # 是否启用模拟器SDK
    emulator_type: str = "mumu"  # 模拟器类型：mumu, bluestacks, ldplayer
    emulator_display_id: int = 0  # 显示ID
    emulator_buffer_format: str = "RGBA"  # 缓冲区格式
    
    # 智能选择配置
    auto_method_selection: bool = True  # 是否启用自动方法选择
    speed_test_interval: float = 300.0  # 性能测试间隔（秒）
    fallback_to_adb: bool = True  # 是否回退到ADB
    
    # 错误处理和重试配置
    max_connection_retries: int = 3  # 最大连接重试次数
    connection_retry_delay: float = 1.0  # 连接重试延迟（秒）
    capture_timeout: float = 10.0  # 捕获超时时间（秒）
    
    # 图像处理配置
    image_processing_enabled: bool = True  # 是否启用图像处理
    convert_to_bgr: bool = True  # 是否转换为BGR格式
    flip_vertical: bool = False  # 是否垂直翻转
    flip_horizontal: bool = False  # 是否水平翻转
    
    # 调试和监控配置
    enable_performance_monitoring: bool = True  # 是否启用性能监控
    log_capture_statistics: bool = True  # 是否记录捕获统计信息
    debug_mode: bool = False  # 调试模式
    
    def get_adb_connection_string(self) -> str:
        """获取ADB连接字符串"""
        return f"{self.adb_host}:{self.adb_port}"
    
    def validate_config(self) -> bool:
        """验证配置参数是否有效"""
        # 检查MUMU路径是否存在
        if not os.path.exists(self.mumu_path):
            raise FileNotFoundError(f"MUMU模拟器路径不存在: {self.mumu_path}")
        
        # 检查ADB工具是否存在
        if not os.path.exists(self.adb_path):
            raise FileNotFoundError(f"ADB工具不存在: {self.adb_path}")
        
        # 验证参数范围
        if not 0 <= self.screenshot_quality <= 100:
            raise ValueError("截图质量参数必须在0-100之间")
        
        if self.screenshot_interval <= 0:
            raise ValueError("截图间隔必须大于0")
        
        return True
    
    def validate(self) -> bool:
        """验证配置参数是否有效（validate_config的别名方法）"""
        return self.validate_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式"""
        return {
            "mumu_path": self.mumu_path,
            "adb_path": self.adb_path,
            "adb_host": self.adb_host,
            "adb_port": self.adb_port,
            "adb_timeout": self.adb_timeout,
            "screenshot_format": self.screenshot_format,
            "screenshot_quality": self.screenshot_quality,
            "screenshot_interval": self.screenshot_interval,
            "input_delay": self.input_delay,
            "retry_count": self.retry_count,
            "log_level": self.log_level,
            "log_to_console": self.log_to_console,
            "log_to_file": self.log_to_file,
            "log_file_path": self.log_file_path
        }


# 创建默认配置实例
default_config = DeviceConfig()


if __name__ == "__main__":
    """配置模块测试代码"""
    config = DeviceConfig()
    
    try:
        config.validate_config()
        print("✅ 设备配置验证成功")
        print(f"MUMU路径: {config.mumu_path}")
        print(f"ADB工具: {config.adb_path}")
        print(f"连接地址: {config.get_adb_connection_string()}")
        print(f"截图间隔: {config.screenshot_interval}秒")
        print(f"重试次数: {config.retry_count}次")
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")