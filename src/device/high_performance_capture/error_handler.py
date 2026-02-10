"""
错误处理模块 - 基于MAA方案的高性能捕获错误处理

技术原理：
1. 统一的错误分类和编码系统
2. 分级错误处理策略
3. 自动重试和恢复机制
4. 详细的错误日志和统计

基于MAA项目的错误处理模式实现。
"""

import logging
import time
from enum import Enum
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass


class CaptureErrorCode(Enum):
    """捕获错误代码枚举 - 基于MAA的错误分类"""
    
    # 连接错误
    CONNECTION_FAILED = "CONN001"  # 连接失败
    DEVICE_NOT_FOUND = "CONN002"   # 设备未找到
    ADB_NOT_AVAILABLE = "CONN003"  # ADB不可用
    SOCKET_ERROR = "CONN004"       # Socket错误
    
    # 捕获错误
    CAPTURE_TIMEOUT = "CAPT001"    # 捕获超时
    CAPTURE_FAILED = "CAPT002"     # 捕获失败
    FRAME_DROPPED = "CAPT003"      # 帧丢失
    STREAM_BROKEN = "CAPT004"      # 流中断
    
    # 解码错误
    DECODE_FAILED = "DECO001"      # 解码失败
    INVALID_FORMAT = "DECO002"     # 无效格式
    BUFFER_CORRUPTED = "DECO003"   # 缓冲区损坏
    
    # SDK错误
    SDK_NOT_LOADED = "SDK001"      # SDK未加载
    SDK_FUNCTION_ERROR = "SDK002"  # SDK函数错误
    DISPLAY_NOT_FOUND = "SDK003"   # 显示未找到
    
    # 配置错误
    CONFIG_INVALID = "CONF001"     # 配置无效
    PATH_NOT_FOUND = "CONF002"     # 路径不存在
    PERMISSION_DENIED = "CONF003"  # 权限被拒绝


@dataclass
class CaptureError:
    """捕获错误信息类"""
    
    code: CaptureErrorCode
    message: str
    method: str
    timestamp: float
    retry_count: int = 0
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """格式化错误信息"""
        return f"[{self.code.value}] {self.method}: {self.message}"


class ErrorHandler:
    """错误处理器 - 基于MAA的实现"""
    
    def __init__(self, config):
        """
        初始化错误处理器
        
        Args:
            config: 设备配置对象
        """
        self.config = config
        self.logger = logging.getLogger("error_handler")
        
        # 错误统计
        self.error_stats: Dict[CaptureErrorCode, int] = {}
        self.last_error_time: float = 0.0
        self.consecutive_errors: int = 0
        
        # 错误处理策略
        self.retry_strategies: Dict[CaptureErrorCode, Callable] = {
            CaptureErrorCode.CONNECTION_FAILED: self._retry_connection,
            CaptureErrorCode.CAPTURE_TIMEOUT: self._retry_capture,
            CaptureErrorCode.SDK_NOT_LOADED: self._retry_sdk,
        }
        
        self.logger.info("ErrorHandler初始化完成")
    
    def handle_error(self, error: CaptureError) -> bool:
        """
        处理捕获错误 - 基于MAA的错误处理策略
        
        Args:
            error: 捕获错误对象
            
        Returns:
            是否应该重试
        """
        try:
            # 更新错误统计
            self._update_error_stats(error)
            
            # 记录错误日志
            self._log_error(error)
            
            # 检查是否达到错误阈值
            if self._should_escalate(error):
                self._escalate_error(error)
                return False
            
            # 应用重试策略
            if error.code in self.retry_strategies:
                return self.retry_strategies[error.code](error)
            
            # 默认重试策略
            return self._default_retry(error)
            
        except Exception as e:
            self.logger.error(f"错误处理异常: {e}")
            return False
    
    def _update_error_stats(self, error: CaptureError) -> None:
        """更新错误统计信息"""
        # 更新错误计数
        if error.code not in self.error_stats:
            self.error_stats[error.code] = 0
        self.error_stats[error.code] += 1
        
        # 更新连续错误计数
        current_time = time.time()
        if current_time - self.last_error_time < 60:  # 1分钟内
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 1
        
        self.last_error_time = current_time
    
    def _log_error(self, error: CaptureError) -> None:
        """记录错误日志"""
        log_message = f"捕获错误: {error}"
        
        if error.details:
            details_str = ", ".join([f"{k}={v}" for k, v in error.details.items()])
            log_message += f" | 详情: {details_str}"
        
        if error.retry_count > 0:
            log_message += f" | 重试次数: {error.retry_count}"
        
        # 根据错误严重性选择日志级别
        if error.code in [CaptureErrorCode.CONNECTION_FAILED, 
                         CaptureErrorCode.SDK_NOT_LOADED]:
            self.logger.error(log_message)
        elif error.code in [CaptureErrorCode.CAPTURE_TIMEOUT,
                           CaptureErrorCode.CAPTURE_FAILED]:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _should_escalate(self, error: CaptureError) -> bool:
        """检查是否需要升级错误处理"""
        # 检查连续错误次数
        if self.consecutive_errors >= self.config.max_connection_retries:
            return True
        
        # 检查特定错误类型
        if (error.code == CaptureErrorCode.CONNECTION_FAILED and 
            error.retry_count >= self.config.max_connection_retries):
            return True
        
        return False
    
    def _escalate_error(self, error: CaptureError) -> None:
        """升级错误处理"""
        self.logger.error(f"错误升级: {error} - 连续错误次数: {self.consecutive_errors}")
        
        # 重置错误计数器
        self.consecutive_errors = 0
        
        # 记录错误升级事件
        escalation_details = {
            'consecutive_errors': self.consecutive_errors,
            'error_code': error.code.value,
            'method': error.method,
            'timestamp': error.timestamp
        }
        
        self.logger.error(f"错误升级详情: {escalation_details}")
    
    def _retry_connection(self, error: CaptureError) -> bool:
        """连接错误重试策略"""
        if error.retry_count < self.config.max_connection_retries:
            self.logger.info(f"连接错误重试: {error.retry_count + 1}/{self.config.max_connection_retries}")
            time.sleep(self.config.connection_retry_delay)
            return True
        return False
    
    def _retry_capture(self, error: CaptureError) -> bool:
        """捕获错误重试策略"""
        if error.retry_count < 2:  # 捕获错误最多重试2次
            self.logger.info(f"捕获错误重试: {error.retry_count + 1}/2")
            time.sleep(0.5)  # 捕获错误重试间隔较短
            return True
        return False
    
    def _retry_sdk(self, error: CaptureError) -> bool:
        """SDK错误重试策略"""
        if error.retry_count < 1:  # SDK错误只重试1次
            self.logger.info("SDK错误重试: 1/1")
            time.sleep(1.0)
            return True
        return False
    
    def _default_retry(self, error: CaptureError) -> bool:
        """默认重试策略"""
        if error.retry_count < 1:  # 默认只重试1次
            self.logger.info("默认错误重试: 1/1")
            time.sleep(0.5)
            return True
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        return {
            'total_errors': sum(self.error_stats.values()),
            'error_counts': {code.value: count for code, count in self.error_stats.items()},
            'consecutive_errors': self.consecutive_errors,
            'last_error_time': self.last_error_time
        }
    
    def reset_statistics(self) -> None:
        """重置错误统计"""
        self.error_stats.clear()
        self.consecutive_errors = 0
        self.last_error_time = 0.0
        self.logger.info("错误统计已重置")


def create_error(code: CaptureErrorCode, message: str, method: str, 
                details: Optional[Dict[str, Any]] = None) -> CaptureError:
    """
    创建错误对象 - 便捷函数
    
    Args:
        code: 错误代码
        message: 错误消息
        method: 方法名称
        details: 错误详情
        
    Returns:
        捕获错误对象
    """
    return CaptureError(
        code=code,
        message=message,
        method=method,
        timestamp=time.time(),
        details=details
    )