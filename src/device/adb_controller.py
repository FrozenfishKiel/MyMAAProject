"""
ADB控制器模块 - 负责与MUMU模拟器进行ADB通信和设备控制

目的：
1. 封装ADB命令执行和连接管理功能
2. 提供设备状态检测和连接重试机制
3. 实现输入控制（点击、滑动、按键等操作）
4. 支持错误分类和快速重试策略
5. 提供设备信息获取和屏幕分辨率检测

包含：
- ADB连接管理
- 设备状态监控
- 输入控制功能
- 错误处理和重试机制
- 设备信息获取
"""

import subprocess
import time
import logging
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path

from .config import DeviceConfig, ErrorType, default_config


class ADBController:
    """ADB控制器类 - 管理ADB连接和设备控制"""
    
    def __init__(self, config: DeviceConfig = default_config):
        """
        初始化ADB控制器
        
        Args:
            config: 设备配置对象
        """
        self.config = config
        self.logger = self._setup_logger()
        self.connected = False
        self.device_info: Dict[str, Any] = {}
        
        # 验证配置
        try:
            config.validate_config()
            self.logger.info("[OK] 设备配置验证成功")
        except Exception as e:
            self.logger.error(f"[ERROR] 设备配置验证失败: {e}")
            raise
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("adb_controller")
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
                # 创建日志目录
                log_path = Path(self.config.log_file_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(
                    self.config.log_file_path,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def _execute_adb_command(self, command: str, timeout: Optional[int] = None, binary_output: bool = False) -> Tuple[bool, Union[str, bytes]]:
        """
        执行ADB命令
        
        Args:
            command: ADB命令
            timeout: 命令超时时间
            binary_output: 是否返回二进制输出（用于截图等命令）
            
        Returns:
            (成功状态, 输出结果)
        """
        if timeout is None:
            timeout = self.config.adb_timeout
        
        # 如果设备已连接，自动检测并使用已连接的设备
        if self.connected:
            # 获取已连接的设备ID
            device_id = self._get_connected_device_id()
            if device_id:
                full_command = f'"{self.config.adb_path}" -s {device_id} {command}'
            else:
                # 如果没有检测到已连接的设备，使用配置的设备ID
                device_id = self.config.get_adb_connection_string()
                full_command = f'"{self.config.adb_path}" -s {device_id} {command}'
        else:
            full_command = f'"{self.config.adb_path}" {command}'
            
        self.logger.debug(f"执行ADB命令: {full_command}")
        
        for attempt in range(self.config.retry_count):
            try:
                # 根据输出类型选择不同的参数
                if binary_output:
                    # 二进制输出模式（用于截图等命令）
                    result = subprocess.run(
                        full_command,
                        shell=True,
                        capture_output=True,
                        timeout=timeout
                    )
                else:
                    # 文本输出模式（默认）
                    result = subprocess.run(
                        full_command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        encoding='utf-8'
                    )
                
                if result.returncode == 0:
                    self.logger.debug(f"ADB命令执行成功: {command}")
                    if binary_output:
                        return True, result.stdout
                    else:
                        return True, result.stdout.strip()
                else:
                    if binary_output:
                        error_msg = result.stderr.decode('utf-8', errors='ignore').strip() if result.stderr else "未知错误"
                    else:
                        error_msg = result.stderr.strip() if result.stderr else "未知错误"
                    
                    self.logger.warning(f"ADB命令执行失败 (尝试 {attempt + 1}/{self.config.retry_count}): {error_msg}")
                    
                    if attempt == self.config.retry_count - 1:
                        return False, error_msg
                    
                    time.sleep(self.config.retry_delay)
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"ADB命令超时 (尝试 {attempt + 1}/{self.config.retry_count}): {command}")
                if attempt == self.config.retry_count - 1:
                    return False, "命令执行超时"
                
            except Exception as e:
                self.logger.error(f"ADB命令执行异常 (尝试 {attempt + 1}/{self.config.retry_count}): {e}")
                if attempt == self.config.retry_count - 1:
                    return False, str(e)
        
        return False, "所有重试尝试均失败"
    
    def _get_connected_device_id(self) -> Optional[str]:
        """
        获取已连接的设备ID
        
        Returns:
            设备ID字符串或None
        """
        try:
            # 执行adb devices命令获取设备列表
            result = subprocess.run(
                f'"{self.config.adb_path}" devices',
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # 解析设备列表，找到状态为'device'的设备
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # 跳过第一行标题
                    if line.strip() and 'device' in line:
                        device_id = line.split('\t')[0].strip()
                        if device_id and device_id != 'List of devices attached':
                            self.logger.debug(f"检测到已连接设备: {device_id}")
                            return device_id
            
            self.logger.warning("未检测到已连接的设备")
            return None
            
        except Exception as e:
            self.logger.error(f"获取已连接设备ID失败: {e}")
            return None
    
    def connect_device(self) -> bool:
        """
        连接设备
        
        Returns:
            连接是否成功
        """
        self.logger.info("正在连接设备...")
        
        # 先检查ADB服务是否运行
        success, output = self._execute_adb_command("devices")
        if not success:
            self.logger.error("ADB服务检查失败")
            return False
        
        # 连接设备
        connection_string = self.config.get_adb_connection_string()
        success, output = self._execute_adb_command(f"connect {connection_string}")
        
        if success and "connected to" in output.lower():
            self.connected = True
            self.logger.info(f"✅ 设备连接成功: {connection_string}")
            
            # 获取设备信息
            self._get_device_info()
            return True
        else:
            self.logger.error(f"❌ 设备连接失败: {output}")
            return False
    
    def disconnect_device(self) -> bool:
        """
        断开设备连接
        
        Returns:
            断开是否成功
        """
        if not self.connected:
            self.logger.warning("设备未连接，无需断开")
            return True
        
        connection_string = self.config.get_adb_connection_string()
        success, output = self._execute_adb_command(f"disconnect {connection_string}")
        
        if success:
            self.connected = False
            self.device_info.clear()
            self.logger.info("✅ 设备断开成功")
            return True
        else:
            self.logger.error(f"❌ 设备断开失败: {output}")
            return False
    
    def _get_device_info(self) -> None:
        """获取设备信息"""
        if not self.connected:
            return
        
        # 获取设备型号
        success, model = self._execute_adb_command("shell getprop ro.product.model")
        if success:
            self.device_info["model"] = model
        
        # 获取Android版本
        success, version = self._execute_adb_command("shell getprop ro.build.version.release")
        if success:
            self.device_info["android_version"] = version
        
        # 获取屏幕分辨率
        success, resolution = self._execute_adb_command("shell wm size")
        if success:
            # 解析分辨率格式: Physical size: 1920x1080
            if ":" in resolution:
                resolution = resolution.split(":")[1].strip()
            self.device_info["resolution"] = resolution
        
        self.logger.info(f"设备信息: {self.device_info}")
    
    def tap(self, x: int, y: int) -> bool:
        """
        点击屏幕指定位置
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            点击是否成功
        """
        if not self.connected:
            self.logger.error("设备未连接，无法执行点击操作")
            return False
        
        command = f"shell input tap {x} {y}"
        success, output = self._execute_adb_command(command)
        
        if success:
            self.logger.debug(f"点击成功: ({x}, {y})")
        else:
            self.logger.error(f"点击失败: {output}")
        
        return success
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: Optional[int] = None) -> bool:
        """
        从起点滑动到终点
        
        Args:
            x1: 起点X坐标
            y1: 起点Y坐标
            x2: 终点X坐标
            y2: 终点Y坐标
            duration: 滑动持续时间（毫秒）
            
        Returns:
            滑动是否成功
        """
        if not self.connected:
            self.logger.error("设备未连接，无法执行滑动操作")
            return False
        
        if duration is None:
            duration = self.config.swipe_duration
        
        command = f"shell input swipe {x1} {y1} {x2} {y2} {duration}"
        success, output = self._execute_adb_command(command)
        
        if success:
            self.logger.debug(f"滑动成功: ({x1}, {y1}) -> ({x2}, {y2}), 时长: {duration}ms")
        else:
            self.logger.error(f"滑动失败: {output}")
        
        return success
    
    def key_event(self, keycode: int) -> bool:
        """
        发送按键事件
        
        Args:
            keycode: 按键代码（如3=HOME, 4=BACK, 24=VOLUME_UP等）
            
        Returns:
            按键是否成功
        """
        if not self.connected:
            self.logger.error("设备未连接，无法执行按键操作")
            return False
        
        command = f"shell input keyevent {keycode}"
        success, output = self._execute_adb_command(command)
        
        if success:
            self.logger.debug(f"按键成功: {keycode}")
        else:
            self.logger.error(f"按键失败: {output}")
        
        return success
    
    def get_screen_resolution(self) -> Optional[Tuple[int, int]]:
        """
        获取屏幕分辨率
        
        Returns:
            (宽度, 高度) 或 None
        """
        if "resolution" not in self.device_info:
            self._get_device_info()
        
        resolution = self.device_info.get("resolution", "")
        if "x" in resolution:
            try:
                width, height = map(int, resolution.split("x"))
                return width, height
            except ValueError:
                self.logger.error(f"分辨率解析失败: {resolution}")
        
        return None
    
    def is_connected(self) -> bool:
        """检查设备是否连接"""
        return self.connected
    
    def get_devices(self) -> List[str]:
        """
        获取当前连接的设备列表
        
        Returns:
            设备ID列表
        """
        try:
            # 执行adb devices命令获取设备列表
            result = subprocess.run(
                f'"{self.config.adb_path}" devices',
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # 解析设备列表，找到状态为'device'的设备
                devices = []
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # 跳过第一行标题
                    if line.strip() and 'device' in line:
                        device_id = line.split('\t')[0].strip()
                        if device_id and device_id != 'List of devices attached':
                            devices.append(device_id)
                
                self.logger.debug(f"检测到设备列表: {devices}")
                return devices
            else:
                self.logger.error(f"获取设备列表失败: {result.stderr}")
                return []
                
        except Exception as e:
            self.logger.error(f"获取设备列表异常: {e}")
            return []
    
    def deploy_operator(self, operator_type: str, position: int, 
                        game_state: Dict = None, 
                        operator_avatar_coords: Tuple[int, int] = None,
                        grid_coords: Tuple[int, int] = None) -> bool:
        """
        部署干员 - 基于手动操作流程实现，支持使用检测结果坐标
        
        操作流程：
        1. 点击干员头像（根据干员类型）
        2. 拖拽到对应的格子位置
        3. 点击拖拽选择方向
        4. 松开鼠标完成部署
        
        Args:
            operator_type: 干员类型（vanguard, guard, sniper, defender, medic, caster, supporter, specialist）
            position: 部署位置（0-7，对应8个格子）
            game_state: 游戏状态信息（可选，包含检测结果）
            operator_avatar_coords: 干员头像坐标（可选，如果提供则使用）
            grid_coords: 格子坐标（可选，如果提供则使用）
            
        Returns:
            部署是否成功
        """
        if not self.connected:
            self.logger.error("设备未连接，无法执行部署操作")
            return False
        
        # 获取屏幕分辨率
        resolution = self.get_screen_resolution()
        if not resolution:
            self.logger.error("无法获取屏幕分辨率")
            return False
        
        width, height = resolution
        
        # 1. 确定干员头像位置（优先使用检测结果，否则使用估算）
        if operator_avatar_coords:
            # 使用提供的检测坐标
            operator_x, operator_y = operator_avatar_coords
            self.logger.info(f"使用检测到的干员头像坐标: ({operator_x}, {operator_y})")
        elif game_state and 'available_operators' in game_state:
            # 从游戏状态中获取干员头像位置
            operator_coords = self._get_operator_coords_from_game_state(game_state, operator_type)
            if operator_coords:
                operator_x, operator_y = operator_coords
                self.logger.info(f"从游戏状态获取干员头像坐标: ({operator_x}, {operator_y})")
            else:
                # 使用估算坐标
                operator_x, operator_y = self._get_estimated_operator_position(width, height, operator_type)
                self.logger.info(f"使用估算的干员头像坐标: ({operator_x}, {operator_y})")
        else:
            # 使用估算坐标
            operator_x, operator_y = self._get_estimated_operator_position(width, height, operator_type)
            self.logger.info(f"使用估算的干员头像坐标: ({operator_x}, {operator_y})")
        
        # 2. 确定部署格子位置（优先使用检测结果，否则使用估算）
        if grid_coords:
            # 使用提供的检测坐标
            grid_x, grid_y = grid_coords
            self.logger.info(f"使用检测到的格子坐标: ({grid_x}, {grid_y})")
        elif game_state and 'deployable_positions' in game_state:
            # 从游戏状态中获取格子位置
            grid_coords = self._get_grid_coords_from_game_state(game_state, position)
            if grid_coords:
                grid_x, grid_y = grid_coords
                self.logger.info(f"从游戏状态获取格子坐标: ({grid_x}, {grid_y})")
            else:
                # 使用估算坐标
                grid_x, grid_y = self._get_estimated_grid_position(width, height, position)
                self.logger.info(f"使用估算的格子坐标: ({grid_x}, {grid_y})")
        else:
            # 使用估算坐标
            grid_x, grid_y = self._get_estimated_grid_position(width, height, position)
            self.logger.info(f"使用估算的格子坐标: ({grid_x}, {grid_y})")
        
        # 3. 执行部署操作
        try:
            # 第一步：点击干员头像
            self.logger.info(f"点击干员头像: {operator_type} 位置({operator_x}, {operator_y})")
            if not self.tap(operator_x, operator_y):
                return False
            
            # 等待短暂时间让游戏响应
            time.sleep(0.5)
            
            # 第二步：拖拽到格子位置
            self.logger.info(f"拖拽到格子位置: {position} 位置({grid_x}, {grid_y})")
            if not self.swipe(operator_x, operator_y, grid_x, grid_y, duration=500):
                return False
            
            # 等待短暂时间
            time.sleep(0.3)
            
            # 第三步：选择部署方向（简化处理：向上部署）
            direction_x, direction_y = self._get_direction_position(grid_x, grid_y, "up")
            self.logger.info(f"选择部署方向: 向上 目标位置({direction_x}, {direction_y})")
            
            # 从格子位置拖拽到方向位置
            if not self.swipe(grid_x, grid_y, direction_x, direction_y, duration=300):
                return False
            
            # 第四步：松开鼠标（通过点击完成部署）
            time.sleep(0.2)
            if not self.tap(direction_x, direction_y):
                return False
            
            self.logger.info(f"✅ 干员部署成功: {operator_type} -> 位置{position} -> 向上")
            return True
            
        except Exception as e:
            self.logger.error(f"部署干员过程中出现错误: {e}")
            return False
    
    def _get_operator_positions(self, width: int, height: int) -> Dict[str, Tuple[int, int]]:
        """
        获取干员头像位置映射
        
        Args:
            width: 屏幕宽度
            height: 屏幕高度
            
        Returns:
            干员类型到坐标的映射
        """
        # 简化处理：假设干员头像在屏幕底部区域
        # 实际实现需要根据游戏UI布局调整
        base_y = height * 0.85  # 底部85%位置
        
        # 8类干员在水平方向均匀分布
        positions = {}
        operator_types = ['vanguard', 'guard', 'sniper', 'defender', 
                         'medic', 'caster', 'supporter', 'specialist']
        
        for i, op_type in enumerate(operator_types):
            x = width * (i + 0.5) / len(operator_types)  # 水平居中
            positions[op_type] = (int(x), int(base_y))
        
        return positions
    
    def _get_grid_positions(self, width: int, height: int) -> List[Tuple[int, int]]:
        """
        获取部署格子位置
        
        Args:
            width: 屏幕宽度
            height: 屏幕高度
            
        Returns:
            8个部署位置的坐标列表
        """
        # 简化处理：假设部署区域在屏幕中央区域
        # 实际实现需要根据游戏地图布局调整
        base_x = width * 0.3
        base_y = height * 0.4
        grid_width = width * 0.4
        grid_height = height * 0.3
        
        positions = []
        
        # 2行4列的网格布局
        for row in range(2):
            for col in range(4):
                x = base_x + grid_width * (col + 0.5) / 4
                y = base_y + grid_height * (row + 0.5) / 2
                positions.append((int(x), int(y)))
        
        return positions
    
    def _get_operator_coords_from_game_state(self, game_state: Dict, operator_type: str) -> Optional[Tuple[int, int]]:
        """
        从游戏状态中获取干员头像坐标 - 充分利用已有检测能力
        
        Args:
            game_state: 游戏状态信息
            operator_type: 干员类型
            
        Returns:
            干员头像中心坐标 (x, y) 或 None
        """
        # 方法1：从available_operators获取（编队界面）
        if 'available_operators' in game_state and game_state['available_operators']:
            # 查找对应类型的干员（根据检测到的干员类型）
            for operator in game_state['available_operators']:
                # 如果检测器提供了干员类型信息，可以精确匹配
                if 'avatar_bbox' in operator:
                    x, y, w, h = operator['avatar_bbox']
                    # 返回头像中心点（点击位置）
                    return (x + w // 2, y + h // 2)
        
        # 方法2：从operator_types获取（干员职业图标检测）
        if 'operator_types' in game_state and game_state['operator_types']:
            # 查找对应类型的干员职业图标
            for operator_info in game_state['operator_types']:
                if operator_info.get('role') == operator_type and 'bbox' in operator_info:
                    x, y, w, h = operator_info['bbox']
                    # 返回职业图标中心点
                    return (x + w // 2, y + h // 2)
        
        # 方法3：如果检测到了干员但无法匹配类型，使用第一个检测到的干员
        if 'available_operators' in game_state and game_state['available_operators']:
            first_operator = game_state['available_operators'][0]
            if 'avatar_bbox' in first_operator:
                x, y, w, h = first_operator['avatar_bbox']
                self.logger.warning(f"使用第一个检测到的干员头像（类型匹配失败: {operator_type}）")
                return (x + w // 2, y + h // 2)
        
        return None
    
    def _get_grid_coords_from_game_state(self, game_state: Dict, position: int) -> Optional[Tuple[int, int]]:
        """
        从游戏状态中获取部署格子坐标 - 充分利用已有检测能力
        
        Args:
            game_state: 游戏状态信息
            position: 部署位置索引
            
        Returns:
            格子中心坐标 (x, y) 或 None
        """
        # 方法1：从deployed_operators获取已部署干员位置（避免重复部署）
        if 'deployed_operators' in game_state and game_state['deployed_operators']:
            # 检查该位置是否已被占用
            deployed_positions = []
            for operator in game_state['deployed_operators']:
                if 'bbox' in operator:
                    x, y, w, h = operator['bbox']
                    deployed_positions.append((x + w // 2, y + h // 2))
            
            # 如果该位置已被占用，可以选择相邻位置
            if len(deployed_positions) > position:
                self.logger.warning(f"位置{position}已被占用，使用相邻位置")
                # 简单处理：使用下一个可用位置
                for i in range(len(deployed_positions), 8):
                    if i != position:
                        return self._get_estimated_grid_position(
                            self.get_screen_resolution()[0] if self.get_screen_resolution() else 1920,
                            self.get_screen_resolution()[1] if self.get_screen_resolution() else 1080,
                            i
                        )
        
        # 方法2：如果有地图格子检测结果，使用精确坐标
        # 这里可以集成Arknights-Tile-Pos-main的检测结果
        # 目前先使用估算坐标，但保留接口
        
        # 方法3：使用估算坐标（保底方案）
        resolution = self.get_screen_resolution()
        if resolution:
            width, height = resolution
            return self._get_estimated_grid_position(width, height, position)
        
        return None
    
    def _get_estimated_operator_position(self, width: int, height: int, operator_type: str) -> Tuple[int, int]:
        """
        估算干员头像位置（当检测结果不可用时）
        
        Args:
            width: 屏幕宽度
            height: 屏幕高度
            operator_type: 干员类型
            
        Returns:
            估算的干员头像坐标
        """
        # 使用之前的估算逻辑
        base_y = height * 0.85  # 底部85%位置
        operator_types = ['vanguard', 'guard', 'sniper', 'defender', 
                         'medic', 'caster', 'supporter', 'specialist']
        
        if operator_type in operator_types:
            index = operator_types.index(operator_type)
            x = width * (index + 0.5) / len(operator_types)
            return (int(x), int(base_y))
        else:
            # 默认返回第一个位置
            return (int(width * 0.5 / len(operator_types)), int(base_y))
    
    def _get_estimated_grid_position(self, width: int, height: int, position: int) -> Tuple[int, int]:
        """
        估算部署格子位置（当检测结果不可用时）
        
        Args:
            width: 屏幕宽度
            height: 屏幕高度
            position: 部署位置索引
            
        Returns:
            估算的格子坐标
        """
        # 使用之前的估算逻辑
        base_x = width * 0.3
        base_y = height * 0.4
        grid_width = width * 0.4
        grid_height = height * 0.3
        
        if 0 <= position < 8:
            row = position // 4
            col = position % 4
            x = base_x + grid_width * (col + 0.5) / 4
            y = base_y + grid_height * (row + 0.5) / 2
            return (int(x), int(y))
        else:
            # 默认返回第一个位置
            return (int(base_x + grid_width * 0.5 / 4), int(base_y + grid_height * 0.5 / 2))

    def _get_direction_position(self, grid_x: int, grid_y: int, direction: str) -> Tuple[int, int]:
        """
        获取部署方向的目标位置
        
        Args:
            grid_x: 格子X坐标
            grid_y: 格子Y坐标
            direction: 部署方向（up, down, left, right）
            
        Returns:
            方向目标位置
        """
        # 方向偏移量（像素）
        offset = 100  # 可以根据屏幕分辨率调整
        
        if direction == "up":
            return grid_x, grid_y - offset
        elif direction == "down":
            return grid_x, grid_y + offset
        elif direction == "left":
            return grid_x - offset, grid_y
        elif direction == "right":
            return grid_x + offset, grid_y
        else:
            # 默认向上
            return grid_x, grid_y - offset

    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        return self.device_info.copy()

    def __enter__(self):
        """上下文管理器入口"""
        self.connect_device()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect_device()


# 常用按键代码常量
class KeyCodes:
    """Android按键代码常量"""
    HOME = 3
    BACK = 4
    MENU = 82
    VOLUME_UP = 24
    VOLUME_DOWN = 25
    POWER = 26
    ENTER = 66
    DELETE = 67


if __name__ == "__main__":
    """ADB控制器测试代码"""
    # 创建ADB控制器实例
    adb = ADBController()
    
    try:
        # 连接设备
        if adb.connect_device():
            print("✅ 设备连接成功")
            
            # 获取设备信息
            info = adb.get_device_info()
            print(f"设备信息: {info}")
            
            # 获取屏幕分辨率
            resolution = adb.get_screen_resolution()
            if resolution:
                print(f"屏幕分辨率: {resolution[0]}x{resolution[1]}")
            
            # 测试点击操作（点击屏幕中心）
            if resolution:
                center_x, center_y = resolution[0] // 2, resolution[1] // 2
                print(f"测试点击屏幕中心: ({center_x}, {center_y})")
                # adb.tap(center_x, center_y)  # 注释掉实际点击，避免误操作
            
            # 断开连接
            adb.disconnect_device()
            print("✅ 设备断开成功")
        else:
            print("❌ 设备连接失败")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")