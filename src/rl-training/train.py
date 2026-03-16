from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# 添加项目路径到sys.path
ROOT = Path(__file__).resolve().parent.parent
DEPS_ROOT = ROOT.parent / "maa-deps" / "maafw-5.2.6-win_amd64"
if DEPS_ROOT.exists():
    sys.path.insert(0, str(DEPS_ROOT))
sys.path.insert(0, str(ROOT / "ai-plugins"))
sys.path.insert(0, str(ROOT / "rl-environment"))

from yolo_recognizer import YoloRecognizer
from template_matcher import TemplateMatcher
from game_env import GameEnv


def check_pause_training() -> bool:
    """
    检查是否需要暂停训练
    
    Returns:
        bool: 是否需要暂停训练
    """
    # 检查是否按下暂停键（P键）
    try:
        import msvcrt
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'p' or key == b'P':
                return True
    except:
        pass
    return False


class TrainingCallback(BaseCallback):
    """
    训练回调函数
    
    用于记录训练过程中的信息
    """
    
    def __init__(self, verbose: int = 0, check_pause_func=None) -> None:
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.check_pause_func = check_pause_func
        self.paused = False
        self.pause_count = 0
    
    def _on_step(self) -> None:
        """
        每一步之后调用
        """
        # 检查是否需要暂停训练
        if self.check_pause_func and self.check_pause_func():
            if not self.paused:
                # 第一次检测到需要暂停
                self.paused = True
                self.pause_count += 1
                print(f"⏸️  暂停训练 #{self.pause_count}")
                print("   请在过场动画结束后按任意键继续...")
            # 跳过这一步
            return
        
        # 检查是否从暂停中恢复
        if self.paused:
            self.paused = False
            print(f"▶️  继续训练 #{self.pause_count}")
        
        # 获取当前奖励
        if "rewards" in self.locals:
            reward = self.locals["rewards"][0]
            self.current_episode_reward += reward
            self.current_episode_length += 1
        
        # 检查是否episode结束
        if "dones" in self.locals and self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            if self.verbose > 0:
                print(f"Episode {len(self.episode_rewards)}: reward = {self.current_episode_reward:.2f}, length = {self.current_episode_length}")
            
            # 重置当前episode
            self.current_episode_reward = 0.0
            self.current_episode_length = 0


def train_rl_model(
    controller: Any,
    yolo_model_path: str,
    rl_model_save_path: str,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    verbose: int = 1
) -> None:
    """
    训练RL模型
    
    Args:
        controller: MaaFramework控制器
        yolo_model_path: YOLO模型路径
        rl_model_save_path: RL模型保存路径
        total_timesteps: 总训练步数
        learning_rate: 学习率
        n_steps: 每次更新的步数
        batch_size: 批次大小
        n_epochs: 训练轮数
        gamma: 折扣因子
        gae_lambda: GAE lambda参数
        verbose: 日志详细程度
    """
    # 加载YOLO识别器
    print(f"Loading YOLO model from {yolo_model_path}...")
    
    # 检查CUDA是否可用
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    yolo_recognizer = YoloRecognizer(model_path=yolo_model_path, device=device)
    yolo_recognizer.load()
    print("YOLO model loaded successfully!")
    
    # 加载模板匹配识别器
    print("Loading template matcher...")
    template_matcher = TemplateMatcher(controller)
    template_matcher.load()
    print("Template matcher loaded successfully!")
    
    # 创建RL环境
    print("Creating RL environment...")
    env = DummyVecEnv([lambda: GameEnv(controller, yolo_recognizer, template_matcher)])
    print("RL environment created successfully!")
    
    # 创建训练回调函数
    callback = TrainingCallback(verbose=verbose, check_pause_func=check_pause_training)
    
    # 创建PPO模型
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        tensorboard_log=None,
    )
    print("PPO model created successfully!")
    
    # 训练模型
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    print("Training completed!")
    
    # 保存模型
    print(f"Saving model to {rl_model_save_path}...")
    os.makedirs(os.path.dirname(rl_model_save_path), exist_ok=True)
    model.save(rl_model_save_path)
    print("Model saved successfully!")
    
    # 打印训练统计信息
    print("\nTraining statistics:")
    print(f"Total episodes: {len(callback.episode_rewards)}")
    if len(callback.episode_rewards) > 0:
        print(f"Average reward: {np.mean(callback.episode_rewards):.2f}")
        print(f"Max reward: {np.max(callback.episode_rewards):.2f}")
        print(f"Min reward: {np.min(callback.episode_rewards):.2f}")
    else:
        print("⚠️  No episodes completed during training")
        print("   This may indicate an issue with the environment setup")


def main() -> None:
    """
    主函数
    """
    # 配置参数
    # 使用绝对路径
    yolo_model_path = r"D:\BiShe\MaAutomaton-main\MaAutomaton-main\models\yolo\best.pt"
    rl_model_save_path = r"D:\BiShe\MaAutomaton-main\MaAutomaton-main\models\rl\policy.zip"
    
    # 创建MaaFramework控制器
    # 使用正确的 MaaFramework API
    try:
        from maa.controller import AdbController
        from maa.toolkit import Toolkit
        
        # 初始化 MaaFramework
        Toolkit.init_option(str(ROOT))
        
        # 查找 ADB 设备
        print("⏳ 正在查找 ADB 设备...")
        adb_devices = Toolkit.find_adb_devices()
        
        if not adb_devices:
            print("❌ 未找到 ADB 设备")
            print("  请确保:")
            print("    1. MuMu 模拟器已启动")
            print("    2. ADB 调试已开启")
            print("    3. 端口配置正确(默认 7555)")
            controller = None
        else:
            print(f"✅ 找到 {len(adb_devices)} 个设备:")
            for i, device in enumerate(adb_devices):
                print(f"    设备 {i+1}: {device.name} ({device.address})")
            
            # 使用第一个设备
            device = adb_devices[0]
            print(f"\n📱 设备配置:")
            print(f"   - 类型: ADB")
            print(f"   - ADB 路径: {device.adb_path}")
            print(f"   - 设备地址: {device.address}")
            print()
            
            # 创建控制器
            print("⏳ 正在连接设备...")
            controller = AdbController(
                adb_path=device.adb_path,
                address=device.address,
                screencap_methods=device.screencap_methods,
                input_methods=device.input_methods,
                config=device.config,
            )
            
            # 连接设备
            connection_job = controller.post_connection()
            connection_job.wait()
            
            if connection_job.succeeded:
                print("✅ 设备连接成功!")
                print()
            else:
                print("❌ 设备连接失败")
                controller = None
                print()
    except Exception as e:
        print(f"❌ MaaFramework 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        controller = None
        print()
    
    # 训练RL模型
    train_rl_model(
        controller=controller,
        yolo_model_path=yolo_model_path,
        rl_model_save_path=rl_model_save_path,
        total_timesteps=100000,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1
    )


if __name__ == "__main__":
    main()
