from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.ai_plugins.yolo_recognizer import YoloRecognizer
from src.rl_environment.game_env import GameEnv


class TrainingCallback(BaseCallback):
    """
    训练回调函数
    
    用于记录训练过程中的信息
    """
    
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def _on_step(self) -> None:
        """
        每一步之后调用
        """
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
    yolo_recognizer = YoloRecognizer(model_path=yolo_model_path, device="cuda")
    yolo_recognizer.load()
    print("YOLO model loaded successfully!")
    
    # 创建RL环境
    print("Creating RL environment...")
    env = GameEnv(controller, yolo_recognizer)
    env = DummyVecEnv([env])
    print("RL environment created successfully!")
    
    # 创建训练回调函数
    callback = TrainingCallback(verbose=verbose)
    
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
        tensorboard_log="./rl_training_logs",
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
    print(f"Average reward: {np.mean(callback.episode_rewards):.2f}")
    print(f"Max reward: {np.max(callback.episode_rewards):.2f}")
    print(f"Min reward: {np.min(callback.episode_rewards):.2f}")


def main() -> None:
    """
    主函数
    """
    # 配置参数
    yolo_model_path = "models/yolo/best.pt"
    rl_model_save_path = "models/rl/policy.zip"
    
    # 创建MaaFramework控制器
    # 这里需要根据实际情况配置
    # 例如：
    # device_config = {
    #     "type": "adb",
    #     "adb_path": "path/to/adb.exe",
    #     "address": "127.0.0.1:5555",
    #     "screencap_methods": ["minicap", "raw"],
    #     "config": {"screencap_raw_stream": True}
    # }
    # from src.maa_wrapper.runtime import MaaFwAdapter
    # adapter = MaaFwAdapter(device_config)
    # adapter.connect()
    # controller = adapter._controller
    
    # 暂时使用None，实际使用时需要替换为真实的controller
    controller = None
    
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
