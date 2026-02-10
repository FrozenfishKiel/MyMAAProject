"""
PPO训练管理器
实现完整的PPO训练循环：经验收集、参数更新、模型保存、训练监控
"""

import torch
import numpy as np
import time
import os
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime

from .ppo_agent import PPOBattleAgent
from .game_environment import GameEnvironment


class PPOTrainer:
    """PPO训练管理器 - 实现完整的训练循环"""
    
    def __init__(self, 
                 config_path: str = "config/config.yaml",
                 model_save_dir: str = "models/ppo",
                 log_dir: str = "logs/ppo"):
        """
        初始化PPO训练管理器
        
        Args:
            config_path: 配置文件路径
            model_save_dir: 模型保存目录
            log_dir: 日志保存目录
        """
        self.config_path = config_path
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        
        # 创建目录
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化PPO智能体和游戏环境
        self.agent = PPOBattleAgent(state_dim=50, action_dim=14)
        self.env = GameEnvironment(config_path)
        
        # 训练参数
        self.max_episodes = 1000  # 最大训练回合数
        self.max_steps_per_episode = 200  # 每回合最大步数
        self.update_frequency = 10  # 每N回合更新一次参数
        self.batch_size = 64  # 批次大小
        
        # 训练状态
        self.current_episode = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        
        # 经验缓冲区
        self.experience_buffer = []
        
        # 训练日志
        self.training_log = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print("✅ PPO训练管理器初始化完成")
    
    def collect_experience(self, num_episodes: int = 1) -> List[Dict[str, Any]]:
        """
        收集训练经验
        
        Args:
            num_episodes: 收集经验的回合数
            
        Returns:
            experiences: 经验列表
        """
        all_experiences = []
        
        for episode in range(num_episodes):
            print(f"🎮 开始收集经验 - 回合 {episode + 1}/{num_episodes}")
            
            # 重置环境
            state = self.env.reset()
            episode_experiences = []
            episode_reward = 0
            episode_steps = 0
            
            for step in range(self.max_steps_per_episode):
                # 选择动作
                action, action_prob = self.agent.choose_action(state)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 记录经验
                experience = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done,
                    'log_prob': np.log(action_prob + 1e-8)
                }
                episode_experiences.append(experience)
                
                # 更新状态
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1
                
                # 检查回合是否结束
                if done:
                    break
            
            # 记录回合信息
            self.training_log['episode_rewards'].append(episode_reward)
            self.training_log['episode_lengths'].append(episode_steps)
            
            print(f"📊 回合 {episode + 1} 完成 - 奖励: {episode_reward:.2f}, 步数: {episode_steps}")
            
            # 添加到经验缓冲区
            all_experiences.extend(episode_experiences)
        
        return all_experiences
    
    def train_step(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        执行一次训练步骤
        
        Args:
            experiences: 经验列表
            
        Returns:
            loss_info: 损失信息
        """
        if len(experiences) < self.batch_size:
            print("⚠️ 经验数量不足，跳过训练步骤")
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # 随机采样批次
        indices = np.random.choice(len(experiences), self.batch_size, replace=False)
        batch_experiences = [experiences[i] for i in indices]
        
        # 更新参数
        loss_info = self.agent.update_parameters(batch_experiences)
        
        # 记录损失
        self.training_log['losses'].append(loss_info)
        
        return loss_info
    
    def evaluate_agent(self, num_episodes: int = 3) -> Dict[str, float]:
        """
        评估智能体性能
        
        Args:
            num_episodes: 评估回合数
            
        Returns:
            eval_stats: 评估统计信息
        """
        print(f"🧪 开始评估智能体 - {num_episodes}回合")
        
        self.agent.eval()
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(self.max_steps_per_episode):
                with torch.no_grad():
                    action, _ = self.agent.choose_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_steps)
            
            print(f"📈 评估回合 {episode + 1} - 奖励: {episode_reward:.2f}, 步数: {episode_steps}")
        
        self.agent.train()
        
        eval_stats = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'max_reward': np.max(eval_rewards)
        }
        
        return eval_stats
    
    def save_checkpoint(self, episode: int, is_best: bool = False):
        """
        保存检查点
        
        Args:
            episode: 当前回合数
            is_best: 是否是最佳模型
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存模型
        if is_best:
            model_path = os.path.join(self.model_save_dir, f'ppo_best_{episode}.pth')
        else:
            model_path = os.path.join(self.model_save_dir, f'ppo_{episode}_{timestamp}.pth')
        
        self.agent.save_model(model_path)
        
        # 保存训练日志
        log_path = os.path.join(self.log_dir, f'training_log_{timestamp}.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)
        
        print(f"💾 检查点已保存 - 模型: {model_path}, 日志: {log_path}")
    
    def train(self, num_episodes: int = None):
        """
        执行完整的训练循环
        
        Args:
            num_episodes: 训练回合数（默认使用max_episodes）
        """
        if num_episodes is None:
            num_episodes = self.max_episodes
        
        print(f"🚀 开始PPO训练 - 目标回合数: {num_episodes}")
        
        start_time = time.time()
        
        for episode in range(self.current_episode, num_episodes):
            self.current_episode = episode
            
            print(f"\n{'='*50}")
            print(f"🎯 训练回合 {episode + 1}/{num_episodes}")
            print(f"{'='*50}")
            
            # 1. 收集经验
            experiences = self.collect_experience(num_episodes=1)
            self.experience_buffer.extend(experiences)
            
            # 2. 训练步骤
            if len(self.experience_buffer) >= self.batch_size:
                loss_info = self.train_step(self.experience_buffer)
                print(f"📚 训练步骤完成 - 策略损失: {loss_info['policy_loss']:.4f}, "
                      f"价值损失: {loss_info['value_loss']:.4f}, 熵: {loss_info['entropy']:.4f}")
            
            # 3. 定期评估和保存
            if (episode + 1) % self.update_frequency == 0:
                # 评估智能体
                eval_stats = self.evaluate_agent(num_episodes=2)
                
                # 检查是否是最佳模型
                if eval_stats['mean_reward'] > self.best_reward:
                    self.best_reward = eval_stats['mean_reward']
                    self.save_checkpoint(episode + 1, is_best=True)
                    print(f"🏆 新的最佳模型! 平均奖励: {self.best_reward:.2f}")
                else:
                    self.save_checkpoint(episode + 1)
                
                # 打印训练进度
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(self.training_log['episode_rewards'][-self.update_frequency:])
                
                print(f"📊 训练进度 - 平均奖励: {avg_reward:.2f}, "
                      f"总步数: {self.total_steps}, 已用时间: {elapsed_time:.1f}s")
        
        # 训练完成
        total_time = time.time() - start_time
        final_avg_reward = np.mean(self.training_log['episode_rewards'][-10:])  # 最后10回合平均奖励
        
        print(f"\n🎉 训练完成!")
        print(f"📊 最终统计:")
        print(f"   - 总回合数: {num_episodes}")
        print(f"   - 总步数: {self.total_steps}")
        print(f"   - 最终平均奖励: {final_avg_reward:.2f}")
        print(f"   - 最佳平均奖励: {self.best_reward:.2f}")
        print(f"   - 总训练时间: {total_time:.1f}s")
        
        # 保存最终模型
        self.save_checkpoint(num_episodes, is_best=False)
        
        # 关闭环境
        self.env.close()
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        self.agent.load_model(checkpoint_path)
        print(f"✅ 检查点已从 {checkpoint_path} 加载")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        获取训练统计信息
        
        Returns:
            stats: 训练统计信息
        """
        if not self.training_log['episode_rewards']:
            return {"status": "未开始训练"}
        
        recent_rewards = self.training_log['episode_rewards'][-20:]  # 最近20回合
        
        return {
            "current_episode": self.current_episode,
            "total_steps": self.total_steps,
            "best_reward": self.best_reward,
            "recent_avg_reward": np.mean(recent_rewards),
            "recent_std_reward": np.std(recent_rewards),
            "total_episodes": len(self.training_log['episode_rewards'])
        }


def main():
    """主函数 - 用于测试训练管理器"""
    print("🧪 测试PPO训练管理器")
    
    # 创建训练管理器
    trainer = PPOTrainer()
    
    # 测试训练循环（仅运行少量回合）
    try:
        trainer.train(num_episodes=5)
        print("✅ PPO训练管理器测试通过")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()