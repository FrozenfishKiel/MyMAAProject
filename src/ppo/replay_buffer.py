"""
经验回放缓冲区
存储和管理PPO训练经验
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque


class ReplayBuffer:
    """经验回放缓冲区 - 存储PPO训练经验"""
    
    def __init__(self, capacity: int = 10000):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
        print(f"✅ 经验回放缓冲区初始化完成 - 容量: {capacity}")
    
    def add(self, experience: Dict[str, Any]) -> None:
        """
        添加经验到缓冲区
        
        Args:
            experience: 经验字典
        """
        self.buffer.append(experience)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        从缓冲区随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch: 经验批次
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def sample_sequential(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        顺序采样一批经验（保持时间顺序）
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch: 经验批次
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        start_idx = random.randint(0, len(self.buffer) - batch_size)
        batch = list(self.buffer)[start_idx:start_idx + batch_size]
        return batch
    
    def sample_recent(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        采样最近的经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch: 最近的经验批次
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        batch = list(self.buffer)[-batch_size:]
        return batch
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()
        self.position = 0
        print("✅ 经验回放缓冲区已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        if not self.buffer:
            return {
                'size': 0,
                'capacity': self.capacity,
                'fullness': 0.0
            }
        
        # 计算奖励统计
        rewards = [exp['reward'] for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'fullness': len(self.buffer) / self.capacity,
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'reward_std': np.std(rewards)
        }
    
    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """通过索引获取经验"""
        return self.buffer[idx]


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区 - 基于TD误差的优先级采样"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        """
        初始化优先经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数
            beta: 重要性采样权重指数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        # 存储经验和优先级
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        print(f"✅ 优先经验回放缓冲区初始化完成 - 容量: {capacity}")
    
    def add(self, experience: Dict[str, Any], priority: Optional[float] = None) -> None:
        """
        添加经验到缓冲区
        
        Args:
            experience: 经验字典
            priority: 优先级，如果为None则使用最大优先级
        """
        if priority is None:
            # 使用最大优先级
            if self.size > 0:
                priority = np.max(self.priorities[:self.size])
            else:
                priority = 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> tuple:
        """
        基于优先级采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch: 经验批次
            indices: 采样索引
            weights: 重要性采样权重
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probabilities = priorities / np.sum(priorities)
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # 计算重要性采样权重
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # 归一化
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[idx] for idx in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        更新经验的优先级
        
        Args:
            indices: 经验索引
            priorities: 新的优先级
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-5) ** self.alpha  # 避免零优先级
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        print("✅ 优先经验回放缓冲区已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'fullness': 0.0
            }
        
        priorities = self.priorities[:self.size]
        rewards = [exp['reward'] for exp in self.buffer[:self.size]]
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'fullness': self.size / self.capacity,
            'avg_priority': np.mean(priorities),
            'max_priority': np.max(priorities),
            'min_priority': np.min(priorities),
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }
    
    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        return self.size


class EpisodeBuffer:
    """回合缓冲区 - 存储完整的回合经验"""
    
    def __init__(self, capacity: int = 100):
        """
        初始化回合缓冲区
        
        Args:
            capacity: 缓冲区容量（回合数）
        """
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        
        print(f"✅ 回合缓冲区初始化完成 - 容量: {capacity} 回合")
    
    def add_episode(self, episode: List[Dict[str, Any]]) -> None:
        """
        添加一个完整回合的经验
        
        Args:
            episode: 回合经验列表
        """
        self.episodes.append(episode)
    
    def sample_episodes(self, num_episodes: int) -> List[List[Dict[str, Any]]]:
        """
        采样完整回合
        
        Args:
            num_episodes: 回合数
            
        Returns:
            episodes: 回合列表
        """
        if len(self.episodes) < num_episodes:
            return list(self.episodes)
        
        episodes = random.sample(self.episodes, num_episodes)
        return episodes
    
    def sample_transitions(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        从所有回合中采样过渡
        
        Args:
            batch_size: 批次大小
            
        Returns:
            transitions: 过渡列表
        """
        all_transitions = []
        for episode in self.episodes:
            all_transitions.extend(episode)
        
        if len(all_transitions) < batch_size:
            return all_transitions
        
        transitions = random.sample(all_transitions, batch_size)
        return transitions
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.episodes.clear()
        print("✅ 回合缓冲区已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        if not self.episodes:
            return {
                'num_episodes': 0,
                'capacity': self.capacity,
                'total_transitions': 0
            }
        
        episode_lengths = [len(episode) for episode in self.episodes]
        all_rewards = [exp['reward'] for episode in self.episodes for exp in episode]
        
        return {
            'num_episodes': len(self.episodes),
            'capacity': self.capacity,
            'total_transitions': sum(episode_lengths),
            'avg_episode_length': np.mean(episode_lengths),
            'max_episode_length': np.max(episode_lengths),
            'min_episode_length': np.min(episode_lengths),
            'avg_reward': np.mean(all_rewards) if all_rewards else 0.0,
            'max_reward': np.max(all_rewards) if all_rewards else 0.0,
            'min_reward': np.min(all_rewards) if all_rewards else 0.0
        }
    
    def __len__(self) -> int:
        """返回缓冲区中的回合数"""
        return len(self.episodes)


if __name__ == "__main__":
    # 测试经验回放缓冲区
    buffer = ReplayBuffer(capacity=100)
    
    # 添加测试经验
    for i in range(50):
        experience = {
            'state': np.random.randn(5),
            'action': np.random.randint(14),
            'reward': np.random.randn(),
            'next_state': np.random.randn(5),
            'done': False,
            'log_prob': 0.5
        }
        buffer.add(experience)
    
    print(f"缓冲区测试 - 大小: {len(buffer)}")
    
    # 测试采样
    batch = buffer.sample(10)
    print(f"采样测试 - 批次大小: {len(batch)}")
    
    # 测试统计信息
    stats = buffer.get_stats()
    print(f"统计信息测试 - {stats}")
    
    # 测试优先缓冲区
    prioritized_buffer = PrioritizedReplayBuffer(capacity=50)
    
    for i in range(30):
        experience = {
            'state': np.random.randn(5),
            'action': np.random.randint(14),
            'reward': np.random.randn(),
            'next_state': np.random.randn(5),
            'done': False,
            'log_prob': 0.5
        }
        prioritized_buffer.add(experience, priority=abs(experience['reward']))
    
    batch, indices, weights = prioritized_buffer.sample(10)
    print(f"优先缓冲区测试 - 批次大小: {len(batch)}, 权重形状: {weights.shape}")
    
    # 测试回合缓冲区
    episode_buffer = EpisodeBuffer(capacity=10)
    
    for i in range(3):
        episode = []
        for j in range(5):  # 每个回合5步
            experience = {
                'state': np.random.randn(5),
                'action': np.random.randint(14),
                'reward': np.random.randn(),
                'next_state': np.random.randn(5),
                'done': j == 4,  # 最后一步终止
                'log_prob': 0.5
            }
            episode.append(experience)
        episode_buffer.add_episode(episode)
    
    episodes = episode_buffer.sample_episodes(2)
    print(f"回合缓冲区测试 - 采样回合数: {len(episodes)}")
    
    print("✅ 经验回放缓冲区测试通过")