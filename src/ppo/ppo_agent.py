"""
PPO智能体主类
实现PPO算法的核心逻辑：动作选择、价值估计、参数更新
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Any

from .networks import PPOPolicyNetwork, PPOValueNetwork


class PPOBattleAgent:
    """PPO智能体 - 用于明日方舟游戏决策"""
    
    def __init__(self, 
                 state_dim: int = 50,  # 修改为状态编码器实际输出的50维
                 action_dim: int = 14,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01):
        """
        初始化PPO智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            gamma: 折扣因子
            clip_epsilon: PPO裁剪参数
            entropy_coef: 熵系数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        
        # 初始化策略网络和价值网络
        self.policy_net = PPOPolicyNetwork(state_dim, action_dim)
        self.value_net = PPOValueNetwork(state_dim)
        
        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.policy_net.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=learning_rate)
        
        # 训练状态
        self.training = True
        
        print(f"✅ PPO智能体初始化完成 - 状态维度: {state_dim}, 动作维度: {action_dim}")
    
    def choose_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态向量
            
        Returns:
            action: 选择的动作编号
            action_prob: 动作概率
        """
        # 转换为Tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 策略网络前向传播
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            
            # 采样动作
            action = action_dist.sample()
            action_prob = action_probs[0, action.item()].item()
            
        return action.item(), action_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作的概率和熵
        
        Args:
            states: 状态批次
            actions: 动作批次
            
        Returns:
            log_probs: 动作对数概率
            entropy: 策略熵
            values: 状态价值
        """
        action_probs = self.policy_net(states)
        action_dist = torch.distributions.Categorical(action_probs)
        
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        values = self.value_net(states)
        
        return log_probs, entropy, values
    
    def compute_advantages(self, rewards: List[float], values: List[float], 
                          dones: List[bool]) -> List[float]:
        """
        计算优势函数
        
        Args:
            rewards: 奖励序列
            values: 价值估计序列
            dones: 终止状态序列
            
        Returns:
            advantages: 优势函数序列
        """
        advantages = []
        advantage = 0.0
        
        # 反向计算优势
        for t in reversed(range(len(rewards))):
            if dones[t]:
                advantage = 0.0
            
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * advantage * (1 - dones[t])
            advantages.insert(0, advantage)
        
        return advantages
    
    def update_parameters(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        PPO参数更新
        
        Args:
            experiences: 经验列表
            
        Returns:
            loss_info: 损失信息
        """
        if not experiences:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # 提取经验数据
        states = torch.FloatTensor([exp['state'] for exp in experiences])
        actions = torch.LongTensor([exp['action'] for exp in experiences])
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences])
        next_states = torch.FloatTensor([exp['next_state'] for exp in experiences])
        dones = torch.BoolTensor([exp['done'] for exp in experiences])
        old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in experiences])
        
        # 计算优势函数
        with torch.no_grad():
            current_values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            
            # 计算TD目标
            targets = rewards + self.gamma * next_values * (~dones).float()
            advantages = targets - current_values
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新循环
        policy_losses = []
        value_losses = []
        entropies = []
        
        for _ in range(4):  # PPO更新次数
            # 评估当前策略
            log_probs, entropy, values = self.evaluate_actions(states, actions)
            
            # 概率比率
            ratios = torch.exp(log_probs - old_log_probs)
            
            # PPO裁剪目标
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = nn.MSELoss()(values.squeeze(), targets)
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies)
        }
    
    def save_model(self, filepath: str):
        """保存模型参数"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        torch.save(checkpoint, filepath)
        print(f"✅ 模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型参数"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ 模型已从 {filepath} 加载")
    
    def train(self):
        """设置为训练模式"""
        self.training = True
        self.policy_net.train()
        self.value_net.train()
    
    def eval(self):
        """设置为评估模式"""
        self.training = False
        self.policy_net.eval()
        self.value_net.eval()


if __name__ == "__main__":
    # 测试PPO智能体
    agent = PPOBattleAgent(state_dim=5, action_dim=14)
    
    # 测试动作选择
    test_state = np.random.randn(5)
    action, prob = agent.choose_action(test_state)
    print(f"测试动作选择 - 动作: {action}, 概率: {prob:.4f}")
    
    print("✅ PPO智能体测试通过")