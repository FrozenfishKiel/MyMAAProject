"""
PPO神经网络定义
包含策略网络和价值网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOPolicyNetwork(nn.Module):
    """PPO策略网络 - 输出动作概率分布"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化策略网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(PPOPolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 策略网络架构
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        print(f"✅ 策略网络初始化完成 - 输入: {state_dim}, 输出: {action_dim}")
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入状态
            
        Returns:
            action_probs: 动作概率分布
        """
        # 特征提取
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # 动作概率分布
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs
    
    def get_action_distribution(self, state: torch.Tensor):
        """获取动作分布"""
        action_probs = self.forward(state)
        return torch.distributions.Categorical(action_probs)


class PPOValueNetwork(nn.Module):
    """PPO价值网络 - 估计状态价值"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """
        初始化价值网络
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
        """
        super(PPOValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # 价值网络架构
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        print(f"✅ 价值网络初始化完成 - 输入: {state_dim}, 输出: 1")
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入状态
            
        Returns:
            value: 状态价值估计
        """
        # 特征提取
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # 价值估计
        value = self.value_head(x)
        
        return value


class PPONetwork(nn.Module):
    """PPO共享网络 - 策略和价值网络共享特征提取层"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化共享网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(PPONetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 共享特征提取层
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 策略头
        self.policy_fc = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
        # 价值头
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        print(f"✅ 共享网络初始化完成 - 输入: {state_dim}, 策略输出: {action_dim}, 价值输出: 1")
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            if module is self.policy_head:
                nn.init.orthogonal_(module.weight, gain=0.01)
            elif module is self.value_head:
                nn.init.orthogonal_(module.weight, gain=1.0)
            else:
                nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入状态
            
        Returns:
            action_probs: 动作概率分布
            value: 状态价值估计
        """
        # 共享特征提取
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        
        # 策略分支
        policy_features = F.relu(self.policy_fc(x))
        action_logits = self.policy_head(policy_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 价值分支
        value_features = F.relu(self.value_fc(x))
        value = self.value_head(value_features)
        
        return action_probs, value
    
    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        """仅获取策略输出"""
        action_probs, _ = self.forward(x)
        return action_probs
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """仅获取价值输出"""
        _, value = self.forward(x)
        return value


if __name__ == "__main__":
    # 测试策略网络
    policy_net = PPOPolicyNetwork(state_dim=10, action_dim=14)
    test_input = torch.randn(1, 10)
    output = policy_net(test_input)
    print(f"策略网络测试 - 输入形状: {test_input.shape}, 输出形状: {output.shape}")
    
    # 测试价值网络
    value_net = PPOValueNetwork(state_dim=10)
    value_output = value_net(test_input)
    print(f"价值网络测试 - 输入形状: {test_input.shape}, 输出形状: {value_output.shape}")
    
    # 测试共享网络
    shared_net = PPONetwork(state_dim=10, action_dim=14)
    policy_out, value_out = shared_net(test_input)
    print(f"共享网络测试 - 策略输出形状: {policy_out.shape}, 价值输出形状: {value_out.shape}")
    
    print("✅ 神经网络测试通过")