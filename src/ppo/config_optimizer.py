"""
配置优化器 - PPO超参数和奖励函数参数优化
基于网格搜索和贝叶斯优化的参数调优框架
"""

import os
import time
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path

from .ppo_agent import PPOBattleAgent
from .game_environment import GameEnvironment


class PPOConfigOptimizer:
    """PPO配置优化器 - 超参数和奖励函数参数优化"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化配置优化器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.optimization_results = {}
        self.best_config = None
        
        # 默认参数搜索空间
        self.param_space = {
            # PPO超参数
            'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
            'gamma': [0.95, 0.97, 0.99, 0.995],
            'clip_epsilon': [0.1, 0.2, 0.3],
            'entropy_coef': [0.01, 0.05, 0.1],
            
            # 奖励函数参数
            'victory_reward': [50, 100, 200],
            'defeat_penalty': [-20, -50, -100],
            'enemy_reward_multiplier': [1.0, 2.0, 5.0],
            'hp_reward_multiplier': [0.5, 1.0, 2.0],
            'step_penalty': [-0.05, -0.1, -0.2]
        }
        
        # 优化配置
        self.optimization_config = {
            'max_iterations': 20,  # 最大迭代次数
            'evaluation_episodes': 3,  # 每次评估的回合数
            'early_stopping_patience': 5,  # 早停耐心
            'target_reward': 50.0  # 目标奖励
        }
        
        print("✅ PPO配置优化器初始化完成")
    
    def evaluate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估特定配置的性能
        
        Args:
            config: 配置参数
            
        Returns:
            评估结果
        """
        print(f"🧪 评估配置: {config}")
        
        try:
            # 初始化PPO智能体
            ppo_config = {
                'state_dim': 50,
                'action_dim': 14,
                'learning_rate': config.get('learning_rate', 3e-4),
                'gamma': config.get('gamma', 0.99),
                'clip_epsilon': config.get('clip_epsilon', 0.2),
                'entropy_coef': config.get('entropy_coef', 0.01)
            }
            
            agent = PPOBattleAgent(**ppo_config)
            
            # 初始化游戏环境
            env = GameEnvironment(self.config_path)
            
            # 评估配置性能
            total_rewards = []
            episode_lengths = []
            
            for episode in range(self.optimization_config['evaluation_episodes']):
                state = env.reset()
                episode_reward = 0
                episode_steps = 0
                
                for step in range(100):  # 最大步数限制
                    action, _ = agent.choose_action(state)
                    next_state, reward, done, _ = env.step(action)
                    
                    # 应用奖励函数参数
                    adjusted_reward = self._adjust_reward(reward, config)
                    episode_reward += adjusted_reward
                    episode_steps += 1
                    
                    state = next_state
                    
                    if done:
                        break
                
                total_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)
            
            # 计算性能指标
            mean_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)
            mean_length = np.mean(episode_lengths)
            
            evaluation_result = {
                'config': config,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'mean_length': mean_length,
                'total_rewards': total_rewards,
                'episode_lengths': episode_lengths,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"📊 配置评估结果 - 平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
            return evaluation_result
            
        except Exception as e:
            error_msg = f"配置评估失败: {e}"
            print(f"❌ {error_msg}")
            return {
                'config': config,
                'mean_reward': -100.0,  # 失败时给予极低奖励
                'std_reward': 0.0,
                'mean_length': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _adjust_reward(self, original_reward: float, config: Dict[str, Any]) -> float:
        """
        根据配置调整奖励值
        
        Args:
            original_reward: 原始奖励
            config: 配置参数
            
        Returns:
            调整后的奖励
        """
        # 这里可以根据具体的奖励函数参数进行调整
        # 目前先简单实现
        adjusted_reward = original_reward
        
        # 应用奖励乘数
        if original_reward > 0:  # 正奖励
            adjusted_reward *= config.get('enemy_reward_multiplier', 1.0)
        elif original_reward < 0:  # 负奖励（惩罚）
            adjusted_reward *= config.get('hp_reward_multiplier', 1.0)
        
        return adjusted_reward
    
    def grid_search(self, param_subset: List[str] = None) -> Dict[str, Any]:
        """
        网格搜索优化
        
        Args:
            param_subset: 要优化的参数子集
            
        Returns:
            优化结果
        """
        print("🔍 开始网格搜索优化")
        
        if param_subset is None:
            param_subset = ['learning_rate', 'gamma', 'clip_epsilon']
        
        # 生成参数组合
        param_combinations = self._generate_param_combinations(param_subset)
        
        best_result = None
        all_results = []
        
        for i, config in enumerate(param_combinations):
            print(f"\n🔄 进度: {i+1}/{len(param_combinations)}")
            
            # 评估配置
            result = self.evaluate_config(config)
            all_results.append(result)
            
            # 更新最佳配置
            if best_result is None or result['mean_reward'] > best_result['mean_reward']:
                best_result = result
                print(f"🏆 新的最佳配置! 平均奖励: {best_result['mean_reward']:.2f}")
        
        # 保存优化结果
        self.optimization_results['grid_search'] = {
            'method': 'grid_search',
            'param_subset': param_subset,
            'best_result': best_result,
            'all_results': all_results,
            'total_evaluations': len(param_combinations)
        }
        
        self.best_config = best_result['config']
        
        return self.optimization_results['grid_search']
    
    def _generate_param_combinations(self, param_subset: List[str]) -> List[Dict[str, Any]]:
        """
        生成参数组合
        
        Args:
            param_subset: 参数子集
            
        Returns:
            参数组合列表
        """
        from itertools import product
        
        # 获取参数值列表
        param_values = []
        param_names = []
        
        for param_name in param_subset:
            if param_name in self.param_space:
                param_values.append(self.param_space[param_name])
                param_names.append(param_name)
        
        # 生成所有组合
        combinations = list(product(*param_values))
        
        # 转换为字典格式
        param_combinations = []
        for combo in combinations:
            config = {}
            for i, param_name in enumerate(param_names):
                config[param_name] = combo[i]
            param_combinations.append(config)
        
        return param_combinations
    
    def random_search(self, num_samples: int = 10) -> Dict[str, Any]:
        """
        随机搜索优化
        
        Args:
            num_samples: 采样数量
            
        Returns:
            优化结果
        """
        print("🎲 开始随机搜索优化")
        
        best_result = None
        all_results = []
        
        for i in range(num_samples):
            print(f"\n🔄 进度: {i+1}/{num_samples}")
            
            # 随机生成配置
            config = self._generate_random_config()
            
            # 评估配置
            result = self.evaluate_config(config)
            all_results.append(result)
            
            # 更新最佳配置
            if best_result is None or result['mean_reward'] > best_result['mean_reward']:
                best_result = result
                print(f"🏆 新的最佳配置! 平均奖励: {best_result['mean_reward']:.2f}")
        
        # 保存优化结果
        self.optimization_results['random_search'] = {
            'method': 'random_search',
            'num_samples': num_samples,
            'best_result': best_result,
            'all_results': all_results,
            'total_evaluations': num_samples
        }
        
        if best_result is not None:
            self.best_config = best_result['config']
        
        return self.optimization_results['random_search']
    
    def _generate_random_config(self) -> Dict[str, Any]:
        """
        随机生成配置
        
        Returns:
            随机配置
        """
        config = {}
        
        for param_name, values in self.param_space.items():
            config[param_name] = np.random.choice(values)
        
        return config
    
    def optimize(self, method: str = 'grid_search', **kwargs) -> Dict[str, Any]:
        """
        执行优化
        
        Args:
            method: 优化方法 ('grid_search', 'random_search')
            **kwargs: 方法特定参数
            
        Returns:
            优化结果
        """
        print(f"🚀 开始{method}优化")
        
        start_time = time.time()
        
        if method == 'grid_search':
            result = self.grid_search(**kwargs)
        elif method == 'random_search':
            result = self.random_search(**kwargs)
        else:
            raise ValueError(f"不支持的优化方法: {method}")
        
        # 添加优化统计信息
        result['optimization_time'] = f"{time.time() - start_time:.2f}s"
        result['timestamp'] = datetime.now().isoformat()
        
        print(f"\n🎉 {method}优化完成!")
        print(f"📊 最佳平均奖励: {result['best_result']['mean_reward']:.2f}")
        print(f"⏱️ 优化用时: {result['optimization_time']}")
        
        return result
    
    def save_optimization_results(self, filepath: str = None):
        """
        保存优化结果
        
        Args:
            filepath: 文件路径
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"optimization_results/ppo_optimization_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.optimization_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 优化结果已保存到: {filepath}")
    
    def generate_optimization_report(self) -> str:
        """
        生成优化报告
        
        Returns:
            优化报告文本
        """
        if not self.optimization_results:
            return "尚未进行优化"
        
        report = "# PPO配置优化报告\n\n"
        
        for method, result in self.optimization_results.items():
            report += f"## {method.replace('_', ' ').title()}\n\n"
            report += f"- 方法: {result['method']}\n"
            report += f"- 评估次数: {result['total_evaluations']}\n"
            report += f"- 优化用时: {result.get('optimization_time', 'N/A')}\n"
            
            if 'best_result' in result and result['best_result']:
                best = result['best_result']
                report += f"- 最佳平均奖励: {best['mean_reward']:.2f} ± {best['std_reward']:.2f}\n"
                report += f"- 最佳平均步数: {best['mean_length']:.1f}\n"
                report += f"- 最佳配置: {json.dumps(best['config'], indent=2)}\n"
            
            report += "\n"
        
        if self.best_config:
            report += "## 推荐配置\n\n"
            report += f"```json\n{json.dumps(self.best_config, indent=2)}\n```\n"
        
        return report


def main():
    """主函数 - 运行配置优化"""
    print("🧪 启动PPO配置优化器")
    
    # 创建优化器
    optimizer = PPOConfigOptimizer()
    
    # 运行网格搜索优化
    print("\n" + "="*50)
    result1 = optimizer.optimize('grid_search', param_subset=['learning_rate', 'gamma'])
    
    # 运行随机搜索优化
    print("\n" + "="*50)
    result2 = optimizer.optimize('random_search', num_samples=5)
    
    # 生成优化报告
    report = optimizer.generate_optimization_report()
    print("\n" + report)
    
    # 保存优化结果
    optimizer.save_optimization_results()
    
    return optimizer.optimization_results


if __name__ == "__main__":
    main()