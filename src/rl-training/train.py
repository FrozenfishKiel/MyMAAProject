from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import datetime

# --- 配置双重日志输出 ---
class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 强制写入硬盘，防止崩溃丢失

    def flush(self):
        self.terminal.flush()

# 获取当前时间生成带时间戳的日志文件
log_dir = Path(r"D:\BiShe\MaAutomaton-main\MaAutomaton-main\logs")
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"training_{timestamp}.log"

# 重定向标准输出
sys.stdout = Logger(str(log_file))
# -------------------------

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
    """检查是否按下了P键暂停训练"""
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
    def __init__(self, verbose: int = 0, check_pause_func=None) -> None:
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.check_pause_func = check_pause_func
        self.paused = False
        self.pause_count = 0
        self.total_steps = 0

    def _on_step(self) -> bool:
        self.total_steps += 1

        if self.check_pause_func and self.check_pause_func():
            if not self.paused:
                self.paused = True
                self.pause_count += 1
                print(f"\n[PAUSE] 暂停训练 #{self.pause_count}，按P键继续...")
            else:
                self.paused = False
                print(f"\n[RESUME] 继续训练 #{self.pause_count}")

        if self.paused:
            import time
            time.sleep(1)
            return True

        if "rewards" in self.locals:
            self.current_episode_reward += self.locals["rewards"][0]
            self.current_episode_length += 1

        if "dones" in self.locals and self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            print(f"--> [回合结束] 第 {len(self.episode_rewards)} 回合 | 奖励: {self.current_episode_reward:.1f} | 步数: {self.current_episode_length}")

            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        # === 新增：保存中间模型 ===
        # 每隔 20 步自动存一次档（为了让你尽早看到文件生成，缩短了保存间隔）
        if self.total_steps > 0 and self.total_steps % 20 == 0:
            save_path = Path(r"D:\BiShe\MaAutomaton-main\MaAutomaton-main\models\rl\policy_latest.zip")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(save_path))
            print(f"[SAVE] 自动保存进度 (Step {self.total_steps}) -> {save_path.name}")
        # ==========================

        return True

def train_rl_model(
    controller: Any,
    yolo_model_path: str,
    rl_model_save_path: str,
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 128,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    verbose: int = 1
) -> None:

    print(f"[INFO] 正在加载 YOLO 模型: {yolo_model_path}")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"       使用设备: {device}")

    yolo_recognizer = YoloRecognizer(model_path=yolo_model_path, device=device)
    yolo_recognizer.load()
    print("[SUCCESS] YOLO 模型加载完成!")

    print("[INFO] 正在加载模板匹配器...")
    template_matcher = TemplateMatcher(controller)
    template_matcher.load()
    print("[SUCCESS] 模板匹配器加载完成!")

    print("[INFO] 正在创建强化学习环境...")
    env = DummyVecEnv([lambda: GameEnv(controller, yolo_recognizer, template_matcher)])
    print("[SUCCESS] 环境创建完成!")

    callback = TrainingCallback(verbose=verbose, check_pause_func=check_pause_training)

    print("[INFO] 正在创建 PPO 模型...")
    # ================= 修改处：加载已有的记忆 =================
    import os
    latest_model_path = r"D:\BiShe\MaAutomaton-main\MaAutomaton-main\models\rl\policy_latest.zip"

    # 使用绝对路径来保存 tensorboard 日志，防止相对路径在不同执行目录下找不到文件夹
    tensorboard_path = str(ROOT.parent / "tensorboard_logs")
    os.makedirs(tensorboard_path, exist_ok=True)

    if os.path.exists(latest_model_path):
        print(f"[SUCCESS] 发现已有模型存档，从 {latest_model_path} 恢复记忆继续训练！")
        model = PPO.load(
            latest_model_path,
            env=env,
            verbose=verbose,
            tensorboard_log=tensorboard_path,
            # ==== 关键改动：恢复探索欲 ====
            # 这里必须写上 ent_coef 否则 load 后探索欲可能会归零
            ent_coef=0.05
        )
        # 手动重置一下学习率等参数，防止因为加载而覆盖了这里设置的超参
        model.learning_rate = learning_rate
        model.n_steps = n_steps
        model.batch_size = batch_size
    else:
        print("[INFO] 未发现已有模型存档，创建全新的 PPO 婴儿大脑！")
        # 使用 CnnPolicy 和 MultiDiscrete 动作空间，SB3 会自动适配
        model = PPO(
            "CnnPolicy",
            env,
            verbose=verbose,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            # ==== 关键改动：提高初始探索欲 ====
            # 原来是 0.01 太低了，导致它一旦找到一个能得分的点就死磕
            ent_coef=0.05,
            tensorboard_log=tensorboard_path, # 开启 Tensorboard，方便看训练曲线
        )
    print("[SUCCESS] PPO 模型创建完成!")

    print(f"\n================ 开始训练 ===================")
    print(f"总步数: {total_timesteps}, 每次更新前收集步数: {n_steps}, 批次: {batch_size}")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except Exception as e:
        print(f"[ERROR] 训练中断: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n================ 保存模型 ===================")
    os.makedirs(os.path.dirname(rl_model_save_path), exist_ok=True)
    model.save(rl_model_save_path)
    print(f"[SUCCESS] 模型已保存至: {rl_model_save_path}")

def main() -> None:
    yolo_model_path = r"D:\BiShe\MaAutomaton-main\MaAutomaton-main\models\yolo\best.pt"
    rl_model_save_path = r"D:\BiShe\MaAutomaton-main\MaAutomaton-main\models\rl\policy.zip"

    try:
        from maa.controller import AdbController
        from maa.toolkit import Toolkit

        Toolkit.init_option(str(ROOT))
        print("[INFO] 正在查找 ADB 设备...")
        adb_devices = Toolkit.find_adb_devices()

        if not adb_devices:
            print("[ERROR] 未找到 ADB 设备。模拟器启动了吗？")
            controller = None
        else:
            device = adb_devices[0]
            print(f"[SUCCESS] 连接到设备: {device.name} ({device.address})")
            controller = AdbController(
                adb_path=device.adb_path,
                address=device.address,
                screencap_methods=device.screencap_methods,
                input_methods=device.input_methods,
                config=device.config,
            )
            connection_job = controller.post_connection()
            connection_job.wait()

            if connection_job.succeeded:
                print("[SUCCESS] 设备握手成功!")
            else:
                print("[ERROR] 设备连接失败")
                controller = None
    except Exception as e:
        print(f"[ERROR] MAA 初始化失败: {e}")
        controller = None

    if controller is None:
        print("[WARNING] 控制器为空，程序可能在干跑报错")

    train_rl_model(
        controller=controller,
        yolo_model_path=yolo_model_path,
        rl_model_save_path=rl_model_save_path,
        total_timesteps=20000, # 增加到两万步，让模型有充足的时间收敛
        n_steps=128,          # 每走128步（大约4个回合），模型更新一次大脑
        batch_size=64
    )

if __name__ == "__main__":
    main()
