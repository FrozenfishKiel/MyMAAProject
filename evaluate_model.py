import os
import sys
import time
from pathlib import Path

# 将模块路径加入 sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "ai-plugins"))
sys.path.insert(0, str(ROOT / "src" / "rl-environment"))

# 修复 maa 模块找不到的问题 (使用项目内的实际依赖路径)
DEPS_ROOT = ROOT / "maa-deps" / "maafw-5.2.6-win_amd64"
if DEPS_ROOT.exists():
    sys.path.insert(0, str(DEPS_ROOT))
else:
    print(f"[WARN] 找不到 MAA 依赖目录: {DEPS_ROOT}")

from stable_baselines3 import PPO
from yolo_recognizer import YoloRecognizer
from template_matcher import TemplateMatcher
from game_env import GameEnv

def main():
    model_path = r"D:\BiShe\MaAutomaton-main\MaAutomaton-main\models\rl\policy_latest.zip"
    yolo_model_path = r"D:\BiShe\MaAutomaton-main\MaAutomaton-main\models\yolo\best.pt"
    
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}，请先运行 train.py 训练一个晚上！")
        return

    try:
        from maa.controller import AdbController
        from maa.toolkit import Toolkit
        Toolkit.init_option(str(ROOT))
        adb_devices = Toolkit.find_adb_devices()
        if not adb_devices:
            print("未找到 ADB 设备，请确保模拟器已启动。")
            return
        device = adb_devices[0]
        controller = AdbController(
            adb_path=device.adb_path,
            address=device.address,
            screencap_methods=device.screencap_methods,
            input_methods=device.input_methods,
            config=device.config,
        )
        controller.post_connection().wait()
        print("[SUCCESS] 设备连接成功！")
    except Exception as e:
        print(f"[ERROR] 模拟器连接失败: {e}")
        return

    print("加载 YOLO 模型...")
    import torch
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YoloRecognizer(model_path=yolo_model_path, device=device_str)
    yolo.load()

    print("加载 TemplateMatcher...")
    matcher = TemplateMatcher(controller)
    matcher.load()

    print("初始化游戏环境...")
    env = GameEnv(controller, yolo, matcher)

    print(f"加载 PPO 模型 ({model_path})...")
    model = PPO.load(model_path, env=env)

    print("=============================================")
    print(" 开始评估 (Evaluation Mode) ")
    print(" AI 将关闭探索随机性，完全使用已学到的策略！")
    print(" 按 Ctrl+C 可以终止评估")
    print("=============================================")

    # 进入关卡，开始评估
    obs, _ = env.reset()
    try:
        consecutive_blocks = 0
        while True:
            # deterministic=True 是评估模式的核心，AI 不会乱试错，只会选它认为得分最高的动作
            action, _states = model.predict(obs, deterministic=True)

            # 如果连续被 CV 阻断 2 次，强行剥夺控制权，修改它的动作为“挂机等费”，打破死锁
            if consecutive_blocks >= 2:
                print(f"\n[ACTION MASKING] AI连续 {consecutive_blocks} 次被阻断陷入死锁！强行覆盖动作为挂机...")
                action[0] = 10
                consecutive_blocks = 0

            obs, reward, terminated, truncated, info = env.step(action)

            # 记录连续非法拦截（在 game_env 中我们刚改成了 -15.0 或者之前的 -0.5）
            if reward <= -0.5:
                consecutive_blocks += 1
            else:
                consecutive_blocks = 0

            if terminated or truncated:
                print("回合结束！")
                obs, _ = env.reset()
                consecutive_blocks = 0
                
    except KeyboardInterrupt:
        print("评估被用户终止。")
    finally:
        env.close()

if __name__ == "__main__":
    main()
