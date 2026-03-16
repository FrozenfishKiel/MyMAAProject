"""
测试动作空间

测试内容:
1. 动作1：点击干员头像
2. 动作2：拖拽到放置区域
3. 动作3：调整方向
4. 动作4：松手完成部署

每个动作都会添加详细的调试信息
"""

import sys
from pathlib import Path
import cv2
import random

# 添加项目路径到sys.path
ROOT = Path(__file__).resolve().parent.parent
DEPS_ROOT = ROOT / "maa-deps" / "maafw-5.2.6-win_amd64"

if DEPS_ROOT.exists():
    sys.path.insert(0, str(DEPS_ROOT))
sys.path.insert(0, str(ROOT / "src" / "ai-plugins"))
sys.path.insert(0, str(ROOT / "src" / "rl-environment"))

from yolo_recognizer import YoloRecognizer
from template_matcher import TemplateMatcher
import time

def test_action1_click_operator_avatar(controller, template_matcher, save_debug_images=True):
    """
    测试动作1：点击干员头像
    
    Args:
        controller: MaaFramework控制器
        template_matcher: 模板匹配识别器
        save_debug_images: 是否保存调试图像
    """
    print("=" * 70)
    print("🎯 测试动作1：点击干员头像")
    print("=" * 70)
    print()
    
    # 部署区范围：1272 x 116
    # 左上角坐标：(2, 598)
    # 右下角坐标：(1274, 714)
    deployment_area = {
        "x1": 2,
        "y1": 598,
        "x2": 1274,
        "y2": 714
    }
    
    print(f"📊 部署区范围:")
    print(f"   - 左上角: ({deployment_area['x1']}, {deployment_area['y1']})")
    print(f"   - 右下角: ({deployment_area['x2']}, {deployment_area['y2']})")
    print()
    
    # 步骤1：在部署区范围内随机点击
    print("【步骤1】在部署区范围内随机点击")
    x = random.randint(deployment_area["x1"], deployment_area["x2"])
    y = random.randint(deployment_area["y1"], deployment_area["y2"])
    print(f"   - 点击位置: ({x}, {y})")
    
    # 点击干员头像
    print("   - 正在点击...")
    click_job = controller.post_click(x, y)
    click_job.wait()
    print("   ✅ 点击完成")
    
    # 等待游戏响应，显示干员信息血条

    time.sleep(0.5)

    # 步骤2：截图
    print("【步骤2】截图")
    print("   - 正在截图...")
    image = controller.post_screencap().wait().get()
    print(f"   ✅ 截图完成，图像形状: {image.shape}")
    
    if save_debug_images:
        output_path = str(ROOT / "tests" / "output" / "action1_click_screenshot.png")
        cv2.imwrite(output_path, image)
        print(f"   - 截图已保存到: {output_path}")
    print()

    time.sleep(2)

    # 步骤3：使用模板匹配识别干员信息血条
    print("【步骤3】使用模板匹配识别干员信息血条")
    hp_bar_template_path = str(ROOT / "data" / "templates" / "hp_bar.png")
    print(f"   - 模板路径: {hp_bar_template_path}")
    
    # 检查模板文件是否存在
    import os
    if not os.path.exists(hp_bar_template_path):
        print(f"   ❌ 模板文件不存在!")
        return False, None
    
    print(f"   - 模板文件存在")
    print(f"   - 正在进行模板匹配...")
    
    try:
        hp_bar_result = template_matcher.match(image, hp_bar_template_path, threshold=0.4)
        
        if hp_bar_result is not None:
            print(f"   ✅ 模板匹配成功!")
            print(f"      - 标签: {hp_bar_result.label}")
            print(f"      - 置信度: {hp_bar_result.confidence:.2f}")
            print(f"      - 位置: {hp_bar_result.box_xyxy}")
            
            if save_debug_images:
                # 在图像上绘制匹配框
                debug_image = image.copy()
                x1, y1, x2, y2 = hp_bar_result.box_xyxy
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_image, f"Confidence: {hp_bar_result.confidence:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                output_path = str(ROOT / "tests" / "output" / "action1_template_match_debug.png")
                cv2.imwrite(output_path, debug_image)
                print(f"      - 调试图像已保存到: {output_path}")
            
            print()
            print("✅ 动作1成功：成功点击干员头像")
            return True, (x, y)
        else:
            print(f"   ❌ 模板匹配失败（未找到匹配）")
            print()
            print("❌ 动作1失败：未找到干员信息血条")
            return False, None
    except Exception as e:
        print(f"   ❌ 模板匹配出错: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("❌ 动作1失败：模板匹配出错")
        return False, None


def test_action2_drag_to_deployment_area(controller, template_matcher, start_position, save_debug_images=True):
    """
    测试动作2：拖拽到放置区域
    
    Args:
        controller: MaaFramework控制器
        template_matcher: 模板匹配识别器
        start_position: 干员位置 (x, y)
        save_debug_images: 是否保存调试图像
    """
    import time
    
    def cancel_deployment_and_wait():
        """
        取消部署并等待费用恢复
        """
        print()
        print("【步骤6】点击随机位置取消部署")
        
        # 点击随机位置取消部署
        click_x = random.randint(100, 1180)
        click_y = random.randint(100, 620)
        print(f"   - 点击位置: ({click_x}, {click_y})")
        print(f"   - 正在点击...")
        click_job = controller.post_click(click_x, click_y)
        click_job.wait()
        print(f"   ✅ 点击完成")
        
        # 等待5秒，让费用恢复
        print()
        print("【步骤7】等待5秒，让费用恢复")
        time.sleep(5.0)
        print(f"   ✅ 等待完成")
    
    print("=" * 70)
    print("🎯 测试动作2：拖拽到放置区域")
    print("=" * 70)
    print()
    
    if start_position is None:
        print("❌ 错误：start_position 为 None")
        cancel_deployment_and_wait()
        return False, None
    
    print(f"📊 干员位置: {start_position}")
    print()
    
    # 步骤1：截图
    print("【步骤1】截图")
    print("   - 正在截图...")
    image = controller.post_screencap().wait().get()
    print(f"   ✅ 截图完成，图像形状: {image.shape}")
    
    if save_debug_images:
        output_path = str(ROOT / "tests" / "output" / "action2_drag_screenshot.png")
        cv2.imwrite(output_path, image)
        print(f"   - 截图已保存到: {output_path}")
    print()
    
    # 步骤2：找到绿色高亮区域的中心点
    print("【步骤2】找到绿色高亮区域的中心点")
    from green_highlight import find_green_highlight
    green_center = find_green_highlight(image)
    
    if green_center:
        print(f"   ✅ 找到绿色高亮区域中心点: {green_center}")
    else:
        print(f"   ❌ 未找到绿色高亮区域")
        cancel_deployment_and_wait()
        print()
        print("❌ 动作2失败：未找到绿色高亮区域")
        return False, None
    print()
    
    # 可视化：绘制绿色高亮区域中心点
    if save_debug_images:
        debug_image = image.copy()
        green_x, green_y = green_center
        # 绘制中心点（绿色圆圈）
        cv2.circle(debug_image, (green_x, green_y), 10, (0, 255, 0), 2)
        cv2.circle(debug_image, (green_x, green_y), 3, (0, 255, 0), -1)
        # 添加文字标注
        cv2.putText(debug_image, f"Green Center: ({green_x}, {green_y})", 
                   (green_x + 15, green_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        output_path = str(ROOT / "tests" / "output" / "action2_green_center_debug.png")
        cv2.imwrite(output_path, debug_image)
        print(f"   - 绿色高亮区域中心点可视化已保存到: {output_path}")
    print()
    
    # 步骤3：拖拽到绿色高亮区域中心点
    print("【步骤3】拖拽到绿色高亮区域中心点")
    start_x, start_y = start_position
    end_x, end_y = green_center
    print(f"   - 起点: ({start_x}, {start_y})")
    print(f"   - 终点: ({end_x}, {end_y})")
    
    # 可视化：绘制拖拽路径
    if save_debug_images:
        debug_image = image.copy()
        # 绘制起点（红色圆圈）
        cv2.circle(debug_image, (start_x, start_y), 10, (0, 0, 255), 2)
        cv2.circle(debug_image, (start_x, start_y), 3, (0, 0, 255), -1)
        cv2.putText(debug_image, f"Start: ({start_x}, {start_y})", 
                   (start_x + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 绘制终点（绿色圆圈）
        cv2.circle(debug_image, (end_x, end_y), 10, (0, 255, 0), 2)
        cv2.circle(debug_image, (end_x, end_y), 3, (0, 255, 0), -1)
        cv2.putText(debug_image, f"End: ({end_x}, {end_y})", 
                   (end_x + 15, end_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        output_path = str(ROOT / "tests" / "output" / "action2_drag_path_debug.png")
        cv2.imwrite(output_path, debug_image)
        print(f"   - 拖拽路径可视化已保存到: {output_path}")
    
    print(f"   - 正在拖拽...")
    swipe_job = controller.post_swipe(start_x, start_y, end_x, end_y, 500)
    swipe_job.wait()
    print(f"   ✅ 拖拽完成")

    time.sleep(0.5)
    # 截图（包含cancel_ui）
    print("【步骤4】截图（包含cancel_ui）")
    print("   - 正在截图...")
    image = controller.post_screencap().wait().get()
    print(f"   ✅ 截图完成，图像形状: {image.shape}")
    
    if save_debug_images:
        output_path = str(ROOT / "tests" / "output" / "action2_after_drag_screenshot.png")
        cv2.imwrite(output_path, image)
        print(f"   - 截图已保存到: {output_path}")
    print()
    
    # 步骤5：使用模板匹配识别点击取消UI
    print("【步骤5】使用模板匹配识别点击取消UI")
    cancel_ui_template_path = str(ROOT / "data" / "templates" / "cancel_ui.png")
    print(f"   - 模板路径: {cancel_ui_template_path}")
    
    # 检查模板文件是否存在
    import os
    if not os.path.exists(cancel_ui_template_path):
        print(f"   ❌ 模板文件不存在!")
        cancel_deployment_and_wait()
        return False, None
    
    print(f"   - 模板文件存在")
    print(f"   - 正在进行模板匹配...")
    
    try:
        cancel_ui_result = template_matcher.match(image, cancel_ui_template_path, threshold=0.4)


        time.sleep(1.0)
        # 添加调试信息：显示所有可能的匹配结果（即使低于阈值）
        template = cv2.imread(cancel_ui_template_path, cv2.IMREAD_COLOR)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        print(f"   - 最佳匹配置信度: {max_val:.4f}")
        print(f"   - 当前匹配阈值: 0.4")
        print(f"   - 是否达到阈值: {'✅ 是' if max_val >= 0.4 else '❌ 否'}")

        
        if cancel_ui_result is not None:
            print(f"   ✅ 模板匹配成功!")
            print(f"      - 标签: {cancel_ui_result.label}")
            print(f"      - 置信度: {cancel_ui_result.confidence:.2f}")
            print(f"      - 位置: {cancel_ui_result.box_xyxy}")
            
            if save_debug_images:
                # 在图像上绘制匹配框
                debug_image = image.copy()
                x1, y1, x2, y2 = cancel_ui_result.box_xyxy
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(debug_image, f"Confidence: {cancel_ui_result.confidence:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                output_path = str(ROOT / "tests" / "output" / "action2_template_match_debug.png")
                cv2.imwrite(output_path, debug_image)
                print(f"      - 调试图像已保存到: {output_path}")
            
            print()
            print("✅ 动作2成功：成功拖拽到放置区域")
            
            # 等待2秒，让用户看到结果

            
            return True, green_center
        else:
            print(f"   ❌ 模板匹配失败（未找到匹配）")
            print()
            print("⚠️  可能是费用不足或取消UI未显示")
            cancel_deployment_and_wait()
            
            print()
            print("❌ 动作2失败：未找到点击取消UI，已取消部署并等待费用恢复")
            return False, None
    except Exception as e:
        print(f"   ❌ 模板匹配出错: {e}")
        import traceback
        traceback.print_exc()
        cancel_deployment_and_wait()
        print()
        print("❌ 动作2失败：模板匹配出错")
        
        return False, None


def test_action3_adjust_direction(controller, center_position):
    """
    测试动作3：调整方向
    
    Args:
        controller: MaaFramework控制器
        center_position: 放置区域中心点 (x, y)
    """
    print("=" * 70)
    print("🎯 测试动作3：调整方向")
    print("=" * 70)
    print()
    
    if center_position is None:
        print("❌ 错误：center_position 为 None")
        return None, None
    
    print(f"📊 放置区域中心点: {center_position}")
    print()
    
    # 步骤1：随机选择一个方向
    print("【步骤1】随机选择一个方向")
    direction = random.choice([0, 1, 2, 3])
    direction_names = {0: "上", 1: "下", 2: "左", 3: "右"}
    print(f"   - 选择的方向: {direction} ({direction_names[direction]})")
    print()
    
    # 步骤2：滑动选择方向
    print("【步骤2】滑动选择方向")
    distance = 200
    x, y = center_position
    
    if direction == 0:  # 上
        end_position = (x, y - distance)
        print(f"   - 向上滑动 {distance} 像素")
    elif direction == 1:  # 下
        end_position = (x, y + distance)
        print(f"   - 向下滑动 {distance} 像素")
    elif direction == 2:  # 左
        end_position = (x - distance, y)
        print(f"   - 向左滑动 {distance} 像素")
    else:  # 右
        end_position = (x + distance, y)
        print(f"   - 向右滑动 {distance} 像素")
    
    print(f"   - 起点: ({x}, {y})")
    print(f"   - 终点: {end_position}")
    print(f"   - 正在滑动...")
    
    swipe_job = controller.post_swipe(x, y, end_position[0], end_position[1], 300)
    swipe_job.wait()
    print(f"   ✅ 滑动完成")
    print()
    
    print("✅ 动作3成功：调整方向完成")
    return direction, end_position


def test_action4_release_to_deploy(controller, yolo_recognizer, center_position):
    """
    测试动作4：松手完成部署
    
    Args:
        controller: MaaFramework控制器
        yolo_recognizer: YOLO识别器
        center_position: 放置区域中心点 (x, y)
    """
    import time
    
    print("=" * 70)
    print("🎯 测试动作4：松手完成部署")
    print("=" * 70)
    print()
    
    if center_position is None:
        print("❌ 错误：center_position 为 None")
        return False
    
    print(f"📊 放置区域中心点: {center_position}")
    print()
    
    # 步骤1：等待一小段时间让游戏响应
    print("【步骤1】等待一小段时间让游戏响应")
    import time
    time.sleep(0.5)
    print("   ✅ 等待完成")
    print()
    
    # 步骤2：截图
    print("【步骤2】截图")
    print("   - 正在截图...")
    image = controller.post_screencap().wait().get()
    print(f"   ✅ 截图完成，图像形状: {image.shape}")
    print()
    
    # 步骤3：使用YOLO识别干员血条
    print("【步骤3】使用YOLO识别干员血条")
    print("   - 正在进行YOLO检测...")
    
    try:
        detections = yolo_recognizer.detect(image, conf=0.25)
        print(f"   ✅ YOLO检测完成，检测到 {len(detections)} 个目标")
        
        # 打印所有检测结果
        for i, detection in enumerate(detections):
            print(f"      目标 {i+1}:")
            print(f"         - 标签: {detection.label}")
            print(f"         - 置信度: {detection.confidence:.2f}")
            print(f"         - 位置: {detection.box_xyxy}")
        
        operator_hp_bar_detected = any(d.label == "operator_hp_bar" for d in detections)
        
        if operator_hp_bar_detected:
            print(f"   ✅ 检测到干员血条")
            print()
            print("✅ 动作4成功：成功部署")
            return True
        else:
            print(f"   ❌ 未检测到干员血条")
            print()
            print("❌ 动作4失败：未检测到干员血条")
            return False
    except Exception as e:
        print(f"   ❌ YOLO检测出错: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("❌ 动作4失败：YOLO检测出错")
        return False


def test_deployment_process(controller, template_matcher, yolo_recognizer):
    """
    测试部署过程，模拟AI的训练过程
    
    Args:
        controller: MaaFramework控制器
        template_matcher: 模板匹配识别器
        yolo_recognizer: YOLO识别器
    
    Returns:
        success: 是否成功部署
        retry_count: 尝试次数
    """
    print("\n" + "=" * 70)
    print("🎯 测试部署过程（模拟AI训练）")
    print("=" * 70)
    print()
    
    retry_count = 0
    deployment_success = False
    
    while not deployment_success:
        print(f"🔄 第 {retry_count + 1} 次尝试")
        print()
        
        # 步骤1：点击干员头像
        action1_success, operator_position = test_action1_click_operator_avatar(
            controller, template_matcher, save_debug_images=True
        )
        
        if not action1_success:
            print(f"❌ 动作1失败，第 {retry_count + 1} 次重试")
            retry_count += 1
            continue
        
        # 步骤2：拖拽到放置区域
        action2_success, deployment_position = test_action2_drag_to_deployment_area(
            controller, template_matcher, operator_position, save_debug_images=True
        )
        
        if not action2_success:
            print(f"❌ 动作2失败（可能是费用不足），重新选择干员")
            retry_count += 1
            continue
        
        # 步骤3：调整方向
        action3_direction, end_position = test_action3_adjust_direction(
            controller, deployment_position
        )
        
        # 步骤4：松手完成部署
        action4_success = test_action4_release_to_deploy(
            controller, yolo_recognizer, deployment_position
        )
        
        if action4_success:
            print(f"✅ 部署成功！")
            deployment_success = True
        else:
            print(f"❌ 动作4失败，重新开始")
            retry_count += 1
    
    print()
    print("=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print()
    print(f"✅ 部署成功！尝试次数：{retry_count + 1}")
    print()
    
    return deployment_success, retry_count + 1


def main(test_mode="deployment"):
    """
    主函数
    
    Args:
        test_mode: 测试模式
            - "deployment": 测试部署过程（模拟AI训练）
            - "individual": 测试单个动作
    """
    print("\n" + "=" * 70)
    print("🧪 动作空间测试")
    print("=" * 70)
    print()
    print(f"📊 测试模式: {test_mode}")
    if test_mode == "deployment":
        print(f"🔄 无限重试，直到成功部署")
    print()
    
    # 步骤1：连接设备
    print("【步骤1】连接设备")
    try:
        import maa
        from maa.controller import AdbController
        from maa.toolkit import Toolkit
        
        # 初始化 MaaFramework
        Toolkit.init_option(str(ROOT))
        
        # 查找 ADB 设备
        print("⏳ 正在查找 ADB 设备...")
        adb_devices = Toolkit.find_adb_devices()
        
        if not adb_devices:
            print("❌ 未找到 ADB 设备")
            return
        else:
            print(f"✅ 找到 {len(adb_devices)} 个设备")
            device = adb_devices[0]
            print(f"   - 设备: {device.name} ({device.address})")
        
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
        else:
            print("❌ 设备连接失败")
            return
    except Exception as e:
        print(f"❌ 设备连接失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # 步骤2：加载YOLO识别器
    print("【步骤2】加载YOLO识别器")
    try:
        yolo_model_path = str(ROOT / "models" / "yolo" / "best.pt")
        print(f"⏳ 正在加载YOLO模型: {yolo_model_path}")
        
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📊 使用设备: {device}")
        
        yolo_recognizer = YoloRecognizer(model_path=yolo_model_path, device=device)
        yolo_recognizer.load()
        print("✅ YOLO模型加载成功!")
    except Exception as e:
        print(f"❌ YOLO模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # 步骤3：加载模板匹配识别器
    print("【步骤3】加载模板匹配识别器")
    try:
        print("⏳ 正在加载模板匹配器...")
        template_matcher = TemplateMatcher(controller)
        template_matcher.load()
        print("✅ 模板匹配器加载成功!")
    except Exception as e:
        print(f"❌ 模板匹配器加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # 根据测试模式执行测试
    if test_mode == "deployment":
        # 测试部署过程（模拟AI训练）
        deployment_success, retry_count = test_deployment_process(
            controller, template_matcher, yolo_recognizer
        )
    elif test_mode == "individual":
        # 测试单个动作
        # 步骤4：测试动作1
        action1_success, operator_position = test_action1_click_operator_avatar(
            controller, template_matcher, save_debug_images=True
        )
        
        if not action1_success:
            print("\n" + "=" * 70)
            print("📊 测试总结")
            print("=" * 70)
            print()
            print("❌ 动作1失败，终止测试")
            print()
            print("💡 建议:")
            print("   1. 检查游戏是否在战斗界面")
            print("   2. 检查部署区是否有干员")
            print("   3. 检查模板图像是否正确")
            print()
            return
        
        print()
        
        # 步骤5：测试动作2
        action2_success, deployment_position = test_action2_drag_to_deployment_area(
            controller, template_matcher, operator_position, save_debug_images=True
        )
        
        if not action2_success:
            print("\n" + "=" * 70)
            print("📊 测试总结")
            print("=" * 70)
            print()
            print("❌ 动作2失败，终止测试")
            print()
            print("💡 建议:")
            print("   1. 检查游戏是否正确响应了拖拽操作")
            print("   2. 检查是否有绿色高亮区域")
            print("   3. 检查模板图像是否正确")
            print()
            return
        
        print()
        
        # 步骤6：测试动作3
        action3_direction, end_position = test_action3_adjust_direction(
            controller, deployment_position
        )
        
        print()
        
        # 步骤7：测试动作4
        action4_success = test_action4_release_to_deploy(
            controller, yolo_recognizer, deployment_position
        )
        
        print()
        
        # 测试总结
        print("\n" + "=" * 70)
        print("📊 测试总结")
        print("=" * 70)
        print()
        print("✅ 已完成的测试:")
        print("  1. 设备连接")
        print("  2. YOLO模型加载")
        print("  3. 模板匹配器加载")
        print("  4. 动作1：点击干员头像")
        print("  5. 动作2：拖拽到放置区域")
        print("  6. 动作3：调整方向")
        print("  7. 动作4：松手完成部署")
        print()
        print("🎯 测试结果:")
        if action4_success:
            print("  - 所有动作都成功了!")
            print("  - 部署流程正常!")
        else:
            print("  - 动作4失败")
            print("  - 部署流程未完成")
        print()
    else:
        print(f"❌ 未知的测试模式: {test_mode}")
        return
    
    print("=" * 70)


if __name__ == "__main__":
    main()
