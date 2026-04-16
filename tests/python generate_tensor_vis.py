import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 配置路径 =================
# 把这里的路径换成你真实的游戏截图路径
image_path = r"D:\BiShe\combat_screenshot.png"

# ================= 2. 读取并降采样 =================
try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("找不到图片，请检查路径！")
except Exception as e:
    print(e)
    # 如果你懒得找图，代码会为你生成一个纯灰色的假图作为演示
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 100

# 降采样到我们论文里说的 128x72 尺寸
img_resized = cv2.resize(img, (128, 72))

# ================= 3. 模拟生成四个通道的数据 =================

# 通道1: 全局环境 (灰度图)
ch1_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# 通道2: 物理约束 (地形掩膜)
# 我们用代码画一些黑白块，模拟不可部署的高台和禁区
ch2_terrain = np.ones((72, 128), dtype=np.uint8) * 200  # 灰色代表可通行低地
cv2.rectangle(ch2_terrain, (20, 10), (40, 50), 255, -1) # 白色代表高台
cv2.rectangle(ch2_terrain, (80, 20), (100, 60), 0, -1)  # 黑色代表不可部署的墙壁或禁区

# 通道3: 威胁感知 (雷达掩膜)
# 全黑背景，点上几个白点，模拟 MOG2 提取的敌人质心
ch3_radar = np.zeros((72, 128), dtype=np.uint8)
cv2.circle(ch3_radar, (60, 30), 2, 255, -1)
cv2.circle(ch3_radar, (65, 35), 1, 255, -1)
cv2.circle(ch3_radar, (90, 50), 3, 255, -1) # 假装这里有一簇敌人

# 通道4: 资源映射 (手牌状态)
# 底部画几个灰度不同的方块，模拟不同职业的干员卡牌
ch4_resource = np.zeros((72, 128), dtype=np.uint8)
# 画底部背景条
cv2.rectangle(ch4_resource, (0, 60), (128, 72), 30, -1)
# 模拟干员卡槽 (不同灰度代表不同职业：如重装、狙击、医疗)
cv2.rectangle(ch4_resource, (10, 62), (25, 70), 180, -1)
cv2.rectangle(ch4_resource, (30, 62), (45, 70), 220, -1)
cv2.rectangle(ch4_resource, (50, 62), (65, 70), 120, -1)
cv2.rectangle(ch4_resource, (70, 62), (85, 70), 50, -1) # 暗色模拟冷却中

# ================= 4. 用 Matplotlib 渲染出高大上的论文用图 =================

# 设置字体大小和排版
plt.rcParams['font.sans-serif'] = ['SimHei'] # 保证能显示中文标题
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

axes[0, 0].imshow(ch1_gray, cmap='gray')
axes[0, 0].set_title('C1: 全局环境 (Global Env)')

axes[0, 1].imshow(ch2_terrain, cmap='gray')
axes[0, 1].set_title('C2: 物理约束 (Terrain Mask)')

axes[1, 0].imshow(ch3_radar, cmap='gray')
axes[1, 0].set_title('C3: 威胁感知 (Enemy Radar)')

axes[1, 1].imshow(ch4_resource, cmap='gray')
axes[1, 1].set_title('C4: 资源映射 (Resource Map)')

# 隐藏坐标轴的刻度
for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.savefig("four_channel_state_tensor.png", dpi=300) # 保存为高清图片
print("生成完毕！图片已保存为 four_channel_state_tensor.png")
plt.show()