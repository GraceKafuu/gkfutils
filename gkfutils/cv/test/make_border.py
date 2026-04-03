import os
import random
import cv2
import numpy as np


# ===================== 配置参数 =====================
INPUT_FOLDER = r"G:\Gosion\projects\011.ZhongyuanGuanxi\data\001\images"    # 输入图片文件夹
OUTPUT_FOLDER = r"G:\Gosion\projects\011.ZhongyuanGuanxi\data\001\images_output"  # 输出图片文件夹
TARGET_WIDTH = 1920       # 目标宽度
TARGET_HEIGHT = 1080      # 目标高度
FILL_COLORS = [0, 114, 255]  # 随机填充颜色
# ====================================================

# 创建输出文件夹
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def process_image(img_path, save_path):
    """
    处理单张图片：缩放 + 补齐到 1920*1080，随机填充颜色
    """
    # 1. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  无法读取图片: {img_path}")
        return

    h, w = img.shape[:2]
    print(f"处理: {os.path.basename(img_path)} | 原始尺寸: {w}x{h}")

    # 2. 计算缩放比例（只在图片超目标尺寸时缩放）
    scale_w = TARGET_WIDTH / w
    scale_h = TARGET_HEIGHT / h
    scale = min(scale_w, scale_h)  # 等比例缩放，保证不裁剪

    # 只有图片比目标大时才进行 resize
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h

    # 3. 计算需要填充的边距（居中补齐）
    top = (TARGET_HEIGHT - new_h) // 2
    bottom = TARGET_HEIGHT - new_h - top
    left = (TARGET_WIDTH - new_w) // 2
    right = TARGET_WIDTH - new_w - left

    # 4. 随机选择填充颜色
    fill_val = random.choice(FILL_COLORS)

    # 5. 执行填充
    img_padded = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=[fill_val, fill_val, fill_val]  # 灰度值，三通道相同
    )

    # 6. 保存最终图片
    cv2.imwrite(save_path, img_padded)
    print(f"✅ 完成 | 填充颜色: {fill_val} | 保存至: {save_path}\n")


if __name__ == "__main__":
    # 批量处理文件夹所有图片
    for filename in os.listdir(INPUT_FOLDER):
        # 支持常见图片格式
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            img_path = os.path.join(INPUT_FOLDER, filename)
            save_path = os.path.join(OUTPUT_FOLDER, filename)
            process_image(img_path, save_path)

    print("🎉 所有图片处理完成！")