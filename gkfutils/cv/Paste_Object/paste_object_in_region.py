import cv2
import numpy as np
import random
import os
from pathlib import Path
from shapely.geometry import Point, Polygon
from typing import List, Tuple, Optional
from scipy.spatial import ConvexHull


def order_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    将点按照逆时针方向排序
    """
    if len(points) < 3:
        return points
    
    # 转换为numpy数组
    pts = np.array(points)
    
    # 计算几何中心
    center = pts.mean(axis=0)
    
    # 计算每个点相对于中心的角度
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # 按照角度排序（逆时针方向）
    sorted_indices = np.argsort(-angles)  # 负号表示逆时针
    sorted_points = pts[sorted_indices]
    
    return [tuple(p) for p in sorted_points]

def create_polygon_mask(
        points: List[Tuple[float, float]],
        image_size: Optional[Tuple[int, int]] = None,
        order_ccw: bool = True
    ) -> np.ndarray:
    """
    从多边形点集创建掩码(mask)
    
    参数:
    - points: 多边形顶点坐标列表 [(x1, y1), (x2, y2), ...]
    - image_size: 图像大小 (height, width)，如果为None则自动计算
    - return_image: 是否返回OpenCV图像格式
    - order_ccw: 是否按逆时针方向排序点
    
    返回:
    - mask: 二值掩码图像，多边形区域为255，背景为0
    """
    if len(points) < 3:
        raise ValueError("多边形至少需要3个点")
    
    # 如果需要，按逆时针方向排序点
    if order_ccw:
        points = order_points(points)
    
    # 将点转换为整数坐标
    points_array = np.array(points, dtype=np.int32)
    
    # 自动计算图像大小
    if image_size is None:
        max_x = int(np.max(points_array[:, 0])) + 10  # 加一些边界
        max_y = int(np.max(points_array[:, 1])) + 10
        min_x = max(0, int(np.min(points_array[:, 0])) - 10)
        min_y = max(0, int(np.min(points_array[:, 1])) - 10)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # 调整点坐标到新的坐标系
        points_array[:, 0] -= min_x
        points_array[:, 1] -= min_y
        image_size = (height, width)
    else:
        height, width = image_size
    
    # 创建空白图像
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 创建多边形填充
    # 使用cv2.fillPoly填充多边形内部
    # 注意：OpenCV坐标顺序是(x, y)，数组顺序是(height, width)
    polygon = [points_array.reshape(-1, 1, 2)]
    cv2.fillPoly(mask, polygon, color=255)
    
    return mask


def paste_object_in_region(
        imageA_path,
        imageB_path,
        region_points=None, 
        output_dir='output',
        shrink_range=(5, 15),
        use_seamless=True,
        min_scale=0.2,
        max_scale=0.8,
        class_id=0,
        max_attempts=1000
    ):
    """
    将图A随机缩小后粘贴到图B的限定区域内，生成图C和YOLO标签文件
    
    Args:
        imageA_path: 图A路径
        imageB_path: 图B路径
        region_points: 限定区域的多边形点列表，格式为[(x1,y1), (x2,y2), ...]
                      如果为None，则使用整个图B作为区域
        output_dir: 输出目录
        shrink_range: 目标框比图A缩小的像素范围
        use_seamless: 是否使用融合技术
        min_scale: 最小缩放比例
        max_scale: 最大缩放比例
        class_id: YOLO类别ID
        max_attempts: 最大尝试次数，用于在区域内寻找合适位置
    """
    # 创建输出目录
    img_save_path = output_dir + "/images"
    lbl_save_path = output_dir + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    # 读取图像
    imgA = cv2.imread(imageA_path)
    imgB = cv2.imread(imageB_path)

    imgBsz = imgB.shape[:2]
    mask = create_polygon_mask(region_points, imgBsz, order_ccw=False)
    imgB = cv2.bitwise_and(imgB, imgB, mask=mask)
    
    if imgA is None or imgB is None:
        print("错误：无法读取图像文件")
        return None
    
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    
    # 如果没有指定区域，使用整个图像区域
    if region_points is None:
        region_points = [(0, 0), (wB-1, 0), (wB-1, hB-1), (0, hB-1)]
    
    # 创建多边形对象用于判断点是否在区域内
    polygon = Polygon(region_points)
    
    # 获取区域的外接矩形
    min_x, min_y, max_x, max_y = polygon.bounds
    min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
    
    # 1. 随机缩小图A
    scale = random.uniform(min_scale, max_scale)
    
    # 确保缩小后的图A小于区域外接矩形
    max_scale_w = (max_x - min_x - 20) / wA if (max_x - min_x - 20) > 0 else 0.1
    max_scale_h = (max_y - min_y - 20) / hA if (max_y - min_y - 20) > 0 else 0.1
    scale = min(scale, max_scale_w, max_scale_h)
    
    if scale < 0.1:
        print("警告：区域太小，无法放入图A，尝试使用最小缩放")
        scale = 0.1
    
    new_w = max(int(wA * scale), 1)
    new_h = max(int(hA * scale), 1)

    # 调整大小
    if new_w < 64 and new_h < 64:
        imgA_resized = imgA
    else:
        imgA_resized = cv2.resize(imgA, (new_w, new_h))
    
    # 2. 在区域内随机选择粘贴位置
    found_position = False
    paste_x, paste_y = 0, 0
    
    for attempt in range(max_attempts):
        # 在区域内随机选择位置
        if attempt < max_attempts // 2:
            # 在前半部分尝试中，使用外接矩形内的随机位置
            candidate_x = random.randint(min_x, max(0, max_x - new_w))
            candidate_y = random.randint(min_y, max(0, max_y - new_h))
        else:
            # 在后半部分尝试中，使用多边形内的随机位置采样
            while True:
                candidate_x = random.randint(min_x, max_x)
                candidate_y = random.randint(min_y, max_y)
                if polygon.contains(Point(candidate_x, candidate_y)):
                    break
            # 调整确保不超出边界
            candidate_x = max(min_x, min(candidate_x, max_x - new_w))
            candidate_y = max(min_y, min(candidate_y, max_y - new_h))
        
        # 检查四个角点是否都在多边形内
        corners = [
            Point(candidate_x, candidate_y),
            Point(candidate_x + new_w - 1, candidate_y),
            Point(candidate_x + new_w - 1, candidate_y + new_h - 1),
            Point(candidate_x, candidate_y + new_h - 1)
        ]
        
        all_inside = all(polygon.contains(corner) for corner in corners)
        
        if all_inside:
            paste_x, paste_y = candidate_x, candidate_y
            found_position = True
            break
    
    if not found_position:
        print(f"警告：在{max_attempts}次尝试后未找到合适位置，使用近似位置")
        # 使用外接矩形的中心作为近似位置
        paste_x = max(min_x, min((min_x + max_x - new_w) // 2, max_x - new_w))
        paste_y = max(min_y, min((min_y + max_y - new_h) // 2, max_y - new_h))
    
    # 3. 创建结果图像（复制图B）
    imgC = imgB.copy()
    
    # 4. 粘贴图A到图B
    if use_seamless:
        # 使用seamlessClone融合技术
        mask = 255 * np.ones(imgA_resized.shape, imgA_resized.dtype)
        center = (paste_x + new_w // 2, paste_y + new_h // 2)
        # NORMAL_CLONE、MIXED_CLONE和MONOCHROME_TRANSFER
        # imgC = cv2.seamlessClone(imgA_resized, imgB, mask, center, cv2.NORMAL_CLONE)
        imgC = cv2.seamlessClone(imgA_resized, imgB, mask, center, cv2.MIXED_CLONE)
    else:
        # 直接粘贴
        # 创建图A的掩码（如果是PNG带透明度）
        if imgA_resized.shape[2] == 4:
            # 分离颜色和透明度通道
            rgb = imgA_resized[:, :, :3]
            alpha = imgA_resized[:, :, 3] / 255.0
            
            # 提取ROI区域
            roi = imgC[paste_y:paste_y+new_h, paste_x:paste_x+new_w]
            
            # 混合图像
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + rgb[:, :, c] * alpha
        else:
            imgC[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = imgA_resized
    
    # 5. 生成YOLO标签（目标框比图A小一些）
    shrink_pixels = random.randint(shrink_range[0], shrink_range[1])
    
    # 计算边界框（YOLO格式：中心点x, 中心点y, 宽度, 高度，均归一化到[0,1]）
    bbox_x = paste_x + new_w / 2
    bbox_y = paste_y + new_h / 2
    bbox_w = max(new_w - shrink_pixels, 1)  # 确保宽度至少为1像素
    bbox_h = max(new_h - shrink_pixels, 1)  # 确保高度至少为1像素
    
    # 归一化
    bbox_x_norm = bbox_x / wB
    bbox_y_norm = bbox_y / hB
    bbox_w_norm = bbox_w / wB
    bbox_h_norm = bbox_h / hB
    
    # 确保归一化值在[0,1]范围内
    bbox_x_norm = max(0.0, min(1.0, bbox_x_norm))
    bbox_y_norm = max(0.0, min(1.0, bbox_y_norm))
    bbox_w_norm = max(0.0, min(1.0, bbox_w_norm))
    bbox_h_norm = max(0.0, min(1.0, bbox_h_norm))
    
    # 6. 保存图像和标签
    base_nameA = os.path.splitext(os.path.basename(imageA_path))[0]
    base_nameB = os.path.splitext(os.path.basename(imageB_path))[0]
    random_num = str(np.random.random()).replace(".", "").replace("-", "")
    output_img_path = "{}/{}_{}_{}.jpg".format(img_save_path, base_nameA, base_nameB, random_num)
    output_lbl_path = "{}/{}_{}_{}.txt".format(lbl_save_path, base_nameA, base_nameB, random_num)

    cv2.imwrite(output_img_path, imgC)
    # 保存YOLO格式标签
    with open(output_lbl_path, 'w') as f:
        f.write(f"{class_id} {bbox_x_norm:.6f} {bbox_y_norm:.6f} {bbox_w_norm:.6f} {bbox_h_norm:.6f}\n")
    
    # 7. 打印信息
    print(f"图A原始大小: {wA}x{hA}")
    print(f"图A缩放后大小: {new_w}x{new_h}")
    print(f"粘贴位置: x={paste_x}, y={paste_y}")
    print(f"边界框位置: 中心({bbox_x:.1f}, {bbox_y:.1f}), 大小{bbox_w}x{bbox_h}")
    print(f"YOLO标签: {class_id} {bbox_x_norm:.6f} {bbox_y_norm:.6f} {bbox_w_norm:.6f} {bbox_h_norm:.6f}")
    print(f"结果图像已保存: {output_img_path}")
    print(f"标签文件已保存: {output_lbl_path}")
    
    # 8. 返回结果信息
    result_info = {
        'image_path': output_img_path,
        'label_path': output_lbl_path,
        'paste_position': (paste_x, paste_y),
        'resized_size': (new_w, new_h),
        'bbox': (bbox_x, bbox_y, bbox_w, bbox_h),
        'yolo_bbox': (bbox_x_norm, bbox_y_norm, bbox_w_norm, bbox_h_norm),
        'region_points': region_points
    }
    
    return result_info, imgC


def visualize_result(imgC, label_path, region_points=None, save_path=None):
    """
    可视化结果，显示边界框和限定区域
    Args:
        imgC: 结果图像
        label_path: YOLO标签文件路径
        region_points: 限定区域的多边形点列表
        save_path: 可视化结果保存路径
    """
    # 创建副本用于可视化
    vis_img = imgC.copy()
    
    # 绘制限定区域（如果提供）
    if region_points is not None:
        # 将点转换为numpy数组
        points = np.array(region_points, dtype=np.int32)
        # 绘制多边形
        cv2.polylines(vis_img, [points], True, (255, 0, 0), 2)
        
        # 标记多边形顶点
        for i, (x, y) in enumerate(region_points):
            cv2.circle(vis_img, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(vis_img, str(i), (x+5, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # 读取并绘制边界框
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if line:
                parts = line.split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                h, w = vis_img.shape[:2]
                
                # 将归一化坐标转换回像素坐标
                bbox_w = int(width * w)
                bbox_h = int(height * h)
                bbox_x = int(x_center * w - bbox_w / 2)
                bbox_y = int(y_center * h - bbox_h / 2)
                
                # 绘制边界框
                cv2.rectangle(vis_img, (bbox_x, bbox_y), 
                            (bbox_x + bbox_w, bbox_y + bbox_h), 
                            (0, 255, 0), 2)
                
                # 添加标签
                label = f"Class {class_id}"
                cv2.putText(vis_img, label, (bbox_x, bbox_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 绘制边界框中心点
                center_x = int(x_center * w)
                center_y = int(y_center * h)
                cv2.circle(vis_img, (center_x, center_y), 4, (0, 0, 255), -1)
    
    # 显示图像
    cv2.imshow('Result with Bounding Box and Region', vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存可视化结果
    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"可视化结果已保存: {save_path}")
    
    return vis_img





# 主程序示例
if __name__ == "__main__":
    # imageA_path = r"G:\Gosion\data\009.TuoGun_Det\images\ScreenShot_2025-12-31_103825_318.png"  # 替换为实际路径
    # imageB_path = r"G:\Gosion\data\009.TuoGun_Det\images\IMG_20251204_113908.jpg"  # 替换为实际路径

    n = 6
    imgA_data_path = r"G:\Gosion\data\009.TuoGun_Det\images\A"
    imgB_data_path = r"G:\Gosion\data\009.TuoGun_Det\videos\frames\20251218_merged\{}".format(n)

    save_path = r"G:\Gosion\data\009.TuoGun_Det\data\v1\train"

    regions = {
        '0': [
        (167, 115),
        (309, 231),
        (575, 431), 
        (1033, 720),
        (1280, 720),
        (1280, 300)
    ],
    '1': [
        (365, 643),
        (451, 675),
        (557, 699), 
        (639, 720),
        (1280, 720),
        (1280, 460)
    ],
    '2': [
        (224, 68),
        (983, 720),
        (1280, 720), 
        (1280, 300)
    ],
    '3': [
        (395, 540),
        (480, 580),
        (654, 633),
        (1280, 720), 
        (1280, 470)
    ],
    '4': [
        (76, 141),
        (640, 720),
        (1280, 720), 
        (1280, 300)
    ],
    '5': [
        (335, 688),
        (354, 720),
        (1280, 720), 
        (1280, 470)
    ]
    }
    
    a_file_list = sorted(os.listdir(imgA_data_path))
    b_file_list = sorted(os.listdir(imgB_data_path))

    for f in b_file_list:
        try:
            selected_a = random.sample(a_file_list, 1)
            imageA_path = os.path.join(imgA_data_path, selected_a[0])

            imageB_path = os.path.join(imgB_data_path, f)

            result_info, imgC = paste_object_in_region(
                imageA_path, 
                imageB_path, 
                region_points=regions["{}".format(n - 1)],
                output_dir=save_path,
                use_seamless=False,  # 这次使用直接粘贴
                shrink_range=(5, 15)
            )
        except Exception as e:
            print(e)
            continue
    
    
    
    