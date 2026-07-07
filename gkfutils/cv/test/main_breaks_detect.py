import cv2
import numpy as np
import math
from collections import defaultdict

# -------------------------- 核心工具函数 --------------------------
def get_skeleton(binary_img):
    """提取图像骨架（细化为1像素宽）"""
    skeleton = cv2.ximgproc.thinning(binary_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return skeleton

def get_endpoints(skeleton):
    """检测骨架所有端点（8邻域像素和为1）"""
    endpoints = []
    h, w = skeleton.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x] == 255:
                # 计算8邻域白色像素数量
                neighbors = skeleton[y-1:y+2, x-1:x+2].sum() // 255
                if neighbors == 2:  # 中心1个 + 邻域1个 = 2
                    endpoints.append((x, y))
    return endpoints

def pair_endpoints(endpoints, max_gap=30):
    """将距离很近的端点配对（一个裂口）"""
    pairs = []
    used = [False] * len(endpoints)
    
    for i in range(len(endpoints)):
        if used[i]:
            continue
        x1, y1 = endpoints[i]
        min_dist = float('inf')
        min_j = -1
        
        for j in range(len(endpoints)):
            if i == j or used[j]:
                continue
            x2, y2 = endpoints[j]
            dist = math.hypot(x1-x2, y1-y2)
            if dist < min_dist and dist <= max_gap:
                min_dist = dist
                min_j = j
        
        if min_j != -1:
            pairs.append((endpoints[i], endpoints[min_j]))
            used[i] = True
            used[min_j] = True
    return pairs

def get_local_region(gray, x, y, k=7):
    """获取端点周围7x7局部区域（用于分析特征）"""
    h, w = gray.shape
    x1 = max(0, x - k)
    x2 = min(w, x + k + 1)
    y1 = max(0, y - k)
    y2 = min(h, y + k + 1)
    return gray[y1:y2, x1:x2], (x1, y1)

def classify_gap(binary, gray, p1, p2, k=7):
    """
    裂口三分类：
    0: 平滑类（端点曲线平滑）
    1: 渐变变暗类（端点周围亮度逐渐降低）
    2: 偏折粗端类（端点偏折、变粗、外延）
    """
    x1, y1 = p1
    x2, y2 = p2

    # ==================== 特征1：检测偏折/粗端/外延 ====================
    def check_bend_thicken(p):
        x, y = p
        # 端点周围连通域面积（正常端点面积很小，粗端/外延面积大）
        mask = np.zeros_like(binary)
        cv2.circle(mask, (x, y), k, 255, -1)
        area = cv2.countNonZero(cv2.bitwise_and(binary, mask))
        
        # 端点方向突变检测（偏折）
        dirs = []
        for dx in [-2,-1,0,1,2]:
            for dy in [-2,-1,0,1,2]:
                if dx==0 and dy==0: continue
                if 0<=y+dy<binary.shape[0] and 0<=x+dx<binary.shape[1]:
                    if binary[y+dy, x+dx]==255:
                        dirs.append((dx, dy))
        if len(dirs) < 2:
            return False
        
        # 计算方向夹角
        v1 = dirs[0]
        v2 = dirs[-1]
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mod = math.hypot(*v1) * math.hypot(*v2)
        if mod == 0:
            angle = 0
        else:
            angle = math.acos(max(-1, min(1, dot/mod))) * 180/math.pi
        
        # 面积大 或 方向突变 → 偏折粗端类
        return area > 40 or angle > 60

    bend1 = check_bend_thicken(p1)
    bend2 = check_bend_thicken(p2)
    if bend1 or bend2:
        return 2

    # ==================== 特征2：检测渐变变暗 ====================
    def check_gradient_dim(p):
        x, y = p
        patch, _ = get_local_region(gray, x, y, k)
        # 计算从中心向外亮度梯度
        center_val = patch[k, k]
        edge_vals = [patch[0,0], patch[0,k], patch[0,-1],
                     patch[k,0], patch[k,-1],
                     patch[-1,0], patch[-1,k], patch[-1,-1]]
        mean_edge = np.mean(edge_vals)
        # 中心亮 → 边缘暗 → 渐变变暗
        return (center_val - mean_edge) > 25

    grad1 = check_gradient_dim(p1)
    grad2 = check_gradient_dim(p2)
    if grad1 or grad2:
        return 1

    # ==================== 默认：平滑类 ====================
    return 0

# -------------------------- 主流程函数 --------------------------
def detect_and_classify_curve_gaps(image_path, output_path="result.png"):
    """
    主函数：检测曲线断裂并分类
    :param image_path: 二值化图像路径
    :param output_path: 结果保存路径
    """
    # 1. 读取图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 确保是纯二值图（0和255）
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 2. 提取骨架 + 检测端点
    skeleton = get_skeleton(binary)
    endpoints = get_endpoints(skeleton)
    print(f"检测到端点数量：{len(endpoints)}")
    
    # 3. 端点配对（形成裂口）
    gap_pairs = pair_endpoints(endpoints, max_gap=40)
    print(f"检测到裂口数量：{len(gap_pairs)}")
    
    # 4. 分类 + 绘制结果
    color_map = {
        0: (0, 255, 0),    # 平滑类：绿色
        1: (255, 0, 0),    # 渐变变暗类：蓝色
        2: (0, 0, 255)     # 偏折粗端类：红色
    }
    label_map = {0:"平滑", 1:"渐变变暗", 2:"偏折粗端"}
    
    result = img.copy()
    for idx, (p1, p2) in enumerate(gap_pairs):
        # 分类
        gap_type = classify_gap(binary, gray, p1, p2)
        color = color_map[gap_type]
        label = label_map[gap_type]
        
        # 计算包围框（包住两个端点）
        x_min = min(p1[0], p2[0]) - 10
        y_min = min(p1[1], p2[1]) - 10
        x_max = max(p1[0], p2[0]) + 10
        y_max = max(p1[1], p2[1]) + 10
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(result.shape[1], x_max), min(result.shape[0], y_max)
        
        # 画框 + 标注
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(result, f"{label}", (x_min, y_min-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"裂口{idx+1}：类别={label}")

    # 保存并显示
    cv2.imwrite(output_path, result)
    cv2.imshow("Gap Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result


import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_left_right_points(contour):
    """
    计算一个轮廓的 最左点 和 最右点
    :param contour: 单个轮廓 (cv2.findContours 输出的其中一个 cnt)
    :return: left_point(x最小), right_point(x最大)
    """
    # 把轮廓展平成 (N,2) 坐标点
    points = contour.reshape(-1, 2)
    
    # 最左：x 最小的点
    left_idx = np.argmin(points[:, 0])
    left_point = tuple(points[left_idx])
    
    # 最右：x 最大的点
    right_idx = np.argmax(points[:, 0])
    right_point = tuple(points[right_idx])
    
    return left_point, right_point


def detect_curve_breaks(image_path, low_thresh=50, high_thresh=150, gap_threshold=10):
    """
    使用Canny边缘检测检测曲线断裂并框出裂口
    :param image_path: 图片路径
    :param low_thresh: Canny低阈值
    :param high_thresh: Canny高阈值
    :param gap_threshold: 断裂判定阈值（像素）
    :return: 标注后的图像
    """
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图片，请检查路径是否正确！")
    output_img = img.copy()  # 用于绘制结果
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊：降噪，提升Canny检测精度
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 3. Canny边缘检测
    # edges = cv2.Canny(blurred, low_thresh, high_thresh, L2gradient=True)
    edges = cv2.Canny(blurred, low_thresh, high_thresh, L2gradient=False)
    
    # 4. 查找所有轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. 检测断裂并绘制矩形框
    break_detected = False
    break_pairs = []
    # 如果轮廓数量 > 1，说明曲线存在断裂（完整曲线只有1个轮廓）
    if len(contours) > 1:
        break_detected = True
        # 遍历所有轮廓，计算外接矩形并框出
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 过滤极小噪点轮廓
            if w > 5 and h > 5:
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                break_pairs.append([(x, y), (x + w, y + h)])
                # 获取最左、最右点
                left_pt, right_pt = get_left_right_points(cnt)
                
                # 画出点
                cv2.circle(output_img, left_pt, 5, (255,0,0), -1)   # 蓝：最左
                cv2.circle(output_img, right_pt, 5, (0,255,0), -1) # 绿：最右

    
    # 6. 添加文字提示
    if break_detected:
        cv2.putText(output_img, "Curve Broken!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(output_img, "Curve Complete!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 7. 结果可视化
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(132), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
    plt.subplot(133), plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)), plt.title('Break Detection')
    plt.show()
    
    return output_img, edges, break_detected, break_pairs


# -------------------------- 运行代码 --------------------------
if __name__ == "__main__":
    # 替换成你的二值化图像路径
    # INPUT_IMAGE = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\laser_detect\v1\images\006_Belt_Torn_Det_aug_0_0000116.jpg"
    # INPUT_IMAGE = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\laser_detect\v1\images\Video_2025_03_08_111948_1_output_000003826.jpg"
    # INPUT_IMAGE = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\laser_detect\v1\images\Video_2025_07_31_115021_2_0000365.jpg"
    # detect_and_classify_curve_gaps(INPUT_IMAGE)


    # 替换为你的图片路径
    # IMAGE_PATH = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\laser_detect\v1\images\006_Belt_Torn_Det_aug_0_0000116.jpg"  
    IMAGE_PATH = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\laser_detect\v1\images\Video_2025_03_08_111948_1_output_000003826.jpg"  
    # IMAGE_PATH = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\laser_detect\v1\images\Video_2025_07_31_115021_2_0000365.jpg"  
    
    # 执行检测
    result_img, canny_edges, is_broken = detect_curve_breaks(
        image_path=IMAGE_PATH,
        low_thresh=50,    # Canny低阈值
        high_thresh=150,  # Canny高阈值
        gap_threshold=10  # 断裂判定阈值
    )
    
    # 输出结果
    if is_broken:
        print("检测结果：曲线存在断裂！已框出裂口位置")
    else:
        print("检测结果：曲线完整，无断裂")



    
    # INPUT_IMAGE = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\laser_detect\v1\images\006_Belt_Torn_Det_aug_0_0000116.jpg"