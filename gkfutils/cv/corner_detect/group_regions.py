import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def group_regions_by_distance(binary_image, max_distance=50, min_samples=1):
    """
    根据白色区域中心点之间的距离进行分组
    
    参数:
    - binary_image: 二值图像(白色区域为255，背景为0)
    - max_distance: 判断为同一组的最大距离阈值
    - min_samples: 每组最少包含的区域数量
    
    返回:
    - grouped_regions: 分组结果，每个元素是一组区域的轮廓列表
    - centers: 所有区域的中心点坐标
    - labels: 每个区域所属的组别标签
    """
    
    # 1. 查找所有白色区域的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return [], [], []
    
    # 2. 计算每个区域的中心点
    centers = []
    valid_contours = []
    
    for contour in contours:
        # 过滤掉太小的区域（可选）
        if cv2.contourArea(contour) < 10:  # 面积阈值
            continue
            
        # 计算矩并获取中心点
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append([cx, cy])
            valid_contours.append(contour)
    
    if len(centers) == 0:
        return [], [], []
    
    centers = np.array(centers)
    
    # 3. 基于距离进行聚类分组
    # 使用DBSCAN聚类算法，eps为距离阈值
    clustering = DBSCAN(eps=max_distance, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    
    # 4. 根据聚类结果分组轮廓
    grouped_regions = []
    n_groups = len(set(labels)) - (1 if -1 in labels else 0)
    
    for group_id in range(n_groups):
        group_contours = [valid_contours[i] for i in range(len(valid_contours)) if labels[i] == group_id]
        grouped_regions.append(group_contours)
    
    # 处理噪声点（label为-1的区域，各自单独成组）
    noise_indices = np.where(labels == -1)[0]
    for idx in noise_indices:
        grouped_regions.append([valid_contours[idx]])
    
    return grouped_regions, centers, labels

def visualize_grouping_result(binary_image, grouped_regions, centers, labels):
    """可视化分组结果"""
    # 创建彩色图像用于可视化
    if len(binary_image.shape) == 2:
        result_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    else:
        result_img = binary_image.copy()
    
    # 定义颜色列表
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]
    
    # 绘制每个组的轮廓和中心点
    for group_id, group_contours in enumerate(grouped_regions):
        color = colors[group_id % len(colors)]

        if len(group_contours) > 1:
            all_points = np.vstack([contour for contour in group_contours])
            group_box = cv2.boundingRect(all_points)
            cv2.rectangle(result_img, group_box, color, 2)
        
        
        for contour in group_contours:
            # 绘制轮廓
            cv2.drawContours(result_img, [contour], -1, color, 2)

            # 计算并绘制中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(result_img, (cx, cy), 5, color, -1)
                cv2.putText(result_img, f'G{group_id}', (cx+10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return result_img


# 使用示例
if __name__ == "__main__":
    # 示例：创建一个测试的二值图像
    test_img = np.zeros((300, 300), dtype=np.uint8)
    
    # 添加几组白色区域
    cv2.rectangle(test_img, (30, 30), (60, 60), 255, -1)  # 组1
    cv2.rectangle(test_img, (40, 100), (70, 130), 255, -1) # 组1
    cv2.rectangle(test_img, (150, 50), (180, 80), 255, -1) # 组2
    cv2.rectangle(test_img, (160, 120), (190, 150), 255, -1) # 组2
    cv2.rectangle(test_img, (250, 200), (280, 230), 255, -1) # 单独区域
    
    # 进行分组
    grouped_regions, centers, labels = group_regions_by_distance(
        test_img, max_distance=80, min_samples=1
    )
    
    print(f"找到 {len(grouped_regions)} 个组")
    for i, group in enumerate(grouped_regions):
        print(f"组 {i}: 包含 {len(group)} 个区域")
    
    # 可视化结果
    result_img = visualize_grouping_result(test_img, grouped_regions, centers, labels)
    
    # 显示结果
    cv2.imshow('Original', test_img)
    cv2.imshow('Grouping Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()