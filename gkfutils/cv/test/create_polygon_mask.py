import numpy as np
import cv2
import matplotlib.pyplot as plt
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



if __name__ == "__main__":
    # points = [(336, 692), (369, 720), (1280, 720), (1280, 490)]
    points = [(170, 122), (1005, 720), (1280, 720), (1280, 190), (969, 119), (715, 90), (646, 127), (458, 160)]
    # img = cv2.imread(r"G:\Gosion\data\009.TuoGun_Det\videos\frames\10.58.136_137_20251219_20251225_rename_0000271.jpg")
    img = cv2.imread(r"G:\Gosion\data\009.TuoGun_Det\videos\frames\10.58.136.137_552_20251219_20251225_rename_0000008.jpg")
    imgsz = img.shape[:2]
    mask = create_polygon_mask(points, imgsz, order_ccw=True)

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("mask", mask)
    cv2.imshow("masked_img", masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
