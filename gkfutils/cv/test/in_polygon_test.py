import numpy as np
import cv2

def is_point_in_polygon(point, polygon):
    """
    判断点是否在多边形内
    :param point: (x, y) 坐标
    :param polygon: 多边形顶点列表 [(x1, y1), (x2, y2), ...]
    :return: True如果在多边形内，否则False
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def filter_boxes_by_center(boxes, polygon):
    """
    根据检测框中心点是否在多边形内进行过滤
    :param boxes: 检测框列表，格式为 [[x1, y1, x2, y2], ...]
    :param polygon: 多边形顶点列表 [(x1, y1), (x2, y2), ...]
    :return: 过滤后的检测框列表
    """
    filtered_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        # 计算检测框中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 判断中心点是否在多边形内
        if is_point_in_polygon((center_x, center_y), polygon):
            filtered_boxes.append(box)
    
    return filtered_boxes

# 方法2：使用OpenCV的更高效方法
def filter_boxes_opencv(boxes, polygon):
    """
    使用OpenCV判断检测框是否在多边形内
    :param boxes: 检测框列表
    :param polygon: 多边形顶点列表
    :return: 过滤后的检测框列表
    """
    filtered_boxes = []
    
    # 将多边形转换为numpy数组
    polygon_np = np.array(polygon, dtype=np.int32)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        # 计算检测框中心点
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # 使用OpenCV的pointPolygonTest函数
        # 返回正值表示点在多边形内，0表示在边界上，负值表示在多边形外
        result = cv2.pointPolygonTest(polygon_np, (center_x, center_y), False)
        
        if result >= 0:  # 点在多边形内或在边界上
            filtered_boxes.append(box)
    
    return filtered_boxes

# 方法3：严格判断（检测框完全在多边形内）
def filter_boxes_completely_inside(boxes, polygon):
    """
    判断检测框的四个角点是否都在多边形内
    :param boxes: 检测框列表
    :param polygon: 多边形顶点列表
    :return: 过滤后的检测框列表
    """
    filtered_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        # 检查四个角点是否都在多边形内
        all_inside = True
        for corner in corners:
            if not is_point_in_polygon(corner, polygon):
                all_inside = False
                break
        
        if all_inside:
            filtered_boxes.append(box)
    
    return filtered_boxes


def visPolygon(img, polygon):
    for p in polygon:
        cv2.circle(img, p, 5, (255, 0, 0), -1)

    for pi in range(len(polygon)):
        if pi == len(polygon) - 1:
            cv2.line(img, polygon[pi], polygon[0], (255, 0, 0), 2)
        else:
            cv2.line(img, polygon[pi], polygon[pi + 1], (255, 0, 0), 2)

    return img



# 示例使用
def main():
    # 示例多边形区域（可以修改为任意多边形）
    polygon = [
        (100, 100),
        (300, 50),
        (500, 150),
        (450, 400),
        (200, 450),
        (50, 300)
    ]
    
    # 示例检测框 [x1, y1, x2, y2]
    boxes = [
        [120, 120, 180, 180],   # 在多边形内
        [350, 200, 420, 280],   # 在多边形内
        [50, 50, 150, 150],     # 部分在多边形外
        [400, 50, 500, 150],    # 在多边形边界上
        [10, 10, 60, 60],       # 完全在多边形外
        [200, 200, 300, 300]    # 在多边形内
    ]
    
    print("原始检测框数量:", len(boxes))
    
    # 使用方法1：中心点判断
    filtered_boxes1 = filter_boxes_by_center(boxes, polygon)
    print("使用方法1过滤后（中心点判断）:", len(filtered_boxes1))
    print("保留的检测框:", filtered_boxes1)
    
    # 使用方法3：严格判断
    filtered_boxes3 = filter_boxes_completely_inside(boxes, polygon)
    print("\n使用方法3过滤后（完全在内）:", len(filtered_boxes3))
    print("保留的检测框:", filtered_boxes3)


    img = np.zeros((1080, 1920, 3), np.uint8)

    img = visPolygon(img, polygon)

    for b in boxes:
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

    for b1 in filtered_boxes1:
        cv2.rectangle(img, (b1[0], b1[1]), (b1[2], b1[3]), (0, 255, 0), 2)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()