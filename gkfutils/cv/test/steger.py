import cv2
import numpy as np

def steger_algorithm(img, sigmaX=1.1, sigmaY=1.1, threshold=30):
    """
    使用Steger算法提取图像中的光条中心线
    
    参数:
        img: 输入图像 (彩色或灰度)
        sigmaX, sigmaY: 高斯滤波的标准差
        threshold: 用于筛选中心点的阈值，基于图像灰度或Hessian特征值
    
    返回:
        centers: 光条中心的亚像素坐标列表 [(x0,y0), (x1,y1), ...]
        result_img: 绘制了中心线的图像
    """
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 高斯滤波
    gray_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigmaX, sigmaY=sigmaY)
    
    # 计算一阶和二阶导数 (使用Scharr滤波器)
    Ix = cv2.Scharr(gray_blur, cv2.CV_32F, 1, 0)
    Iy = cv2.Scharr(gray_blur, cv2.CV_32F, 0, 1)
    Ixx = cv2.Scharr(Ix, cv2.CV_32F, 1, 0)
    Ixy = cv2.Scharr(Ix, cv2.CV_32F, 0, 1)
    Iyy = cv2.Scharr(Iy, cv2.CV_32F, 0, 1)
    
    # 初始化结果
    centers = []
    result_img = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width = gray.shape
    
    for v in range(height):
        for u in range(width):
            # 初步筛选：灰度值高于阈值才处理
            if gray[v, u] < threshold:
                continue
                
            # 构建Hessian矩阵
            H = np.array([[Ixx[v, u], Ixy[v, u]], 
                          [Ixy[v, u], Iyy[v, u]]], dtype=np.float32)
            
            # 计算特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eig(H)
            
            # 获取最大特征值对应的特征向量（法线方向）
            max_idx = np.argmax(np.abs(eigenvalues))
            nx, ny = eigenvectors[:, max_idx]
            max_eigenvalue = eigenvalues[max_idx]
            
            # 检查二阶导数特性（亮条纹要求特征值为负）
            if max_eigenvalue > -1.0:  # 调整阈值以适应你的图像
                continue
                
            # 计算泰勒展开式中的t
            denominator = nx*nx*Ixx[v, u] + 2*nx*ny*Ixy[v, u] + ny*ny*Iyy[v, u]
            if denominator == 0:
                continue
                
            t = -(nx * Ix[v, u] + ny * Iy[v, u]) / denominator
            
            # 检查亚像素偏移是否在合理范围内
            if abs(t * nx) <= 0.5 and abs(t * ny) <= 0.5:
                px = u + t * nx
                py = v + t * ny
                
                # 确保坐标在图像范围内
                if 0 <= px < width and 0 <= py < height:
                    centers.append((px, py))
                    # 在图像上绘制中心点（红色）
                    cv2.circle(result_img, (int(px), int(py)), 1, (0, 0, 255), -1)
    
    return centers, result_img

# 使用示例
if __name__ == "__main__":
    # 读取图像
    img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\DataFromDifferentLocations\NinemeiDaba\CollectedDataByProgram\20251030_merged\data1jia_20251027_20251027131204.jpg"
    img = cv2.imread(img_path)
    
    # 应用Steger算法
    centers, result_img = steger_algorithm(img, sigmaX=1.1, sigmaY=1.1, threshold=100)
    
    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Steger Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"找到 {len(centers)} 个中心点")




