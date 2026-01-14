import cv2
import numpy as np
import math

def detect_lines_hough(image_path):
    """
    使用霍夫变换检测直线并计算倾斜角度
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用Canny边缘检测[citation:2][citation:4]
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 使用霍夫变换检测直线[citation:4]
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    
    # 创建原图的副本用于绘制结果
    result_image = image.copy()
    angles = []
    
    if lines is not None:
        for i, line in enumerate(lines):
            rho, theta = line[0]
            
            # 计算直线的倾斜角度（转换为度数）[citation:3]
            angle = np.degrees(theta)
            
            # 将角度转换到0-180度范围
            if angle > 90:
                angle = angle - 180
            
            angles.append(angle)
            
            # 计算直线上的点
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # 计算直线的两个端点
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            # 在图像上绘制直线
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 显示角度
            cv2.putText(result_image, f"{angle:.2f}deg", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            print(f"直线 {i+1} 倾斜角度: {angle:.2f} 度")
    
    if angles:
        avg_angle = np.mean(angles)
        print(f"\n平均倾斜角度: {avg_angle:.2f} 度")
    else:
        print("未检测到直线")
        avg_angle = 0
    
    # 显示结果
    cv2.imshow('边缘检测', edges)
    cv2.imshow('直线检测结果', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return angles



def detect_lines_houghp(image_path):
    """
    使用概率霍夫变换检测直线线段
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # _, edges = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # 使用概率霍夫变换检测线段[citation:3]
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
    
    result_image = image.copy()
    angles = []
    
    if lines is not None:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            
            # 计算斜率并转换为角度
            slope = (y2 - y1) / (x2 - x1)
            angle = np.degrees(np.arctan(slope))

            # # 将角度转换到0~180度范围
            # if angle < 0:
            #     angle += 180

            if angle < 10 or angle > 60:
                continue
            
            # # 计算线段的角度[citation:3]
            # if x2 - x1 == 0:  # 垂直线
            #     angle = 90
            # elif y2 - y1 == 0:  # 水平线
            #     angle = 0
            # else:
            #     # 计算斜率并转换为角度
            #     slope = (y2 - y1) / (x2 - x1)
            #     angle = np.degrees(np.arctan(slope))

            #     # # 将角度转换到0~180度范围
            #     # if angle < 0:
            #     #     angle += 180

            #     if angle < 10 or angle > 60:
            #         continue
            
            angles.append(angle)

            # 绘制检测到的线段
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在图像上显示角度
            cv2.putText(result_image, f"{angle:.2f}deg", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            print(f"线段 {i+1} 倾斜角度: {angle:.2f} 度")
    
    if angles:
        avg_angle = np.mean(angles)
        print(f"\n平均倾斜角度: {avg_angle:.2f} 度")
    else:
        print("未检测到直线")
        avg_angle = 0
    
    # 显示结果
    cv2.imshow('边缘检测', edges)
    cv2.imshow('线段检测结果', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return angles


def detect_v2(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像，请检查路径！")
        exit()

    # 2. 转换为HSV颜色空间（便于精准筛选黑色）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. 定义黑色的HSV阈值范围（可根据实际光照调整）
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])  # V通道上限控制黑色区域，可微调

    # 4. 生成掩码（仅保留黑色区域）
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 5. 形态学操作（开运算）去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 6. 查找外部轮廓 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. 遍历轮廓，筛选目标滚筒（通过面积、形状特征）
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # 排除过小的噪声轮廓（根据实际场景调整阈值）
            continue
        
        # 计算边界矩形与长宽比
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h  # 滚筒近似圆柱形，长宽比应接近1
        
        # 计算圆度（4π*面积/周长²，圆形的圆度接近1）
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # 筛选符合滚筒特征的轮廓（圆度和长宽比阈值可微调）
        if 0.5 < circularity < 1.0 and 0.7 < aspect_ratio < 1.3:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制检测框

    # 8. 显示结果
    cv2.imshow("Mask（黑色区域掩码）", mask)
    cv2.imshow("Result（检测结果）", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 使用方法
    img_path = r"G:\Gosion\data\009.TuoGun_Det\obb\v1\images\2025-07-22 14_33_08.jpg"
    # detect_lines_hough(img_path)
    # detect_lines_houghp(img_path)
    detect_v2(img_path)

