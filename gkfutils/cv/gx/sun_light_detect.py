import os
import cv2
import numpy as np
from datetime import datetime


def greenToblack(img):
    # 将BGR图像转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义绿色的HSV范围
    # 可以根据需要调整这些值
    lower_green = np.array([35, 50, 50])   # 绿色范围下限
    upper_green = np.array([85, 255, 255]) # 绿色范围上限

    # 创建掩码，标记绿色区域
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 将绿色区域变为黑色
    result = img.copy()
    # 将掩码中绿色区域设置为黑色(BGR)
    result[mask > 0] = [0, 0, 0]

    return result


def isInTargetPeriods(sunRisePeriods=(6, 9), sunSetPeriods=(16, 19)):
    """判断当前时间是否在早上6-9点或下午4-7点"""
    now = datetime.now()
    hour = now.hour
    
    # 早上6-9点或下午4-7点
    result = (sunRisePeriods[0] <= hour < sunRisePeriods[1]) or (sunSetPeriods[0] <= hour < sunSetPeriods[1])
    return result


def detectSunLight(img, areaThresh=100, rs=(0.375, 0.625), edgePixelsThresh=15, sunRisePeriods=(6, 9), sunSetPeriods=(16, 19)):
    """
    这是大图版本
    太阳光基本上在皮带的两边, 所以可以去除中间部分的情况
    可以分成4分, 取中间部分为1/4-3/4, 则 rs = (0.25, 0.75)  
    可以分成8分, 取中间部分为3/8-5/8, 则 rs = (0.375, 0.625) 
    """
    imgsz = img.shape[:2]
    vis = img.copy()

    # 判断当前时间是否处于日出或日落时分
    timeResult = isInTargetPeriods(sunRisePeriods, sunSetPeriods)

    img = greenToblack(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    contours0, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = []
    box = []
    for c in range(len(contours0)):
        area = cv2.contourArea(contours0[c])
        if area < areaThresh:
            continue
        
        # 去除在中间的情况，光照影响基本在皮带两边
        bbx = cv2.boundingRect(contours0[c])
        bbxCenterX = bbx[0] + bbx[2] / 2
        if bbxCenterX > imgsz[1] * rs[0] and bbxCenterX < imgsz[1] * rs[1]:
            continue
        
        # 确保结果是在两边
        bbx_x1 = bbx[0]
        bbx_x2 = bbx[0] + bbx[2]
        if bbx_x1 < edgePixelsThresh or bbx_x2 > imgsz[1] - edgePixelsThresh:
            if not timeResult:
                continue
            contours.append(contours0[c])
            box.append(bbx)
            cv2.rectangle(vis, (bbx[0], bbx[1]), (bbx[0] + bbx[2], bbx[1] + bbx[3]), (0, 255, 0), 2)

    return vis, contours, box


def checkIfAffecttedbySunLight(img, sunRisePeriods=(6, 9), sunSetPeriods=(16, 19)):
    """
    这个是针对裁剪之后的小图
    与上面的detectSunLight需要区别
    """
    imgsz = img.shape[:2]
    vis = img.copy()

    # 判断当前时间是否处于日出或日落时分
    timeResult = isInTargetPeriods(sunRisePeriods, sunSetPeriods)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    # _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    mean = np.mean(blurred)
    if mean > 150:
        return True
    else:
        return False




if __name__ == "__main__":
    img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\DataFromDifferentLocations\NinemeiDaba\CollectedDataByProgram\20251030_merged\data5yi_20251104_20251104083059.jpg"
    # img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\DataFromDifferentLocations\NinemeiDaba\CollectedDataByProgram\20251030_merged\data5yi_20251104_20251104071558.jpg"
    img = cv2.imread(img_path)
    vis, contours, box = detectSunLight(img, areaThresh=100)
    
    cv2.imshow("vis", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


