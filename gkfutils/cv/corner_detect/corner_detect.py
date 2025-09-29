import os
import cv2
import time
import numpy as np



def corner_detect(img, blockSize=10, ksize=7, k=0.04, show=False):
    imgsz = img.shape

    if len(imgsz) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(imgsz) == 2:
        gray = img
    else:
        raise Exception("img shape error!")

    gray = np.float32(gray)  # 转换为浮点型
    dst = cv2.cornerHarris(gray, blockSize, ksize, k)  # Harris 角点检测

    r, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)  # 二值化阈值处理
    dst = np.uint8(dst)  # 转换为整型

    return dst