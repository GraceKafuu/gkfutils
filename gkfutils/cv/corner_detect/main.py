import os
import cv2
import time
import numpy as np



def main():
    img_path = r"G:\\Gosion\\data\\006.Belt_Torn_Det\\data\\seg\\v2_mini\\val\\images\\1_output_000000005.jpg"
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)  # 转换为浮点型
    # dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)  # Harris 角点检测
    t1 = time.time()
    dst = cv2.cornerHarris(gray, blockSize=10, ksize=7, k=0.04)  # Harris 角点检测
    t2 = time.time()
    print("Harris角点检测耗时：", t2 - t1)

    r, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)  # 二值化阈值处理
    dst = np.uint8(dst)  # 转换为整型
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()
