import os
import cv2
import time
import numpy as np
from group_regions import group_regions_by_distance, visualize_grouping_result


def main0():
    # img_path = r"G:\\Gosion\\data\\006.Belt_Torn_Det\\data\\seg\\v2_mini\\val\\images\\1_output_000000005.jpg"
    img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\videos\lainjiangsilie\Video_2025_09_26_103048_1_frames\Video_2025_09_26_103048_1\Video_2025_09_26_103048_1_output_000000001.jpg"
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)  # 转换为浮点型
    # dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)  # Harris 角点检测
    t1 = time.time()
    dst = cv2.cornerHarris(gray, blockSize=10, ksize=7, k=0.04)  # Harris 角点检测
    t2 = time.time()
    print("Harris角点检测耗时: ", t2 - t1)

    r, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)  # 二值化阈值处理
    dst = np.uint8(dst)  # 转换为整型

    grouped_regions, centers, labels = group_regions_by_distance(
        dst, max_distance=50, min_samples=1
    )
    
    print(f"找到 {len(grouped_regions)} 个组")
    for i, group in enumerate(grouped_regions):
        print(f"组 {i}: 包含 {len(group)} 个区域")
    
    # 可视化结果
    result_img = visualize_grouping_result(dst, grouped_regions, centers, labels)



    dstBGR = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    added = cv2.addWeighted(img, 0.5, dstBGR, 0.5, 0, img)

    cv2.imshow('dst', dst)
    cv2.imshow('added', added)
    cv2.imshow('result_img', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(laser_ort, fit_deg):
    out_points = laser_ort.detect(img, vis=False)
    fit_model = np.polyfit(out_points[:, 0], out_points[:, 1], deg=fit_deg)
    fit_model = check_fitted_model(img, fit_model)
    # 假设当前拟合的曲线模型无效，俺么应该是画面有变动导致的，则使用上一次拟合曲线模型
    if fit_model is None:
        if last_fit_model is not None:
            fit_model = last_fit_model
    else:
        last_fit_model = fit_model


if __name__ == "__main__":
    main()
