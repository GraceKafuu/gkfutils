import os
import cv2
import numpy as np


def z_score(x, mean, std):
    return (x - mean) / std


def find_nonzero_groups(arr):
    """高性能版本"""
    # 创建非零掩码
    mask = arr != 0
    
    if not np.any(mask):
        return []

    # 找到变化点
    change_points = np.where(np.diff(np.r_[False, mask, False]))[0]
    
    # 重组为(start, end)对
    starts = change_points[::2]
    ends = change_points[1::2] - 1
    
    return list(zip(starts, ends))


def createMask(img, zscore_thr=5, expand_pixels=25):
    """ 只适用于黑白相机版本 """
    mask = np.zeros(img.shape, np.uint8)

    imgsz = img.shape[:2]

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rowSums = np.sum(img, axis=1)

    row_sums_mean = np.mean(rowSums)
    row_sums_std = np.std(rowSums)
    zscore = z_score(rowSums, row_sums_mean, row_sums_std)

    x_coords1 = np.where(abs(zscore) > zscore_thr)
    groups_zscore = find_nonzero_groups(x_coords1[0])

    laser_row_idx = imgsz[0] // 2

    for i, (start, end) in enumerate(groups_zscore):
        start = x_coords1[0][start]
        end = x_coords1[0][end]

        mid = (start + end) // 2

        if abs(end - start) < 5: continue

        if mid > imgsz[0] * 0.25 and mid < imgsz[0] * 0.75:
            laser_row_idx = mid
            expand_pixels = max(expand_pixels, end - mid)

    laser_row_dix0 = laser_row_idx - expand_pixels
    laser_row_dix1 = laser_row_idx + expand_pixels
    if laser_row_dix0 < 0: laser_row_dix0 = 0
    if laser_row_dix1 > imgsz[0]: laser_row_dix1 = imgsz[0]

    mask[laser_row_dix0:laser_row_dix1, :] = 255
    mask_row_idxs = (laser_row_dix0, laser_row_dix1)

    return mask, mask_row_idxs




    






if __name__ == "__main__":
    main()