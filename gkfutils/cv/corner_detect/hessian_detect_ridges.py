# Source - https://stackoverflow.com/a
# Posted by billylanchantin
# Retrieved 2025-11-07, License - CC BY-SA 4.0

import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from numba import jit, prange
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # 屏蔽numba首次编译警告


def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def plot_images(*images):
    images = list(images)
    n = len(images)
    fig, ax = plt.subplots(ncols=n, sharey=True)
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.show()


def main():
    data_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\DataFromDifferentLocations\NinemeiDaba\CollectedDataByProgram\20251030_merged"
    save_path = data_path + "_detect_ridges_results"
    os.makedirs(save_path, exist_ok=True)   

    file_list = os.listdir(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path, 0) # 0 imports a grayscale
        if img is None:
            raise(ValueError(f"Image didn\'t load. Check that '{f_abs_path}' exists."))
        
        t1 = time.time()
        a, b = detect_ridges(img, sigma=3.0)
        t2 = time.time()
        print(f"{fname} took {t2-t1} seconds")
        # plot_images(img, a, b)
        # pass
        # a = np.uint8(a * 255)
        b = np.uint8(b * 255)
        # res = np.vstack((a, b))
        f_dst_path = save_path + "/{}.png".format(fname)
        cv2.imwrite(f_dst_path, b)


# =======================================================================

def fast_hessian_matrix_opencv(gray, sigma=1.0):
    """
    使用OpenCV快速计算Hessian矩阵及其特征值
    """
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # 使用Scharr算子计算一阶和二阶导数（比Sobel更精确）
    Ix = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
    Iy = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
    
    Ixx = cv2.Scharr(Ix, cv2.CV_32F, 1, 0)
    Ixy = cv2.Scharr(Ix, cv2.CV_32F, 0, 1)
    Iyy = cv2.Scharr(Iy, cv2.CV_32F, 0, 1)
    
    return Ixx, Ixy, Iyy

def fast_hessian_eigvals_opencv(Ixx, Ixy, Iyy):
    """
    快速计算Hessian矩阵特征值（向量化实现）
    """
    # 解析解计算2x2矩阵特征值，避免通用特征值分解
    trace = Ixx + Iyy
    determinant = Ixx * Iyy - Ixy * Ixy
    
    # 计算特征值: λ = (trace ± sqrt(trace² - 4*determinant)) / 2
    discriminant = trace * trace - 4 * determinant
    discriminant = np.maximum(discriminant, 0)  # 避免负数
    
    sqrt_discriminant = np.sqrt(discriminant)
    
    lambda1 = (trace + sqrt_discriminant) / 2
    lambda2 = (trace - sqrt_discriminant) / 2
    
    # 按绝对值大小排序
    maxima_ridges = np.where(np.abs(lambda1) > np.abs(lambda2), lambda1, lambda2)
    minima_ridges = np.where(np.abs(lambda1) > np.abs(lambda2), lambda2, lambda1)
    
    return maxima_ridges, minima_ridges

def fast_detect_ridges(gray, sigma=1.0):
    """
    快速脊线检测
    """
    Ixx, Ixy, Iyy = fast_hessian_matrix_opencv(gray, sigma)
    return fast_hessian_eigvals_opencv(Ixx, Ixy, Iyy)


def multiscale_fast_hessian(gray, sigma=1.0, scale_factor=0.5):
    """
    多尺度下采样加速，处理后再上采样恢复分辨率
    """
    # 下采样
    if scale_factor < 1.0:
        h, w = gray.shape
        new_size = (int(w * scale_factor), int(h * scale_factor))
        small_gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    else:
        small_gray = gray
    
    # 在小图上计算Hessian
    Ixx, Ixy, Iyy = fast_hessian_matrix_opencv(small_gray, sigma)
    maxima_small, minima_small = fast_hessian_eigvals_opencv(Ixx, Ixy, Iyy)
    
    # 上采样恢复分辨率
    if scale_factor < 1.0:
        maxima_ridges = cv2.resize(maxima_small, (w, h), interpolation=cv2.INTER_LINEAR)
        minima_ridges = cv2.resize(minima_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return maxima_ridges, minima_ridges
    else:
        return maxima_small, minima_small
    

@jit(nopython=True, parallel=True, fastmath=True)
def numba_hessian_eigvals(Ixx, Ixy, Iyy):
    """
    使用Numba并行计算Hessian特征值
    """
    h, w = Ixx.shape
    maxima = np.zeros((h, w), dtype=np.float32)
    minima = np.zeros((h, w), dtype=np.float32)
    
    for i in prange(h):
        for j in prange(w):
            a = Ixx[i, j]
            b = Ixy[i, j] 
            c = Iyy[i, j]
            
            # 计算特征值
            trace = a + c
            det = a * c - b * b
            discriminant = trace * trace - 4 * det
            
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                lambda1 = (trace + sqrt_disc) / 2
                lambda2 = (trace - sqrt_disc) / 2
                
                if abs(lambda1) > abs(lambda2):
                    maxima[i, j] = lambda1
                    minima[i, j] = lambda2
                else:
                    maxima[i, j] = lambda2
                    minima[i, j] = lambda1
    
    return maxima, minima


def numba_detect_ridges(gray, sigma=1.0):
    """
    Numba加速的脊线检测
    """
    Ixx, Ixy, Iyy = fast_hessian_matrix_opencv(gray, sigma)
    return numba_hessian_eigvals(Ixx, Ixy, Iyy)


def roi_fast_hessian(gray, sigma=1.0, intensity_threshold=50):
    """
    只在感兴趣区域计算Hessian，大幅减少计算量
    """
    # 创建二值掩码，只在高强度区域计算
    _, mask = cv2.threshold(gray, intensity_threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    
    # 计算整个图像的Hessian（但只在mask区域使用结果）
    Ixx, Ixy, Iyy = fast_hessian_matrix_opencv(gray, sigma)
    maxima, minima = fast_hessian_eigvals_opencv(Ixx, Ixy, Iyy)
    
    # 将非ROI区域置零
    maxima = maxima * (mask > 0)
    minima = minima * (mask > 0)
    
    return maxima, minima


def benchmark_methods(gray_image):
    """
    性能基准测试
    """
    methods = {
        "Original skimage": lambda: detect_ridges(gray_image, sigma=3.0),
        "OpenCV Fast": lambda: fast_detect_ridges(gray_image, sigma=3.0),
        "Multiscale (0.5x)": lambda: multiscale_fast_hessian(gray_image, sigma=3.0, scale_factor=0.5),
        "Numba JIT": lambda: numba_detect_ridges(gray_image, sigma=3.0),
        "ROI Processing": lambda: roi_fast_hessian(gray_image, sigma=3.0, intensity_threshold=50)
    }
    
    print(f"图像尺寸: {gray_image.shape}")
    print("=" * 50)
    
    for name, method in methods.items():
        # 预热
        if "Numba" in name:
            method()
        
        # 计时
        start_time = time.time()
        result = method()
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000
        print(f"{name:<20}: {execution_time:6.2f} ms")
        
    print("=" * 50)


def ultimate_fast_ridges(gray, sigma=1.0, use_roi=True, use_multiscale=True):
    """
    终极优化方案：结合多种加速技术
    """
    if use_multiscale:
        # 多尺度处理
        return multiscale_fast_hessian(gray, sigma, scale_factor=0.5)
    elif use_roi:
        # ROI处理
        return roi_fast_hessian(gray, sigma)
    else:
        # 纯OpenCV加速
        return fast_detect_ridges(gray, sigma)
    
# ===================================================================================


def detect_ridges_original(gray, sigma=1.0):
    """原脊线检测函数"""
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

# -------------------------- 优化后的函数 --------------------------
@jit(nopython=True, cache=True, fastmath=True)
def _compute_eigenvalues_numba(Hrr, Hcc, Hrc):
    """
    Numba加速的2x2矩阵特征值计算（解析解）
    输入：Hrr(二阶行导数), Hcc(二阶列导数), Hrc(混合导数)
    输出：max特征值（maxima_ridges）、min特征值（minima_ridges）
    """
    rows, cols = Hrr.shape
    lambda1 = np.empty((rows, cols), dtype=np.float32)
    lambda2 = np.empty((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            hrr = Hrr[i, j]
            hcc = Hcc[i, j]
            hrc = Hrc[i, j]
            
            # 2x2对称矩阵特征值解析公式
            trace = hrr + hcc
            det = hrr * hcc - hrc * hrc
            sqrt_term = np.sqrt((hrr - hcc)**2 + 4 * hrc**2)
            
            lambda1[i, j] = (trace + sqrt_term) / 2.0
            lambda2[i, j] = (trace - sqrt_term) / 2.0
    return lambda1, lambda2

def detect_ridges_optimized(gray, sigma=1.0):
    """
    优化后的脊线检测函数
    :param gray: 灰度图（uint8/float32）
    :param sigma: 高斯核标准差
    :return: maxima_ridges, minima_ridges（与原函数结果一致）
    """
    # 1. 数据预处理：转为float32 + 内存连续化
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    gray = np.ascontiguousarray(gray)
    
    # 2. 计算高斯核大小（覆盖99%的高斯分布）
    ksize = 2 * int(3 * sigma) + 1
    if ksize < 3:  # 最小核大小，避免Sobel出错
        ksize = 3
    
    # 3. OpenCV计算二阶导数（Hrr=dyy, Hcc=dxx, Hrc=dxy）
    # 一阶导数
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)  # 一阶x（列）
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)  # 一阶y（行）
    
    # 二阶导数
    Hcc = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=ksize)  # 二阶x (dxx)
    Hrr = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=ksize)  # 二阶y (dyy)
    Hrc = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=ksize)  # 混合导数 (dxy)
    
    # 4. Numba加速计算特征值
    maxima_ridges, minima_ridges = _compute_eigenvalues_numba(Hrr, Hcc, Hrc)
    
    return maxima_ridges, minima_ridges

# -------------------------- 测试对比 --------------------------
def test_performance():
    """测试原函数与优化函数的速度和效果一致性"""
    # 生成测试图像（1700x600灰度图）
    img = np.random.randint(0, 255, (600, 1700), dtype=np.uint8)
    sigma = 1.0
    
    # 预热（Numba第一次运行需要编译，排除编译时间）
    _ = detect_ridges_optimized(img, sigma)
    
    # 测试原函数速度
    t1 = time.time()
    max_orig, min_orig = detect_ridges_original(img, sigma)
    t_orig = (time.time() - t1) * 1000  # 转毫秒
    print(f"原函数耗时：{t_orig:.2f} ms")
    
    # 测试优化函数速度
    t2 = time.time()
    max_opt, min_opt = detect_ridges_optimized(img, sigma)
    t_opt = (time.time() - t2) * 1000
    print(f"优化函数耗时：{t_opt:.2f} ms")
    print(f"速度提升：{t_orig/t_opt:.2f} 倍")
    
    # 验证效果一致性（误差小于1e-4视为一致）
    max_diff = np.max(np.abs(max_orig.astype(np.float32) - max_opt))
    min_diff = np.max(np.abs(min_orig.astype(np.float32) - min_opt))
    print(f"最大值脊线最大误差：{max_diff:.6f}")
    print(f"最小值脊线最大误差：{min_diff:.6f}")
    print(f"效果是否一致：{max_diff < 1e-4 and min_diff < 1e-4}")


def keep_top5_largest_components(img, k=5, connectivity=8):
    """
    保留连通域分析中面积最大的前5个连通域，其余置0
    :param img: 输入二值图像（建议uint8类型，0为背景，255为前景）
    :param connectivity: 连通域连接方式，4或8（默认8）
    :return: 仅保留前5大连通域的二值图像
    """
    # 1. 连通域分析（注意：输入需为二值图，非二值图需先二值化）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=connectivity
    )
    
    # 2. 处理边界情况：无前景连通域（仅背景）
    if num_labels <= 1:
        return np.zeros_like(img)
    
    # 3. 提取前景连通域的标签和面积（排除标签0：背景）
    # stats[:,4] 是每个标签的面积，索引0为背景，从1开始是前景
    foreground_labels = np.arange(1, num_labels)
    foreground_areas = stats[1:, (k - 1)]  # 前景面积
    
    # 4. 按面积降序排序，取前5个标签（不足5个则取全部）
    # 得到面积排序后的标签索引
    sorted_indices = np.argsort(foreground_areas)[::-1]  # 降序
    top5_indices = sorted_indices[:k]  # 前5大的索引
    top5_labels = foreground_labels[top5_indices]  # 前5大的连通域标签
    
    # 5. 创建掩码：仅保留前5大连通域的像素
    mask = np.isin(labels, top5_labels).astype(np.uint8) * 255
    
    # 6. 应用掩码到原图像（确保输出与原图像格式一致）
    result = cv2.bitwise_and(img, img, mask=mask)
    
    return result

from typing import List, Dict, Tuple
def process_top5_components(
    img: np.ndarray,
    connectivity: int = 8,
    min_area_threshold: float = 0.0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    保留连通域分析中面积最大的前5个连通域（其余置0）；若前5个面积均小于阈值，则全部置0
    :param img: 输入二值图像（uint8类型，0为背景，255为前景）
    :param connectivity: 连通域连接方式，4或8（默认8）
    :param min_area_threshold: 面积阈值，若前5大连通域的最大面积小于此值则全置0
    :return: 
        - processed_img: 处理后的二值图像
        - component_info: 所有连通域的详细信息列表（含背景），每个元素为字典：
            {
                "label": 连通域标签（0为背景）,
                "area": 连通域面积,
                "width": 连通域宽度,
                "height": 连通域高度,
                "bbox_xywh": 外包围框（xywh格式：[x, y, width, height]）,
                "bbox_xyxy": 外包围框（xyxy格式：[x1, y1, x2, y2]）,
                "centroid": 质心坐标（(cx, cy)）,
                "is_kept": 是否被保留（True/False）
            }
    """
    # 1. 连通域分析（输入需为二值图，非二值图需先二值化）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=connectivity
    )
    
    # 2. 初始化连通域信息列表（包含背景）
    component_info = []
    for label in range(num_labels):
        # stats[label] 结构：[x, y, width, height, area]
        x, y, w, h, area = stats[label]
        cx, cy = centroids[label]
        
        # 计算外包围框的xyxy格式（x1,y1为左上角，x2,y2为右下角）
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        component_info.append({
            "label": label,
            "area": int(area),
            "width": int(w),
            "height": int(h),
            "bbox_xywh": [x1, y1, int(w), int(h)],  # xywh格式
            "bbox_xyxy": [x1, y1, x2, y2],            # xyxy格式（更通用的包围框格式）
            "centroid": (float(cx), float(cy)),
            "is_kept": False  # 初始标记为未保留
        })
    
    # 3. 处理边界情况：无前景连通域（仅背景）
    if num_labels <= 1:
        return np.zeros_like(img), component_info
    
    # 4. 提取前景连通域（排除背景label=0）
    foreground_info = [info for info in component_info if info["label"] != 0]
    
    # 5. 按面积降序排序前景连通域，取前5个（不足5个则取全部）
    # 优化：若连通域极多，可替换为np.partition（见下方备注）
    foreground_info_sorted = sorted(foreground_info, key=lambda x: x["area"], reverse=True)
    top5_info = foreground_info_sorted[:5]
    
    # 6. 阈值判断：前5个中最大面积是否≥阈值
    keep_top5 = False
    if top5_info:
        max_top5_area = max([info["area"] for info in top5_info])
        if max_top5_area >= min_area_threshold:
            keep_top5 = True
            # 标记前5个连通域为保留
            for info in top5_info:
                info["is_kept"] = True
    
    # 7. 生成掩码：根据是否保留前5个决定掩码内容
    if keep_top5:
        # 提取前5个的标签，生成保留掩码
        top5_labels = [info["label"] for info in top5_info]
        mask = np.isin(labels, top5_labels).astype(np.uint8) * 255
    else:
        # 全置0
        mask = np.zeros_like(img)
    
    # 8. 应用掩码到原图像
    processed_img = cv2.bitwise_and(img, img, mask=mask)
    
    return processed_img, component_info


def main2():
    data_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\DataFromDifferentLocations\NinemeiDaba\CollectedDataByProgram\20251030_merged"
    save_path = data_path + "_detect_ridges_results_numba"
    os.makedirs(save_path, exist_ok=True)   

    file_list = os.listdir(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path, 0) # 0 imports a grayscale
        if img is None:
            raise(ValueError(f"Image didn\'t load. Check that '{f_abs_path}' exists."))
        
        t1 = time.time()
        # a, b = detect_ridges(img, sigma=3.0)
        a, b = detect_ridges_optimized(img, 1.0)
        t2 = time.time()
        print(f"{fname} took {t2-t1} seconds")
        # plot_images(img, a, b)
        # pass
        # a = np.uint8(a * 255)
        b = np.uint8(b * 255)
        b = cv2.bitwise_not(b)
        _, b = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)

        b = cv2.erode(b, np.ones((3, 3), np.uint8), iterations=1)
        b = cv2.dilate(b, np.ones((3, 3), np.uint8), iterations=1)

        # b = keep_top5_largest_components(b, k=5, connectivity=8)

        connectivity = 8
        min_area_threshold = 500  # 面积阈值：小于500则全置0

        # 3. 处理连通域
        b, comp_info = process_top5_components(
            b, 
            connectivity=connectivity, 
            min_area_threshold=min_area_threshold
        )

        # res = np.vstack((a, b))
        f_dst_path = save_path + "/{}.png".format(fname)
        cv2.imwrite(f_dst_path, b)



# 使用示例
if __name__ == "__main__":
    # main()
    main2()

    # # 生成测试图像 (1700x660)
    # test_image = np.random.randint(0, 255, (660, 1700), dtype=np.uint8)
    
    # # 性能测试
    # benchmark_methods(test_image)
    
    # # 验证结果一致性
    # print("\n验证结果一致性:")
    # original_maxima, original_minima = detect_ridges(test_image, sigma=3.0)
    # fast_maxima, fast_minima = fast_detect_ridges(test_image, sigma=3.0)
    # # fast_maxima, fast_minima = numba_detect_ridges(test_image, sigma=3.0)
    
    # diff_maxima = np.mean(np.abs(original_maxima - fast_maxima))
    # diff_minima = np.mean(np.abs(original_minima - fast_minima))
    
    # print(f"最大特征值平均差异: {diff_maxima:.6f}")
    # print(f"最小特征值平均差异: {diff_minima:.6f}")

    # test_performance()
    
    # # 可视化对比（可选）
    # img = cv2.imread(r"G:\Gosion\data\006.Belt_Torn_Det\data\DataFromDifferentLocations\NinemeiDaba\CollectedDataByProgram\20251030_merged\data1jia_20251027_20251027135020.jpg", 0)  # 替换为你的图像路径
    # if img is not None:
    #     max_orig, min_orig = detect_ridges_original(img, 1.0)
    #     max_opt, min_opt = detect_ridges_optimized(img, 1.0)
        
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(221), plt.imshow(max_orig, cmap='gray'), plt.title('原函数-最大值脊线')
    #     plt.subplot(222), plt.imshow(max_opt, cmap='gray'), plt.title('优化函数-最大值脊线')
    #     plt.subplot(223), plt.imshow(min_orig, cmap='gray'), plt.title('原函数-最小值脊线')
    #     plt.subplot(224), plt.imshow(min_opt, cmap='gray'), plt.title('优化函数-最小值脊线')
    #     plt.tight_layout()
    #     plt.show()


