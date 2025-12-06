import onnxruntime
import numpy as np
import cv2
from PIL import Image
import time
import os
import random
import shutil
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')  # 屏蔽numba首次编译警告
from typing import Union, List, Dict, Tuple

class LaserDetect:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = self.session.get_inputs()[0].shape
        self.ouput_size = self.session.get_outputs()[0].shape

        self.imgsz = self.input_size[2:]

    def pil2cv(self, image):
        assert isinstance(image, Image.Image), f'Input image type is not PIL.image and is {type(image)}!'
        if len(image.split()) == 1:
            return np.asarray(image)
        elif len(image.split()) == 3:
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        elif len(image.split()) == 4:
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGR)
        else:
            return None
    
    def preprocess(self, input):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        if isinstance(input, str):
            img = cv2.imread(input)
        elif isinstance(input, Image.Image):
            img = self.pil2cv(input)
        else:
            assert isinstance(input, np.ndarray), f'input is not np.ndarray and is {type(input)}!'
            img = input

        # Get the height and width of the input image
        self.origsize = img.shape[:2]

        # Convert the image color space from BGR to RGB
        # img = self.img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, self.imgsz[::-1])
        # img = pre_transform(img)

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data


    def inference(self, input):
        ort_outs = self.session.run(None, {self.input_name: input})
        return ort_outs
    
    def postprocess(self, ort_outs):
        rh = 64 / self.origsize[0]
        rw = 256 / self.origsize[1]

        w_gird = np.linspace(0, self.imgsz[1], 256)

        output_rows = ort_outs[0][0, :]
        # print(output_rows)

        output_rows_orig = output_rows / rh
        output_cols_orig = w_gird / rw
        # print(output_rows_orig)

        out = []

        for i in range(256):
            pi = (round(output_cols_orig[i]), round(output_rows_orig[i]))
            # cv2.circle(img_orig, pi, 5, (255, 0, 255), -1)
            out.append(pi)

        return out
    
    def detect(self, img, vis=False):
        img_np = self.preprocess(img)
        ort_outs = self.inference(img_np)
        out = self.postprocess(ort_outs)

        if vis:
            img_vis = self.visualize(img, out)
            cv2.imshow("img_vis", img_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return np.asarray(out)
    
    def visualize(self, img, out):
        for i in range(len(out)):
            cv2.circle(img, out[i], 5, (255, 0, 255), -1)

        return img
    

def fit_plot(img, model, r=4, color=(255, 0, 255)):
    """
    a, b, c = model
    print(f"拟合的抛物线方程: y = {a:.3f}x² + {b:.3f}x + {c:.3f}")
    """
    if model is None: return img

    vis = img.copy()
    imgsz = img.shape[:2]
    x = np.linspace(0, imgsz[1], imgsz[1] - 1)
    y = np.polyval(model, x)

    for i in range(len(x)):
        cv2.circle(vis, (int(x[i]), int(y[i])), r, color, -1)

    return vis


def createMaskByModel_v2(img, fit_model, fit_deg=4, expand_pixels=25):
    vis = np.zeros(shape=img.shape[:2], dtype=np.uint8)
    if fit_model is None:
        return vis

    mask = fit_plot(vis, fit_model)

    if expand_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, expand_pixels))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


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


def process_largest_component(
    img: np.ndarray,
    connectivity: int = 8,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    保留连通域分析中面积最大的前景连通域（其余置0）；
    若该最大连通域的宽度<原图宽度1/4 且 高度<原图高度1/4，则全部置0
    :param img: 输入二值图像（uint8类型，0为背景，255为前景）
    :param connectivity: 连通域连接方式，4或8（默认8）
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
    # 1. 基础校验：确保输入是二值图（非必须，但符合函数设计初衷）
    if not np.all(np.isin(img, [0, 255])):
        raise ValueError("输入图像必须是二值图像（仅包含0和255）")
    
    # 获取原图宽高（注意：numpy数组shape是 (高度, 宽度)）
    orig_height, orig_width = img.shape[:2]
    # 计算宽高阈值（原图对应维度的1/4）
    width_threshold = orig_width / 4
    height_threshold = orig_height / 4

    # 2. 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=connectivity
    )
    
    # 3. 初始化连通域信息列表（包含背景）
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
            "bbox_xyxy": [x1, y1, x2, y2],            # xyxy格式
            "centroid": (float(cx), float(cy)),
            "is_kept": False  # 初始标记为未保留
        })
    
    # 4. 处理边界情况：无前景连通域（仅背景）
    if num_labels <= 1:
        return np.zeros_like(img), component_info
    
    # 5. 提取前景连通域（排除背景label=0）并找到面积最大的
    foreground_info = [info for info in component_info if info["label"] != 0]
    # 按面积降序排序，取第一个（面积最大的前景连通域）
    largest_comp_info = sorted(foreground_info, key=lambda x: x["area"], reverse=True)[0]
    
    # 6. 判断是否保留最大连通域：宽/高同时小于原图1/4则舍弃
    keep_largest = False
    largest_comp_width = largest_comp_info["width"]
    largest_comp_height = largest_comp_info["height"]
    
    if not (largest_comp_width < width_threshold and largest_comp_height < height_threshold):
        keep_largest = True
        largest_comp_info["is_kept"] = True  # 标记为保留

    # 7. 生成掩码并应用到原图像
    if keep_largest:
        # 仅保留面积最大的连通域
        largest_label = largest_comp_info["label"]
        mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        # 全部置0
        mask = np.zeros_like(img)
    
    processed_img = cv2.bitwise_and(img, img, mask=mask)
    
    return processed_img, component_info



def extract_green_mask(
    img_input: Union[str, np.ndarray],
    top_extend: int = 0,    # Mask向上扩展的像素数
    bottom_extend: int = 0, # Mask向下扩展的像素数
    green_hsv_low: Tuple[int, int, int] = (35, 40, 40),  # 绿色HSV下限（可调整）
    green_hsv_high: Tuple[int, int, int] = (77, 255, 255),# 绿色HSV上限（可调整）
    save_path: str = None   # Mask保存路径（None则不保存）
) -> np.ndarray:
    """
    提取图像中的绿色区域生成二值Mask，并对Mask进行上下像素扩展
    :param img_input: 输入图像（路径字符串 或 cv2读取的numpy数组）
    :param top_extend: Mask向上扩展像素数（≥0）
    :param bottom_extend: Mask向下扩展像素数（≥0）
    :param green_hsv_low: 绿色HSV范围下限（H:0-179, S/V:0-255）
    :param green_hsv_high: 绿色HSV范围上限
    :param save_path: Mask保存路径（如"green_mask.png"）
    :return: 处理后的绿色Mask图（uint8，255=绿色区域，0=背景）
    """
    # 1. 读取/校验输入图像
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise ValueError(f"无法读取图像：{img_input}")
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy()
    else:
        raise TypeError("img_input必须是图像路径字符串或cv2读取的numpy数组")
    
    # 2. 图像预处理：转HSV + 去噪
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 高斯模糊去除小噪声（可选，根据需求调整核大小）
    hsv_img = cv2.GaussianBlur(hsv_img, (5, 5), 0)
    
    # 3. 提取绿色区域的初始Mask
    mask = cv2.inRange(hsv_img, np.array(green_hsv_low), np.array(green_hsv_high))
    
    # 4. 形态学操作优化Mask（去除小噪点、填充孔洞）
    # 结构元素（可根据图像分辨率调整大小）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 开运算：先腐蚀后膨胀，去除小亮点噪声
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 闭运算：先膨胀后腐蚀，填充绿色区域内部小孔洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 5. 对Mask进行上下扩展
    h, w = mask.shape
    if top_extend > 0 or bottom_extend > 0:
        # 找到所有非0像素的行范围
        non_zero_rows = np.where(mask.sum(axis=1) > 0)[0]
        if len(non_zero_rows) > 0:
            min_row = max(0, non_zero_rows[0] - top_extend)    # 向上扩展，不越界
            max_row = min(h-1, non_zero_rows[-1] + bottom_extend)  # 向下扩展，不越界
            # 扩展区域置为255
            mask[min_row:max_row+1, :] = 255
    
    # 6. 保存Mask（可选）
    if save_path is not None:
        cv2.imwrite(save_path, mask)
        print(f"绿色Mask已保存至：{save_path}")
    
    return mask


def main():
    data_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\cls3\v3\train\Random_Selected\0_random_selected_1000"
    save_path = data_path + "_detect_ridges_results_numba"
    os.makedirs(save_path, exist_ok=True)   

    model_path = r"G:\Gosion\code\Ultra-Fast-Lane-Detection-v2-master\weights\20251114_OK2\best.onnx"
    laserDetect = LaserDetect(model_path)

    file_list = os.listdir(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = os.path.join(data_path, f)
        img = cv2.imread(f_abs_path)
        # img_cp = img.copy()

        points = laserDetect.detect(img, vis=False)

        fit_model = np.polyfit(points[:, 0], points[:, 1], deg=4)
        print("fname: {} fit_model: {}".format(fname, fit_model))
        plot_out = fit_plot(img, fit_model, r=4)
        # cv2.imshow("plot_out", plot_out)
        # f_dst_path = os.path.join(save_path, f)
        # cv2.imwrite(f_dst_path, plot_out)

        mask = createMaskByModel_v2(img, fit_model, fit_deg=4, expand_pixels=200)

        # green_mask = extract_green_mask(img, top_extend=0, bottom_extend=0)

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        # thresh, comp_info = process_largest_component(
        #     thresh, 
        #     connectivity=8
        # )

        # mask_test = cv2.bitwise_and(green_mask, thresh)
        

        # cv2.imshow("img", img)
        # cv2.imshow("mask", mask)
        # cv2.imshow("green_mask", green_mask)
        # cv2.imshow("thresh", thresh)
        # # cv2.imshow("mask_test", mask_test)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        laser_by_mask = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imwrite(save_path + "/{}_laser_by_mask.png".format(fname), laser_by_mask)
        
        mask2 = cv2.bitwise_not(mask)
        laser_by_mask_gray = cv2.cvtColor(laser_by_mask, cv2.COLOR_BGR2GRAY)

        t1 = time.time()
        # a, b = detect_ridges(img, sigma=3.0)
        a, b = detect_ridges_optimized(laser_by_mask_gray, 1.0)
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

        b = cv2.bitwise_not(b, b, mask=mask2)
        # b = b - mask2 * 255
        
        # b = keep_top5_largest_components(b, k=5, connectivity=8)

        connectivity = 8
        min_area_threshold = 500  # 面积阈值：小于500则全置0

        # 3. 处理连通域
        # b, comp_info = process_top5_components(
        #     b, 
        #     connectivity=connectivity, 
        #     min_area_threshold=min_area_threshold
        # )

        b, comp_info = process_largest_component(
            b, 
            connectivity=connectivity
        )

        kernel_size = 5  # 可改为3、7、9等，根据空洞尺寸调整
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # ---------------------- 3. 执行闭运算 ----------------------
        # cv2.MORPH_CLOSE：闭运算（先膨胀后腐蚀）
        closed_img = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)

        # res = np.vstack((a, b))
        f_dst_path = save_path + "/{}.png".format(fname)
        cv2.imwrite(f_dst_path, closed_img)



if __name__ == "__main__":
    main()
