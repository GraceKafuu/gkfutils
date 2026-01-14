import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
import onnxruntime as ort
import random
import requests
import time
import shutil
import threading


def corner_detect(img, blockSize=10, ksize=7, k=0.04):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)  # 转换为浮点型
    dst = cv2.cornerHarris(gray, blockSize, ksize, k)  # Harris 角点检测

    r, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)  # 二值化阈值处理
    dst = np.uint8(dst)  # 转换为整型

    return dst


def group_regions_by_distance(binary_image, max_distance=50, min_samples=1):
    """
    根据白色区域中心点之间的距离进行分组
    
    参数:
    - binary_image: 二值图像(白色区域为255，背景为0)
    - max_distance: 判断为同一组的最大距离阈值
    - min_samples: 每组最少包含的区域数量
    
    返回:
    - grouped_regions: 分组结果，每个元素是一组区域的轮廓列表
    - centers: 所有区域的中心点坐标
    - labels: 每个区域所属的组别标签
    """
    
    # 1. 查找所有白色区域的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return [], [], []
    
    # 2. 计算每个区域的中心点
    centers = []
    valid_contours = []
    
    for contour in contours:
        # 过滤掉太小的区域（可选）
        if cv2.contourArea(contour) < 10:  # 面积阈值
            continue
            
        # 计算矩并获取中心点
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append([cx, cy])
            valid_contours.append(contour)
    
    if len(centers) == 0:
        return [], [], []
    
    centers = np.array(centers)
    
    # 3. 基于距离进行聚类分组
    # 使用DBSCAN聚类算法，eps为距离阈值
    clustering = DBSCAN(eps=max_distance, min_samples=min_samples).fit(centers)
    labels = clustering.labels_
    
    # 4. 根据聚类结果分组轮廓
    grouped_regions = []
    n_groups = len(set(labels)) - (1 if -1 in labels else 0)
    
    for group_id in range(n_groups):
        group_contours = [valid_contours[i] for i in range(len(valid_contours)) if labels[i] == group_id]
        grouped_regions.append(group_contours)
    
    # 处理噪声点（label为-1的区域，各自单独成组）
    noise_indices = np.where(labels == -1)[0]
    for idx in noise_indices:
        grouped_regions.append([valid_contours[idx]])
    
    return grouped_regions, centers, labels


def visualize_grouping_result(binary_image, grouped_regions, centers, labels, group_num_thr=3):
    """可视化分组结果"""
    # 创建彩色图像用于可视化
    if len(binary_image.shape) == 2:
        result_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    else:
        result_img = binary_image.copy()
    
    # 定义颜色列表
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]

    group_boxes = []
    
    # 绘制每个组的轮廓和中心点
    for group_id, group_contours in enumerate(grouped_regions):
        color = colors[group_id % len(colors)]

        if len(group_contours) > group_num_thr:
            all_points = np.vstack([contour for contour in group_contours])
            group_box = cv2.boundingRect(all_points)
            cv2.rectangle(result_img, group_box, color, 2)
            group_boxes.append(group_box)
        
        
        for contour in group_contours:
            # 绘制轮廓
            cv2.drawContours(result_img, [contour], -1, color, 2)

            # 计算并绘制中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(result_img, (cx, cy), 5, color, -1)
                cv2.putText(result_img, f'G{group_id}', (cx+10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return result_img, group_boxes


def get_laser_area_image(img, mask, expand_pixels=25):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if expand_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, expand_pixels))
        mask = cv2.dilate(mask, kernel, iterations=1)
    laserGray = cv2.bitwise_and(image, image, mask=mask)
    return laserGray


def z_score(x, mean, std):
    return (x - mean) / std


def cal_column_min_max_y(img):
    imgsz = img.shape[:2]
    # 向量化计算
    # 为每列创建行索引矩阵
    row_indices = np.arange(imgsz[0]).reshape(-1, 1)
    
    # 找到每列非零元素的行索引
    nonzero_mask = img > 0
    
    # 初始化结果
    min_y_coords = np.full(imgsz[1], imgsz[0], dtype=int)
    max_y_coords = np.full(imgsz[1], -1, dtype=int)
    
    # 对每列应用计算
    for col in range(imgsz[1]):
        col_nonzero = nonzero_mask[:, col]
        if np.any(col_nonzero):
            col_indices = np.where(col_nonzero)[0]
            min_y_coords[col] = col_indices[0]  # 第一个非零元素的索引
            max_y_coords[col] = col_indices[-1] # 最后一个非零元素的索引
    
    return min_y_coords, max_y_coords


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


def detect_by_frames_diff_v5(imgPre, img, mask, vis_box_expand_pixels=25, laser_area_expand_pixels=25, zscore_thr=5, min_dis=10, max_dis=1000):

    # initHsvMeans, initAreas = init_hsvs_means[0], init_hsvs_means[1]
    # initHsvMeansExpand, initAreasExpand = init_hsvs_means[2], init_hsvs_means[3]

    img_vis = img.copy()
    imgsz = img.shape[:2]

    gray_img_expand = get_laser_area_image(img, mask, laser_area_expand_pixels)

    diff = cv2.absdiff(imgPre, gray_img_expand)
    column_sums = np.sum(diff, axis=0)

    # nonzero_mask = diff != 0
    # nonzero_counts = np.count_nonzero(nonzero_mask, axis=0)
    # nonzero_counts[nonzero_counts == 0] = 1
    # column_means = column_sums / nonzero_counts
    # column_means = np.where(column_means < pixel_thr, 0, column_means)

    # # print("column_means: ", column_means)

    # column_means_mean = np.mean(column_means)
    # column_means_std = np.std(column_means)
    # zscore = z_score(column_means, column_means_mean, column_means_std)


    column_sums_mean = np.mean(column_sums)
    column_sums_std = np.std(column_sums)
    zscore = z_score(column_sums, column_sums_mean, column_sums_std)

    x_coords1 = np.where(abs(zscore) > zscore_thr)
    groups_zscore = find_nonzero_groups(x_coords1[0])

    # x_coords2 = np.where(column_sums > 500)
    # groups = find_nonzero_groups(x_coords2[0])

    min_y_coords, max_y_coords = cal_column_min_max_y(diff)
    
    wides = []
    boxes = []
    for i, (start, end) in enumerate(groups_zscore):
        start = x_coords1[0][start]
        end = x_coords1[0][end]
        
        # assert end >= start, "end < start!"
        # print(f"组 {i+1}: 起始={start}, 结束={end}, 值={column_means[start:end+1]}")

        fakeDis = end - start

        if fakeDis >= imgsz[1]: continue
        if fakeDis < min_dis: continue
        if fakeDis > max_dis: continue

        wides.append(fakeDis)

        x1_vis = start - vis_box_expand_pixels
        x2_vis = end + vis_box_expand_pixels
        y1_vis = round(np.mean(min_y_coords[start:end])) - vis_box_expand_pixels * 2
        y2_vis = round(np.mean(max_y_coords[start:end])) + vis_box_expand_pixels * 2

        if x1_vis < 0: x1_vis = 0
        if y1_vis < 0: y1_vis = 0
        if x2_vis > imgsz[1]: x2_vis = imgsz[1]
        if y2_vis > imgsz[0]: y2_vis = imgsz[0]

        visBox = [x1_vis, y1_vis, x2_vis - x1_vis, y2_vis - y1_vis]
        # boxes.append([start, 0, fakeDis, imgsz[0]])
        boxes.append(visBox)

        cv2.rectangle(img_vis, (x1_vis, y1_vis), (x2_vis, y2_vis), (255, 0, 255), 5)

    return img_vis, boxes, wides


# ------------------------------------ 2025.09.17 WJH. ------------------------------------
class LaserDetect:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
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


def check_fitted_model(img, model):
    imgsz = img.shape[:2]

    p_left_x = 0  # 左边缘
    p_middle_x = imgsz[1] // 2  # 中间点
    p_right_x = imgsz[1] - 1  # 右边缘
    p_1_8_x = imgsz[1] // 8  # 1/8点位
    p_7_8_x = imgsz[1] * 7 // 8  # 7/8点位

    p_left_y_pred = np.polyval(model, p_left_x)
    p_middle_y_pred = np.polyval(model, p_middle_x)
    p_right_y_pred = np.polyval(model, p_right_x)
    p_1_8_y_pred = np.polyval(model, p_1_8_x)
    p_7_8_y_pred = np.polyval(model, p_7_8_x)

    left_res = p_left_y_pred > 0 and p_left_y_pred < imgsz[0]
    middle_res = (p_middle_y_pred > 0 and p_middle_y_pred < imgsz[0] * 0.60) or (p_middle_y_pred > imgsz[0] * 0.40 and p_middle_y_pred < imgsz[0])
    right_res = p_right_y_pred > 0 and p_right_y_pred < imgsz[0]
    eighth_res = abs(p_1_8_y_pred - p_7_8_y_pred) < imgsz[0] / 5

    if left_res and middle_res and right_res and eighth_res:
        return model
    else:
        return None
    

def fit_plot(img, model, color=(255, 0, 255)):
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
        cv2.circle(vis, (int(x[i]), int(y[i])), 2, color, -1)

    return vis


def createMaskByModel(img, laser_model, fit_deg=4, expand_pixels=25):
    vis = np.zeros(shape=img.shape[:2], dtype=np.uint8)
    out_points = laser_model.detect(img, vis=False)
    fit_model = np.polyfit(out_points[:, 0], out_points[:, 1], deg=fit_deg)
    mask = fit_plot(vis, fit_model, deg=fit_deg)

    if expand_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, expand_pixels))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask



def start_c_v3(imgPre, img0, mask, vis_box_expand_pixels=25, laser_area_expand_pixels=25, zscore_thr=5, min_dis=10, max_dis=500, p_crnerDetectFlag=True):
    imgPre = cv2.cvtColor(imgPre, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)

    if p_crnerDetectFlag:
        corner_res = corner_detect(img, blockSize=10, ksize=7, k=0.04)
        grouped_regions, centers, labels = group_regions_by_distance(corner_res, max_distance=50, min_samples=1)
        result_img, group_boxes = visualize_grouping_result(img, grouped_regions, centers, labels, group_num_thr=3)

        img_vis, boxes, wides = detect_by_frames_diff_v5(imgPre, img, vis_box_expand_pixels, laser_area_expand_pixels, zscore_thr, min_dis, max_dis)
        img_vis_new = cv2.addWeighted(result_img, 0.5, img_vis, 0.5, 0)

        for gb in group_boxes:
            wides.append(gb[2])

        return img_vis_new, wides

    else:
        img_vis, boxes, wides = detect_by_frames_diff_v5(imgPre, img, vis_box_expand_pixels, laser_area_expand_pixels, zscore_thr, min_dis, max_dis)

        return img_vis, wides

