import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import time
import os
import random
from sklearn.cluster import DBSCAN


def pil2cv(image):
    assert isinstance(image, Image.Image), f'Input image type is not PIL.image and is {type(image)}!'
    if len(image.split()) == 1:
        return np.asarray(image)
    elif len(image.split()) == 3:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    elif len(image.split()) == 4:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGR)
    else:
        return None
        

def preprocess(input, imgsz=(64, 256)):
    """
    Preprocesses the input image before performing inference.

    Returns:
        image_data: Preprocessed image data ready for inference.
    """
    if isinstance(input, str):
        img = cv2.imread(input)
    elif isinstance(input, Image.Image):
        img = pil2cv(input)
    else:
        assert isinstance(input, np.ndarray), f'input is not np.ndarray and is {type(input)}!'
        img = input

    # Get the height and width of the input image
    img_height, img_width = img.shape[:2]

    # Convert the image color space from BGR to RGB
    # img = self.img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to match the input shape
    img = cv2.resize(img, imgsz[::-1])
    # img = pre_transform(img)

    # Normalize the image data by dividing it by 255.0
    image_data = np.array(img) / 255.0

    # Transpose the image to have the channel dimension as the first dimension
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

    # Expand the dimensions of the image data to match the expected input shape
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    # Return the preprocessed image data
    return image_data


def inference(img):
    # img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\seg\v2_mini\val\images\1_output_000000005.jpg"

    # fname = os.path.basename(img_path)
    # img_name = os.path.splitext(fname)[0]

    # img = cv2.imread(img_path)
    img_orig = img.copy()

    imgsz = img.shape
    rh = 64 / imgsz[0]
    rw = 256 / imgsz[1]

    w_gird = np.linspace(0, imgsz[1], 256)

    img = preprocess(img, imgsz=(64, 256))

    model_path = "weights/best.onnx"
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name

    t1 = time.time()
    ort_outs = ort_session.run(None, {input_name: img})
    t2 = time.time()
    print("inference time: ", t2 - t1)


    # print(ort_outs)
    # print(ort_outs[0].shape)

    # output_rows = ort_outs[0][0, :256]
    output_rows = ort_outs[0][0, :]
    # print(output_rows)

    output_rows_orig = output_rows / rh
    # print(output_rows_orig)

    for i in range(256):
        pi = (round(w_gird[i]), round(output_rows_orig[i]))
        cv2.circle(img_orig, pi, 5, (255, 0, 255), -1)

    # cv2.imshow('img_orig', img_orig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite('test.png', img_orig)
    return img_orig


def main0():
    # data_path = r'G:\Gosion\data\006.Belt_Torn_Det\data\seg\v2_mini\val\images'
    data_path = r'G:\Gosion\data\006.Belt_Torn_Det\data\videos\lainjiangsilie\Video_2025_09_26_103048_1_frames\Video_2025_09_26_103048_1'
    file_list = sorted(os.listdir(data_path))

    # save_path = r'G:\Gosion\data\006.Belt_Torn_Det\data\seg\v2_mini\val\images_out'
    save_path = r'G:\Gosion\data\006.Belt_Torn_Det\data\videos\lainjiangsilie\Video_2025_09_26_103048_1_frames\Video_2025_09_26_103048_1_out'
    os.makedirs(save_path, exist_ok=True)

    for f in file_list:
        f_abs_path = os.path.join(data_path, f)
        fname = os.path.splitext(f)[0]
        img = cv2.imread(f_abs_path)
        imgsz = img.shape
        if len(imgsz) == 2: 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = inference(img)
        cv2.imwrite(os.path.join(save_path, fname + 'out_20250928.png'), out)


def corner_detect(img, blockSize=10, ksize=7, k=0.04):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)  # 转换为浮点型
    dst = cv2.cornerHarris(gray, blockSize, ksize, k)  # Harris 角点检测

    r, dst = cv2.threshold(dst, 0.001 * dst.max(), 255, 0)  # 二值化阈值处理
    dst = np.uint8(dst)  # 转换为整型

    return dst


def group_regions_by_distance(binary_image, fit_model, max_distance=50, min_samples=1, disWithFitModelThresh=15):
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

            # 计算与激光曲线的距离
            y_pred = np.polyval(fit_model, cx)
            dis = abs(cy - y_pred)
            if dis > disWithFitModelThresh: continue

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

        if len(group_contours) < group_num_thr:
            continue

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
                cv2.putText(result_img, f'G{group_id}', (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
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


def detect_by_frames_diff_v5(imgPre, img, mask, vis_box_expand_pixels=25, laser_area_expand_pixels=25, zscore_thr1=-5, zscore_thr2=5, min_dis=10, max_dis=1000):

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

    # x_coords1 = np.where(abs(zscore) > zscore_thr)
    x_coords1 = np.where((zscore < zscore_thr1) | (zscore > zscore_thr2))
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



def detect_by_frames_diff_v5_patent(img, diff, mask, vis_box_expand_pixels=25, laser_area_expand_pixels=25, zscore_thr1=-5, zscore_thr2=5, min_dis=10, max_dis=1000):

    # initHsvMeans, initAreas = init_hsvs_means[0], init_hsvs_means[1]
    # initHsvMeansExpand, initAreasExpand = init_hsvs_means[2], init_hsvs_means[3]

    img_vis = img.copy()
    imgsz = img.shape[:2]

    # gray_img_expand = get_laser_area_image(img, mask, laser_area_expand_pixels)

    # diff = cv2.absdiff(imgPre, gray_img_expand)
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

    # x_coords1 = np.where(abs(zscore) > zscore_thr)
    x_coords1 = np.where((zscore < zscore_thr1) | (zscore > zscore_thr2))
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

        cv2.rectangle(img_vis, (x1_vis, y1_vis), (x2_vis, y2_vis), (0, 255, 255), 5)

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
    

def fit_plot(img, model, color=(255, 255, 255)):
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
    mask = fit_plot(vis, fit_model)

    if expand_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, expand_pixels))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def main():
    deg = 4

    # img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\seg\v2_mini\val\images\1_output_000000005.jpg"
    # img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\seg\v2_mini\val\images\Video_2025_03_08_111948_1_output_000005805.jpg"
    img_path = r"G:\Gosion\data\006.Belt_Torn_Det\resources\patent\Video_2025_03_01_165140_1_output_000007529.jpg"

    img2_path = r"G:\Gosion\data\006.Belt_Torn_Det\resources\patent\Video_2025_03_01_165140_1_output_000009359.jpg"
    img2 = cv2.imread(img2_path)
    img2_cp = img2.copy()

    save_path = r"G:\Gosion\data\006.Belt_Torn_Det\resources\patent\output"
    os.makedirs(save_path, exist_ok=True)

    fname = os.path.basename(img_path)
    img_name = os.path.splitext(fname)[0]

    img = cv2.imread(img_path)
    img_orig = img.copy()
    img_orig2 = img.copy()

    imgsz = img.shape
    rh = 64 / imgsz[0]
    rw = 256 / imgsz[1]

    w_gird = np.linspace(0, imgsz[1], 256)

    img = preprocess(img, imgsz=(64, 256))

    model_path = r"G:\Gosion\code\Ultra-Fast-Lane-Detection-v2-master\weights\best.onnx"
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name

    t1 = time.time()
    ort_outs = ort_session.run(None, {input_name: img})
    t2 = time.time()
    print("inference time: ", t2 - t1)


    laser_ort = LaserDetect(model_path=model_path)

    # print(ort_outs)
    # print(ort_outs[0].shape)

    # output_rows = ort_outs[0][0, :256]
    output_rows = ort_outs[0][0, :]
    # print(output_rows)

    output_rows_orig = output_rows / rh
    # print(output_rows_orig)

    points = []

    for i in range(256):
        pi = (round(w_gird[i]), round(output_rows_orig[i]))
        points.append(pi)
        cv2.circle(img_orig, pi, 5, (255, 0, 255), -1)

    cv2.imwrite(save_path + "/laser_ort.png", img_orig)

    points = np.array(points)
    model = np.polyfit(points[:, 0], points[:, 1], deg)

    polyval_out = fit_plot(img_orig, model, color=(255, 255, 0))

    cv2.imwrite(save_path + "/polyval_out.png", polyval_out)


    mask = createMaskByModel(img_orig, laser_ort, fit_deg=4, expand_pixels=25)
    cv2.imwrite(save_path + "/mask.png", mask)

    laser_area_image = get_laser_area_image(img_orig2, mask, expand_pixels=25)
    cv2.imwrite(save_path + "/laser_area_image.png", laser_area_image)


    # img_vis, _, _ = detect_by_frames_diff_v5_patent(img_orig2, mask, vis_box_expand_pixels=25, laser_area_expand_pixels=25, zscore_thr1=-5, zscore_thr2=5, min_dis=10, max_dis=1000)
    # cv2.imwrite(r"G:\Gosion\data\006.Belt_Torn_Det\data\seg\v2_mini\val\images_out\img_vis.png", img_vis)

    corner_res = corner_detect(img2_cp, 10, 7, 0.04)
    grouped_regions, centers, labels = group_regions_by_distance(corner_res, model, 25, 1, 15)
    result_img, group_boxes = visualize_grouping_result(img2_cp, grouped_regions, centers, labels, 1)
    cv2.imwrite(save_path + "/cornerOut.png", result_img)

    img2_cp_laser_area = get_laser_area_image(img2_cp, mask, 25)



    diff = cv2.absdiff(laser_area_image, img2_cp_laser_area)
    cv2.imwrite(save_path + "/diff.png", diff)


    img_vis, _, _ = detect_by_frames_diff_v5_patent(result_img, diff, mask, vis_box_expand_pixels=25, laser_area_expand_pixels=25, zscore_thr1=-5, zscore_thr2=5, min_dis=10, max_dis=1000)
    cv2.imwrite(save_path + "/diff_out.png", img_vis)

# def main2():
#     img1_path = r""




if __name__ == '__main__':
    main()




