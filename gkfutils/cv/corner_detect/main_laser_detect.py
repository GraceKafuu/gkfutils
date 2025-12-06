import onnxruntime
import numpy as np
import cv2
from PIL import Image
import time
import os
import random
import shutil


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
    


def generate_polynomial(x, model, degree):
    """
    根据阶数生成多项式预测表达式
    
    参数:
        x: 输入的自变量数组
        model: 多项式系数数组，长度应为 degree + 1
        degree: 多项式的阶数(如2表示二次多项式)
    
    返回:
        y: 多项式预测结果
    """
    # 检查系数数量是否与阶数匹配
    if len(model) != degree + 1:
        raise ValueError(f"系数数量应为 {degree + 1}，但实际为 {len(model)}")
    
    # 初始化预测结果
    y = 0.0
    
    # 循环生成每一项并累加
    for i in range(degree + 1):
        # 对于degree阶多项式，第i项为 model[i] * x_all^(degree - i)
        exponent = degree - i
        y += model[i] * (x ** exponent)
    
    return y


def ransac_fit(points, deg=4, select_num=100, num_iterations=500, threshold=15.0):
    """
    # # 生成测试数据
    # np.random.seed(42)
    # x = np.linspace(-10, 10, 100)
    # y_true = 0.5 * x**2 + 2 * x + 3
    # y = y_true + np.random.normal(scale=3.0, size=x.shape)  # 添加噪声
    # # 添加外点
    # outlier_indices = np.random.choice(len(x), size=20, replace=False)
    # y[outlier_indices] += np.random.normal(scale=30, size=20)
    # points = np.column_stack((x, y))

    # # 运行RANSAC
    # model = ransac_parabola(points, num_iterations=1000, threshold=3.0)
    # if model is not None:
    #     a, b, c = model
    #     print(f"拟合的抛物线方程: y = {a:.3f}x² + {b:.3f}x + {c:.3f}")
    # else:
    #     print("拟合失败")
    """
    best_model = None
    best_inliers = []
    max_inliers = 0

    n_points = points.shape[0]
    if n_points < select_num:
        return None  # 无法拟合

    for _ in range(num_iterations):
        # 随机选择select_num个点
        sample_indices = random.sample(range(n_points), select_num)
        sample_points = points[sample_indices]
        sx = sample_points[:, 0]
        sy = sample_points[:, 1]

        model = np.polyfit(sx, sy, deg)

        # 计算所有点的垂直距离
        x_all = points[:, 0]
        y_all = points[:, 1]

        # y_pred = model[0] * x_all**2 + model[1] * x_all + model[2]
        # y_pred = model[0] * x_all**3 + model[1] * x_all**2 + model[2] * x_all + model[3]
        # y_pred = model[0] * x_all**4 + model[1] * x_all**3 + model[2] * x_all**2 + model[3] * x_all + model[4]

        y_pred = generate_polynomial(x_all, model, deg)

        distances = np.abs(y_all - y_pred)

        # 统计内点
        inliers = np.where(distances < threshold)[0]
        n_inliers = inliers.size

        # 更新最佳模型
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_inliers = inliers
            best_model = model

    # 使用所有内点重新拟合
    if best_model is not None and len(best_inliers) >= select_num:
        x_inliers = points[best_inliers, 0]
        y_inliers = points[best_inliers, 1]
        model = np.polyfit(x_inliers, y_inliers, deg)
        best_model = model
    else:
        return None  # 无有效模型

    return best_model


def fit_curve(img, deg=4, lower=None, upper=None, laser_color="green", FLAG_RANSAC_FIT=True, select_num=1000, num_iterations=1000, threshold=50):
    mask, result, binary, binary_otsu = extract_specific_color(img, lower=lower, upper=upper, color=laser_color)

    # cv2.imshow("binary", binary)
    # cv2.imshow("binary_otsu", binary_otsu)
    # cv2.waitKey(0)

    points = np.where(binary == 255)
    points = np.hstack((points[1].reshape(-1, 1), points[0].reshape(-1, 1)))

    fit_model = None
    if FLAG_RANSAC_FIT:
        fit_model = ransac_fit(points, deg, select_num, num_iterations, threshold)
        # if fit_model is not None:
        #     a, b, c = fit_model
        #     print(f"拟合的抛物线方程: y = {a:.12f}x² + {b:.12f}x + {c:.12f}")
        # else:
        #     print("拟合失败")
    else:
        fit_model = np.polyfit(points[:, 0], points[:, 1], deg)

    # fit_Q.put(fit_model)

    return fit_model


def fit_plot(img, model, deg=4, color=(255, 0, 255)):
    """
    a, b, c = model
    print(f"拟合的抛物线方程: y = {a:.3f}x² + {b:.3f}x + {c:.3f}")
    """
    vis = img.copy()
    imgsz = img.shape[:2]
    x = np.linspace(0, imgsz[1], imgsz[1] - 1)
    # y = model[0] * x**2 + model[1] * x + model[2]
    # y = model[0] * x**3 + model[1] * x**2 + model[2] * x + model[3]
    # y = model[0] * x**4 + model[1] * x**3 + model[2] * x**2 + model[3] * x + model[4]
    y = generate_polynomial(x, model, deg)

    for i in range(len(x)):
        cv2.circle(vis, (int(x[i]), int(y[i])), 2, color, -1)

    return vis


def main():
    # img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\DataFromDifferentLocations\NinemeiDaba\CollectedDataByProgram\20251030_merged"
    img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\Jingye\data\20251201"
    # img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\DataFromDifferentLocations\NinemeiDaba\CollectedDataByProgram\compare_test"
    save_path = img_path + "_detect_results_20251206"
    os.makedirs(save_path, exist_ok=True)

    model_path = r"G:\Gosion\code\Ultra-Fast-Lane-Detection-v2-master\weights\20251114_OK2\best.onnx"
    laserDetect = LaserDetect(model_path)

    file_list = os.listdir(img_path)
    for f in file_list:
        img_name = os.path.splitext(f)[0]
        f_abs_path = os.path.join(img_path, f)
        img = cv2.imread(f_abs_path)
        # img_cp = img.copy()

        points = laserDetect.detect(img, vis=False)

        fit_model = np.polyfit(points[:, 0], points[:, 1], deg=4)
        print("img_name: {} fit_model: {}".format(img_name, fit_model))
        plot_out = fit_plot(img, fit_model, deg=4)
        # cv2.imshow("plot_out", plot_out)
        f_dst_path = os.path.join(save_path, f)
        cv2.imwrite(f_dst_path, plot_out)


def check_fitted_model(img, model):
    """ 检查拟合的曲线方程是否有问题 """
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
    

def main2():
    data_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\LaserDetLikeLaneMarksDet\fake_data_v2"
    img_path = data_path + "\\images"
    mask_path = data_path + "\\masks"
    save_path = data_path + "_selected"
    img_save_path = save_path + "\\images"
    mask_save_path = save_path + "\\masks"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)

    model_path = r"G:\Gosion\code\Ultra-Fast-Lane-Detection-v2-master\weights\best.onnx"
    laserDetect = LaserDetect(model_path)

    file_list = os.listdir(mask_path)
    for f in file_list:
        img_name = os.path.splitext(f)[0]
        img_abs_path = os.path.join(img_path, "{}.jpg".format(img_name))
        mask_abs_path = os.path.join(mask_path, f)
        img = cv2.imread(mask_abs_path)
        imgsz = img.shape[:2]
        # img_cp = img.copy()

        b = img[:, :, 0]
        mask_brightness = np.mean(b)

        # points = laserDetect.detect(img, vis=False)

        points = np.where(b == 255)
        if len(points[1]) == 0:
            continue
        points = np.hstack((points[1].reshape(-1, 1), points[0].reshape(-1, 1)))

        bbx = cv2.boundingRect(b)
        res = False
        if bbx[2] < imgsz[1]:
            res = True

        fit_model = np.polyfit(points[:, 0], points[:, 1], deg=4)
        plot_out = fit_plot(img, fit_model, deg=4)
        # cv2.imshow("plot_out", plot_out)

        fit_model_check = check_fitted_model(img, fit_model)
        if fit_model_check is None and mask_brightness > 0 and res:
            img_dst_path = os.path.join(img_save_path, "{}.jpg".format(img_name))
            mask_dst_path = os.path.join(mask_save_path, f)
            shutil.move(img_abs_path, img_dst_path)
            shutil.move(mask_abs_path, mask_dst_path)


        # cv2.imwrite(f_dst_path, plot_out)


if __name__ == '__main__':
    main()
    # main2()

