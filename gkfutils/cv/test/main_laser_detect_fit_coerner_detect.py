import onnxruntime
import numpy as np
import cv2
from PIL import Image
import time
import os
import random



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
        

def extract_specific_color(img, lower=None, upper=None, color="green"):
    """
    提取图中指定颜色区域
    :param img: 输入图像
    :param lower: 颜色下限
    :param upper: 颜色上限
    :param color: 颜色名称
    :return: mask, result, binary, binary_otsu
    
    https://news.sohu.com/a/569474695_120265289
    https://blog.csdn.net/yuan2019035055/article/details/140495066
    """
    assert len(img.shape) == 3, "len(img.shape) != 3"

    if color == "black":
        lower = (0, 0, 0)
        upper = (180, 255, 46)
    elif color == "gray":
        lower = (0, 0, 46)
        upper = (180, 43, 220)
    elif color == "white":
        lower = (0, 0, 221)
        upper = (180, 30, 255)
    elif color == "red":
        lower0 = (0, 43, 46)
        upper0 = (10, 255, 255)
        lower1 = (156, 43, 46)
        upper1 = (180, 255, 255)
    elif color == "orange":
        lower = (11, 43, 46)
        upper = (25, 255, 255)
    elif color == "yellow":
        lower = (26, 43, 46)
        upper = (34, 255, 255)
    elif color == "green":
        lower = (35, 43, 46)
        upper = (77, 255, 255)
    elif color == "cyan":
        lower = (78, 43, 46)
        upper = (99, 255, 255)
    elif color == "blue":
        lower = (100, 43, 46)
        upper = (124, 255, 255)
    elif color == "purple":
        lower = (125, 43, 46)
        upper = (155, 255, 255)
    else:
        assert lower is not None and upper is not None and color not in ['black', 'gray', 'white', 'red', 'orange', 'yellow','green', 'cyan', 'blue', 'purple'], "Please choose color \
        from ['black', 'gray', 'white', 'red', 'orange', 'yellow','green', 'cyan', 'blue', 'purple']. If not in the list, please input the 'lower' and 'upper' HSV value of the color."
        
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color == "red":
        mask0 = cv2.inRange(hsv_img, lower0, upper0)
        mask1 = cv2.inRange(hsv_img, lower1, upper1)
        mask = cv2.bitwise_or(mask0, mask1)
    else:
        mask = cv2.inRange(hsv_img, lower, upper)

    # 可视化结果（可选）
    # 将掩码应用到原图上，显示提取的颜色区域
    result = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask, result, binary, binary_otsu


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


def corner_detect_v5(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)  # 转换为浮点型
    # dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)  # Harris 角点检测
    dst = cv2.cornerHarris(gray, blockSize=10, ksize=7, k=0.04)  # Harris 角点检测

    r, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)  # 二值化阈值处理
    dst = np.uint8(dst)  # 转换为整型
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    results = []
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[c])
        results.append([x, y, w, h])

    return results


def filter_box_by_fitted_model(results, model, deg=4, dis_thresh=15, area_thresh=36):
    new_results = []

    for b in results:
        x, y, w, h = b
        centerPoint = (x + w / 2, y + h / 2)

        if w * h < area_thresh: continue
        y_pred = generate_polynomial(x, model, deg)
        if abs(centerPoint[1] - y_pred) > dis_thresh: continue
        new_results.append(b)

    return new_results


def vis_corner_detect_results(img, results):
    for b in results:
        x, y, w, h = b
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    return img



# def main0():
#     data_path = r'G:\Gosion\data\006.Belt_Torn_Det\data\seg\v2_mini\val\images'
#     file_list = sorted(os.listdir(data_path))

#     save_path = r'G:\Gosion\data\006.Belt_Torn_Det\data\seg\v2_mini\val\images_out'
#     os.makedirs(save_path, exist_ok=True)

#     for f in file_list:
#         f_abs_path = os.path.join(data_path, f)
#         fname = os.path.splitext(f)[0]
#         img = cv2.imread(f_abs_path)
#         out = inference(img)
#         cv2.imwrite(os.path.join(save_path, fname + 'out_20250908.png'), out)


def main():
    img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\seg\v2_mini\val\images\1_output_000000005.jpg"
    # img_path = r"G:\Gosion\data\000.Test_Data\images\wechat_2025-09-17_171219_819.png"

    fname = os.path.basename(img_path)
    img_name = os.path.splitext(fname)[0]

    img = cv2.imread(img_path)
    img_cp = img.copy()

    model_path = r"G:\Gosion\code\Ultra-Fast-Lane-Detection-v2-master\weights\best.onnx"
    laserDetect = LaserDetect(model_path)
    points = laserDetect.detect(img, vis=False)

    fit_model = np.polyfit(points[:, 0], points[:, 1], deg=4)
    plot_out = fit_plot(img, fit_model, deg=4)
    cv2.imshow("plot_out", plot_out)
    

    results = corner_detect_v5(img_cp)
    new_results = filter_box_by_fitted_model(results, fit_model, deg=4, dis_thresh=15, area_thresh=36)
    img_vis = vis_corner_detect_results(img_cp, new_results)
    cv2.imshow("corner_detect", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







if __name__ == '__main__':
    main()




