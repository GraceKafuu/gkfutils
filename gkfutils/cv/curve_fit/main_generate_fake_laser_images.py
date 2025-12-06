import os
import cv2
import numpy as np
import random


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



def coefficients_add_noise_with_constraints(coefficients, x_range=(0, 2048), y_range=(0, 512), max_attempts=100000):
    """
    为多项式系数添加噪声，并确保在指定x范围内，y值在指定范围内
    
    参数:
    coefficients: 原始多项式系数列表
    x_range: x的范围 (min, max)
    y_range: y的范围 (min, max)
    max_attempts: 最大尝试次数
    
    返回:
    添加噪声后的系数列表，如果找不到满足条件的系数则返回None
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # 生成测试x值
    x_test = np.linspace(x_min, x_max, 1000)
    
    # 尝试多次添加噪声，直到找到满足条件的系数
    for attempt in range(max_attempts):
        noisy_coefficients = []
        
        for coef in coefficients:
            # 确定系数的量级
            if coef == 0:
                magnitude = 1  # 如果系数为0，使用1作为默认量级
            else:
                magnitude = 10 ** np.floor(np.log10(abs(coef)))
            
            # 随机选择使用当前量级或更大一级的量级
            use_larger_magnitude = random.choice([True, False])
            if use_larger_magnitude:
                noise_magnitude = magnitude * 10
            else:
                noise_magnitude = magnitude
            
            # 生成随机噪声，噪声大小在0.5-1.5倍量级范围内
            noise = random.uniform(0.5, 1.5) * noise_magnitude * random.choice([-1, 1])
            
            # 添加噪声到系数
            noisy_coef = coef + noise
            noisy_coefficients.append(noisy_coef)
        
        # 检查在x范围内，y值是否在指定范围内
        y_test = np.polyval(noisy_coefficients, x_test)
        if np.all(y_test >= y_min) and np.all(y_test <= y_max):
            return noisy_coefficients
    
    # 如果尝试多次后仍未找到满足条件的系数
    print(f"在{max_attempts}次尝试后未找到满足条件的系数, 返回原值！")
    return coefficients


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

    line_width = random.choice([1, 2, 3, 4])

    for i in range(len(x)):
        cv2.circle(vis, (int(x[i]), int(y[i])), line_width, color, -1)

    return vis


def add_fake_laser(img):
    # imgsz = (512, 2048)
    imgsz = img.shape[:2]
    degree = random.choice([2, 4])
    x_range=(0, imgsz[1])
    y_range=(0, imgsz[0])
    max_attempts=100000

    if degree == 2:
        coefficients = [2.19821900e-04, -3.47445927e-01, 2.07871764e+02]
    elif degree == 4:
        coefficients = [1.09052339e-10, -4.13956426e-07,  7.68820732e-04, -6.42365746e-01, 2.60268902e+02]
    else:
        print("Error!")

    coefficients = coefficients_add_noise_with_constraints(coefficients, x_range, y_range, max_attempts)
    print("coefficients:", coefficients)

    laser_color = random.choice([(0, 0, 255), (0, 255, 0)])
    img = fit_plot(img, coefficients, deg=degree, color=laser_color)

    mask = np.zeros((imgsz[0], imgsz[1], 3), np.uint8)
    mask = fit_plot(mask, coefficients, deg=degree, color=(255, 255, 255))

    return img, mask


def add_fake_laser_demo():
        # degree = random.choice([2, 3, 4])
    degree = random.choice([2, 4])

    imgsz = (512, 2048)
    x_range=(0, imgsz[1])
    y_range=(0, imgsz[0])
    max_attempts=100000

    if degree == 2:
        coefficients = [2.19821900e-04, -3.47445927e-01, 2.07871764e+02]
    elif degree == 4:
        coefficients = [1.09052339e-10, -4.13956426e-07,  7.68820732e-04, -6.42365746e-01, 2.60268902e+02]
    else:
        print("Error!")

    coefficients = coefficients_add_noise_with_constraints(coefficients, x_range, y_range, max_attempts)
    print("coefficients:", coefficients)

    line_width = random.choice([1, 2, 3, 4])
    img = np.zeros((imgsz[0], imgsz[1], line_width), np.uint8)
    img = fit_plot(img, coefficients, deg=degree)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main_TUSimple():
    TUSimple_base_path = r"G:\Gosion\data\000.OpenDatasets\TUSimple\train_set"
    TUSimple_data_path = TUSimple_base_path + "\\train_gt.txt"

    save_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\LaserDetLikeLaneMarksDet\fake_data"
    img_save_path = save_path + "\\images"
    lbl_save_path = save_path + "\\masks"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    add_str = "TUSimple"

    fr = open(TUSimple_data_path, "r")
    lines = fr.readlines()
    fr.close()

    for i, l in enumerate(lines):
        l = l.strip().split(" ")
        img_name = os.path.basename(l[0])
        fname = os.path.splitext(img_name)[0]
        
        img0_path = TUSimple_base_path + "\\" + l[0]
        img1_path = TUSimple_base_path + "\\" + l[1]

        img_path = random.choice([img0_path, img1_path])
        img = cv2.imread(img_path)
        imgsz = img.shape

        rnd_h1 = np.random.randint(0, imgsz[0])
        rnd_h = np.random.randint(180, 601)

        if rnd_h1 + rnd_h > imgsz[0]: continue

        img_part = img[rnd_h1:rnd_h1 + rnd_h, :]

        img, mask = add_fake_laser(img_part)
        img_save_path_i = img_save_path + "\\{}_{:07d}_{}.jpg".format(add_str, i, fname)
        lbl_save_path_i = lbl_save_path + "\\{}_{:07d}_{}.png".format(add_str, i, fname)
        cv2.imwrite(img_save_path_i, img)
        cv2.imwrite(lbl_save_path_i, mask)
        

def main_others():
    add_str = "videos_frames3"
    # data_path = r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v2\pexels_frames_labelbee\images"
    # data_path = r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\videos_frames\2025-05-29 11-23-55~1"
    data_path = r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\videos_frames\zhanting\images"
    # data_path = r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v2\New\002\images"
    # data_path = r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v1\train\images"
    file_list = sorted(os.listdir(data_path))


    save_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\LaserDetLikeLaneMarksDet\fake_data_v2\001"
    img_save_path = save_path + "\\images"
    lbl_save_path = save_path + "\\masks"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    for i, f in enumerate(file_list):
        fname = os.path.splitext(f)[0]
        img_path = os.path.join(data_path, f)

        img = cv2.imread(img_path)
        imgsz = img.shape

        rnd_h1 = np.random.randint(0, imgsz[0])
        rnd_h = np.random.randint(180, 601)

        if rnd_h1 + rnd_h > imgsz[0]: continue

        img_part = img[rnd_h1:rnd_h1 + rnd_h, :]
        img_partsz = img_part.shape

        # img, mask = add_fake_laser(img_part)
        # img_save_path_i = img_save_path + "\\{}_{:07d}_{}.jpg".format(add_str, i, fname)
        # lbl_save_path_i = lbl_save_path + "\\{}_{:07d}_{}.png".format(add_str, i, fname)
        # cv2.imwrite(img_save_path_i, img)
        # cv2.imwrite(lbl_save_path_i, mask)

        mask = np.zeros((img_partsz[0], img_partsz[1], 3), np.uint8)
        img_save_path_i = img_save_path + "\\{}_{:07d}_{}.jpg".format(add_str, i, fname)
        lbl_save_path_i = lbl_save_path + "\\{}_{:07d}_{}.png".format(add_str, i, fname)
        cv2.imwrite(img_save_path_i, img_part)
        cv2.imwrite(lbl_save_path_i, mask)


def main_20251106():
    add_str = "xianchangshuju"
    # data_path = r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v2\pexels_frames_labelbee\images"
    # data_path = r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\videos_frames\2025-05-29 11-23-55~1"
    # data_path = r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\videos_frames\zhanting\images"
    # data_path = r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v2\New\002\images"
    # data_path = r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v1\train\images"
    data_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\LaserDetLikeLaneMarksDet\fake_data_v2\000\images"
    file_list = sorted(os.listdir(data_path))


    save_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\LaserDetLikeLaneMarksDet\fake_data_v2\000"
    # img_save_path = save_path + "\\images"
    lbl_save_path = save_path + "\\masks"
    # os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    for i, f in enumerate(file_list):
        fname = os.path.splitext(f)[0]
        img_path = os.path.join(data_path, f)

        img = cv2.imread(img_path)
        imgsz = img.shape

        # rnd_h1 = np.random.randint(0, imgsz[0])
        # rnd_h = np.random.randint(180, 601)

        # if rnd_h1 + rnd_h > imgsz[0]: continue

        # img_part = img[rnd_h1:rnd_h1 + rnd_h, :]
        # img_partsz = img_part.shape

        # img, mask = add_fake_laser(img_part)
        # img_save_path_i = img_save_path + "\\{}_{:07d}_{}.jpg".format(add_str, i, fname)
        # lbl_save_path_i = lbl_save_path + "\\{}_{:07d}_{}.png".format(add_str, i, fname)
        # cv2.imwrite(img_save_path_i, img)
        # cv2.imwrite(lbl_save_path_i, mask)

        mask = np.zeros((imgsz[0], imgsz[1], 3), np.uint8)
        # img_save_path_i = img_save_path + "\\{}_{:07d}_{}.jpg".format(add_str, i, fname)
        lbl_save_path_i = lbl_save_path + "\\{}.png".format(fname)
        # cv2.imwrite(img_save_path_i, img_part)
        cv2.imwrite(lbl_save_path_i, mask)


if __name__ == "__main__":
    # main_TUSimple()
    # main_others()
    main_20251106()






