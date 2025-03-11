# -*- coding:utf-8 -*-

"""
# @Time       : 2022/05/13 13:56 Update
#               2024/03/29 14:30 Update
#               2024/10/14 16:15 Update
# @Author     : GraceKafuu
# @Email      : gracekafuu@gmail.com
# @File       : main_test.py
# @Software   : PyCharm

Description:
1.
2.
3.

Change Log:
1.
2.
3.

"""


from cv.utils import *
from cv.YOLO.yolo import (
    YOLOv5_ONNX, YOLOv8_ONNX
)

import os
import re
import cv2
import json
import random
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import inspect
import importlib


def image_processing():
    img_path = "./data/images/0.jpg"
    dst_path = img_path.replace(".jpg", "_res.jpg")
    img = cv2.imread(img_path)
    # res = rotate(img, random=False, p=1, algorithm=algorithm, center=(100, 100), angle=angle, scale=1, expand=expand)
    # res = flip(img, random=False, p=1, m=-1)
    # res = scale(img, random=False, p=1, fx=0.0, fy=0.5)
    # res = resize(img, random=False, p=1, dsz=(1920, 1080), interpolation=cv2.INTER_LINEAR)
    # res = equalize_hist(img, random=False, p=1, m=1)
    # res = change_brightness(img, random=False, p=1, value=100)
    # res = gamma_correction(img, random=False, p=1, value=1.3)
    # res = gaussian_noise(img, random=False, p=1, mean=0, var=0.1)
    # res = poisson_noise(img, random=False, p=1)
    # res = sp_noise(img, random=False, p=1, salt_p=0.0, pepper_p=0.001)
    # res = make_sunlight_effect(img, random=False, p=1, center=(200, 200), effect_r=70, light_strength=170)
    # res = color_distortion(img, random=False, p=1, value=-50)
    # res = change_contrast_and_brightness(img, random=False, p=1, alpha=0.5, beta=90)
    # res = clahe(img, random=False, p=1, m=1, clipLimit=2.0, tileGridSize=(8, 8))
    # res = change_hsv(img, random=False, p=1, hgain=0.5, sgain=0.5, vgain=0.5)
    # res = gaussian_blur(img, random=False, p=1, k=5)
    # res = motion_blur(img, random=False, p=1, k=15, angle=90)
    # res = median_blur(img, random=False, p=1, k=3)
    # res = transperent_overlay(img, random=False, p=1, rect=(50, 50, 80, 100))
    # res = dilation_erosion(img, random=False, p=1, flag="erode", scale=(6, 8))
    # res = make_rain_effect(img, random=False, p=1, m=1, length=20, angle=75, noise=500)
    # res = compress(img, random=False, p=1, quality=80)
    # res = exposure(img, random=False, p=1, rect=(100, 150, 200, 180))
    # res = change_definition(img, random=False, p=1, r=0.5)
    # res = stretch(img, random=False, p=1, r=0.5)
    # res = crop(img, random=False, p=1, rect=(0, 0, 100, 200))
    # res = make_mask(img, random=False, p=1, rect=(0, 0, 100, 200), color=(255, 0, 255))
    # res = squeeze(img, random=False, p=1, degree=20)
    # res = make_haha_mirror_effect(img, random=False, p=1, center=(150, 150), r=10, degree=20)
    # res = warp_img(img, random=False, p=1, degree=10)
    # res = enhance_gray_value(img, random=False, p=1, gray_range=(0, 255))
    # res = homomorphic_filter(img, random=False, p=1)
    # res = contrast_stretch(img, random=False, p=1, alpha=0.25, beta=0.75)
    # res = log_transformation(img, random=False, p=1)
    res = translate(img, random=False, p=1, tx=-20, ty=30, border_color=(114, 0, 114), dstsz=None)
    cv2.imwrite(dst_path, res)


def image_processing_aug():
    # img_path = "./data/images/0.jpg"
    # dst_path = img_path.replace(".jpg", "_res.jpg")
    # if os.path.exists(dst_path): os.remove(dst_path)
    # shutil.rmtree("./data/images_results")

    data_path = "./data/images"
    save_path = make_save_path(data_path=data_path, relative=".", add_str="results")
    file_list = get_file_list(data_path)
    p = 0.5

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        
        img = dilate_erode(img, random=True, p=p, flag=np.random.choice(["dilate", "erode"]))

        rdm0 = np.random.choice(np.arange(2))
        if rdm0 == 0:
            img = scale(img, random=True, p=p, fx=(0.5, 1.5), fy=(0.5, 1.5))
        else:
            img = stretch(img, random=True, p=p, r=(0.25, 1.25))

        rdm1 = np.random.choice(np.arange(5))
        if rdm1 == 0:
            img = change_brightness(img, random=True, p=p, value=(-75, 75))
        elif rdm1 == 1:
            img = gamma_correction(img, random=True, p=p, value=(0.5, 1.5))
        elif rdm1 == 2:
            img = change_contrast_and_brightness(img, random=True, p=p, alpha=(0.25, 0.75), beta=(0, 75))
        elif rdm1 == 3:
            img = clahe(img, random=True, p=p, m=np.random.choice([0, 1]),  clipLimit=(2.0, 4.0), tileGridSize=(4, 16))
        else:
            img = log_transformation(img, random=True, p=p)

        rdm2 = np.random.choice(np.arange(6))
        if rdm2 == 0:
            img = gaussian_noise(img, random=True, p=p, mean=(0, 1), var=(0.1, 0.25))
        elif rdm2 == 1:
            img = poisson_noise(img, random=True, p=p, n=(2, 5))
        elif rdm2 == 2:
            img = sp_noise(img, random=True, p=p, salt_p=(0.01, 0.025), pepper_p=(0.01, 0.025))
        elif rdm2 == 3:
            img = gaussian_blur(img, random=True, p=p)
        elif rdm2 == 4:
            img = motion_blur(img, random=True, p=p, angle=(-180, 180))
        else:
            img = median_blur(img, random=True, p=p)
        
        rdm3 = np.random.choice(np.arange(2))
        if rdm3 == 0:
            img = color_distortion(img, random=True, p=p, value=(-360, 360))
        else:
            img = change_hsv(img, random=True, p=p, hgain=(0.25, 0.75), sgain=(0.25, 0.75), vgain=(0.25, 0.75))
        
        img = transperent_overlay(img, random=True, p=p, max_h_r=1.0, max_w_r=0.5, alpha=(0.1, 0.6))

        # rdm4 = np.random.choice(np.arange(3))
        # if rdm4 == 0:
        #     img = dilate_erode(img, random=True, p=p, flag=np.random.choice(["dilate", "erode"]))
        # elif rdm4 == 1:
        #     img = open_close_gradient(img, random=True, p=p, flag=np.random.choice(["open", "close", "gradient"]))
        # else:
        #     img = tophat_blackhat(img, random=True, p=p, flag=np.random.choice(["tophat", "blackhat"]))

        rdm5 = np.random.choice(np.arange(2))
        if rdm5 == 0:
            img = make_sunlight_effect(img, random=True, p=p, effect_r=(10, 80), light_strength=(50, 80))
        else:
            img = make_rain_effect(img, random=True, p=p, m=np.random.choice([0, 1]), length=(10, 90), angle=(0, 180), noise=(100, 500))
        
        img = compress(img, random=True, p=p, quality=(25, 95))
        img = rotate(img, random=True, p=p, algorithm="pil", angle=(-45, 45), expand=True)

        # 以下OCR数据增强时不建议使用:
        # img = flip(img, random=True, p=p, m=np.random.choice([-1, 0, 1]))  # m=np.random.choice([-1, 0, 1])
        # img = equalize_hist(img, random=True, p=p, m=1)  # m=np.random.choice([0, 1])
        # img = translate(img, random=True, p=p, tx=(-50, 50), ty=(-50, 50), dstsz=None)

        # 以下还存在问题, 需要优化:
        # img = warp_and_deform(img, random=True, p=p, a=(5, 15), b=(1, 5), gridspace=(10, 20))
        # img = normalize(img, random=True, p=p, alpha=0, beta=1, norm_type=np.random.choice([cv2.NORM_MINMAX, cv2.NORM_L2]))  # 容易变黑图

        f_dst_path = save_path + "/{}.jpg".format(fname)
        cv2.imwrite(f_dst_path, img)


def image_processing_aug_det_data(data_path):
    # img_path = "./data/images/0.jpg"
    # dst_path = img_path.replace(".jpg", "_res.jpg")
    # if os.path.exists(dst_path): os.remove(dst_path)
    # shutil.rmtree("./data/images_results")

    # data_path = "./data/images"
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    
    save_path = make_save_path(data_path, relative=".", add_str="aug")
    img_save_path = save_path + "/images"
    lbl_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    file_list = get_file_list(img_path)
    p = 0.5

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = img_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)

        img_dst_path = img_save_path + "/{}".format(f)
        lbl_abs_path = lbl_path + "/{}.txt".format(fname)
        lbl_dst_path = lbl_save_path + "/{}.txt".format(fname)
        
        # img = dilate_erode(img, random=True, p=p, flag=np.random.choice(["dilate", "erode"]))

        # rdm0 = np.random.choice(np.arange(2))
        # if rdm0 == 0:
        #     img = scale(img, random=True, p=p, fx=(0.5, 1.5), fy=(0.5, 1.5))
        # else:
        #     img = stretch(img, random=True, p=p, r=(0.25, 1.25))

        rdm1 = np.random.choice(np.arange(5))
        if rdm1 == 0:
            img = change_brightness(img, random=True, p=p, value=(-25, 25))
        elif rdm1 == 1:
            img = gamma_correction(img, random=True, p=p, value=(0.5, 1.5))
        # elif rdm1 == 2:
        #     img = change_contrast_and_brightness(img, random=True, p=p, alpha=(0.25, 0.75), beta=(0, 25))
        # elif rdm1 == 3:
        #     img = clahe(img, random=True, p=p, m=np.random.choice([0, 1]),  clipLimit=(2.0, 4.0), tileGridSize=(4, 16))
        else:
            img = log_transformation(img, random=True, p=p)

        rdm2 = np.random.choice(np.arange(6))
        if rdm2 == 0:
            img = gaussian_noise(img, random=True, p=p, mean=(0, 1), var=(0.1, 0.25))
        elif rdm2 == 1:
            img = poisson_noise(img, random=True, p=p, n=(2, 5))
        elif rdm2 == 2:
            img = sp_noise(img, random=True, p=p, salt_p=(0.01, 0.025), pepper_p=(0.01, 0.025))
        elif rdm2 == 3:
            img = gaussian_blur(img, random=True, p=p)
        # elif rdm2 == 4:
        #     img = motion_blur(img, random=True, p=p, angle=(-180, 180))
        else:
            img = median_blur(img, random=True, p=p)
        
        # rdm3 = np.random.choice(np.arange(2))
        # # img = color_distortion(img, random=True, p=p, value=(-360, 360))
        # # img = change_hsv(img, random=True, p=p, hgain=(0.25, 0.95), sgain=(0.25, 0.95), vgain=(0.25, 0.95))
        # img = change_color(img, random=True, p=1, hue_shift=30)
        # if rdm3 == 0:
        #     img = color_distortion(img, random=True, p=p, value=(-360, 360))
        # else:
        #     img = change_hsv(img, random=True, p=p, hgain=(0.45, 0.95), sgain=(0.45, 0.95), vgain=(0.45, 0.95))
        
        # img = transperent_overlay(img, random=True, p=p, max_h_r=1.0, max_w_r=0.5, alpha=(0.1, 0.6))

        # rdm4 = np.random.choice(np.arange(3))
        # if rdm4 == 0:
        #     img = dilate_erode(img, random=True, p=p, flag=np.random.choice(["dilate", "erode"]))
        # elif rdm4 == 1:
        #     img = open_close_gradient(img, random=True, p=p, flag=np.random.choice(["open", "close", "gradient"]))
        # else:
        #     img = tophat_blackhat(img, random=True, p=p, flag=np.random.choice(["tophat", "blackhat"]))

        # rdm5 = np.random.choice(np.arange(2))
        # if rdm5 == 0:
        #     img = make_sunlight_effect(img, random=True, p=p, effect_r=(10, 80), light_strength=(50, 80))
        # else:
        #     img = make_rain_effect(img, random=True, p=p, m=np.random.choice([0, 1]), length=(10, 90), angle=(0, 180), noise=(100, 500))

        # img = make_rain_effect(img, random=True, p=p, m=np.random.choice([0, 1]), length=(10, 90), angle=(0, 180), noise=(100, 500))
        
        # img = compress(img, random=True, p=p, quality=(25, 95))
        # img = rotate(img, random=True, p=p, algorithm="pil", angle=(-45, 45), expand=True)

        # 以下OCR数据增强时不建议使用:
        # img = flip(img, random=True, p=p, m=np.random.choice([-1, 0, 1]))  # m=np.random.choice([-1, 0, 1])
        # img = equalize_hist(img, random=True, p=p, m=1)  # m=np.random.choice([0, 1])
        # img = translate(img, random=True, p=p, tx=(-50, 50), ty=(-50, 50), dstsz=None)

        # 以下还存在问题, 需要优化:
        # img = warp_and_deform(img, random=True, p=p, a=(5, 15), b=(1, 5), gridspace=(10, 20))
        # img = normalize(img, random=True, p=p, alpha=0, beta=1, norm_type=np.random.choice([cv2.NORM_MINMAX, cv2.NORM_L2]))  # 容易变黑图

        brightness = cal_brightness_v2(img)
        if brightness > 200: continue

        cv2.imwrite(img_dst_path, img)
        shutil.copy(lbl_abs_path, lbl_dst_path)


def make_border():
    # # img_path = "./data/images/3.jpg"
    # # dst_path = img_path.replace(".jpg", "_res.jpg")
    # img_path = "./data/images/long.png"
    # dst_path = img_path.replace(".png", "_res.png")
    # img = cv2.imread(img_path)
    # # res = make_border_v7(img, (64, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # # res = make_border_v7(img, (256, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # # res = make_border_v7(img, (64, 256), random=False, base_side="H", ppocr_format=True, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # # cv2.imwrite(dst_path, res)

    # res = make_border_v7(img, (64, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=True, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # if isinstance(res, list):
    #     for i in range(len(res)):
    #         cv2.imwrite(dst_path.replace(".png", "_res_{}.png".format(i)), res[i])
    # else:
    #     cv2.imwrite(dst_path, res)

    
    data_path = r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v3\from_YanDajun_checked\train_makeBorder\images-orig"
    save_path = make_save_path(data_path, relative=".", add_str="make_border")
    file_list = os.listdir(data_path)

    rs = list(range(10, 30))
    rs = [i * 0.01 for i in rs]
    print(rs)

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        f_dst_path = save_path + "/{}.jpg".format(fname)
        img = cv2.imread(f_abs_path)
        imgsz = img.shape[:2]
        # r = np.random.uniform(0.50, 0.99)
        r = np.random.choice(rs)
        # img = cv2.resize(img, (int(imgsz[1] * np.random.uniform(0.50, 0.99)), int(imgsz[0] * np.random.uniform(0.50, 0.99))))
        img = cv2.resize(img, (int(imgsz[1] * r), int(imgsz[0] * r)))
        imgsz_new = img.shape[:2]

        # color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        color = (0, 0, 0)
        top = (1080 - imgsz_new[0]) // 2
        bottom = 1080 - top - imgsz_new[0]
        left = (1920 - imgsz_new[1]) // 2
        right = 1920 - left - imgsz_new[1]

        res = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        cv2.imwrite(f_dst_path, res)


        # res = make_border_v7(img, (1080, 1920), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=True, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
        # if isinstance(res, list):
        #     for i in range(len(res)):
        #         cv2.imwrite(f_dst_path.replace(".jpg", "_res_{}.jpg".format(i)), res[i])
        # else:
        #     cv2.imwrite(f_dst_path, res)


def yolov5_inference():
    onnx_path = r"E:\GraceKafuu\Python\yolov5-6.2\yolov5s.onnx"
    img_path = r"E:\GraceKafuu\Python\yolov5-6.2\data\images\bus.jpg"
    
    model = YOLOv5_ONNX(onnx_path)
    model_input_size = (448, 768)
    img0, img, src_size = model.pre_process(img_path, img_size=model_input_size)
    print("src_size: ", src_size)
    
    t1 = time.time()
    pred = model.inference(img)
    t2 = time.time()
    print("{:.12f}".format(t2 - t1))
    
    out_bbx = model.post_process(pred, src_size, img_size=model_input_size)
    print("out_bbx: ", out_bbx)
    for b in out_bbx:
        cv2.rectangle(img0, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
    cv2.imshow("test", img0)
    cv2.waitKey(0)


def yolov8_inference():
    onnx_path = r"E:\GraceKafuu\Python\ultralytics-main\yolov8n.onnx"
    img_path = r"E:\GraceKafuu\Python\yolov5-6.2\data\images\bus.jpg"
    
    model = YOLOv8_ONNX(onnx_path)
    model_input_size = (640, 640)
    img0, img, src_size = model.pre_process(img_path, img_size=model_input_size)
    print("src_size: ", src_size)
    
    t1 = time.time()
    pred = model.inference(img)
    t2 = time.time()
    print("{:.12f}".format(t2 - t1))
    
    out_bbx = model.post_process(pred, src_size, img_size=model_input_size)
    print("out_bbx: ", out_bbx)
    for b in out_bbx:
        cv2.rectangle(img0, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
    cv2.imshow("test", img0)
    cv2.waitKey(0)


def list_module_functions():
    """
    列出模块中所有的函数
    """
    current_file = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file)
    os.chdir(current_dir)
    module = importlib.import_module(os.path.basename(current_file)[:-3])
    functions = [func for func in dir(module) if callable(getattr(module, func))]
    print(sorted(functions))


def hog_test():
    data_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data"
    img_list = sorted(os.listdir(data_path))
    for img in img_list:
        img_name = os.path.splitext(img)[0]
        img_abs_path = data_path + "/{}".format(img)
        img = cv2.imread(img_abs_path, 0)
        hog_res = apply_hog(img)
        cv2.imwrite("{}/{}_hog.jpg".format(data_path, img_name), hog_res * 255)


def dehaze_test():
    m = dehaze(cv2.imread('/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data/20221111152824_8b46d8_75_0028505.jpg') / 255.0) * 255
    cv2.imwrite('/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data/20221111152824_8b46d8_75_0028505_defog.jpg', m)


def wt_test():
    from pywt import dwt, idwt, dwt2, idwt2

    img_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/cmp_data2/smoke"
    img_list = os.listdir(img_path)

    save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/output/cmp_data2/smoke"
    os.makedirs(save_path, exist_ok=True)

    Es = []

    for img in img_list:
        img_name = os.path.splitext(img)[0]
        img_abs_path = img_path + "/{}".format(img)
        img = cv2.imread(img_abs_path, 0)
        cA, (cH, cV, cD) = dwt2(img, "haar")

        cAH = np.hstack((cA, cH))
        cVD = np.hstack((cV, cD))
        cAHVD = np.vstack((cAH, cVD))
        cv2.imwrite("{}/{}_dwt2.jpg".format(save_path, img_name), cAHVD)

        img_resz = cv2.resize(img, (cA.shape[1], cA.shape[0]))
        img_cha = cv2.subtract(np.uint8(cv2.merge([img_resz, img_resz, img_resz])), np.uint8(cv2.merge([cA, cA, cA])))
        # img_cha = cv2.subtract(cv2.merge([img_resz, img_resz, img_resz]), cv2.merge([cA, cA, cA]))
        # img_cha = cv2.subtract(cv2.merge([img_resz, img_resz, img_resz]), cv2.merge([cA, cA, cA]))
        print(img_cha.sum())
        cv2.imwrite("{}/{}_img_cha.jpg".format(save_path, img_name), img_cha)

        energy = (cH ** 2 + cV ** 2 + cD ** 2).sum() / img.size
        print("E: ", energy)

        Es.append(energy)

    print("E mean: ", np.mean(Es))


def saliency_map_ft_test():
    from pywt import dwt, idwt, dwt2, idwt2

    # img = cv2.imread("/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data/fire_smoke_20230203_0000133.jpg")
    # saliency_map = cal_saliency_map_FT(img)
    # cv2.imwrite("/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data/fire_smoke_20230203_0000133_saliency_map_ft.jpg", saliency_map * 255)
    data_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/smoke_PS/data"

    bg_path = os.path.abspath(os.path.join(data_path, "../..")) + "/bg"
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/output"
    os.makedirs(save_path, exist_ok=True)

    img_list = sorted(os.listdir(data_path))
    for imgi in img_list:
        img_name = os.path.splitext(imgi)[0]
        img_abs_path = data_path + "/{}".format(imgi)
        bg_abs_path = bg_path + "/{}".format(imgi)
        img = cv2.imread(img_abs_path)
        H, W = img.shape[:2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_cp = img.copy()
        bg_img = cv2.imread(bg_abs_path)
        bg_b, bg_g, bg_r = cv2.split(bg_img)
        bg_img_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        # bg_img_cp = bg_img.copy()
        saliency_map2 = cal_saliency_map(img_abs_path, algorithm="FT2")
        saliency_map2_merge = cv2.merge([saliency_map2, saliency_map2, saliency_map2])
        saliency_map = cal_saliency_map(img) * 255
        b, g, r = cv2.split(saliency_map)
        ret, b_bin = cv2.threshold(np.uint8(b), 70, 255, cv2.THRESH_BINARY)
        cv2.imwrite("{}/{}_saliency_map2.jpg".format(save_path, img_name), saliency_map2)
        cv2.imwrite("{}/{}_saliency_map2_merge.jpg".format(save_path, img_name), saliency_map2_merge)
        cv2.imwrite("{}/{}_saliency_map.jpg".format(save_path, img_name), saliency_map)
        cv2.imwrite("{}/{}_saliency_map_b.jpg".format(save_path, img_name), b)
        cv2.imwrite("{}/{}_saliency_map_g.jpg".format(save_path, img_name), g)
        cv2.imwrite("{}/{}_saliency_map_r.jpg".format(save_path, img_name), r)
        cv2.imwrite("{}/{}_saliency_map_b_bin.jpg".format(save_path, img_name), b_bin)

        b_bin_merge = cv2.merge([b_bin, b_bin, b_bin])

        cnts, hierarchy = cv2.findContours(b_bin.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_cnts = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_cnts)
        cv2.rectangle(b_bin, (x, y), (x + w, y + h), (255, 0, 255), 5)
        cv2.rectangle(img_cp, (x, y), (x + w, y + h), (255, 0, 255), 5)
        cv2.imwrite("{}/{}_saliency_map_b_bin_rect.jpg".format(save_path, img_name), b_bin)
        # cv2.imwrite("{}/{}_saliency_map_img_rect.jpg".format(data_path, img_name), img_cp)

        bg_roi = bg_img_gray[y:y + h, x:x + w]
        bgcA, (bgcH, bgcV, bgcD) = dwt2(bg_roi, "haar")
        # bgcAH = np.hstack((bgcA, bgcH))
        # bgcVD = np.hstack((bgcV, bgcD))
        # bgcAHVD = np.vstack((bgcAH, bgcVD))
        # bg_img_resz = cv2.resize(img, (cA.shape[1], cA.shape[0]))
        bg_energy_gray = (bgcH ** 2 + bgcV ** 2 + bgcD ** 2).sum() / bg_roi.size
        print("E_bg_gray: ", bg_energy_gray)

        bg_roi = bg_b[y:y + h, x:x + w]
        bgcA, (bgcH, bgcV, bgcD) = dwt2(bg_roi, "haar")
        # bgcAH = np.hstack((bgcA, bgcH))
        # bgcVD = np.hstack((bgcV, bgcD))
        # bgcAHVD = np.vstack((bgcAH, bgcVD))
        # bg_img_resz = cv2.resize(img, (cA.shape[1], cA.shape[0]))
        bg_energy = (bgcH ** 2 + bgcV ** 2 + bgcD ** 2).sum() / bg_roi.size
        print("E_bg_b: ", bg_energy)

        b_roi = img_gray[y:y + h, x:x + w]
        cA, (cH, cV, cD) = dwt2(b_roi, "haar")
        cAH = np.hstack((cA, cH))
        cVD = np.hstack((cV, cD))
        cAHVD = np.vstack((cAH, cVD))
        img_resz = cv2.resize(cAHVD, (W, H))
        img_resz_merge = cv2.merge([img_resz, img_resz, img_resz])
        energy_gray = (cH ** 2 + cV ** 2 + cD ** 2).sum() / b_roi.size
        print("E_gray: ", energy_gray)

        b_roi = b[y:y + h, x:x + w]
        cA, (cH, cV, cD) = dwt2(b_roi, "haar")
        cAH = np.hstack((cA, cH))
        cVD = np.hstack((cV, cD))
        cAHVD = np.vstack((cAH, cVD))
        img_resz = cv2.resize(cAHVD, (W, H))
        energy = (cH ** 2 + cV ** 2 + cD ** 2).sum() / b_roi.size
        print("E_b: ", energy)

        E_ratio_gray = energy_gray / bg_energy_gray
        E_ratio_b = energy / bg_energy
        print("E_ratio_gray: {}".format(E_ratio_gray))
        print("E_ratio_b: {}".format(E_ratio_b))

        img_roi = img[y:y + h, x:x + w]
        # img_roi_b, img_roi_g, img_roi_r = cv2.split(img_roi)
        B_, G_, R_ = np.mean(img_roi[:, :, 0]), np.mean(img_roi[:, :, 1]), np.mean(img_roi[:, :, 2])
        print("B_, G_, R_: ", B_, G_, R_)

        cv2.putText(img_cp, "E_bg: {:.2f}".format(bg_energy), (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
        cv2.putText(img_cp, "E: {:.2f}".format(energy), (20, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
        cv2.putText(img_cp, "E_ratio: {:.2f}".format(E_ratio_b), (20, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
        cv2.putText(img_cp, "B_, G_, R_: {:.2f} {:.2f} {:.2f}".format(B_, G_, R_), (20, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

        cv2.imwrite("{}/{}_saliency_map_img_cp.jpg".format(save_path, img_name), img_cp)
        out_img = np.hstack((img, saliency_map, b_bin_merge, img_resz_merge, img_cp))
        cv2.imwrite("{}/{}_saliency_map_stacked.jpg".format(save_path, img_name), out_img)


def gkfocr_test():
    data = "data/doc/imgs"
    # data = "data/doc/imgs/11.jpg"
    ocr = GKFOCR(cfg_path="configs/cfg_gkfocr.yaml", debug=False)
    out_img = ocr.inference(data)
    if out_img is not None:
        cv2.imwrite("data/doc/test_out_img.jpg", out_img)


def seamless_clone_test(save_path):
    os.makedirs(save_path, exist_ok=True)
    bg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/gen_fake/gen_AbC/bg/2.jpg"
    fg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/gen_fake/gen_AbC/0-9_AbC_new"
    fg_list = sorted(os.listdir(fg_path))

    bg_img = cv2.imread(bg_path)
    bgsz = bg_img.shape[:2]

    rdm = random.random()

    label_str = ""
    for i in range(4):
        fgi = random.sample(fg_list, 1)[0]
        fgi_name = os.path.splitext(fgi)[0]
        if "AN" in fgi_name:
            label_str += "A"
        elif "bN" in fgi_name:
            label_str += "b"
        elif "CN" in fgi_name:
            label_str += "C"
        elif "0N" in fgi_name:
            label_str += "0"
        elif "1N" in fgi_name:
            label_str += "1"
        elif "2N" in fgi_name:
            label_str += "2"
        elif "3N" in fgi_name:
            label_str += "3"
        elif "4N" in fgi_name:
            label_str += "4"
        elif "5N" in fgi_name:
            label_str += "5"
        elif "6N" in fgi_name:
            label_str += "6"
        elif "7N" in fgi_name:
            label_str += "7"
        elif "8N" in fgi_name:
            label_str += "8"
        elif "9N" in fgi_name:
            label_str += "9"
        elif "0.N" in fgi_name:
            label_str += "0."
        elif "1.N" in fgi_name:
            label_str += "1."
        elif "2.N" in fgi_name:
            label_str += "2."
        elif "3.N" in fgi_name:
            label_str += "3."
        elif "4.N" in fgi_name:
            label_str += "4."
        elif "5.N" in fgi_name:
            label_str += "5."
        elif "6.N" in fgi_name:
            label_str += "6."
        elif "7.N" in fgi_name:
            label_str += "7."
        elif "8.N" in fgi_name:
            label_str += "8."
        elif "9.N" in fgi_name:
            label_str += "9."
        elif "space" in fgi_name:
            label_str += ""
        else:
            print("Error!")

        fg_abs_path = fg_path + "/{}".format(fgi)
        fg_img = cv2.imread(fg_abs_path)
        fg_img = cv2.resize(fg_img, (48, 70))
        fgsz = fg_img.shape[:2]

        mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
        out = cv2.seamlessClone(fg_img, bg_img, mask_img, (30 + 56 * i, 36), cv2.MIXED_CLONE)
        bg_img = out

    cv2.imwrite("{}/20231008_{}_{}={}.jpg".format(save_path, str(rdm).replace(".", ""), fgi_name, label_str), out)


def det_labels_convertion():
    """ yolo <-> labelbee """
    # labelbee_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_labelbee_format", copy_images=True, small_bbx_thresh=3, cls_plus=-1)  # OK
    # yolo_to_labelbee(data_path="E:/GraceKafuu/yolo/coco128/data", copy_images=True, small_bbx_thresh=3, cls_plus=1)  # OK

    """ yolo <-> voc """
    # coco_classes = get_coco_names()
    # voc_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_voc_format", classes=coco_classes, copy_images=True, small_bbx_thresh=3, cls_plus=0)  # OK
    # yolo_to_voc(data_path="E:/GraceKafuu/yolo/coco128/data", classes=coco_classes, copy_images=True, small_bbx_thresh=3, cls_plus=0)  # OK

    """ yolo <-> coco """
    # categories = get_coco_categories()
    # coco_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_coco_format", json_name="instances_val2017_20241121.json", copy_images=False, small_bbx_thresh=3, cls_plus=0)  # OK
    # yolo_to_coco(data_path="E:/GraceKafuu/yolo/coco128/data", json_name="instances_val2017_20241121.json", categories=categories, copy_images=False, small_bbx_thresh=3, cls_plus=0)  # OK

    """ yolo <-> labelme """
    # TODO
    # labelme_to_yolo()
    # yolo_to_labelme()


def warmup_schedule(optimizer, name):
    import pytorch_warmup

    if name == 'linear':
        return pytorch_warmup.UntunedLinearWarmup(optimizer)
    elif name == 'exponential':
        return pytorch_warmup.UntunedExponentialWarmup(optimizer)
    elif name == 'radam':
        return pytorch_warmup.RAdamWarmup(optimizer)
    elif name == 'none':
        return pytorch_warmup.LinearWarmup(optimizer, 1)
    

def pytorch_warmup_test1():
    """ OK """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pytorch_warmup
    from torchvision.models import mobilenet_v2

    device = torch.device('cpu')

    init_lr = 0.01
    model = mobilenet_v2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), weight_decay=0.01)
    epochs = 500
    len_train_loader = 1000
    num_steps = len_train_loader * epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)
    warmup_scheduler = warmup_schedule(optimizer, 'exponential')  # linear exponential radam none

    for epoch in range(1, epochs + 1):
        for i in range(250):
            optimizer.zero_grad()
            optimizer.step()

            with warmup_scheduler.dampening():
                lr_scheduler.step()

        lr_scheduler.step()

        lr1 = optimizer.param_groups[0]['lr']
        lr2 = lr_scheduler.get_last_lr()[0]
        print("Epoch: {}, lr1: {} lr2: {}".format(epoch, lr1, lr2))


def adjust_lr(optimizer, lr_scheduler, epoch, warmup_epochs, init_lr):
    """ warmup 调整学习率 """
    if epoch <= warmup_epochs:
        lr = init_lr * (epoch + 1) / warmup_epochs
        optimizer.param_groups[0]['lr'] = lr
        lr_scheduler.get_last_lr()[0] = lr
    else:
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]

    return lr


def pytorch_warmup_test2():
    """ OK """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pytorch_warmup
    from torchvision.models import mobilenet_v2

    device = torch.device('cpu')

    model = mobilenet_v2().to(device)

    init_lr = 0.01
    epochs = 500
    warmup_epochs = 5

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # num_steps = len(train_loader) * epochs
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)

    for e in range(epochs):
        lr1 = adjust_lr(optimizer, lr_scheduler, e, warmup_epochs, init_lr)

        for i in range(250):
            optimizer.zero_grad()
            optimizer.step()

        lr2 = lr_scheduler.get_last_lr()[0]
        print("Epoch: {}, lr1: {}, lr2: {}".format(e, lr1, lr2))


def main_merge_ocr_rec_txt():
    merge_txt_files(data_path="E:\\GraceKafuu\\Resources\\data\\OCR\\rec_exp\\val\\labels")


def cal_params_flops_test():
    bias_flag = True
    conv_model = TestConv2dNet(bias=bias_flag)
    linear_model = TestLinearNet(bias=bias_flag)
    lstm_model = TestLSTMNet(bias=bias_flag)
    x1 = torch.randn(1, 3, 224, 224)
    x2 = torch.randn(1, 128, 16)
    cal_params_flops(conv_model, x1, bias_flag=bias_flag, method="thop")
    cal_params_flops(linear_model, x2, bias_flag=bias_flag, method="thop")
    cal_params_flops(lstm_model, x2, bias_flag=bias_flag, method="thop")
    





if __name__ == '__main__':
    # image_processing()
    # image_processing_aug()
    image_processing_aug_det_data(data_path=r"D:\Gosion\Projects\006.Belt_Torn_Det\data\pose\v3\train")
    # make_border()

    # det_labels_convertion()

    # pytorch_warmup_test1()  # OK
    # pytorch_warmup_test2()  # OK

    # main_merge_ocr_rec_txt()

    # cal_params_flops_test()


    
























