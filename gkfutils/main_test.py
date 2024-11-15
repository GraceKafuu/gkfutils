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
from cv.yolo import (
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
    img_path = "./data/images/0.jpg"
    dst_path = img_path.replace(".jpg", "_res.jpg")
    if os.path.exists(dst_path): os.remove(dst_path)
    shutil.rmtree("./data/images_results")
    data_path = "./data/images"
    save_path = make_save_path(data_path=data_path, relative=".", add_str="results")
    file_list = get_file_list(data_path)
    p = 1

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


def make_border():
    # img_path = "./data/images/3.jpg"
    # dst_path = img_path.replace(".jpg", "_res.jpg")
    img_path = "./data/images/long.png"
    dst_path = img_path.replace(".png", "_res.png")
    img = cv2.imread(img_path)
    # res = make_border_v7(img, (64, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # res = make_border_v7(img, (256, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # res = make_border_v7(img, (64, 256), random=False, base_side="H", ppocr_format=True, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # cv2.imwrite(dst_path, res)

    res = make_border_v7(img, (64, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=True, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    if isinstance(res, list):
        for i in range(len(res)):
            cv2.imwrite(dst_path.replace(".png", "_res_{}.png".format(i)), res[i])
    else:
        cv2.imwrite(dst_path, res)


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


def main_test():
    pass


if __name__ == '__main__':
    # image_processing()
    # image_processing_aug()
    # make_border()

    main_test()

    
























