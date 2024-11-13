# -*- coding:utf-8 -*-

"""
# @Time       : 2022/5/13 13:56, 2024/3/29 14:30 Update
# @Author     : GraceKafuu
# @Email      : 
# @File       : det.py
# @Software   : PyCharm

Description:
1.
2.
3.

"""

# from ..utils import (
#     get_file_list, get_base_name,
#     get_dir_name, make_save_path,
#     is_all_chinese, is_all_digits
# )


import os
import re
import sys
import cv2
import json
import time
import math
import copy
import glob
import yaml
import random
import shutil
import codecs
import imghdr
import struct
import pickle
import hashlib
import base64
import socket
import argparse
import threading
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import PIL
from PIL import Image, ImageDraw
import skimage
import scipy
import torch
import torchvision
import onnxruntime as ort
from torchvision import transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyclipper
from shapely.geometry import Polygon


# Base utils ===================================================
def get_file_list(data_path: str, abspath=False) -> list:
    file_list = []
    list_ = sorted(os.listdir(data_path))
    for f in list_:
        f_path = data_path + "/{}".format(f)
        if os.path.isfile(f_path):
            if abspath:
                file_list.append(f_path)
            else:
                file_list.append(f)
    return file_list


def get_dir_list(data_path: str, abspath=False):
    dir_list = []
    list_ = sorted(os.listdir(data_path))
    for f in list_:
        f_path = data_path + "/{}".format(f)
        if os.path.isdir(f_path):
            if abspath:
                dir_list.append(f_path)
            else:
                dir_list.append(f)
    return dir_list


def get_dir_file_list(data_path: str, abspath=False):
    list_ = sorted(os.listdir(data_path))
    if abspath:
        list_new = []
        for f in list_:
            f_path = data_path + "/{}".format(f)
            list_new.append(f_path)
        return list_new
    else:
        return list_


def get_base_name(data_path: str):
    base_name = os.path.basename(data_path)
    return base_name


def get_dir_name(data_path: str):
    assert os.path.isdir(data_path), "{} is not a dir!".format(data_path)
    dir_name = os.path.basename(data_path)
    return dir_name


def get_file_name(data_path: str):
    """
    without suffix
    """
    assert os.path.isfile(data_path), "{} is not a file!".format(data_path)
    base_name = os.path.basename(data_path)
    file_name = os.path.splitext(base_name)[0]
    return file_name


def get_file_name_with_suffix(data_path: str):
    assert os.path.isfile(data_path), "{} is not a file!".format(data_path)
    base_name = os.path.basename(data_path)
    return base_name


def get_suffix(data_path: str):
    assert os.path.isfile(data_path), "{} is not a file!".format(data_path)
    base_name = os.path.basename(data_path)
    suffix = os.path.splitext(base_name)[1]
    return suffix


def make_save_path(data_path: str, relative=".", add_str="results"):
    base_name = get_base_name(data_path)
    if relative == ".":
        save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_{}".format(base_name, add_str)
    elif relative == "..":
        save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_{}".format(base_name, add_str)
    elif relative == "...":
        save_path = os.path.abspath(os.path.join(data_path, "../../..")) + "/{}_{}".format(base_name, add_str)
    else:
        print("relative should be . or .. or ...")
        raise ValueError
    os.makedirs(save_path, exist_ok=True)
    return save_path


def save_file_path_to_txt(data_path: str, abspath=True):
    assert type(data_path) == str, "{} should be str!".format(data_path)
    dirname = os.path.basename(data_path)
    data_list = sorted(os.listdir(data_path))
    txt_save_path = os.path.abspath(os.path.join(data_path, "../{}_list.txt".format(dirname)))
    with open(txt_save_path, 'w', encoding='utf-8') as fw:
        for f in data_list:
            if abspath:
                f_abs_path = data_path + "/{}".format(f)
                fw.write("{}\n".format(f_abs_path))
            else:
                fw.write("{}\n".format(f))

    print("Success! --> {}".format(txt_save_path))


def is_all_digits(string):
    pattern = r'^\d+$'
    if re.match(pattern, string):
        return True
    else:
        return False


def is_all_chinese(string):
    pattern = '[\u4e00-\u9fa5]+'
    if re.match(pattern, string) and len(string) == len(set(string)):
        return True
    else:
        return False
    

# Image processing utils ===================================================
def cv2pil(image):
    assert isinstance(image, np.ndarray), f'Input image type is not cv2 and is {type(image)}!'
    if len(image.shape) == 2:
        return Image.fromarray(image)
    elif len(image.shape) == 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        return None


def pil2cv(image):
    assert isinstance(image, PIL.Image.Image), f'Input image type is not PIL.image and is {type(image)}!'
    if len(image.split()) == 1:
        return np.asarray(image)
    elif len(image.split()) == 3:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    elif len(image.split()) == 4:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGR)
    else:
        return None


def rotate(img, random=False, p=1, algorithm="pil", center=(50, 50), angle=(-45, 45), scale=1, expand=True) -> np.ndarray:
    if random:
        if np.random.random() <= p:
            assert isinstance(angle, tuple), "if random=True, angle is tuple."
            angle = np.random.randint(angle[0], angle[1] + 1)
            if algorithm == "cv2":
                assert isinstance(scale, tuple), "if random=True and algorithm='cv2', scale is tuple."
                scale = np.random.uniform(scale[0], scale[1] + 1e-6)
                if isinstance(img, PIL.Image.Image):
                    img = np.asarray(img)
                imgsz = img.shape[:2]
                x = np.random.randint(0, imgsz[1])
                y = np.random.randint(0, imgsz[0])
                center = np.random.randint(x, y)
                M = cv2.getRotationMatrix2D(center, angle, scale)
                img = cv2.warpAffine(img, M, imgsz[::-1])
            else:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(np.uint8(img))
                img = np.asarray(img.rotate(angle, expand=expand))
                
            return img
        
        else:
            return img
    else:
        assert isinstance(angle, int), "if random=False, angle is int."
        if algorithm == "cv2":
            assert isinstance(scale, float), "if random=False, scale is float."
            if isinstance(img, PIL.Image.Image):
                img = np.asarray(img)
            imgsz = img.shape[:2]
            M = cv2.getRotationMatrix2D(center, angle, scale)
            img = cv2.warpAffine(img, M, imgsz[::-1])
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(np.uint8(img))
            img = np.asarray(img.rotate(angle, expand=expand))
            
        return img


def flip(img, random=False, p=1, m=0):
    """
    0:垂直翻转(沿x轴翻转)
    1:水平翻转(沿y轴翻转)
    -1:同时在水平和垂直方向翻转

    """
    assert m in [-1, 0, 1], "m(flip direction) should be one of [-1, 0, 1]"

    if random:
        if np.random.random() <= p:
            ms = [-1, 0, 1]
            m = np.random.choice(ms)
            img = cv2.flip(img, m)
            return img
        else:
            return img
    else:
        img = cv2.flip(img, m)
        return img


def scale(img, random=False, p=1, fx=0.5, fy=0.5):
    if random:
        assert isinstance(fx, tuple), "if random=True, fx is tuple."
        assert isinstance(fy, tuple), "if random=True, fy is tuple."
        assert fx[0] > 0 and fx[1] > fx[0], "fx[0] > 0 and fx[1] > fx[0]."
        assert fy[0] > 0 and fy[1] > fy[0], "fy[0] > 0 and fy[1] > fy[0]."
        if np.random.random() <= p:
            fx = np.random.uniform(fx[0], fx[1] + 1e-6)
            fy = np.random.uniform(fy[0], fy[1] + 1e-6)
            img = cv2.resize(img, None, fx=fx, fy=fy)
            return img
        else:
            return img
    else:
        assert isinstance(fx, float), "if random=False, fx is float."
        assert isinstance(fy, float), "if random=False, fy is float."
        assert fx > 0 and fy > 0, "fx > 0 and fy > 0."
        img = cv2.resize(img, None, fx=fx, fy=fy)
        return img

    
def resize(img, random=False, p=1, dsz=(1920, 1080), r=(0.01, 2.0), interpolation=cv2.INTER_LINEAR):
    if random:
        assert isinstance(r, tuple), "if random=True, r is tuple."
        if np.random.random() <= p:
            imgsz = img.shape[:2]
            rx = np.random.uniform(r[0], r[1])
            ry = np.random.uniform(r[0], r[1])
            dsz = (int(imgsz[1] * rx), int(imgsz[0] * ry))
            if dsz[0] <= 0: dsz[0] = 1
            if dsz[1] <= 0: dsz[1] = 1
            img = cv2.resize(img, dsz, interpolation=interpolation)
            return img
        else:
            return img
    else:
        assert isinstance(dsz, tuple), "if random=False, dsz is tuple."
        img = cv2.resize(img, dsz, interpolation=interpolation)
        return img


def equalize_hist(img, random=False, p=1, m=0):
    assert m in [0, 1], "m should be one of [0, 1]"

    if random:
        if np.random.random() <= p:
            m = np.random.choice([0, 1])
            if m == 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.equalizeHist(img)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                b, g, r = cv2.split(img)
                B = cv2.equalizeHist(b)
                G = cv2.equalizeHist(g)
                R = cv2.equalizeHist(r)
                img = cv2.merge([B, G, R])
            return img
        else:
            return img
    else:
        if m == 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            b, g, r = cv2.split(img)
            B = cv2.equalizeHist(b)
            G = cv2.equalizeHist(g)
            R = cv2.equalizeHist(r)
            img = cv2.merge([B, G, R])
        
        return img


def change_brightness(img, random=False, p=1, value=30):
    if random:
        assert isinstance(value, tuple), "if random=True, value is tuple."
        if np.random.random() <= p:
            brightness_value = np.random.randint(value[0], value[1] + 1)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, brightness_value)
            v[v > 255] = 255
            v[v < 0] = 0
            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            return img
        else:
            return img
    else:
        assert isinstance(value, int), "if random=False, value is int."
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        
        return img


def change_brightness_opencv_official(img, alpha=1.0, beta=0):
    """
    https://docs.opencv2.org/4.5.3/d3/dc1/tutorial_basic_linear_transform.html
    Parameters
    ----------
    img
    alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    beta = int(input('* Enter the beta value [0-100]: '))
    Returns
    -------

    """
    new_image = np.zeros(img.shape, img.dtype)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)

    return new_image


def gamma_correction(img, random=False, p=1, value=(0.4, 1.7)):
    if random:
        assert isinstance(value, tuple), "If random=True, value should be tuple!"
        if np.random.random() <= p:
            value = np.random.uniform(value[0], value[1])
            lookUpTable = np.empty((1, 256), np.uint8)
            for i in range(256):
                lookUpTable[0, i] = np.clip(pow(i / 255.0, value) * 255.0, 0, 255)
            img = cv2.LUT(img, lookUpTable)
            return img
        else:
            return img
    else:
        assert isinstance(value, float), "If random=False, value should be float!"
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, value) * 255.0, 0, 255)
        img = cv2.LUT(img, lookUpTable)
        
        return img
    

def gamma_transformation(img, gamma=0.8):
    # Apply Gamma=0.4 on the normalised image and then multiply by scaling constant (For 8 bit, c=255)
    gamma_res = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    return gamma_res


def gamma_correction_auto(img, method=2):
    """
    https://stackoverflow.com/questions/61695773/how-to-set-the-best-value-for-gamma-correction

    Here are two ways to do that in Python/Opencv2. Both are based upon the ratio of the log(mid-gray)/log(mean).
    Results are often reasonable, especially for dark image, but do not work in all cases. For bright image,
    invert the gray or value image, process as for dark images, then invert again and recombine if using the value image.

    Read the input
    Convert to gray or HSV value
    Compute the ratio log(mid-gray)/log(mean) on the gray or value channel
    Raise the input or value to the power of the ratio
    If using the value channel, combine the new value channel with the hue and saturation channels and convert back to RGB

    :param img:
    :return:
    """

    if method == 1:
        if len(img.shape) == 2:
            gray = img
            # compute gamma = log(mid*255)/log(mean)
            mid = 0.5
            mean = np.mean(gray)
            gamma = math.log(mid * 255) / math.log(mean)
            print("gamma: ", gamma)

            imgbgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # do gamma correction
            img_gamma1 = np.power(imgbgr, gamma).clip(0, 255).astype(np.uint8)
            return img_gamma1, gamma
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # compute gamma = log(mid*255)/log(mean)
            mid = 0.5
            mean = np.mean(gray)
            gamma = math.log(mid * 255) / math.log(mean)
            print("gamma: ", gamma)

            # do gamma correction
            img_gamma1 = np.power(img, gamma).clip(0, 255).astype(np.uint8)
            return img_gamma1, gamma
    elif method == 2:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(hsv)

            # compute gamma = log(mid*255)/log(mean)
            mid = 0.5
            mean = np.mean(val)
            gamma = math.log(mid * 255) / math.log(mean)
            print("gamma: ", gamma)

            # do gamma correction on value channel
            val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)

            # combine new value channel with original hue and sat channels
            hsv_gamma = cv2.merge([hue, sat, val_gamma])
            img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
            return img_gamma2, gamma
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(hsv)

            # compute gamma = log(mid*255)/log(mean)
            mid = 0.5
            mean = np.mean(val)
            gamma = math.log(mid * 255) / math.log(mean)
            print("gamma: ", gamma)

            # do gamma correction on value channel
            val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)

            # combine new value channel with original hue and sat channels
            hsv_gamma = cv2.merge([hue, sat, val_gamma])
            img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
            return img_gamma2, gamma
    else:
        print("Method should be 1 or 2!")
        return None
    

def gaussian_noise(img, random=False, p=1, mean=0, var=0.25):
    """
    Examples
        # --------
        # Draw samples from the distribution:
        #
        # >>> mu, sigma = 0, 0.1 # mean and standard deviation
        # >>> s = np.random.normal(mu, sigma, 1000)
        #
        # Verify the mean and the variance:
        #
        # >>> abs(mu - np.mean(s))
        # 0.0  # may vary
        #
        # >>> abs(sigma - np.std(s, ddof=1))
        # 0.1  # may vary
        #
        # Display the histogram of the samples, along with
        # the probability density function:
        #
        # >>> import matplotlib.pyplot as plt
        # >>> count, bins, ignored = plt.hist(s, 30, density=True)
        # >>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        # ...                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        # ...          linewidth=2, color='r')
        # >>> plt.show()
    Parameters
    ----------
    img

    Returns
    -------

    """
    # 生成高斯噪声
    assert var >= 0.1 and var <= 100, "var >= 0.1 and var <= 100!"
    if random:
        if np.random.random() <= p:
            var = np.random.uniform(0.1, var)
            mu, sigma = mean, var ** 0.5
            gaussian = np.random.normal(mu, sigma, img.shape).astype('uint8')
            img = cv2.add(img, gaussian)
            return img
        else:
            return img
    else:
        mu, sigma = mean, var ** 0.5
        gaussian = np.random.normal(mu, sigma, img.shape).astype('uint8')
        img = cv2.add(img, gaussian)
        return img


def poisson_noise(img, random=False, p=1):
    if random:
        if np.random.random() <= p:
            vals = len(np.unique(img))
            vals = 2 ** np.ceil(np.log2(vals))
            poisson = (np.random.poisson(img * vals) / float(vals)).astype('uint8')
            img = cv2.add(img, poisson)
            return img
        else:
            return img
    else:
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        poisson = (np.random.poisson(img * vals) / float(vals)).astype('uint8')
        img = cv2.add(img, poisson)
        return img


def sp_noise(img, random=False, p=1, salt_p=0.01, pepper_p=0.01):
    """
    salt and pepper noise
    """
    assert salt_p >= 0 and salt_p <= 1, "salt_p >= 0 and salt_p <= 1!"
    assert pepper_p >= 0 and pepper_p <= 1, "salt_p >= 0 and salt_p <= 1!"
    if random:
        if np.random.random() <= p:
            salt_p = np.random.uniform(0.0, salt_p)
            pepper_p = np.random.uniform(0.0, pepper_p)

            noisy_image = np.copy(img)
            total_pixels = img.shape[0] * img.shape[1]  # 计算图像的总像素数

            num_salt = int(total_pixels * salt_p)  # 通过将总像素数与指定的椒盐噪声比例相乘,得到要添加的椒盐噪声的数量。
            salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
            noisy_image[salt_coords[0], salt_coords[1]] = 255

            num_pepper = int(total_pixels * pepper_p)
            pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
            noisy_image[pepper_coords[0], pepper_coords[1]] = 0

            return noisy_image
        else:
            return img
    else:
        noisy_image = np.copy(img)
        total_pixels = img.shape[0] * img.shape[1]  # 计算图像的总像素数

        num_salt = int(total_pixels * salt_p)  # 通过将总像素数与指定的椒盐噪声比例相乘,得到要添加的椒盐噪声的数量。
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
        noisy_image[salt_coords[0], salt_coords[1]] = 255

        num_pepper = int(total_pixels * pepper_p)
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_image


def make_sunlight_effect(img, random=False, p=1, center=(50, 50), effect_r=(50, 200), light_strength=150):
    if random:
        assert isinstance(effect_r, tuple), "If random=True, effect_r should be tuple!"
        if np.random.random() <= p:
            imgsz = img.shape[:2]
            center = (np.random.randint(0, imgsz[1]), np.random.randint(0, imgsz[0]))
            effectR = np.random.randint(effect_r[0], effect_r[1])
            lightStrength = np.random.randint(0, light_strength)

            dst = np.zeros(shape=img.shape, dtype=np.uint8)

            for i in range(imgsz[0]):
                for j in range(imgsz[1]):
                    dis = (center[0] - j) ** 2 + (center[1] - i) ** 2
                    B, G, R = img[i, j][0], img[i, j][1], img[i, j][2]
                    if dis < effectR * effectR:
                        result = int(lightStrength * (1.0 - np.sqrt(dis) / effectR))
                        B += result
                        G += result
                        R += result

                        B, G, R = min(max(0, B), 255), min(max(0, G), 255), min(max(0, R), 255)
                        dst[i, j] = np.uint8((B, G, R))
                    else:
                        dst[i, j] = np.uint8((B, G, R))
            return dst
        else:
            return img
    else:
        assert isinstance(effect_r, int), "If random=False, effect_r should be int!"
        imgsz = img.shape[:2]
        dst = np.zeros(shape=img.shape, dtype=np.uint8)

        for i in range(imgsz[0]):
            for j in range(imgsz[1]):
                dis = (center[0] - j) ** 2 + (center[1] - i) ** 2
                B, G, R = img[i, j][0], img[i, j][1], img[i, j][2]
                if dis < effect_r * effect_r:
                    result = int(light_strength * (1.0 - np.sqrt(dis) / effect_r))
                    B += result
                    G += result
                    R += result

                    B, G, R = min(max(0, B), 255), min(max(0, G), 255), min(max(0, R), 255)
                    dst[i, j] = np.uint8((B, G, R))
                else:
                    dst[i, j] = np.uint8((B, G, R))
        return dst

        
def color_distortion(img, random=False, p=1, value=(-50, 50)):
    """
    TODO: PIL format
    col = ImageEnhance.Color(img)
    res = col.enhance(random.uniform(lower, upper))

    def random_jitter(image):
        # 对图像进行颜色抖动
        # :param image: PIL的图像image
        # :return: 有颜色色差的图像image

        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    def random_sharpness(img, p=0.5, lower=0.5, upper=1.5):
        assert upper >= lower, "upper must be >= lower."
        assert lower >= 0, "lower must be non-negative."
        if np.random.random() < p:
            img = getpilimage(img)
            sha = ImageEnhance.Sharpness(img)
            return sha.enhance(random.uniform(lower, upper))
        else:
            return img
            
    """
    if random:
        assert isinstance(value, tuple), "If random=True, value should be tuple!"
        if np.random.random() <= p:
            hue_v = np.random.randint(value[0], value[1])
            hsv_image = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_v) % 180  # 在Hue通道上增加30
            hsv_image = np.clip(hsv_image, 0, 255)
            img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            return img
        else:
            return img
    else:
        assert isinstance(value, int), "If random=False, value should be int!"
        hsv_image = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + value) % 180  # 在Hue通道上增加30
        hsv_image = np.clip(hsv_image, 0, 255)
        img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return img


def change_contrast_and_brightness(img, random=False, p=1, alpha=0.5, beta=30):
    """
    # 使用公式f(x)=α.g(x)+β, α调节对比度, β调节亮度
    # 小心使用
    # TODO: PIL format
    # con = ImageEnhance.Contrast(img)
    # res = con.enhance(random.uniform(lower, upper))
    # 
    # bri = ImageEnhance.Brightness(img)
    # res = bri.enhance(random.uniform(lower, upper))
    """
    assert alpha >= 0 and alpha <= 1, "alpha >= 0 and alpha <= 1"
    if random:
        # 容易变黑图,不太建议使用这个
        if np.random.random() <= p:
            alpha = np.random.uniform(0, alpha)
            beta = np.random.uniform(-beta, beta + 1)
            blank = np.zeros(img.shape, img.dtype)  # 创建图片类型的零矩阵
            img = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)  # 图像混合加权
            return img
        else:
            return img
    else:
        blank = np.zeros(img.shape, img.dtype)  # 创建图片类型的零矩阵
        img = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)  # 图像混合加权
        return img


def contrast_stretch(img, random=False, p=1, alpha=0, beta=1):
    """
    对比拉伸, 设置好alpha和beta效果比较理想
    """
    if random:
        assert isinstance(alpha, tuple), "If random=True, alpha should be tuple!"
        assert isinstance(beta, tuple), "If random=True, beta should be tuple!"
        alpha = np.random.uniform(alpha[0], alpha[1])
        beta = np.random.uniform(beta[0], beta[1])

        if np.random.random() <= p:
            norm_img1 = cv2.normalize(img, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            norm_img = (255 * norm_img1).astype(np.uint8)
            return norm_img
        else:
            return img
    else:
        assert isinstance(alpha, float), "If random=False, alpha should be float!"
        assert isinstance(beta, float), "If random=False, beta should be float!"

        norm_img1 = cv2.normalize(img, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img = (255 * norm_img1).astype(np.uint8)
        return norm_img


def clahe(img, random=False, p=1, m=0, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    直方图适应均衡化
    该函数包含以下参数:
    clipLimit: 用于控制直方图均衡化的局部对比度,值越高,越容易出现失真和噪声。建议值为2-4,若使用默认值0则表示自动计算。
    tileGridSize: 表示每个块的大小,推荐16x16。
    tileGridSize.width: 块的宽度。
    tileGridSize.height: 块的高度。
    函数返回一个CLAHE对象,可以通过该对象调用apply函数来实现直方图均衡化。
    """
    assert m in [0, 1], "m should be one of [0, 1]!"
    if random:
        if np.random.random() <= p:
            clipLimit = np.random.randint(0, 40 + 1)
            tgs = np.random.choice([4, 8, 16, 32])
            tileGridSize = (tgs, tgs)
            if m == 0:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
                res = clahe.apply(img)
                img = cv2.merge([res, res, res])
            else:
                b, g, r = cv2.split(img.astype(np.uint8))
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
                clahe_b = clahe.apply(b)
                clahe_g = clahe.apply(g)
                clahe_r = clahe.apply(r)
                img = cv2.merge([clahe_b, clahe_g, clahe_r])

            return img
        else:
            return img
    else:
        if m == 0:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            res = clahe.apply(img)
            img = cv2.merge([res, res, res])
        else:
            b, g, r = cv2.split(img.astype(np.uint8))
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            clahe_b = clahe.apply(b)
            clahe_g = clahe.apply(g)
            clahe_r = clahe.apply(r)
            img = cv2.merge([clahe_b, clahe_g, clahe_r])
        return img


def change_hsv(img, random=False, p=1, hgain=0.5, sgain=0.5, vgain=0.5):
    if random:
        if np.random.random() <= p:
            img = img.astype(np.uint8)
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
            img = img.astype(np.float32)
            return img
        else:
            return img
    else:
        img = img.astype(np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        img = img.astype(np.float32)
        return img


def gaussian_blur(img, random=False, p=1, k=3):
    if random:
        if np.random.random() <= p:
            h, w, _ = img.shape
            ks = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
            if h > 16 and w > 16:
                if h <= 128 and w <= 128:
                    k = np.random.choice(ks[:3])
                elif h <= 256 and w <= 256:
                    k = np.random.choice(ks[:8])
                elif h <= 512 and w <= 512:
                    k = np.random.choice(ks[:11])
                else:
                    k = np.random.choice(ks[5:])
                img = cv2.GaussianBlur(img, (k, k), 1)

            return img
        else:
            return img
    else:
        h, w, _ = img.shape
        img = cv2.GaussianBlur(img, (k, k), 1)
        return img


def motion_blur(img, random=False, p=1, k=3, angle=30):
    """
    假如用于增强OCR数据, 则k不宜太大!
    """
    if random:
        if np.random.random() <= p:
            angle = np.random.randint(-180, 181)
            imgsz = img.shape[:2]
            # ks = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
            # ks = [3, 5, 7, 9, 11, 13, 15, 17]
            ks = [3, 5, 7, 9]
            if imgsz[0] > 16 and imgsz[1] > 16:
                if imgsz[0] <= 128 and imgsz[1] <= 128:
                    k = np.random.choice(ks[:3])
                elif imgsz[0] <= 256 and imgsz[1] <= 256:
                    k = np.random.choice(ks[:8])
                elif imgsz[0] <= 512 and imgsz[1] <= 512:
                    k = np.random.choice(ks[:11])
                else:
                    k = np.random.choice(ks)

            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[(k - 1) // 2, :] = np.ones(k, dtype=np.float32)
            m =  cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
            kernel = cv2.warpAffine(kernel, m, (k, k))
            kernel = kernel * (1.0 / np.sum(kernel))
            img = cv2.filter2D(img, -1, kernel)
            return img
        else:
            return img
    else:
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[(k - 1) // 2, :] = np.ones(k, dtype=np.float32)
        m =  cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, m, (k, k))
        kernel = kernel * (1.0 / np.sum(kernel))
        img = cv2.filter2D(img, -1, kernel)
        
        return img


def median_blur(img, random=False, p=1, k=3):
    if random:
        if np.random.random() <= p:
            ks = [3, 5, 7, 9]
            k = np.random.choice(ks)
            img = cv2.medianBlur(img, k)
            return img
        else:
            return img
    else:
        img = cv2.medianBlur(img, k)

        return img


def dilation_erosion(img, random=False, p=1, flag="dilate", scale=(6, 8)):
    """
    dilate, erode
    """
    assert flag in ["dilate", "erode"], 'flag should be one of ["dilate", "erode"]!'
    if random:
        if np.random.random() <= p:
            imgsz = img.shape[:2]
            if imgsz[0] > 1024:
                scale = (10, 12)
            elif imgsz[0] > 512:
                scale = (8, 10)
            elif imgsz[0] > 256:
                scale = (6, 8)
            elif imgsz[0] > 128:
                scale = (4, 6)
            elif imgsz[0] > 32:
                scale = (2, 3)
            else:
                return img

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, tuple(np.random.randint(scale[0], scale[1], 2))
            )
            if flag == "dilate":
                img = cv2.dilate(img, kernel, iterations=1)
            else:
                img = cv2.erode(img, kernel, iterations=1)
            return img
        else:
            return img
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(scale[0], scale[1], 2))
        )
        if flag == "dilate":
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)
        return img


# Rain effect --------------------------------------------------------------------------------
def rain_noise(img, value=10):
    '''
    #生成噪声图像
    >>> 输入: img图像

        value= 大小控制雨滴的多少
    >>> 返回图像大小的模糊噪声图像
    '''

    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平,取浮点数,只保留最大的一部分作为噪声
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    # 可以输出噪声看看
    '''
    cv2.imshow('img',noise)
    cv2.waitKey()
    cv2.destroyWindow('img')
    '''
    return noise


def rain_blur(noise, length=10, angle=0, w=1):
    '''
    将噪声加上运动模糊,模仿雨滴

    >>>输入
    noise:输入噪声图,shape = img.shape[0:2]
    length: 对角矩阵大小,表示雨滴的长度
    angle: 倾斜的角度,逆时针为正
    w:      雨滴大小

    >>>输出带模糊的噪声

    '''

    # 这里由于对角阵自带45度的倾斜,逆时针为正,所以加了-45度的误差,保证开始为正
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle + 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # 生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
    k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核,使得雨有宽度

    # k = k / length                         #是否归一化

    blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核,进行滤波

    # 转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    '''
    cv2.imshow('img',blurred)
    cv2.waitKey()
    cv2.destroyWindow('img')
    '''

    return blurred


def alpha_rain(rain, img, beta=0.8):
    # 输入雨滴噪声和图像
    # beta = 0.8   #results weight
    # 显示下雨效果

    # expand dimensin
    # 将二维雨噪声扩张为三维单通道
    # 并与图像合成在一起形成带有alpha通道的4通道图像
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel

    rain_result = img.copy()  # 拷贝一个掩膜
    rain = np.array(rain, dtype=np.float32)  # 数据类型变为浮点数,后面要叠加,防止数组越界要用32位
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    # 对每个通道先保留雨滴噪声图对应的黑色(透明)部分,再叠加白色的雨滴噪声部分(有比例因子)

    """
    cv2.imshow('rain_effct_result', rain_result)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """
    return rain_result


def add_rain(rain, img, alpha=0.9):
    # 输入雨滴噪声和图像
    # alpha:原图比例因子
    # 显示下雨效果

    # chage rain into  3-dimenis
    # 将二维rain噪声扩张为与原图相同的三通道图像
    rain = np.expand_dims(rain, 2)
    rain = np.repeat(rain, 3, 2)

    # 加权合成新图
    result = cv2.addWeighted(img, alpha, rain, 1 - alpha, 1)

    """
    cv2.imshow('rain_effect', result)
    cv2.waitKey()
    cv2.destroyWindow('rain_effect')
    """
    return result


def make_rain_effect(img, random=False, p=1, m=0, length=(10, 80), angle=(-45, 46), noise=(100, 500)):
    assert m in [0, 1], "m should be one of [0, 1]!"
    if random:
        assert isinstance(length, tuple), "If random=True, length should be tuple!"
        assert isinstance(angle, tuple), "If random=True, angle should be tuple!"
        assert isinstance(noise, tuple), "If random=True, noise should be tuple!"
        if np.random.random() <= p:
            rain_length = np.random.randint(length[0], length[1] + 1)
            rain_angle = np.random.randint(angle[0], angle[1] + 1)
            noise_value = np.random.randint(noise[0], noise[1] + 1)
            rain_w = np.random.choice([1, 3, 5])
            noise = rain_noise(img, value=noise_value)
            rain = rain_blur(noise, length=rain_length, angle=rain_angle, w=rain_w)

            if m == 0:
                rain_beta = 0.1 * np.random.randint(4, 8)
                img = alpha_rain(rain, img, beta=rain_beta)  # 方法一,透明度赋值
            else:
                rain_alpha = 0.1 * np.random.randint(7, 10)
                img = add_rain(rain, img, alpha=rain_alpha)  # 方法二, 加权后有玻璃外的效果
            return img
        else:
            return img
    else:
        assert isinstance(length, int), "If random=False, length should be int!"
        assert isinstance(angle, int), "If random=False, length should be int!"
        assert isinstance(noise, int), "If random=False, length should be int!"

        rain_w = np.random.choice([1, 3, 5])
        noise = rain_noise(img, value=noise)
        rain = rain_blur(noise, length=length, angle=angle, w=rain_w)

        if m == 0:
            rain_beta = 0.1 * np.random.randint(4, 8)
            img = alpha_rain(rain, img, beta=rain_beta)  # 方法一,透明度赋值
        else:
            rain_alpha = 0.1 * np.random.randint(7, 10)
            img = add_rain(rain, img, alpha=rain_alpha)  # 方法二, 加权后有玻璃外的效果
        return img
# Rain effect --------------------------------------------------------------------------------


def compress(img, random=False, p=1, quality=(25, 90)):
    """
    like change_definition
    """
    if random:
        assert isinstance(quality, tuple), "If random=True, quality should be tuple!"
        if np.random.random() <= p:
            q = np.random.randint(quality[0], quality[1])
            param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            img_encode = cv2.imencode('.jpeg', img, param)
            img_decode = cv2.imdecode(img_encode[1], cv2.IMREAD_COLOR)
            return img_decode
        else:
            return img
    else:
        assert isinstance(quality, int), "If random=False, quality should be int!"
        param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        img_encode = cv2.imencode('.jpeg', img, param)
        img_decode = cv2.imdecode(img_encode[1], cv2.IMREAD_COLOR)
        return img_decode
    

def change_definition(img, random=False, p=1, r=(0.5, 0.95)):
    """
    like compress
    """
    if random:
        assert isinstance(r, tuple), "If random=True, r should be tuple!"
        if np.random.random() <= p:
            h, w = img.shape[:2]
            rate = np.random.uniform(r[0], r[1])
            w2 = int(w * rate)
            h2 = int(h * rate)
            img = cv2.resize(img, (w2, h2))
            img = cv2.resize(img, (w, h))
            return img
        else:
            return img
    else:
        assert isinstance(r, float), "If random=False, r should be float!"
        h, w = img.shape[:2]
        w2 = int(w * r)
        h2 = int(h * r)
        img = cv2.resize(img, (w2, h2))
        img = cv2.resize(img, (w, h))
        return img


def exposure(img, random=False, p=1, rect=(50, 50, 100, 100)):
    # 目前有问题, 2024.11.13
    if random:
        if np.random.random() <= p:
            h, w = img.shape[:2]
            x0 = random.randint(0, w)
            y0 = random.randint(0, h)
            x1 = random.randint(x0, w)
            y1 = random.randint(y0, h)
            area = (x0, y0, x1, y1)
            mask = Image.new('L', (w, h), color=255)
            draw = ImageDraw.Draw(mask)
            mask = np.array(mask)
            if len(img.shape) == 3:
                mask = mask[:, :, np.newaxis]
                mask = np.concatenate([mask, mask, mask], axis=2)
            draw.rectangle(area, fill=np.random.randint(150, 255))
            res = img + (255 - mask)
            res = np.clip(res, 0, 255)
            return res
        else:
            return img
    else:
        h, w = img.shape[:2]
        mask = Image.new('L', (w, h), color=255)
        draw = ImageDraw.Draw(mask)
        mask = np.array(mask)
        if len(img.shape) == 3:
            mask = mask[:, :, np.newaxis]
            mask = np.concatenate([mask, mask, mask], axis=2)
        draw.rectangle(rect, fill=np.random.randint(150, 255))
        res = img + (255 - mask)
        res = np.clip(res, 0, 255)
        return res


def stretch(img, random=False, p=1, r=(0.8, 1.2)):
    if random:
        assert isinstance(r, tuple), "If random=True, r should be tuple!"
        if np.random.random() <= p:
            h, w = img.shape[:2]
            rate = np.random.uniform(r[0], r[1])
            w2 = int(w * rate)
            h2 = int(h * rate)
            if np.random.random() <= 0.5:
                img = cv2.resize(img, (w2, h))
            else:
                img = cv2.resize(img, (w, h2))
            return img
        else:
            return img
    else:
        assert isinstance(r, float), "If random=False, r should be float!"
        h, w = img.shape[:2]
        w2 = int(w * r)
        h2 = int(h * r)
        if np.random.random() <= 0.5:
            img = cv2.resize(img, (w2, h))
        else:
            img = cv2.resize(img, (w, h2))
        return img


def crop(img, random=False, p=1, fix_size=False, crop_size=(256, 256), min_size=(64, 64), rect=(0, 0, 100, 200)):
    # crop_size: [H, W]
    if random:
        if np.random.random() <= p:
            imgsz = img.shape[:2]
            assert crop_size[0] >= 0 and crop_size[0] <= imgsz[0], "crop_size[0] < 0 or crop_size[0] > imgsz[0]"
            assert crop_size[1] >= 0 and crop_size[1] <= imgsz[1], "crop_size[1] < 0 or crop_size[1] > imgsz[1]"

            if not fix_size:
                crop_size_h = np.random.randint(min_size[0], crop_size[0])
                crop_size_w = np.random.randint(min_size[1], crop_size[1])
                crop_size = (crop_size_h, crop_size_w)
                
            x = np.random.randint(0, imgsz[1] - crop_size[1])
            y = np.random.randint(0, imgsz[0] - crop_size[0])

            try:
                cropped_img = img[y:(y + crop_size[0]), x:(x + crop_size[1])]
            except Exception as Error:
                print(Error)
                return None
            
            return cropped_img
        else:
            return img
    else:
        imgsz = img.shape[:2]
        assert rect[0] >= 0 and rect[0] <= imgsz[1], "rect[0] >= 0 and rect[0] <= imgsz[1]"
        assert rect[1] >= 0 and rect[1] <= imgsz[0], "rect[1] >= 0 and rect[1] <= imgsz[0]"
        cropped_img = img[rect[1]:rect[3], rect[0]:rect[2]]
        
        return cropped_img


def make_mask(img, random=False, p=1, fix_size=False, mask_size=(256, 256), min_size=(64, 64), rect=(0, 0, 100, 200), color=(255, 0, 255)):
    """
    like transperent_overlay
    """
    imgcp = copy.copy(img)
    if random:
        if np.random.random() <= p:
            # 在图像上随机生成一个矩形遮挡,遮挡的位置和大小都是随机生成的。遮挡的颜色也是随机选择的
            # 生成随机遮挡位置和大小
            imgsz = imgcp.shape[:2]

            if not fix_size:
                mask_size_x = np.random.randint(min_size[1], mask_size[1])
                mask_size_y = np.random.randint(min_size[0], mask_size[0])
                mask_size = (mask_size_x, mask_size_y)

            mask_x = np.random.randint(0, imgsz[1] - mask_size[1])
            mask_y = np.random.randint(0, imgsz[0] - mask_size[0])

            # 生成随机颜色的遮挡
            mask_color = np.random.randint(0, 256, (1, 1, 3))
            imgcp[mask_y:mask_y + mask_size[0], mask_x:mask_x + mask_size[1]] = mask_color
            return imgcp
        else:
            return imgcp
    else:
        imgsz = imgcp.shape[:2]
        assert rect[0] >= 0 and rect[2] <= imgsz[1], "rect[0] >= 0 and rect[2] <= imgsz[1]"
        assert rect[1] >= 0 and rect[3] <= imgsz[0], "rect[1] >= 0 and rect[3] <= imgsz[0]"

        imgcp[rect[1]:rect[3], rect[0]:rect[2]] = color
        return imgcp
    

def transperent_overlay(img, random=False, p=1, rect=(50, 50, 100, 80), max_h_r=1.0, max_w_r=0.25):
    """
    rect: [x1, y1, x2, y2]
    """
    if random:
        if np.random.random() <= p:
            imgsz = img.shape
            orig_c = imgsz[2]
            max_h = int(imgsz[0] * max_h_r)
            max_w = int(imgsz[1] * max_w_r)

            alpha = 0.1 * np.random.randint(1, 5)

            x = np.random.randint(0, max(imgsz[1] - max_w, 1))
            y = np.random.randint(0, max(imgsz[0] - max_h, 1))
            bw = np.random.randint(0, max(max_w, 1))
            bh = np.random.randint(0, max(max_h, 1))
            color = [np.random.randint(0, 256) for _ in range(3)]

            if imgsz[2] < 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            # 创建一个与图片大小相同的覆盖层
            # overlay = img.copy()
            overlay = np.ones(shape=img.shape, dtype=np.uint8)
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, -1)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # Convert the image back to the original number of channels
            if orig_c != img.shape[2]:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        else:
            return img
    else:
        imgsz = img.shape
        orig_c = imgsz[2]
        alpha = 0.1 * np.random.randint(1, 5)
        color = [np.random.randint(0, 256) for _ in range(3)]

        if imgsz[2] < 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # 创建一个与图片大小相同的覆盖层
        # overlay = img.copy()
        overlay = np.ones(shape=img.shape, dtype=np.uint8)
        x1, y1 = rect[0], rect[1]
        x2, y2 = rect[2], rect[3]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Convert the image back to the original number of channels
        if orig_c != img.shape[2]:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def squeeze(img, random=False, p=1, center=(5, 50), degree=11):
    """
    产生向中心点挤压的效果。效果不太好,速度也慢,谨慎使用!
    """
    if random:
        assert isinstance(degree, tuple), "If random=True, degree should be tuple!"
        if np.random.random() <= p:
            imgsz = img.shape
            center_x = np.random.randint(0, imgsz[1])
            center_y = np.random.randint(0, imgsz[0])
            center = (center_x, center_y)
            degree = np.random.randint(degree[0], degree[1])
            new_data = img.copy()
            for i in range(imgsz[1]):
                for j in range(imgsz[0]):
                    tx = i - center[0]
                    ty = j - center[1]
                    theta = math.atan2(ty, tx)
                    # 半径
                    radius = math.sqrt(tx ** 2 + ty ** 2)
                    radius = math.sqrt(radius) * degree
                    new_x = int(center[0] + radius * math.cos(theta))
                    new_y = int(center[1] + radius * math.sin(theta))
                    if new_x < 0:
                        new_x = 0
                    if new_x >= imgsz[1]:
                        new_x = imgsz[1] - 1
                    if new_y < 0:
                        new_y = 0
                    if new_y >= imgsz[0]:
                        new_y = imgsz[0] - 1

                    for c in range(imgsz[2]):
                        new_data[j][i][c] = img[new_y][new_x][c]
            return new_data
        else:
            return img
    else:
        assert isinstance(degree, int), "If random=False, degree should be float!"
        imgsz = img.shape
        new_data = img.copy()
        for i in range(imgsz[1]):
            for j in range(imgsz[0]):
                tx = i - center[0]
                ty = j - center[1]
                theta = math.atan2(ty, tx)
                # 半径
                radius = math.sqrt(tx ** 2 + ty ** 2)
                radius = math.sqrt(radius) * degree
                new_x = int(center[0] + radius * math.cos(theta))
                new_y = int(center[1] + radius * math.sin(theta))
                if new_x < 0:
                    new_x = 0
                if new_x >= imgsz[1]:
                    new_x = imgsz[1] - 1
                if new_y < 0:
                    new_y = 0
                if new_y >= imgsz[0]:
                    new_y = imgsz[0] - 1

                for c in range(imgsz[2]):
                    new_data[j][i][c] = img[new_y][new_x][c]
        return new_data
    

def make_haha_mirror_effect(img, random=False, p=1, center=(50, 50), r=40, degree=4):
    """
    效果不太好,速度也慢,谨慎使用!
    """
    if random:
        assert isinstance(r, tuple), "If random=False, r should be tuple!"
        assert isinstance(degree, tuple), "If random=False, degree should be tuple!"

        if np.random.random() <= p:
            height, width, n = img.shape
            center = (np.random.randint(0, width), np.random.randint(0, height))
            r = np.random.randint(r[0], r[1])
            degree = np.random.randint(degree[0], degree[1])
            randius = r * degree  # 直径
            real_randius = int(randius / 2)  # 半径
            new_data = img.copy()
            for i in range(width):
                for j in range(height):
                    tx = i - center[0]
                    ty = j - center[1]
                    distance = tx ** 2 + tx ** 2
                    # 为了保证选择的像素是图片上的像素
                    if distance < randius ** 2:
                        new_x = tx / 2
                        new_y = ty / 2
                        # 图片的每个像素的坐标按照原来distance 之后的distance(real_randius**2)占比放大即可
                        new_x = int(new_x * math.sqrt(distance) / real_randius + center[0])
                        new_y = int(new_y * math.sqrt(distance) / real_randius + center[1])
                        # 当不超过new_data 的边界时候就可赋值
                        if new_x < width and new_y < height:
                            new_data[j][i][0] = img[new_y][new_x][0]
                            new_data[j][i][1] = img[new_y][new_x][1]
                            new_data[j][i][2] = img[new_y][new_x][2]
            return new_data
        else:
            return img
    else:
        assert isinstance(r, int), "If random=False, r should be int!"
        assert isinstance(degree, int), "If random=False, degree should be int!"

        height, width, n = img.shape
        randius = r * degree  # 直径
        real_randius = int(randius / 2)  # 半径
        new_data = img.copy()
        for i in range(width):
            for j in range(height):
                tx = i - center[0]
                ty = j - center[1]
                distance = tx ** 2 + tx ** 2
                # 为了保证选择的像素是图片上的像素
                if distance < randius ** 2:
                    new_x = tx / 2
                    new_y = ty / 2
                    # 图片的每个像素的坐标按照原来distance 之后的distance(real_randius**2)占比放大即可
                    new_x = int(new_x * math.sqrt(distance) / real_randius + center[0])
                    new_y = int(new_y * math.sqrt(distance) / real_randius + center[1])
                    # 当不超过new_data 的边界时候就可赋值
                    if new_x < width and new_y < height:
                        new_data[j][i][0] = img[new_y][new_x][0]
                        new_data[j][i][1] = img[new_y][new_x][1]
                        new_data[j][i][2] = img[new_y][new_x][2]
        return new_data

    
def warp_img(img, random=False, p=1, degree=4):
    """
    产生上下波浪状扭曲的效果.
    目前发现下底存在遮盖原图的情况, 需要优化! 2024.11.13
    """
    if random:
        assert isinstance(degree, tuple), "If random=False, degree should be tuple!"
        if np.random.random() <= p:
            degree = np.random.randint(degree[0], degree[1])
            height, width, channels = img.shape
            new_data = np.zeros([height, width, 3], np.uint8)  # null img
            for j in range(width):
                temp = degree * math.sin(360 * j / width * math.pi / 180)  # [-degree,degree]
                temp = degree + temp  # [0, 2*degree]
                for i in range(int(temp + 0.5), int(height + temp - 2 * degree)):
                    x = int((i - temp) * height / (height - degree))
                    if x >= height:
                        x = height - 1
                    if x < 0:
                        x = 0
                    for channel in range(channels):
                        new_data[i][j][channel] = img[x][j][channel]
            return new_data
        else:
            return img
    else:
        assert isinstance(degree, int), "If random=False, degree should be int!"
        height, width, channels = img.shape
        new_data = np.zeros([height, width, 3], np.uint8)  # null img
        for j in range(width):
            temp = degree * math.sin(360 * j / width * math.pi / 180)  # [-degree,degree]
            temp = degree + temp  # [0, 2*degree]
            for i in range(int(temp + 0.5), int(height + temp - 2 * degree)):
                x = int((i - temp) * height / (height - degree))
                if x >= height:
                    x = height - 1
                if x < 0:
                    x = 0
                for channel in range(channels):
                    new_data[i][j][channel] = img[x][j][channel]
        return new_data

    
def enhance_gray_value(img, random=False, p=1, gray_range=(0, 255)):
    """
    灰度变换, 通过将像素值映射到新的范围来增强图像的灰度
    看起来好像没什么效果,不建议使用。。。
    """
    if random:
        if np.random.random() <= p:
            img = cv2.convertScaleAbs(img, alpha=(gray_range[1] - gray_range[0]) / 255, beta=gray_range[0])
            return img
        else:
            return img
    else:
        img = cv2.convertScaleAbs(img, alpha=(gray_range[1] - gray_range[0]) / 255, beta=gray_range[0])
        return img
    

def homomorphic_filter(img, random=False, p=1):
    """
    目前程序有问题,需要优化!2024.11.13
    """
    if random:
        if np.random.random() <= p:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 15, 75, 75)
            # 对数变换和傅里叶变换
            H, W = gray.shape[:2]
            gray_log = np.log(gray + 1)
            gray_fft = np.fft.fft2(gray_log)
            # 设置同态滤波器参数
            c, d, gamma_L, gamma_H, gamma_C = 1, 10, 0.2, 2.5, 1
            # 构造同态滤波器
            u, v = np.meshgrid(range(W), range(H))
            Duv = np.sqrt((u - W / 2) ** 2 + (v - H / 2) ** 2)
            Huv = (gamma_H - gamma_L) * (1 - np.exp(-c * (Duv ** 2) / (d ** 2))) + gamma_L
            Huv = Huv * (1 - gamma_C) + gamma_C
            # 进行频域滤波
            gray_fft_filtered = Huv * gray_fft
            gray_filtered = np.fft.ifft2(gray_fft_filtered)
            gray_filtered = np.exp(np.real(gray_filtered)) - 1
            # 转为uint8类型
            gray_filtered = cv2.normalize(gray_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            return gray_filtered
        else:
            return img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 15, 75, 75)
        # 对数变换和傅里叶变换
        H, W = gray.shape[:2]
        gray_log = np.log(gray + 1)
        gray_fft = np.fft.fft2(gray_log)
        # 设置同态滤波器参数
        c, d, gamma_L, gamma_H, gamma_C = 1, 10, 0.2, 2.5, 1
        # 构造同态滤波器
        u, v = np.meshgrid(range(W), range(H))
        Duv = np.sqrt((u - W / 2) ** 2 + (v - H / 2) ** 2)
        Huv = (gamma_H - gamma_L) * (1 - np.exp(-c * (Duv ** 2) / (d ** 2))) + gamma_L
        Huv = Huv * (1 - gamma_C) + gamma_C
        # 进行频域滤波
        gray_fft_filtered = Huv * gray_fft
        gray_filtered = np.fft.ifft2(gray_fft_filtered)
        gray_filtered = np.exp(np.real(gray_filtered)) - 1
        # 转为uint8类型
        gray_filtered = cv2.normalize(gray_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return gray_filtered


def log_transformation(img, random=False, p=1):
    """
    对数变换
    """
    if random:
        if np.random.random() <= p:
            img = np.clip(img, 2, 255)
            c = 255 / np.log(1 + np.max(img))
            log_image = c * (np.log(img))
            # Specify the data type so that
            # float value will be converted to int
            log_image = np.clip(log_image, 0, 255)
            log_image = np.array(log_image, dtype=np.uint8)
            return log_image
        else:
            return img
    else:
        img = np.clip(img, 2, 255)
        c = 255 / np.log(1 + np.max(img))
        log_image = c * (np.log(img))
        # Specify the data type so that
        # float value will be converted to int
        log_image = np.clip(log_image, 0, 255)
        log_image = np.array(log_image, dtype=np.uint8)
        return log_image

        
def translate(img, random=False, p=1, tx=20, ty=30, border_color=(114, 114, 114), dstsz=None):
    if random:
        assert isinstance(tx, tuple), "If random=True, tx should be tuple!"
        assert isinstance(ty, tuple), "If random=True, ty should be tuple!"

        if np.random.random() <= p:
            tx = np.random.randint(tx[0], tx[1])
            ty = np.random.randint(ty[0], ty[1])
            border_color = [np.random.randint(0, 256) for _ in range(3)]
            M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            img = cv2.warpAffine(img, M, dsize=dstsz, borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)
            return img
        else:
            return img
    else:
        assert isinstance(tx, int), "If random=False, tx should be int!"
        assert isinstance(ty, int), "If random=False, ty should be int!"
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        img = cv2.warpAffine(img, M, dsize=dstsz, borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)
        return img


def resize_images(data_path, size=(1920, 1080)):
    dir_name = get_dir_name(data_path)
    img_list = get_file_list(data_path)
    save_path = make_save_path(data_path=data_path, relative=".", add_str="resized")
    os.makedirs(save_path, exist_ok=True)

    for img in img_list:
        img_abs_path = data_path + "/" + img
        img_name = os.path.splitext(img)[0]
        img = cv2.imread(img_abs_path)
        resz_img = cv2.resize(img, size)
        cv2.imwrite("{}/{}.jpg".format(save_path, img_name), resz_img)


# Object detection utils ===================================================

def bbox_voc_to_yolo(size, box):
    """
    VOC --> YOLO
    :param size: [H, W]
    :param box:
    orig: [xmin, xmax, ymin, ymax], deprecated;
    new:  [xmin, ymin, xmax, ymax], 2024.03.29, WJH.
    :return: [x, y, w, h]
    """
    dh = 1. / (size[0])
    dw = 1. / (size[1])
    # x = (box[0] + box[1]) / 2.0
    # y = (box[2] + box[3]) / 2.0
    # w = box[1] - box[0]
    # h = box[3] - box[2]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = int(round(x)) * dw
    w = int(round(w)) * dw
    y = int(round(y)) * dh
    h = int(round(h)) * dh

    if x < 0: x = 0
    if y < 0: y = 0
    if w > 1: w = 1
    if h > 1: h = 1

    return [x, y, w, h]


def bbox_yolo_to_voc(size, bbx):
    """
    YOLO --> VOC
    !!!!!! orig: (bbx, size) 20230329 changed to (size, bbx)
    :param bbx: yolo format bbx
    :param size: [H, W]
    :return: [x_min, y_min, x_max, y_max]
    """
    bbx_ = (bbx[0] * size[1], bbx[1] * size[0], bbx[2] * size[1], bbx[3] * size[0])
    x_min = int(round(bbx_[0] - (bbx_[2] / 2)))
    y_min = int(round(bbx_[1] - (bbx_[3] / 2)))
    x_max = int(round(bbx_[0] + (bbx_[2] / 2)))
    y_max = int(round(bbx_[1] + (bbx_[3] / 2)))

    return [x_min, y_min, x_max, y_max]


def write_labelbee_det_json(bbx, size):
    """
    {"x":316.6583427922815,"y":554.4245175936436,"width":1419.1872871736662,"height":556.1679909194097,
    "attribute":"1","valid":true,"id":"tNd2HY6C","sourceID":"","textAttribute":"","order":1}
    :param bbx: x1, y1, x2, y2
    :param size: H, W
    :return:
    """

    chars = ""
    for i in range(48, 48 + 9):
        chars += chr(i)
    for j in range(65, 65 + 25):
        chars += chr(j)
    for k in range(97, 97 + 25):
        chars += chr(k)

    json_ = {}
    json_["width"] = size[1]
    json_["height"] = size[0]
    json_["valid"] = True
    json_["rotate"] = 0

    step_1_ = {}
    step_1_["toolName"] = "rectTool"

    result_list = []
    for i in range(len(bbx)):
        result_list_dict = {}
        result_list_dict["x"] = bbx[i][0]
        result_list_dict["y"] = bbx[i][1]
        result_list_dict["width"] = bbx[i][2] - bbx[i][0]
        result_list_dict["height"] = bbx[i][3] - bbx[i][1]
        result_list_dict["attribute"] = "{}".format(bbx[i][4])
        result_list_dict["valid"] = True
        id_ = random.sample(chars, 8)
        result_list_dict["id"] = "".join(d for d in id_)
        result_list_dict["sourceID"] = ""
        result_list_dict["textAttribute"] = ""
        result_list_dict["order"] = i + 1
        result_list.append(result_list_dict)

    step_1_["result"] = result_list
    json_["step_1"] = step_1_

    return json_


# def yolo_to_labelbee(data_path):
#     img_path = data_path + "/images"
#     txt_path = data_path + "/labels"
#     json_path = data_path + "/jsons"
#     os.makedirs(json_path, exist_ok=True)

#     txt_list = sorted(os.listdir(txt_path))

#     for t in tqdm(txt_list):
#         base_name = os.path.splitext(t)[0]
#         txt_abs_path = txt_path + "/{}".format(t)
#         try:
#             img_abs_path, json_abs_path, size = get_img_json_path_and_size(img_path, json_path, base_name)

#             bbx_for_json = []
#             with open(txt_abs_path, "r", encoding="utf-8") as fo:
#                 lines = fo.readlines()
#                 for l in lines:
#                     l_ = l.strip().split(" ")
#                     # bbx_ = [float(l_[1]), float(l_[2]), float(l_[3]), float(l_[4])]
#                     # VOC_bbx = convert_bbx_yolo_to_VOC(size, bbx_)
#                     bbx_ = [float(l_[1]), float(l_[2]), float(l_[3]), float(l_[4])]
#                     VOC_bbx = bbox_yolo_to_voc(size, bbx_)
#                     VOC_bbx = list(VOC_bbx)

#                     w_, h_ = VOC_bbx[2] - VOC_bbx[0], VOC_bbx[3] - VOC_bbx[1]
#                     if w_ < 3 or h_ < 3:
#                         print("img_abs_path: ", img_abs_path)
#                         print("txt_abs_path: ", txt_abs_path)

#                     VOC_bbx.append(int(l_[0]) + 1)
#                     bbx_for_json.append(VOC_bbx)

#             with open(json_abs_path, "w", encoding="utf-8") as jfw:
#                 jfw.write(json.dumps(write_labelbee_det_json(bbx_for_json, size)))

#             # print("labelbee json saved to --> {}".format(json_abs_path))

#         except Exception as Error:
#             print(Error)
#             print("txt_abs_path: ", txt_abs_path)


def convert_annotation(img_name, data_path, classes):
    import xml.etree.ElementTree as ET

    in_file = open('{}/xmls/{}.xml'.format(data_path, img_name), encoding='utf-8')
    out_file = open('{}/labels/{}.txt'.format(data_path, img_name), 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    class_names = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in class_names:
            class_names.append(cls)
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        # b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        # bb = convert_bbx_VOC_to_yolo((h, w), b)
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        bb = bbox_voc_to_yolo((h, w), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    return class_names


def voc_to_yolo(data_path, classes, val_percent=0.1):
    # classes = ["fire"]  # own data sets which classes which category to write, in the order
    # test set proportion of the total data set, the default 0.1, if the test set and the training set have been demarcated, the corresponding code is modified

    images_path = data_path + "/images/"  # darknet relative path folder, see description github, and they need to modify, according to note here the absolute path can also be used

    if not os.path.exists("{}/labels".format(data_path)):
        os.makedirs("{}/labels".format(data_path))

    xml_list = [f for f in os.listdir('{}/xmls'.format(data_path))]  # XML data storage folder
    train_file = open('{}/train.txt'.format(data_path), 'w', encoding="utf-8")
    val_file = open('{}/val.txt'.format(data_path), 'w', encoding="utf-8")
    class_names_all = []
    for i, xml_ in enumerate(xml_list):
        img_name = os.path.splitext(xml_)[0]
        if xml_.endswith(".xml"):  # Sometimes jpg and xml files are placed in the same folder, so to determine what suffixes
            if i < (len(xml_list) * val_percent):
                val_file.write("{}/{}.jpg".format(images_path, img_name))
            else:
                train_file.write("{}/{}.jpg".format(images_path, img_name))

        try:
            class_names = convert_annotation(img_name, data_path, classes)
            for nm in class_names:
                if nm not in class_names_all:
                    class_names_all.append(nm)
        except Exception as Error:
            print("Error: {}".format(img_name))

    print(class_names_all)

    train_file.close()
    val_file.close()


def labelbee_to_yolo(data_path, copy_image=False):
    img_path = data_path + "/images"
    json_path = data_path + "/jsons"

    det_images_path = data_path + "/{}".format("selected_images")
    det_labels_path = data_path + "/labels"
    if copy_image:
        os.makedirs(det_images_path, exist_ok=True)
    os.makedirs(det_labels_path, exist_ok=True)

    json_list = sorted(os.listdir(json_path))

    for j in tqdm(json_list):
        try:
            img_name = os.path.splitext(j)[0]
            img_base_name = os.path.splitext(img_name)[0]

            json_abs_path = json_path + "/{}".format(j)
            json_ = json.load(open(json_abs_path, 'r', encoding='utf-8'))
            if not json_: continue
            w, h = json_["width"], json_["height"]

            result_ = json_["step_1"]["result"]
            if not result_: continue

            if copy_image:
                img_abs_path = img_path + "/{}".format(img_name)
                shutil.copy(img_abs_path, det_images_path + "/{}".format(img_name))

            len_result = len(result_)

            txt_save_path = det_labels_path + "/{}.txt".format(img_base_name)
            with open(txt_save_path, "w", encoding="utf-8") as fw:
                for i in range(len_result):
                    x_ = result_[i]["x"]
                    y_ = result_[i]["y"]
                    w_ = result_[i]["width"]
                    h_ = result_[i]["height"]

                    cls_id = int(result_[i]["attribute"])
                    # if result_[i]["attribute"] == "roller_skating":
                    #     cls_id = 1
                    # elif result_[i]["attribute"] == "board_skating":
                    #     cls_id = 2
                    # else:
                    #     cls_id = int(result_[i]["attribute"])

                    x_min = x_
                    x_max = x_ + w_
                    y_min = y_
                    y_max = y_ + h_

                    # bb = convert_bbx_VOC_to_yolo((h, w), (x_min, x_max, y_min, y_max))
                    bb = bbox_voc_to_yolo((h, w), (x_min, y_min, x_max, y_max))
                    txt_content = "{}".format(cls_id) + " " + " ".join([str(b) for b in bb]) + "\n"
                    # txt_content = "{}".format(cls_id - 1) + " " + " ".join([str(b) for b in bb]) + "\n"
                    # txt_content = "{}".format(cls_id + 1) + " " + " ".join([str(b) for b in bb]) + "\n"
                    fw.write(txt_content)

        except Exception as Error:
            print(Error)


def coco_to_yolo(root):
    json_trainfile = root + '/annotations/instances_train2017.json'  # COCO Object Instance 类型的标注
    json_valfile = root + '/annotations/instances_val2017.json'  # COCO Object Instance 类型的标注
    train_ana_txt_save_path = root + '/train2017_labels/'  # 保存的路径
    val_ana_txt_save_path = root + '/val2017_labels/'  # 保存的路径

    traindata = json.load(open(json_trainfile, 'r'))
    valdata = json.load(open(json_valfile, 'r'))

    # 重新映射并保存class 文件
    if not os.path.exists(train_ana_txt_save_path):
        os.makedirs(train_ana_txt_save_path)
    if not os.path.exists(val_ana_txt_save_path):
        os.makedirs(val_ana_txt_save_path)

    id_map = {}  # coco数据集的id不连续!重新映射一下再输出!
    with open(os.path.join(root, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(traindata['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i

    '''
    保存train txt
    '''
    # print(id_map)
    # 这里需要根据自己的需要,更改写入图像相对路径的文件位置。
    list_file = open(os.path.join(root, 'train2017.txt'), 'w')
    for img in tqdm(traindata['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字,与jpg一致
        f_txt = open(os.path.join(train_ana_txt_save_path, ana_txt_name), 'w')
        for ann in traindata['annotations']:
            if ann['image_id'] == img_id:
                # box = convert((img_width, img_height), ann["bbox"])
                # box = convert_bbx_VOC_to_yolo((img_height, img_width), ann["bbox"])
                ann_np = np.array([ann["bbox"]])
                ann_np = ann_np[:, [0, 2, 1, 3]]
                ann_list = list(ann_np[0])
                box = bbox_voc_to_yolo((img_height, img_width), ann_list)
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
        # 将图片的相对路径写入train2017或val2017的路径
        list_file.write('./images/train2017/%s.jpg\n' % (head))
    list_file.close()

    '''
    保存val txt
    '''
    # print(id_map)
    # 这里需要根据自己的需要,更改写入图像相对路径的文件位置。
    list_file = open(os.path.join(root, 'val2017.txt'), 'w')
    for img in tqdm(valdata['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字,与jpg一致
        f_txt = open(os.path.join(val_ana_txt_save_path, ana_txt_name), 'w')
        for ann in valdata['annotations']:
            if ann['image_id'] == img_id:
                # box = convert((img_width, img_height), ann["bbox"])
                # box = convert_bbx_VOC_to_yolo((img_height, img_width), ann["bbox"])
                ann_np = np.array([ann["bbox"]])
                ann_np = ann_np[:, [0, 2, 1, 3]]
                ann_list = list(ann_np[0])
                box = bbox_voc_to_yolo((img_height, img_width), ann_list)
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
        # 将图片的相对路径写入train2017或val2017的路径
        list_file.write('./images/val2017/%s.jpg\n' % (head))
    list_file.close()


def write_point(doc, root, label1, label2, value1, value2):
    root = root.appendChild(doc.createElement('points'))
    root.appendChild(doc.createElement(label1)).appendChild(doc.createTextNode(value1))
    root.appendChild(doc.createElement(label2)).appendChild(doc.createTextNode(value2))


def write_one(doc, root, label, value):
    root.appendChild(doc.createElement(label)).appendChild(doc.createTextNode(value))


# def yolo_to_voc(args):
#     # TODO: rewrite and check
#     from xml.dom import minidom

#     input_file_record_path = args.input_file_record_path
#     input_label_checker_path = args.input_label_checker_path
#     input_xml_file_path = args.input_xml_file_path
#     output_folder_path = args.output_folder_path

#     text_list = get_file_list(input_file_record_path)
#     label_list = get_label_list(input_label_checker_path)
#     content_dict = generate_dict(text_list, label_list)

#     for key in content_dict.keys():
#         file_name = key
#         doc = minidom.Document()
#         annotationlist = doc.createElement('annotation')
#         doc.appendChild(annotationlist)

#         # folder = doc.createElement('folder')
#         # annotationlist.appendChild(folder)
#         # folder_name = doc.createTextNode(sys.argv[0].strip().split('/')[-2])
#         # folder.appendChild(folder_name)

#         annotationlist.appendChild(doc.createElement('filename')).appendChild(doc.createTextNode(sys.argv[0]))

#         xml_size = minidom.parse(os.path.join(input_xml_file_path, '{}.xml'.format(file_name)))
#         width_value = xml_size.getElementsByTagName('width')
#         width_value = width_value[0].firstChild.data
#         height_value = xml_size.getElementsByTagName('height')
#         height_value = height_value[0].firstChild.data
#         depth_value = xml_size.getElementsByTagName('depth')
#         depth_value = depth_value[0].firstChild.data

#         size = doc.createElement('size')
#         annotationlist.appendChild(size)
#         write_one(doc, size, 'width', width_value)
#         write_one(doc, size, 'height', height_value)
#         write_one(doc, size, 'depth', depth_value)

#         for i in range(len(content_dict[key])):
#             x_min = content_dict[key][i][0]
#             y_min = content_dict[key][i][1]
#             x_max = content_dict[key][i][2]
#             y_max = content_dict[key][i][3]
#             label = content_dict[key][i][4]

#             objectlist = doc.createElement('object')
#             annotationlist.appendChild(objectlist)
#             write_one(doc, objectlist, 'name', label)
#             write_one(doc, objectlist, 'difficult', '0')
#             write_one(doc, objectlist, 'truncated', '0')

#             bndbox = doc.createElement('bndbox')
#             objectlist.appendChild(bndbox)
#             write_one(doc, bndbox, 'xmin', x_min)
#             write_one(doc, bndbox, 'ymin', y_min)
#             write_one(doc, bndbox, 'xmax', x_max)
#             write_one(doc, bndbox, 'ymax', y_max)

#             segmentation = doc.createElement('segmentation')
#             objectlist.appendChild(segmentation)
#             write_point(doc, segmentation, 'x', 'y', x_min, y_min)
#             write_point(doc, segmentation, 'x', 'y', x_max, y_min)
#             write_point(doc, segmentation, 'x', 'y', x_max, y_max)
#             write_point(doc, segmentation, 'x', 'y', x_min, y_max)

#             if not os.path.exists(output_folder_path):
#                 os.makedirs(output_folder_path)
#             with open(os.path.join(output_folder_path, '{}.xml').format(file_name), 'w', encoding='UTF-8') as fh:
#                 doc.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')


def labelbee_kpt_to_yolo(data_path, copy_image=True):
    img_path = data_path + "/images"
    json_path = data_path + "/jsons"

    kpt_images_path = data_path + "/{}".format("selected_images")
    kpt_labels_path = data_path + "/labels"
    if copy_image:
        os.makedirs(kpt_images_path, exist_ok=True)
    os.makedirs(kpt_labels_path, exist_ok=True)

    json_list = sorted(os.listdir(json_path))

    for j in tqdm(json_list):
        try:
            json_abs_path = json_path + "/{}".format(j)
            json_ = json.load(open(json_abs_path, 'r', encoding='utf-8'))
            if not json_: continue
            w, h = json_["width"], json_["height"]

            result_ = json_["step_1"]["result"]
            if not result_: continue

            if copy_image:
                img_abs_path = img_path + "/{}".format(j.replace(".json", ""))
                # shutil.move(img_path, det_images_path + "/{}".format(j.replace(".json", "")))
                shutil.copy(img_abs_path, kpt_images_path + "/{}".format(j.replace(".json", "")))

            len_result = len(result_)

            txt_save_path = kpt_labels_path + "/{}.txt".format(j.replace(".json", "").split(".")[0])
            with open(txt_save_path, "w", encoding="utf-8") as fw:
                kpts = []
                for i in range(len_result):
                    x_ = result_[i]["x"]
                    y_ = result_[i]["y"]
                    attribute_ = result_[i]["attribute"]
                    x_normalized = x_ / w
                    y_normalized = y_ / h

                    visible = True
                    if visible:
                        kpts.append([x_normalized, y_normalized, 2])

                kpts = np.asarray(kpts).reshape(-1, 12)
                for ki in range(kpts.shape[0]):
                    txt_content = " ".join([str(k) for k in kpts[ki]]) + "\n"
                    fw.write(txt_content)

        except Exception as Error:
            print(Error)


def labelbee_kpt_to_dbnet(data_path, copy_image=True):
    img_path = data_path + "/images"
    json_path = data_path + "/jsons"

    kpt_images_path = data_path + "/{}".format("selected_images")
    kpt_labels_path = data_path + "/gts"
    if copy_image:
        os.makedirs(kpt_images_path, exist_ok=True)
    os.makedirs(kpt_labels_path, exist_ok=True)

    json_list = sorted(os.listdir(json_path))

    for j in tqdm(json_list):
        try:
            json_abs_path = json_path + "/{}".format(j)
            json_ = json.load(open(json_abs_path, 'r', encoding='utf-8'))
            if not json_: continue
            w, h = json_["width"], json_["height"]

            result_ = json_["step_1"]["result"]
            if not result_: continue

            if copy_image:
                img_abs_path = img_path + "/{}".format(j.replace(".json", ""))
                # shutil.move(img_path, det_images_path + "/{}".format(j.replace(".json", "")))
                shutil.copy(img_abs_path, kpt_images_path + "/{}".format(j.replace(".json", "")))

            len_result = len(result_)

            txt_save_path = kpt_labels_path + "/{}.gt".format(os.path.splitext(j.replace(".json", ""))[0])
            with open(txt_save_path, "w", encoding="utf-8") as fw:
                result_ = sorted(result_, key=lambda k: int(k["order"]))
                kpts = []
                for i in range(len_result):
                    # x_ = int(round(result_[i]["x"]))
                    # y_ = int(round(result_[i]["y"]))
                    x_ = result_[i]["x"]
                    y_ = result_[i]["y"]
                    attribute_ = result_[i]["attribute"]
                    # x_normalized = x_ / w
                    # y_normalized = y_ / h

                    # visible = True
                    # if visible:
                    #     kpts.append([x_normalized, y_normalized, 2])
                    kpts.append([x_, y_])

                kpts = np.asarray(kpts).reshape(-1, 8)
                for ki in range(kpts.shape[0]):
                    txt_content = ", ".join([str(k) for k in kpts[ki]]) + ", 0\n"
                    fw.write(txt_content)

        except Exception as Error:
            print(Error)


def parse_json(json_abs_path):
    json_data = json.load(open(json_abs_path, "r", encoding="utf-8"))
    w, h = json_data["width"], json_data["height"]
    len_object = len(json_data["step_1"]["result"])
    polygon_list = []
    label_list = []
    for i in range(len_object):
        pl_ = json_data["step_1"]["result"][i]["pointList"]

        # x_, y_ = [], []
        xy_ = []  # x, y, x, y. x. y, x, y
        for i in range(len(pl_)):
            # x_.append(float(pl_[i]["x"]))
            # y_.append(float(pl_[i]["y"]))

            xy_.append(float(pl_[i]["x"]))
            xy_.append(float(pl_[i]["y"]))

        polygon = list(map(float, xy_))
        polygon = list(map(math.floor, polygon))
        polygon = np.array(polygon, np.int32).reshape(-1, 1, 2)
        polygon_list.append(polygon)

        label_list.append(0)

    return polygon_list, label_list, (w, h)


def labelbee_seg_to_png(data_path):
    images_path = data_path + "/{}".format("images")
    json_path = data_path + "/{}".format("jsons")

    seg_images_path = data_path + "/{}".format("images_select")
    png_vis_path = data_path + "/{}".format("masks_vis")
    png_path = data_path + "/{}".format("masks")
    os.makedirs(seg_images_path, exist_ok=True)
    os.makedirs(png_vis_path, exist_ok=True)
    os.makedirs(png_path, exist_ok=True)

    json_list = []
    file_list = os.listdir(json_path)
    for f in file_list:
        if f.endswith(".json"):
            json_list.append(f)

    for j in json_list:
        try:
            json_abs_path = json_path + "/{}".format(j)
            polygon_list, label_list, img_size = parse_json(json_abs_path)

            if not polygon_list: continue

            img_vis, img = draw_label(size=(img_size[1], img_size[0], 3), polygon_list=polygon_list)
            png_vis_save_path = png_vis_path + "/{}".format(j.split(".")[0] + ".png")
            img_vis.save(png_vis_save_path)
            png_save_path = png_path + "/{}".format(j.split(".")[0] + ".png")
            img.save(png_save_path)

            img_src_path = images_path + "/{}".format(j.replace(".json", ""))
            img_dst_path = seg_images_path + "/{}".format(j.replace(".json", ""))
            shutil.copy(img_src_path, img_dst_path)
            print("{} copy to --> {}".format(img_src_path, img_dst_path))

        except Exception as Error:
            print(Error, Error.__traceback__.tb_lineno)


def convert_points(size, p):
    """
    convert 8 points to yolo format.
    :param size:
    :param p:
    :return:
    """
    dw, dh = 1. / (size[0]), 1. / (size[1])

    res = []
    for i in range(len(p)):
        if i % 2 == 0:
            res.append(p[i] * dw)
        else:
            res.append(p[i] * dh)

    return res


def labelbee_seg_json_to_yolo_txt(data_path):
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}".format("labels")

    removed_damaged_img = os.path.abspath(os.path.join(data_path, "../..")) + "/{}".format("removed")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(removed_damaged_img, exist_ok=True)

    keypoint_flag = False

    img_list = []
    json_list = []
    file_list = os.listdir(data_path)
    for f in file_list:
        if f.endswith(".jpg") or f.endswith(".jpeg"):
            img_list.append(f)
        elif f.endswith(".json"):
            json_list.append(f)

    for j in json_list:
        img_abs_path = data_path + "/{}".format(j.strip(".json"))
        # img = cv2.imread(img_abs_path)
        # if img is None: continue

        img_dst_path = removed_damaged_img + "/{}".format(j.strip(".json"))
        shutil.copy(img_abs_path, img_dst_path)

        json_abs_path = data_path + "/{}".format(j)
        json_ = json.load(open(json_abs_path, "r", encoding="utf-8"))
        w, h = json_["width"], json_["height"]

        txt_save_path = save_path + "/{}".format(j.replace(".json", ".txt"))
        with open(txt_save_path, "w", encoding="utf-8") as fw:
            len_object = len(json_["step_1"]["result"])
            pl = []
            for i in range(len_object):
                pl_ = json_["step_1"]["result"][i]["pointList"]

                x_, y_ = [], []
                xy_ = []  # x, y, x, y. x. y, x, y
                for i in range(len(pl_)):
                    x_.append(float(pl_[i]["x"]))
                    y_.append(float(pl_[i]["y"]))

                    xy_.append(float(pl_[i]["x"]))
                    xy_.append(float(pl_[i]["y"]))

                # yolov5 keypoint format
                if keypoint_flag:
                    if len(xy_) == 8 and len(x_) == 4 and len(y_) == 4:
                        x_min, x_max = min(x_), max(x_)
                        y_min, y_max = min(y_), max(y_)

                        # bb = convert_bbx_VOC_to_yolo((h, w), (x_min, x_max, y_min, y_max))
                        bb = bbox_voc_to_yolo((h, w), (x_min, y_min, x_max, y_max))
                        p_res = convert_points((w, h), xy_)

                        txt_content = "0" + " " + " ".join([str(a) for a in bb]) + " " + " ".join([str(c) for c in p_res]) + "\n"
                        fw.write(txt_content)
                else:
                    x_min, x_max = min(x_), max(x_)
                    y_min, y_max = min(y_), max(y_)

                    # bb = convert_bbx_VOC_to_yolo((h, w), (x_min, x_max, y_min, y_max))
                    bb = bbox_voc_to_yolo((h, w), (x_min, y_min, x_max, y_max))
                    txt_content = "0" + " " + " ".join([str(a) for a in bb]) + "\n"
                    fw.write(txt_content)

            print("Saved --> {}".format(txt_save_path))


def labelme_json_to_voc_xml(data_path):
    img_path = data_path + "/images"
    labelme_path = data_path + "/jsons"  # Original labelme label data path
    saved_path = data_path + "/xmls"  # Save path
    os.makedirs(saved_path, exist_ok=True)
    # Get pending files
    files = glob.glob(labelme_path + "/*.json")
    files = [i.split("/")[-1].split(".json")[0] for i in files]

    # Read annotation information and write to xml
    for json_file_ in files:
        json_filename = labelme_path + "/" + json_file_ + ".json"
        # json_filename = json_file_ + ".json"
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        height, width, channels = cv2.imread(img_path + "/" + json_file_ + ".jpg").shape
        with codecs.open(saved_path + "/" + json_file_ + ".xml", "w", "utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'Shanghai360_ZP_data' + '</folder>\n')
            xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
            xml.write('\t<source>\n')
            xml.write('\t\t<database>The UAV autolanding</database>\n')
            xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
            xml.write('\t\t<image>flickr</image>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t</source>\n')
            xml.write('\t<owner>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t\t<name>ChaojieZhu</name>\n')
            xml.write('\t</owner>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(width) + '</width>\n')
            xml.write('\t\t<height>' + str(height) + '</height>\n')
            xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t\t<segmented>0</segmented>\n')
            for multi in json_file["shapes"]:
                points = np.array(multi["points"])
                xmin = min(points[:, 0])
                xmax = max(points[:, 0])
                ymin = min(points[:, 1])
                ymax = max(points[:, 1])
                label = multi["label"]
                if xmax <= xmin:
                    pass
                elif ymax <= ymin:
                    pass
                else:
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + label + '</name>\n')
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    xml.write('\t\t<truncated>0</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>' + str(int(round(xmin))) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' + str(int(round(ymin))) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' + str(int(round(xmax))) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' + str(int(round(ymax))) + '</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')
                    print(json_filename, xmin, ymin, xmax, ymax, label)
            xml.write('</annotation>')


def vis_yolo_label(data_path, print_flag=True, color_num=1000, rm_small_object=False, rm_size=32):
    colors = []
    for i in range(color_num * 2):
        c = list(np.random.choice(range(256), size=3))
        if c not in colors:
            colors.append(c)

    colors = colors[:color_num]

    img_path = data_path + "/images"
    txt_path = data_path + "/labels"
    vis_path = data_path + "/vis_bbx"
    # vis_path = os.path.abspath(os.path.join(data_path, "..")) + "/vis_bbx"
    # txt_new_path = data_path + "/labels_new"
    os.makedirs(vis_path, exist_ok=True)
    # os.makedirs(txt_new_path, exist_ok=True)

    img_list = os.listdir(img_path)

    for img in tqdm(img_list):
        try:
            img_name = os.path.splitext(img)[0]
            img_abs_path = img_path + "/{}".format(img)
            txt_abs_path = txt_path + "/{}.txt".format(img_name)
            img = cv2.imread(img_abs_path)
            h, w = img.shape[:2]

            # txt_new_abs_path = txt_new_path + "/{}.txt".format(img_name)
            # txt_data_new = open(txt_new_abs_path, "w", encoding="utf-8")

            with open(txt_abs_path, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
                for l_orig in lines:
                    l = l_orig.strip()
                    cls = int(l.split(" ")[0])
                    l_ = [float(l.split(" ")[1]), float(l.split(" ")[2]), float(l.split(" ")[3]), float(l.split(" ")[4])]
                    # bbx_VOC_format = convert_bbx_yolo_to_VOC((h, w), l_)
                    bbx_VOC_format = bbox_yolo_to_voc((h, w), l_)
                    # if rm_small_object:
                    #     ow, oh = bbx_VOC_format[2] - bbx_VOC_format[0], bbx_VOC_format[3] - bbx_VOC_format[1]
                    #     if ow >= rm_size and oh >= rm_size:
                    #         txt_data_new.write(l_orig)

                    cv2.rectangle(img, (bbx_VOC_format[0], bbx_VOC_format[1]), (bbx_VOC_format[2], bbx_VOC_format[3]), (int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])), 2)
                    cv2.putText(img, "{}".format(cls), (bbx_VOC_format[0], bbx_VOC_format[1] - 4), cv2.FONT_HERSHEY_PLAIN, 2, (int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])))

                    cv2.imwrite("{}/{}".format(vis_path, img), img)
                    if print_flag:
                        print("--> {}/{}".format(vis_path, img))

            # txt_data_new.close()
            #
            # # Remove empty file
            # txt_data_new_r = open(txt_new_abs_path, "r", encoding="utf-8")
            # lines_new_r = txt_data_new_r.readlines()
            # txt_data_new_r.close()
            # if not lines_new_r:
            #     os.remove(txt_new_abs_path)
            #     print("os.remove: {}".format(txt_new_abs_path))

        except Exception as Error:
            print(Error)


def list_yolo_labels(label_path):
    file_list = get_file_list(label_path)
    labels = []
    for f in tqdm(file_list):
        f_abs_path = label_path + "/{}".format(f)
        with open(f_abs_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            for l in lines:
                cls = int(l.strip().split(" ")[0])
                if cls not in labels:
                    labels.append(cls)

    print("\n{}:".format(label_path))
    print("Len: {}, Labels: {}".format(len(labels), sorted(labels)))


def random_select_yolo_images_and_labels(data_path, select_num=1000, move_or_copy="copy", select_mode=0):
    orig_img_path = data_path + "/images"
    orig_lbl_path = data_path + "/labels"
    data_list = sorted(os.listdir(orig_img_path))

    assert select_num <= len(data_list), "{} is grater than total num!".format(select_num)

    selected_img_save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_random_selected_{}/images".format(data_path.split("/")[-1], select_num)
    selected_lbl_save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_random_selected_{}/labels".format(data_path.split("/")[-1], select_num)
    os.makedirs(selected_img_save_path, exist_ok=True)
    os.makedirs(selected_lbl_save_path, exist_ok=True)

    if select_mode == 0:
        selected = random.sample(data_list, select_num)
    else:
        selected = random.sample(data_list, len(data_list) - select_num)

    for f in tqdm(selected):
        f_name = os.path.splitext(f)[0]
        img_src_path = orig_img_path + "/{}".format(f)
        lbl_src_path = orig_lbl_path + "/{}.txt".format(f_name)

        img_dst_path = selected_img_save_path + "/{}".format(f)
        lbl_dst_path = selected_lbl_save_path + "/{}.txt".format(f_name)

        if move_or_copy == "copy":
            try:
                shutil.copy(img_src_path, img_dst_path)
                shutil.copy(lbl_src_path, lbl_dst_path)
            except Exception as Error:
                print(Error)
        elif move_or_copy == "move":
            shutil.move(img_src_path, img_dst_path)
            shutil.move(lbl_src_path, lbl_dst_path)
        else:
            print("Error!")


def change_yolo_labels(txt_base_path):
    """
    Just a simple example.
    :param txt_base_path:
    :return:
    """
    txt_path = txt_base_path + "/labels"
    save_path = txt_base_path + "/labels_new"
    os.makedirs(save_path, exist_ok=True)

    txt_list = sorted(os.listdir(txt_path))
    for txt in tqdm(txt_list):
        txt_abs_path = txt_path + "/{}".format(txt)
        txt_new_abs_path = save_path + "/{}".format(txt)

        txt_data = open(txt_abs_path, "r", encoding="utf-8")
        txt_data_new = open(txt_new_abs_path, "w", encoding="utf-8")
        lines = txt_data.readlines()
        for l_ in lines:
            l = l_.strip().split(" ")
            cls = int(l[0])

            # if cls == 0:
            #     cls_new = 1
            #     l_new = str(cls_new) + " " + " ".join([i for i in l[1:]]) + "\n"
            # elif cls == 1:
            #     cls_new = 1
            #     l_new = str(cls_new) + " " + " ".join([i for i in l[1:]]) + "\n"

            if cls == 80 or cls == 81:
                cls_new = cls - 80
                l_new = str(cls_new) + " " + " ".join([i for i in l[1:]]) + "\n"

                # if cls == 0:
                #     l_new = str(cls) + " " + " ".join([i for i in l[1:]]) + "\n"

                txt_data_new.write(l_new)

        txt_data.close()
        txt_data_new.close()

        # Remove empty file
        txt_data_new_r = open(txt_new_abs_path, "r", encoding="utf-8")
        lines_new_r = txt_data_new_r.readlines()
        txt_data_new_r.close()
        if len(lines_new_r) == 0:
            os.remove(txt_new_abs_path)
            print("os.remove: {}".format(txt_new_abs_path))
        # else:
        #     shutil.copy(txt_abs_path, txt_new_abs_path)


def merge_det_bbx_and_kpt_points_to_yolov5_pose_labels(data_path, cls=0):
    det_path = data_path + "/det"
    det_img_path = det_path + "/images"
    det_lbl_path = det_path + "/labels"
    kpt_path = data_path + "/kpt"
    kpt_img_path = kpt_path + "/images"
    kpt_lbl_path = kpt_path + "/labels"

    save_path = data_path + "/det_kpt"
    save_lbl_path = save_path + "/labels"
    os.makedirs(save_lbl_path, exist_ok=True)

    det_lbl_list = sorted(os.listdir(det_lbl_path))
    kpt_lbl_list = sorted(os.listdir(kpt_lbl_path))
    same_list = list(set(det_lbl_list) & set(kpt_lbl_list))

    for s in tqdm(same_list):
        try:
            fname = os.path.splitext(s)[0]
            img_s_abs_path = det_img_path + "/{}.jpg".format(fname)
            det_s_abs_path = det_lbl_path + "/{}".format(s)
            kpt_s_abs_path = kpt_lbl_path + "/{}".format(s)

            img = cv2.imread(img_s_abs_path)
            imgsz = img.shape[:2]

            det_bbxs = []

            with open(det_s_abs_path, "r", encoding="utf-8") as frd:
                det_lines = frd.readlines()
                for dl in det_lines:
                    dl = dl.strip().split(" ")
                    cls = int(dl[0])
                    bbx = list(map(float, dl[1:]))
                    bbx = np.asarray(bbx).reshape(-1, 4)
                    for b in range(bbx.shape[0]):
                        # bbx_voc = convert_bbx_yolo_to_VOC(imgsz, list(bbx[b]))
                        bbx_voc = bbox_yolo_to_voc(imgsz, list(bbx[b]))
                        det_bbxs.append(bbx_voc)

            dst_lbl_path = save_lbl_path + "/{}.txt".format(fname)
            with open(dst_lbl_path, "w", encoding="utf-8") as fwdk:
                with open(kpt_s_abs_path, "r", encoding="utf-8") as frk:
                    kpt_lines = frk.readlines()

                    for detbbx in det_bbxs:
                        # detbbx_new = [detbbx[0], detbbx[2], detbbx[1], detbbx[3]]
                        # bbx_yolo = convert_bbx_VOC_to_yolo(imgsz, detbbx_new)
                        detbbx_new = [detbbx[0], detbbx[1], detbbx[2], detbbx[3]]
                        bbx_yolo = bbox_voc_to_yolo(imgsz, detbbx_new)
                        for kl in kpt_lines:
                            kl_ = kl.strip().split(" ")
                            points = list(map(float, kl_))
                            points = np.asarray(points).reshape(-1, 3)
                            points_ = points[:, :2]
                            points_ = list(points_.reshape(1, -1)[0])
                            points_ = np.asarray(points_).reshape(-1, 8)[0]

                            p_bbx = [points_[0] * imgsz[1], points_[1] * imgsz[0], points_[4] * imgsz[1], points_[5] * imgsz[0]]
                            iou = cal_iou(detbbx, p_bbx)
                            if iou > 0:
                                txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bbx_yolo]) + " " + kl
                                fwdk.write(txt_content)
        except Exception as Error:
            print(Error)


def write_one(doc, root, label, value):
    root.appendChild(doc.createElement(label)).appendChild(doc.createTextNode(value))


def create_xml(xml_name, date, lineName, direction, startStation, endStation, startTime, endTime, startKm, endKm, startPoleNo, endPoleNo, panoramisPixel, partPixel):
    from xml.dom import minidom

    doc = minidom.Document()
    root = doc.createElement("detect")
    doc.appendChild(root)
    baseinfolist = doc.createElement("baseInfo")
    root.appendChild(baseinfolist)
    write_one(doc, baseinfolist, "date", date)
    write_one(doc, baseinfolist, "lineName", lineName)
    write_one(doc, baseinfolist, "direction", direction)
    write_one(doc, baseinfolist, "startStation", startStation)
    write_one(doc, baseinfolist, "endStation", endStation)

    appendinfolist = doc.createElement("appendInfo")
    root.appendChild(appendinfolist)
    write_one(doc, appendinfolist, "startTime", startTime)
    write_one(doc, appendinfolist, "endTime", endTime)
    write_one(doc, appendinfolist, "startKm", startKm)
    write_one(doc, appendinfolist, "endKm", endKm)
    write_one(doc, appendinfolist, "startPoleNo", startPoleNo)
    write_one(doc, appendinfolist, "endPoleNo", endPoleNo)
    write_one(doc, appendinfolist, "panoramisPixel", panoramisPixel)
    write_one(doc, appendinfolist, "partPixel", partPixel)

    with open(os.path.join('{}').format(xml_name), 'w', encoding='UTF-8') as fh:
        doc.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')


def create_mdb_if_not_exists(ACCESS_DATABASE_FILE):
    import pypyodbc

    ODBC_CONN_STR = 'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;' % ACCESS_DATABASE_FILE
    if not os.path.exists(ACCESS_DATABASE_FILE):
        mdb_file = pypyodbc.win_create_mdb(ACCESS_DATABASE_FILE)

        # ODBC_CONN_STR = 'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;' % ACCESS_DATABASE_FILE
        conn = pypyodbc.connect(ODBC_CONN_STR)
        cur = conn.cursor()

        SQL = """CREATE TABLE PICINDEX (id COUNTER PRIMARY KEY, SETLOC VARCHAR(255) NOT NULL, KM NUMBER NOT NULL, ST VARCHAR(255), PANORAMIS_START_FRAME NUMBER NOT NULL,
                                                PANORAMIS_START_PATH VARCHAR(255) NOT NULL, PANORAMIS_END_FRAME NUMBER NOT NULL, PANORAMIS_END_PATH VARCHAR(255) NOT NULL,
                                                PART_START_FRAME NUMBER NOT NULL, PART_START_PATH VARCHAR(255) NOT NULL, PART_END_FRAME NUMBER NOT NULL, PART_END_PATH VARCHAR(255) NOT NULL);"""
        cur.execute(SQL)
        conn.commit()
        cur.close()
        conn.close()


def write_data_to_mdb(ACCESS_DATABASE_FILE, insert_data):
    import pypyodbc

    ODBC_CONN_STR = 'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;' % ACCESS_DATABASE_FILE

    conn = pypyodbc.connect(ODBC_CONN_STR)
    cur = conn.cursor()

    SQL_ = """insert into PICINDEX (id, SETLOC, KM, ST, PANORAMIS_START_FRAME, PANORAMIS_START_PATH, PANORAMIS_END_FRAME, PANORAMIS_END_PATH, PART_START_FRAME, 
                        PART_START_PATH, PART_END_FRAME, PART_END_PATH) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    cur.execute(SQL_, insert_data)
    conn.commit()
    cur.close()
    conn.close()


def change_xml_content(filename, content_orig, content_chg):
    import xml.etree.ElementTree as ET

    xmlTree = ET.parse(filename)
    rootElement = xmlTree.getroot()
    for element in rootElement.findall("object"):
        if element.find('name').text == content_orig:
            element.find('name').text = content_chg
    xmlTree.write(filename, encoding='UTF-8', xml_declaration=True)


def extract_one_gif_frames(gif_path):
    img_name = os.path.splitext(os.path.basename(gif_path))[0]
    save_path = os.path.abspath(os.path.join(gif_path, "../..")) + "/{}_gif_frames".format(img_name.split("/")[-1])
    os.makedirs(save_path, exist_ok=True)

    gif_img = Image.open(gif_path)
    try:
        gif_img.save("{}/{}_{}.png".format(save_path, img_name, gif_img.tell()))
        while True:
            gif_img.seek(gif_img.tell() + 1)
            gif_img.save("{}/{}_{}.png".format(save_path, img_name, gif_img.tell()))
    except Exception as Error:
        print(Error)


def extract_one_video_frames(video_path, gap=5):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.abspath(os.path.join(video_path, "../..")) + "/{}_video_frames".format(video_name.split("/")[-1])
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if i % gap == 0:
                cv2.imwrite("{}/{}_{:07d}.jpg".format(save_path, video_name, i), frame)

            i += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()


def extract_videos_frames(base_path, gap=5, save_path=""):
    video_list = sorted(os.listdir(base_path))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = os.path.abspath(os.path.join(base_path, "../..")) + "/{}_video_frames".format(base_path.split("/")[-1])
        os.makedirs(save_path, exist_ok=True)

    for v in tqdm(video_list):
        try:
            video_abs_path = base_path + "/{}".format(v)
            video_name = os.path.splitext(v)[0]
            v_save_path = save_path + "/{}".format(video_name)
            if not os.path.exists(v_save_path): os.makedirs(v_save_path)

            cap = cv2.VideoCapture(video_abs_path)
            i = 0
            while True:

                ret, frame = cap.read()
                if ret:
                    if i % gap == 0:
                        cv2.imwrite("{}/{}_{:07d}.jpg".format(v_save_path, video_name, i), frame)

                    i += 1

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
        except Exception as Error:
            print(Error)


def scale_uint16(img, size):
    img1 = img // 256
    img2 = img % 256
    img1 = cv2.resize(img1.astype('uint8'), size, interpolation=cv2.INTER_NEAREST)
    img2 = cv2.resize(img2.astype('uint8'), size, interpolation=cv2.INTER_NEAREST)
    img3 = img1.astype('uint16') * 256 + img2.astype('uint16')
    return img3


def cal_mean_std_var(data_path, size=(64, 64)):
    img_h, img_w = size[0], size[1]  # 根据自己数据集适当调整,影响不大
    means, stds, vars = [], [], []
    img_list = []

    i = 0
    dir_list = os.listdir(data_path)
    for d in dir_list:
        imgs_path_list = os.listdir(data_path + "/{}".format(d))
        for item in tqdm(imgs_path_list):
            img = cv2.imread(os.path.join(data_path + "/{}".format(d), item))
            img = cv2.resize(img, (img_w, img_h))
            img = img[:, :, :, np.newaxis]
            img_list.append(img)
            i += 1

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))
        vars.append(np.var(pixels))

    return means, stds, vars


def convert_to_jpg_format(data_path):
    img_list = sorted(os.listdir(data_path))

    for img in img_list:
        img_name = os.path.splitext(img)[0]
        img_abs_path = data_path + "/{}".format(img)

        if img.endswith(".jpeg") or img.endswith(".png") or img.endswith(".bmp") or img.endswith(".JPG") or img.endswith(".JPEG") or img.endswith(".PNG") or img.endswith(".BMP"):
            img = cv2.imread(img_abs_path)
            cv2.imwrite("{}/{}.jpg".format(data_path, img_name), img)
            os.remove(img_abs_path)
            print("remove --> {} | write --> {}.jpg".format(img_abs_path, img_name))
        elif img.endswith(".jpg"):
            continue
        elif img.endswith(".gif") or img.endswith(".GIF") or img.endswith(".webp"):
            os.remove(img_abs_path)
            print("remove --> {}".format(img_abs_path))
        else:
            print(img_abs_path)


def convert_to_png_format(data_path):
    img_list = sorted(os.listdir(data_path))

    for img in img_list:
        img_abs_path = data_path + "/{}".format(img)
        try:
            img_name = os.path.splitext(img)[0]
            if img.endswith(".jpeg") or img.endswith(".jpg") or img.endswith(".bmp") or img.endswith(".JPEG") or img.endswith(".JPG") or img.endswith(".BMP"):
                # img_abs_path = data_path + "/{}".format(img)
                img = cv2.imread(img_abs_path)
                cv2.imwrite("{}/{}.png".format(data_path, img_name), img)
                os.remove(img_abs_path)
                print("write --> {}.png  |  remove --> {}".format(img_name, img))

            elif img.endswith(".png"):
                continue
            else:
                print(img)
        except Exception as Error:
            os.remove(img_abs_path)
            print("os.remove: {}".format(img_abs_path))


def HORIZON_quant_cal_mean_std(torchvision_mean, torchvision_std, print_flag=True):
    """
    ll = [0.5079259, 0.43544242, 0.40075096]
    for i in ll:
        print(i * 255)

    ll2 = [0.27482128, 0.26032233, 0.2618361]
    for i in ll2:
        print(1 / (i * 255))
    :param torchvision_mean:
    :param torchvision_std:
    :return:
    """
    HORIZON_quant_mean = []
    HORIZON_quant_std = []

    for i in torchvision_mean:
        HORIZON_quant_mean.append(i * 255)

    for i in torchvision_std:
        HORIZON_quant_std.append(1 / (i * 255))

    if print_flag:
        print("HORIZON_quant_mean: {} HORIZON_quant_std: {}".format(HORIZON_quant_mean, HORIZON_quant_std))

    return HORIZON_quant_mean, HORIZON_quant_std


def cal_green_sensitivity(hsv_img, mask_img):
    """
    My patent calculation
    :param hsv_img:
    :param mask_img:
    :return:
    """

    assert hsv_img.shape[:2] == mask_img.shape, "hsv_img.shape != mask_img.shape"
    mask = np.where((mask_img[:, :] > 127))

    h_, s_, v_ = [], [], []
    for x, y in zip(mask[1], mask[0]):
        try:
            h_.append(hsv_img[y, x, 0])
            s_.append(hsv_img[y, x, 1])
            v_.append(hsv_img[y, x, 2])
        except Exception as Error:
            print(Error)

    h_mean = np.mean(h_)
    s_mean = np.mean(s_)
    v_mean = np.mean(v_)

    h_green1, h_green2 = [], []
    for hi in h_:
        if hi >= 35 and hi <= 90:
            h_green1.append(hi)
        if hi > 45 and hi < 70:
            h_green2.append(hi)
    sigma1, sigma2 = 0.3, 0.7
    phi = len(h_green1) / len(mask[0]) * sigma1 + len(h_green2) / len(mask[0]) * sigma2
    sen = 1 / 3 * np.pi * s_mean ** 1.2 * v_mean ** 0.6 * phi

    return sen


def exit_light_patent_algorithm_test(img_path):
    img = cv2.imread(img_path)
    g_img = cv2.split(img)[1]
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ret, thresh = cv2.threshold(g_img, 127, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    # hsvimg = cv2.resize(hsvimg, (96, 64))
    # thresh = cv2.resize(thresh, (96, 64))

    t1 = time.time()
    sensitivity = cal_green_sensitivity(hsvimg, thresh)
    t2 = time.time()
    print(t2 - t1)
    print(sensitivity)

    if sensitivity > 1000:
        res = "ON"
        print(res)
    else:
        res = "OFF"
        print(res)


def black_area_change_pixel(img_path):
    save_path = img_path.replace(img_path.split("/")[-1], "{}_change_10".format(img_path.split("/")[-1]))
    os.makedirs(save_path, exist_ok=True)

    img_list = os.listdir(img_path)

    for img in img_list:
        img_abs_path = img_path + "/{}".format(img)
        img = cv2.imread(img_abs_path)
        img_cp = img.copy()

        # black_area = np.where((img[:, :, 0] < 5) & (img[:, :, 1] < 5) & (img[:, :, 2] < 5))
        black_area = np.where((img[:, :, 0] < 10) & (img[:, :, 1] < 10) & (img[:, :, 2] < 10))
        # black_area = np.where((img[:, :, 0] < 20) & (img[:, :, 1] < 20) & (img[:, :, 2] < 20))
        # black_area = np.where((img[:, :, 0] < 30) & (img[:, :, 1] < 30) & (img[:, :, 2] < 30))

        # bg_img = bg_img.copy()
        for x_b, y_b in zip(black_area[1], black_area[0]):
            try:
                img_cp[y_b, x_b] = (255, 0, 255)
            except Exception as Error:
                print(Error)

        cv2.imwrite("{}/{}".format(save_path, img), img_cp)


def perspective_transform(image, rect):
    """
    透视变换
    """
    tl, tr, br, bl = rect
    # tl, tr, br, bl = np.array([tl[0] - 20, tl[1] - 20]), np.array([tr[0] + 20, tr[1] - 20]), np.array([br[0] + 20, br[1] + 20]), np.array([bl[0] - 20, bl[1] + 20])
    # rect_new = np.array([tl[0] - 20, tl[1] - 20]), np.array([tr[0] + 20, tr[1] - 20]), np.array([br[0] + 20, br[1] + 20]), np.array([bl[0] - 20, bl[1] + 20])
    # 计算宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # 计算高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 定义变换后新图像的尺寸
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1],
                   [0, maxHeight-1]], dtype='float32')
    # 变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # 透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def crop_img_via_perspective_transform(img_path: str):
    # Need to test
    
    def click_event(event, x, y, flags, param):
        xy = []
        if event == cv2.EVENT_LBUTTONDOWN:
            xy.append((x, y))
            cv2.circle(img, (x, y), 1, (255, 0, 255), -1)
            cv2.putText(img, "({}, {})".format(x, y), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
            cv2.imshow("img", img)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    global xy
    xy = []

    cv2.imshow("img", img)
    cv2.setMouseCallback("img", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("xy: ", xy)

    # p1 = np.array([[8, 26], [137, 44], [16, 162], [147, 209]], dtype=np.float32)
    p1 = np.array(xy, dtype=np.float32)
    p2 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(p1, p2)
    warped = cv2.warpPerspective(img, M, (w, h))
    cv2.imwrite("{}".format(img_path.replace(".jpg", "_perspective_transform_cropped.jpg")), warped)


def process_black_images(img_path, flag="mv", pixel_sum=100000):
    img_list = sorted(os.listdir(img_path))

    if flag == "mv":
        dir_name = os.path.basename(img_path)
        save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_moved_black_images_{}".format(dir_name, pixel_sum)
        os.makedirs(save_path, exist_ok=True)

    for img in img_list:
        if os.path.isdir(img): continue
        img_abs_path = img_path + "/{}".format(img)
        try:

            img = cv2.imread(img_abs_path)
            img = cv2.resize(img, (128, 128))
            h, w = img.shape[:2]
            sum_ = np.sum(img[:, :, :])
            if sum_ < pixel_sum:
                if flag == "mv":
                    shutil.move(img_abs_path, save_path)
                elif flag == "rm":
                    os.remove(img_abs_path)

            # if w > 120 and h > 200:
            #     shutil.move(img_abs_path, save_path)

        except Exception as Error:
            if flag == "mv":
                shutil.move(img_abs_path, save_path)
            elif flag == "rm":
                os.remove(img_abs_path)
            print(Error)


def check_image(img_path):
    """
    remove file: Corrupt JPEG data: premature end of file / data segment.
    :param img_path:
    :return:
    """

    try:
        img = Image.open(img_path).load()
        img.verify()
        return True
    except:
        # os.remove(img_path)
        print("PIL check_image: Error! {}".format(img_path))
        return False


def remove_corrupt_images_pil(img_path, move_or_delete="delete"):
    img_list = sorted(os.listdir(img_path))
    dir_name = os.path.basename(img_path)

    if move_or_delete == "move":
        move_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_moved".format(dir_name)
        os.makedirs(move_path, exist_ok=True)

    for img in img_list:
        img_abs_path = img_path + "/{}".format(img)
        if move_or_delete == "move":
            img_dst_path = move_path + "/{}".format(img)
        # img = cv2.imread(img_abs_path)
        try:
            res = check_image(img_abs_path)
            if not res:
                if move_or_delete == "move":
                    shutil.move(img_abs_path, img_dst_path)
                    print("shutil.move {} --> {}".format(img_abs_path, img_dst_path))
                elif move_or_delete == "delete":
                    os.remove(img_abs_path)
                    print("Removed --> {}".format(img_abs_path))

        except Exception as Error:
            print(Error)
            os.remove(img_abs_path)
            print("Removed --> {}".format(img_abs_path))


def remove_corrupt_images_pil_v2(img_path, move_or_delete="delete"):
    img_list = sorted(os.listdir(img_path))
    dir_name = os.path.basename(img_path)

    if move_or_delete == "move":
        move_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_moved".format(dir_name)
        os.makedirs(move_path, exist_ok=True)

    for img in img_list:
        img_abs_path = img_path + "/{}".format(img)
        if move_or_delete == "move":
            img_dst_path = move_path + "/{}".format(img)
        try:
            img = Image.open(img_abs_path)
            img = np.asarray(img)
        except Exception as Error:
            print(Error)
            if move_or_delete == "move":
                shutil.move(img_abs_path, img_dst_path)
                print("shutil.move: {} --> {}".format(img_abs_path, img_dst_path))
            elif move_or_delete == "delete":
                os.remove(img_abs_path)
                print("Removed: --> {}".format(img_abs_path))


def badImgFast(fn, imgType=None):
    if os.path.getsize(fn) < 512:
        return True
    valid = False
    with open(fn, "rb") as f:
        f.seek(-2, 2)
        buf = f.read()
        valid = buf.endswith(b'\xff\xd9') or buf.endswith(b'\xae\x82') or buf.endswith(b'\x00\x3B') or buf.endswith(b'\x60\x82')  # 检测jpg图片完整性, 检测png图片完整性
        buf.endswith(b'\x00\x00')
    # return valid or (imghdr.what(nm) =="webp")
    return valid


def remove_corrupt_images_pil_v2_base(img_path, move_or_delete, img_list_i):
    dir_name = os.path.basename(img_path)
    if move_or_delete == "move":
        move_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_moved".format(dir_name)
        os.makedirs(move_path, exist_ok=True)

    for img in tqdm(img_list_i):
        suffix = os.path.splitext(img)[1][1:]
        img_abs_path = img_path + "/{}".format(img)
        if move_or_delete == "move":
            img_dst_path = move_path + "/{}".format(img)
        try:
            res = imghdr.what(img_abs_path)

            flag = False
            if suffix.lower()[:2] == res.lower()[:2]:
                flag = True

            if res == None or not flag:
                if move_or_delete == "move":
                    shutil.move(img_abs_path, img_dst_path)
                    print("shutil.move: {} --> {}".format(img_abs_path, img_dst_path))
                elif move_or_delete == "delete":
                    os.remove(img_abs_path)
                    print("Removed: --> {}".format(img_abs_path))
        except Exception as Error:
            print(Error)


def remove_corrupt_images_pil_v2_main(img_path, move_or_delete="delete"):
    img_list = sorted(os.listdir(img_path))

    len_ = len(img_list)
    img_lists = []
    split_n = 8
    for j in range(split_n):
        img_lists.append(img_list[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

    t_list = []
    for i in range(split_n):
        img_list_i = img_lists[i]
        t = threading.Thread(target=remove_corrupt_images_pil_v2_base, args=(img_path, move_or_delete, img_list_i,))
        t_list.append(t)

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


def remove_corrupt_images_cv2_v2_base(img_path, move_or_delete, img_list_i):
    dir_name = os.path.basename(img_path)
    if move_or_delete == "move":
        move_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_moved".format(dir_name)
        os.makedirs(move_path, exist_ok=True)

    for img in tqdm(img_list_i):
        suffix = os.path.splitext(img)[1][1:]
        img_abs_path = img_path + "/{}".format(img)
        if move_or_delete == "move":
            img_dst_path = move_path + "/{}".format(img)
        try:
            # res = imghdr.what(img_abs_path)

            res = cv2.imread(img_abs_path)
            if res is None:
                if move_or_delete == "move":
                    shutil.move(img_abs_path, img_dst_path)
                    print("shutil.move: {} --> {}".format(img_abs_path, img_dst_path))
                elif move_or_delete == "delete":
                    os.remove(img_abs_path)
                    print("Removed: --> {}".format(img_abs_path))
        except Exception as Error:
            print(Error)


def remove_corrupt_images_cv2_v2_main(img_path, move_or_delete="delete"):
    img_list = sorted(os.listdir(img_path))

    len_ = len(img_list)
    img_lists = []
    split_n = 8
    for j in range(split_n):
        img_lists.append(img_list[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

    t_list = []
    for i in range(split_n):
        img_list_i = img_lists[i]
        t = threading.Thread(target=remove_corrupt_images_cv2_v2_base, args=(img_path, move_or_delete, img_list_i,))
        t_list.append(t)

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


def remove_corrupt_images_opencv(img_path):
    img_list = sorted(os.listdir(img_path))
    for img in img_list:
        img_abs_path = img_path + "/{}".format(img)
        try:
            # img = cv2.imdecode(np.fromfile(img_abs_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.imread(img_abs_path)

            if img is None:
                os.remove(img_abs_path)
                print("[img is None]: Removed --> {}".format(img_abs_path))
                continue

            img_ = np.asarray(img)

        except Exception as Error:
            print(Error)
            os.remove(img_abs_path)
            print("Removed --> {}".format(img_abs_path))


def process_same_images_via_ssim(img_path, imgsz=(64, 64), flag="move"):
    from skimage.metrics import structural_similarity

    img_list = sorted(os.listdir(img_path))
    dir_name = os.path.basename(img_path)

    if flag == "move":
        move_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_same_images_moved".format(dir_name)
        os.makedirs(move_path, exist_ok=True)

    for i in range(len(img_list)):
        try:
            img_path_i = img_path + "/{}".format(img_list[i])
            img_i = cv2.imread(img_path_i)
            imgisz = img_i.shape[:2]
            if imgisz[0] < 10 or imgisz[1] < 10: continue
            if img_i is None: continue
            img_i = cv2.resize(img_i, imgsz)

            for j in range(i + 1, len(img_list)):
                img_path_j = img_path + "/{}".format(img_list[j])
                img_j = cv2.imread(img_path_j)
                imgjsz = img_j.shape[:2]
                if imgjsz[0] < 10 or imgjsz[1] < 10: continue
                if img_j is None: continue
                img_j = cv2.resize(img_j, imgsz)

                ssim = structural_similarity(img_i, img_j, multichannel=True)
                print("N: {} i: {}, j: {}, ssim: {}".format(len(img_list), i, j, ssim))

                if ssim > 0.95:
                    if flag == "remove" or flag == "delete":
                        os.remove(img_path_j)
                        print("{}, {} 两张图片相似度很高, ssim: {}  |  Removed: {}".format(img_list[i], img_list[j], ssim, img_path_j))
                    elif flag == "move":
                        shutil.move(img_path_j, move_path + "/{}".format(img_list[j]))
                        print("{}, {} 两张图片相似度很高, ssim: {}   |  {} --> {}/{}.".format(img_list[i], img_list[j], ssim, img_path_j, move_path, img_list[j]))
                    else:
                        print("'flag' should be one of [remove, delete, move]!")

            print(" ----------- {} ----------- ".format(i))

        except Exception as Error:
            print(Error, Error.__traceback__.tb_lineno)


def ssim_process_base(img_path, imgsz, img_list, ii, img_i, img_list_i, flag, move_path):
    from skimage.metrics import structural_similarity

    for j in range(len(img_list_i)):
        img_path_j = img_path + "/{}".format(img_list_i[j])
        img_j = cv2.imread(img_path_j)
        imgsz = img_j.shape[:2]
        if imgsz[0] < 7 or imgsz[1] < 7: continue
        if img_j is None: continue
        img_j = cv2.resize(img_j, imgsz)

        ssim = structural_similarity(img_i, img_j, multichannel=True)
        print("N: {} i: {}, j: {}, ssim: {}".format(len(img_list), ii, j, ssim))

        if ssim > 0.95:
            if flag == "remove" or flag == "delete":
                os.remove(img_path_j)
                print("{}, {} 两张图片相似度很高, ssim: {}  |  Removed: {}".format(img_list[ii], img_list_i[j], ssim, img_path_j))
            elif flag == "move":
                shutil.move(img_path_j, move_path + "/{}".format(img_list_i[j]))
                print("{}, {} 两张图片相似度很高, ssim: {}   |  {} --> {}/{}.".format(img_list[ii], img_list_i[j], ssim, img_path_j, move_path, img_list_i[j]))
            else:
                print("'flag' should be one of [remove, delete, move]!")

    print(" ----------- {} ----------- ".format(ii))


def process_same_images_multithread_via_ssim(img_path, imgsz=(64, 64), flag="move"):
    img_list = sorted(os.listdir(img_path))
    dir_name = os.path.basename(img_path)

    if flag == "move" or flag == "mv":
        move_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_same_images_moved".format(dir_name)
        os.makedirs(move_path, exist_ok=True)

    for ii in range(len(img_list)):
        try:
            img_path_i = img_path + "/{}".format(img_list[ii])
            img_i = cv2.imread(img_path_i)
            if img_i is None: continue
            img_i = cv2.resize(img_i, imgsz)

            img_list_left = img_list[ii + 1:]

            len_ = len(img_list_left)
            img_lists_left = []
            split_n = 8
            for j in range(split_n):
                img_lists_left.append(img_list_left[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

            t_list = []
            for i in range(split_n):
                img_list_i = img_lists_left[i]
                t = threading.Thread(target=ssim_process_base, args=(img_path, imgsz, img_list, ii, img_i, img_list_i, flag, move_path,))
                t_list.append(t)

            for t in t_list:
                t.start()
            for t in t_list:
                t.join()

        except Exception as Error:
            print(Error)


def remove_small_area(img_path):
    img_list = sorted(os.listdir(img_path))
    dir_name = os.path.basename(img_path)
    save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_removed_small_area".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    for img in img_list:
        try:
            img_abs_path = img_path + "/{}".format(img)
            img = cv2.imread(img_abs_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(img_gray.astype(np.uint8), 5, 255, cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                cx, cy, cw, ch = cv2.boundingRect(c)
                if cw * ch < 5000:
                    img[cy:cy + ch, cx:cx + cw] = (0, 0, 0)
                    print("{}: {}_{}_{}_{}".format(img, cx, cy, cw, ch))

            cv2.imwrite("{}/{}".format(save_path, img), img)

        except Exception as Error:
            print(Error)


def process_small_images(img_path, rmsz=48, mode=0):
    img_list = sorted(os.listdir(img_path))
    dir_name = os.path.basename(img_path)
    save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_small".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    for img in tqdm(img_list):
        if os.path.isdir(img): continue
        try:
            img_abs_path = img_path + "/{}".format(img)
            img_dst_path = save_path + "/{}".format(img)
            img = cv2.imdecode(np.fromfile(img_abs_path, dtype=np.uint8), cv2.IMREAD_COLOR)

            h, w = img.shape[:2]
            if mode == 0:
                if (h < rmsz and w < rmsz) or (h > 8 * w or w > 5 * h):
                    shutil.move(img_abs_path, img_dst_path)
            elif mode == 1:
                if h < rmsz or w < rmsz:
                    shutil.move(img_abs_path, img_dst_path)
            else:
                if (h < rmsz or w < rmsz) or (h > 3 * w or w > 5 * h):
                    shutil.move(img_abs_path, img_dst_path)

        except Exception as Error:
            print(Error)


def apply_hog(img):
    from skimage import feature, exposure

    fd, hog_img = feature.hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    return hog_img_rescaled


def MinFilterGray(src, r=7):
    '''最小值滤波,r是滤波器半径'''
    # 使用opencv的erode函数更高效

    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    ''''引导滤波,直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):
    # 输入rgb图像,值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)  # 得到暗通道图像
    V1 = guidedfilter(V1, MinFilterGray(V1, 7), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return Y


def cal_saliency_map(src, algorithm="FT"):
    if algorithm == "FT":
        lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        # gaussian_blur = cv2.GaussianBlur(src, (17, 17), 0)
        blur = cv2.medianBlur(src, 7)

        mean_lab = np.mean(lab, axis=(0, 1))
        saliency_map = (blur - mean_lab) * (blur - mean_lab)
        saliency_map = (saliency_map - np.amin(saliency_map)) / (np.amax(saliency_map) - np.amin(saliency_map))

        return saliency_map


def get_saliency_ft(img_path):
    from skimage.util import img_as_float

    # Saliency map calculation based on:

    img = skimage.io.imread(img_path)

    img_rgb = img_as_float(img)

    img_lab = skimage.color.rgb2lab(img_rgb)
    avgl, avga, avgb = np.mean(img_lab, axis=(0, 1))

    mean_val = np.mean(img_lab, axis=(0, 1))
    kernel_h = (1.0 / 16.0) * np.array([[1, 4, 6, 4, 1]])
    # kernel_h = (1.0/4.0) * np.array([[1,2,1]])
    kernel_w = kernel_h.transpose()

    blurred_l = scipy.signal.convolve2d(img_lab[:, :, 0], kernel_h, mode='same')
    blurred_a = scipy.signal.convolve2d(img_lab[:, :, 1], kernel_h, mode='same')
    blurred_b = scipy.signal.convolve2d(img_lab[:, :, 2], kernel_h, mode='same')

    blurred_l2 = scipy.signal.convolve2d(blurred_l, kernel_w, mode='same')
    blurred_a2 = scipy.signal.convolve2d(blurred_a, kernel_w, mode='same')
    blurred_b2 = scipy.signal.convolve2d(blurred_b, kernel_w, mode='same')

    im_blurred = np.dstack([blurred_l2, blurred_a2, blurred_b2])

    # sal = np.linalg.norm(mean_val - im_blurred,axis = 2)
    sal = np.square(blurred_l2 - avgl) + np.square(blurred_a2 - avga) + np.square(blurred_b2 - avgb)
    sal_max = np.max(sal)
    sal_min = np.min(sal)
    range = sal_max - sal_min
    if range == 0:
        range = 1
    sal = 255 * ((sal - sal_min) / range)

    sal = sal.astype(int)
    return sal


def binarise_saliency_map(saliency_map):
    adaptive_threshold = 2.0 * saliency_map.mean()
    return (saliency_map > adaptive_threshold)


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
        saliency_map2 = get_saliency_ft(img_abs_path)
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


def thresh_img(img, threshold_min_thr=10, adaptiveThreshold=True):
    if adaptiveThreshold:
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
        return None, thresh
    else:
        ret, thresh = cv2.threshold(img, threshold_min_thr, 255, cv2.THRESH_BINARY)
        return ret, thresh


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


def dehaze_test():
    m = deHaze(cv2.imread('/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data/20221111152824_8b46d8_75_0028505.jpg') / 255.0) * 255
    cv2.imwrite('/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data/20221111152824_8b46d8_75_0028505_defog.jpg', m)


def hog_test():
    data_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data"
    img_list = sorted(os.listdir(data_path))
    for img in img_list:
        img_name = os.path.splitext(img)[0]
        img_abs_path = data_path + "/{}".format(img)
        img = cv2.imread(img_abs_path, 0)
        hog_res = apply_hog(img)
        cv2.imwrite("{}/{}_hog.jpg".format(data_path, img_name), hog_res * 255)


def convert_to_gray_image(data_path):
    img_list = sorted(os.listdir(data_path))
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/images_gray"
    os.makedirs(save_path, exist_ok=True)

    for i in img_list:
        img_abs_path = data_path + "/{}".format(i)
        img = cv2.imread(img_abs_path, 0)
        cv2.imwrite("{}/{}".format(save_path, i), img)


def create_pure_images(save_path, size=(1080, 1920), max_pixel_value=20, save_num=1000, p=0.8):
    os.makedirs(save_path, exist_ok=True)
    colors = [[0, 0, 0],
              [10, 0, 0],
              [0, 10, 0],
              [0, 0, 10],
              [10, 10, 0],
              [10, 0, 10],
              [0, 10, 10],
              [10, 10, 10],
              [10, 15, 0],
              [10, 0, 15],
              [15, 10, 0],
              [2, 3, 5],
              [5, 2, 2],
              [5, 6, 2],
              [5, 7, 2],
              [5, 2, 8],
              [5, 54, 2],
              [5, 5, 2],
              ]

    colors2 = []
    for i in range(save_num):
        r = np.random.random()
        if r < p:
            c0 = np.random.choice(range(max_pixel_value))
            c1 = np.random.choice(range(max_pixel_value))
            c = [np.random.choice([c0, c1]), np.random.choice([c0, c1]), np.random.choice([c0, c1])]
        else:
            c = list(np.random.choice(range(max_pixel_value), size=3))
        if c not in colors2:
            colors2.append(c)

    if len(colors2) > 1000 and len(colors2) < 5000:
        colors2 = colors2 * 5
    elif len(colors2) <= 1000:
        colors2 = colors2 * 10
    elif len(colors2) >= 5000:
        colors2 = colors2 * 2

    for i in range(len(colors2)):
        img_init = np.ones(shape=[size[0], size[1], 3])
        img_b = img_init[:, :, 0] * colors2[i][0]
        img_g = img_init[:, :, 1] * colors2[i][1]
        img_r = img_init[:, :, 2] * colors2[i][2]
        img = cv2.merge([img_b, img_g, img_r])
        cv2.imwrite("{}/{}.jpg".format(save_path, i), img)


def classify_images_via_bgr_values(img_path):
    img_list = sorted(os.listdir(img_path))
    save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/cls_res"
    save_path_0 = save_path + "/0"
    save_path_1 = save_path + "/1"
    os.makedirs(save_path_0, exist_ok=True)
    os.makedirs(save_path_1, exist_ok=True)

    for i in img_list:
        img_abs_path = img_path + "/{}".format(i)
        img = cv2.imread(img_abs_path)
        imgsz = img.shape[:2]
        b, g, r = cv2.split(img)
        b_ = np.mean(np.asarray(b).reshape(1, -1))
        g_ = np.mean(np.asarray(g).reshape(1, -1))
        r_ = np.mean(np.asarray(r).reshape(1, -1))

        print("b_, g_, r_: ", b_, g_, r_)

        bg_mean = np.mean([b_, g_])

        if abs(r_ - bg_mean) < 30:
            img_dst_path = save_path_0 + "/{}".format(i)
            shutil.move(img_abs_path, img_dst_path)
        else:
            img_dst_path = save_path_1 + "/{}".format(i)
            shutil.move(img_abs_path, img_dst_path)


def GetRed(img):
    """
    提取图中的红色部分
    """
    # 转化为hsv空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # print(hsv.shape)
    # 颜色在HSV空间下的上下限
    low_hsv = np.array([0, 180, 80])
    high_hsv = np.array([10, 255, 255])

    # 使用opencv的inRange函数提取颜色
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    Red = cv2.bitwise_and(img, img, mask=mask)
    return Red


def find_red_bbx(img, expand_p=2):
    src = GetRed(img)
    binary = cv2.Canny(src, 80, 80 * 2)
    k = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, k)

    results = []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        arclen = cv2.arcLength(contours[c], True)
        if area < 20 or arclen < 100:
            continue
        rect = cv2.minAreaRect(contours[c])
        cx, cy = rect[0]

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        listX = [box[0][0], box[1][0], box[2][0], box[3][0]]
        listY = [box[0][1], box[1][1], box[2][1], box[3][1]]
        x1 = min(listX)
        y1 = min(listY)
        x2 = max(listX)
        y2 = max(listY)
        # print(x1, y1, x2, y2)
        width = np.int32(x2 - x1)
        height = np.int32(y2 - y1)

        roi = img[y1 + expand_p: y2 - expand_p, x1 + expand_p:x2 - expand_p]
        # print
        # print(x1,y1,x2,y2)
        if width < 80 or height < 80:
            continue

        # cv2.imshow("roi", roi)
        # cv2.waitKey(0)
        if len(roi):
            # cv2.imwrite("{}/{}_{}.jpg".format(lbl_path, img_name, c), roi)
            bbx_voc = [int(round(x1)) + expand_p, int(round(x2)) - expand_p, int(round(y1)) + expand_p, int(round(y2)) - expand_p]
            results.append(bbx_voc)

    return results


def detect_shape(c):
    """
    approxPolyDP()函数是opencv中对指定的点集进行多边形逼近的函数
    :param c:
    :return: 返回形状和折点的坐标
    """
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) == 3:
        shape = "triangle"
        return shape, approx

    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        return shape, approx

    elif len(approx) == 5:
        shape = "pentagon"
        return shape, approx

    elif len(approx) == 6:
        shape = "hexagon"
        return shape, approx

    elif len(approx) == 8:
        shape = "octagon"
        return shape, approx

    elif len(approx) == 10:
        shape = "star"
        return shape, approx

    else:
        shape = "circle"
        return shape, approx


def seg_crop_object(img, bgimg, maskimg):
    # imgsz = img.shape
    outimg = np.zeros(img.shape)
    # outimg2 = bgimg.copy()
    # roi = np.where(maskimg[:, :, 0] != 0 & maskimg[:, :, 1] != 0 & maskimg[:, :, 2] != 0)
    roi = np.where(maskimg[:, :, 0] != 0)
    outimg[roi] = img[roi]
    # outimg2[roi] = (0, 0, 0)

    conts, hierarchy = cv2.findContours(maskimg[:, :, 0].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxc = max(conts, key=cv2.contourArea)
    bbox = cv2.boundingRect(maxc)
    outimg_crop = outimg[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
    # outimg2_crop = outimg2[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
    # merged = outimg_crop + outimg2_crop

    # relative_roi = []
    # for ri in range(len(roi[0])):
    #     y_new, x_new = y - bbox[1], x - bbox[0]
    #     relative_roi.append([y_new, x_new])

    # relative_roi = np.asarray(relative_roi)

    relative_roi = (roi[0] - bbox[1], roi[1] - bbox[0])

    return outimg_crop, bbox, relative_roi


def vstack_two_images(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/0-9_white"):
    # save_path = os.path.abspath(os.path.join(data_path, "..")) + "/llj_0-9_vstack_output"
    # os.makedirs(save_path, exist_ok=True)
    #
    # for i in range(10):
    #     imgi_path = data_path + "/{}.png".format(i)
    #     imgi1_path = data_path + "/{}.png".format(i + 1)
    #     imgi = cv2.imread(imgi_path)
    #     imgi1 = cv2.imread(imgi1_path)
    #
    #     vstack = np.vstack((imgi, imgi1))
    #     vstacksz = vstack.shape
    #
    #     # while (True):
    #     #     rdm = np.random.random()
    #     #     if rdm < 0.25:
    #     #         break
    #     # out = vstack[int(round(rdm * vstacksz[0])):int(round((1 - rdm) * vstacksz[0])), 0:vstacksz[1]]
    #     # cv2.imwrite("{}/{}.jpg".format(save_path, i), out)
    #
    #     rdm = np.random.random()
    #     out = vstack[int(round(rdm * vstacksz[0])):int(round((1 - rdm) * vstacksz[0])), 0:vstacksz[1]]
    #     if rdm < 0.25:
    #         cv2.imwrite("{}/{}.jpg".format(save_path, i), out)
    #     # else:
    #     #     cv2.imwrite("{}/{}.jpg".format(save_path, i + 1), out)

    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/llj_0-9_vstack_output"
    os.makedirs(save_path, exist_ok=True)

    for i in range(10):
        imgi_path = data_path + "/{}N.png".format(i)
        imgi1_path = data_path + "/{}N.png".format(i + 1)
        imgi = cv2.imread(imgi_path)
        imgi1 = cv2.imread(imgi1_path)

        vstack = np.vstack((imgi, imgi1))
        vstacksz = vstack.shape

        # while (True):
        #     rdm = np.random.random()
        #     if rdm < 0.25:
        #         break
        # out = vstack[int(round(rdm * vstacksz[0])):int(round((1 - rdm) * vstacksz[0])), 0:vstacksz[1]]
        # cv2.imwrite("{}/{}.jpg".format(save_path, i), out)

        rdm = np.random.random()
        out = vstack[int(round(rdm * vstacksz[0])):int(round((1 - rdm) * vstacksz[0])), 0:vstacksz[1]]
        if rdm < 0.25:
            cv2.imwrite("{}/{}_{}.jpg".format(save_path, i, str(rdm).replace(".", "")), out)
        # else:
        #     cv2.imwrite("{}/{}_{}.jpg".format(save_path, i + 1, str(rdm).replace(".", "")), out)


def cut_images(data_path):
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/0-9_output_ud"
    os.makedirs(save_path, exist_ok=True)

    for i in range(10):
        imgi_path = data_path + "/{}N.png".format(i)
        imgi = cv2.imread(imgi_path)
        imgisz = imgi.shape
        for j in range(1, 10):
            outi_u = imgi[int(round(j * 0.1 * imgisz[0])):imgisz[0], 0:imgisz[1]]
            outi_d = imgi[0:int(round((1 - j * 0.1) * imgisz[0])), 0:imgisz[1]]
            cv2.imwrite("{}/{}_{}_u.png".format(save_path, i, j), outi_u)
            cv2.imwrite("{}/{}_{}_d.png".format(save_path, i, j), outi_d)


def stack_images(data_path):
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/0-9_output_ud_stack"
    os.makedirs(save_path, exist_ok=True)

    for i in range(10):
        for j in range(1, 10):
            imgi_j_u_path = data_path + "/{}_{}_u.png".format(i, j)
            if i == 9:
                imgi1_j_d_path = data_path + "/{}_{}_d.png".format(0, 10 - j)
            else:
                imgi1_j_d_path = data_path + "/{}_{}_d.png".format(i + 1, 10 - j)
            imgi_j_u = cv2.imread(imgi_j_u_path)
            imgi1_j_d = cv2.imread(imgi1_j_d_path)
            # imgi_j_u_sz = imgi_j_u.shape

            stack = np.vstack((imgi_j_u, imgi1_j_d))
            if j <= 5:
                cv2.imwrite("{}/{}_{}_stack={}.png".format(save_path, i, 10 - j, i), stack)
            else:
                if i == 9:
                    cv2.imwrite("{}/{}_{}_stack={}.png".format(save_path, i, 10 - j, 0), stack)
                else:
                    cv2.imwrite("{}/{}_{}_stack={}.png".format(save_path, i, 10 - j, i + 1), stack)


def crop_images(data_path):
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_cropped".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    data_list = sorted(os.listdir(data_path))
    for f in tqdm(data_list):
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        imgsz = img.shape[:2]
        # cropped = img[212:imgsz[0], :]
        cropped = img[0:165, :]
        cv2.imwrite("{}/{}".format(save_path, f), cropped)


def get_color(specific=True):
    global color
    color1 = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

    if specific:
        color2, color3, color4 = (0, 0, 0), (114, 114, 114), (255, 255, 255)
        color_rdm = np.random.rand()
        if color_rdm <= 0.85:
            color = color1
        elif color_rdm > 0.85 and color_rdm <= 0.90:
            color = color2
        elif color_rdm > 0.90 and color_rdm <= 0.95:
            color = color3
        else:
            color = color4
    else:
        color = color1

    return color


def makeBorder_base(im, new_shape=(64, 256), r1=0.75, specific=True):
    """
    :param im:
    :param new_shape: (H, W)
    :param r1:
    :param r2:
    :param sliding_window:
    :return:
    """
    color = get_color(specific=specific)

    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape * 4)

    # if im is too small(shape[0] < new_shape[0] * 0.75), first pad H, then calculate r.
    if shape[0] < new_shape[0] * r1:
        padh = new_shape[0] - shape[0]
        padh1 = padh // 2
        padh2 = padh - padh1
        im = cv2.copyMakeBorder(im, padh1, padh2, 0, 0, cv2.BORDER_CONSTANT, value=color)  # add border

    shape = im.shape[:2]  # current shape [height, width]
    r = new_shape[0] / shape[0]

    # Compute padding
    new_unpad_size = (int(round(shape[0] * r)), int(round(shape[1] * r)))
    ph, pw = new_shape[0] - new_unpad_size[0], new_shape[1] - new_unpad_size[1]  # wh padding

    rdm = np.random.random()
    if rdm > 0.5:
        top = ph // 2
        bottom = ph - top
        left = pw // 2
        right = pw - left

        if shape != new_unpad_size:
            im = cv2.resize(im, new_unpad_size[::-1], interpolation=cv2.INTER_LINEAR)

        if im.shape[1] <= new_shape[1]:
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        else:
            im = cv2.resize(im, new_shape[::-1])
    else:
        rdmh = np.random.random()
        rmdw = np.random.random()
        top = int(round(ph * rdmh))
        bottom = ph - top
        left = int(round(pw * rmdw))
        right = pw - left

        if shape != new_unpad_size:
            im = cv2.resize(im, new_unpad_size[::-1], interpolation=cv2.INTER_LINEAR)

        if im.shape[1] <= new_shape[1]:
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        else:
            im = cv2.resize(im, new_shape[::-1])

    return im


def sliding_window_crop_v2(img, cropsz=(64, 256), gap=(0, 128), makeBorder=True, r1=0, r2=0.25, specific=True):
    cropped_imgs = []
    imgsz = img.shape[:2]

    if gap[0] == 0 and gap[1] > 0:
        cropsz = (imgsz[0], cropsz[1])
        for i in range(0, imgsz[1], gap[1]):
            if i + cropsz[1] > imgsz[1]:
                cp_img = img[0:imgsz[0], i:imgsz[1]]
                if makeBorder:
                    cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific=specific)
                cropped_imgs.append(cp_img)
                break
            else:
                cp_img = img[0:imgsz[0], i:i + cropsz[1]]
                cropped_imgs.append(cp_img)
    elif gap[0] > 0 and gap[1] == 0:
        cropsz = (cropsz[0], imgsz[1])
        for j in range(0, imgsz[0], gap[0]):
            if j + cropsz[0] > imgsz[0]:
                cp_img = img[j:imgsz[0], 0:imgsz[1]]
                if makeBorder:
                    cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific=specific)
                cropped_imgs.append(cp_img)
                break
            else:
                cp_img = img[j:j + cropsz[0], 0:imgsz[1]]
                cropped_imgs.append(cp_img)
    elif gap[0] == 0 and gap[1] == 0:
        print("Error! gap[0] == 0 and gap[1] == 0!")
    else:
        for j in range(0, imgsz[0], gap[0]):
            if j + cropsz[0] > imgsz[0]:
                for i in range(0, imgsz[1], gap[1]):
                    if i + cropsz[1] > imgsz[1]:
                        cp_img = img[j:imgsz[0], i:imgsz[1]]
                        if makeBorder:
                            cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific=specific)
                        cropped_imgs.append(cp_img)
                        break
                    else:
                        cp_img = img[j:imgsz[0], i:i + cropsz[1]]
                        cropped_imgs.append(cp_img)
                break

            else:
                for i in range(0, imgsz[1], gap[1]):
                    if i + cropsz[1] > imgsz[1]:
                        cp_img = img[j:j + cropsz[0], i:imgsz[1]]
                        if makeBorder:
                            cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific=specific)
                        cropped_imgs.append(cp_img)
                        break
                    else:
                        cp_img = img[j:j + cropsz[0], i:i + cropsz[1]]
                        cropped_imgs.append(cp_img)

    return cropped_imgs


def makeBorder_v6(im, new_shape=(64, 256), r1=0.75, r2=0.25, sliding_window=False, specific=True, gap_r=(0, 7 / 8), last_img_makeBorder=True):
    """
    :param im:
    :param new_shape: (H, W)
    :param r1:
    :param r2:
    :param sliding_window:
    :return:
    """
    color = get_color(specific=specific)

    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape * 4)

    # if im is too small(shape[0] < new_shape[0] * 0.75), first pad H, then calculate r.
    if shape[0] < new_shape[0] * r1:
        padh = new_shape[0] - shape[0]
        padh1 = padh // 2
        padh2 = padh - padh1
        im = cv2.copyMakeBorder(im, padh1, padh2, 0, 0, cv2.BORDER_CONSTANT, value=color)  # add border

    shape = im.shape[:2]  # current shape [height, width]
    r = new_shape[0] / shape[0]

    # Compute padding
    new_unpad_size = (int(round(shape[0] * r)), int(round(shape[1] * r)))
    ph, pw = new_shape[0] - new_unpad_size[0], new_shape[1] - new_unpad_size[1]  # wh padding

    rdm = np.random.random()
    if rdm > 0.5:
        top = ph // 2
        bottom = ph - top
        left = pw // 2
        right = pw - left

        if shape != new_unpad_size:
            im = cv2.resize(im, new_unpad_size[::-1], interpolation=cv2.INTER_LINEAR)

        if im.shape[1] <= new_shape[1]:
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        elif (im.shape[1] > new_shape[1]) and (im.shape[1] <= (new_shape[1] + int(round(new_shape[1] * r2)))):
            im = cv2.resize(im, new_shape[::-1])
        else:  # TODO sliding window: 2023.09.27 Done
            if sliding_window:
                final_imgs = sliding_window_crop_v2(im, cropsz=new_shape, gap=(int(gap_r[0] * 0), int(gap_r[1] * new_shape[1])), makeBorder=last_img_makeBorder)
                return final_imgs
            else:
                im = cv2.resize(im, new_shape[::-1])
    else:
        rdmh = np.random.random()
        rmdw = np.random.random()
        top = int(round(ph * rdmh))
        bottom = ph - top
        left = int(round(pw * rmdw))
        right = pw - left

        if shape != new_unpad_size:
            im = cv2.resize(im, new_unpad_size[::-1], interpolation=cv2.INTER_LINEAR)

        if im.shape[1] <= new_shape[1]:
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        elif (im.shape[1] > new_shape[1]) and (im.shape[1] <= (new_shape[1] + int(round(new_shape[1] * r2)))):
            im = cv2.resize(im, new_shape[::-1])
        else:  # TODO sliding window: 2023.09.27 Done
            if sliding_window:
                final_imgs = sliding_window_crop_v2(im, cropsz=new_shape, gap=(int(gap_r[0] * 0), int(gap_r[1] * new_shape[1])), makeBorder=last_img_makeBorder)
                return final_imgs
            else:
                im = cv2.resize(im, new_shape[::-1])

    return im


def makeBorder_inference(im, new_shape=(64, 256), r1=0, r2=0.25, sliding_window=False, color=(0, 0, 0), gap_r=(0, 7 / 8), last_img_makeBorder=True):
    """
    :param im:
    :param new_shape: (H, W)
    :param r1:
    :param r2:
    :param sliding_window:
    :return:
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape * 4)

    # if im is too small(shape[0] < new_shape[0] * 0.75), first pad H, then calculate r.
    if shape[0] < new_shape[0] * r1:
        padh = new_shape[0] - shape[0]
        padh1 = padh // 2
        padh2 = padh - padh1
        im = cv2.copyMakeBorder(im, padh1, padh2, 0, 0, cv2.BORDER_CONSTANT, value=color)  # add border

    shape = im.shape[:2]  # current shape [height, width]
    r = new_shape[0] / shape[0]

    # Compute padding
    new_unpad_size = (int(round(shape[0] * r)), int(round(shape[1] * r)))
    ph, pw = new_shape[0] - new_unpad_size[0], new_shape[1] - new_unpad_size[1]  # wh padding

    top = ph // 2
    bottom = ph - top
    left = 0
    right = pw - left

    if shape != new_unpad_size:
        im = cv2.resize(im, new_unpad_size[::-1], interpolation=cv2.INTER_LINEAR)

    if im.shape[1] <= new_shape[1]:
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    elif (im.shape[1] > new_shape[1]) and (im.shape[1] <= (new_shape[1] + int(round(new_shape[1] * r2)))):
        im = cv2.resize(im, new_shape[::-1])
    else:  # TODO sliding window: 2023.09.27 Done
        if sliding_window:
            final_imgs = sliding_window_crop_v2(im, cropsz=new_shape, gap=(int(gap_r[0] * 0), int(gap_r[1] * new_shape[1])), makeBorder=last_img_makeBorder, r1=r1)
            return final_imgs
        else:
            im = cv2.resize(im, new_shape[::-1])

    return im


def sample_asym(magnitude, size=None):
    return np.random.beta(1, 4, size) * magnitude


def sample_sym(magnitude, size=None):
    return (np.random.beta(4, 4, size=size) - 0.5) * 2 * magnitude


def sample_uniform(low, high, size=None):
    return np.random.uniform(low, high, size=size)


def get_interpolation(type='random'):
    if type == 'random':
        choice = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
        interpolation = choice[random.randint(0, len(choice) - 1)]
    elif type == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif type == 'linear':
        interpolation = cv2.INTER_LINEAR
    elif type == 'cubic':
        interpolation = cv2.INTER_CUBIC
    elif type == 'area':
        interpolation = cv2.INTER_AREA
    else:
        raise TypeError('Interpolation types only nearest, linear, cubic, area are supported!')
    return interpolation


def blend_mask(image, mask, alpha=0.5, cmap='jet', color='b', color_alpha=1.0):
    # normalize mask
    mask = (mask - mask.min()) / (mask.max() - mask.min() + np.finfo(float).eps)
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    # get color map
    color_map = plt.get_cmap(cmap)
    mask = color_map(mask)[:, :, :3]
    # convert float to uint8
    mask = (mask * 255).astype(dtype=np.uint8)

    # set the basic color
    basic_color = np.array(mpl.colors.to_rgb(color)) * 255
    basic_color = np.tile(basic_color, [image.shape[0], image.shape[1], 1])
    basic_color = basic_color.astype(dtype=np.uint8)
    # blend with basic color
    blended_img = cv2.addWeighted(image, color_alpha, basic_color, 1 - color_alpha, 0)
    # blend with mask
    blended_img = cv2.addWeighted(blended_img, alpha, mask, 1 - alpha, 0)

    return blended_img


def onehot(label, depth, device=None):
    """
    Args:
        label: shape (n1, n2, ..., )
        depth: a scalar

    Returns:
        onehot: (n1, n2, ..., depth)
    """
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, device=device)
    onehot = torch.zeros(label.size() + torch.Size([depth]), device=device)
    onehot = onehot.scatter_(-1, label.unsqueeze(-1), 1)

    return onehot


def bmp2jpg(bmp_abs_path):
    base_name = os.path.basename(bmp_abs_path)
    save_name = base_name.replace(".bmp", ".jpg")
    cv2_img = cv2.imread(bmp_abs_path)

    return cv2_img, save_name


def dcm2jpg(dcm_path):
    import pydicom
    ds = pydicom.read_file(dcm_path)  # 读取.dcm文件
    img = ds.pixel_array  # 提取图像信息
    # scipy.misc.imsave(out_path, img)
    return img


def calc_brightness(img):
    # 把图片转换为单通道的灰度图
    img = cv2.resize(img, (16, 16))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取形状以及长宽
    img_shape = gray_img.shape
    height, width = img_shape[0], img_shape[1]
    size = gray_img.size
    # 灰度图的直方图
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # 计算灰度图像素点偏离均值(128)程序
    a = 0
    ma = 0
    reduce_matrix = np.full((height, width), 128)
    shift_value = gray_img - reduce_matrix
    shift_sum = sum(map(sum, shift_value))
    da = shift_sum / size
    # 计算偏离128的平均偏差
    for i in range(256):
        ma += (abs(i - 128 - da) * hist[i])
    m = abs(ma / size)

    # 亮度系数
    if m == 0:
        print("ZeroDivisionError!")
        return 100, -100
    else:
        k = abs(da) / m
        return k[0], da


def cv2_img_add_text(img, text, left, top, font_path="simsun.ttc", textColor=(0, 255, 0), textSize=20):
    from PIL import ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageFile

    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(font_path, textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def random_color():
    b = random.randint(0, 256)
    g = random.randint(0, 256)
    r = random.randint(0, 256)

    return (b, g, r)


def cal_SVD_var(img):
    img_r, img_g, img_b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    u_r, sigma_r, v_r = np.linalg.svd(img_r)
    u_g, sigma_g, v_g = np.linalg.svd(img_r)
    u_b, sigma_b, v_b = np.linalg.svd(img_r)
    # r
    len_sigma_r = len(sigma_r)
    len_sigma_r_50 = int(round(.5 * len_sigma_r))
    len_sigma_r_20 = int(round(.2 * len_sigma_r))
    var_r_50 = np.var(sigma_r[:len_sigma_r_50])
    var_r_last_20 = np.var(sigma_r[-len_sigma_r_20:])
    # g
    len_sigma_g = len(sigma_g)
    len_sigma_g_50 = int(round(.5 * len_sigma_g))
    len_sigma_g_20 = int(round(.2 * len_sigma_g))
    var_g_50 = np.var(sigma_r[:len_sigma_g_50])
    var_g_last_20 = np.var(sigma_r[-len_sigma_g_20:])
    # b
    len_sigma_b = len(sigma_b)
    len_sigma_b_50 = int(round(.5 * len_sigma_b))
    len_sigma_b_20 = int(round(.2 * len_sigma_b))
    var_b_50 = np.var(sigma_r[:len_sigma_b_50])
    var_b_last_20 = np.var(sigma_r[-len_sigma_b_20:])

    var_50 = np.mean([var_r_50, var_g_50, var_b_50])
    var_last_20 = np.mean([var_r_last_20, var_g_last_20, var_b_last_20])

    return var_50, var_last_20


def find_specific_color(img, lower=(0, 0, 100), upper=(80, 80, 255)):
    """
    https://stackoverflow.com/questions/42592234/python-opencv-morphologyex-remove-specific-color
    Parameters
    ----------
    img

    Returns
    -------

    """
    mask = cv2.inRange(img, np.array(lower), np.array(upper))
    # mask = 255 - mask
    res = cv2.bitwise_and(img, img, mask=mask)  # -- Contains pixels having the gray color--

    return res


def change_pixels_value(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    thresh1 = cv2.threshold(s, 92, 255, cv2.THRESH_BINARY)[1]
    thresh2 = cv2.threshold(v, 10, 255, cv2.THRESH_BINARY)[1]
    thresh2 = 255 - thresh2
    mask = cv2.add(thresh1, thresh2)

    H, W, _ = img.shape
    newimg = img.copy()

    for i in range(H):
        for j in range(W):
            if mask[i, j] != 0:
                newimg[i, j] = img[i - 12, j - 12]

    return newimg


def get_peak_points(heatmaps):
    """

    :param heatmaps: numpy array (N,4,256,256)
    :return:numpy array (N,4,2) #
    """
    N,C,H,W = heatmaps.shape   # N= batch size C=4 hotmaps
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i, j] == heatmaps[i, j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x, y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points


# ========================================= Color Identification =========================================
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_colors(image, n_colors, show_chart, size):
    from sklearn.cluster import KMeans
    from collections import Counter

    modified_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=n_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if show_chart:
        plt.figure(figsize=(8, 6))
        plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
        plt.show()

    return rgb_colors


def match_image_by_color(image, color, threshold=60, n_colors=10, size=(128, 32)):
    from skimage.color import rgb2lab, deltaE_cie76

    image_colors = get_colors(image, n_colors, False, size)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    selected_image = False
    for i in range(n_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if diff < threshold:
            selected_image = True

    return selected_image


def show_selected_images(images, color, threshold, colors_to_match):
    index = 1
    for i in range(len(images)):
        selected = match_image_by_color(images[i], color, threshold, colors_to_match)
        if selected:
            # image_ = cv2.resize(images[i], (1920, 1080))
            # cv2.imshow("image_{}".format(i), image_)
            # cv2.waitKey(0)
            plt.subplot(1, 5, index)
            plt.imshow(images[i])
            index += 1


def colors_dict():
    COLORS = {
        # 'RED_128': [128, 0, 0],
        'GREEN_128': [0, 128, 0],
        # 'BLUE_128': [0, 0, 128],
        # 'RED_255': [255, 0, 0],
        'GREEN_255': [0, 255, 0],
        # 'BLUE_255': [0, 0, 255],
        # 'YELLOW_128': [128, 128, 0],
        'CYAN_128': [0, 128, 128],
        # 'MAGENTA_128': [128, 0, 128],
        # 'YELLOW_255': [255, 255, 0],
        'CYAN_255': [0, 255, 255],
        # 'MAGENTA_255': [255, 0, 255],
        'BLACK': [0, 0, 0],
        # 'GRAY': [128, 128, 128],
        'WHITE': [255, 255, 255]
    }

    return COLORS


def identify_colors(img, COLORS, THRESHOLD=60, N_COLORS=5, SIZE=(32, 16)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    RES = {}
    for k in COLORS:
        selected = match_image_by_color(img, COLORS[k], THRESHOLD, N_COLORS, SIZE)
        RES[k] = selected

    return RES


def color_identify(data_path):
    """
    An example.
    :param data_path:
    :return:
    """
    off_path = os.path.abspath(os.path.join(data_path, "../..")) + "/cropped_on_or_off/off"
    on_path = os.path.abspath(os.path.join(data_path, "../..")) + "/cropped_on_or_off/on"
    unsure_path = os.path.abspath(os.path.join(data_path, "../..")) + "/cropped_on_or_off/unsure"
    os.makedirs(off_path, exist_ok=True)
    os.makedirs(on_path, exist_ok=True)
    os.makedirs(unsure_path, exist_ok=True)

    COLORS = colors_dict()

    img_list = os.listdir(data_path)
    for img in img_list:
        img_abs_path = data_path + "/{}".format(img)
        img = cv2.imread(img_abs_path)
        RES = identify_colors(img, COLORS, THRESHOLD=60, N_COLORS=5, SIZE=(32, 16))
        if RES["WHITE"] and not RES["GREEN_128"] and not RES["GREEN_255"]:
            img_dst_path = off_path + "/{}".format(img)
            shutil.copy(img_abs_path, img_dst_path)
            print("{}: {} --> {}".format("OFF", img_abs_path, img_dst_path))
        elif RES["GREEN_128"] and RES["GREEN_255"] and not RES["WHITE"]:
            img_dst_path = on_path + "/{}".format(img)
            shutil.copy(img_abs_path, img_dst_path)
            print("{}: {} --> {}".format("ON", img_abs_path, img_dst_path))
        else:
            img_dst_path = unsure_path + "/{}".format(img)
            shutil.copy(img_abs_path, img_dst_path)
            print("{}: {} --> {}".format("ON", img_abs_path, img_dst_path))
    

def cal_images_mean_height_width(data_path):
    img_list = sorted(os.listdir(data_path))

    hs, ws = [], []

    for img in img_list:
        img_abs_path = data_path + "/{}".format(img)
        img = cv2.imread(img_abs_path)
        h, w = img.shape[:2]
        hs.append(h)
        ws.append(w)

    h_mean = np.mean(hs)
    w_mean = np.mean(ws)

    print(h_mean)  # 511.35578569681155
    print(w_mean)  # 478.03767430481935

    return h_mean, w_mean


# OCR ===================================================
class SegDetectorRepresenter():
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, batch, pred, is_output_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, H, W)
        '''
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        # for batch_index in range(pred.size(0)):  # train
        for batch_index in range(pred.shape[0]):  # inference
            height, width = batch['shape'][batch_index]
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch
    
    def binarize(self, pred):
        return pred > self.thresh

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height, onnx_flag=True):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        # bitmap = _bitmap.cpu().numpy()  # The first channel
        # pred = pred.cpu().detach().numpy()

        # inference
        if onnx_flag:
            bitmap = _bitmap  # The first channel
            pred = pred
        else:
            bitmap = _bitmap.cpu().numpy()  # The first channel
            pred = pred.cpu().detach().numpy()

        # ## train
        # bitmap = _bitmap.cpu().numpy()  # The first channel
        # pred = pred.cpu().detach().numpy()


        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height, onnx_flag=True):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2

        # # inference
        if onnx_flag:
            bitmap = _bitmap  # The first channel
            pred = pred
        else:
            bitmap = _bitmap.cpu().numpy()  # The first channel
            pred = pred.cpu().detach().numpy()

        # # ## train
        # bitmap = _bitmap.cpu().numpy()  # The first channel
        # pred = pred.cpu().detach().numpy()

        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue
            # print('===points:', points)
            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores
    
    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int_), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int_), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int_), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int_), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def select_horizontal_vertical_images(data_path, flag="horizontal", mvcp="move", r=1.0):
    file_list = sorted(os.listdir(data_path))
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_selected_horizontal_images".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        imgsz = img.shape[:2]
        # if imgsz[0] > imgsz[1]:
        if flag == "horizontal" or flag == "h" or flag == "H":
            if imgsz[0] * r <= imgsz[1]:
                f_dst_path = save_path + "/{}".format(f)
                if mvcp == "copy" or mvcp == "cp":
                    shutil.copy(f_abs_path, f_dst_path)
                elif mvcp == "move" or mvcp == "mv":
                    shutil.move(f_abs_path, f_dst_path)
                elif mvcp == "delete" or mvcp == "del":
                    os.remove(f_abs_path)
        elif flag == "vertical" or flag == "v" or flag == "V":
            if imgsz[0] >= imgsz[1] * r:
                f_dst_path = save_path + "/{}".format(f)
                if mvcp == "copy" or mvcp == "cp":
                    shutil.copy(f_abs_path, f_dst_path)
                elif mvcp == "move" or mvcp == "mv":
                    shutil.move(f_abs_path, f_dst_path)
                elif mvcp == "delete" or mvcp == "del":
                    os.remove(f_abs_path)
        else:
            print("Error!")
            


def draw_bbox(img, result, color=(0, 0, 255), thickness=2):
    for point in result:
        point = point.astype(int)
        cv2.polylines(img, [point], True, color, thickness)
    return img


# def softmax(x):
#     exps = np.exp(x - np.max(x))
#     return exps / np.sum(exps)


def expand_kpt(imgsz, pts, r):
    minSide = min(imgsz[0], imgsz[1])
    if minSide > 400:
        minSide = minSide / 5
    elif minSide > 300:
        minSide = minSide / 4
    elif minSide > 200:
        minSide = minSide / 3
    elif minSide > 100:
        minSide = minSide / 2
    else:
        minSide = minSide

    expandP = round(minSide * r)
    expandP_half = round(minSide * r / 2)
    expandP_quarter = round(minSide * r / 4)
    expandP_one_sixth = round(minSide * r / 6)
    expandP_one_eighth = round(minSide * r / 8)

    for i in range(len(pts)):
        if pts[i][0] - expandP >= 0:
            if i == 0 or i == 3:
                pts[i][0] = pts[i][0] - expandP
            else:
                pts[i][0] = pts[i][0] + expandP
        elif pts[i][0] - expandP_half >= 0:
            if i == 0 or i == 3:
                pts[i][0] = pts[i][0] - expandP_half
            else:
                pts[i][0] = pts[i][0] + expandP_half
        elif pts[i][0] - expandP_quarter >= 0:
            if i == 0 or i == 3:
                pts[i][0] = pts[i][0] - expandP_quarter
            else:
                pts[i][0] = pts[i][0] + expandP_quarter
        elif pts[i][0] - expandP_one_sixth >= 0:
            if i == 0 or i == 3:
                pts[i][0] = pts[i][0] - expandP_one_sixth
            else:
                pts[i][0] = pts[i][0] + expandP_one_sixth
        elif pts[i][0] - expandP_one_eighth >= 0:
            if i == 0 or i == 3:
                pts[i][0] = pts[i][0] - expandP_one_eighth
            else:
                pts[i][0] = pts[i][0] + expandP_one_eighth
        else:
            pts[i][0] = pts[i][0]

        if pts[i][1] - expandP >= 0:
            if i == 0 or i == 1:
                pts[i][1] = pts[i][1] - expandP
            else:
                pts[i][1] = pts[i][1] + expandP
        elif pts[i][1] - expandP_half >= 0:
            if i == 0 or i == 1:
                pts[i][1] = pts[i][1] - expandP_half
            else:
                pts[i][1] = pts[i][1] + expandP_half
        elif pts[i][1] - expandP_quarter >= 0:
            if i == 0 or i == 1:
                pts[i][1] = pts[i][1] - expandP_quarter
            else:
                pts[i][1] = pts[i][1] + expandP_quarter
        elif pts[i][1] - expandP_one_sixth >= 0:
            if i == 0 or i == 1:
                pts[i][1] = pts[i][1] - expandP_one_sixth
            else:
                pts[i][1] = pts[i][1] + expandP_one_sixth
        elif pts[i][1] - expandP_one_eighth >= 0:
            if i == 0 or i == 1:
                pts[i][1] = pts[i][1] - expandP_one_eighth
            else:
                pts[i][1] = pts[i][1] + expandP_one_eighth
        else:
            pts[i][1] = pts[i][1]

    for i in range(len(pts)):
        pts[i][0] = int(round(pts[i][0]))
        pts[i][1] = int(round(pts[i][1]))

    return pts


def cal_hw(b):
    MIN_X = 1e6
    MAX_X = -1e6
    MIN_Y = 1e6
    MAX_Y = -1e6

    for bi in b:
        if bi[0] <= MIN_X:
            MIN_X = bi[0]
        if bi[0] >= MAX_X:
            MAX_X = bi[0]
        if bi[1] <= MIN_Y:
            MIN_Y = bi[1]
        if bi[1] >= MAX_Y:
            MAX_Y = bi[1]

    h = int(round(abs(MAX_Y - MIN_Y)))
    w = int(round(abs(MAX_X - MIN_X)))
    return (h, w)


def get_new_boxes(boxes, rhw, r=0.12):
    boxes_orig = []
    for bi in boxes:
        bi_ = []
        for bj in bi:
            bi_orig = [bj[0] / rhw[1], bj[1] / rhw[0]]
            bi_.append(bi_orig)
        boxes_orig.append(bi_)

    boxes_new = []
    for bbi in boxes_orig:
        # x1, x2 = round(min(bi[0], bi[0])), round(max(bi[0], bi[0]))
        # y1, y2 = round(min(bi[1], bi[1])), round(max(bi[1], bi[1]))
        # basesz = (abs(y2 - y1), abs(x2 - x1))
        basesz = cal_hw(bbi)
        bi_ = expand_kpt(basesz, bbi, r)
        boxes_new.append(bi_)

    return boxes_new


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)


def contain_chinese(string):
    pattern = r'[\u4e00-\u9fff]'
    return re.search(pattern, string) is not None


def process_sliding_window_results(res):
    # TODO
    final_res = ""
    for i, resi in enumerate(res):
        if i == 0:
            final_res += resi
        else:
            resi_new = resi
            for j in range(len(resi)):
                if len(resi) >= j + 1 and len(final_res) >= j + 1:
                    if resi[0:j + 1] == final_res[-(j + 1):]:
                        resi_new = resi[j + 1:]
            final_res += resi_new

    return final_res


def get_label(img_name):
    label = ""
    if "=" in img_name:
        equal_num = img_name.count("=")
        if equal_num > 1:
            print("equal_num > 1!")
        else:
            # label = img_name.split("=")[-1]

            img_name_r = img_name[::-1]
            idx_r = img_name_r.find("=")
            idx = -(idx_r + 1)
            label = img_name[(idx + 1):]

    return label


def get_alpha(flag="digits_19"):
    global alpha

    if flag == "digits_15":
        # alpha = ' ' + '0123456789.' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # alpha = ' ' + '0123456789.-:' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        alpha = ' ' + '0123456789.' + 'AbC'
    elif flag == "digits_19":
        alpha = ' ' + '0123456789' + '.:/\\-' + 'AbC'
    elif flag == "digits_20":
        alpha = ' ' + '0123456789' + '.:/\\-' + 'ABbC'
    elif flag == "digits_26":
        alpha = ' ' + '0123456789' + '.:/\\-' + 'AbC' + '℃' + 'MPa' + '㎡m³'
    elif flag == "Chinese1":
        CH_SIM_CHARS = ' ' + '0123456789.' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        CH_SIM_CHARS += ',;~!@#$%^&*()_+-={}:"<>?-=[]/|\\' + "'"
        CH_SIM_CHARS += '、。┅《》「」【】¥®πи‰℃№Ⅱ←↑→↓①②③④▪☆❤'
        print(len(CH_SIM_CHARS))

        ch_sim_chars = open("words/ch_sim_char.txt", "r", encoding="utf-8")
        lines = ch_sim_chars.readlines()
        for l in lines:
            CH_SIM_CHARS += l.strip()
        alpha = CH_SIM_CHARS  # len = 6738  7568
    elif flag == "Chinese_6867":
        CH_SIM_CHARS = ' '
        ch_sim_chars = open("words/chinese_simple_with_special_chars.txt", "r", encoding="utf-8")
        lines = ch_sim_chars.readlines()
        for l in lines:
            CH_SIM_CHARS += l.strip()
        alpha = CH_SIM_CHARS  # len = 6867
    elif flag == "ppocr_6625":
        CH_SIM_CHARS = ' '
        ch_sim_chars = open("words/ppocr_keys_v1.txt", "r", encoding="utf-8")
        lines = ch_sim_chars.readlines()
        for l in lines:
            CH_SIM_CHARS += l.strip()
        alpha = CH_SIM_CHARS
    else:
        raise NotImplementedError

    return alpha


def resize_norm_padding_img(img, imgsz, max_wh_ratio):
    # max_wh_ratio: 320 / 48
    imgC, imgH, imgW = imgsz
    assert imgC == img.shape[2]
    imgW = int((imgH * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def putText_Chinese(img_pil, p, string, color=(255, 0, 255)):
    from PIL import ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageFile

    # img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('./utils/gen_fake/Fonts/chinese_2/仿宋_GB2312.ttf', 20)
    draw.text(p, string, font=font, fill=color)
    # img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_pil


def draw_e2e_res(image, boxes, txts, font_path="utils/gen_fake/Fonts/chinese_2/楷体_GB2312.ttf"):
    from PIL import ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageFile

    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))

    font = ImageFont.truetype(font_path, 15, encoding="utf-8")
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        box = np.array(box)
        box = [tuple(x) for x in box]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(box, outline=color)
        draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))

    return np.array(img_show)[:, :, ::-1]


class GKFOCR(object):
    """
    input support: 1.image path
    2024.09.14
    """

    def __init__(self, cfg_path: str = "configs/cfg_gkfocr.yaml", debug: bool = False):
        with open(cfg_path, errors='ignore') as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.m_FLAG_DeBug = debug
        self.alpha = get_alpha(flag="Chinese_6867")  # digits Chinese

        self.det_model_path = self.cfg["det"]["model_path"]
        self.rec_model_path = self.cfg["rec"]["model_path"]
        self.det_input_shape = eval(self.cfg["det"]["input_shape"])
        self.rec_input_shape = eval(self.cfg["rec"]["input_shape"])
        self.det_mean = eval(self.cfg["det"]["mean"])
        self.det_std = eval(self.cfg["det"]["std"])
        self.rec_mean = eval(self.cfg["rec"]["mean"])
        self.rec_std = eval(self.cfg["rec"]["std"])

        self.det_ort_session = self.init_model(self.det_model_path)
        print("Load det model: {}\tSuccessful".format(self.det_model_path))
        self.rec_ort_session = self.init_model(self.rec_model_path)
        print("Load rec model: {}\tSuccessful".format(self.rec_model_path))

        self.det_thresh = float(self.cfg["det"]["thresh"])
        self.det_box_thresh = float(self.cfg["det"]["box_thresh"])
        self.det_max_candidates = float(self.cfg["det"]["max_candidates"])
        self.det_unclip_ratio = float(self.cfg["det"]["unclip_ratio"])

        self.rec_make_border_flag = bool(self.cfg["rec"]["make_border_flag"])
        self.rec_batch_first = bool(self.cfg["rec"]["batch_first"])
        self.rec_ppocr_flag = bool(self.cfg["rec"]["ppocr_flag"])
        self.rec_c = int(self.cfg["rec"]["c"])
        self.rec_r1 = float(self.cfg["rec"]["r1"])
        self.rec_r2 = float(self.cfg["rec"]["r2"])
        self.rec_sliding_window_flag = bool(self.cfg["rec"]["sliding_window_flag"])
        self.rec_color = eval(self.cfg["rec"]["color"])
        self.rec_gap_r = eval(self.cfg["rec"]["gap_r"])
        self.rec_medianblur_flag = bool(self.cfg["rec"]["medianblur_flag"])
        self.rec_k = int(self.cfg["rec"]["k"])
        self.rec_clahe_flag = bool(self.cfg["rec"]["clahe_flag"])
        self.rec_clipLimit = int(self.cfg["rec"]["clipLimit"])
        self.rec_score_thr = float(self.cfg["rec"]["score_thr"])

    def init_model(self, model_path: str):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        return ort_session

    def inference(self, data):
        if isinstance(data, str):
            if os.path.isfile(data):
                img = cv2.imread(data)
                img_cp = img.copy()
                mask, boxes, scores, draw_img_resize, boxs_new = self.det_inference(img)
                txts = self.rec_inference_v2(img_cp, boxs_new)
                out_img = draw_e2e_res(img_cp, boxs_new, txts, font_path=self.cfg["chinese_font_path"])
                return out_img
            elif os.path.isdir(data):
                dirname = os.path.basename(data)
                save_path = os.path.abspath(os.path.join(data, "../{}_output".format(dirname)))
                os.makedirs(save_path, exist_ok=True)

                file_list = sorted(os.listdir(data))
                for f in tqdm(file_list):
                    fname = os.path.splitext(f)[0]
                    f_abs_path = data + "/{}".format(f)
                    img = cv2.imread(f_abs_path)
                    img_cp = img.copy()
                    mask, boxes, scores, draw_img_resize, boxs_new = self.det_inference(img)
                    txts = self.rec_inference(img_cp, boxs_new, save_path, fname)

                    if self.m_FLAG_DeBug:
                        pred_mask_path = save_path + "/{}_pred_mask.jpg".format(fname)
                        cv2.imwrite(pred_mask_path, mask * 255)

                    out_img = draw_e2e_res(img_cp, boxs_new, txts, font_path=self.cfg["chinese_font_path"])
                    cv2.imwrite("{}/{}_out_img.jpg".format(save_path, fname), out_img)
                return None
            else:
                print("data should be test image file path or directory path!")
        elif isinstance(data, np.ndarray) or isinstance(data, Image.Image):
            if isinstance(data, np.ndarray):
                out_img = self.inference_one_array(data)
                return out_img
            else:
                out_img = self.inference_one_array(np.asarray(data))
                return out_img
        else:
            print("data should be: 1.test image file path. 2. test image directory path. 3. test image np.ndarray or PIL.Image.Image!")

    def inference_one_array(self, img):
        img_cp = img.copy()
        mask, boxes, scores, draw_img_resize, boxs_new = self.det_inference(img)
        txts = self.rec_inference_v2(img_cp, boxs_new)
        out_img = draw_e2e_res(img_cp, boxs_new, txts, font_path=self.cfg["chinese_font_path"])
        return out_img

    def det_inference(self, img):
        imgsz_orig = img.shape[:2]
        rhw = (self.det_input_shape[0] / imgsz_orig[0], self.det_input_shape[1] / imgsz_orig[1])

        img, img_resize = self.det_preprocess(img)
        outputs = self.det_ort_session.run(None, {'input': img})
        mask, boxes, scores = self.det_postprocess(outputs)

        draw_img = draw_bbox(img_resize, boxes)
        draw_img_resize = cv2.resize(draw_img, imgsz_orig[::-1])
        boxs_new = get_new_boxes(boxes, rhw, r=0.12)

        return mask, boxes, scores, draw_img_resize, boxs_new
    
    def det_preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.det_input_shape)
        img_resize = img.copy()
        img = (img / 255. - np.array(self.det_mean)) / np.array(self.det_std)
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        return img, img_resize
    
    def det_postprocess(self, outputs, is_output_polygon=False):
        b, c, h, w = outputs[0].shape
        mask = outputs[0][0, 0, ...]
        batch = {'shape': [(h, w)]}

        box_list, score_list = SegDetectorRepresenter(thresh=self.det_thresh, box_thresh=self.det_box_thresh, max_candidates=self.det_max_candidates, unclip_ratio=self.det_unclip_ratio)(batch, outputs[0], is_output_polygon)
        box_list, score_list = box_list[0], score_list[0]

        if len(box_list) > 0:
            if is_output_polygon:
                idx = [x.sum() > 0 for x in box_list]
                box_list = [box_list[i] for i, v in enumerate(idx) if v]
                score_list = [score_list[i] for i, v in enumerate(idx) if v]
            else:
                idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []

        return mask, box_list, score_list

    def rec_inference(self, img, boxes, save_path, fname):
        txts = []
        mask_vis = np.zeros(shape=img.shape, dtype=np.uint8)
        mask_vis_pil = Image.fromarray(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB))
        for b in boxes:
            try:
                dstsz = cal_hw(b)
                warpped = perspective_transform(b, dstsz, img)

                makeBorderRes = makeBorder_inference(warpped, new_shape=self.rec_input_shape, sliding_window=self.rec_sliding_window_flag, gap_r=self.rec_gap_r)
                pred, score = self.rec_inference_one(makeBorderRes)
                txts.append(pred)

                if self.m_FLAG_DeBug:
                    print("pred: {}\tscore: {}".format(pred, score))
                    cv2.imwrite("{}/{}_cropped_img={}.jpg".format(save_path, fname, pred), warpped)

                p0 = tuple(map(int, b[0]))
                mask_vis_pil = putText_Chinese(mask_vis_pil, p0, pred, color=(255, 0, 255))

            except Exception as Error:
                print(Error)

        if self.m_FLAG_DeBug:
            mask_vis = cv2.cvtColor(np.array(mask_vis_pil), cv2.COLOR_RGB2BGR)
            cv2.imwrite("{}/{}_vis_results.jpg".format(save_path, fname), mask_vis)

        return txts
    
    def rec_inference_v2(self, img, boxes):
        txts = []
        mask_vis = np.zeros(shape=img.shape, dtype=np.uint8)
        mask_vis_pil = Image.fromarray(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB))
        for b in boxes:
            try:
                dstsz = cal_hw(b)
                warpped = perspective_transform(b, dstsz, img)

                makeBorderRes = makeBorder_inference(warpped, new_shape=self.rec_input_shape, sliding_window=self.rec_sliding_window_flag, gap_r=self.rec_gap_r)
                pred, score = self.rec_inference_one(makeBorderRes)
                txts.append(pred)

                p0 = tuple(map(int, b[0]))
                mask_vis_pil = putText_Chinese(mask_vis_pil, p0, pred, color=(255, 0, 255))

            except Exception as Error:
                print(Error)

        return txts
    
    def rec_inference_one(self, img):
        img = self.rec_preprocess(img)
        ort_outs = self.rec_ort_session.run(["output"], {self.rec_ort_session.get_inputs()[0].name: img})
        pred, scores_mean = self.rec_postprocess(ort_outs[0])
        return pred, scores_mean

    def rec_preprocess(self, img):
        """
        """
        if self.rec_medianblur_flag:
            img = median_blur(img, k=self.rec_k)
        if self.rec_clahe_flag:
            img = clahe(img, clipLimit=self.rec_clipLimit)

        imgsz = (self.rec_c, self.rec_input_shape[0], self.rec_input_shape[1])

        if self.rec_ppocr_flag:
            max_wh_ratio = self.rec_input_shape[1] / self.rec_input_shape[0]
            img = resize_norm_padding_img(img, imgsz=imgsz, max_wh_ratio=max_wh_ratio)
            img = img[np.newaxis, :].astype(np.float32)
        else:
            imgsz_ = img.shape[:2]
            if imgsz_ != self.rec_input_shape:
                img = cv2.resize(img, self.rec_input_shape[::-1])
            img = (img / 255. - np.array(self.rec_mean)) / np.array(self.rec_std)
            img = img.transpose(2, 0, 1)
            img = img[np.newaxis, :].astype(np.float32)

        return img
    
    def rec_postprocess(self, pred):
        res = []
        scores = []

        if self.rec_batch_first:
            for i in range(pred.shape[1]):
                argmax_i = np.argmax(pred[0][i])
                res.append(argmax_i)

                sc_ = softmax(pred[0][i])
                sc = sc_[1:]
                max_ = max(sc)
                if max_ >= self.rec_score_thr:
                    scores.append(max_)
        else:
            for i in range(pred.shape[0]):
                argmax_i = np.argmax(pred[i][0])
                res.append(argmax_i)

                sc_ = softmax(pred[i][0])
                sc = sc_[1:]
                max_ = max(sc)
                if max_ >= self.rec_score_thr:
                    scores.append(max_)

        scores_mean = np.mean(scores)

        pred_ = [self.alpha[class_id] for class_id in res]
        pred_ = [k for k, g in itertools.groupby(list(pred_))]
        pred = ''.join(pred_).replace(' ', '')

        return pred, scores_mean


def main_gkfocr():
    data = "data/doc/imgs"
    # data = "data/doc/imgs/11.jpg"
    ocr = GKFOCR(cfg_path="configs/cfg_gkfocr.yaml", debug=False)
    out_img = ocr.inference(data)
    if out_img is not None:
        cv2.imwrite("data/doc/test_out_img.jpg", out_img)


def cal_distance(p1, p2):
    dis = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dis


def cal_similar_height_width(rect):
    """
    top left --> top right --> bottom right --> bottom left
    """
    dis01 = cal_distance(rect[0], rect[1])
    dis12 = cal_distance(rect[1], rect[2])
    dis23 = cal_distance(rect[2], rect[3])
    dis30 = cal_distance(rect[3], rect[0])

    sh = int(max(dis12, dis30))
    sw = int(max(dis01, dis23))

    return (sh, sw)


def convert_mtwi_to_ocr_rec_data(data_path):
    save_path = data_path + "/rec_cropped"
    os.makedirs(save_path, exist_ok=True)

    img_path = data_path + "/image_train"
    txt_path = data_path + "/txt_train"

    file_list = sorted(os.listdir(img_path))
    for f in tqdm(file_list):
        fname = os.path.splitext(f)[0]
        f_abs_path = img_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        # print(f)
        if img is None: continue
        imgsz = img.shape[:2]

        txt_abs_path = txt_path + "/{}.txt".format(fname)

        with open(txt_abs_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                pos, label = line.split(",")[:8], line.split(",")[-1]
                pos = list(map(float, pos))
                pos = list(map(round, pos))
                pos = list(map(int, pos))
                # pos = np.array([[pos[0], pos[1]], [pos[2], pos[3]], [pos[4], pos[5]], [pos[6], pos[7]]])
                pos = np.array([[pos[0], pos[1]], [pos[6], pos[7]], [pos[4], pos[5]], [pos[2], pos[3]]])
                similar_hw = cal_similar_height_width(pos)
                warped = perspective_transform(pos, similar_hw, img)
                save_path_i = save_path + "/{}_{}={}.jpg".format(fname, i, label)
                cv2.imwrite(save_path_i, warped)


def convert_ShopSign_to_ocr_rec_data(data_path):
    save_path = data_path + "/rec_cropped"
    os.makedirs(save_path, exist_ok=True)

    img_path = data_path + "/images"
    txt_path = data_path + "/labels"

    file_list = sorted(os.listdir(img_path))
    for f in tqdm(file_list):
        fname = os.path.splitext(f)[0]
        f_abs_path = img_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        # print(f)
        if img is None: continue
        imgsz = img.shape[:2]

        txt_abs_path = txt_path + "/{}.txt".format(fname.replace("image", "gt_img"))

        with open(txt_abs_path, "r", encoding="gbk") as fr:
            lines = fr.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                pos, label = line.split(",")[:8], line.split(",")[-1]
                pos = list(map(float, pos))
                pos = list(map(round, pos))
                pos = list(map(int, pos))
                pos = np.array([[pos[0], pos[1]], [pos[2], pos[3]], [pos[4], pos[5]], [pos[6], pos[7]]])
                # pos = np.array([[pos[0], pos[1]], [pos[6], pos[7]], [pos[4], pos[5]], [pos[2], pos[3]]])
                similar_hw = cal_similar_height_width(pos)
                warped = perspective_transform(pos, similar_hw, img)
                save_path_i = save_path + "/{}_{}={}.jpg".format(fname, i, label)
                cv2.imwrite(save_path_i, warped)


def gen_dbnet_train_test_txt(data_path, data_type="test"):
    img_path = data_path + "/{}images".format(data_type)
    gt_path = data_path + "/{}gts".format(data_type)

    img_list = sorted(os.listdir(img_path))
    gt_list = sorted(os.listdir(gt_path))

    with open(data_path + "/{}images_list.txt".format(data_type), "w", encoding="utf-8") as fw:
        for s in tqdm(img_list):
            sname = os.path.splitext(s)[0]
            s_img_path = img_path + "/{}".format(s)
            s_gt_path = gt_path + "/{}.gt".format(sname)

            # s_img_path = "/{}".format(sname)
            # s_gt_path = "/{}.gt".format(sname)
            l = "{}\t{}\n".format(s_img_path, s_gt_path)
            fw.write(l)


def aug_dbnet_data(data_path, bg_path, maxnum=20000):
    img_path = data_path + "/images"
    mask_path = data_path + "/masks_vis"
    save_path = data_path + "/output"
    save_img_path = save_path + "/images"
    save_gts_path = save_path + "/gts"
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_gts_path, exist_ok=True)

    # dbnet_gt_path = os.path.abspath(os.path.join(data_path, "..")) + "/kpt/gts"
    dbnet_gt_path = data_path + "/gts"

    img_list = sorted(os.listdir(img_path))
    bg_list = sorted(os.listdir(bg_path))

    N = 0
    for bg in tqdm(bg_list):
        if N >= maxnum:
            break

        bg_name = os.path.splitext(bg)[0]
        bg_abs_path = bg_path + "/{}".format(bg)
        bgimg = cv2.imread(bg_abs_path)
        bgsz = bgimg.shape[:2]

        rdmN = np.random.randint(5, 50)
        img_list_selected = random.sample(img_list, rdmN)

        for img in img_list_selected:
            try:
                img_name = os.path.splitext(img)[0]
                img_abs_path = img_path + "/{}".format(img)
                mask_abs_path = mask_path + "/{}.png".format(img_name)
                gt_abs_path = dbnet_gt_path + "/{}.gt".format(img_name)

                img = cv2.imread(img_abs_path)
                imgsz = img.shape[:2]
                maskimg = cv2.imread(mask_abs_path)

                rdmnum = np.random.random()
                if imgsz[0] > 3000 and imgsz[1] > 3000:
                    if rdmnum < 0.25:
                        img = cv2.resize(img, (imgsz[1] // 2, imgsz[0] // 2))
                        maskimg = cv2.resize(maskimg, (imgsz[1] // 2, imgsz[0] // 2))
                    elif rdmnum > 0.75:
                        img = cv2.resize(img, (imgsz[1] // 4, imgsz[0] // 4))
                        maskimg = cv2.resize(maskimg, (imgsz[1] // 4, imgsz[0] // 4))
                else:
                    if rdmnum < 0.45:
                        img = cv2.resize(img, (imgsz[1] // 2, imgsz[0] // 2))
                        maskimg = cv2.resize(maskimg, (imgsz[1] // 2, imgsz[0] // 2))

                outimg_crop, bbox, relative_roi = seg_crop_object(img, bgimg, maskimg)

                with open(gt_abs_path, "r", encoding="utf-8") as fo:
                    lines = fo.readlines()
                    assert len(lines) == 1, "{}: lines > 1!".format(gt_abs_path)
                    for line in lines:
                        # line = line.strip().split(", ")[:8]
                        # line = list(map(float, line))
                        # relative_points_x = np.array(line[::2]) - bbox[0]
                        # relative_points_y = np.array(line[1::2]) - bbox[1]

                        if imgsz[0] > 3000 and imgsz[1] > 3000:
                            if rdmnum < 0.25:
                                line = line.strip().split(", ")[:8]
                                line = list(map(float, line))
                                line = np.array(line) / 2

                                relative_points_x = np.array(line[::2]) - bbox[0]
                                relative_points_y = np.array(line[1::2]) - bbox[1]
                            elif rdmnum > 0.75:
                                line = line.strip().split(", ")[:8]
                                line = list(map(float, line))
                                line = np.array(line) / 4
                                relative_points_x = np.array(line[::2]) - bbox[0]
                                relative_points_y = np.array(line[1::2]) - bbox[1]
                            else:
                                line = line.strip().split(", ")[:8]
                                line = list(map(float, line))
                                relative_points_x = np.array(line[::2]) - bbox[0]
                                relative_points_y = np.array(line[1::2]) - bbox[1]
                        else:
                            if rdmnum < 0.45:
                                line = line.strip().split(", ")[:8]
                                line = list(map(float, line))
                                line = np.array(line) / 2
                                relative_points_x = np.array(line[::2]) - bbox[0]
                                relative_points_y = np.array(line[1::2]) - bbox[1]
                            else:
                                line = line.strip().split(", ")[:8]
                                line = list(map(float, line))
                                relative_points_x = np.array(line[::2]) - bbox[0]
                                relative_points_y = np.array(line[1::2]) - bbox[1]

                paste_rdm_pos = (np.random.randint(0, (bgsz[1] - bbox[2])), np.random.randint(0, (bgsz[0] - bbox[3])))

                new_roi = (relative_roi[0] + paste_rdm_pos[1], relative_roi[1] + paste_rdm_pos[0])

                bgcp = bgimg.copy()
                bgcp[new_roi] = (0, 0, 0)

                bgcp_crop = bgcp[paste_rdm_pos[1]:(paste_rdm_pos[1] + bbox[3]), paste_rdm_pos[0]:(paste_rdm_pos[0] + bbox[2])]
                merged = outimg_crop + bgcp_crop

                bg1 = bgimg[0:paste_rdm_pos[1], 0:bgsz[1]]
                bg2 = bgimg[paste_rdm_pos[1]:(paste_rdm_pos[1] + bbox[3]), 0:paste_rdm_pos[0]]
                bg3 = merged
                bg4 = bgimg[(paste_rdm_pos[1]):(paste_rdm_pos[1] + bbox[3]), (paste_rdm_pos[0] + bbox[2]):bgsz[1]]
                bg5 = bgimg[(paste_rdm_pos[1] + bbox[3]):bgsz[0], 0:bgsz[1]]
                bg_mid = np.hstack((bg2, bg3, bg4))
                bg_final = np.vstack((bg1, bg_mid, bg5))

                cv2.imwrite("{}/{}_{}.jpg".format(save_img_path, bg_name, img_name), bg_final)

                new_points_x = relative_points_x + paste_rdm_pos[0]
                new_points_y = relative_points_y + paste_rdm_pos[1]
                new_points = np.vstack((new_points_x, new_points_y))
                new_points = new_points.T.reshape(1, -1)[0]

                gt_abs_path = save_gts_path + "/{}_{}.gt".format(bg_name, img_name)
                with open(gt_abs_path, "w", encoding="utf-8") as fw:
                    content = ", ".join(str(p) for p in new_points) + ", 0\n"
                    fw.write(content)

                print(new_points)
                N += 1

            except Exception as Error:
                print(Error)


def vis_dbnet_gt(data_path):
    img_path = data_path + "/images"
    gt_path = data_path + "/gts"
    vis_path = data_path + "/vis"
    os.makedirs(vis_path, exist_ok=True)

    gt_list = sorted(os.listdir(gt_path))
    for gt in tqdm(gt_list):
        gt_name = os.path.splitext(gt)[0]
        gt_abs_path = gt_path + "/{}".format(gt)
        img_abs_path = img_path + "/{}.jpg".format(gt_name)

        img = cv2.imread(img_abs_path)

        with open(gt_abs_path, "r", encoding="utf-8") as fo:
            lines = fo.readlines()
            for line in lines:
                line = line.strip().split(", ")[:8]
                line = list(map(int, map(round, map(float, line))))
                for j in range(0, 8, 2):
                    cv2.circle(img, (line[j], line[j + 1]), 4, (255, 0, 255), 2)

        cv2.imwrite("{}/{}.jpg".format(vis_path, gt_name), img)
        

def crop_ocr_rec_img_via_labelbee_det_json(data_path):
    dir_name = os.path.basename(data_path)
    img_path = data_path + "/images"
    json_path = data_path + "/jsons"

    cropped_path = data_path + "/{}_cropped".format(dir_name)
    det_images_path = data_path + "/{}_selected_images".format(dir_name)

    os.makedirs(cropped_path, exist_ok=True)
    os.makedirs(det_images_path, exist_ok=True)

    json_list = os.listdir(json_path)

    for j in tqdm(json_list):
        img_name = os.path.splitext(j.replace(".json", ""))[0]
        json_abs_path = json_path + "/{}".format(j)
        img_abs_path = img_path + "/{}".format(j.replace(".json", ""))
        img = cv2.imread(img_abs_path)
        json_ = json.load(open(json_abs_path, 'r', encoding='utf-8'))
        if not json_: continue
        w, h = json_["width"], json_["height"]

        result_ = json_["step_1"]["result"]
        if not result_: continue

        try:
            img_abs_path = img_path + "/{}".format(j.replace(".json", ""))
            shutil.copy(img_abs_path, det_images_path + "/{}".format(j.replace(".json", "")))
        except Exception as Error:
            print(Error)

        len_result = len(result_)
        for i in range(len_result):
            x_ = result_[i]["x"]
            y_ = result_[i]["y"]
            w_ = result_[i]["width"]
            h_ = result_[i]["height"]

            x_min = int(round(x_))
            x_max = int(round(x_ + w_))
            y_min = int(round(y_))
            y_max = int(round(y_ + h_))

            label = result_[i]["textAttribute"]

            try:
                cropped_img0 = img[y_min:y_max, x_min:x_max]
                cv2.imwrite("{}/{}_{}_{}={}.jpg".format(cropped_path, img_name, i, 0, label), cropped_img0)
                if "A" in label or "b" in label or "C" in label:
                    rdm_w = np.random.randint(55, 76)
                    alpha0 = cropped_img0[0:cropped_img0.shape[0], 0:rdm_w]
                    digital0 = cropped_img0[0:cropped_img0.shape[0], rdm_w:cropped_img0.shape[1]]
                    alpha0_label = label[0]
                    digital0_label = label[1:]
                    cv2.imwrite("{}/{}_{}_{}_alpha={}.jpg".format(cropped_path, img_name, i, 0, alpha0_label), alpha0)
                    cv2.imwrite("{}/{}_{}_{}_digital={}.jpg".format(cropped_path, img_name, i, 0, digital0_label), digital0)

            except Exception as Error:
                print(Error)

            try:
                cropped_img1 = img[y_min - np.random.randint(0, 4):y_max + np.random.randint(0, 4), x_min - np.random.randint(0, 4):x_max + np.random.randint(0, 4)]
                cv2.imwrite("{}/{}_{}_{}={}.jpg".format(cropped_path, img_name, i, 1, label), cropped_img1)
                if "A" in label or "b" in label or "C" in label:
                    rdm_w = np.random.randint(55, 76)
                    alpha0 = cropped_img1[0:cropped_img1.shape[0], 0:rdm_w]
                    digital0 = cropped_img1[0:cropped_img1.shape[0], rdm_w:cropped_img1.shape[1]]
                    alpha0_label = label[0]
                    digital0_label = label[1:]
                    cv2.imwrite("{}/{}_{}_{}_alpha={}.jpg".format(cropped_path, img_name, i, 1, alpha0_label), alpha0)
                    cv2.imwrite("{}/{}_{}_{}_digital={}.jpg".format(cropped_path, img_name, i, 1, digital0_label), digital0)

            except Exception as Error:
                print(Error)

            try:
                cropped_img2 = img[y_min - np.random.randint(0, 4):y_max - np.random.randint(0, 4), x_min - np.random.randint(0, 4):x_max - np.random.randint(0, 4)]
                cv2.imwrite("{}/{}_{}_{}={}.jpg".format(cropped_path, img_name, i, 2, label), cropped_img2)
                if "A" in label or "b" in label or "C" in label:
                    rdm_w = np.random.randint(55, 76)
                    alpha0 = cropped_img2[0:cropped_img2.shape[0], 0:rdm_w]
                    digital0 = cropped_img2[0:cropped_img2.shape[0], rdm_w:cropped_img2.shape[1]]
                    alpha0_label = label[0]
                    digital0_label = label[1:]
                    cv2.imwrite("{}/{}_{}_{}_alpha={}.jpg".format(cropped_path, img_name, i, 2, alpha0_label), alpha0)
                    cv2.imwrite("{}/{}_{}_{}_digital={}.jpg".format(cropped_path, img_name, i, 2, digital0_label), digital0)

            except Exception as Error:
                print(Error)

            try:
                cropped_img3 = img[y_min + np.random.randint(0, 4):y_max - np.random.randint(0, 4), x_min + np.random.randint(0, 4):x_max - np.random.randint(0, 4)]
                cv2.imwrite("{}/{}_{}_{}={}.jpg".format(cropped_path, img_name, i, 3, label), cropped_img3)
                if "A" in label or "b" in label or "C" in label:
                    rdm_w = np.random.randint(55, 76)
                    alpha0 = cropped_img3[0:cropped_img3.shape[0], 0:rdm_w]
                    digital0 = cropped_img3[0:cropped_img3.shape[0], rdm_w:cropped_img3.shape[1]]
                    alpha0_label = label[0]
                    digital0_label = label[1:]
                    cv2.imwrite("{}/{}_{}_{}_alpha={}.jpg".format(cropped_path, img_name, i, 3, alpha0_label), alpha0)
                    cv2.imwrite("{}/{}_{}_{}_digital={}.jpg".format(cropped_path, img_name, i, 3, digital0_label), digital0)

            except Exception as Error:
                print(Error)

            try:
                cropped_img4 = img[y_min + np.random.randint(0, 4):y_max + np.random.randint(0, 4), x_min + np.random.randint(0, 4):x_max + np.random.randint(0, 4)]
                cv2.imwrite("{}/{}_{}_{}={}.jpg".format(cropped_path, img_name, i, 4, label), cropped_img4)
                if "A" in label or "b" in label or "C" in label:
                    rdm_w = np.random.randint(55, 76)
                    alpha0 = cropped_img4[0:cropped_img4.shape[0], 0:rdm_w]
                    digital0 = cropped_img4[0:cropped_img4.shape[0], rdm_w:cropped_img4.shape[1]]
                    alpha0_label = label[0]
                    digital0_label = label[1:]
                    cv2.imwrite("{}/{}_{}_{}_alpha={}.jpg".format(cropped_path, img_name, i, 4, alpha0_label), alpha0)
                    cv2.imwrite("{}/{}_{}_{}_digital={}.jpg".format(cropped_path, img_name, i, 4, digital0_label), digital0)

            except Exception as Error:
                print(Error)

            
def convert_ICDAR_to_custom_format(data_path):
    dir_name = os.path.basename(data_path)
    train_or_test = "train"
    img_path = data_path + '/{}'.format(train_or_test)
    if train_or_test == "train":
        lbl_path = data_path + "/annotation.txt"
    elif train_or_test == "test":
        lbl_path = data_path + "/annotation_test.txt"
    else:
        print("Error")

    save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_renamed".format(train_or_test)
    os.makedirs(save_path, exist_ok=True)

    with open(lbl_path, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
        for line in lines:
            l = line.strip().split(" ")
            f_name = os.path.basename(l[0])
            img_abs_path = img_path + "/{}".format(f_name)
            label = " ".join([l[ii] for ii in range(1, len(l))])
            if "/" in label: continue
            img_name, suffix = os.path.splitext(f_name)[0], os.path.splitext(f_name)[1]
            img_dst_path = save_path + "/{}_{}_{}={}{}".format(dir_name, train_or_test, img_name, label, suffix)
            try:
                shutil.copy(img_abs_path, img_dst_path)
            except Exception as Error:
                print(Error)


def get_font_chars(font_path):
    from fontTools.ttLib import TTFont

    font = TTFont(font_path, fontNumber=0)
    glyph_names = font.getGlyphNames()
    char_list = []
    for idx, glyph in enumerate(glyph_names):
        if glyph[0] == '.':  # 跳过'.notdef', '.null'
            continue
        if glyph == 'union':
            continue
        if glyph[:3] == 'uni':
            glyph = glyph.replace('uni', '\\u')
        if glyph[:2] == 'uF':
            glyph = glyph.replace('uF', '\\u')
        if glyph == '\\uversal':
            continue

        char = json.loads("glyph")
        char_list.append(char)
    return char_list


def is_char_visible(font, char):
    from PIL import ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageFile

    """
    是否可见字符
    :param font:
    :param char:
    :return:
    """
    gray = Image.fromarray(np.zeros((20, 20), dtype=np.uint8))
    draw = ImageDraw.Draw(gray)
    draw.text((0, 0), char, 100, font=font)
    visible = np.max(np.array(gray)) > 0
    return visible


def get_all_font_chars(font_dir, word_set):
    from PIL import ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageFile

    font_path_list = [os.path.join(font_dir, font_name) for font_name in os.listdir(font_dir)]
    font_list = [ImageFont.truetype(font_path, size=10) for font_path in font_path_list]
    font_chars_dict = dict()
    for font, font_path in zip(font_list, font_path_list):
        font_chars = get_font_chars(font_path)
        # font_chars = [c.strip() for c in font_chars if len(c) == 1 and word_set.__contains__(c) and is_char_visible(font, c)]  # 可见字符
        font_chars = [c.strip() for c in font_chars if len(c) == 1 and word_set.__contains__(c)]  # 可见字符
        font_chars = list(set(font_chars))  # 去重
        font_chars.sort()
        font_chars_dict[font_path] = font_chars

    return font_chars_dict


def gen_background(imgsz):
    """
    生成背景;随机背景|纯色背景|合成背景
    :return:
    """
    # a = random.random()
    # pure_bg = np.ones((imgsz[0], imgsz[1], 3)) * np.array(random_color(0, 100))
    # random_bg = np.random.rand(imgsz[0], imgsz[1], 3) * 100
    # if a < 0.1:
    #     return random_bg
    # elif a < 0.8:
    #     return pure_bg
    # else:
    #     b = random.random()
    #     mix_bg = b * pure_bg + (1 - b) * random_bg
    #     return mix_bg

    a = random.random()
    pure_bg1 = np.zeros((imgsz[0], imgsz[1], 3))
    pure_bg2 = np.ones((imgsz[0], imgsz[1], 3)) * 255
    # if a < 0.5:
    #     return pure_bg1
    # else:
    #     return pure_bg2
    return pure_bg1
    # return pure_bg2


def horizontal_draw(draw, text, font, color, imgsz, char_w, char_h, easyFlag):
    """
    水平方向文字合成
    :param draw:
    :param text:
    :param font:
    :param color:
    :param char_w:
    :param char_h:
    :return:
    """
    text_w = len(text) * char_w
    h_margin = max(imgsz[0] - char_h, 1)
    w_margin = max(imgsz[1] - text_w, 1)

    # y_shift_high = h_margin - int(round(0.5 * char_h))
    # if y_shift_high < 0:
    #     y_shift_high = h_margin
    #
    # x_shift = np.random.randint(0, w_margin)
    # y_shift = np.random.randint(0, y_shift_high)
    # # y_shift = np.random.randint(0, 4)
    # # y_shift = np.random.randint(char_h + 5, self.imgsz[0] - char_h - 5)
    # y_shift_cp = y_shift
    # # x_shift = 20
    # # y_shift = 2

    x_shift = 9
    y_shift = 30 - (char_h // 2) - 5
    y_shift_cp = y_shift

    i = 0
    while i < len(text):
        draw.text((x_shift, y_shift), text[i], color, font=font)
        i += 1
        # x_shift += char_w + 0.25 * np.random.random() * np.random.randint(5, 9)
        # y_shift = y_shift_cp + 0.45 * np.random.randn()
        # y_shift = 2 + 0.3 * np.random.randn()

        # if easyFlag:
        #     x_shift += char_w + 5
        #     y_shift = y_shift_cp + np.random.rand()
        # else:
        #     x_shift += char_w + np.random.uniform(0, 1) * np.random.randint(5, 8)
        #     y_shift = y_shift_cp + 0.45 * np.random.randint(-5, 6)

        x_shift += char_w + 5
        y_shift = y_shift_cp

        # 如果下个字符超出图像,则退出
        if x_shift + 1.5 * char_w > imgsz[1]:
            break

    return text[:i]


def gen_ocr_img(imgsz=(64, 128), font=None, alpha="0123456789.AbC", target_len=1):
    from PIL import ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageFile

    # # font_size_list = [35, 32, 30, 28, 25]
    # font_size_list = [48]
    # font_path_list = list(FONT_CHARS_DICT.keys())
    # font_list = []  # 二位列表[size,font]
    # for size in font_size_list:
    #     font_list.append([ImageFont.truetype(font_path, size=size) for font_path in font_path_list])

    text = np.random.choice(list(alpha), target_len)
    text = ''.join(text)
    # size_idx = np.random.randint(len(font_size_list))
    # font_idx = np.random.randint(len(font_path_list))
    # font = font_list[size_idx][font_idx]
    # font_path = font_path_list[font_idx]

    w, char_h = font.getsize(text)
    char_w = int(w / len(text))

    imgsz = (56, char_w + 8)

    image = gen_background(imgsz)
    image = image.astype(np.uint8)

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)
    # color = tuple(random_color(105, 255))
    # color = (0, 0, 0)
    color = (255, 255, 255)

    text = horizontal_draw(draw, text, font, color, imgsz, char_w, char_h, easyFlag=True)
    target_len = len(text)  # target_len可能变小了
    indices = np.array([alpha.index(c) for c in text])
    image = np.array(im)

    # rmdnum = random.random()
    # if rmdnum > 0.75:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # else:
    #     image = 255 - image

    image = 255 - image

    return image


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


def convert_baidu_chinese_ocr_dataset_to_custom_dataset_format(data_path):
    # labels = "０１２３４５６７８９"
    train_images_path = data_path + "/train_images"
    train_list_path = data_path + "/train.list"
    img_list = sorted(os.listdir(train_images_path))

    save_Chinese_path = make_save_path(train_images_path, "is_all_chinese")
    save_digits_path = make_save_path(train_images_path, "is_all_digits")

    with open(train_list_path, "r", encoding="utf-8") as fo:
        lines = fo.readlines()
        for line in tqdm(lines):
            try:
                line = line.strip()
                img_name = line.split("\t")[2]
                label = line.split("\t")[3]

                res1 = is_all_chinese(label)
                res2 = is_all_digits(label)

                img_abs_path = train_images_path + "/{}".format(img_name)
                img_base_name, suffix = os.path.splitext(img_name)[0], os.path.splitext(img_name)[1]
                img_new_name = "{}={}{}".format(img_base_name, label, suffix)
                img_dst_Chines_path = save_Chinese_path + "/{}".format(img_new_name)
                img_dst_digits_path = save_digits_path + "/{}".format(img_new_name)
                if res1:
                    os.rename(img_abs_path, img_dst_Chines_path)
                if res2:
                    os.rename(img_abs_path, img_dst_digits_path)

            except Exception as Error:
                print(Error)


# KPT
def crop_img_via_labelbee_kpt_json(data_path):
    img_path = data_path + "/images"
    json_path = data_path + "/jsons"
    save_path = data_path + "/output_warp_test_resize"
    os.makedirs(save_path, exist_ok=True)

    json_list = sorted(os.listdir(json_path))

    for j in tqdm(json_list):
        try:
            fname = os.path.splitext(j.replace(".json", ""))[0]
            json_abs_path = json_path + "/{}".format(j)
            json_ = json.load(open(json_abs_path, 'r', encoding='utf-8'))
            if not json_: continue
            w, h = json_["width"], json_["height"]

            result_ = json_["step_1"]["result"]
            if not result_: continue

            # if copy_image:
            #     img_abs_path = img_path + "/{}".format(j.replace(".json", ""))
            #     # shutil.move(img_path, det_images_path + "/{}".format(j.replace(".json", "")))
            #     shutil.copy(img_abs_path, kpt_images_path + "/{}".format(j.replace(".json", "")))

            len_result = len(result_)

            # txt_save_path = kpt_labels_path + "/{}.gt".format(j.replace(".json", "").split(".")[0])
            # with open(txt_save_path, "w", encoding="utf-8") as fw:

            img_abs_path = img_path + "/{}.jpg".format(fname)
            img = cv2.imread(img_abs_path)

            kpts = []
            for i in range(len_result):
                x_ = int(round(result_[i]["x"]))
                y_ = int(round(result_[i]["y"]))
                attribute_ = result_[i]["attribute"]
                # x_normalized = x_ / w
                # y_normalized = y_ / h

                # visible = True
                # if visible:
                #     kpts.append([x_normalized, y_normalized, 2])
                kpts.append([x_, y_])

            x1, x2 = round(min(kpts[0][0], kpts[3][0])), round(max(kpts[1][0], kpts[2][0]))
            y1, y2 = round(min(kpts[0][1], kpts[1][1])), round(max(kpts[2][1], kpts[3][1]))
            cropped_base = img[y1:y2, x1:x2]
            basesz = cropped_base.shape[:2]

            kpts = expand_kpt(basesz, kpts, r=0.12)

            kpts = np.asarray(kpts).reshape(-1, 8)
            for ki in range(kpts.shape[0]):
                # txt_content = ", ".join([str(k) for k in kpts[ki]]) + ", 0\n"
                # fw.write(txt_content)

                if h > w:
                    src_points = np.float32([[kpts[ki][0], kpts[ki][1]], [kpts[ki][2], kpts[ki][3]], [kpts[ki][6], kpts[ki][7]], [kpts[ki][4], kpts[ki][5]]])
                    dst_points = np.float32([[0, 0], [h // 2, 0], [0, w // 2], [h // 2, w // 2]])
                    M = cv2.getPerspectiveTransform(src_points, dst_points)
                    warpped = cv2.warpPerspective(img, M, (h // 2, w // 2))
                    cv2.imwrite("{}/{}_{}.jpg".format(save_path, fname, ki), warpped)
                else:
                    src_points = np.float32([[kpts[ki][0], kpts[ki][1]], [kpts[ki][2], kpts[ki][3]], [kpts[ki][6], kpts[ki][7]], [kpts[ki][4], kpts[ki][5]]])
                    dst_points = np.float32([[0, 0], [w // 2, 0], [0, h // 2], [w // 2, h // 2]])
                    M = cv2.getPerspectiveTransform(src_points, dst_points)
                    warpped = cv2.warpPerspective(img, M, (w // 2, h // 2))
                    cv2.imwrite("{}/{}_{}.jpg".format(save_path, fname, ki), warpped)

        except Exception as Error:
            print(Error)


# def labelbee_kpt_to_labelme_kpt(data_path):
#     import labelme

#     save_path = make_save_path(data_path, "labelme_format")
#     img_save_path = save_path + "/images"
#     json_save_path = save_path + "/jsons"
#     os.makedirs(img_save_path, exist_ok=True)
#     os.makedirs(json_save_path, exist_ok=True)

#     images_path = data_path + "/images"
#     jsons_path = data_path + "/jsons"
#     file_list = get_file_list(jsons_path)
#     for f in tqdm(file_list):
#         try:
#             img_name = os.path.splitext(f)[0]
#             fname = os.path.splitext(img_name)[0]
#             f_abs_path = jsons_path + "/{}".format(f)
#             img_abs_path = images_path + "/{}.jpg".format(fname)
#             img = cv2.imread(img_abs_path)
#             imgsz = img.shape[:2]

#             with open(f_abs_path, "r") as fr:
#                 src_data = json.load(fr)
#             assert len(src_data["step_1"]["result"]) == 4, "N points should == 4!"

#             p1 = (src_data["step_1"]["result"][0]["x"], src_data["step_1"]["result"][0]["y"])
#             p2 = (src_data["step_1"]["result"][1]["x"], src_data["step_1"]["result"][1]["y"])
#             p3 = (src_data["step_1"]["result"][2]["x"], src_data["step_1"]["result"][2]["y"])
#             p4 = (src_data["step_1"]["result"][3]["x"], src_data["step_1"]["result"][3]["y"])

#             shapes_data = []
#             shapes_data.append({"label": "ul", "points": [[p1[0], p1[1]]], "group_id": None, "shape_type": "point", "flags": {}})
#             shapes_data.append({"label": "ur", "points": [[p2[0], p2[1]]], "group_id": None, "shape_type": "point", "flags": {}})
#             shapes_data.append({"label": "br", "points": [[p3[0], p3[1]]], "group_id": None, "shape_type": "point", "flags": {}})
#             shapes_data.append({"label": "bl", "points": [[p4[0], p4[1]]], "group_id": None, "shape_type": "point", "flags": {}})

#             json_labelme = {}
#             json_labelme["version"] = "4.5.9"
#             json_labelme["flags"] = eval("{}")
#             json_labelme["shapes"] = shapes_data
#             json_labelme["imagePath"] = img_name
#             json_labelme["imageData"] = labelme.utils.img_arr_to_b64(img).strip()
#             json_labelme["imageHeight"] = imgsz[0]
#             json_labelme["imageWidth"] = imgsz[1]

#             json_dst_path = json_save_path + "/{}.json".format(fname)
#             with open(json_dst_path, 'w') as fw:
#                 json.dump(json_labelme, fw, indent=2)

#             img_src_path = images_path + "/{}.jpg".format(fname)
#             img_dst_path = img_save_path + "/{}.jpg".format(fname)
#             shutil.copy(img_src_path, img_dst_path)

#         except Exception as Error:
#             print(Error)


def aug_points(pts, n=10, imgsz=None, r=0.05):
    minSide = min(imgsz[0], imgsz[1])
    rdmp = round(minSide * r)
    ptsnew = []

    assert len(pts) == 4, "len(pts) should == 4!"

    for ni in range(n):
        ptsnewi = []
        for i in range(4):
            pi = (pts[i][0] + np.random.randint(-rdmp, rdmp), pts[i][1] + np.random.randint(-rdmp, rdmp))
            ptsnewi.append(pi)
        ptsnew.append(ptsnewi)

    return ptsnew


# def labelbee_kpt_to_labelme_kpt_multi_points(data_path):
#     import labelme

#     save_path = make_save_path(data_path, "labelme_format")
#     img_save_path = save_path + "/images"
#     json_save_path = save_path + "/jsons"
#     os.makedirs(img_save_path, exist_ok=True)
#     os.makedirs(json_save_path, exist_ok=True)

#     images_path = data_path + "/images"
#     jsons_path = data_path + "/jsons"
#     file_list = get_file_list(jsons_path)
#     for f in tqdm(file_list):
#         try:
#             img_name = os.path.splitext(f)[0]
#             fname = os.path.splitext(img_name)[0]
#             f_abs_path = jsons_path + "/{}".format(f)
#             img_abs_path = images_path + "/{}.jpeg".format(fname)
#             img = cv2.imread(img_abs_path)
#             imgsz = img.shape[:2]

#             with open(f_abs_path, "r") as fr:
#                 src_data = json.load(fr)
#             assert len(src_data["step_1"]["result"]) != 0 and len(src_data["step_1"]["result"]) % 4 == 0, "N points should % 4 == 0 and != 0!"

#             pts = []
#             ni = 0
#             for i in range(0, len(src_data["step_1"]["result"]), 4):
#                 if src_data["step_1"]["result"][i + 0]["attribute"] == "1":
#                     p1 = [src_data["step_1"]["result"][i + 0]["x"], src_data["step_1"]["result"][i + 0]["y"]]
#                 if src_data["step_1"]["result"][i + 1]["attribute"] == "2":
#                     p2 = [src_data["step_1"]["result"][i + 1]["x"], src_data["step_1"]["result"][i + 1]["y"]]
#                 if src_data["step_1"]["result"][i + 2]["attribute"] == "3":
#                     p3 = [src_data["step_1"]["result"][i + 2]["x"], src_data["step_1"]["result"][i + 2]["y"]]
#                 if src_data["step_1"]["result"][i + 3]["attribute"] == "4":
#                     p4 = [src_data["step_1"]["result"][i + 3]["x"], src_data["step_1"]["result"][i + 3]["y"]]

#                 pts.append([p1, p2, p3, p4])

#                 pt = [p1, p2, p3, p4]
#                 pt_copy = copy.deepcopy(pt)
#                 augNum = 3
#                 x1, x2 = round(min(p1[0], p4[0])), round(max(p2[0], p3[0]))
#                 y1, y2 = round(min(p1[1], p2[1])), round(max(p3[1], p4[1]))
#                 cropped_base = img[y1:y2, x1:x2]
#                 basesz = cropped_base.shape[:2]
#                 # ptsnew = aug_points(pt, n=10, imgsz=basesz, r=0.25)
#                 # # ptsnew = list(set(ptsnew))

#                 ni += 1

#                 ptsnew = []
#                 for i in range(augNum):
#                     r_ = 0.01 * np.random.randint(10, 16)
#                     pt_ = expand_kpt(basesz, pt, r=r_)
#                     pt_cp = copy.deepcopy(pt_)
#                     ptsnew.append(pt_cp)

#                 # pt_ = expand_kpt(basesz, pt, r=0.10)

#                 for idx, pi in enumerate(ptsnew):
#                     # for idx, pi in enumerate([pt]):
#                     ix1, ix2 = round(min(pi[0][0], pi[3][0])), round(max(pi[1][0], pi[2][0]))
#                     iy1, iy2 = round(min(pi[0][1], pi[1][1])), round(max(pi[2][1], pi[3][1]))
#                     cropped = img[iy1:iy2, ix1:ix2]
#                     croppedsz = cropped.shape[:2]

#                     shapes_data = []
#                     shapes_data.append({"label": "ul", "points": [[pt_copy[0][0] - ix1, pt_copy[0][1] - iy1]], "group_id": None, "shape_type": "point", "flags": {}})
#                     shapes_data.append({"label": "ur", "points": [[pt_copy[1][0] - ix1, pt_copy[1][1] - iy1]], "group_id": None, "shape_type": "point", "flags": {}})
#                     shapes_data.append({"label": "br", "points": [[pt_copy[2][0] - ix1, pt_copy[2][1] - iy1]], "group_id": None, "shape_type": "point", "flags": {}})
#                     shapes_data.append({"label": "bl", "points": [[pt_copy[3][0] - ix1, pt_copy[3][1] - iy1]], "group_id": None, "shape_type": "point", "flags": {}})

#                     json_labelme = {}
#                     json_labelme["version"] = "4.5.9"
#                     json_labelme["flags"] = eval("{}")
#                     json_labelme["shapes"] = shapes_data
#                     json_labelme["imagePath"] = fname + "_{}_{}.jpg".format(ni, idx)
#                     json_labelme["imageData"] = labelme.utils.img_arr_to_b64(cropped).strip()
#                     json_labelme["imageHeight"] = croppedsz[0]
#                     json_labelme["imageWidth"] = croppedsz[1]

#                     json_dst_path = json_save_path + "/{}_{}_{}.json".format(fname, ni, idx)
#                     with open(json_dst_path, 'w') as fw:
#                         json.dump(json_labelme, fw, indent=2)

#                     img_dst_path = img_save_path + "/{}_{}_{}.jpg".format(fname, ni, idx)
#                     cv2.imwrite(img_dst_path, cropped)

#         except Exception as Error:
#             print(Error)


def crop_img_via_labelme_json(data_path, r=0.10):
    save_path = data_path + "/images_perspective_transform"
    os.makedirs(save_path, exist_ok=True)

    img_path = data_path + "/images"
    json_path = data_path + "/jsons"

    file_list = get_file_list(img_path)
    for f in tqdm(file_list):
        fname = os.path.splitext(f)[0]
        img_abs_path = img_path + "/{}".format(f)
        json_abs_path = json_path + "/{}.json".format(fname)

        img = cv2.imread(img_abs_path)
        imgsz = img.shape[:2]

        with open(json_abs_path, "r") as fr:
            json_ = json.load(fr)

        p0 = json_["shapes"][0]["points"][0]
        p1 = json_["shapes"][1]["points"][0]
        p2 = json_["shapes"][2]["points"][0]
        p3 = json_["shapes"][3]["points"][0]
        pts = [p0, p1, p2, p3]

        pts = expand_kpt(imgsz, pts, r=r)

        dis_x01 = np.sqrt((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2)
        dis_x23 = np.sqrt((pts[3][0] - pts[2][0]) ** 2 + (pts[3][1] - pts[2][1]) ** 2)
        dis_y03 = np.sqrt((pts[3][0] - pts[0][0]) ** 2 + (pts[3][1] - pts[0][1]) ** 2)
        dis_y12 = np.sqrt((pts[2][0] - pts[1][0]) ** 2 + (pts[2][1] - pts[1][1]) ** 2)
        dstW = round(max(dis_x01, dis_x23))
        dstH = round(max(dis_y03, dis_y12))

        srcPoints = np.array([pts[0], pts[1], pts[3], pts[2]], dtype=np.float32)
        dstPoints = np.array([[0, 0], [dstW, 0], [0, dstH], [dstW, dstH]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        warped = cv2.warpPerspective(img, M, (dstW, dstH))
        cv2.imwrite("{}/{}".format(save_path, f), warped)


# CLS
class GKFCLS():
    def __init__(self, model_path, n_classes=2, input_size=(128, 128), keep_ratio_flag=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), device="cuda:0", print_infer_time=False):
        self.transforms_test = transforms.Compose([transforms.Resize((input_size[1], input_size[0])),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=mean, std=std),
                                                   ])
        self.model_path = model_path
        self.n_classes = n_classes
        self.input_size = input_size
        self.keep_ratio_flag = keep_ratio_flag
        self.device = device
        self.print_infer_time = print_infer_time
        self.ort_session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', "CPUExecutionProvider"])

    def keep_ratio(self, pilimg, flag=True, shape=(128, 128)):
        if flag:
            img = np.array(np.uint8(pilimg))
            img_src, ratio, (dw, dh) = letterbox(img, new_shape=shape)
            keep_ratio_pilimg = Image.fromarray(img_src)
            return keep_ratio_pilimg
        else:
            return pilimg

    def preprocess(self, img_path):
        pilimg = Image.open(img_path).convert("RGB")
        pilimg = self.keep_ratio(pilimg, flag=self.keep_ratio_flag, shape=self.input_size)
        pilimg = self.transforms_test(pilimg).unsqueeze(0)
        pilimg = pilimg.to(self.device)

        return pilimg

    def inference(self, pilimg):
        t1 = time.time()
        ort_outs = self.ort_session.run(["output"], {self.ort_session.get_inputs()[0].name: to_numpy(pilimg)})
        ort_out = ort_outs[0]
        t2 = time.time()
        if self.print_infer_time:
            print("inference time: {}".format(t2 - t1))
        return ort_out

    def postprocess(self, ort_out):
        cls = np.argmax(ort_out)
        return cls

    def cal_acc_n_cls(self, test_path="", output_path=None, save_pred_true=False, save_pred_false=True, save_dir_name="", mv_or_cp="copy"):
        """
        :param test_path:
        :param output_path:
        :param save_pred_false_img:
        :param save_dir_name:
        :param mv_or_cp:
        :return:
        """
        dir_name = get_dir_name(test_path)
        save_path = make_save_path(test_path, dir_name_add_str="pred_res")
        save_path_true = save_path + "/true"
        save_path_false = save_path + "/false"
        os.makedirs(save_path_true, exist_ok=True)
        os.makedirs(save_path_false, exist_ok=True)

        res_list = []
        img_list = sorted(os.listdir(test_path))
        for img in tqdm(img_list):
            img_abs_path = test_path + "/{}".format(img)
            img_name = os.path.splitext(img)[0]
            pilimg = self.preprocess(img_abs_path)
            ort_out = self.inference(pilimg)
            cls = self.postprocess(ort_out)
            res_list.append(cls)

            for ci in range(self.n_classes):
                if cls == int(dir_name):
                    if save_pred_true:
                        img_dst_path = save_path_true + "/{}={}.jpg".format(img_name, cls)
                        if mv_or_cp == "copy" or mv_or_cp == "cp":
                            shutil.copy(img_abs_path, img_dst_path)
                        else:
                            shutil.move(img_abs_path, img_dst_path)
                    else:
                        pass
                else:
                    if save_pred_false:
                        img_dst_path = save_path_false + "/{}={}.jpg".format(img_name, cls)
                        if mv_or_cp == "copy" or mv_or_cp == "cp":
                            shutil.copy(img_abs_path, img_dst_path)
                        else:
                            shutil.move(img_abs_path, img_dst_path)
                    else:
                        pass

        acc_i = {}
        for i in range(self.n_classes):
            acc_i["{}".format(i)] = res_list.count(i) / len(res_list)

        print(acc_i)

        return acc_i

    def cal_acc_2_cls(self, test_path="", output_path=None, save_FP_FN_img=True, save_dir_name="", mv_or_cp="copy", NP="P", metrics=True):
        """
        :param test_path: Should just be one class
        :param output_path: If None, will create output dir in current path, others will create in the output_path
        :param save_img: Save FP images(Type I error), FN images(Type II error)
        :param NP: Current dir images is Positive or Negative
        :param metrics: Cal Precisioin Recall F1 Score AUC-ROC
        :return:
        """
        dir_name = os.path.basename(test_path)
        if save_FP_FN_img:
            if output_path is None:
                output_path = os.path.abspath(os.path.join(test_path, "../..")) + "/{}_output_{}".format(dir_name, save_dir_name)
                FP_Path = output_path + "/FP"
                FN_Path = output_path + "/FN"
                os.makedirs(FP_Path, exist_ok=True)
                os.makedirs(FN_Path, exist_ok=True)
            else:
                FP_Path = output_path + "/FP"
                FN_Path = output_path + "/FN"
                os.makedirs(FP_Path, exist_ok=True)
                os.makedirs(FN_Path, exist_ok=True)

        res_list = []
        TP, FP, FN, TN = 0, 0, 0, 0
        img_list = sorted(os.listdir(test_path))
        for img in tqdm(img_list):
            img_abs_path = test_path + "/{}".format(img)
            img_name = os.path.splitext(img)[0]
            pilimg = self.preprocess(img_abs_path)
            ort_out = self.inference(pilimg)
            cls = self.postprocess(ort_out)
            res_list.append(cls)

            if NP == "P":
                if cls == 0:
                    FN += 1
                    if save_FP_FN_img:
                        img_dst_path = FN_Path + "/{}".format(img)
                        if mv_or_cp == "copy":
                            shutil.copy(img_abs_path, img_dst_path)
                        elif mv_or_cp == "move":
                            shutil.move(img_abs_path, img_dst_path)
                        else:
                            print("mv_or_cp should be: move, copy.")
                        print("Predicted cls: {} True label: {} img_path: {}".format(cls, NP, img_abs_path))
                elif cls == 1:
                    TP += 1
                else:
                    print("Just 2 classes!")
            elif NP == "N":
                if cls == 0:
                    TN += 1
                elif cls == 1:
                    FP += 1
                    if save_FP_FN_img:
                        img_dst_path = FP_Path + "/{}".format(img)
                        if mv_or_cp == "copy":
                            shutil.copy(img_abs_path, img_dst_path)
                        elif mv_or_cp == "move":
                            shutil.move(img_abs_path, img_dst_path)
                        else:
                            print("mv_or_cp should be: move, copy.")
                        print("Predicted cls: {} True label: {} img_path: {}".format(cls, NP, img_abs_path))
                else:
                    print("Just 2 classes!")
            else:
                print("NP should be 'N' or 'P'!")

        acc_i = {}
        for i in range(self.n_classes):
            acc_i["{}".format(i)] = res_list.count(i) / len(res_list)

        print(acc_i)

        if metrics:
            Accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-12)
            Precision = TP / (TP + FP + 1e-12)
            Recall = TP / (TP + FN + 1e-12)
            Specificity = TN / (TN + FP + 1e-12)
            F1 = 2 * (Precision * Recall) / (Precision + Recall + + 1e-12)
            print("TP, FP, FN, TN: {}, {}, {}, {}".format(TP, FP, FN, TN))
            print("Accuracy: {:.12f} Precision: {:.12f} Recall: {:.12f} F1: {:.12f} Specificity: {:.12f}".format(Accuracy, Precision, Recall, F1, Specificity))

        return acc_i
    

def gen_cls_negatives_via_random_crop(data_path, random_size=(96, 100, 128, 160), randint_low=10, randint_high=51, hw_dis=100, dst_num=20000):
    img_list = sorted(os.listdir(data_path))

    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_random_cropped".format(data_path.split("/")[-1])
    os.makedirs(save_path, exist_ok=True)

    total_num = 0

    for img in tqdm(img_list):

        if total_num >= dst_num:
            break

        img_name = os.path.splitext(img)[0]
        img_abs_path = data_path + "/{}".format(img)
        try:
            img = cv2.imread(img_abs_path)
            h, w = img.shape[:2]
            n = np.random.randint(randint_low, randint_high)
            for i in range(n):
                try:

                    if total_num == dst_num:
                        break

                    size_i_h = random.sample(random_size, 1)
                    size_i_w = random.sample(random_size, 1)

                    while abs(size_i_h[0] - size_i_w[0]) > hw_dis:
                        size_i_w = random.sample(random_size, 1)

                    size_i = (size_i_h, size_i_w)

                    random_pos = [np.random.randint(0, w - size_i[1][0]), np.random.randint(0, h - size_i[0][0])]
                    random_cropped = img[random_pos[1]:(random_pos[1] + size_i[0][0]), random_pos[0]:(random_pos[0] + size_i[1][0])]
                    cv2.imwrite("{}/{}_{}_{}_{}.jpg".format(save_path, img_name, size_i[0][0], size_i[1][0], i), random_cropped)

                    total_num += 1

                except Exception as Error:
                    print(Error, Error.__traceback__.tb_lineno)
        except Exception as Error:
            print(Error, Error.__traceback__.tb_lineno)


def random_erasing(data_path):
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_random_erasing_aug".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomErasing()])

    img_list = sorted(os.listdir(data_path))
    for img in img_list:
        img_abs_path = data_path + "/{}".format(img)
        img_name = os.path.splitext(img)[0]
        pilimg = Image.open(img_abs_path)
        random_erased = transform(pilimg)
        random_erased_pil = transforms.ToPILImage()(random_erased)
        random_erased_pil.save("{}/{}".format(save_path, img))


def random_paste_four_corner(positive_img_path, negative_img_path):
    dir_name = os.path.basename(positive_img_path)
    save_path = os.path.abspath(os.path.join(positive_img_path, "../..")) + "/{}_random_paste_four_corner_aug".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    pimg_list = sorted(os.listdir(positive_img_path))
    nimg_list = sorted(os.listdir(negative_img_path))
    for pimg in pimg_list[:20000]:
        try:
            pimg_abs_path = positive_img_path + "/{}".format(pimg)
            pimg_name = os.path.splitext(pimg)[0]
            ppilimg = Image.open(pimg_abs_path)
            (pw, ph) = ppilimg.size

            nimg_path = random.sample(nimg_list, 1)[0]
            nimg_abs_path = negative_img_path + "/{}".format(nimg_path)
            nimg_name = os.path.splitext(nimg_path)[0]
            npilimg = Image.open(nimg_abs_path)
            (nw, nh) = npilimg.size

            # narrayimg = np.array(npilimg, dtype=np.uint8)
            paste_n = np.random.randint(1, 5)
            pwh_min = min(pw, ph)
            # crop_size = [int(round(pwh_min * 0.10)), int(round(pwh_min * 0.25)), int(round(pwh_min * 0.30)), int(round(pwh_min * 0.35)), int(round(pwh_min * 0.45)), int(round(pwh_min * 0.55))]
            crop_size = [int(round(pwh_min * 0.45)), int(round(pwh_min * 0.50)), int(round(pwh_min * 0.60)), int(round(pwh_min * 0.65)), int(round(pwh_min * 0.70)), int(round(pwh_min * 0.75))]
            # cropped_pimgs = []
            crop_coor1s = [(np.random.randint(0, int(pw * 0.25)), np.random.randint(0, int(ph * 0.25))), (np.random.randint(int(pw * 0.75), pw + 1), np.random.randint(0, int(ph * 0.25))),
                           (np.random.randint(0, int(pw * 0.25)), np.random.randint(int(ph * 0.75), ph + 1)), (np.random.randint(int(pw * 0.75), pw + 1), np.random.randint(int(ph * 0.75), ph + 1))]
            crop_coor1 = random.sample(crop_coor1s, paste_n)
            for i in range(paste_n):
                crop_coor2_wh = random.sample(crop_size, 2)
                crop_box = (crop_coor1[i][0], crop_coor1[i][1], crop_coor1[i][0] + crop_coor2_wh[0], crop_coor1[i][1] + crop_coor2_wh[1])
                cropped = npilimg.crop(crop_box)
                # cropped_pimgs.append(cropped)

                ppilimg.paste(cropped, crop_box)

            ppilimg.save("{}/{}_{}.jpg".format(save_path, pimg_name, nimg_name))

        except Exception as Error:
            print(Error, )


def crop_red_bbx_area(data_path, expand_p=5):
    file_list = get_file_list(data_path)
    save_path = make_save_path(data_path, dir_name_add_str="crop_red_bbx")
    # expand_p = 10

    for f in tqdm(file_list):
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        results = find_red_bbx(img, expand_p=expand_p)
        for ri, r in enumerate(results):
            try:
                f_ri_dst_path = save_path + "/{}_{}_{}_cropped.jpg".format(fname, expand_p, ri)
                cropped = img[r[2]:r[3], r[0]:r[1]]
                cv2.imwrite(f_ri_dst_path, cropped)
            except Exception as Error:
                print(Error)


def is_gray_img(img, dstsz=(64, 64), mean_thr=1):
    img = cv2.resize(img, dstsz)
    imgsz = img.shape[:2]
    psum = []
    for hi in range(imgsz[0]):
        for wi in range(imgsz[1]):
            pgap = (abs(img[hi, wi, 0] - img[hi, wi, 1]) + abs(img[hi, wi, 1] - img[hi, wi, 2]) + abs(img[hi, wi, 0] - img[hi, wi, 2])) / 3
            psum.append(pgap)

    pmean = np.mean(psum)

    if pmean < mean_thr:
        return True
    
    return False


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def cal_iou(bbx1, bbx2):
    """
    b1 = [0, 0, 10, 10]
    b2 = [2, 2, 12, 12]
    iou = cal_iou(b1, b2)  # 0.47058823529411764

    p --> bbx1
    q --> bbx2
    :param bbx1:
    :param bbx2:
    :return:
    """

    px1, py1, px2, py2 = bbx1[0], bbx1[1], bbx1[2], bbx1[3]
    qx1, qy1, qx2, qy2 = bbx2[0], bbx2[1], bbx2[2], bbx2[3]
    area1 = abs(px2 - px1) * abs(py2 - py1)
    area2 = abs(qx2 - qx1) * abs(qy2 - qy1)

    # cross point --> c
    cx1 = max(px1, qx1)
    cy1 = max(py1, qy1)
    cx2 = min(px2, qx2)
    cy2 = min(py2, qy2)

    cw = cx2 - cx1
    ch = cy2 - cy1
    if cw <= 0 or ch <= 0:
        return 0

    carea = cw * ch
    iou = carea / (area1 + area2 - carea)
    return iou


def seamless_clone(bg_path, obj_path):
    img1 = cv2.imread(bg_path)
    img2 = cv2.imread(obj_path)
    img2 = cv2.resize(img2, (1920, 1080))

    # src_mask = np.zeros(img2.shape, img2.dtype)
    h, w = img1.shape[:2]
    mask = 255 * np.ones(img2.shape, img2.dtype)
    center = (w // 2, h // 2)
    output_normal = cv2.seamlessClone(img2, img1, mask, center, cv2.NORMAL_CLONE)
    output_mixed = cv2.seamlessClone(img2, img1, mask, center, cv2.MIXED_CLONE)
    output_MONOCHROME = cv2.seamlessClone(img2, img1, mask, center, cv2.MONOCHROME_TRANSFER)

    # cv2.imshow("output_normal", output_normal)
    # cv2.imshow("output_mixed", output_mixed)
    # cv2.waitKey(0)
    cv2.imwrite("/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/output_normal.png", output_normal)
    cv2.imwrite("/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/output_mixed.png", output_mixed)
    cv2.imwrite("/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/output_MONOCHROME.png", output_MONOCHROME)


def draw_label(size=(384, 384, 3), polygon_list=None):
    image = np.zeros(size, np.uint8)
    img_vis = cv2.fillPoly(image, polygon_list, (128, 128, 128))
    img_vis = Image.fromarray(img_vis)
    img = cv2.fillPoly(image, polygon_list, (1, 1, 1))
    img = Image.fromarray(img)

    return img_vis, img


def crop_img_via_expand(img, bbx, size, n=1.0):
    """
    left & right expand pixels should be (n - 1) / 2
    :param img:
    :param bbx: [x1, y1, x2, y2]
    :param size: image size --> [H, W]
    :param n: 1, 1.5, 2, 2.5, 3
    :return:
    """

    x1, y1, x2, y2 = bbx
    bbx_h, bbx_w = y2 - y1, x2 - x1
    expand_x = int(round((n - 1) / 2 * bbx_w))
    expand_y = int(round((n - 1) / 2 * bbx_h))
    expand_x_half = int(round(expand_x / 2))
    expand_y_half = int(round(expand_y / 2))
    # center_p = [int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))]
    if n == 1:
        cropped = img[y1:y2, x1:x2]
        return cropped
    else:
        if x1 - expand_x >= 0:
            x1_new = x1 - expand_x
        elif x1 - expand_x_half >= 0:
            x1_new = x1 - expand_x_half
        else:
            x1_new = x1

        if y1 - expand_y >= 0:
            y1_new = y1 - expand_y
        elif y1 - expand_y_half >= 0:
            y1_new = y1 - expand_y_half
        else:
            y1_new = y1

        if x2 + expand_x <= size[1]:
            x2_new = x2 + expand_x
        elif x2 + expand_x_half <= size[1]:
            x2_new = x2 + expand_x_half
        else:
            x2_new = x2

        if y2 + expand_y <= size[0]:
            y2_new = y2 + expand_y
        elif y2 + expand_y_half <= size[0]:
            y2_new = y2 + expand_y_half
        else:
            y2_new = y2

        cropped = img[y1_new:y2_new, x1_new:x2_new]
        return cropped


def crop_image_via_labelbee_labels(data_path, crop_ratio=(1, 1.5, 2, 2.5, 3)):
    dir_name = os.path.basename(data_path)
    img_path = data_path + "/images"
    json_path = data_path + "/jsons"

    cropped_path = data_path + "/{}_cropped".format(dir_name)
    det_images_path = data_path + "/{}_selected_images".format(dir_name)
    det_labels_path = data_path + "/{}_labels".format(dir_name)

    os.makedirs(cropped_path, exist_ok=True)
    os.makedirs(det_images_path, exist_ok=True)
    os.makedirs(det_labels_path, exist_ok=True)

    json_list = os.listdir(json_path)

    for j in json_list:
        img_name = os.path.splitext(j.replace(".json", ""))[0]
        json_abs_path = json_path + "/{}".format(j)
        img_abs_path = img_path + "/{}".format(j.replace(".json", ""))
        img = cv2.imread(img_abs_path)
        json_ = json.load(open(json_abs_path, 'r', encoding='utf-8'))
        if not json_: continue
        w, h = json_["width"], json_["height"]

        result_ = json_["step_1"]["result"]
        if not result_: continue

        try:
            img_abs_path = img_path + "/{}".format(j.replace(".json", ""))
            # shutil.move(img_path, det_images_path + "/{}".format(j.strip(".json")))
            shutil.copy(img_abs_path, det_images_path + "/{}".format(j.replace(".json", "")))
        except Exception as Error:
            print(Error)

        len_result = len(result_)

        txt_save_path = det_labels_path + "/{}.txt".format(j.replace(".json", "").split(".")[0])
        with open(txt_save_path, "w", encoding="utf-8") as fw:
            for i in range(len_result):
                x_ = result_[i]["x"]
                y_ = result_[i]["y"]
                w_ = result_[i]["width"]
                h_ = result_[i]["height"]

                x_min = int(round(x_))
                x_max = int(round(x_ + w_))
                y_min = int(round(y_))
                y_max = int(round(y_ + h_))

                for nx in crop_ratio:
                    try:
                        cropped_img = crop_img_via_expand(img, [x_min, y_min, x_max, y_max], [h, w], nx)
                        cropped_nx_path = cropped_path + "/{}".format(nx)
                        os.makedirs(cropped_nx_path, exist_ok=True)
                        cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, i, nx), cropped_img)
                    except Exception as Error:
                        print(Error)
                        # cropped_img = crop_img_expand_n_times_v2(img, [x_min, y_min, x_max, y_max], [h, w], 1)
                        # cropped_nx_path = cropped_path + "/{}".format(nx)
                        # os.makedirs(cropped_nx_path, exist_ok=True)
                        # cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, i, nx), cropped_img)

                # bb = convert_bbx_VOC_to_yolo((h, w), (x_min, x_max, y_min, y_max))
                bb = bbox_voc_to_yolo((h, w), (x_min, y_min, x_max, y_max))
                txt_content = "0" + " " + " ".join([str(b) for b in bb]) + "\n"
                fw.write(txt_content)


def crop_image_via_yolo_labels(data_path, CLS=(1, 2), crop_ratio=(1, 1.5, 2, 2.5, 3)):
    dir_name = os.path.basename(data_path)
    img_path = data_path + "/images"
    txt_path = data_path + "/labels"

    cropped_path = data_path + "/{}_cropped".format(dir_name)
    os.makedirs(cropped_path, exist_ok=True)

    # txt_list = os.listdir(txt_path)
    img_list = os.listdir(img_path)

    for j in tqdm(img_list):
        try:
            img_name = os.path.splitext(j)[0]
            txt_abs_path = txt_path + "/{}.txt".format(img_name)
            img_abs_path = img_path + "/{}".format(j)
            img = cv2.imread(img_abs_path)
            if img is None: continue
            h, w = img.shape[:2]

            txt_o = open(txt_abs_path, "r", encoding="utf-8")
            lines = txt_o.readlines()
            txt_o.close()

            for i, l in enumerate(lines):
                l_s = l.strip().split(" ")
                cls = int(l_s[0])
                if cls in CLS:
                    bbx_yolo = list(map(float, l_s[1:]))
                    # bbx_voc = convert_bbx_yolo_to_VOC([h, w], bbx_yolo)
                    bbx_voc = bbox_yolo_to_voc([h, w], bbx_yolo)

                    # crop_ratio_rdm = np.random.randint(20, 31)
                    # crop_ratio_ = [crop_ratio_rdm * 0.1]
                    for nx in crop_ratio:
                        try:
                            cropped_img = crop_img_via_expand(img, bbx_voc, [h, w], nx)
                            cropped_nx_path = cropped_path + "/{}/{}".format(cls, nx)
                            os.makedirs(cropped_nx_path, exist_ok=True)
                            cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, i, nx), cropped_img)
                        except Exception as Error:
                            print(Error)
        except Exception as Error:
            print(Error)


def select_images_via_gosuncn_cpp_output(txt_path, save_path_flag="current", save_path="", save_no_det_res_img=False, save_crop_img=True, save_src_img=False, save_vis_img=False, crop_expand_ratio=1.5, n_cls=4):
    if save_path_flag == "current":
        save_base_path = os.path.abspath(os.path.join(txt_path, "../.."))
    else:
        save_base_path = save_path

    dataset_name = os.path.basename(txt_path).split("_list_res")[0]

    if save_src_img:
        for i in range(n_cls):
            save_path_cls_i_src_0 = save_base_path + "/C_Plus_Plus_det_output/{}/src_images/cls_{}/0".format(dataset_name, i)
            save_path_cls_i_src_1 = save_base_path + "/C_Plus_Plus_det_output/{}/src_images/cls_{}/1".format(dataset_name, i)
            os.makedirs(save_path_cls_i_src_0, exist_ok=True)
            os.makedirs(save_path_cls_i_src_1, exist_ok=True)

    if save_vis_img:
        for i in range(n_cls):
            save_path_cls_i_vis_0 = save_base_path + "/C_Plus_Plus_det_output/{}/vis_images/cls_{}/0".format(dataset_name, i)
            save_path_cls_i_vis_1 = save_base_path + "/C_Plus_Plus_det_output/{}/vis_images/cls_{}/1".format(dataset_name, i)
            os.makedirs(save_path_cls_i_vis_0, exist_ok=True)
            os.makedirs(save_path_cls_i_vis_1, exist_ok=True)

    if save_crop_img:
        for i in range(n_cls):
            save_path_cls_i_crop_0 = save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/cls_{}/0/{}".format(dataset_name, i, crop_expand_ratio)
            save_path_cls_i_crop_1 = save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/cls_{}/1/{}".format(dataset_name, i, crop_expand_ratio)
            os.makedirs(save_path_cls_i_crop_0, exist_ok=True)
            os.makedirs(save_path_cls_i_crop_1, exist_ok=True)

    if save_no_det_res_img:
        save_path_no_det_res = save_base_path + "/C_Plus_Plus_det_output/{}/no_det_res".format(dataset_name)
        os.makedirs(save_path_no_det_res, exist_ok=True)

    with open(txt_path, "r", encoding="utf-8") as fo:
        lines = fo.readlines()
        for l in tqdm(lines):
            ff = l.strip().split(" ")
            fpath = ff[0]
            fname = os.path.basename(fpath)
            if len(ff) <= 1:
                if save_no_det_res_img:
                    shutil.copy(fpath, "{}/{}".format(save_path_no_det_res, fname))
                continue

            res = list(map(float, ff[1:]))
            np_res = np.asarray(res).reshape(-1, 7)

            img = cv2.imread(fpath)
            img_cp = img.copy()
            h, w = img.shape[:2]

            label_i_sum_ = {}
            label_i_flag = {}

            for j in range(n_cls):
                label_i_sum_["label_{}_sum_".format(j)] = 0
                label_i_flag["label_{}_flag".format(j)] = False

            for i in range(len(np_res)):
                pred_label = int(np_res[i][4])

                for n in range(n_cls):
                    if pred_label == n:
                        label_i_flag["label_{}_flag".format(n)] = True
                        x1y1x2y2_VOC = [int(np_res[i][0]), int(np_res[i][1]), int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])]
                        cropped_img = crop_img_via_expand(img_cp, x1y1x2y2_VOC, [h, w], crop_expand_ratio)

                        cls_np_res = int(np_res[i][6])
                        if cls_np_res == 0:
                            if save_crop_img:
                                cv2.imwrite("{}/{}_{}_{}.jpg".format(save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/cls_{}/0/{}".format(dataset_name, n, crop_expand_ratio), fname.split(".")[0], i, crop_expand_ratio), cropped_img)
                            if save_vis_img:
                                cv2.rectangle(img, (int(np_res[i][0]), int(np_res[i][1])), (int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])), (0, 0, 255), 2)
                                cv2.putText(img, "{}: {}".format(int(np_res[i][4]), np_res[i][5]), (int(np_res[i][0]), int(np_res[i][1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                        else:
                            if save_crop_img:
                                cv2.imwrite("{}/{}_{}_{}.jpg".format(save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/cls_{}/1/{}".format(dataset_name, n, crop_expand_ratio), fname.split(".")[0], i, crop_expand_ratio), cropped_img)
                            if save_vis_img:
                                cv2.rectangle(img, (int(np_res[i][0]), int(np_res[i][1])), (int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])), (0, 255, 0), 2)
                                cv2.putText(img, "{}: {}".format(int(np_res[i][4]), np_res[i][5]), (int(np_res[i][0]), int(np_res[i][1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                            label_i_sum_["label_{}_sum_".format(n)] += 1

            for n in range(n_cls):
                if label_i_flag["label_{}_flag".format(n)]:
                    if label_i_sum_["label_{}_sum_".format(n)] == 0:
                        if save_vis_img:
                            cv2.imwrite("{}/{}".format(save_base_path + "/C_Plus_Plus_det_output/{}/vis_images/cls_{}/0".format(dataset_name, n), fname), img)
                        if save_src_img:
                            shutil.copy(fpath, "{}/{}".format(save_base_path + "/C_Plus_Plus_det_output/{}/src_images/cls_{}/0".format(dataset_name, n), fname))
                    else:
                        if save_vis_img:
                            cv2.imwrite("{}/{}".format(save_base_path + "/C_Plus_Plus_det_output/{}/vis_images/cls_{}/1".format(dataset_name, n), fname), img)
                        if save_src_img:
                            shutil.copy(fpath, "{}/{}".format(save_base_path + "/C_Plus_Plus_det_output/{}/src_images/cls_{}/1".format(dataset_name, n), fname))


def remove_yolo_label_specific_class(data_path, rm_cls=(1, 2,)):
    curr_labels_path = data_path + "/labels"
    save_labels_path = data_path + "/labels_new"
    os.makedirs(save_labels_path, exist_ok=True)

    txt_list = sorted(os.listdir(curr_labels_path))
    for txt in tqdm(txt_list):
        txt_abs_path = curr_labels_path + "/{}".format(txt)
        txt_new_abs_path = save_labels_path + "/{}".format(txt)
        txt_data = open(txt_abs_path, "r", encoding="utf-8")
        txt_data_new = open(txt_new_abs_path, "w", encoding="utf-8")
        lines = txt_data.readlines()
        for l in lines:
            cls = l.strip().split(" ")[0]
            correctN = 0
            for rmclsi in rm_cls:
                if int(cls) != rmclsi:
                    correctN += 1

            if correctN == len(rm_cls):
                l_new = l
                # l_new = str(int(cls) - 1) + l[1:]
                txt_data_new.write(l_new)

        txt_data.close()
        txt_data_new.close()

        # Remove empty file
        txt_data_new_r = open(txt_new_abs_path, "r", encoding="utf-8")
        lines_new_r = txt_data_new_r.readlines()
        txt_data_new_r.close()
        if not lines_new_r:
            os.remove(txt_new_abs_path)
            print("os.remove: {}".format(txt_new_abs_path))


def convert_Stanford_Dogs_to_yolo_format(data_path):
    import xml.etree.ElementTree as ET

    img_path = data_path + "/Images"
    anno_path = data_path + "/annotation/Annotation"

    # save_path = data_path + "/annotation/yolo_labels"
    # os.makedirs(save_path, exist_ok=True)

    img_dir_list = os.listdir(img_path)
    xml_dir_list = os.listdir(anno_path)

    classes = []
    for d in img_dir_list:
        dog_name = d.split("-")[1]
        if dog_name not in classes:
            classes.append(dog_name)

    for d in xml_dir_list:
        d_path = anno_path + "/{}".format(d)

        save_path = data_path + "/annotation/yolo_labels/{}".format(d)
        os.makedirs(save_path, exist_ok=True)

        xml_list = os.listdir(d_path)
        for i, f_name in enumerate(xml_list):
            xml_abs_path = d_path + "/{}".format(f_name)
            try:
                in_file = open(xml_abs_path, "r", encoding='utf-8')
                out_file = open('{}/{}.txt'.format(save_path, f_name), 'w', encoding='utf-8')

                tree = ET.parse(in_file)
                root = tree.getroot()
                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)

                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    if cls not in classes or int(difficult) == 1:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
                    bb = bbox_voc_to_yolo((h, w), b)
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

                in_file.close()
                out_file.close()

            except Exception as Error:
                print("Error: {} {}".format(Error, f_name))


def convert_WiderPerson_to_yolo_format(data_path):
    img_path = data_path + "/Images"
    lbl_path = data_path + "/Annotations"

    save_path = data_path + "/labels"
    os.makedirs(save_path, exist_ok=True)

    lbl_list = sorted(os.listdir(lbl_path))
    for lbl in lbl_list:
        f_name = os.path.splitext(lbl)[0]
        lbl_abs_path = lbl_path + "/{}".format(lbl)
        lbl_new_path = save_path + "/{}".format(lbl)
        img_abs_path = img_path + "/{}.jpg".format(f_name)
        img = cv2.imread(img_abs_path)
        img_shape = img.shape[:2]

        orig_lbl = open(lbl_abs_path, "r", encoding="utf-8")
        new_lbl = open(lbl_new_path, "w", encoding="utf-8")

        orig_lbl_data = orig_lbl.readlines()
        for i, l in enumerate(orig_lbl_data):
            if i == 0: continue
            l_ = l.strip()
            cls = l_[0]
            VOC_bb = list(map(int, l_[2:].split(" ")))
            # VOC_bb = list(np.array([VOC_bb])[:, [0, 2, 1, 3]][0])
            # yolo_bb = convert_bbx_VOC_to_yolo(img_shape, VOC_bb)
            yolo_bb = bbox_voc_to_yolo(img_shape, VOC_bb)
            txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in yolo_bb]) + "\n"
            new_lbl.write(txt_content)

        orig_lbl.close()
        new_lbl.close()


def convert_TinyPerson_to_yolo_format(data_path):
    data_type = ["train", "test"]
    dense_or_not = ["", "dense"]
    for dt in data_type:
        for d in dense_or_not:
            save_path = data_path + "/yolo_format/{}/labels_{}".format(dt, d)
            os.makedirs(save_path, exist_ok=True)

            json_data = None
            if not d:
                json_data = json.load(open(data_path + "/annotations/tiny_set_{}.json".format(dt), "r", encoding="utf-8"))
            else:
                json_data = json.load(open(data_path + "/annotations/tiny_set_{}_with_dense.json".format(dt), "r", encoding="utf-8"))

            images = json_data["images"]
            categories = json_data["categories"]

            for i in range(len(images)):
                img_abs_path = data_path + "/{}/{}".format(dt, images[i]["file_name"])
                txt_abs_path = save_path + "/{}.txt".format(os.path.splitext(os.path.basename(images[i]["file_name"]))[0])
                bbxes = []
                for ann in json_data["annotations"]:
                    VOC_bbx = ann["bbox"]
                    VOC_bbx = [VOC_bbx[0], VOC_bbx[1], VOC_bbx[0] + VOC_bbx[2], VOC_bbx[1] + VOC_bbx[3]]

                    category_id = ann["category_id"]
                    area = ann["area"]
                    iscrowd = ann["iscrowd"]
                    image_id = ann["image_id"]
                    id = ann["id"]
                    logo = ann["logo"]
                    ignore = ann["ignore"]
                    in_dense_image = ann["in_dense_image"]

                    if image_id != i:
                        continue
                    if logo:
                        continue

                    img_shape = [images[image_id]["height"], images[image_id]["width"]]
                    yolo_bbx = bbox_voc_to_yolo(img_shape, VOC_bbx)

                    if ignore:
                        # yolo_bbx.insert(0, int(category_id) - 1)
                        yolo_bbx.insert(0, 1)
                        bbxes.append(yolo_bbx)
                    else:
                        # yolo_bbx.insert(0, int(category_id) - 1)
                        yolo_bbx.insert(0, 0)
                        bbxes.append(yolo_bbx)

                with open(txt_abs_path, "w", encoding="utf-8") as fw:
                    for bb in bbxes:
                        # txt_content = "{}".format(bb[0]) + " " + " ".join([str(b) for b in bb[1:]]) + "\n"
                        txt_content = "{}".format(bb[0]) + " " + " ".join([str(b) for b in bb[1:]]) + "\n"
                        fw.write(txt_content)


def convert_AI_TOD_to_yolo_format(data_path):
    # train_data_path = data_path + "/train"
    # test_data_path = data_path + "/test"
    # val_data_path = data_path + "/val"

    # classes = []
    # lbl_path = "/home/zengyifan/wujiahu/data/Open_Dataset/AI-TOD/train/labels"
    # lbl_list = sorted(os.listdir(lbl_path))
    # for lbl in lbl_list:
    #     lbl_abs_path = lbl_path + "/{}".format(lbl)
    #     fo =open(lbl_abs_path, "r", encoding="utf-8")
    #     txt_data = fo.readlines()
    #     fo.close()
    #
    #     for l in txt_data:
    #         l = l.strip().split(" ")
    #         cls = l[-1]
    #         if cls not in classes:
    #             classes.append(cls)
    #
    # print(classes, len(classes))
    # ['person', 'vehicle', 'ship', 'airplane', 'storage-tank', 'bridge', 'wind-mill', 'swimming-pool']

    classes = ['person', 'vehicle', 'ship', 'airplane', 'storage-tank', 'bridge', 'wind-mill', 'swimming-pool']
    dt = ["train", "val"]
    for d in dt:
        d_img_path = data_path + "/{}/images".format(d)
        d_lbl_path = data_path + "/{}/labels-orig".format(d)

        save_lbl_path = data_path + "/{}/labels".format(d)
        os.makedirs(save_lbl_path, exist_ok=True)

        img_list = sorted(os.listdir(d_img_path))
        for img in img_list:
            img_name = os.path.splitext(img)[0]
            img_abs_path = d_img_path + "/{}".format(img)
            lbl_abs_path = d_lbl_path + "/{}.txt".format(img_name)
            lbl_dst_path = save_lbl_path + "/{}.txt".format(img_name)

            img = cv2.imread(img_abs_path)
            img_shape = img.shape[:2]

            txt_fo = open(lbl_abs_path, "r", encoding="utf-8")
            txt_data = txt_fo.readlines()
            txt_fw = open(lbl_dst_path, "w", encoding="utf-8")

            for line in txt_data:
                l = line.strip().split(" ")
                cls = classes.index(l[-1])
                bbx = list(map(float, l[:4]))
                # bbx = list(np.array([bbx])[:, [0, 2, 1, 3]][0])
                # bbx_yolo = convert_bbx_VOC_to_yolo(img_shape, bbx)
                bbx_yolo = bbox_voc_to_yolo(img_shape, bbx)

                txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bbx_yolo]) + "\n"
                txt_fw.write(txt_content)

            txt_fw.close()


def vis_coco_pose_dataset():
    img_path = "/home/zengyifan/wujiahu/data/000.Open_Dataset/coco/train2017/000000000036.jpg"
    label_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/coco_kpts/labels/train2017/000000000036.txt"

    img = cv2.imread(img_path)
    imgsz = img.shape[:2]

    with open(label_path, "r", encoding="utf-8") as fo:
        lines = fo.readlines()
        for l in lines:
            l = l.strip().split(" ")
            cls = int(l[0])
            bbx = list(map(float, l[1:5]))
            # bbx_voc = convert_bbx_yolo_to_VOC(imgsz, bbx)
            bbx_voc = bbox_yolo_to_voc(imgsz, bbx)
            cv2.rectangle(img, (bbx_voc[0], bbx_voc[1]), (bbx_voc[2], bbx_voc[3]), (255, 255, 0))

            points = np.asarray(list(map(float, l[5:]))).reshape(-1, 3)
            points_x = points[:, 0] * imgsz[1]
            points_y = points[:, 1] * imgsz[0]
            for i in range(points_x.shape[0]):
                if points_x[i] == 0 and points_y[i] == 0:
                    continue
                cv2.circle(img, (int(round(points_x[i])), int(round(points_y[i]))), 3, (255, 0, 255), 2)

    cv2.imshow("test", img)
    cv2.waitKey(0)


def get_bbx(kpts, imgsz, r=0.68):
    minx = min([xi for xi in kpts[:, 0]])
    maxx = max([xi for xi in kpts[:, 0]])
    miny = min([yi for yi in kpts[:, 1]])
    maxy = max([yi for yi in kpts[:, 1]])
    ymid = (miny + maxy) / 2
    w_ = maxx - minx
    y_half = w_ * r
    y1 = ymid - y_half
    y2 = ymid + y_half
    area = abs(maxx - minx) * abs(y2 - y1)

    y1_ = ymid - y_half - y_half * 0.5
    y2_ = ymid + y_half + y_half * 0.005
    minx_ = minx - minx * 0.025
    maxx_ = maxx + maxx * 0.025
    if y1_ < 0: y1_ = 0
    if y2_ > imgsz[0]: y2_ = imgsz[0]
    if minx_ < 0: minx_ < 0
    if maxx_ > imgsz[1]: maxx_ = imgsz[1]
    bbx = [minx_, y1_, maxx_, y2_]

    return bbx, area


def write_label(fpath, bboxes, cls):
    with open(fpath, "w", encoding="utf-8") as fw:
        for bb in bboxes:
            txt_content = "{} ".format(cls) + " ".join([str(bi) for bi in bb]) + "\n"
            fw.write(txt_content)


def create_labels_via_yolo_pose(data_path, cls=2):
    from ultralytics import YOLO

    model = YOLO("/home/zengyifan/wujiahu/yolo/ultralytics-main/yolov8s-pose.pt")

    dir_name = get_dir_name(data_path)
    file_list = get_file_list(data_path)
    save_path = make_save_path(data_path, dir_name_add_str="labels_head")

    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        base_name = get_base_name(f_abs_path)
        file_name = os.path.splitext(base_name)[0]
        suffix = os.path.splitext(base_name)[1]
        img = cv2.imread(f_abs_path)
        imgsz = img.shape[:2]
        bboxes = []

        results = model(f_abs_path)
        for r in results:
            keypoints = r.keypoints
            kpt_np = keypoints.xy.cpu().numpy()
            for pi in kpt_np:
                # for k in pi[:5]:
                # cv2.circle(img, (int(k[0]), int(k[1])), 2, (255, 0, 255))
                if len(pi[:5]) < 5: continue
                bbx, area = get_bbx(pi[:5], imgsz, r=0.68)
                if area < 500:
                    continue

                # bbx_yolo = convert_bbx_VOC_to_yolo(imgsz, bbx)
                bbx_yolo = bbox_voc_to_yolo(imgsz, bbx)
                bboxes.append(bbx_yolo)

        txt_save_path = "{}/{}.txt".format(save_path, file_name)
        write_label(txt_save_path, bboxes, cls=cls)


def get_coco_names():
    names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}
    return names


# 读取出图像中的目标框
def read_xml(root, image_id):
    import xml.etree.ElementTree as ET

    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    bndbox = root.find('object').find('bndbox')
    return bndboxlist  # 以多维数组的形式保存


# 将xml文件中的旧坐标值替换成新坐标值,并保存,这个程序里面没有使用
# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml(root, image_id, new_target):
    import xml.etree.ElementTree as ET

    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str(image_id) + "_aug" + '.xml'))


# SEG 
def convert_seg_0_255_to_0_n(image, c="3"):
    """
    根据实际进行修改
    c = 1 or c = 3
    """
    target = np.where((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 128))
    # yellow = np.where((image_to_write[:, :, 0] != 255) & (image_to_write[:, :, 1] != 255) & (image_to_write[:, :, 2] != 255))
    # green = np.where((image_to_write[:, :, 0] == 0) & (image_to_write[:, :, 1] == 128) & (image_to_write[:, :, 2] == 0))

    dst = None
    if c == 1:
        dst = np.zeros((image.shape[:2]), dtype=np.int32)
        dst[target] = 1
    elif c == 3:
        dst = np.zeros(image.shape, dtype=np.int32)
        dst[target] = (1, 1, 1)
    else:
        print("Error!")

    return dst
    

def create_Camvid_train_val_txt(base_path):
    img_path = base_path + "\\train"
    lbl_path = base_path + "\\trainanno"
    img_list = os.listdir(img_path)

    save_path = "{}/camvid_trainval_list.txt".format(base_path).replace("\\", "/")
    with open(save_path, "w+", encoding="utf8") as f:
        for img in img_list:
            img_abs_path = "train" + "/" + img
            label_name = img.replace("jpg", "png")
            lbl_abs_path = "trainanno" + "/" + label_name
            f.writelines(img_abs_path + " " + lbl_abs_path + "\n")

    print("Created --> {}".format(save_path))


if __name__ == '__main__':
    pass






















