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


import os
import re
import sys
import PIL.Image
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
from PIL import (
    Image, ImageDraw,
    ImageOps, ImageFont
)
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
from torch import nn
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from labelme import utils
import subprocess


# Base utils ===================================================
def timestamp_to_strftime(timestamp: float):
    if timestamp is None or timestamp == "":
        strftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        return strftime
    else:
        assert type(timestamp) == float, "timestamp should be float!"
        strftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        return strftime


def strftime_to_timestamp(strftime: str):
    """
    strftime = "2024-11-06 12:00:00"
    """
    assert strftime is not None or strftime != "", "strftime is empty!"
    struct_time = time.strptime(strftime, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(struct_time)
    return timestamp


def get_date_time(mode=0):
    """
    0: %Y-%m-%d %H:%M:%S
    1: %Y %m %d %H:%M:%S
    2: %Y/%m/%d %H:%M:%S
    """
    if mode == 0:
        datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        return datetime
    elif mode == 1:
        datetime = time.strftime("%Y %m %d %H:%M:%S", time.localtime(time.time()))
        return datetime
    elif mode == 2:
        datetime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))
        return datetime
    else:
        print("mode should be 0, 1, 2!")


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
    print("Create directory successful! save_path: {}".format(save_path))
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
    assert algorithm in ["pil", "cv2"], 'algorithm in ["pil", "cv2"]!'
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


def flip(img, random=False, p=1, m=np.random.choice([-1, 0, 1])):
    """
    0:垂直翻转(沿x轴翻转)
    1:水平翻转(沿y轴翻转)
    -1:同时在水平和垂直方向翻转

    """
    assert m in [-1, 0, 1], "m(flip direction) should be one of [-1, 0, 1]"

    if random:
        if np.random.random() <= p:
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


def normalize(img, random=False, p=1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX):
    """
    """
    assert norm_type == cv2.NORM_MINMAX or norm_type == cv2.NORM_L2, "norm_type: cv2.NORM_MINMAX or cv2.NORM_L2!"
    if random:
        if np.random.random() <= p:
            if norm_type == cv2.NORM_MINMAX:
                norm_img = cv2.normalize(img, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            else:
                norm_img = cv2.normalize(img, None, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            norm_img = (255 * norm_img).astype(np.uint8)
            return norm_img
        else:
            return img
    else:
        if norm_type == cv2.NORM_MINMAX:
            norm_img = cv2.normalize(img, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            norm_img = cv2.normalize(img, None, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        norm_img = (255 * norm_img).astype(np.uint8)
        return norm_img
    

def equalize_hist(img, random=False, p=1, m=np.random.choice([0, 1])):
    assert m in [0, 1], "m should be one of [0, 1]"
    if random:
        if np.random.random() <= p:
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
            v = np.clip(v, 0, 255)
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
        v = np.clip(v, 0, 255)
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
    
    if random:
        # alpha建议>= 0.1，不然容易变黑图
        assert isinstance(alpha, tuple), "If random=True, alpha should be tuple!"
        assert isinstance(beta, tuple), "If random=True, beta should be tuple!"
        if np.random.random() <= p:
            alpha = np.random.uniform(alpha[0], alpha[1])
            beta = np.random.randint(beta[0], beta[1] + 1)
            blank = np.zeros(img.shape, img.dtype)  # 创建图片类型的零矩阵
            img = cv2.addWeighted(np.uint8(img), alpha, np.uint8(blank), 1 - alpha, beta)  # 图像混合加权
            return img
        else:
            return img
    else:
        assert isinstance(alpha, float), "If random=False, alpha should be float!"
        assert isinstance(beta, int), "If random=False, beta should be int!"
        assert alpha >= 0 and alpha <= 1, "alpha >= 0 and alpha <= 1"
        blank = np.zeros(img.shape, img.dtype)  # 创建图片类型的零矩阵
        img = cv2.addWeighted(np.uint8(img), alpha, np.uint8(blank), 1 - alpha, beta)  # 图像混合加权
        return img
    

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
        assert isinstance(clipLimit, tuple), "If random=True, clipLimit should be tuple!"
        if np.random.random() <= p:
            clipLimit = np.random.randint(clipLimit[0], clipLimit[1] + 1)
            tgs = np.random.randint(tileGridSize[0], tileGridSize[1] + 1)
            # tgs = np.random.choice([4, 8, 16, 32])
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
        assert isinstance(clipLimit, float), "If random=False, clipLimit should be float!"
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
            hgain = np.random.uniform(hgain[0], hgain[1])
            sgain = np.random.uniform(sgain[0], sgain[1])
            vgain = np.random.uniform(vgain[0], vgain[1])
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
            img = img.astype(np.uint8)
            hgain = np.random.uniform(hgain[0], hgain[1])
            sgain = np.random.uniform(sgain[0], sgain[1])
            vgain = np.random.uniform(vgain[0], vgain[1])
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
    

def change_color(img, random=False, p=1, hue_shift=30):
    if random:
        if np.random.random() <= p:
            value = np.random.randint(0, 180)
            # 将图像从BGR颜色空间转换为HSV颜色空间
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 调整Hue值
            hsv_image[:, :, 0] = (hsv_image[:, :, 0] + value) % 180
            
            # 将图像从HSV颜色空间转换回BGR颜色空间
            new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            
            return new_image
        else:
            return new_image
    else:
        # 将图像从BGR颜色空间转换为HSV颜色空间
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 调整Hue值
            hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
            
            # 将图像从HSV颜色空间转换回BGR颜色空间
            new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            
            return new_image


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
            hue_v = np.random.randint(value[0], value[1])
            hsv_image = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_v) % 180  # 在Hue通道上增加30
            hsv_image = np.clip(hsv_image, 0, 255)
            img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            return img
    else:
        assert isinstance(value, int), "If random=False, value should be int!"
        hsv_image = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + value) % 180  # 在Hue通道上增加30
        hsv_image = np.clip(hsv_image, 0, 255)
        img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return img
    

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
                assert min_size[1] < mask_size[1], "min_size[1] < mask_size[1]"
                assert min_size[0] < mask_size[0], "min_size[0] < mask_size[0]"
                mask_size_x = np.random.randint(min_size[1], mask_size[1])
                mask_size_y = np.random.randint(min_size[0], mask_size[0])
                mask_size = (mask_size_x, mask_size_y)

            mask_x = np.random.randint(0, max(imgsz[1] - mask_size[1], 1))
            mask_y = np.random.randint(0, max(imgsz[0] - mask_size[0], 1))

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
    

def transperent_overlay(img, random=False, p=1, rect=(50, 50, 100, 80), max_h_r=1.0, max_w_r=0.25, alpha=(0.1, 1.0)):
    """
    rect: [x1, y1, x2, y2]
    """
    if random:
        if np.random.random() <= p:
            imgsz = img.shape
            orig_c = imgsz[2]
            max_h = int(imgsz[0] * max_h_r)
            max_w = int(imgsz[1] * max_w_r)

            alpha = np.random.uniform(alpha[0], alpha[1])

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
            img = cv2.addWeighted(np.uint8(overlay), alpha, np.uint8(img), 1 - alpha, 0)

            # Convert the image back to the original number of channels
            if orig_c != img.shape[2]:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        else:
            return img
    else:
        imgsz = img.shape
        orig_c = imgsz[2]
        alpha = np.random.uniform(alpha[0], alpha[1])
        color = [np.random.randint(0, 256) for _ in range(3)]

        if imgsz[2] < 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # 创建一个与图片大小相同的覆盖层
        # overlay = img.copy()
        overlay = np.ones(shape=img.shape, dtype=np.uint8)
        x1, y1 = rect[0], rect[1]
        x2, y2 = rect[2], rect[3]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(np.uint8(overlay), alpha, np.uint8(img), 1 - alpha, 0)

        # Convert the image back to the original number of channels
        if orig_c != img.shape[2]:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    

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
    if random:
        assert isinstance(mean, tuple), "If random=True, mean should be tuple!"
        assert isinstance(var, tuple), "If random=True, var should be tuple!"
        if np.random.random() <= p:
            mean = np.random.randint(mean[0], mean[1])
            var = np.random.uniform(var[0], var[1])
            mu, sigma = mean, var ** 0.5
            gaussian = np.random.normal(mu, sigma, img.shape)
            img = cv2.add(np.uint8(img), np.uint8(gaussian))
            return img
        else:
            return img
    else:
        assert isinstance(mean, int), "If random=False, mean should be int!"
        assert isinstance(var, float), "If random=False, var should be float!"
        mu, sigma = mean, var ** 0.5
        gaussian = np.random.normal(mu, sigma, img.shape)
        img = cv2.add(np.uint8(img), np.uint8(gaussian))
        return img


def poisson_noise(img, random=False, p=1, n=2):
    if random:
        assert isinstance(n, tuple), "If random=False, n should be tuple!"
        if np.random.random() <= p:
            vals = len(np.unique(img))
            n = np.random.randint(n[0], n[1])
            vals = n ** np.ceil(np.log2(vals))
            poisson = np.random.poisson(img * vals) / float(vals)
            img = cv2.add(np.uint8(img), np.uint8(poisson))
            return img
        else:
            return img
    else:
        assert isinstance(n, int), "If random=False, n should be int!"
        vals = len(np.unique(img))
        vals = n ** np.ceil(np.log2(vals))
        poisson = np.random.poisson(img * vals) / float(vals)
        img = cv2.add(np.uint8(img), np.uint8(poisson))
        return img


def sp_noise(img, random=False, p=1, salt_p=0.01, pepper_p=0.01):
    """
    salt and pepper noise
    """
    
    if random:
        assert isinstance(salt_p, tuple), "If random=True, salt_p should be tuple!"
        assert isinstance(pepper_p, tuple), "If random=True, pepper_p should be tuple!"
        if np.random.random() <= p:
            salt_p = np.random.uniform(salt_p[0], salt_p[1])
            pepper_p = np.random.uniform(pepper_p[0], pepper_p[1])

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
        assert isinstance(salt_p, float), "If random=False, salt_p should be float!"
        assert isinstance(pepper_p, float), "If random=False, pepper_p should be float!"
        assert salt_p >= 0 and salt_p <= 1, "salt_p >= 0 and salt_p <= 1!"
        assert pepper_p >= 0 and pepper_p <= 1, "salt_p >= 0 and salt_p <= 1!"

        noisy_image = np.copy(img)
        total_pixels = img.shape[0] * img.shape[1]  # 计算图像的总像素数

        num_salt = int(total_pixels * salt_p)  # 通过将总像素数与指定的椒盐噪声比例相乘,得到要添加的椒盐噪声的数量。
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
        noisy_image[salt_coords[0], salt_coords[1]] = 255

        num_pepper = int(total_pixels * pepper_p)
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_image


def gaussian_blur(img, random=False, p=1, k=3):
    if random:
        if np.random.random() <= p:
            h, w, _ = img.shape
            ks = [3, 5, 7, 9]
            if h > 16 and w > 16:
                if h <= 128 and w <= 128:
                    k = np.random.choice(ks[:2])
                else:
                    k = np.random.choice(ks)
                img = cv2.GaussianBlur(img, (k, k), 0)

            return img
        else:
            return img
    else:
        img = cv2.GaussianBlur(img, (k, k), 0)
        return img


def motion_blur(img, random=False, p=1, k=3, angle=30):
    """
    假如用于增强OCR数据, 则k不宜太大!
    """
    if random:
        if np.random.random() <= p:
            angle = np.random.randint(angle[0], angle[1] + 1)
            imgsz = img.shape[:2]
            ks = [3, 5, 7, 9]
            if imgsz[0] > 16 and imgsz[1] > 16:
                if imgsz[0] <= 128 and imgsz[1] <= 128:
                    k = np.random.choice(ks[:2])
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
            imgsz = img.shape[:2]
            ks = [3, 5, 7, 9]
            if imgsz[0] > 16 and imgsz[1] > 16:
                if imgsz[0] <= 128 and imgsz[1] <= 128:
                    k = np.random.choice(ks[:2])
                else:
                    k = np.random.choice(ks)
                img = cv2.medianBlur(np.uint8(img), k)
            return img
        else:
            return img
    else:
        img = cv2.medianBlur(np.uint8(img), k)

        return img


def dilate_erode(img, random=False, p=1, flag="dilate", k=(3, 3)):
    """
    dilate, erode
    """
    assert flag in ["dilate", "erode"], 'flag should be one of ["dilate", "erode"]!'
    if random:
        if np.random.random() <= p:
            imgsz = img.shape[:2]
            if min(imgsz) > 512:
                k = (5, 5)
            else:
                k = (3, 3)
            
            kernel = np.ones(k, dtype=np.uint8)
            if flag == "dilate":
                img = cv2.dilate(img, kernel, iterations=1)
            else:
                img = cv2.erode(img, kernel, iterations=1)
            return img
        else:
            return img
    else:
        # kernel = cv2.getStructuringElement(
        #     cv2.MORPH_ELLIPSE, tuple(np.random.randint(scale[0], scale[1], 2))
        # )

        kernel = np.ones(k, dtype=np.uint8)
        if flag == "dilate":
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)
        return img


def open_close_gradient(img, random=False, p=1, flag="open", k=(3, 3)):
    """
    open, close, gradient
    """
    assert flag in ["open", "close", "gradient"], 'flag should be one of ["open", "close", "gradient"]!'
    if random:
        if np.random.random() <= p:
            imgsz = img.shape[:2]
            if min(imgsz) > 512:
                k = (5, 5)
            else:
                k = (3, 3)
            
            kernel = np.ones(k, dtype=np.uint8)
            if flag == "open":
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            elif flag == "close":
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            else:
                img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
            return img
        else:
            return img
    else:
        # kernel = cv2.getStructuringElement(
        #     cv2.MORPH_ELLIPSE, tuple(np.random.randint(scale[0], scale[1], 2))
        # )

        kernel = np.ones(k, dtype=np.uint8)
        if flag == "open":
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif flag == "close":
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        else:
            img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        return img
    

def tophat_blackhat(img, random=False, p=1, flag="tophat", k=(3, 3)):
    """
    tophat, blackhat
    """
    assert flag in ["tophat", "blackhat"], 'flag should be one of ["tophat", "blackhat"]!'
    if random:
        if np.random.random() <= p:
            imgsz = img.shape[:2]
            if min(imgsz) > 512:
                k = (5, 5)
            else:
                k = (3, 3)
            
            kernel = np.ones(k, dtype=np.uint8)
            if flag == "tophat":
                img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
            else:
                img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
            return img
        else:
            return img
    else:
        # kernel = cv2.getStructuringElement(
        #     cv2.MORPH_ELLIPSE, tuple(np.random.randint(scale[0], scale[1], 2))
        # )

        kernel = np.ones(k, dtype=np.uint8)
        if flag == "tophat":
            img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        else:
            img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
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
    result = cv2.addWeighted(np.uint8(img), alpha, np.uint8(rain), 1 - alpha, 1)

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


def make_sunlight_effect(img, random=False, p=1, center=(50, 50), effect_r=(50, 200), light_strength=(50, 150)):
    if random:
        assert isinstance(effect_r, tuple), "If random=True, effect_r should be tuple!"
        assert isinstance(light_strength, tuple), "If random=True, light_strength should be tuple!"
        if np.random.random() <= p:
            imgsz = img.shape[:2]
            center = (np.random.randint(0, imgsz[1]), np.random.randint(0, imgsz[0]))
            effectR = np.random.randint(effect_r[0], effect_r[1])
            lightStrength = np.random.randint(light_strength[0], light_strength[1])

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
        assert isinstance(light_strength, int), "If random=False, light_strength should be int!"

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
    

class WaveDeformer():
    def __init__(self, a=10, b=40, gridspace=20):
        self.a = a
        self.b = b
        self.gridspace = gridspace

    def transform(self, x, y):
        y = y + self.a * math.sin(x / self.b)
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (*self.transform(x0, y0),
                *self.transform(x0, y1),
                *self.transform(x1, y1),
                *self.transform(x1, y0),
                )

    def getmesh(self, img):
        self.w, self.h = img.size
        gridspace = self.gridspace

        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))

        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]

        return [t for t in zip(target_grid, source_grid)]
    
    
def warp_and_deform(img, random=False, p=1, a=10, b=40, gridspace=20):
    """
    """
    if random:
        if np.random.random() <= p:
            assert isinstance(a, tuple), "If random=True, q should be tuple!"
            assert isinstance(b, tuple), "If random=True, b should be tuple!"
            assert isinstance(gridspace, tuple), "If random=True, gridspace should be tuple!"
            a = np.random.randint(a[0], a[1])
            b = np.random.randint(b[0], b[1])
            gridspace = np.random.randint(gridspace[0], gridspace[1])
            img = cv2pil(img)
            img = ImageOps.deform(img, WaveDeformer(a=a, b=b, gridspace=gridspace))
            return pil2cv(img)
        else:
            return img
    else:
        assert isinstance(a, float), "If random=False, q should be float!"
        assert isinstance(b, float), "If random=False, b should be float!"
        assert isinstance(gridspace, int), "If random=False, gridspace should be int!"
        img = cv2pil(img)
        img = ImageOps.deform(img, WaveDeformer(a=a, b=b, gridspace=gridspace))
        return pil2cv(img)


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


def byte2img(byte_data):
    """
    byte_data = b'Your byte data here'
    byte_io = io.BytesIO(byte_data)
    image = Image.open(byte_io)
    ----------------------------------

    with open("1.jpg", "r") as f:
        data=f.read()
    base64.b64encode(data)  # 图片转字节
    base64.b64decode(data)  # 字节转图片
    """
    byte_data = base64.b64decode(byte_data)
    nparr = np.frombuffer(byte_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img


def img2byte(img_path):
    with open(img_path, "rb") as f:
        data=f.read()
    byte_data = base64.b64encode(data)

    return byte_data


def connected_components_analysis(img, connectivity=8, area_thr=100, h_thr=8, w_thr=8):
    """
    stats: [x, y, w, h, area]
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=connectivity)
    
    areas = stats[:, -1]  # stats[:, cv2.CC_STAT_AREA]
    for i in range(1, num_labels):
        if areas[i] < area_thr:
            labels[labels == i] = 0
        else:
            if stats[i, 2] < w_thr or stats[i, 3] < h_thr:
                labels[labels == i] = 0

    # 不同的连通域赋予不同的颜色
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 256)
        output[:, :, 1][mask] = np.random.randint(0, 256)
        output[:, :, 2][mask] = np.random.randint(0, 256)

    return output, num_labels, labels, stats, centroids


def write_video(video, video_path, save_path):
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    if save_path is None or save_path == "":
        save_path = video_path + ".avi"
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width * 2,  height))

    return out


def x1y1wh_to_x1y1x2y2(x):
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] # bottom right y
    """
    y = [x[0], x[1], x[0] + x[2], x[1] + x[3]]

    return y


def merge_two_bboxes(b1, b2):
    xmin = min(b1[0], b2[0])
    ymin = min(b1[1], b2[1])
    xmax = max(b1[2], b2[2])
    ymax = max(b1[3], b2[3])

    assert xmin <= xmax and ymin <= ymax, "Merge bboxes error!"

    return [xmin, ymin, xmax, ymax]


def merge_bboxes(bboxes, iou_thresh=0.0):
    out_bboxes = []
    len_boxes = len(bboxes)
    merge_idxes = []

    for i in range(len_boxes - 1):
        for j in range(i + 1, len_boxes):
            iou = cal_iou(bboxes[i], bboxes[j])
            if iou > iou_thresh and bboxes[i] != bboxes[j]:
                merge_idxes.append([i, j])
    
    for idx, mi in enumerate(merge_idxes):
        merged_box = merge_two_bboxes(bboxes[mi[0]], bboxes[mi[1]])
        out_bboxes.append(merged_box)
        
    mi_list = []  # merge_idxes_list
    all_list = list(range(len_boxes))
    for idx, mi in enumerate(merge_idxes):
        if mi[0] not in mi_list:
            mi_list.append(mi[0])
        if mi[1] not in mi_list:
            mi_list.append(mi[1])

    nmi_list = list(set(mi_list) ^ set(all_list))  # not_merge_idxes_list
    for nmi in nmi_list:
        out_bboxes.append(bboxes[nmi])

    return out_bboxes, len(merge_idxes)


def draw_rect(frameDet, frameNowBGR, area_thresh=100, iou_thresh=0.0, object_thresh=200, flag_merge_bboxes=True):
    contours, hierarchy = cv2.findContours(frameDet, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tmp_bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < area_thresh: continue
        x1y1x2y2 = x1y1wh_to_x1y1x2y2([x, y, w, h])
        tmp_bboxes.append(x1y1x2y2)

    """
    # 显示未合并目标框的效果
    frameNowBGR_tmp = np.copy(frameNowBGR)
    for b_tmp in tmp_bboxes:
        x1_tmp, y1_tmp, x2_tmp, y2_tmp = b_tmp[0], b_tmp[1], b_tmp[2], b_tmp[3]
        cv2.rectangle(frameNowBGR_tmp, (x1_tmp, y1_tmp), (x2_tmp, y2_tmp), (255, 0, 255), 2)
    cv2.imshow('frameNowBGR_tmp', frameNowBGR_tmp)
    cv2.waitKey(1)
    """
    
    final_bboxes = tmp_bboxes
    if flag_merge_bboxes:
        id = 0
        len_mi = 0
        while True:
            if id > 0 and len_mi == 0 or len_mi > object_thresh or len(final_bboxes) > object_thresh: break
            final_bboxes, len_mi = merge_bboxes(final_bboxes, iou_thresh=iou_thresh)
            id += 1

    """
    # 显示合并目标框的效果
    for b in final_bboxes:
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        cv2.rectangle(frameNowBGR, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.imshow('frameNowBGR_merge_bboxes', frameNowBGR)
    cv2.waitKey(1)
    """

    for b in final_bboxes:
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        cv2.rectangle(frameNowBGR, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return frameNowBGR


def shield_time_watermark(img, time_watermark):
    imgsz = img.shape
    if time_watermark is not None:
        for twm in time_watermark:
            x, y, w, h = twm
            x = int(round(x * imgsz[1]))
            y = int(round(y * imgsz[0]))
            w = int(round(w * imgsz[1]))
            h = int(round(h * imgsz[0]))
            if len(imgsz) == 2:
                img[y:y + h, x:x + w] = 0
            else:
                img[y:y + h, x:x + w] = (0, 0, 0)

    return img


def moving_object_detect(video_path, m=3, area_thresh=100, scale_r=(0.5, 0.5), time_watermark=None, cca=True, flag_merge_bboxes=True, vis_result=False, save_path=None, debug=False):
    """
    param m: [2, 3], [两帧帧间差分法, 三帧帧间差分法]
    param cca: connected components analysis
    param time_watermark: [[x, y, w, h], [x, y, w, h]], ratio not pixel value! e.g. [[0, 0.0488, 0.4370, 0.0651]]
    """
    assert m in [2, 3], "m should be one of [2, 3]!"
    base_name = os.path.basename(video_path)
    suffix = os.path.splitext(base_name)[1]
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Open {} failed!".format(video_path))
        return -1
    
    if vis_result:
        out = write_video(video, video_path, save_path)

    # 为了效率没有将判断放进while True里面
    if m == 2:
        retPre, framePre = video.read()  # 上一帧

        if scale_r is not None:
            framePre = cv2.resize(framePre, dsize=None, fx=scale_r[0], fy=scale_r[1])

        framePreBGR = framePre.copy()
        framePre = cv2.cvtColor(framePre, cv2.COLOR_BGR2GRAY)
        framePre = shield_time_watermark(framePre, time_watermark)

        while True:
            ret, frameNow = video.read()  # 当前帧
            if not ret: break

            if scale_r is not None:
                frameNow = cv2.resize(frameNow, dsize=None, fx=scale_r[0], fy=scale_r[1])

            frameNowBGR = frameNow.copy()
            frameNow = cv2.cvtColor(frameNow, cv2.COLOR_BGR2GRAY)
            frameNow = shield_time_watermark(frameNow, time_watermark)
            frameDet = cv2.absdiff(framePre, frameNow)
            framePre = frameNow

            _, frameDet = cv2.threshold(frameDet, 127, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            # e = cv2.getStructuringElement(0, (3, 3))
            # frameDet = cv2.erode(frameDet, e)
            # frameDet = cv2.dilate(frameDet, e)

            if cca:
                analysis_output = connected_components_analysis(frameDet, connectivity=8, area_thr=area_thresh, h_thr=8, w_thr=8)
                output = analysis_output[0]  # output, num_labels, labels, stats, centroids
                output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                _, frameDet = cv2.threshold(output_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

            framePreBGR = frameNowBGR
            framePre = frameNow

            framePreBGR = draw_rect(frameDet, framePreBGR, area_thresh, 0.0, 200, flag_merge_bboxes)

            if vis_result:
                frameDet_3c = cv2.merge([frameDet, frameDet, frameDet])
                dst = np.hstack((framePreBGR, frameDet_3c))
                out.write(dst)

            if debug:
                cv2.imshow("frameDet", frameDet)
                cv2.waitKey(1)
    else:
        retPrePre, framePrePre = video.read()  # 上上帧
        retPre, framePre = video.read()  # 上一帧

        if scale_r is not None:
            framePrePre = cv2.resize(framePrePre, dsize=None, fx=scale_r[0], fy=scale_r[1])
            framePre = cv2.resize(framePre, dsize=None, fx=scale_r[0], fy=scale_r[1])

        framePrePreBGR = framePrePre.copy()
        framePreBGR = framePre.copy()
        framePrePre = cv2.cvtColor(framePrePre, cv2.COLOR_BGR2GRAY)
        framePre = cv2.cvtColor(framePre, cv2.COLOR_BGR2GRAY)
        framePrePre = shield_time_watermark(framePrePre, time_watermark)
        framePre = shield_time_watermark(framePre, time_watermark)
        
        while True:
            ret, frameNow = video.read()  # 当前帧
            if not ret: break

            if scale_r is not None:
                frameNow = cv2.resize(frameNow, dsize=None, fx=scale_r[0], fy=scale_r[1])

            frameNowBGR = frameNow.copy()
            frameNow = cv2.cvtColor(frameNow, cv2.COLOR_BGR2GRAY)
            frameNow = shield_time_watermark(frameNow, time_watermark)
            d1 = cv2.absdiff(framePrePre, framePre)
            d2 = cv2.absdiff(framePre, frameNow)
            _, thresh1 = cv2.threshold(d1, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            _, thresh2 = cv2.threshold(d2, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            e = cv2.getStructuringElement(0, (3, 3))
            thresh1 = cv2.dilate(thresh1, e)
            thresh2 = cv2.dilate(thresh2, e)
            frameDet = cv2.bitwise_and(thresh1, thresh2)

            if cca:
                analysis_output = connected_components_analysis(frameDet, connectivity=8, area_thr=area_thresh, h_thr=8, w_thr=8)
                output = analysis_output[0]  # output, num_labels, labels, stats, centroids
                output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                _, frameDet = cv2.threshold(output_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

            framePrePreBGR = framePreBGR
            framePreBGR = frameNowBGR
            framePrePre = framePre
            framePre = frameNow

            framePrePreBGR = draw_rect(frameDet, framePrePreBGR, area_thresh, 0.0, 200, flag_merge_bboxes)
            
            if vis_result:
                frameDet_3c = cv2.merge([frameDet, frameDet, frameDet])
                dst = np.hstack((framePrePreBGR, frameDet_3c))
                out.write(dst)

            if debug:
                cv2.imshow("frameDet", frameDet)
                cv2.waitKey(1)

    video.release()
    if vis_result:
        out.release()
    cv2.destroyAllWindows()

    return 0


# Object detection utils ===================================================
def bbox_voc_to_yolo(imgsz, box):
    """
    VOC --> YOLO
    :param imgsz: [H, W]
    :param box:
    orig: [xmin, xmax, ymin, ymax], deprecated;
    new:  [xmin, ymin, xmax, ymax], 2024.03.29, WJH.
    :return: [x, y, w, h]
    """
    dh = 1. / (imgsz[0])
    dw = 1. / (imgsz[1])
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
    assert x <= 1, "x: {}".format(x)
    assert y <= 1, "y: {}".format(y)
    assert w >= 0, "w: {}".format(w)
    assert h >= 0, "h: {}".format(h)

    return [x, y, w, h]


def bbox_yolo_to_voc(imgsz, bbx, return_decimal=False):
    """
    YOLO --> VOC
    !!!!!! orig: (bbx, imgsz) 20230329 changed to (imgsz, bbx)
    :param bbx: yolo format bbx
    :param imgsz: [H, W]
    :return: [x_min, y_min, x_max, y_max]
    """
    bbx_ = (bbx[0] * imgsz[1], bbx[1] * imgsz[0], bbx[2] * imgsz[1], bbx[3] * imgsz[0])

    if return_decimal:
        x_min = bbx_[0] - (bbx_[2] / 2)
        y_min = bbx_[1] - (bbx_[3] / 2)
        x_max = bbx_[0] + (bbx_[2] / 2)
        y_max = bbx_[1] + (bbx_[3] / 2)
    else:
        x_min = int(round(bbx_[0] - (bbx_[2] / 2)))
        y_min = int(round(bbx_[1] - (bbx_[3] / 2)))
        x_max = int(round(bbx_[0] + (bbx_[2] / 2)))
        y_max = int(round(bbx_[1] + (bbx_[3] / 2)))

    if x_min < 0: x_min = 0
    if y_min < 0: y_min = 0
    if x_max > imgsz[1]: x_max = imgsz[1]
    if y_max > imgsz[0]: y_max = imgsz[0]

    assert x_min >= 0 and x_min <= imgsz[1], "x_min: {}".format(x_min)
    assert y_min >= 0 and y_min <= imgsz[0], "y_min: {}".format(y_min)
    assert x_max >= 0 and x_max <= imgsz[1], "x_max: {}".format(x_max)
    assert y_max >= 0 and y_max <= imgsz[0], "y_max: {}".format(y_max)

    return [x_min, y_min, x_max, y_max]


def write_labelbee_det_json(bbx, imgsz):
    """
    {"x":316.6583427922815,"y":554.4245175936436,"width":1419.1872871736662,"height":556.1679909194097,
    "attribute":"1","valid":true,"id":"tNd2HY6C","sourceID":"","textAttribute":"","order":1}
    :param bbx: x1, y1, x2, y2
    :param imgsz: H, W
    :return:
    """

    chars = ""
    for i in range(48, 48 + 9):
        chars += chr(i)
    for j in range(65, 65 + 25):
        chars += chr(j)
    for k in range(97, 97 + 25):
        chars += chr(k)

    j = {}
    j["width"] = imgsz[1]
    j["height"] = imgsz[0]
    j["valid"] = True
    j["rotate"] = 0

    step_1 = {}
    step_1["toolName"] = "rectTool"

    ids = []
    while (len(ids) < len(bbx)):
        id_ = random.sample(chars, 8)
        id_ = "".join([d for d in id_])
        if id_ not in ids:
            ids.append(id_)
    assert len(ids) == len(bbx), "len(ids) != len(bbxs)"

    result = []
    for i in range(len(bbx)):
        result_dict = {}
        result_dict["x"] = bbx[i][0]
        result_dict["y"] = bbx[i][1]
        result_dict["width"] = bbx[i][2] - bbx[i][0]
        result_dict["height"] = bbx[i][3] - bbx[i][1]
        result_dict["attribute"] = "{}".format(bbx[i][4])
        result_dict["valid"] = True
        # id_ = random.sample(chars, 8)
        # result_dict["id"] = "".join(d for d in id_)
        result_dict["id"] = ids[i]
        result_dict["sourceID"] = ""
        result_dict["textAttribute"] = ""
        result_dict["order"] = i + 1
        result.append(result_dict)

    step_1["result"] = result
    j["step_1"] = step_1

    return j


def print_small_bbx_message(voc_bbx, small_bbx_thresh, txt_src_path):
    bw = voc_bbx[2] - voc_bbx[0]
    bh = voc_bbx[3] - voc_bbx[1]
    if bw <= small_bbx_thresh and bh <= small_bbx_thresh:
        print("\nAttention! Have very small bxx: bw <= {} and bh <= {}! \
                txt_src_path: {}".format(small_bbx_thresh, small_bbx_thresh, txt_src_path))


def yolo_to_labelbee(data_path, save_path="", copy_images=True, small_bbx_thresh=3, cls_plus=1):
    """
    Usually labelbee's class 0 is background, 1 is the first class.
    So yolo -> labelbee: class = int(l[0]) + cls_plus, where cls_plus == 1.
    """
    img_path = data_path + "/images"
    txt_path = data_path + "/labels"

    if save_path is None or save_path == "":
        save_path = make_save_path(data_path, ".", "labelbee_format")
    else:
        os.makedirs(save_path, exist_ok=True)

    img_save_path = save_path + "/images"
    json_save_path = save_path + "/jsons"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(json_save_path, exist_ok=True)

    file_list = sorted(os.listdir(img_path))

    for f in tqdm(file_list):
        file_name = os.path.splitext(f)[0]
        img_src_path = img_path + "/{}".format(f)
        txt_src_path = txt_path + "/{}.txt".format(file_name)
        if not os.path.exists(img_src_path): continue
        if not os.path.exists(txt_src_path): continue

        img = cv2.imread(img_src_path)
        if img is None: continue
        imgsz = img.shape[:2]

        if copy_images:
            img_dst_path = img_save_path + "/{}".format(f)
            shutil.copy(img_src_path, img_dst_path)
        json_dst_path = json_save_path + "/{}.json".format(f)

        bbx_for_json = []
        with open(txt_src_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            for line in lines:
                l = line.strip().split(" ")
                bbx = list(map(float, l[1:]))
                voc_bbx = bbox_yolo_to_voc(imgsz, bbx)
                print_small_bbx_message(voc_bbx, small_bbx_thresh, txt_src_path)
                
                voc_bbx.append(int(l[0]) + cls_plus)
                bbx_for_json.append(voc_bbx)

        with open(json_dst_path, "w", encoding="utf-8") as jw:
            jw.write(json.dumps(write_labelbee_det_json(bbx_for_json, imgsz)))

    print("OK!")


def voc_to_yolo(data_path, save_path="", classes={}, copy_images=True, small_bbx_thresh=3, cls_plus=0):
    import xml.etree.ElementTree as ET

    img_path = data_path + "/images"
    xml_path = data_path + "/xmls"

    if save_path is None or save_path == "":
        save_path = make_save_path(data_path, ".", "yolo_format")
    else:
        os.makedirs(save_path, exist_ok=True)

    img_save_path = save_path + "/images"
    txt_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(txt_save_path, exist_ok=True)

    file_list = sorted(os.listdir(img_path))
    class_names = []
    for f in tqdm(file_list):
        file_name = os.path.splitext(f)[0]
        img_src_path = img_path + "/{}".format(f)
        xml_src_path = xml_path + "/{}.xml".format(file_name)

        if not os.path.exists(xml_src_path): continue

        img = cv2.imread(img_src_path)
        if img is None: continue
        imgsz = img.shape

        if copy_images:
            img_dst_path = img_save_path + "/{}".format(f)
            shutil.copy(img_src_path, img_dst_path)

        txt_dst_path = txt_save_path + "/{}.txt".format(file_name)
        fw = open(txt_dst_path, "w", encoding="utf-8")

        try:
            tree = ET.parse(xml_src_path)
            root = tree.getroot()
            size = root.find('size')
            imgsz = (int(size.find('height').text), int(size.find('width').text))

            class_names_i = []
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in class_names_i:
                    class_names_i.append(cls)
                if classes is not None and classes != {}:
                    if cls not in list(classes.values()):
                        print("{} is not in {}!".format(cls, classes))
                        continue
                    if int(difficult) == 1:
                        print("int(difficult) == 1!")
                        continue

                cls_id = list(classes.values()).index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
                bb = bbox_voc_to_yolo(imgsz, b)
                content = str(int(cls_id) + cls_plus) + " " + " ".join([str(a) for a in bb]) + '\n'
                fw.write(content)
        
            for n in class_names_i:
                if n not in class_names:
                    class_names.append(n)
        except Exception as Error:
            print("Error: {}".format(xml_src_path))

        fw.close()

    print("class_names: {}".format(class_names))
    print("OK!")


def labelbee_to_yolo(data_path, save_path="", copy_images=True, small_bbx_thresh=3, cls_plus=-1):
    """
    Usually labelbee's class 0 is background, 1 is the first class.
    So labelbee -> yolo: cls_id = cls_id + cls_plus, where cls_plus == -1.
    """
    img_path = data_path + "/images"
    json_path = data_path + "/jsons"

    if save_path is None or save_path == "":
        save_path = make_save_path(data_path, ".", "yolo_format")
    else:
        os.makedirs(save_path, exist_ok=True)

    img_save_path = save_path + "/images"
    txt_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(txt_save_path, exist_ok=True)

    json_list = sorted(os.listdir(json_path))
    for j in tqdm(json_list):
        try:
            img_name_ws= os.path.splitext(j)[0]  # img_name_with_suffix
            img_name = os.path.splitext(img_name_ws)[0]

            json_abs_path = json_path + "/{}".format(j)
            json_ = json.load(open(json_abs_path, 'r', encoding='utf-8'))
            if not json_: continue
            imgsz = (json_["height"], json_["width"])

            result = json_["step_1"]["result"]
            if not result: continue

            if copy_images:
                img_src_path = img_path + "/{}".format(img_name_ws)
                img_dst_path = img_save_path + "/{}".format(img_name_ws)
                shutil.copy(img_src_path, img_dst_path)

            len_result = len(result)

            txt_dst_path = txt_save_path + "/{}.txt".format(img_name)
            with open(txt_dst_path, "w", encoding="utf-8") as fw:
                for i in range(len_result):
                    cls_id = int(result[i]["attribute"])

                    x = result[i]["x"]
                    y = result[i]["y"]
                    w = result[i]["width"]
                    h = result[i]["height"]
                    voc_bbx = (x, y, x + w, y + h)

                    print_small_bbx_message(voc_bbx, small_bbx_thresh, txt_dst_path)
                    bb = bbox_voc_to_yolo(imgsz, voc_bbx)
                    txt_content = "{}".format(cls_id + cls_plus) + " " + " ".join([str(b) for b in bb]) + "\n"
                    fw.write(txt_content)

        except Exception as Error:
            print(Error)

    print("OK!")


def coco_to_yolo(data_path, json_name="instances_train2017.json", save_path="", copy_images=False, small_bbx_thresh=3, cls_plus=0):
    """
    json_path = data_path/annotations/json_name
    """
    img_path = data_path + "/images"
    json_path = data_path + '/annotations/{}'.format(json_name)

    if save_path is None or save_path == "":
        save_path = make_save_path(data_path, ".", "yolo_format")
    else:
        os.makedirs(save_path, exist_ok=True)

    img_save_path = save_path + "/images"
    txt_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(txt_save_path, exist_ok=True)

    j = json.load(open(json_path, 'r', encoding="utf-8"))

    # 重新映射并保存class 文件
    id_map = {}  # coco数据集的id不连续!重新映射一下再输出!
    with open(os.path.join(save_path, 'classes.txt'), 'w', encoding="utf-8") as fw:
        for i, category in enumerate(j['categories']):
            fw.write(f"{category['name']}\n")
            id_map[category['id']] = i

    for img in tqdm(j['images']):
        file_name_ws = img["file_name"]
        file_name = os.path.splitext(file_name_ws)[0]
        imgsz = (img["height"], img["width"])
        img_id = img["id"]

        img_src_path = img_path + "/{}".format(file_name_ws)
        if copy_images:
            img_dst_path = img_save_path + "/{}".format(file_name_ws)
            shutil.copy(img_src_path, img_dst_path)

        txt_dst_path = txt_save_path + "/{}.txt".format(file_name)
        txt_fw = open(txt_dst_path, 'w', encoding="utf-8")
        for ann in j['annotations']:
            if ann['image_id'] == img_id:
                ann_np = np.array([ann["bbox"]])
                # ann_np = ann_np[:, [0, 2, 1, 3]]
                ann_list = list(ann_np[0])
                ann_list = [ann_list[0], ann_list[1], ann_list[0] + ann_list[2], ann_list[1] + ann_list[3]]
                bbx_yolo = bbox_voc_to_yolo(imgsz, ann_list)
                content = str(int(id_map[ann["category_id"]]) + cls_plus) + " " + " ".join([str(a) for a in bbx_yolo]) + '\n'
                txt_fw.write(content)

        txt_fw.close()

    print("OK!")


def yolo_to_coco(data_path, save_path="", json_name="instances_val2017_20241121.json", categories=[], copy_images=True, small_bbx_thresh=3, cls_plus=0):
    img_path = data_path + "/images"
    txt_path = data_path + "/labels"

    if save_path is None or save_path == "":
        save_path = make_save_path(data_path, ".", "coco_format")
    else:
        os.makedirs(save_path, exist_ok=True)

    img_save_path = save_path + "/images"
    json_save_path = save_path + "/annotations"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(json_save_path, exist_ok=True)

    file_list = sorted(os.listdir(img_path))

    if json_name is None or json_name == "":
        date = get_date_time().split(" ")[0]
        json_dst_path = json_save_path + "/instances_train2017_{}.json".format(date)
    else:
        json_dst_path = json_save_path + "/{}".format(json_name)

    # json content -----------------------------------------
    info = {
        "year": 2024,
        "version": '1.0',
        "date_created": 2024 - 10 - 16
    }

    licenses = {
        "id": 1,
        "name": "null",
        "url": "null",
    }

    # 自己的标签类别，跟yolo的数据集类别要对应好
    assert isinstance(categories, list), "categories is not list!"
    assert categories is not None and categories != [], "Please input categories!"
    # categories = [
    #     {
    #         "id": 0,
    #         "name": 'ship',
    #         "supercategory": 'sar',
    #     },
    #     {
    #         "id": 1,
    #         "name": 'aircraft',
    #         "supercategory": 'sar',
    #     },
    #     {
    #         "id": 2,
    #         "name": 'car',
    #         "supercategory": 'sar',
    #     },
    # ]

    jdata = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}
    images = []
    annotations = []
    # -----------------------------------------

    with open(json_dst_path, "w", encoding="utf-8") as jw:
        for i, f in tqdm(enumerate(file_list)):
            file_name = os.path.splitext(f)[0]
            img_src_path = img_path + "/{}".format(f)
            txt_src_path = txt_path + "/{}.txt".format(file_name)

            if not os.path.exists(txt_src_path): continue

            img = cv2.imread(img_src_path)
            if img is None: continue
            imgsz = img.shape

            img_info = {}
            img_info['id'] = i
            img_info['file_name'] = f
            img_info['width'] = imgsz[1]
            img_info['height'] = imgsz[0]

            if img_info != {}:
                images.append(img_info)

            if copy_images:
                img_dst_path = img_save_path + "/{}".format(f)
                shutil.copy(img_src_path, img_dst_path)

            with open(txt_src_path, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
                for idx, line in enumerate(lines):
                    annotation_info = {}
                    l = line.strip().split(" ")
                    bbx = list(map(float, l[1:]))
                    voc_bbx = bbox_yolo_to_voc(imgsz, bbx)
                    box_xywh = [voc_bbx[0], voc_bbx[1], voc_bbx[2] - voc_bbx[0], voc_bbx[3] - voc_bbx[1]]

                    print_small_bbx_message(voc_bbx, small_bbx_thresh, txt_src_path)

                    annotation_info["category_id"] = int(l[0])
                    annotation_info['bbox'] = box_xywh
                    annotation_info['area'] = box_xywh[2] * box_xywh[3]
                    annotation_info['image_id'] = i
                    annotation_info['id'] = i * 100 + idx
                    annotation_info['segmentation'] = [[voc_bbx[0], voc_bbx[1], voc_bbx[2], voc_bbx[1], voc_bbx[2], voc_bbx[3], voc_bbx[0], voc_bbx[3]]]  # 四个点的坐标
                    annotation_info['iscrowd'] = 0  # 单例
                    annotations.append(annotation_info)

        jdata['images'] = images
        jdata['annotations'] = annotations
        jw.write(json.dumps(jdata, indent=2))

    print("OK!")


def labelme_to_yolo():
    pass


def yolo_to_labelme():
    pass


class Labelme2YOLO(object):
    
    def __init__(self, json_dir, to_seg=False):
        self._json_dir = json_dir
        
        self._label_id_map = self._get_label_id_map(self._json_dir)
        self._to_seg = to_seg

        # i = 'YOLODataset'
        # i += '_seg/' if to_seg else '/'
        # self._save_path_pfx = os.path.join(self._json_dir, i)
        
        self._save_path_pfx = self._json_dir

    def _make_train_val_dir(self):
        self._label_dir_path = os.path.abspath(os.path.join(self._save_path_pfx, '../labels'))
        self._image_dir_path = os.path.abspath(os.path.join(self._save_path_pfx, '../images'))

        for yolo_path in (os.path.join(self._label_dir_path, 'train'),
                          os.path.join(self._label_dir_path, 'val'),
                          os.path.join(self._image_dir_path, 'train'), 
                          os.path.join(self._image_dir_path, 'val')):
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)
            
            os.makedirs(yolo_path, exist_ok=True)    
                
    def _get_label_id_map(self, json_dir):
        label_set = set()
    
        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data['shapes']:
                    label_set.add(shape['label'])
        
        return OrderedDict([(label, label_id) for label_id, label in enumerate(label_set)])
    
    def _train_test_split(self, folders, json_names, val_size):
        if len(folders) > 0 and 'train' in folders and 'val' in folders:
            train_folder = os.path.join(self._json_dir, 'train/')
            train_json_names = [train_sample_name + '.json' \
                                for train_sample_name in os.listdir(train_folder) \
                                if os.path.isdir(os.path.join(train_folder, train_sample_name))]
            
            val_folder = os.path.join(self._json_dir, 'val/')
            val_json_names = [val_sample_name + '.json' \
                              for val_sample_name in os.listdir(val_folder) \
                              if os.path.isdir(os.path.join(val_folder, val_sample_name))]
            
            return train_json_names, val_json_names
        
        train_idxs, val_idxs = train_test_split(range(len(json_names)), 
                                                test_size=val_size)
        train_json_names = [json_names[train_idx] for train_idx in train_idxs]
        val_json_names = [json_names[val_idx] for val_idx in val_idxs]
        
        return train_json_names, val_json_names
    
    def convert(self, val_size):
        json_names = [file_name for file_name in os.listdir(self._json_dir) \
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and \
                      file_name.endswith('.json')]
        folders =  [file_name for file_name in os.listdir(self._json_dir) \
                    if os.path.isdir(os.path.join(self._json_dir, file_name))]
        train_json_names, val_json_names = self._train_test_split(folders, json_names, val_size)
        
        self._make_train_val_dir()
    
        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        for target_dir, json_names in zip(('train/', 'val/'), (train_json_names, val_json_names)):
            for json_name in json_names:
                json_path = os.path.join(self._json_dir, json_name)
                json_data = json.load(open(json_path))
                
                print('Converting %s for %s ...' % (json_name, target_dir.replace('/', '')))
                
                # img_path = self._save_yolo_image(json_data, 
                #                                  json_name, 
                #                                  self._image_dir_path, 
                #                                  target_dir)

                img_name = json_name.replace('.json', '.jpg')
                img_path = os.path.abspath(os.path.join(self._json_dir, "../images/{}".format(img_name)))
                if not os.path.exists(img_path):
                    img_path = img_path.replace(".jpg", ".JPG")
                    
                yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
                self._save_yolo_label(json_name, 
                                      self._label_dir_path, 
                                      target_dir, 
                                      yolo_obj_list)
        
        print('Generating dataset.yaml file ...')
        self._save_dataset_yaml()
                
    def convert_one(self, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))
        
        print('Converting %s ...' % json_name)
        
        # img_path = self._save_yolo_image(json_data, json_name, 
        #                                  self._json_dir, '')

        img_name = json_name.replace('.json', '.jpg')
        img_path = os.path.join(self._json_dir, img_name)
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".JPG")
        
        yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
        self._save_yolo_label(json_name, self._json_dir, 
                              '', yolo_obj_list)
    
    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []
        
        img_h, img_w, _ = cv2.imread(img_path).shape
        for shape in json_data['shapes']:
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point
            if shape['shape_type'] == 'circle':
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            else:
                yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)
            
            yolo_obj_list.append(yolo_obj)
            
        return yolo_obj_list
    
    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]
        obj_center_x, obj_center_y = shape['points'][0]

        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 +
                           (obj_center_y - shape['points'][1][1]) ** 2)

        if self._to_seg:
            retval = [label_id]

            n_part = radius / 10
            n_part = int(n_part) if n_part > 4 else 4
            n_part2 = n_part << 1

            pt_quad = [None for i in range(0, 4)]
            pt_quad[0] = [[obj_center_x + math.cos(i * math.pi / n_part2) * radius,
                         obj_center_y - math.sin(i * math.pi / n_part2) * radius]
                         for i in range(1, n_part)]
            pt_quad[1] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[0]]
            pt_quad[1].reverse()
            pt_quad[3] = [[x1, obj_center_y * 2 - y1] for x1, y1 in pt_quad[0]]
            pt_quad[3].reverse()
            pt_quad[2] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[3]]
            pt_quad[2].reverse()

            pt_quad[0].append([obj_center_x, obj_center_y - radius])
            pt_quad[1].append([obj_center_x - radius, obj_center_y])
            pt_quad[2].append([obj_center_x, obj_center_y + radius])
            pt_quad[3].append([obj_center_x + radius, obj_center_y])

            for i in pt_quad:
                for j in i:
                    j[0] = round(float(j[0]) / img_w, 6)
                    j[1] = round(float(j[1]) / img_h, 6)
                    retval.extend(j)
            return retval

        obj_w = 2 * radius
        obj_h = 2 * radius
        
        yolo_center_x= round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]

        if self._to_seg:
            retval = [label_id]
            for i in shape['points']:
                i[0] = round(float(i[0]) / img_w, 6)
                i[1] = round(float(i[1]) / img_h, 6)
                retval.extend(i)
            return retval

        def __get_object_desc(obj_port_list):
            __get_dist = lambda int_list: max(int_list) - min(int_list)
            
            x_lists = [port[0] for port in obj_port_list]        
            y_lists = [port[1] for port in obj_port_list]
            
            return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)
        
        obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape['points'])
                    
        yolo_center_x= round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h
    
    def _save_yolo_label(self, json_name, label_dir_path, target_dir, yolo_obj_list):
        txt_path = os.path.join(label_dir_path, 
                                target_dir, 
                                json_name.replace('.json', '.txt'))

        with open(txt_path, 'w+') as f:
            for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
                yolo_obj_line = ""
                for i in yolo_obj:
                    yolo_obj_line += f'{i} '
                yolo_obj_line = yolo_obj_line[:-1]
                if yolo_obj_idx != len(yolo_obj_list) - 1:
                    yolo_obj_line += '\n'
                f.write(yolo_obj_line)

    def _save_yolo_image(self, json_data, json_name, image_dir_path, target_dir):
        img_name = json_name.replace('.json', '.jpg')
        img_path = os.path.join(image_dir_path, img_name)
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".JPG")
        
        # if not os.path.exists(img_path):
        #     img = utils.img_b64_to_arr(json_data['imageData'])
        #     PIL.Image.fromarray(img).save(img_path)
        
        return img_path
    
    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._save_path_pfx, 'dataset.yaml')

        with open(yaml_path, 'w+') as yaml_file:
            yaml_file.write('train: %s\n' % \
                            os.path.join(self._image_dir_path, 'train/'))
            yaml_file.write('val: %s\n\n' % \
                            os.path.join(self._image_dir_path, 'val/'))
            yaml_file.write('nc: %i\n\n' % len(self._label_id_map))
            
            names_str = ''
            for label, _ in self._label_id_map.items():
                names_str += "'%s', " % label
            names_str = names_str.rstrip(', ')
            yaml_file.write('names: [%s]' % names_str)


image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')


def tobase64(file_path):
    with open(file_path, "rb") as image_file:
        data = base64.b64encode(image_file.read())
        return data.decode()


def img_filename_to_ext(img_filename, ext='txt'):
    for img_ext in image_extensions:
        if img_filename.lower().endswith(img_ext):
            return img_filename[:-len(img_ext)] + ext


def is_image_file(file_path):
    file_path = file_path.lower()
    for ext in image_extensions:
        if file_path.endswith(ext):
            return True
    return False


def get_shapes(txt_path, width, height, class_labels):
    shapes = open(txt_path).read().split('\n')
    result = []
    for shape in shapes:
        if not shape:
            continue
        values = shape.split()

        class_id = values[0]
        r_shape = dict()
        r_shape["label"] = class_labels[int(class_id)]

        values = [float(value) for value in values[1:]]
        bbox_voc = bbox_yolo_to_voc((height, width), values)
        points = []
        points.append([bbox_voc[0], bbox_voc[1]])
        points.append([bbox_voc[2], bbox_voc[1]])
        points.append([bbox_voc[2], bbox_voc[3]])
        points.append([bbox_voc[0], bbox_voc[3]])

        # for i in range(len(values)//2):
        #     points.append([values[2*i]*width, values[2*i+1]*height])
        r_shape['points'] = points

        r_shape.update({ "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {}
        })
        result.append(r_shape)
    return result


def yolo2labelme_single(txt_path, img_path, class_labels, out_dir):
    img = Image.open(img_path)
    result = {"version": "5.2.1", "flags": {}}
    result['shapes'] = get_shapes(txt_path, img.width, img.height, class_labels)
    result["imagePath"] = img_path
    result["imageData"] = tobase64(img_path)
    result["imageHeight"] = img.height
    result["imageWidth"] = img.width

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    img_filename = os.path.basename(img_path)
    json_path = img_filename_to_ext(img_filename,'.json')
    json_path = os.path.join(out_dir,json_path)
    with open(json_path,'w') as f:
        f.write(json.dumps(result))
    shutil.copyfile(img_path, os.path.join(out_dir, img_filename) )


def yolo2labelme(data_path, out=None, skip=False):
    # yaml_path = os.path.join(data,"dataset.yaml")
    # with open(yaml_path, 'r') as stream:
    #     data_loaded = yaml.safe_load(stream)
    #     class_labels = data_loaded['names']
    class_labels = {0: "person", 1: "smoking"}

    if out is None:
        out = os.path.join(os.path.abspath(data_path),'..','labelmeDataset')
        os.makedirs(out, exist_ok=True)
    
    # for dir_type in ['test', 'train','val']:
        # dir_path = os.path.join(data, data_loaded[dir_type])
        # dir_path = os.path.abspath(dir_path)

    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    for filename in os.listdir(img_path):
        img_file = os.path.join(img_path, filename)
        base_name = os.path.splitext(filename)[0]
        txt_abs_path = lbl_path + "/{}.txt".format(base_name)
        if is_image_file(img_file):
            # txt_file = img_filename_to_ext(img_file.replace('images','labels'), '.txt')
            if os.path.exists(txt_abs_path):
                yolo2labelme_single(txt_abs_path, img_file, class_labels, out)
            else:
                if skip == False:
                    raise FileNotFoundError(f"{txt_abs_path} is expected to exist."
                                            +"Pass skip=True to skip silently.\n"
                                            +"skip='print' to print missed paths.")
                elif skip == 'print':
                    print(f'Missing {txt_abs_path}')


def write_xml_point(root, node, label1, value1, label2, value2):
    node = node.appendChild(root.createElement('points'))
    node.appendChild(root.createElement(label1)).appendChild(root.createTextNode(value1))
    node.appendChild(root.createElement(label2)).appendChild(root.createTextNode(value2))


def write_xml_node(root, node, label, value):
    node.appendChild(root.createElement(label)).appendChild(root.createTextNode(value))


def yolo_to_voc(data_path, save_path="", classes={}, copy_images=True, small_bbx_thresh=3, cls_plus=0):
    from xml.dom import minidom

    img_path = data_path + "/images"
    txt_path = data_path + "/labels"

    if save_path is None or save_path == "":
        save_path = make_save_path(data_path, ".", "voc_format")
    else:
        os.makedirs(save_path, exist_ok=True)

    img_save_path = save_path + "/images"
    xml_save_path = save_path + "/xmls"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(xml_save_path, exist_ok=True)

    file_list = sorted(os.listdir(img_path))
    for f in tqdm(file_list):
        file_name = os.path.splitext(f)[0]
        img_src_path = img_path + "/{}".format(f)
        txt_src_path = txt_path + "/{}.txt".format(file_name)

        if not os.path.exists(txt_src_path): continue

        img = cv2.imread(img_src_path)
        if img is None: continue
        imgsz = img.shape

        if copy_images:
            img_dst_path = img_save_path + "/{}".format(f)
            shutil.copy(img_src_path, img_dst_path)

        bbxs = []
        with open(txt_src_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            for line in lines:
                l = line.strip().split(" ")
                bbx = list(map(float, l[1:]))
                voc_bbx = bbox_yolo_to_voc(imgsz, bbx)

                print_small_bbx_message(voc_bbx, small_bbx_thresh, txt_src_path)

                voc_bbx.append(int(l[0]) + cls_plus)
                bbxs.append(voc_bbx)

        xml_abs_path = xml_save_path + "/{}.xml".format(file_name)

        root = minidom.Document()
        annotation = root.createElement('annotation')
        root.appendChild(annotation)

        write_xml_node(root, annotation, 'filename', img_src_path)

        size = root.createElement('size')
        annotation.appendChild(size)
        write_xml_node(root, size, 'width', str(imgsz[1]))
        write_xml_node(root, size, 'height', str(imgsz[0]))
        write_xml_node(root, size, 'depth', str(imgsz[2]))

        for i in range(len(bbxs)):
            x_min = str(bbxs[i][0])
            y_min = str(bbxs[i][1])
            x_max = str(bbxs[i][2])
            y_max = str(bbxs[i][3])
            label = str(bbxs[i][4])

            object_ = root.createElement('object')
            annotation.appendChild(object_)
            if classes is not None and classes != {}:
                write_xml_node(root, object_, 'name', classes[label])
            else:
                write_xml_node(root, object_, 'name', label)
            write_xml_node(root, object_, 'difficult', '0')
            write_xml_node(root, object_, 'truncated', '0')

            bndbox = root.createElement('bndbox')
            object_.appendChild(bndbox)
            write_xml_node(root, bndbox, 'xmin', x_min)
            write_xml_node(root, bndbox, 'ymin', y_min)
            write_xml_node(root, bndbox, 'xmax', x_max)
            write_xml_node(root, bndbox, 'ymax', y_max)

            segmentation = root.createElement('segmentation')
            object_.appendChild(segmentation)
            write_xml_point(root, segmentation, 'x', x_min, 'y', y_min)
            write_xml_point(root, segmentation, 'x', x_max, 'y', y_min)
            write_xml_point(root, segmentation, 'x', x_max, 'y', y_max)
            write_xml_point(root, segmentation, 'x', x_min, 'y', y_max)

        with open(xml_abs_path, 'w', encoding='UTF-8') as fw:
            root.writexml(fw, indent='', addindent='\t', newl='\n', encoding='UTF-8')

    print("OK!")


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

        xy_ = []
        for i in range(len(pl_)):
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


def labelbee_seg_json_to_yolo_txt(data_path, keypoint_flag=False, cls_plus=-1):
    dir_name = get_dir_name(data_path)
    img_path = data_path + "/{}".format("images")
    json_path = data_path + "/{}".format("jsons")

    save_path = os.path.abspath(os.path.join(data_path, "../")) + "/{}_{}".format(dir_name, "yolo_labels")
    img_save_path = save_path + "/{}".format("images")
    lbl_save_path = save_path + "/{}".format("labels")
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    json_list = sorted(os.listdir(json_path))

    for j in json_list:
        img_abs_path = img_path + "/{}".format(j.strip(".json"))
        img_dst_path = img_save_path + "/{}".format(j.strip(".json"))

        json_abs_path = json_path + "/{}".format(j)
        json_ = json.load(open(json_abs_path, "r", encoding="utf-8"))
        w, h = json_["width"], json_["height"]

        len_object = len(json_["step_1"]["result"])
        if len_object == 0: continue
        shutil.copy(img_abs_path, img_dst_path)

        txt_save_path = lbl_save_path + "/{}".format(j.replace(".jpg.json", ".txt"))
        with open(txt_save_path, "w", encoding="utf-8") as fw:
            
            pl = []
            for i in range(len_object):
                pl_ = json_["step_1"]["result"][i]["pointList"]
                cls_ = int(json_["step_1"]["result"][i]["attribute"])

                x_, y_ = [], []
                xy_ = []  # x, y, x, y. x. y, x, y
                for i in range(len(pl_)):
                    # x_.append(float(pl_[i]["x"]))
                    # y_.append(float(pl_[i]["y"]))

                    # xy_.append(float(pl_[i]["x"]))
                    # xy_.append(float(pl_[i]["y"]))

                    dw = 1. / w
                    dh = 1. / h
                    xy_.append(float(pl_[i]["x"] * dw))
                    xy_.append(float(pl_[i]["y"] * dh))

                txt_content = "{}".format(cls_ + cls_plus) + " " + " ".join([str(a) for a in xy_]) + "\n"
                fw.write(txt_content)


                # # yolov5 keypoint format
                # if keypoint_flag:
                #     if len(xy_) == 8 and len(x_) == 4 and len(y_) == 4:
                #         x_min, x_max = min(x_), max(x_)
                #         y_min, y_max = min(y_), max(y_)

                #         bb = bbox_voc_to_yolo((h, w), (x_min, y_min, x_max, y_max))
                #         p_res = convert_points((w, h), xy_)

                #         txt_content = "{}".format(cls_ + cls_plus) + " " + " ".join([str(a) for a in bb]) + " " + " ".join([str(c) for c in p_res]) + "\n"
                #         fw.write(txt_content)
                # else:
                #     x_min, x_max = min(x_), max(x_)
                #     y_min, y_max = min(y_), max(y_)

                #     bb = bbox_voc_to_yolo((h, w), (x_min, y_min, x_max, y_max))
                #     txt_content = "{}".format(cls_ + cls_plus) + " " + " ".join([str(a) for a in bb]) + "\n"
                #     fw.write(txt_content)

            print("Saved --> {}".format(txt_save_path))


def labelme_to_voc(data_path):
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


def labelme_det_kpt_to_yolo_labels(data_path, class_list, keypoint_list):
    img_path = data_path + "/images"
    json_path = data_path + "/jsons"

    save_path = make_save_path(data_path=data_path, relative=".", add_str="yolo_format")
    img_save_path = save_path + "/images"
    lbl_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)
    
    file_list = get_file_list(img_path)

    for f in file_list:
        fname = os.path.splitext(f)[0]
        img_src_path = img_path + "/{}".format(f)
        json_src_path = json_path + "/{}.json".format(fname)
            
        img = cv2.imread(img_src_path)

        if not os.path.exists(json_src_path): continue

        with open(json_src_path) as json_file:
            json_data = json.load(json_file)

        h,w = img.shape[:2]
        # 步骤：
        # 1. 找出所有的矩形，记录下矩形的坐标，以及对应group_id
        # 2. 遍历所有的head和tail，记下点的坐标，以及对应group_id，加入到对应的矩形中
        # 3. 转为yolo格式

        # rectangles = {}
        rectangles = []
        # 遍历初始化
        for shape in json_data["shapes"]:
            label = shape["label"] # pen, head, tail
            group_id = shape["group_id"] # 0, 1, 2, ...
            points = shape["points"] # x,y coordinates
            shape_type = shape["shape_type"]

            # 只处理矩形
            if shape_type == "rectangle" and label == "torn":
                # if group_id not in rectangles:
                #     rectangles[group_id] = {
                #         "label": label,
                #         "rect": points[0] + points[1],  # Rectangle [x1, y1, x2, y2]
                #         "keypoints_list": []
                #     }

                rectangles.append(points[0] + points[1])

        # 遍历更新，将点加入对应group_id的矩形中
        # for keypoint in keypoint_list:
        #     for shape in json_data["shapes"]:
        #         label = shape["label"]
        #         group_id = shape["group_id"]
        #         points = shape["points"]
        #         # 如果匹配到了对应的keypoint
        #         if label == keypoint:
        #             rectangles[group_id]["keypoints_list"].append(points[0])

        # for shape in json_data["shapes"]:
        #     label = shape["label"]
        #     group_id = shape["group_id"]
        #     points = shape["points"]
        #     # 如果匹配到了对应的keypoint
        #     if label in keypoint_list:
        #         rectangles[group_id]["keypoints_list"].append(points[0])

        keypoints = []
        for shape in json_data["shapes"]:
            label = shape["label"]
            group_id = shape["group_id"]
            points = shape["points"]
            # 如果匹配到了对应的keypoint
            if label in keypoint_list:
                keypoints.append(points[0])

        
        # 转为yolo格式
        img_dst_path = img_save_path + "/{}".format(f)
        shutil.copy(img_src_path, img_dst_path)

        yolo_dst_path = lbl_save_path + "/{}.txt".format(fname)
        with open(yolo_dst_path, "w") as f:

            yolo_list = []
            for rectangle in rectangles:
                result_list  = []
                # label_id = class_list.index(rectangle["label"])
                # x1,y1,x2,y2
                # x1,y1,x2,y2 = rectangle["rect"]
                x1,y1,x2,y2 = rectangle
                # center_x, center_y, width, height
                center_x = (x1+x2)/2
                center_y = (y1+y2)/2
                width = abs(x1-x2)
                height = abs(y1-y2)
                # normalize
                center_x /= w
                center_y /= h
                width /= w
                height /= h

                # 保留6位小数
                center_x = round(center_x, 6)
                center_y = round(center_y, 6)
                width = round(width, 6)
                height = round(height, 6)


                # 添加 label_id, center_x, center_y, width, height
                label_id = 0
                result_list = [label_id, center_x, center_y, width, height]
            
                # 添加 p1_x, p1_y, p1_v, p2_x, p2_y, p2_v
                # for point in rectangle["keypoints_list"]:

                points_bbox = []
                for point in keypoints:
                    x,y = point
                    x,y = int(x), int(y)

                    if x > x1 and x < x2 and y > y1 and y < y2:
                        # normalize
                        x /= w
                        y /= h
                        # 保留6位小数
                        x = round(x, 6)
                        y = round(y, 6)
                        
                        # result_list.extend([x,y,2])
                        points_bbox.append([x,y,2])

                assert len(points_bbox) == 2, "len(points_bbox) != 2"
                if points_bbox[0][0] > points_bbox[1][0]:
                    points_bbox = points_bbox[::-1]
                    print("points_bbox = points_bbox[::-1]")

                result_list.extend(points_bbox[0])
                result_list.extend(points_bbox[1])

                yolo_list.append(result_list)
                
            
            for yolo in yolo_list:
                # for i in range(len(yolo)):
                #     if i == 0:
                #         f.write(str(yolo[i]))
                #     else:
                #         f.write(" " + str(yolo[i]))
                # f.write("\n")

                content = " ".join([str(i) for i in yolo]) + "\n"
                f.write(content)


def labelbee_multi_step_det_kpt_to_yolo_labels(data_path, save_path="", copy_images=True, small_bbx_thresh=3, cls_plus=-1):
    """
    Usually labelbee's class 0 is background, 1 is the first class.
    So labelbee -> yolo: cls_id = cls_id + cls_plus, where cls_plus == -1.
    """
    img_path = data_path + "/images"
    json_path = data_path + "/jsons"

    if save_path is None or save_path == "":
        save_path = make_save_path(data_path, ".", "yolo_format")
    else:
        os.makedirs(save_path, exist_ok=True)

    img_save_path = save_path + "/images"
    txt_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(txt_save_path, exist_ok=True)

    json_list = sorted(os.listdir(json_path))
    for j in tqdm(json_list):
        try:
            img_name_ws= os.path.splitext(j)[0]  # img_name_with_suffix
            img_name = os.path.splitext(img_name_ws)[0]

            json_abs_path = json_path + "/{}".format(j)
            json_ = json.load(open(json_abs_path, 'r', encoding='utf-8'))
            if not json_: continue
            imgsz = (json_["height"], json_["width"])

            step_1_result = json_["step_1"]["result"]
            step_2_result = json_["step_2"]["result"]
            if not step_1_result: continue
            if not step_2_result: continue

            if copy_images:
                img_src_path = img_path + "/{}".format(img_name_ws)
                img_dst_path = img_save_path + "/{}".format(img_name_ws)
                shutil.copy(img_src_path, img_dst_path)

            len_s1_result = len(step_1_result)
            len_s2_result = len(step_2_result)
            assert len_s1_result * 2 == len_s2_result, "len_s1_result * 2 != len_s2_result"

            txt_dst_path = txt_save_path + "/{}.txt".format(img_name)
            with open(txt_dst_path, "w", encoding="utf-8") as fw:
                for i in range(len_s1_result):
                    cls_id = int(step_1_result[i]["attribute"])
                    rect_curr_id = str(step_1_result[i]["id"])

                    x = step_1_result[i]["x"]
                    y = step_1_result[i]["y"]
                    w = step_1_result[i]["width"]
                    h = step_1_result[i]["height"]
                    voc_bbx = (x, y, x + w, y + h)

                    print_small_bbx_message(voc_bbx, small_bbx_thresh, txt_dst_path)
                    bb = bbox_voc_to_yolo(imgsz, voc_bbx)
                    # txt_content = "{}".format(cls_id + cls_plus) + " " + " ".join([str(b) for b in bb]) + "\n"
                    txt_content = "{}".format(cls_id + cls_plus) + " " + " ".join([str(b) for b in bb])

                    point_content_list = []
                    for j in range(len_s2_result):
                        point_curr_id = step_2_result[j]["id"]
                        point_src_id = str(step_2_result[j]["sourceID"])
                        order = step_2_result[j]["order"]

                        if point_src_id == rect_curr_id:
                            px = step_2_result[j]["x"]
                            py = step_2_result[j]["y"]
                            px /= imgsz[1]
                            py /= imgsz[0]
                            point_content_list.append([px, py, order])

                    point_content_list = sorted(point_content_list, key=lambda x: x[2])
                    point_content = ""
                    for p in point_content_list:
                        point_content += " " + " ".join([str(p[0]), str(p[1]), str(2)])

                    txt_content_new = txt_content + point_content + "\n"
                    fw.write(txt_content_new)

        except Exception as Error:
            print(Error)

    print("OK!")


def write_labelbee_det_kpt_json(bbxs, kpts, imgsz):
    """
    {"x":316.6583427922815,"y":554.4245175936436,"width":1419.1872871736662,"height":556.1679909194097,
    "attribute":"1","valid":true,"id":"tNd2HY6C","sourceID":"","textAttribute":"","order":1}
    :param bbx: x1, y1, x2, y2
    :param imgsz: H, W
    :return:
    """
    assert len(bbxs) == len(kpts), "len(bbxs) != len(kpts)"

    chars = ""
    for i in range(48, 48 + 9):
        chars += chr(i)
    for j in range(65, 65 + 25):
        chars += chr(j)
    for k in range(97, 97 + 25):
        chars += chr(k)

    j = {}
    j["width"] = imgsz[1]
    j["height"] = imgsz[0]
    j["valid"] = True
    j["rotate"] = 0

    # --------------------------------
    step_1 = {}
    step_1["dataSourceStep"] = 0
    step_1["toolName"] = "rectTool"

    ids = []
    while (len(ids) < len(bbxs)):
        id_ = random.sample(chars, 8)
        id_ = "".join([d for d in id_])
        if id_ not in ids:
            ids.append(id_)
    assert len(ids) == len(bbxs), "len(ids) != len(bbxs)"

    result = []
    for i in range(len(bbxs)):
        result_dict = {}
        result_dict["x"] = bbxs[i][0]
        result_dict["y"] = bbxs[i][1]
        result_dict["width"] = bbxs[i][2] - bbxs[i][0]
        result_dict["height"] = bbxs[i][3] - bbxs[i][1]
        result_dict["attribute"] = "{}".format(bbxs[i][4])
        result_dict["valid"] = True
        result_dict["id"] = ids[i]
        result_dict["sourceID"] = ""
        result_dict["textAttribute"] = ""
        result_dict["order"] = i + 1
        result.append(result_dict)

    step_1["result"] = result
    j["step_1"] = step_1

    # --------------------------------
    step_2 = {}
    step_2["dataSourceStep"] = 1
    step_2["toolName"] = "pointTool"

    point_ids = []
    while (len(point_ids) < len(kpts)):
        pid_ = random.sample(chars, 8)
        pid_ = "".join([d for d in pid_])
        if pid_ not in ids and pid_ not in point_ids:
            point_ids.append(pid_)
    assert len(point_ids) == len(kpts), "len(point_ids) != len(kpts)"
    
    point_result = []
    for i in range(len(kpts)):
        for n in range(2):
            point_result_dict = {}
            if n == 0:
                point_result_dict["x"] = kpts[i][0]
                point_result_dict["y"] = kpts[i][1]
            else:
                point_result_dict["x"] = kpts[i][2]
                point_result_dict["y"] = kpts[i][3]
            if n == 0:
                point_result_dict["attribute"] = "1"
            else:
                point_result_dict["attribute"] = "2"
            point_result_dict["valid"] = True
            point_result_dict["id"] = point_ids[i]
            point_result_dict["sourceID"] = ids[i]
            point_result_dict["textAttribute"] = ""
            if n == 0:
                point_result_dict["order"] = 1
            else:
                point_result_dict["order"] = 2
            point_result.append(point_result_dict)

    step_2["result"] = point_result
    j["step_2"] = step_2

    return j


def det_kpt_yolo_labels_to_labelbee_multi_step_json(data_path, save_path="", copy_images=True, small_bbx_thresh=3, cls_plus=1, return_decimal=True):
    """
    Usually labelbee's class 0 is background, 1 is the first class.
    So yolo -> labelbee: class = int(l[0]) + cls_plus, where cls_plus == 1.
    """
    img_path = data_path + "/images"
    txt_path = data_path + "/labels"

    if save_path is None or save_path == "":
        save_path = make_save_path(data_path, ".", "labelbee_format")
    else:
        os.makedirs(save_path, exist_ok=True)

    img_save_path = save_path + "/images"
    json_save_path = save_path + "/jsons"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(json_save_path, exist_ok=True)

    file_list = sorted(os.listdir(img_path))

    for f in tqdm(file_list):
        file_name = os.path.splitext(f)[0]
        img_src_path = img_path + "/{}".format(f)
        txt_src_path = txt_path + "/{}.txt".format(file_name)
        if not os.path.exists(img_src_path): continue
        if not os.path.exists(txt_src_path): continue

        img = cv2.imread(img_src_path)
        if img is None: continue
        imgsz = img.shape[:2]

        if copy_images:
            img_dst_path = img_save_path + "/{}".format(f)
            shutil.copy(img_src_path, img_dst_path)
        json_dst_path = json_save_path + "/{}.json".format(f)

        step_one_bbx = []
        step_two_kpt = []
        with open(txt_src_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            for line in lines:
                l = line.strip().split(" ")
                bbx = list(map(float, l[1:5]))
                voc_bbx = bbox_yolo_to_voc(imgsz, bbx, return_decimal=return_decimal)
                print_small_bbx_message(voc_bbx, small_bbx_thresh, txt_src_path)
                
                voc_bbx.append(int(l[0]) + cls_plus)
                step_one_bbx.append(voc_bbx)

                yolo_pts = list(map(float, l[5:7])) + list(map(float, l[8:10]))
                voc_pts = [yolo_pts[0] * imgsz[1], yolo_pts[1] * imgsz[0], yolo_pts[2] * imgsz[1], yolo_pts[3] * imgsz[0]]
                step_two_kpt.append(voc_pts)

        with open(json_dst_path, "w", encoding="utf-8") as jw:
            jw.write(json.dumps(write_labelbee_det_kpt_json(step_one_bbx, step_two_kpt, imgsz)))

    print("OK!")







def vis_yolo_labels(data_path, print_flag=True, color_num=1000, rm_small_object=False, rm_size=32):
    colors = []
    for i in range(color_num * 2):
        c = list(np.random.choice(range(256), size=3))
        if c not in colors:
            colors.append(c)

    colors = colors[:color_num]

    img_path = data_path + "/images"
    txt_path = data_path + "/labels"
    vis_path = data_path + "/vis_bbx"
    os.makedirs(vis_path, exist_ok=True)

    img_list = os.listdir(img_path)

    for f in tqdm(img_list):
        try:
            img_name = os.path.splitext(f)[0]
            img_abs_path = img_path + "/{}".format(f)
            txt_abs_path = txt_path + "/{}.txt".format(img_name)
            img = cv2.imread(img_abs_path)
            h, w = img.shape[:2]

            with open(txt_abs_path, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
                for l_orig in lines:
                    l = l_orig.strip()
                    cls = int(l.split(" ")[0])
                    l_ = [float(l.split(" ")[1]), float(l.split(" ")[2]), float(l.split(" ")[3]), float(l.split(" ")[4])]
                    bbx_VOC_format = bbox_yolo_to_voc((h, w), l_)

                    cv2.rectangle(img, (bbx_VOC_format[0], bbx_VOC_format[1]), (bbx_VOC_format[2], bbx_VOC_format[3]), (int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])), 2)
                    cv2.putText(img, "{}".format(cls), (bbx_VOC_format[0], bbx_VOC_format[1] - 4), cv2.FONT_HERSHEY_PLAIN, 2, (int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])))

                    cv2.imwrite("{}/{}".format(vis_path, f), img)
                    if print_flag:
                        print("--> {}/{}".format(vis_path, f))

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
                try:
                    cls = int(l.strip().split(" ")[0])
                except Exception as Error:
                    print(Error)
                    print(l)
                if cls not in labels:
                    labels.append(cls)

    print("\n{}:".format(label_path))
    print("Len: {}, Labels: {}".format(len(labels), sorted(labels)))


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


def extract_gif_frames(gif_path):
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


def extract_video_frames(video_path, gap=5):
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


def cal_mean_std_var_2(data_path, size=(64, 64), step=1):
    dir_list = os.listdir(data_path)
    
    means = [0 for i in range(3)]
    stds = [0 for i in range(3)]
    cnt = 0
    for idx in tqdm(range(0, len(dir_list), step)):
        cnt+=1
        filename = dir_list[idx]
        img = cv2.imread(os.path.join(data_path, filename)) 
        img = img /255.0
        b, g, r = cv2.split(img)
        means[0] += np.mean(r)
        means[1] += np.mean(g)
        means[2] += np.mean(b)
    means = np.array(means) / cnt

    # std要另外算，计算减去的均值是所有图片的均值，而不是某张图片的均值。
    for idx in tqdm(range(0, len(dir_list), step)):
        filename = dir_list[idx]
        img = cv2.imread(os.path.join(data_path, filename)) 
        img = img /255.0
        b, g, r = cv2.split(img)
        stds[0] += np.mean((r - means[0]) ** 2)
        stds[1] += np.mean((g - means[1]) ** 2)
        stds[2] += np.mean((b - means[2]) ** 2)
    stds = np.sqrt(np.array(stds) / cnt)

    print("RGB MEAN:",means,"RBG STD:",stds) 


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
            raise NotImplementedError


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
                print(img_abs_path)
                raise NotImplementedError
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


def change_black_area_pixel(img_path):
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


def perspective_transform(img, rect):
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
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped


def crop_img_via_perspective_transform(img):
    """
    标注4个点并通过透视变换裁剪出这个区域
    输入可以是图片路径或np.ndarray或PIL.Image
    """
    
    def click_event(event, x, y, flags, param):
        xy = []
        if event == cv2.EVENT_LBUTTONDOWN:
            xy.append((x, y))
            cv2.circle(img, (x, y), 1, (255, 0, 255), -1)
            cv2.putText(img, "({}, {})".format(x, y), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
            cv2.imshow("img", img)

    if isinstance(img, str) and os.path.exists(img):
        img = cv2.imread(img)
    elif isinstance(img, PIL.Image.Image):
        img = pil2cv(img)
    else:
        assert isinstance(img, np.ndarray)
    
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
    
    return warped


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

        except Exception as Error:
            if flag == "mv":
                shutil.move(img_abs_path, save_path)
            elif flag == "rm":
                os.remove(img_abs_path)
            print(Error)


def process_small_images(img_path, size=48, mode=0):
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
                if (h < size and w < size) or (h > 8 * w or w > 5 * h):
                    shutil.move(img_abs_path, img_dst_path)
            elif mode == 1:
                if h < size or w < size:
                    shutil.move(img_abs_path, img_dst_path)
            else:
                if (h < size or w < size) or (h > 3 * w or w > 5 * h):
                    shutil.move(img_abs_path, img_dst_path)

        except Exception as Error:
            print(Error)


def process_corrupt_images(img_path, algorithm="pil", flag="delete"):
    assert algorithm == "pil" or algorithm == "imghdr" or algorithm == "cv2", "algorithm: pil, imghdr, cv2"
    assert flag == "delete" or flag == "del" or flag == "move" or flag == "mv", "flag: delete, del, move, mv"

    file_list = sorted(os.listdir(img_path))

    if flag == "move" or flag == "mv":
        save_path = make_save_path(img_path, relative=".", add_str="corrupt_images")
        os.makedirs(save_path, exist_ok=True)

    for f in file_list:
        suffix = os.path.splitext(f)[1][1:]
        img_abs_path = img_path + "/{}".format(f)
        img_dst_path = save_path + "/{}".format(f)

        try:
            if algorithm == "pil":
                img = Image.open(img_abs_path)
                img.load().verify()
                img = np.asarray(img)
            elif algorithm == "imghdr":
                is_corrupt = True
                res = imghdr.what(img_abs_path)
                if suffix.lower()[:2] == res.lower()[:2]:
                    is_corrupt = False

                if is_corrupt:
                    if flag == "move" or flag == "mv":
                        shutil.move(img_abs_path, img_dst_path)
                        print("shutil.move: {} --> {}".format(img_abs_path, img_dst_path))
                    else:
                        os.remove(img_abs_path)
                        print("os.remove: {}".format(img_abs_path))
            else:
                res = cv2.imread(img_abs_path)

        except Exception as Error:
            print(Error)

            if flag == "move" or flag == "mv":
                shutil.move(img_abs_path, img_dst_path)
                print("shutil.move: {} --> {}".format(img_abs_path, img_dst_path))
            else:
                os.remove(img_abs_path)
                print("os.remove: {}".format(img_abs_path))


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


def apply_hog(img):
    from skimage import feature, exposure

    fd, hog_img = feature.hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    return hog_img_rescaled


def min_filter_gray(src, r=7):
    '''最小值滤波,r是滤波器半径'''
    # 使用opencv的erode函数更高效

    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guided_filter(I, p, r, eps):
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


def get_v1(m, r, eps, w, maxV1):
    # 输入rgb图像,值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)  # 得到暗通道图像
    V1 = guided_filter(V1, min_filter_gray(V1, 7), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

    return V1, A


def dehaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    V1, A = get_v1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return Y


def cal_saliency_map(img_path, algorithm="FT"):
    if algorithm == "FT":
        img = cv2.imread(img_path)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # gaussian_blur = cv2.GaussianBlur(src, (17, 17), 0)
        blur = cv2.medianBlur(img, 7)

        mean_lab = np.mean(lab, axis=(0, 1))
        saliency_map = (blur - mean_lab) * (blur - mean_lab)
        saliency_map = (saliency_map - np.amin(saliency_map)) / (np.amax(saliency_map) - np.amin(saliency_map))

        return saliency_map
    elif algorithm == "FT2":
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


def thresh_img(img, threshold_min_thr=10, adaptiveThreshold=True):
    if adaptiveThreshold:
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
        return None, thresh
    else:
        ret, thresh = cv2.threshold(img, threshold_min_thr, 255, cv2.THRESH_BINARY)
        return ret, thresh


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


def get_red(img):
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
    src = get_red(img)
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
    outimg = np.zeros(img.shape)
    # roi = np.where(maskimg[:, :, 0] != 0 & maskimg[:, :, 1] != 0 & maskimg[:, :, 2] != 0)
    roi = np.where(maskimg[:, :, 0] != 0)
    outimg[roi] = img[roi]

    conts, hierarchy = cv2.findContours(maskimg[:, :, 0].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxc = max(conts, key=cv2.contourArea)
    bbox = cv2.boundingRect(maxc)
    outimg_crop = outimg[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
    relative_roi = (roi[0] - bbox[1], roi[1] - bbox[0])

    return outimg_crop, bbox, relative_roi


def crop_image_to_create_rolling_numbers(data_path):
    """
    OCR
    裁剪图片的目的是模拟例如电表中滚动的数字
    """
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


def create_rolling_numbers(data_path):
    """
    OCR
    模拟例如电表中滚动的数字
    """
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

            stack = np.vstack((imgi_j_u, imgi1_j_d))
            if j <= 5:
                cv2.imwrite("{}/{}_{}_stack={}.png".format(save_path, i, 10 - j, i), stack)
            else:
                if i == 9:
                    cv2.imwrite("{}/{}_{}_stack={}.png".format(save_path, i, 10 - j, 0), stack)
                else:
                    cv2.imwrite("{}/{}_{}_stack={}.png".format(save_path, i, 10 - j, i + 1), stack)


def get_color(specific_color=True):
    """
    specific_color: type -> bool, tuple, list
    if tuple or list: specific_color = ((c1), (c2), ...)
    """
    # 使用这个传入copyMakeBorder的value参数会报错，不知道为啥，结果是<class 'tuple'>
    # cv2.error: OpenCV(4.8.0) :-1: error: (-5:Bad argument) in function 'copyMakeBorder'
    # > Overload resolution failed:
    # >  - Scalar value for argument 'value' is not numeric
    # >  - Scalar value for argument 'value' is not numeric
    # color0 = tuple(np.random.randint(0, 256, 3))

    color0 = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

    if isinstance(specific_color, bool):
        color1 = (0, 0, 0)
        color2 = (114, 114, 114)
        color3 = (255, 255, 255)
        colors = [color0, color1, color2, color3]

        if specific_color:
            color = random.sample(colors, 1)[0]
            return color
        return color0
    elif isinstance(specific_color, tuple):
        colors = []
        for c in specific_color:
            colors.append(c)
        color = random.sample(colors, 1)[0]
        return color
    elif isinstance(specific_color, list):
        colors = []
        for c in specific_color:
            colors.append(tuple(c))
        color = random.sample(colors, 1)[0]
        return color
    else:
        print("specific_color should be bool or tuple or list!")
        raise ValueError


def make_border_base(im, new_shape=(64, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, specific_color=True):
    """
    :param im:
    :param new_shape: (H, W)
    :param r1:
    :param r2:
    :param sliding_window:
    :return:
    """
    assert base_side in [-1, 0, 1, "H", "h", "Height", "height", "W", "w", "Width", "width"], "arg -> base_side error!"
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape * 4)

    color = get_color(specific_color=specific_color)
    shape = im.shape[:2]  # current shape [height, width]

    # Compute padding
    if ppocr_format:
        ratio = shape[1] / shape[0]
        if math.ceil(new_shape[0] * ratio) >= new_shape[1]:
            unpad_size = new_shape
        else:
            unpad_size = (new_shape[0], int(math.ceil(new_shape[0] * ratio)))
    else:
        if base_side == 0 or base_side == "H" or base_side == "h" or base_side == "Height" or base_side == "height":
            r = new_shape[0] / shape[0]
        elif base_side == 1 or base_side == "W" or base_side == "w" or base_side == "Width" or base_side == "width":
            r = new_shape[1] / shape[1]
        else:
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        unpad_size = (int(round(shape[0] * r)), int(round(shape[1] * r)))
    
    if shape != unpad_size:
        im = cv2.resize(im, unpad_size[::-1], interpolation=cv2.INTER_LINEAR)

    ph, pw = new_shape[0] - unpad_size[0], new_shape[1] - unpad_size[1]  # wh padding
    if random:
        rdmh = np.random.random()
        rmdw = np.random.random()
        top = int(round(ph * rdmh))
        bottom = ph - top
        left = int(round(pw * rmdw))
        right = pw - left
    else:
        top = ph // 2
        bottom = ph - top
        left = 0
        right = pw - left
        
    if base_side == -1:
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        if im.shape[1] <= new_shape[1]:
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        else:
            im = cv2.resize(im, new_shape[::-1])

    return im


def sliding_window_crop(img, cropsz=(64, 256), random=True, base_side=-1, ppocr_format=False, gap=(0, 128), r1=0, r2=0.25, specific_color=True, make_border=True):
    cropped_imgs = []
    imgsz = img.shape[:2]

    if gap[0] == 0 and gap[1] > 0:
        cropsz = (imgsz[0], cropsz[1])
        for i in range(0, imgsz[1], gap[1]):
            if i + cropsz[1] > imgsz[1]:
                cp_img = img[0:imgsz[0], i:imgsz[1]]
                if make_border:
                    cp_img = make_border_base(cp_img, new_shape=cropsz, random=random, base_side=base_side, ppocr_format=ppocr_format, r1=r1, r2=r2, specific_color=specific_color)
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
                if make_border:
                    cp_img = make_border_base(cp_img, new_shape=cropsz, random=random, base_side=base_side, ppocr_format=ppocr_format, r1=r1, r2=r2, specific_color=specific_color)
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
                        if make_border:
                            cp_img = make_border_base(cp_img, new_shape=cropsz, random=random, base_side=base_side, ppocr_format=ppocr_format, r1=r1, r2=r2, specific_color=specific_color)
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
                        if make_border:
                            cp_img = make_border_base(cp_img, new_shape=cropsz, random=random, base_side=base_side, ppocr_format=ppocr_format, r1=r1, r2=r2, specific_color=specific_color)
                        cropped_imgs.append(cp_img)
                        break
                    else:
                        cp_img = img[j:j + cropsz[0], i:i + cropsz[1]]
                        cropped_imgs.append(cp_img)

    return cropped_imgs


def make_border_v7(im, new_shape=(64, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True):
    """
    :param im:
    :param new_shape: (H, W)
    :param base_side: [-1, 0, 1, "H", "h", "Height", "height", "W", "w", "Width", "width"]
    :param r1:
    :param r2:
    :param sliding_window:
    :return:
    """
    assert base_side in [-1, 0, 1, "H", "h", "Height", "height", "W", "w", "Width", "width"], "arg -> base_side error!"
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape * 4)

    color = get_color(specific_color=specific_color)
    shape = im.shape[:2]  # current shape [height, width]

    # Compute padding
    if ppocr_format:
        ratio = shape[1] / shape[0]
        if math.ceil(new_shape[0] * ratio) >= new_shape[1]:
            unpad_size = new_shape
        else:
            unpad_size = (new_shape[0], int(math.ceil(new_shape[0] * ratio)))
    else:
        if base_side == 0 or base_side == "H" or base_side == "h" or base_side == "Height" or base_side == "height":
            r = new_shape[0] / shape[0]
        elif base_side == 1 or base_side == "W" or base_side == "w" or base_side == "Width" or base_side == "width":
            r = new_shape[1] / shape[1]
        else:
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        unpad_size = (int(round(shape[0] * r)), int(round(shape[1] * r)))
    
    if shape != unpad_size:
        im = cv2.resize(im, unpad_size[::-1], interpolation=cv2.INTER_LINEAR)

    ph, pw = new_shape[0] - unpad_size[0], new_shape[1] - unpad_size[1]  # wh padding
    if random:
        rdmh = np.random.random()
        rmdw = np.random.random()
        top = int(round(ph * rdmh))
        bottom = ph - top
        left = int(round(pw * rmdw))
        right = pw - left
    else:
        top = ph // 2
        bottom = ph - top
        left = 0
        right = pw - left
        
    if base_side == -1:
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        if im.shape[1] <= new_shape[1]:
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        elif (im.shape[1] > new_shape[1]) and (im.shape[1] <= (new_shape[1] + int(round(new_shape[1] * r2)))):
            im = cv2.resize(im, new_shape[::-1])
        else:
            if sliding_window:
                final_imgs = sliding_window_crop(im, cropsz=new_shape, random=random, base_side=base_side, ppocr_format=ppocr_format, gap=(int(gap_r[0] * 0), int(gap_r[1] * new_shape[1])), r1=r1, r2=r2, specific_color=specific_color, make_border=last_img_make_border)
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
    """
    blend: 
    释义
    v.
    （使）混合; 融合，结合; 协调
    n.
    融合; 混合（物）
    """
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
    blended_img = cv2.addWeighted(np.uint8(image), color_alpha, np.uint8(basic_color), 1 - color_alpha, 0)
    # blend with mask
    blended_img = cv2.addWeighted(np.uint8(blended_img), alpha, np.uint8(mask), 1 - alpha, 0)

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


def dcm2array(dcm_path):
    import pydicom
    ds = pydicom.read_file(dcm_path)  # 读取.dcm文件
    img = ds.pixel_array  # 提取图像信息
    # scipy.misc.imsave(out_path, img)
    return img


def cal_brightness(img):
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


def cal_brightness_v2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    brightness = np.mean(gray)
    return brightness


def opencv_add_chinese_text(img, text, left, top, font_path="simsun.ttc", textColor=(0, 255, 0), textSize=20):
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


def cal_svd_var(img):
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

    if flag == "digits":
        alpha = ' ' + '0123456789' + '.'
    elif flag == "alphabets":
        alpha = ' ' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    elif flag == "digits_alphabets":
        alpha = ' ' + '0123456789' + '.' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    elif flag == "digits_15":
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
        alpha = CH_SIM_CHARS
    elif flag == "Chinese_21160":
        CH_SIM_CHARS = ' '
        ch_sim_chars = open("words/chinese_chars_v1_21159.txt", "r", encoding="utf-8")
        lines = ch_sim_chars.readlines()
        for l in lines:
            CH_SIM_CHARS += l.strip()
        alpha = CH_SIM_CHARS
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

                makeBorderRes = make_border_v7(img, (64, 256), random=False, base_side="H", ppocr_format=True, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
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

                makeBorderRes = make_border_v7(img, (64, 256), random=False, base_side="H", ppocr_format=True, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
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


def read_ocr_lables(lbl_path):
    CH_SIM_CHARS = ' '
    ch_sim_chars = open(lbl_path, "r", encoding="utf-8")
    lines = ch_sim_chars.readlines()
    for l in lines:
        CH_SIM_CHARS += l.strip()
    alpha = CH_SIM_CHARS  # len = 1 + 6867 = 6868
    return alpha


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


def create_dbnet_train_test_txt(data_path, data_type="test"):
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


def create_ocr_img(imgsz=(64, 128), font=None, alpha="0123456789.AbC", target_len=1):
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


def labelbee_kpt_to_labelme_kpt(data_path):
    import labelme

    save_path = make_save_path(data_path, "labelme_format")
    img_save_path = save_path + "/images"
    json_save_path = save_path + "/jsons"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(json_save_path, exist_ok=True)

    images_path = data_path + "/images"
    jsons_path = data_path + "/jsons"
    file_list = get_file_list(jsons_path)
    for f in tqdm(file_list):
        try:
            img_name = os.path.splitext(f)[0]
            fname = os.path.splitext(img_name)[0]
            f_abs_path = jsons_path + "/{}".format(f)
            img_abs_path = images_path + "/{}.jpg".format(fname)
            img = cv2.imread(img_abs_path)
            imgsz = img.shape[:2]

            with open(f_abs_path, "r") as fr:
                src_data = json.load(fr)
            assert len(src_data["step_1"]["result"]) == 4, "N points should == 4!"

            p1 = (src_data["step_1"]["result"][0]["x"], src_data["step_1"]["result"][0]["y"])
            p2 = (src_data["step_1"]["result"][1]["x"], src_data["step_1"]["result"][1]["y"])
            p3 = (src_data["step_1"]["result"][2]["x"], src_data["step_1"]["result"][2]["y"])
            p4 = (src_data["step_1"]["result"][3]["x"], src_data["step_1"]["result"][3]["y"])

            shapes_data = []
            shapes_data.append({"label": "ul", "points": [[p1[0], p1[1]]], "group_id": None, "shape_type": "point", "flags": {}})
            shapes_data.append({"label": "ur", "points": [[p2[0], p2[1]]], "group_id": None, "shape_type": "point", "flags": {}})
            shapes_data.append({"label": "br", "points": [[p3[0], p3[1]]], "group_id": None, "shape_type": "point", "flags": {}})
            shapes_data.append({"label": "bl", "points": [[p4[0], p4[1]]], "group_id": None, "shape_type": "point", "flags": {}})

            json_labelme = {}
            json_labelme["version"] = "4.5.9"
            json_labelme["flags"] = eval("{}")
            json_labelme["shapes"] = shapes_data
            json_labelme["imagePath"] = img_name
            json_labelme["imageData"] = labelme.utils.img_arr_to_b64(img).strip()
            json_labelme["imageHeight"] = imgsz[0]
            json_labelme["imageWidth"] = imgsz[1]

            json_dst_path = json_save_path + "/{}.json".format(fname)
            with open(json_dst_path, 'w') as fw:
                json.dump(json_labelme, fw, indent=2)

            img_src_path = images_path + "/{}.jpg".format(fname)
            img_dst_path = img_save_path + "/{}.jpg".format(fname)
            shutil.copy(img_src_path, img_dst_path)

        except Exception as Error:
            print(Error)


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


def labelbee_kpt_to_labelme_kpt_multi_points(data_path):
    import labelme

    save_path = make_save_path(data_path, "labelme_format")
    img_save_path = save_path + "/images"
    json_save_path = save_path + "/jsons"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(json_save_path, exist_ok=True)

    images_path = data_path + "/images"
    jsons_path = data_path + "/jsons"
    file_list = get_file_list(jsons_path)
    for f in tqdm(file_list):
        try:
            img_name = os.path.splitext(f)[0]
            fname = os.path.splitext(img_name)[0]
            f_abs_path = jsons_path + "/{}".format(f)
            img_abs_path = images_path + "/{}.jpeg".format(fname)
            img = cv2.imread(img_abs_path)
            imgsz = img.shape[:2]

            with open(f_abs_path, "r") as fr:
                src_data = json.load(fr)
            assert len(src_data["step_1"]["result"]) != 0 and len(src_data["step_1"]["result"]) % 4 == 0, "N points should % 4 == 0 and != 0!"

            pts = []
            ni = 0
            for i in range(0, len(src_data["step_1"]["result"]), 4):
                if src_data["step_1"]["result"][i + 0]["attribute"] == "1":
                    p1 = [src_data["step_1"]["result"][i + 0]["x"], src_data["step_1"]["result"][i + 0]["y"]]
                if src_data["step_1"]["result"][i + 1]["attribute"] == "2":
                    p2 = [src_data["step_1"]["result"][i + 1]["x"], src_data["step_1"]["result"][i + 1]["y"]]
                if src_data["step_1"]["result"][i + 2]["attribute"] == "3":
                    p3 = [src_data["step_1"]["result"][i + 2]["x"], src_data["step_1"]["result"][i + 2]["y"]]
                if src_data["step_1"]["result"][i + 3]["attribute"] == "4":
                    p4 = [src_data["step_1"]["result"][i + 3]["x"], src_data["step_1"]["result"][i + 3]["y"]]

                pts.append([p1, p2, p3, p4])

                pt = [p1, p2, p3, p4]
                pt_copy = copy.deepcopy(pt)
                augNum = 3
                x1, x2 = round(min(p1[0], p4[0])), round(max(p2[0], p3[0]))
                y1, y2 = round(min(p1[1], p2[1])), round(max(p3[1], p4[1]))
                cropped_base = img[y1:y2, x1:x2]
                basesz = cropped_base.shape[:2]
                # ptsnew = aug_points(pt, n=10, imgsz=basesz, r=0.25)
                # # ptsnew = list(set(ptsnew))

                ni += 1

                ptsnew = []
                for i in range(augNum):
                    r_ = 0.01 * np.random.randint(10, 16)
                    pt_ = expand_kpt(basesz, pt, r=r_)
                    pt_cp = copy.deepcopy(pt_)
                    ptsnew.append(pt_cp)

                # pt_ = expand_kpt(basesz, pt, r=0.10)

                for idx, pi in enumerate(ptsnew):
                    # for idx, pi in enumerate([pt]):
                    ix1, ix2 = round(min(pi[0][0], pi[3][0])), round(max(pi[1][0], pi[2][0]))
                    iy1, iy2 = round(min(pi[0][1], pi[1][1])), round(max(pi[2][1], pi[3][1]))
                    cropped = img[iy1:iy2, ix1:ix2]
                    croppedsz = cropped.shape[:2]

                    shapes_data = []
                    shapes_data.append({"label": "ul", "points": [[pt_copy[0][0] - ix1, pt_copy[0][1] - iy1]], "group_id": None, "shape_type": "point", "flags": {}})
                    shapes_data.append({"label": "ur", "points": [[pt_copy[1][0] - ix1, pt_copy[1][1] - iy1]], "group_id": None, "shape_type": "point", "flags": {}})
                    shapes_data.append({"label": "br", "points": [[pt_copy[2][0] - ix1, pt_copy[2][1] - iy1]], "group_id": None, "shape_type": "point", "flags": {}})
                    shapes_data.append({"label": "bl", "points": [[pt_copy[3][0] - ix1, pt_copy[3][1] - iy1]], "group_id": None, "shape_type": "point", "flags": {}})

                    json_labelme = {}
                    json_labelme["version"] = "4.5.9"
                    json_labelme["flags"] = eval("{}")
                    json_labelme["shapes"] = shapes_data
                    json_labelme["imagePath"] = fname + "_{}_{}.jpg".format(ni, idx)
                    json_labelme["imageData"] = labelme.utils.img_arr_to_b64(cropped).strip()
                    json_labelme["imageHeight"] = croppedsz[0]
                    json_labelme["imageWidth"] = croppedsz[1]

                    json_dst_path = json_save_path + "/{}_{}_{}.json".format(fname, ni, idx)
                    with open(json_dst_path, 'w') as fw:
                        json.dump(json_labelme, fw, indent=2)

                    img_dst_path = img_save_path + "/{}_{}_{}.jpg".format(fname, ni, idx)
                    cv2.imwrite(img_dst_path, cropped)

        except Exception as Error:
            print(Error)


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
    

class CLS_ORT:
    def __init__(self, model_path):
        self.model_path = model_path
        self.ort_session = self.get_ort_session(self.model_path)
        self.inputs = self.ort_session.get_inputs()[0]
        self.outputs = self.ort_session.get_outputs()[0]
        self.input_name = self.inputs.name
        self.output_name =self.outputs.name
        self.inputsz = (self.inputs.shape[2], self.inputs.shape[3])

    def get_ort_session(self, model_path):
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = ort.InferenceSession(model_path, providers=providers)
        return ort_session

    def pre_process(self, img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        """
        # mean_ = np.array([0.485, 0.456, 0.406])
        # std_ = np.array([0.229, 0.224, 0.225])
        """
        # img = (img / 255. - np.array(mean)) / np.array(std)
        img = (img / 255. - 0.5) / 0.5
        # img = img / 255.
        img_x = img.transpose(2, 0, 1)
        img_x = img_x[np.newaxis, :]
        img_x = img_x.astype(np.float32)
        return img_x

    def softmax(self, x, axis=None):
        ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return ex / np.sum(ex, axis=axis, keepdims=True)

    def post_process(self, ort_out):
        # ort_out = ort_outs[0][0]
        out = self.softmax(np.array(ort_out), axis=0)
        cls = np.argmax(out)
        return cls

    def inference(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_cp = img
        img = cv2.resize(img, self.inputsz)
        img = self.pre_process(img)

        ort_outs = self.ort_session.run(["output"], {self.inputs.name: img})
        cls = self.post_process(ort_outs[0][0])
        
        return cls
    

def create_cls_negatives_via_random_crop(data_path, random_size=(96, 100, 128, 160), randint_low=10, randint_high=51, hw_dis=100, dst_num=20000):
    img_list = sorted(os.listdir(data_path))

    save_path = make_save_path(data_path, ".", "random_cropped")
    os.makedirs(save_path, exist_ok=True)

    total_num = 0

    for img in tqdm(img_list[6000:]):

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
    """
    Runs Non-Maximum Suppression (NMS) on inference results

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


def seamless_clone(fg_path, bg_path, dst_num=5000):
    """
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
    """
    save_path = make_save_path(fg_path, ".", "seamless_clone_result")
    fg_img_list = get_file_list(fg_path)
    bg_img_list = get_file_list(bg_path)

    for fg in fg_img_list:
        fgname = os.path.splitext(fg)[0]
        fg_abs_path = fg_path + "/{}".format(fg)
        fg_img = cv2.imread(fg_abs_path)
        fgsz = fg_img.shape[:2]

        for bg in bg_img_list:
            bgname = os.path.splitext(bg)[0]
            bg_abs_path = bg_path + "/{}".format(bg)
            bg_img = cv2.imread(bg_abs_path)
            bgsz = bg_img.shape[:2]

            bg_img = cv2.resize(bg_img, (fgsz[1], fgsz[0]))

            mask = 255 * np.ones(fg_img.shape, fg_img.dtype)
            center = (fgsz[1] // 2, fgsz[0] // 2)
            # output_normal = cv2.seamlessClone(fg_img, bg_img, mask, center, cv2.NORMAL_CLONE)
            # output_mixed = cv2.seamlessClone(fg_img, bg_img, mask, center, cv2.MIXED_CLONE)
            output_MONOCHROME = cv2.seamlessClone(fg_img, bg_img, mask, center, cv2.MONOCHROME_TRANSFER)

            output_path = save_path + "/{}_{}_{}.png".format(fgname, bgname, "MIXED_CLONE")
            cv2.imwrite(output_path, output_MONOCHROME)

            if len(os.listdir(save_path)) >= 5000:
                break


            # mask = np.zeros(fgsz, dtype=np.uint8)
            # center = (fgsz[1]//2, fgsz[0]//2)  # 中心点坐标
            # axes = (fgsz[1]//4, fgsz[0]//4)    # 椭圆长短轴
            # cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)  # 绘制白色填充椭圆

            # # 设置克隆位置（这里使用目标图像中心）
            # clone_position = (fgsz[1]//2, fgsz[0]//2)

            # # 执行无缝克隆
            # result = cv2.seamlessClone(
            #     fg_img, 
            #     bg_img, 
            #     mask, 
            #     clone_position, 
            #     cv2.NORMAL_CLONE  # 克隆模式：保持纹理细节
            # )

            # output_path = save_path + "/{}_{}_{}.png".format(fgname, bgname, "MIXED_CLONE")
            # cv2.imwrite(output_path, result)


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
            if l.strip() == "":
                continue
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
    names = {
        '0': 'person', '1': 'bicycle', '2': 'car', '3': 'motorcycle', '4': 'airplane',
        '5': 'bus', '6': 'train', '7': 'truck', '8': 'boat', '9': 'traffic light',
        '10': 'fire hydrant', '11': 'stop sign', '12': 'parking meter', '13': 'bench', '14': 'bird',
        '15': 'cat', '16': 'dog', '17': 'horse', '18': 'sheep', '19': 'cow',
        '20': 'elephant', '21': 'bear', '22': 'zebra', '23': 'giraffe', '24': 'backpack',
        '25': 'umbrella', '26': 'handbag', '27': 'tie', '28': 'suitcase', '29': 'frisbee',
        '30': 'skis', '31': 'snowboard', '32': 'sports ball', '33': 'kite', '34': 'baseball bat',
        '35': 'baseball glove', '36': 'skateboard', '37': 'surfboard', '38': 'tennis racket', '39': 'bottle',
        '40': 'wine glass', '41': 'cup', '42': 'fork', '43': 'knife', '44': 'spoon',
        '45': 'bowl', '46': 'banana', '47': 'apple', '48': 'sandwich', '49': 'orange',
        '50': 'broccoli', '51': 'carrot', '52': 'hot dog', '53': 'pizza', '54': 'donut',
        '55': 'cake', '56': 'chair', '57': 'couch', '58': 'potted plant', '59': 'bed',
        '60': 'dining table', '61': 'toilet', '62': 'tv', '63': 'laptop', '64': 'mouse',
        '65': 'remote', '66': 'keyboard', '67': 'cell phone', '68': 'microwave', '69': 'oven',
        '70': 'toaster', '71': 'sink', '72': 'refrigerator', '73': 'book', '74': 'clock',
        '75': 'vase', '76': 'scissors', '77': 'teddy bear', '78': 'hair drier', '79': 'toothbrush'
        }
    return names


def get_coco_categories():
    categories = [
        {
            "id": 0,
            "name": 'person',
            "supercategory": 'sar',
        },
        {
            "id": 1,
            "name": 'bicycle',
            "supercategory": 'sar',
        },
        {
            "id": 2,
            "name": 'car',
            "supercategory": 'sar',
        },
        {
            "id": 3,
            "name": 'motorcycle',
            "supercategory": 'sar',
        },
        {
            "id": 4,
            "name": 'airplane',
            "supercategory": 'sar',
        },
        {
            "id": 5,
            "name": 'bus',
            "supercategory": 'sar',
        },
        {
            "id": 6,
            "name": 'train',
            "supercategory": 'sar',
        },
        {
            "id": 7,
            "name": 'truck',
            "supercategory": 'sar',
        },
        {
            "id": 8,
            "name": 'boat',
            "supercategory": 'sar',
        },
        {
            "id": 9,
            "name": 'traffic light',
            "supercategory": 'sar',
        },
        {
            "id": 10,
            "name": 'fire hydrant',
            "supercategory": 'sar',
        },
        {
            "id": 11,
            "name": 'stop sign',
            "supercategory": 'sar',
        },
        {
            "id": 12,
            "name": 'parking meter',
            "supercategory": 'sar',
        },
        {
            "id": 13,
            "name": 'bench',
            "supercategory": 'sar',
        },
        {
            "id": 14,
            "name": 'bird',
            "supercategory": 'sar',
        },
        {
            "id": 15,
            "name": 'cat',
            "supercategory": 'sar',
        },
        {
            "id": 16,
            "name": 'dog',
            "supercategory": 'sar',
        },
        {
            "id": 17,
            "name": 'horse',
            "supercategory": 'sar',
        },
        {
            "id": 18,
            "name": 'sheep',
            "supercategory": 'sar',
        },
        {
            "id": 19,
            "name": 'cow',
            "supercategory": 'sar',
        },
        {
            "id": 20,
            "name": 'elephant',
            "supercategory": 'sar',
        },
        {
            "id": 21,
            "name": 'bear',
            "supercategory": 'sar',
        },
        {
            "id": 22,
            "name": 'zebra',
            "supercategory": 'sar',
        },
        {
            "id": 23,
            "name": 'giraffe',
            "supercategory": 'sar',
        },
        {
            "id": 24,
            "name": 'backpack',
            "supercategory": 'sar',
        },
        {
            "id": 25,
            "name": 'umbrella',
            "supercategory": 'sar',
        },
        {
            "id": 26,
            "name": 'handbag',
            "supercategory": 'sar',
        },
        {
            "id": 27,
            "name": 'tie',
            "supercategory": 'sar',
        },
        {
            "id": 28,
            "name": 'suitcase',
            "supercategory": 'sar',
        },
        {
            "id": 29,
            "name": 'frisbee',
            "supercategory": 'sar',
        },
        {
            "id": 30,
            "name": 'skis',
            "supercategory": 'sar',
        },
        {
            "id": 31,
            "name": 'snowboard',
            "supercategory": 'sar',
        },
        {
            "id": 32,
            "name": 'sports ball',
            "supercategory": 'sar',
        },
        {
            "id": 33,
            "name": 'kite',
            "supercategory": 'sar',
        },
        {
            "id": 34,
            "name": 'baseball bat',
            "supercategory": 'sar',
        },
        {
            "id": 35,
            "name": 'baseball glove',
            "supercategory": 'sar',
        },
        {
            "id": 36,
            "name": 'skateboard',
            "supercategory": 'sar',
        },
        {
            "id": 37,
            "name": 'surfboard',
            "supercategory": 'sar',
        },
        {
            "id": 38,
            "name": 'tennis racket',
            "supercategory": 'sar',
        },
        {
            "id": 39,
            "name": 'bottle',
            "supercategory": 'sar',
        },
        {
            "id": 40,
            "name": 'wine glass',
            "supercategory": 'sar',
        },
        {
            "id": 41,
            "name": 'cup',
            "supercategory": 'sar',
        },
        {
            "id": 42,
            "name": 'fork',
            "supercategory": 'sar',
        },
        {
            "id": 43,
            "name": 'knife',
            "supercategory": 'sar',
        },
        {
            "id": 44,
            "name": 'spoon',
            "supercategory": 'sar',
        },
        {
            "id": 45,
            "name": 'bowl',
            "supercategory": 'sar',
        },
        {
            "id": 46,
            "name": 'banana',
            "supercategory": 'sar',
        },
        {
            "id": 47,
            "name": 'apple',
            "supercategory": 'sar',
        },
        {
            "id": 48,
            "name": 'sandwich',
            "supercategory": 'sar',
        },
        {
            "id": 49,
            "name": 'orange',
            "supercategory": 'sar',
        },
        {
            "id": 50,
            "name": 'broccoli',
            "supercategory": 'sar',
        },
        {
            "id": 51,
            "name": 'carrot',
            "supercategory": 'sar',
        },
        {
            "id": 52,
            "name": 'hot dog',
            "supercategory": 'sar',
        },
        {
            "id": 53,
            "name": 'pizza',
            "supercategory": 'sar',
        },
        {
            "id": 54,
            "name": 'donut',
            "supercategory": 'sar',
        },
        {
            "id": 55,
            "name": 'cake',
            "supercategory": 'sar',
        },
        {
            "id": 56,
            "name": 'chair',
            "supercategory": 'sar',
        },
        {
            "id": 57,
            "name": 'couch',
            "supercategory": 'sar',
        },
        {
            "id": 58,
            "name": 'potted plant',
            "supercategory": 'sar',
        },
        {
            "id": 59,
            "name": 'bed',
            "supercategory": 'sar',
        },
        {
            "id": 60,
            "name": 'dining table',
            "supercategory": 'sar',
        },
        {
            "id": 61,
            "name": 'toilet',
            "supercategory": 'sar',
        },
        {
            "id": 62,
            "name": 'tv',
            "supercategory": 'sar',
        },
        {
            "id": 63,
            "name": 'laptop',
            "supercategory": 'sar',
        },
        {
            "id": 64,
            "name": 'mouse',
            "supercategory": 'sar',
        },
        {
            "id": 65,
            "name": 'remote',
            "supercategory": 'sar',
        },
        {
            "id": 66,
            "name": 'keyboard',
            "supercategory": 'sar',
        },
        {
            "id": 67,
            "name": 'cell phone',
            "supercategory": 'sar',
        },
        {
            "id": 68,
            "name": 'microwave',
            "supercategory": 'sar',
        },
        {
            "id": 69,
            "name": 'oven',
            "supercategory": 'sar',
        },
        {
            "id": 70,
            "name": 'toaster',
            "supercategory": 'sar',
        },
        {
            "id": 71,
            "name": 'sink',
            "supercategory": 'sar',
        },
        {
            "id": 72,
            "name": 'refrigerator',
            "supercategory": 'sar',
        },
        {
            "id": 73,
            "name": 'book',
            "supercategory": 'sar',
        },
        {
            "id": 74,
            "name": 'clock',
            "supercategory": 'sar',
        },
        {
            "id": 75,
            "name": 'vase',
            "supercategory": 'sar',
        },
        {
            "id": 76,
            "name": 'scissors',
            "supercategory": 'sar',
        },
        {
            "id": 77,
            "name": 'teddy bear',
            "supercategory": 'sar',
        },
        {
            "id": 78,
            "name": 'hair drier',
            "supercategory": 'sar',
        },
        {
            "id": 79,
            "name": 'toothbrush',
            "supercategory": 'sar',
        }
    ]

    return categories

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


def get_font_char_image(data_path, chars="0123456789.AbC"):
    save_path = make_save_path(data_path, relative=".", add_str="results")
    
    FONT_CHARS_DICT = get_all_font_chars(font_dir=data_path, word_set="0123456789.AbC")
    print(FONT_CHARS_DICT)
    
    font_path_list = list(FONT_CHARS_DICT.keys())
    for ft_path in tqdm(font_path_list):
        font_name = os.path.splitext(os.path.basename(ft_path))[0]
        ft = ImageFont.truetype(ft_path, size=48)
        for a in chars:
            image = gen_img(imgsz=(64, 128), font=ft, alpha=a, target_len=1)
            cv2.imwrite("{}/{}_bg1_{}.jpg".format(save_path, font_name, a), image)


def create_ocr_rec_train_txt_base(data_path, alpha):
    """
    fname=label.jpg --> fname=label.jpg label
    Returns
    -------

    """
    # data_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/test/v2/15_cls/64_256_v5"
    save_path = data_path + ".txt"

    fw = open(save_path, "w", encoding="utf-8")
    file_list = get_file_list(data_path, abspath=True)

    for f in tqdm(file_list):
        fname = os.path.basename(f)
        fname_ = os.path.splitext(fname)[0]
        label = fname_.split("=")[1]

        num_ = 0
        for l in label:
            if l not in alpha:
                num_ += 1

        if os.path.exists(f) and num_ == 0:
            content = "{} {}\n".format(f, label)
            fw.write(content)

    fw.close()


def create_ocr_rec_train_txt(data_path, LABEL):
    """
    fname=label.jpg --> fname=label.jpg label
    Returns
    -------

    """
    save_path = data_path + ".txt"

    fw = open(save_path, "w", encoding="utf-8")

    dirs = sorted(os.listdir(data_path))
    for d in dirs:
        d_path = data_path + "/{}".format(d)
        ddirs = sorted(os.listdir(d_path))
        for dd in ddirs:
            dd_path = d_path + "/{}".format(dd)
            if os.path.isfile(dd_path): continue
            file_list = get_file_list(dd_path)
            for f in tqdm(file_list):
                f_src_path = dd_path + "/{}".format(f)
                if not f.endswith(".jpg") and not f.endswith(".jpeg") and not f.endswith(".png"):
                    print(f_src_path)
                try:
                    # fname = os.path.basename(f_src_path)
                    fname_ = os.path.splitext(f)[0]
                    label = fname_.split("=")[1]

                    num_ = 0
                    for l in label:
                        if l not in LABEL:
                            num_ += 1

                    if os.path.exists(f_src_path) and num_ == 0:
                        content = "{} {}\n".format(f_src_path, label)
                        fw.write(content)
                except Exception as Error:
                    print(Error)
                    print(f_src_path)
    fw.close()


def merge_txt_files(data_path):
    dirname = os.path.basename(data_path)
    file_list = get_file_list(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "../Merged_txt"))
    os.makedirs(save_path, exist_ok=True)
    merged_txt_path = save_path + "/{}.txt".format(dirname)
    fw = open(merged_txt_path, "w", encoding="utf-8")

    for f in file_list:
        f_path = data_path + "/{}".format(f)
        if os.path.isfile(f_path) and f_path.endswith(".txt"):
            with open(f_path, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
                fw.writelines(lines)
    fw.close()


def merge_ocr_rec_train_txt_files(data_path, LABEL):
    """
    fname=label.jpg --> fname=label.jpg label
    Returns
    -------

    """
    # save_path = data_path + ".txt"
    #
    # fw = open(save_path, "w", encoding="utf-8")

    dirs = sorted(os.listdir(data_path))

    for d in dirs:
        d_path = data_path + "/{}".format(d)

        merge_txt_files(d_path)

        # if os.path.isfile(d_path): continue
        # ddirs = sorted(os.listdir(d_path))
        # for dd in ddirs:
        #     dd_path = d_path + "/{}".format(dd)
        #     merge_txt_files(dd_path)

    # fw.close()


def check_ocr_label(data_path, label):
    """
    data_path: *.txt
    fname=label.jpg label
    Parameters
    ----------
    data_path
    label

    Returns
    -------

    """
    assert os.path.isfile(data_path) and data_path.endswith(".txt"), "{} should be *.txt"
    fr = open(data_path, "r", encoding="utf-8")
    lines = fr.readlines()
    fr.close()

    LABEL = ""

    for line in tqdm(lines):
        f, lbl = line.split(" ")[0], line.split(" ")[1].strip()
        for l in lbl:
            if l not in LABEL:
                LABEL += l

    print("label: {}, label length: {}".format(label, len(label)))
    print("LABEL: {}, LABEL length: {}".format(LABEL, len(LABEL)))

    un = ""
    for l in label:
        if l not in LABEL:
            un += l
    print("exclude: {}".format(un))


def list_module_functions():
    import inspect
    import importlib
    """
    列出模块中所有的函数
    """
    current_file = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file)
    os.chdir(current_dir)
    module = importlib.import_module(os.path.basename(current_file)[:-3])
    functions = [func for func in dir(module) if callable(getattr(module, func))]
    print(sorted(functions))


# -------- cal params and flops --------
class TestConv2dNet(nn.Module):
    """
    params: (3 * 3 * 3 + bias) * 16 + (16 * 3 * 3 + bias) * 32 = 5040  # 与thop结果一致
    flops: (3 * 3 * 3 + 3 * (3 * 3 - 1) + bias) * 16 * 224 * 224
           +
           (16 * 3 * 3 + 16 * (3 * 3 - 1) + bias) * 32 * 224 * 224
           =
           480083968
           # 480083968 / 2 = 240041984.0  thop结果为252887040.0 不一致
    """
    def __init__(self, bias):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TestLinearNet(nn.Module):
    """
    params: (16 + bias) * 32 + (32 + bias) * 64 = 2560  # 与thop结果一致
    flops: https://blog.csdn.net/qq_37025073/article/details/106735053
    
    """
    def __init__(self, bias):
        super().__init__()
        self.fc1 = nn.Linear(16, 32, bias=bias)
        self.fc2 = nn.Linear(32, 64, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TestLSTMNet(nn.Module):
    """
    params: 4 * ((16 + bias + 32 + bias) * 32) + 4 * ((32 + bias + 64 + bias) * 64) = 30720  # 与thop结果一致
    flops: 
    """
    def __init__(self, bias):
        super().__init__()
        self.lstm1 = nn.LSTM(16, 32, bidirectional=False, num_layers=1, bias=bias)
        self.lstm2 = nn.LSTM(32, 64, bidirectional=False, num_layers=1, bias=bias)
 
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x
    

def cal_params_flops(model, input, bias_flag=True, method="thop"):
    """
    https://zhuanlan.zhihu.com/p/387349200
    https://www.cnblogs.com/picassooo/p/16343737.html

    bias_flag = True
    bias = 0
    if bias_flag:
        bias = 1

    Params --------------------------------------------------------
    Conv2d:
    params = (k_h * k_w * c_in + bias) * c_out
    
    Linear -> Conv2d:
    params = (H_out * W_out * N + bias) * F_out  # N: 卷积核的数量

    Linear -> Linear:
    params = (F_in + bias) * F_out

    LSTM:
    https://baijiahao.baidu.com/s?id=1735032676336476820&wfr=spider&for=pc
    params = 4 * [(input_dim + hidden_dim + bias) * hidden_dim]


    FLOPs --------------------------------------------------------
    Conv2d:
    flops = ((c_in * k_h * k_w) + (c_in * k_h * k_w - 1) + bias) * c_out * H_feat * W_feat

    Linear:
    flops = ((2 * I - 1) + bias) * O  #  I是全连接输入的神经元数, O是全连接输出的神经元数

    """
    import thop

    assert method in ["thop", "torchstat", "manual"], 'method should be ["thop", "torchstat", "manual"]!'

    bias = 0
    if bias_flag:
        bias = 1

    if method == "thop":
        ops, params = thop.profile(model, inputs=(input, ))
        print("thop -> ops: {}, params: {}".format(ops, params))

    elif method == "torchstat":
        raise NotImplementedError
    # elif method == "torchstat":
    #     model_stat = torchstat.stat(model, tuple(input.shape[1:]))
    #     print("torchstat -> {}".format(model_stat))

    else:
        print("manual, 参考注释手动计算!")


def change_txt_content(txt_path):
    save_path = make_save_path(txt_path, relative=".", add_str="new")
    file_path = get_file_list(txt_path, abspath=False)

    for f in file_path:
        f_abs_path = txt_path + "/{}".format(f)
        fr = open(f_abs_path, "r", encoding="utf-8")
        txt_content = fr.readlines()
        fr.close()

        f_dst_path = save_path + "/{}".format(f)
        with open(f_dst_path, "w", encoding="utf-8") as fw:
            # for line in txt_content:
            #     l = line.strip().split(" ")
            #     cls = int(l[0])
            #     if cls == 1:
            #         cls_new = cls - 1
            #     else:
            #         cls_new = cls
            #     l_new= str(cls_new) + " " + " ".join([str(a) for a in l[1:]]) + '\n'
            #     fw.write(l_new)

            for line in txt_content:
                l = line.strip().split(" ")
                cls = int(l[0])
                if cls == 1:
                    cls_new = 0
                else:
                    cls_new = 1
                l_new= str(cls_new) + " " + " ".join([str(a) for a in l[1:]]) + '\n'
                fw.write(l_new)

            # for i, line in enumerate(txt_content):
            #     l = line.strip().split(" ")
                
            #     if i == len(txt_content) - 1:
            #         cls_new = int(l[0]) + 1
            #         l_new= str(cls_new) + " " + " ".join([str(a) for a in l[1:]]) + '\n'
            #     else:
            #         cls_new = int(l[0])
            #         l_new= str(cls_new) + " " + " ".join([str(a) for a in l[1:]]) + '\n'
            #     fw.write(l_new)

            # sum_0 = 0
            # for line in txt_content:
            #     l = line.strip().split(" ")
            #     cls = int(l[0])
            #     if cls == 0:
            #         sum_0 += 1

            # if sum_0 > 1:
            #     idx_0 = 0
            #     for line in txt_content:
            #         l = line.strip().split(" ")
            #         cls = int(l[0])
            #         if cls == 0:
            #             idx_0 += 1
            #             if idx_0 == 1:
            #                 cls_new = cls
            #                 l_new= str(cls_new) + " " + " ".join([str(a) for a in l[1:]]) + '\n'
            #                 fw.write(l_new)
            #         else:
            #             cls_new = cls
            #             l_new= str(cls_new) + " " + " ".join([str(a) for a in l[1:]]) + '\n'
            #             fw.write(l_new)
            # else:
            #     for line in txt_content:
            #         l = line.strip().split(" ")
            #         cls_new = int(l[0])
            #         l_new= str(cls_new) + " " + " ".join([str(a) for a in l[1:]]) + '\n'
            #         fw.write(l_new)


def expand_yolo_bbox(bbx, size, n=1.0):
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
        return bbx
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

        return [x1_new, y1_new, x2_new, y2_new]

    
def yolo_label_expand_bbox(data_path, classes, r=1.5):
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    save_path = make_save_path(lbl_path, relative=".", add_str="lables_expand")

    file_list = get_file_list(img_path)
    for f in file_list:
        f_base_name = os.path.splitext(f)[0]
        img_abs_path = img_path + "/{}".format(f)
        lbl_abs_path = lbl_path + "/{}.txt".format(f_base_name)

        img = cv2.imread(img_abs_path)
        imgsz = img.shape[:2]

        fr = open(lbl_abs_path, "r", encoding="utf-8")
        txt_content = fr.readlines()
        fr.close()

        lbl_dst_path = save_path + "/{}.txt".format(f_base_name)
        with open(lbl_dst_path, "w", encoding="utf-8") as fw:
            for line in txt_content:
                l = line.strip().split(" ")

                cls = int(l[0])
                if cls == classes:
                    bbox_yolo = list(map(float, l[1:]))
                    bbox_voc = bbox_yolo_to_voc(imgsz, bbox_yolo)
                    bbox_new = expand_yolo_bbox(bbox_voc, imgsz, n=r)
                    bbox_yolo_new= bbox_voc_to_yolo(imgsz, bbox_new)
                    
                    l_new= str(cls) + " " + " ".join([str(a) for a in bbox_yolo_new]) + '\n'
                    fw.write(l_new)
                else:
                    fw.write(line)


def ffmpeg_extract_video_frames(video_path, fps=25):
    save_path = make_save_path(video_path, relative=".", add_str="frames")
    file_list = get_file_list(video_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = video_path + "/{}".format(f)
        f_dst_path = save_path + "/{}".format(fname)
        os.makedirs(f_dst_path, exist_ok=True)

        # # FFmpeg 命令，提取视频的每一帧
        # command = [
        #     'ffmpeg.exe', 
        #     '-i', f_abs_path, 
        #     '-r', '5'
        #     ' -q:v', '1',  # 设定每秒提取1帧
        #     '-f', 'image2',
        #     f'{f_dst_path}/output_%09d.jpg'  # 输出文件格式
        # ]

        # subprocess.run(command)

        command = f"D:/installed/ffmpeg-7.0.2-essentials_build/bin/ffmpeg.exe -i {f_abs_path} -r {fps} -q:v 1 -f image2 {f_dst_path}/{fname}_output_%09d.jpg"
        os.system(command)



def make_border_and_change_yolo_labels(data_path, dstsz=(-1, 1920)):
    """
    补边扩展图像, 同时将yolo label做对应的改变使得变换之后目标框仍然正确.
    """
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    save_path = make_save_path(data_path, relative=".", add_str="make_border_results")
    img_save_path = save_path + "/images"
    lbl_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    rs = list(range(10, 30))
    rs = [i * 0.01 for i in rs]

    file_list = get_file_list(img_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        img_abs_path = img_path + "/{}".format(f)
        lbl_abs_path = lbl_path + "/{}.txt".format(fname)

        fr = open(lbl_abs_path, "r", encoding="utf-8")
        lines = fr.readlines()
        fr.close()

        img = cv2.imread(img_abs_path)
        imgsz = img.shape[:2]

        img_dst_path = img_save_path + "/{}".format(f)
        lbl_dst_path = lbl_save_path + "/{}.txt".format(fname)

        with open(lbl_dst_path, "w", encoding="utf-8") as fw:
            if dstsz[1] > imgsz[1]:
                pad_w = dstsz[1] - imgsz[1]
            else:
                pad_w = 0
            rx = np.random.randint(0, 11) * 0.1
            pad_left = int(pad_w * rx)
            pad_right = pad_w - pad_left

            if dstsz[0] > imgsz[0]:
                pad_h = dstsz[0] - imgsz[0]
            else:
                pad_h = 0
            ry = np.random.randint(0, 11) * 0.1
            pad_top = int(pad_h * ry)
            pad_bottom = pad_h - pad_top
            img_new = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            imgnewsz = img_new.shape[:2]
            cv2.imwrite(img_dst_path, img_new)

            for line in lines:
                l = line.strip().split(" ")
                bbox_yolo = list(map(float, l[1:]))
                bbox_voc = bbox_yolo_to_voc(imgsz, bbox_yolo)

                x1, y1, x2, y2 = bbox_voc
                bbox_voc_new = [pad_left + x1, pad_top + y1, pad_left + x2, pad_top + y2]
                bbox_yolo_new = bbox_voc_to_yolo(imgnewsz, bbox_voc_new)

                txt_content = "{} ".format(str(l[0])) + " ".join([str(b) for b in bbox_yolo_new]) + "\n"
                fw.write(txt_content)


def norm_vector_3d(p1, p2, p3):
    """
    https://blog.51cto.com/u_16213405/12842859
    
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([0, 1, 0])
    n = norm_vector(p1, p2, p3)
    print(n)  # [0 0 1]

    """
    v12 = p2 - p1
    v13 = p3 - p1

    n = np.cross(v12, v13)
    return n


def cal_angle_via_vector_cross(p1, p2, p3):
    """
    通过向量叉乘计算角度
    """

    v12 = p2 - p1
    v13 = p3 - p1

    v = v12[0] * v13[0] + v12[1] * v13[1]
    len_v12 = np.sqrt(v12[0] ** 2 + v12[1] ** 2)
    len_v13 = np.sqrt(v13[0] ** 2 + v13[1] ** 2)
    angle = np.arccos(v / (len_v12 * len_v13))
    angle = angle * 180 / np.pi

    return angle


def append_content_to_txt_test():
    for s in range(1, 12):
        path1 = r"D:\Gosion\Projects\004.GuardArea_Det\data\example\{}_yolo_format\labels".format(s)
        file = os.listdir(path1)[0]
        f_abs_path = path1 + "/{}".format(file)

        with open(f_abs_path, "r") as f:
            # line_src = f.readlines()[0]
            line_src = ""
            lines = f.readlines()
            for line in lines:
                l = line.strip().split(" ")
                cls = int(l[0])
                if cls == 1:
                    line_src = line

        path2 = r"D:\Gosion\Projects\004.GuardArea_Det\data\v2\{}_yolo_format\labels".format(s)
        files = os.listdir(path2)

        for f in files:
            f_abs_path2 = path2 + "/{}".format(f)
            with open(f_abs_path2, "a") as fa:
                fa.write(line_src)


    # path1 = r"D:\Gosion\Projects\004.GuardArea_Det\data\v1\1_yolo_format\labels"
    # file = os.listdir(path1)[0]
    # f_abs_path = path1 + "/{}".format(file)

    # with open(f_abs_path, "r") as f:
    #     line_src = f.readlines()[0]
        

    # path2 = r"D:\Gosion\Projects\004.GuardArea_Det\data\v1\2\labels"
    # files = os.listdir(path2)

    # for f in files:
    #     f_abs_path2 = path2 + "/{}".format(f)
    #     with open(f_abs_path2, "a") as fa:
    #         fa.write(line_src)


def append_jitter_box_yolo_label(data_path, append_label, p=5):
    # lines = ["2 0.4578125 0.750925925925926 0.3625 0.4935185185185185"]

    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    # lbl_path_new = data_path + "/labels_new"
    # os.makedirs(lbl_path_new, exist_ok=True)

    file_list = get_file_list(lbl_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = lbl_path + "/{}".format(f)
        # f_abs_path_new = lbl_path_new + "/{}".format(f)
        img_abs_path = img_path + "/{}.jpg".format(fname)
        img = cv2.imread(img_abs_path)
        imgsz = img.shape[:2]

        with open(f_abs_path, "a") as f_append:
            for line in append_label:
                l = line.strip().split(" ")
                cls = int(l[0])
                
                bbox_yolo = list(map(float, l[1:]))
                bbox_voc = bbox_yolo_to_voc(imgsz, bbox_yolo)
                bbox_voc_new = [
                    bbox_voc[0] + np.random.randint(-p, p + 1),
                    bbox_voc[1] + np.random.randint(-p, p + 1),
                    bbox_voc[2] + np.random.randint(-p, p + 1),
                    bbox_voc[3] + np.random.randint(-p, p + 1)
                ]

                if bbox_voc_new[0] < 0: bbox_voc_new[0] = 0
                if bbox_voc_new[1] < 0: bbox_voc_new[1] = 0
                if bbox_voc_new[2] < 0: bbox_voc_new[2] = 0
                if bbox_voc_new[3] < 0: bbox_voc_new[3] = 0
                if bbox_voc_new[0] > imgsz[1]: bbox_voc_new[0] = imgsz[1]
                if bbox_voc_new[1] > imgsz[0]: bbox_voc_new[1] = imgsz[0]
                if bbox_voc_new[2] > imgsz[1]: bbox_voc_new[2] = imgsz[1]
                if bbox_voc_new[3] > imgsz[0]: bbox_voc_new[3] = imgsz[0]

                bbox_yolo_new = bbox_voc_to_yolo(imgsz, bbox_voc_new)
                txt_content_new = "{} ".format(cls) + " ".join([str(b) for b in bbox_yolo_new]) + "\n"
                f_append.write(txt_content_new)




def jitter_bbox(data_path, classes=(2, ), p=5):
    """
    input is yolo format
    p: jitter pixels
    """
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    lbl_path_new = data_path + "/labels_new"
    os.makedirs(lbl_path_new, exist_ok=True)

    file_list = get_file_list(lbl_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = lbl_path + "/{}".format(f)
        f_abs_path_new = lbl_path_new + "/{}".format(f)
        img_abs_path = img_path + "/{}.jpg".format(fname)
        img = cv2.imread(img_abs_path)
        imgsz = img.shape[:2]

        with open(f_abs_path, "r") as f_read:
            with open(f_abs_path_new, "w") as f_write:
                lines = f_read.readlines()
                for line in lines:
                    l = line.strip().split(" ")
                    cls = int(l[0])

                    if cls in classes:
                         bbox_yolo = list(map(float, l[1:]))
                         bbox_voc = bbox_yolo_to_voc(imgsz, bbox_yolo)
                         bbox_voc_new = [
                             bbox_voc[0] + np.random.randint(-p, p + 1),
                             bbox_voc[1] + np.random.randint(-p, p + 1),
                             bbox_voc[2] + np.random.randint(-p, p + 1),
                             bbox_voc[3] + np.random.randint(-p, p + 1)
                         ]

                         if bbox_voc_new[0] < 0: bbox_voc_new[0] = 0
                         if bbox_voc_new[1] < 0: bbox_voc_new[1] = 0
                         if bbox_voc_new[2] < 0: bbox_voc_new[2] = 0
                         if bbox_voc_new[3] < 0: bbox_voc_new[3] = 0
                         if bbox_voc_new[0] > imgsz[1]: bbox_voc_new[0] = imgsz[1]
                         if bbox_voc_new[1] > imgsz[0]: bbox_voc_new[1] = imgsz[0]
                         if bbox_voc_new[2] > imgsz[1]: bbox_voc_new[2] = imgsz[1]
                         if bbox_voc_new[3] > imgsz[0]: bbox_voc_new[3] = imgsz[0]

                         bbox_yolo_new = bbox_voc_to_yolo(imgsz, bbox_voc_new)
                         txt_content_new = "{} ".format(cls) + " ".join([str(b) for b in bbox_yolo_new]) + "\n"
                         f_write.write(txt_content_new)
                    else:
                        f_write.write(line)


    os.rename(lbl_path, lbl_path + "-old")
    os.rename(lbl_path_new, lbl_path_new.replace("/labels_new", "/labels"))

                    
def check_yolo_labels(data_path):
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"

    save_path = data_path + "/abnormal_images_labels"
    img_save_path = save_path + "/images"
    lbl_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    file_list = sorted(os.listdir(lbl_path))
    for f in file_list:
        fname = os.path.splitext(f)[0]
        lbl_src_path = lbl_path + "/{}".format(f)

        # 1 ------------------------
        sum_object = 0
        sum_all = 0
        with open(lbl_src_path, "r", encoding="utf-8") as f_read:
            lines = f_read.readlines()
            for line in lines:
                l = line.strip().split(" ")
                cls = int(l[0])
                if cls == 0:
                    sum_object += 1
                sum_all += 1
        
        if sum_all != 2 or sum_object > 1:
            img_src_path = img_path + "/{}.jpg".format(fname)
            img_dst_path = img_save_path + "/{}.jpg".format(fname)
            lbl_dst_path = lbl_save_path + "/{}".format(f)
            shutil.move(img_src_path, img_dst_path)
            shutil.move(lbl_src_path, lbl_dst_path)

        

def delete_yolo_labels_high_iou_bbox(data_path, iou_thr=0.8, target_cls=(0, 1), del_cls=0):
    """
    通过模型生成的标签中，有些框可能差不多，几乎重叠，但是类别不一样
    函数的功能是去除指定的重复的类别框
    """
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    lbl_path_new = data_path + "/labels_new"
    os.makedirs(lbl_path_new, exist_ok=True)

    file_list = get_file_list(lbl_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = lbl_path + "/{}".format(f)
        f_abs_path_new = lbl_path_new + "/{}".format(f)
        img_abs_path = img_path + "/{}.jpg".format(fname)
        if not os.path.exists(img_abs_path):
            img_abs_path = img_path + "/{}.png".format(fname)
            
        img = cv2.imread(img_abs_path)
        if img is None: continue
        imgsz = img.shape[:2]

        with open(f_abs_path, "r") as f_read:
            lines = f_read.readlines()

            del_targets = []

            for i in range(len(lines) - 1):
                for j in range(i, len(lines) - 1):
                    line_j = lines[j]
                    line_j1 = lines[j + 1]
                    
                    l_j = line_j.strip().split(" ")
                    l_j1 = line_j1.strip().split(" ")

                    cls_j = int(l_j[0])
                    cls_j1 = int(l_j1[0])

                    bbx_yolo_j = list(map(float, l_j[1:]))
                    bbx_yolo_j1 = list(map(float, l_j1[1:]))
                    bbx_voc_j = bbox_yolo_to_voc(imgsz, bbx_yolo_j)
                    bbx_voc_j1 = bbox_yolo_to_voc(imgsz, bbx_yolo_j1)
                    iou_j_j1 = cal_iou(bbx_voc_j, bbx_voc_j1)

                    if iou_j_j1 > iou_thr and cls_j != cls_j1 and cls_j in target_cls and cls_j1 in target_cls:
                        print(f_abs_path)
                        if cls_j == del_cls:
                            del_targets.append(line_j)
                        elif cls_j1 == del_cls:
                            del_targets.append(line_j1)

        with open(f_abs_path_new, "w") as f_write:
            for line in lines:
                if line not in del_targets:
                    f_write.write(line)
            

def select_specific_images_and_labels(data_path):
    """
    选择符合要求的图片和标签
    """

    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    
    save_path = make_save_path(data_path, relative=".", add_str="selected")
    img_save_path = save_path + "/images"
    lbl_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    file_list = get_file_list(lbl_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        img_src_path = img_path + "/{}.jpg".format(fname)
        lbl_src_path = lbl_path + "/{}".format(f)
        img_dst_path = img_save_path + "/{}.jpg".format(fname)
        lbl_dst_path = lbl_save_path + "/{}".format(f)

        sitting_person_num = 0
        guardarea_num = 0

        with open(lbl_src_path, "r") as f_read:
            lines = f_read.readlines()

            for line in lines:
                l = line.strip().split(" ")
                cls = int(l[0])

                if cls == 0:
                    sitting_person_num += 1
                elif cls == 2:
                    guardarea_num += 1

        if sitting_person_num == 1 and guardarea_num == 1:
            shutil.copy(img_src_path, img_dst_path)
            shutil.copy(lbl_src_path, lbl_dst_path)


                

def remove_corrupt_image(data_path):

    save_path = make_save_path(data_path, relative=".", add_str="corrupt")

    file_list = get_file_list(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        img_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(img_abs_path)
        if img is None:
            shutil.move(img_abs_path, save_path)


def LNG_connected_component_analysis(img, area_low=100, area_high=1000, connectivity=8):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=connectivity)
    areas = stats[:, -1]  # stats[:, cv2.CC_STAT_AREA]
    areas_new = []
    h_new = []
    w_new = []
    for i in range(1, num_labels):
        if areas[i] > area_low and areas[i] < area_high:
            areas_new.append(areas[i])

        if areas[i] > area_low / 2:
            h_new.append(stats[i, cv2.CC_STAT_HEIGHT])
            w_new.append(stats[i, cv2.CC_STAT_WIDTH])

    if len(areas_new) == 0:
        return num_labels, 0, 0, 0
    
    area_avg = np.mean(areas_new)
    h_avg = np.mean(h_new)
    w_avg = np.mean(w_new)
    return num_labels, area_avg, h_avg, w_avg
            


def cal_area_ratio_of_sepecific_color(img, lower=(0, 0, 100), upper=(80, 80, 255), apply_mask=True):
    # h_crop = 68
    h_crop = 0
    assert h_crop <= 68, "h_crop must be less than 68!"
    img = img[h_crop:, :]
    img_orig = img.copy()
    if apply_mask:
        # 640 * 512, (620, 68) (640, 445)
        imgsz = img.shape[:2]
        # rh1, rh2 = 68 / 512, 445 / 512
        rh1, rh2 = (68 - h_crop) / imgsz[0], (512 - h_crop - 68) / imgsz[0]
        rw1, rw2 = 620 / imgsz[1], 640 / imgsz[1]
        img[int(rh1 * imgsz[0]):int(rh2 * imgsz[0]), int(rw1 * imgsz[1]):int(rw2 * imgsz[1])] = (0, 0, 0)
        
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower, upper)

    # # 计算指定颜色区域的像素数量
    # color_pixel_count = cv2.countNonZero(mask)


    # 可视化结果（可选）
    # 将掩码应用到原图上，显示提取的颜色区域
    result = cv2.bitwise_and(img, img, mask=mask)
    # result = np.uint8(result)


    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # gray2 = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(r"D:\Gosion\Projects\GuanWangLNG\cewen\wendutu_result\gray2.jpg", gray2)
    
    _, bw_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # cv2.imwrite(r"D:\Gosion\Projects\GuanWangLNG\cewen\wendutu_result\bw_image.jpg", bw_image)


    num_labels, area_avg, h_avg, w_avg = LNG_connected_component_analysis(bw_image, area_low=100, area_high=1000, connectivity=8)
    print(f"num_labels: {num_labels}, area_avg: {area_avg}, h_avg: {h_avg}, w_avg: {w_avg}")

 
    # white_area = np.sum(bw_image == 255)
    white_area = cv2.countNonZero(bw_image)

    # 计算图像的总像素数量
    total_pixel_count = img.shape[0] * img.shape[1]

    # 计算指定颜色区域占整张图像的面积比例
    color_area_ratio = white_area / total_pixel_count * 100

    # 输出结果
    # print(f"指定颜色区域的像素数量: {white_area}")
    # print(f"图像的总像素数量: {total_pixel_count}")
    print(f"指定颜色区域占整张图像的面积比例: {color_area_ratio:.6f}\n")

    rs = "{}_{}_{:.6f}".format(white_area, total_pixel_count, color_area_ratio)


    # # 显示原图和结果
    # cv2.imshow("Original Image", img)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Result", result)

    # # 等待按键并关闭窗口
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite(r"D:\Gosion\Projects\GuanWangLNG\cewen\wendutu_result\result.jpg", result)

    # res = np.hstack((img_orig, result), dtype=np.uint8)
    
    # return res, color_area_ratio, rs
    return result, color_area_ratio, rs

                    
def extract_parabolic_curve_area(img_path):
    # 读取图片
    image = cv2.imread(img_path)
    if image is None:
        print("Error: 图片未找到，请检查路径。")
        exit()

    # 获取图片的高度和宽度
    height, width = image.shape[:2]

    # 定义抛物线的参数 (y = a*(x - x0)^2 + y0)
    a = 0.001  # 抛物线的开口大小
    x0 = width // 2  # 抛物线的原点 x 坐标（可调整）
    # y0 = height // 2  # 抛物线的原点 y 坐标（可调整）
    y0 = height // 4  # 抛物线的原点 y 坐标（可调整）

    # 定义距离抛物线的上下边缘的像素数
    N = 50

    # 创建一个与图片大小相同的空白掩码
    mask = np.zeros_like(image)

    # 遍历每一列
    for x in range(width):
        # 计算抛物线的 y 值（以 (x0, y0) 为原点）
        y_parabola = int(a * (x - x0) ** 2 + y0)
        
        # 计算上边缘和下边缘的 y 值
        y_upper = max(0, y_parabola - N)
        y_lower = min(height - 1, y_parabola + N)
        
        # 在掩码上绘制矩形区域
        mask[y_upper:y_lower, x] = 255

    # 使用掩码提取感兴趣的区域
    result = cv2.bitwise_and(image, mask)

    # 显示原始图片和提取的区域
    cv2.imshow('Original Image', image)
    cv2.imshow('Extracted Region', result)

    # 等待按键并关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def adjust_putText():
    """
    cv2.putText()遇到了问题：原本是将字符显示在目标框的上方，但是有的目标框靠近图像上边缘导致字符超出图片显示不了。
    可以通过检测目标框的位置来判断是否将文本显示在框的上方或下方。如果目标框靠近图像的上边缘，将文本显示在框的下方；否则，显示在框的上方。
    """
    pass
    # (text_width, text_height), baseline = cv2.getTextSize(str(dis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    # if b[1] - text_height - 10 < 0:  # 如果文本超出图像上边缘
    #     text_y = b[3] + text_height + 8  # 将文本显示在框的下方
    # else:
    #     text_y = b[1] - 5  # 将文本显示在框的上方
    # cv2.putText(img, "{:.2f}".format(dis), (b[0], text_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)


def corner_detect(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # find Harris corners
    # gray = np.float32(gray)
    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # dst = cv2.dilate(dst,None)
    # ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    # dst = np.uint8(dst)
    # # find centroids
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # # define the criteria to stop and refine the corners
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)
    # # Now draw them
    # res = np.hstack((centroids, corners))
    # res = np.intp(res)
    # img[res[:, 1],res[:, 0]] = [0, 0, 255]
    # img[res[:, 3],res[:, 2]] = [0, 255, 0]
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # =======================================================================================
    # https://www.cnblogs.com/GYH2003/articles/GYH-PythonOpenCV-FeatureDetection-Corner.html
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)  # 转换为浮点型
    # dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)  # Harris 角点检测
    t1 = time.time()
    dst = cv2.cornerHarris(gray, blockSize=10, ksize=7, k=0.04)  # Harris 角点检测
    t2 = time.time()
    print("Harris角点检测耗时：", t2 - t1)

    r, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)  # 二值化阈值处理
    dst = np.uint8(dst)  # 转换为整型
    cv2.imshow('dst', dst)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # 不同的连通域赋予不同的颜色
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 256)
        output[:, :, 1][mask] = np.random.randint(0, 256)
        output[:, :, 2][mask] = np.random.randint(0, 256)

    # 标记角点
    dst = cv2.dilate(dst, None)  # 膨胀操作，增强角点标记
    print(dst > 0.01 * dst.max())
    img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 将角点标记为红色

    # 显示结果
    cv2.imshow('Harris Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # =======================================================================================

    # # Shi-Tomasi 角点检测
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    # corners = np.int0(corners)  # 转换为整数坐标

    # # 标记角点
    # for corner in corners:
    #     x, y = corner.ravel()
    #     cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # 用绿色圆点标记角点

    # # 显示结果
    # cv2.imshow('Shi-Tomasi Corners', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # FAST 角点检测
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # fast = cv2.FastFeatureDetector_create()  # 创建 FAST 检测器
    # keypoints = fast.detect(gray, None)  # 检测角点

    # # 标记角点
    # image_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))  # 用绿色标记角点

    # # 显示结果
    # cv2.imshow('FAST Corners', image_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # =======================================================================================
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    # img_gray = np.float32(img_gray)  # 转换为浮点类型
    # dst = cv2.cornerHarris(img_gray, 10, 7, 0.04)  # 查找哈里斯角
    # r, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)  # 二值化阈值处理
    # dst = np.uint8(dst)  # 转换为整型
    # r, l, s, cxys = cv2.connectedComponentsWithStats(dst)  # 查找质点坐标
    # cif = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)  # 定义优化查找条件
    # corners = cv2.cornerSubPix(img_gray, np.float32(cxys), (5, 5), (-1, -1), cif)  # 执行优化查找
    # res = np.hstack((cxys, corners))  # 堆叠构造新数组，便于标注角
    # res = np.intp(res)  # 转换为整型
    # img[res[:, 1], res[:, 0]] = [0, 0, 255]  # 将哈里斯角对应像素设置为红色
    # img[res[:, 3], res[:, 2]] = [0, 255, 0]  # 将优化结果像素设置为绿色
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # =======================================================================================


def draw_triangle_mask(img, areas):
    """
    在图像中划定一个三角形区域，并将区域内的像素值设置为0。
    
    参数:
        image: 输入图像。
        vertices: 三角形的三个顶点坐标，格式为[[x1, y1], [x2, y2], [x3, y3]]。
        [[100, 100], [200, 300], [300, 100]]
    
    返回:
        masked_image: 处理后的图像。
    """
    mask = 255 * np.ones_like(img)
    
    for a in areas:
        a = np.array(a, dtype=np.int32)
        mask = cv2.fillPoly(mask, [a], (0, 0, 0))
    
    return mask


def apply_mask_area(img, flag=True):
    # 去除干扰，屏蔽不是roi区域，roi区域是激光线附近的区域
    if flag:
        imgsz = img.shape[:2]
        areas = [
            [[0, 0], [0, int(0.5 * imgsz[0])], [int(0.25 * imgsz[1]), 0]],
            [[int(0.75 * imgsz[1]), 0], [imgsz[1], 0], [imgsz[1], int(0.5 * imgsz[0])]],
            [[int(0.5 * imgsz[1]), int(0.35 * imgsz[0])], [0, imgsz[0]], [imgsz[1], imgsz[0]]]
        ]
        masked_image = draw_triangle_mask(img, areas)
        img = cv2.bitwise_and(img, masked_image)

    return img


def draw_rectangle_mask(img, areas):
    """
    在图像中划定一个三角形区域，并将区域内的像素值设置为0。
    
    参数:
        image: 输入图像。
        vertices: 三角形的三个顶点坐标，格式为[[x1, y1], [x2, y2], [x3, y3]]。
        [[100, 100], [200, 300], [300, 100]]
    
    返回:
        masked_image: 处理后的图像。
    """
    mask = 255 * np.ones_like(img)
    
    for a in areas:
        a = np.array(a, dtype=np.int32)
        mask = cv2.fillPoly(mask, [a], (0, 0, 0))
    
    return mask


def apply_mask_area_v2(img, miny, maxy, flag=True):
    # 去除干扰，屏蔽不是roi区域，roi区域是激光线附近的区域
    if flag:
        imgsz = img.shape[:2]
        areas = [
            [[0, 0], [imgsz[1], 0], [imgsz[1], miny], [0, miny]],
            [[0, maxy], [imgsz[1], maxy], [imgsz[1], imgsz[0]], [0, imgsz[0]]]
        ]
        masked_image = draw_rectangle_mask(img, areas)
        img = cv2.bitwise_and(img, masked_image)

    return img

    
def corner_detect_test():
    expand_pixels = 25
    dis_thresh = 30
    min_inliers = 5
    data_path = r"D:\Gosion\Projects\006.Belt_Torn_Det\data\video\video_frames\cropped\20250308_frames_merged"
    save_path = make_save_path(data_path, add_str="Harris_Corners_Result")
    file_list = get_file_list(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        imgsz = img.shape[:2]

        img_ama = apply_mask_area(img, flag=True)

        # =======================================================================================
        # https://www.cnblogs.com/GYH2003/articles/GYH-PythonOpenCV-FeatureDetection-Corner.html
        gray = cv2.cvtColor(img_ama, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)  # 转换为浮点型
        # dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)  # Harris 角点检测
        dst = cv2.cornerHarris(gray, blockSize=10, ksize=7, k=0.04)  # Harris 角点检测

        r, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)  # 二值化阈值处理
        dst = np.uint8(dst)

        # 标记角点
        dst = cv2.dilate(dst, None)  # 膨胀操作，增强角点标记
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # # 不同的连通域赋予不同的颜色
        # output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        # for i in range(1, num_labels):
        #     mask = labels == i
        #     output[:, :, 0][mask] = np.random.randint(0, 256)
        #     output[:, :, 1][mask] = np.random.randint(0, 256)
        #     output[:, :, 2][mask] = np.random.randint(0, 256)

        # for i in range(len(centroids_new)):
        #     cv2.circle(img, (int(centroids_new[i][0]), int(centroids_new[i][1])), 5, (0, 0, 255), -1)
        # cv2.imwrite(save_path + "/{}_centroids_new_0.jpg".format(fname), img)

        # centroids_new = img_points_coord_to_cardesian_coord(imgsz, np.array(centroids_new))
        # polynomial, poly_filtered = polyfit_v2(centroids_new)
        # y_plot_initial = polynomial(centroids_new[:, 0])
        # y_plot_filtered = poly_filtered(centroids_new[:, 0])

        # for i in range(len(centroids_new)):
        #     cv2.circle(img, (int(centroids_new[i][0]), int(centroids_new[i][1])), 5, (0, 255, 255), -1)
        # cv2.imwrite(save_path + "/{}_centroids_new_1.jpg".format(fname), img)

        centroids_new = sorted(centroids[1:], key=lambda x: x[0])
        stats_new = sorted(stats[1:], key=lambda x: x[0])
        wides = []

        if len(centroids_new) >= min_inliers:
            centroids_new = np.array(centroids_new)
            best_poly= polyfit_RANSAC(centroids_new, degree=2, n_iters=100, threshold=5, min_inliers=min_inliers)
            fit_y = best_poly(centroids_new[:, 0])

            # center_y = np.array([b[1] + b[3] / 2 for b in stats_new])
            # dis_fc_y = np.abs(center_y - fit_y)
            # dis_median = sorted(dis_fc_y)[len(fit_y) // 2]
            # print("fit_y: {}".format(fit_y))
            # print("center_y: {}".format(center_y))
            # print("dis_fc_y: {}".format(dis_fc_y))
            # print("dis_median: {}".format(dis_median))

            for i in range(len(stats_new)):
                bi = stats_new[i][:-1]
                bi_center = [bi[0] + bi[2] / 2, bi[1] + bi[3] / 2]
                bi_expand = [bi[0] - expand_pixels, bi[1] - round(1.25 * expand_pixels), bi[0] + bi[2] + expand_pixels, bi[1] + bi[3] + round(1.25 * expand_pixels)]
                fit_y_i = best_poly(bi_center[0])

                d = abs(bi_center[1] - fit_y_i)
                if d <= dis_thresh:
                    wides.append(bi[2] - 2)  # 因为前面进行了膨胀，所以需要减2
                    cv2.circle(img, (int(bi_center[0]), int(bi_center[1])), 5, (255, 255, 255), -1)
                    cv2.rectangle(img, (bi_expand[0], bi_expand[1]), (bi_expand[2], bi_expand[3]), (255, 0, 255), 2)
        else:
            for i in range(len(stats_new)):
                bi = stats_new[i][:-1]
                bi_center = [bi[0] + bi[2] / 2, bi[1] + bi[3] / 2]
                bi_expand = [bi[0] - expand_pixels, bi[1] - round(1.25 * expand_pixels), bi[0] + bi[2] + expand_pixels, bi[1] + bi[3] + round(1.25 * expand_pixels)]

                wides.append(bi[2] - 2)  # 因为前面进行了膨胀，所以需要减2
                cv2.circle(img, (int(bi_center[0]), int(bi_center[1])), 5, (255, 255, 255), -1)
                cv2.rectangle(img, (bi_expand[0], bi_expand[1]), (bi_expand[2], bi_expand[3]), (255, 0, 255), 2)

        cv2.imwrite(save_path + "/{}_centroids_new_2.jpg".format(fname), img)
        print("fname: {} wides: {}".format(fname, wides))


        # for i in range(len(centroids_new)):
        #     cv2.circle(img, (int(centroids_new[i, 0]), int(centroids_new[i, 1])), 5, (0, 0, 255), -1)
        #     # cv2.circle(img, (int(centroids_new[i, 0]), int(y_plot_initial[i])), 3, (0, 255, 255), -1)
        #     # cv2.circle(img, (int(centroids_new[i, 0]), int(y_plot_filtered[i])), 6, (255, 255, 0), -1)


        # img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 将角点标记为红色

        # f_dst_path = save_path + "/{}".format(f)
        # cv2.imwrite(f_dst_path, img)

        # 显示结果
        # cv2.imshow('Harris Corners', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # =======================================================================================


def img_points_coord_to_cardesian_coord(imgsz, points):
    """imgsz: [h, w]"""
    points[1] = imgsz[0] - points[1]

    return points


def polyfit_v2(points, degree=2):
    # 多项式拟合（二次）
    coefficients = np.polyfit(points[:, 0], points[:, 1], degree)
    polynomial = np.poly1d(coefficients)

    # 计算点到曲线的距离
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    distances = np.abs(y_vals - polynomial(x_vals)) / np.sqrt(1 + np.polyval(np.polyder(coefficients), x_vals)**2)

    # 过滤离群点（阈值=1.0）
    threshold = 15.0
    filtered_points = points[distances <= threshold]

    # 重新拟合曲线
    if len(filtered_points) > 0:
        coeff_filtered = np.polyfit(filtered_points[:, 0], filtered_points[:, 1], degree)
        poly_filtered = np.poly1d(coeff_filtered)
    else:
        poly_filtered = polynomial  # 若无点保留，使用原始曲线

    # y_plot_initial = polynomial(points[:, 0])
    # y_plot_filtered = poly_filtered(points[:, 0])

    return polynomial, poly_filtered


def polyfit_RANSAC(points, degree=2, n_iters=100, threshold=5, min_inliers=5):
    best_model=None
    best_inliers=[]
    
    for i in range(n_iters):
        sample = random.sample(list(points), min_inliers)
        sample_x = np.array([p[0] for p in sample])
        sample_y = np.array([p[1] for p in sample])
        # 多项式拟合（二次）
        try:
            coefficients = np.polyfit(sample_x, sample_y, degree)
            polynomial = np.poly1d(coefficients)
        except:
            continue

        # 计算点到曲线的距离
        x_vals = points[:, 0]
        y_vals = points[:, 1]
        distances = np.abs(y_vals - polynomial(x_vals)) / np.sqrt(1 + np.polyval(np.polyder(coefficients), x_vals)**2)

        # 统计内点
        inliers = points[distances < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = polynomial

    # 使用最佳内点重新拟合曲线
    if best_model is not None:
        best_coeff = np.polyfit(best_inliers[:, 0], best_inliers[:, 1], 2)
        best_poly = np.poly1d(best_coeff)
    else:
        # 多项式拟合（二次）
        coefficients = np.polyfit(points[:, 0], points[:, 1], degree)
        polynomial = np.poly1d(coefficients)

        # 计算点到曲线的距离
        x_vals = points[:, 0]
        y_vals = points[:, 1]
        distances = np.abs(y_vals - polynomial(x_vals)) / np.sqrt(1 + np.polyval(np.polyder(coefficients), x_vals)**2)

        # 过滤离群点（阈值=1.0）
        filtered_points = points[distances <= threshold]

        # 重新拟合曲线
        if len(filtered_points) > 0:
            coeff_filtered = np.polyfit(filtered_points[:, 0], filtered_points[:, 1], degree)
            best_poly = np.poly1d(coeff_filtered)
        else:
            best_poly = polynomial  # 若无点保留，使用原始曲线

    return best_poly



def polyfit_test():
    # 生成示例点坐标 (x, y)
    points = np.array([
        [1, 2], [2, 3], [3, 5], [4, 7], [5, 11],
        [6, 15], [7, 20], [8, 25], [9, 30], [10, 35]
    ], dtype=np.float32)

    # 定义图像尺寸和坐标系映射参数
    img_width, img_height = 800, 600
    x_min, x_max = 0, 12  # 扩展x范围以便显示
    y_min, y_max = 0, 40  # 扩展y范围

    # 数据坐标到图像坐标的转换函数
    def data_to_img_coord(x, y):
        scale_x = img_width / (x_max - x_min)
        scale_y = img_height / (y_max - y_min)
        img_x = int((x - x_min) * scale_x)
        img_y = int(img_height - (y - y_min) * scale_y)  # OpenCV的y轴向下
        return img_x, img_y

    # 创建空白图像
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # 白色背景

    # 绘制原始点
    for x, y in points:
        img_x, img_y = data_to_img_coord(x, y)
        cv2.circle(img, (img_x, img_y), 5, (0, 0, 255), -1)  # 红色点

    # 多项式拟合（二次）
    degree = 2
    coefficients = np.polyfit(points[:, 0], points[:, 1], degree)
    polynomial = np.poly1d(coefficients)

    # 计算点到曲线的距离
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    distances = np.abs(y_vals - polynomial(x_vals)) / np.sqrt(1 + np.polyval(np.polyder(coefficients), x_vals)**2)

    # 过滤离群点（阈值=1.0）
    threshold = 1.0
    filtered_points = points[distances <= threshold]

    # 重新拟合曲线
    if len(filtered_points) > 0:
        coeff_filtered = np.polyfit(filtered_points[:, 0], filtered_points[:, 1], degree)
        poly_filtered = np.poly1d(coeff_filtered)
    else:
        poly_filtered = polynomial  # 若无点保留，使用原始曲线

    # 在图像上绘制拟合曲线
    x_plot = np.linspace(x_min, x_max, 100)
    y_plot_initial = polynomial(x_plot)
    y_plot_filtered = poly_filtered(x_plot)

    # 绘制原始拟合曲线（蓝色）
    for i in range(len(x_plot)-1):
        x1, y1 = data_to_img_coord(x_plot[i], y_plot_initial[i])
        x2, y2 = data_to_img_coord(x_plot[i+1], y_plot_initial[i+1])
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 绘制过滤后的曲线（绿色）
    for i in range(len(x_plot)-1):
        x1, y1 = data_to_img_coord(x_plot[i], y_plot_filtered[i])
        x2, y2 = data_to_img_coord(x_plot[i+1], y_plot_filtered[i+1])
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 绘制过滤后的点（绿色）
    for x, y in filtered_points:
        img_x, img_y = data_to_img_coord(x, y)
        cv2.circle(img, (img_x, img_y), 5, (0, 255, 0), -1)

    # 添加文本说明
    cv2.putText(img, f"Original Equation: {polynomial}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, f"Filtered Equation: {poly_filtered}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "Red: Original Points | Blue: Initial Fit | Green: Filtered", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 显示结果
    cv2.imshow("Curve Fitting with Outlier Removal", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_specific_color_area_ratio(img, color="green"):
    """
    https://news.sohu.com/a/569474695_120265289
    https://blog.csdn.net/yuan2019035055/article/details/140495066
    """
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
        raise ValueError("Invalid color. Please choose from 'black', 'gray', 'white', 'red', 'orange', 'yellow','green', 'cyan', 'blue', 'purple'.")

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color == "red":
        mask0 = cv2.inRange(hsv_image, lower0, upper0)
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        mask = cv2.bitwise_or(mask0, mask1)
    else:
        mask = cv2.inRange(hsv_image, lower, upper)

    # 可视化结果（可选）
    # 将掩码应用到原图上，显示提取的颜色区域
    result = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    _, bw_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 下面两个方法结果一致
    # white_area = np.sum(bw_image == 255)
    white_area = cv2.countNonZero(bw_image)

    # 计算图像的总像素数量
    total_pixel_count = img.shape[0] * img.shape[1]

    # 计算指定颜色区域占整张图像的面积比例
    r = white_area / total_pixel_count * 100

    return r


def classify_image_by_brightness(data_path, show_brightness=False):
    """
    判断图像的亮度是否超过阈值，并返回对应的类别。
    """
    save_path = make_save_path(data_path, ".", "classify_image_by_brightness")
    paths = []
    for i in range(10):
        pathi = save_path + "/{}".format(i)
        paths.append(pathi)
        os.makedirs(pathi, exist_ok=True)
        

    file_list = get_file_list(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        brightness = cal_brightness_v2(img)
        if show_brightness:
            print("{}: {}".format(fname, brightness))
        else:
            # if brightness > 225:
            #     f_dst_path = paths[0] + "/{}".format(f)
            #     shutil.move(f_abs_path, f_dst_path)
            # elif brightness > 200:
            #     f_dst_path = paths[1] + "/{}".format(f)
            #     shutil.move(f_abs_path, f_dst_path)
            # elif brightness > 175:
            #     f_dst_path = paths[2] + "/{}".format(f)
            #     shutil.move(f_abs_path, f_dst_path)
            # elif brightness > 150:
            #     f_dst_path = paths[3] + "/{}".format(f)
            #     shutil.move(f_abs_path, f_dst_path)
            # elif brightness > 125:
            #     f_dst_path = paths[4] + "/{}".format(f)
            #     shutil.move(f_abs_path, f_dst_path)
            # elif brightness > 100:
            #     f_dst_path = paths[5] + "/{}".format(f)
            #     shutil.move(f_abs_path, f_dst_path)
            # elif brightness > 75:
            #     f_dst_path = paths[6] + "/{}".format(f)
            #     shutil.move(f_abs_path, f_dst_path)
            # elif brightness > 55:
            #     f_dst_path = paths[7] + "/{}".format(f)
            #     shutil.move(f_abs_path, f_dst_path)
            # elif brightness > 30:
            #     f_dst_path = paths[8] + "/{}".format(f)
            #     shutil.move(f_abs_path, f_dst_path)
            # else:
            #     f_dst_path = paths[9] + "/{}".format(f)
            #     shutil.move(f_abs_path, f_dst_path)

            if brightness > 169 and brightness < 172:
                f_dst_path = paths[0] + "/{}".format(f)
                shutil.move(f_abs_path, f_dst_path)
            else:
                f_dst_path = paths[1] + "/{}".format(f)
                shutil.move(f_abs_path, f_dst_path)


def bilateral_filter(img, d=9, csigma=75, ssigma=75):
    filtered = cv2.bilateralFilter(img, d, csigma, ssigma)
    return filtered


def opencv_imread_filename_contain_chinese(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def opencv_imwrite_filename_contain_chinese(img_path, img):
    cv2.imencode('.jpg', img)[1].tofile(img_path)


def z_score(x, mean, std):
    return (x - mean) / std


def expand_bbox(bi, imgsz, expand_pixels):
    """ imgsz: [h, w] """
    x1 = bi[0] - expand_pixels
    y1 = bi[1] - round(1.25 * expand_pixels)
    x2 = bi[0] + bi[2] + expand_pixels
    y2 = bi[1] + bi[3] + round(1.25 * expand_pixels)

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > imgsz[1]:
        x2 = imgsz[1]
    if y2 > imgsz[0]:
        y2 = imgsz[0]

    bi_expand = [x1, y1, x2, y2]

    return bi_expand



def corner_detect_v2(img, expand_pixels=25, brightness_thresh=2, dilate_pixels=2):
    imgsz = img.shape[:2]
    dst_bbx_pts = []
    wides = []
    
    # =======================================================================================
    # https://www.cnblogs.com/GYH2003/articles/GYH-PythonOpenCV-FeatureDetection-Corner.html
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 根据实验，基本全黑（镜头套了盖子）的图误报很多，这种图和密密麻麻的黑点等类似的噪声图相似，对角点检测有影响，导致误报非常多
    # 对于这种情况，为了减少误报，去除这种图的影响
    brt= np.mean(gray)
    if brt < brightness_thresh or brt > (255 - brightness_thresh):
        return img, dst_bbx_pts, wides

    gray = np.float32(gray)  # 转换为浮点型
    # dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)  # Harris 角点检测
    dst = cv2.cornerHarris(gray, blockSize=10, ksize=7, k=0.04)  # Harris 角点检测

    r, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)  # 二值化阈值处理
    dst = np.uint8(dst)

    # 标记角点
    dst = cv2.dilate(dst, None)  # 膨胀操作，增强角点标记
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    centroids_new = sorted(centroids[1:], key=lambda x: x[0])
    stats_new = sorted(stats[1:], key=lambda x: x[0])

    for i in range(len(stats_new)):
        bi = stats_new[i][:-1]
        bi_center = [bi[0] + bi[2] / 2, bi[1] + bi[3] / 2]
        bi_expand = expand_bbox(bi, imgsz, expand_pixels)

        torn_dis = bi[2] - dilate_pixels  # 因为前面进行了膨胀，所以需要减dilate_pixels
        wides.append(torn_dis)
        p_draw_i = [bi[0] + dilate_pixels / 2, bi[1] + bi[3] / 2]
        p_draw_i_1 = [bi[0] + bi[2] - dilate_pixels / 2, bi[1] + bi[3] / 2]
        dst_bp = [bi_expand, [p_draw_i, p_draw_i_1]]  # dst_bp： dst_bbox_points
        dst_bbx_pts.append(dst_bp)

        cv2.circle(img, (round(p_draw_i[0]), round(p_draw_i[1])), 5, (255, 56, 56), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (round(p_draw_i_1[0]), round(p_draw_i_1[1])), 5, (255, 157, 151), -1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (bi_expand[0], bi_expand[1]), (bi_expand[2], bi_expand[3]), (255, 0, 255), 2)

        (text_width, text_height), baseline = cv2.getTextSize(str(torn_dis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        if bi_expand[1] - text_height - 10 < 0:  # 如果文本超出图像上边缘
            text_y = bi_expand[3] + text_height + 8  # 将文本显示在框的下方
        else:
            text_y = bi_expand[1] - 5  # 将文本显示在框的上方

        cv2.putText(img, "{:.2f}".format(torn_dis), (bi_expand[0], text_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    return img, dst_bbx_pts, wides


def group_elements(arr, thr=15):
    if not arr:  # 处理空数组
        return []
    groups = [[arr[0]]]  # 初始化第一个组
    for num in arr[1:]:
        # 如果当前元素与当前组最后一个元素的差超过15，创建新组
        if num - groups[-1][-1] > thr:
            groups.append([num])
        else:
            groups[-1].append(num)
    return groups


def group_numpy_array(arr, thr=15):
    if arr.size == 0:  # 处理空数组
        return []
    if arr.ndim != 1:  # 确保输入为一维数组
        raise ValueError("输入必须是一维数组")
    
    # 计算相邻元素差值并找到分割点
    diffs = np.diff(arr)
    split_indices = np.where(diffs > thr)[0] + 1
    
    # 分割原数组为多个组
    groups = np.split(arr, split_indices)
    
    # # 转换为列表形式输出（可选）
    # grouped_coordinates = [group.tolist() for group in groups]
    # return grouped_coordinates

    return groups


def median_filter_1d(res_list, k=15):
    """
    中值滤波
    """
    edge = int(k / 2)
    new_res = res_list.copy()
    for i in range(len(res_list)):
        if i <= edge or i >= len(res_list) - edge - 1:
            pass
        else:
            medianv = np.median(res_list[i - edge:i + edge + 1])
            if new_res[i] != medianv:
                new_res[i] = medianv
            else:
                pass

    return new_res


def merge_sorted_unique(arr1, arr2):
    merged = np.concatenate((arr1, arr2))
    return np.unique(merged)


def merge_keep_order(arr1, arr2):
    merged = np.concatenate((arr1, arr2))
    _, idx = np.unique(merged, return_index=True)
    return merged[np.sort(idx)]


def get_bbx_btd(g, imgsz, expand_pixels=10, y_draw=25):
    x1 = np.min(g) - expand_pixels
    y1 = y_draw
    x2 = np.max(g) + expand_pixels
    y2 = imgsz[0] - y_draw

    bbx_org = [x1 + expand_pixels, y1, x2 - expand_pixels, y2]

    if x1 < 0:
        x1 = 0
    if x2 > imgsz[1]:
        x2 = imgsz[1]

    bbx_exp = [x1, y1, x2, y2]
    return bbx_org, bbx_exp


def connected_components_analysis_btd(img, connectivity=8, area_thr=100, h_thr=8, w_thr=8):
    """
    stats: [x, y, w, h, area]
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=connectivity)
    
    stats_new = []
    stats_new_idx = []
    areas = stats[:, -1]  # stats[:, cv2.CC_STAT_AREA]
    for i in range(1, num_labels):
        # if areas[i] > area_thr and stats[i, 2] / stats[i, 3] < 2:
        #     labels[labels == i] = 0
        # else:
        #     if stats[i, 2] < w_thr or stats[i, 3] < h_thr:
        #         labels[labels == i] = 0
        #     else:
        #         stats_new.append(stats[i])
        #         stats_new_idx.append(i)
        # if areas[i] < area_thr:
        #     labels[labels == i] = 0
        # else:
        #     if areas[i] > area_thr and stats[i, 2] > stats[i, 3] * 1.5:
        #         stats_new.append(stats[i])
        #         stats_new_idx.append(i)
        #     elif stats[i, 2] < w_thr or stats[i, 3] < h_thr:
        #         labels[labels == i] = 0

        if areas[i] > area_thr and stats[i, 2] > stats[i, 3] * 2:
            stats_new.append(stats[i])
            stats_new_idx.append(i)

    # 不同的连通域赋予不同的颜色
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 256)
        output[:, :, 1][mask] = np.random.randint(0, 256)
        output[:, :, 2][mask] = np.random.randint(0, 256)


    four_points = []
    for si in stats_new_idx:
        mask = np.uint8(labels == si)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 1, "len(contours) != 1"
        contour = contours[0]
        points = contour.reshape(-1, 2)
        top_left = points[np.argmin(points[:, 0] + points[:, 1])]
        bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
        bottom_left = points[np.argmin(points[:, 0] - points[:, 1])]
        top_right = points[np.argmax(points[:, 0] - points[:, 1])]
        four_points.append([top_left, top_right, bottom_right, bottom_left])

    stats_new = sorted(stats_new, key=lambda x: x[0])
    four_points = sorted(four_points, key=lambda x: x[0][0])
    
    return output, num_labels, labels, stats, centroids, stats_new, four_points


def get_draw_points(tr_i, tl_i_1, x1, y1, x2, y2, w1, m1v, m2v, y_gap, gap_thresh):
    """
    tr_i: top_right_i
    tl_i_1: top_left_i_1
    m1v: wides_m1_value
    m2v: wides_m2_value
    """
    points_draw_i = np.empty(shape=(0, 2), dtype=np.int32)
    points_draw_i_1 = np.empty(shape=(0, 2), dtype=np.int32)
    if abs(m1v - m2v) <= gap_thresh:
        if y_gap <= gap_thresh:
            points_draw_i = np.append(points_draw_i, tr_i.reshape(1, 2), axis=0)
            points_draw_i_1 = np.append(points_draw_i_1, tl_i_1.reshape(1, 2), axis=0)
        else:
            points_draw_i = np.append(points_draw_i, np.array([[x1 + w1, y1]]), axis=0)
            points_draw_i_1 = np.append(points_draw_i_1, np.array([[x2, y2]]), axis=0)
    else:
        points_draw_i = np.append(points_draw_i, np.array([[x1 + w1, y1]]), axis=0)
        points_draw_i_1 = np.append(points_draw_i_1, np.array([[x2, y2]]), axis=0)
    
    points_draw_i = np.squeeze(points_draw_i, axis=0)
    points_draw_i_1 = np.squeeze(points_draw_i_1, axis=0)

    return points_draw_i, points_draw_i_1


def get_connected_components_analysis_bbx(img0, img, analysis_output, expand_pixels=10, y_draw=25, draw_circle=False):
    imgsz = img.shape[:2]
    stats_new = analysis_output[-2]
    four_points = analysis_output[-1]

    dst_bbx_pts = []
    wides = []

    for i in range(len(stats_new) - 1):
        x1, y1, w1, h1, area1 = stats_new[i]
        x2, y2, w2, h2, area2 = stats_new[i + 1]
        
        x1_draw = round(x1 + w1 - expand_pixels)
        y1_draw = round((y1 + y2) / 2 - expand_pixels)
        x2_draw = round(x2 + expand_pixels)
        y2_draw = round((y1 + y2) / 2 + expand_pixels * 2)
        if x1_draw < 0: x1_draw = 0
        if x2_draw < 0: x2_draw = 0
        if x1_draw > imgsz[1]: x1_draw = imgsz[1]
        if x2_draw > imgsz[1]: x2_draw = imgsz[1]
        if y1_draw < 0: y1_draw = 0
        if y2_draw < 0: y2_draw = 0
        if y1_draw > imgsz[0]: y1_draw = imgsz[0]
        if y2_draw > imgsz[0]: y2_draw = imgsz[0]

        if y2_draw <= y1_draw or x2_draw <= x1_draw: continue
        if x2_draw - x1_draw > 150: continue
        
        wides_m1_value = abs(x2 - (x1 + w1))
        
        top_right_i = four_points[i][1]
        top_left_i_1 = four_points[i + 1][0]
        y_gap = abs(top_right_i[1] - top_left_i_1[1])
        wides_m2_value = cal_distance(top_right_i, top_left_i_1)

        points_draw_i, points_draw_i_1 = get_draw_points(
            top_right_i, top_left_i_1, x1, y1, x2, y2, w1, wides_m1_value, wides_m2_value, y_gap, gap_thresh=30
        )

        # # 去除一些误报
        # x1_o = x1_draw + expand_pixels
        # x2_o = x2_draw - expand_pixels
        # c = img_src[:, x1_o:x2_o]
        # col_sums = np.sum(c, axis=0)

        # if col_sums.size == 0: continue
        # mean = np.mean(col_sums)
        # median = np.median(col_sums)
        # std = np.std(col_sums)

        # arr = np.array([mean, median, stats[0], stats[1]])
        # mean_arr = np.mean(arr)
        # std_arr = np.std(arr)
        # zscore = z_score(arr, mean_arr, std_arr)
        # x_coords = np.where(abs(zscore) < 3)
        # if x_coords[0].size > 0: continue

        # 去除误报
        bbx_ = [x1_draw, y1_draw, x2_draw, y2_draw]
        bbx_croped = img0[bbx_[1]:bbx_[3], bbx_[0]:bbx_[2]]
        if bbx_croped.size == 0: continue
        m = np.mean(bbx_croped)
        if m < 50 or m > 200: continue


        dis = cal_distance(points_draw_i, points_draw_i_1)
        wides.append(dis)

        # 为了和reduce_sum、z_score方法统一，固定框的y值. 2025.03.28, WJH.
        y1_draw = y_draw
        y2_draw = imgsz[0] - y_draw

        dst_bp = [[x1_draw, y1_draw, x2_draw, y2_draw], [list(points_draw_i), list(points_draw_i_1)]]
        dst_bbx_pts.append(dst_bp)

        # cv2.rectangle(img, (x1_draw, y1_draw), (x2_draw, y2_draw), (255, 0, 255), 2)
        # if draw_circle:
        #     cv2.circle(img, tuple(points_draw_i), 5, (255, 56, 56), -1, lineType=cv2.LINE_AA)
        #     cv2.circle(img, tuple(points_draw_i_1), 5, (255, 157, 151), -1, lineType=cv2.LINE_AA)
        # (text_width, text_height), baseline = cv2.getTextSize(str(dis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # if y1_draw - text_height - 10 < 0:  # 如果文本超出图像上边缘
        #     text_y = y2_draw + text_height + 8  # 将文本显示在框的下方
        # else:
        #     text_y = y1_draw - 5  # 将文本显示在框的上方

        # cv2.putText(img, "{:.2f}".format(dis), (x1_draw, text_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    return img, dst_bbx_pts, wides


def reduceSum_zScore(img0, img, reducesum_thr=100, zscore_thr=3, group_thr=15, flag_no_gap_abnormal=True, abnormal_mean_r=3):
    imgsz = img.shape[:2]
    col_sums = np.sum(img0, axis=0)

    dst_bbx_pts = []
    wides = []

    if col_sums.size > 0:
        mean_value = np.mean(col_sums)
        median_value = np.median(col_sums)
        std_value = np.std(col_sums)

        zscore = z_score(col_sums, mean_value, std_value)

        x_coords1 = np.where(col_sums < reducesum_thr)
        x_coords2 = np.where(abs(zscore) > zscore_thr)

        x_coords = merge_sorted_unique(x_coords1[0], x_coords2[0])
        groups = group_numpy_array(x_coords, thr=group_thr)

        for g in groups:
            if g.shape[0] > 0:
                bbx_org, bbx_exp = get_bbx_btd(g, imgsz, expand_pixels=10, y_draw=25)
                g_mean = np.mean(col_sums[bbx_org[0]:bbx_org[2]])

                if g_mean < reducesum_thr:
                    if abs(bbx_org[0] - 0) < 15 or abs(bbx_org[2] - imgsz[1]) < 15:
                        if g_mean < 5:
                            # cv2.rectangle(img, (bbx_exp[0], bbx_exp[1]), (bbx_exp[2], bbx_exp[3]), (255, 0, 255), 2)
                            dst_bp = [bbx_exp, [[bbx_org[0], imgsz[0] // 2], [bbx_org[2], imgsz[0] // 2]]]
                            dst_bbx_pts.append(dst_bp)
                            wides.append(abs(bbx_org[2] - bbx_org[0]))
                    else:
                        # cv2.rectangle(img, (bbx_exp[0], bbx_exp[1]), (bbx_exp[2], bbx_exp[3]), (255, 0, 255), 2)
                        dst_bp = [bbx_exp, [[bbx_org[0], imgsz[0] // 2], [bbx_org[2], imgsz[0] // 2]]]
                        dst_bbx_pts.append(dst_bp)
                        wides.append(abs(bbx_org[2] - bbx_org[0]))
                else:
                    if flag_no_gap_abnormal:
                        if g_mean > abnormal_mean_r * median_value:
                            dst_bp = [bbx_exp, [[bbx_org[0], imgsz[0] // 2], [bbx_org[2], imgsz[0] // 2]]]
                            dst_bbx_pts.append(dst_bp)
                            wides.append(abs(bbx_org[2] - bbx_org[0]))

    return img, dst_bbx_pts, wides


def merge_bbx(img, input_trad, input_pose, iou_thr=0.80, dis_thresh=15, draw_circle=False):
    imgsz = img.shape[:2]

    img_vis_trad, dst_bbx_pts_trad, wides_trad = input_trad
    img_vis_pose, dst_bbx_pts_pose, wides_pose = input_pose
    
    if len(dst_bbx_pts_trad) == 0 and len(dst_bbx_pts_pose) > 0:
        for d in dst_bbx_pts_pose:
            dis = cal_distance(d[1][0], d[1][1])
            # wides.append(dis)

            cv2.rectangle(img, (d[0][0], d[0][1]), (d[0][2], d[0][3]), (255, 0, 255), 2)
            if draw_circle:
                cv2.circle(img, tuple(map(round, d[1][0])), 5, (255, 56, 56), -1, lineType=cv2.LINE_AA)
                cv2.circle(img, tuple(map(round, d[1][1])), 5, (255, 157, 151), -1, lineType=cv2.LINE_AA)
            # cv2.putText(img, "{}".format(dis), (d[0][0], d[0][1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            (text_width, text_height), baseline = cv2.getTextSize(str(dis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            if d[0][1] - text_height - 10 < 0:  # 如果文本超出图像上边缘
                text_y = d[0][3] + text_height + 8  # 将文本显示在框的下方
            else:
                text_y = d[0][1] - 5  # 将文本显示在框的上方

            cv2.putText(img, "{:.2f}".format(dis), (d[0][0], text_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

        return img, dst_bbx_pts_pose, wides_pose
    
    elif len(dst_bbx_pts_trad) > 0 and len(dst_bbx_pts_pose) == 0:
        for d in dst_bbx_pts_trad:
            dis = cal_distance(d[1][0], d[1][1])
            # wides.append(dis)

            cv2.rectangle(img, (d[0][0], d[0][1]), (d[0][2], d[0][3]), (255, 0, 255), 2)
            if draw_circle:
                cv2.circle(img, tuple(map(round, d[1][0])), 5, (255, 56, 56), -1, lineType=cv2.LINE_AA)
                cv2.circle(img, tuple(map(round, d[1][1])), 5, (255, 157, 151), -1, lineType=cv2.LINE_AA)
            # cv2.putText(img, "{}".format(dis), (d[0][0], d[0][1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            (text_width, text_height), baseline = cv2.getTextSize(str(dis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            if d[0][1] - text_height - 10 < 0:  # 如果文本超出图像上边缘
                text_y = d[0][3] + text_height + 8  # 将文本显示在框的下方
            else:
                text_y = d[0][1] - 5  # 将文本显示在框的上方

            cv2.putText(img, "{:.2f}".format(dis), (d[0][0], text_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

        return img, dst_bbx_pts_trad, wides_trad
    else:
        dst1 = []
        dst2 = []
        wides = []
        for i in range(len(dst_bbx_pts_trad)):
            ib = dst_bbx_pts_trad[i][0]
            ip0 = dst_bbx_pts_trad[i][1][0]
            ip1 = dst_bbx_pts_trad[i][1][1]
            ip_center = [(ip0[0] + ip1[0]) / 2, (ip0[1] + ip1[1]) / 2]

            # 判断当前的框是否是唯一的，也就是dst_bbx_pts_trad有，dst_bbx_pts_pose没有
            check_unique_num = 0
            for j in range(len(dst_bbx_pts_pose)):
                jp0 = dst_bbx_pts_pose[j][1][0]
                jp1 = dst_bbx_pts_pose[j][1][1]
                jp_center = [(jp0[0] + jp1[0]) / 2, (jp0[1] + jp1[1]) / 2]
                dis_ip_jp = cal_distance(ip_center, jp_center)
                if dis_ip_jp > dis_thresh:
                    check_unique_num += 1

            for j in range(len(dst_bbx_pts_pose)):
                jb = dst_bbx_pts_pose[j][0]
                iou = cal_iou(ib, jb)
                if iou > iou_thr:
                    if dst_bbx_pts_pose[j] not in dst2:
                        dst2.append(dst_bbx_pts_pose[j])
                else:
                    if iou == 0 and check_unique_num == len(dst_bbx_pts_pose) and dst_bbx_pts_trad[i] not in dst1:
                        dst1.append(dst_bbx_pts_trad[i])
                    if iou == 0 and dst_bbx_pts_pose[j] not in dst2:
                        dst2.append(dst_bbx_pts_pose[j])
        dst = dst1 + dst2

        for d in dst:
            dis = cal_distance(d[1][0], d[1][1])
            wides.append(dis)

            cv2.rectangle(img, (d[0][0], d[0][1]), (d[0][2], d[0][3]), (255, 0, 255), 2)
            if draw_circle:
                cv2.circle(img, tuple(map(round, d[1][0])), 5, (255, 56, 56), -1, lineType=cv2.LINE_AA)
                cv2.circle(img, tuple(map(round, d[1][1])), 5, (255, 157, 151), -1, lineType=cv2.LINE_AA)
            # cv2.putText(img, "{}".format(dis), (d[0][0], d[0][1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            (text_width, text_height), baseline = cv2.getTextSize(str(dis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            if d[0][1] - text_height - 10 < 0:  # 如果文本超出图像上边缘
                text_y = d[0][3] + text_height + 8  # 将文本显示在框的下方
            else:
                text_y = d[0][1] - 5  # 将文本显示在框的上方

            cv2.putText(img, "{:.2f}".format(dis), (d[0][0], text_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

        return img, dst, wides


def btd_test():
    # data_path = r"D:\Gosion\Projects\006.Belt_Torn_Det\data\大唐发耳皮带撕裂告警图片\24"
    # save_path = make_save_path(data_path, ".", "results")
    # file_list = get_file_list(data_path)
    # for f in file_list:
    #     f_abs_path = data_path + "/{}".format(f)
    #     img = opencv_imread_filename_contain_chinese(f_abs_path)
    #     res = cv2.medianBlur(img, 7)
    #     f_dst_path = save_path + "/{}".format(f)
    #     opencv_imwrite_filename_contain_chinese(f_dst_path, res)

    thresh = 100
    data_path = r"D:\Gosion\Projects\data\silie_data_cp\train\tear"
    # data_path = r"D:\Gosion\Projects\006.Belt_Torn_Det\data\rare_samples\000"
    # data_path = r"D:\Gosion\Projects\data\silie_data_cp\train\not_torn"
    # data_path = r"E:\wujiahu\006.Belt_Torn_Det\data\A7300MG30_DE36009AAK00231_frames\Video_2025_03_27_172528_4"
    # data_path = r"E:\wujiahu\006.Belt_Torn_Det\data\A7300MG30_DE36009AAK00231_frames\cross_test2"
    # data_path = r"D:\Gosion\Projects\data\silie_data_cp\train\not_torn_selected"
    # data_path = r"D:\Gosion\Projects\data\silie_data_cp\train\not_torn_selected2"
    save_path = make_save_path(data_path, ".", "results-mean")
    file_list = get_file_list(data_path)
    for f in file_list:
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        img_vis = img.copy()
        
        img0 = img[:, :, 0]
        
        # -----------------------------------------------------------------------------------------
        # 方法1：连通域分析
        analysis_output = connected_components_analysis_btd(img0, connectivity=8, area_thr=200, h_thr=8, w_thr=8)
        img_vis_cca, dst_bbx_pts_cca, wides_cca = get_connected_components_analysis_bbx(img0, img, analysis_output, expand_pixels=10, draw_circle=False)

        # 方法2：统计每一列的像素值和（reduce sum）、异常分析方法（z_score）
        img_vis_rszs, dst_bbx_pts_rszs, wides_rszs = reduceSum_zScore(img0, img, reducesum_thr=100, zscore_thr=3, group_thr=15, flag_no_gap_abnormal=True, abnormal_mean_r=3)
        
        # 合并方法1和方法2的结果
        input_cca = (img_vis_cca, dst_bbx_pts_cca, wides_cca)
        input_rszs = (img_vis_rszs, dst_bbx_pts_rszs, wides_rszs)
        img_vis, dst_bbx_pts, wides = merge_bbx(img_vis, input_cca, input_rszs, iou_thr=0.005, dis_thresh=50)
        print("wides: {}".format(wides))
        if len(dst_bbx_pts) > 0:
            cv2.imwrite(save_path + "/{}".format(f), img_vis)



def median_filter_1d_v2(signal, kernel_size):
    pad = kernel_size // 2
    signal_padded = np.pad(signal, (pad, pad), mode='edge')
    filtered = np.zeros_like(signal)
    for i in range(len(signal)):
        filtered[i] = np.median(signal_padded[i:i+kernel_size])
    return filtered


def detect_gap(image_path):
    # 读取图像并转为灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found")
        return None

    # 二值化处理
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 计算垂直投影寻找主横线位置
    sum_rows = cv2.reduce(binary, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S).flatten()
    max_sum = 0
    best_y = 0
    height = binary.shape[0]

    for y in range(height - 9):
        current_sum = sum_rows[y:y+10].sum()
        if current_sum > max_sum:
            max_sum = current_sum
            best_y = y

    # 裁剪主横线区域
    cropped = binary[best_y:best_y+10, :]

    # 计算水平投影并处理
    sum_cols = cv2.reduce(cropped, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).flatten()
    sum_cols_smoothed = median_filter_1d_v2(sum_cols, 5)

    # 创建缺口掩膜
    threshold_value = 255  # 基于10%的阈值设定
    gap_mask = (sum_cols_smoothed < threshold_value).astype(np.uint8) * 255

    # 寻找缺口轮廓
    contours, _ = cv2.findContours(gap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gaps = []
    for contour in contours:
        x, _, w, _ = cv2.boundingRect(contour)
        gaps.append((x, w))

    if not gaps:
        print("未检测到缺口")
        return None

    # 选择最大宽度的缺口
    gaps.sort(key=lambda x: x[1], reverse=True)
    gap_start, gap_width = gaps[0]

    return (gap_start, gap_width)


def stats_btd(file_path, start=0, end=100, baseline=[3, [2, 4, 6]]):
    with open(file_path, "r", encoding="utf-8") as f_read:
        lines = f_read.readlines()

    selected = lines[start - 1:end]

    has_res = []
    no_res = []
    correct = 0
    rela_correct = 0
    for line in selected:
        l = line.strip()
        if "-----------------------可能存在" in l:
            if "可能存在 0 个裂痕: 宽 [] 像素" in l:
                no_res.append(0)
            else:
                res = l.split("-----------------------")[1].strip(" 像素----------------------")
                detect_num = int(res[5])
                detect_wides = eval(res.split("宽 ")[1])
                res_i = [detect_num, detect_wides]
                has_res.append(res_i)

                if res_i == baseline:
                    correct += 1

                if detect_num == baseline[0]:
                    rela_correct += 1

    abs_acc = correct / (len(has_res) + len(no_res))
    rela_acc = rela_correct / (len(has_res) + len(no_res))
    print("len_no_res: {}".format(len(no_res)))
    print("len_has_res: {}".format(len(has_res)))
    print("has_res: {}".format(has_res))
    print("abs_acc: {:.2f}".format(abs_acc))
    print("rela_acc: {:.2f}".format(rela_acc))



def reduceSumMaskArea(img):
    imgsz = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    blurred = cv2.medianBlur(enhanced, 3) 

    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    row_sums = np.sum(binary, axis=1)

    # if row_sums.size > 0:
    mean_value = np.mean(row_sums)
    median_value = np.median(row_sums)
    std_value = np.std(row_sums)

    zscore = z_score(row_sums, median_value, std_value)

    x_coords1 = np.where(row_sums > 10000)
    x_coords2 = np.where(abs(zscore) > 3)

    x_coords = merge_sorted_unique(x_coords1[0], x_coords2[0])
    groups = group_numpy_array(x_coords, thr=15)

    # print("x_coords1: {}".format(x_coords1))
    # print("x_coords2: {}".format(x_coords2))
    # print("x_coords: {}".format(x_coords))
    # print("groups: {}".format(groups))

    # common_elements = x_coords2[0][np.isin(x_coords2[0], x_coords1[0])]

    y_min = np.min(x_coords1) - 45
    y_max = np.max(x_coords1) + 45

    if y_min < 0:
        y_min = np.min(x_coords1) - 25
    if y_max > imgsz[0]:
        y_max = np.max(x_coords1) + 25

    if y_min < 0:
        y_min = np.min(x_coords1) - 10
    if y_max > imgsz[0]:
        y_max = np.max(x_coords1) + 10

    if y_min < 0:
        y_min = np.min(x_coords1)
    if y_max > imgsz[0]:
        y_max = np.max(x_coords1)

    masked = apply_mask_area_v2(img, y_min, y_max, flag=True)

    masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    masked_enhanced = clahe.apply(masked_gray)
    masked_enhanced2 = clahe.apply(masked_enhanced)

    masked_blurred = cv2.medianBlur(masked_enhanced, 3)

    masked_thresh = cv2.adaptiveThreshold(
        masked_blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    masked_cleaned = cv2.morphologyEx(masked_thresh, cv2.MORPH_OPEN, kernel)  # 开运算去噪

    return masked, masked_enhanced, masked_enhanced2, masked_blurred, masked_cleaned

       
def reduceSumMaskArea_v2(img, reduceSumThr=10000, zScoreThr=3, yExpand=45):
    imgsz = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.medianBlur(enhanced, 3) 

    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    row_sums = np.sum(binary, axis=1)

    # if row_sums.size > 0:
    mean_value = np.mean(row_sums)
    median_value = np.median(row_sums)
    std_value = np.std(row_sums)

    zscore = z_score(row_sums, median_value, std_value)

    x_coords1 = np.where(row_sums > reduceSumThr)
    # x_coords2 = np.where(abs(zscore) > zScoreThr)

    # x_coords = merge_sorted_unique(x_coords1[0], x_coords2[0])
    # groups = group_numpy_array(x_coords, thr=15)

    y_min_orig = np.min(x_coords1)
    y_max_orig = np.max(x_coords1)

    y_min = y_min_orig - yExpand
    y_max = y_max_orig + yExpand

    if y_min < 0:
        y_min = np.min(x_coords1) - yExpand // 2
    if y_max > imgsz[0]:
        y_max = np.max(x_coords1) + yExpand // 2

    if y_min < 0:
        y_min = np.min(x_coords1) - yExpand // 4
    if y_max > imgsz[0]:
        y_max = np.max(x_coords1) + yExpand // 4

    if y_min < 0:
        y_min = np.min(x_coords1)
    if y_max > imgsz[0]:
        y_max = np.max(x_coords1)

    masked = apply_mask_area_v2(img, y_min, y_max, flag=True)

    return masked, (y_min_orig, y_max_orig), (y_min, y_max)



def connected_components_analysis_20250401(img, connectivity=8, area_thr=100, h_thr=8, w_thr=8):
    """
    stats: [x, y, w, h, area]
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=connectivity)
    
    stats_new = []
    stats_new_idx = []
    areas = stats[:, -1]  # stats[:, cv2.CC_STAT_AREA]
    for i in range(1, num_labels):
        # if areas[i] > area_thr and stats[i, 2] / stats[i, 3] < 2:
        #     labels[labels == i] = 0
        # else:
        #     if stats[i, 2] < w_thr or stats[i, 3] < h_thr:
        #         labels[labels == i] = 0
        #     else:
        #         stats_new.append(stats[i])
        #         stats_new_idx.append(i)

        if areas[i] < area_thr:
            labels[labels == i] = 0
        # else:
        #     if areas[i] > area_thr and stats[i, 2] > stats[i, 3] * 1.5:
        #         stats_new.append(stats[i])
        #         stats_new_idx.append(i)
        #     elif stats[i, 2] < w_thr or stats[i, 3] < h_thr:
        #         labels[labels == i] = 0

        # if areas[i] > area_thr and stats[i, 2] > stats[i, 3] * 2:
        #     stats_new.append(stats[i])
        #     stats_new_idx.append(i)

    # 不同的连通域赋予不同的颜色
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 256)
        output[:, :, 1][mask] = np.random.randint(0, 256)
        output[:, :, 2][mask] = np.random.randint(0, 256)


    four_points = []
    for si in stats_new_idx:
        mask = np.uint8(labels == si)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 1, "len(contours) != 1"
        contour = contours[0]
        points = contour.reshape(-1, 2)
        top_left = points[np.argmin(points[:, 0] + points[:, 1])]
        bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
        bottom_left = points[np.argmin(points[:, 0] - points[:, 1])]
        top_right = points[np.argmax(points[:, 0] - points[:, 1])]
        four_points.append([top_left, top_right, bottom_right, bottom_left])

    stats_new = sorted(stats_new, key=lambda x: x[0])
    four_points = sorted(four_points, key=lambda x: x[0][0])
    
    return output, num_labels, labels, stats, centroids, stats_new, four_points
    


def find_intersections(img):
    # 读取灰度图
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape

    # 预处理：高斯模糊降噪
    blurred = cv2.GaussianBlur(img, (5,5), 0)

    # ========== 提取横向曲线 ==========
    # 高阈值提取亮色横线（>200）
    _, th_high = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    # 横向闭运算（强化水平线）
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
    horizontal = cv2.morphologyEx(th_high, cv2.MORPH_CLOSE, kernel_h)

    # ========== 提取纵向曲线 ==========
    # 低阈值提取暗色竖线（<100）
    _, th_low = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    # 纵向闭运算（强化垂直线）
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
    vertical = cv2.morphologyEx(th_low, cv2.MORPH_CLOSE, kernel_v)

    # ========== 检测交叉点 ==========
    # 计算交叉区域
    intersections = cv2.bitwise_and(horizontal, vertical)
    
    # 优化交叉点检测
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    intersections = cv2.morphologyEx(intersections, cv2.MORPH_OPEN, kernel)

    # 寻找连通区域
    contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 转换为彩色图用于标注
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 存储交点坐标
    points = []
    for cnt in contours:
        # 计算最小外接矩形
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 计算中心点
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))
            
            # 绘制标记
            cv2.circle(result, (cx, cy), 8, (0,0,255), -1)
            cv2.drawContours(result, [box], 0, (0,255,0), 2)

    # # 保存结果
    # cv2.imwrite(output_path, result)
    return points, result




def btd_test_20250401():
    data_path = r"E:\Gosion\data\006.Belt_Torn_Det\data\yitishi\bgr\data\20250331\src"
    save_path = make_save_path(data_path, ".", "results")
    change_brt_save_path = save_path + "/change_brightness"
    equalized_image_save_path = save_path + "/equalized_image"
    enhanced_save_path = save_path + "/enhanced"
    enhanced2_save_path = save_path + "/enhanced2"
    enhanced3_save_path = save_path + "/enhanced3"
    blurred_save_path = save_path + "/blurred"
    masked_save_path = save_path + "/masked"
    masked_enhanced_save_path = save_path + "/masked_enhanced"
    masked_enhanced2_save_path = save_path + "/masked_enhanced2"
    masked_blurred_save_path = save_path + "/masked_blurred"
    masked_cleaned_save_path = save_path + "/masked_cleaned"
    binary_save_path = save_path + "/binary"
    cd_save_path = save_path + "/corner_detect"
    sharpened_save_path = save_path + "/sharpened"
    os.makedirs(change_brt_save_path, exist_ok=True)
    os.makedirs(equalized_image_save_path, exist_ok=True)
    os.makedirs(enhanced_save_path, exist_ok=True)
    os.makedirs(enhanced2_save_path, exist_ok=True)
    os.makedirs(enhanced3_save_path, exist_ok=True)
    os.makedirs(blurred_save_path, exist_ok=True)
    os.makedirs(masked_save_path, exist_ok=True)
    os.makedirs(masked_enhanced_save_path, exist_ok=True)
    os.makedirs(masked_enhanced2_save_path, exist_ok=True)
    os.makedirs(masked_blurred_save_path, exist_ok=True)
    os.makedirs(masked_cleaned_save_path, exist_ok=True)
    os.makedirs(binary_save_path, exist_ok=True)
    os.makedirs(cd_save_path, exist_ok=True)
    os.makedirs(sharpened_save_path, exist_ok=True)



    file_list = get_file_list(data_path)

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        img_vis = img.copy()

        changed_brt = change_brightness(img, random=True, p=1, value=(-50, 50))

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert the YUV image back to RGB format
        equalized_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced2 = clahe.apply(enhanced)
        enhanced3 = clahe.apply(enhanced2)
        # enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


        # kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
        # horizontal = cv2.morphologyEx(enhanced2, cv2.MORPH_CLOSE, kernel_h)
        # kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
        # vertical = cv2.morphologyEx(enhanced2, cv2.MORPH_CLOSE, kernel_v)
        # intersections = cv2.bitwise_and(horizontal, vertical)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        # intersections2 = cv2.morphologyEx(intersections, cv2.MORPH_OPEN, kernel)

        # horizontal_path = save_path + "/{}_horizontal.jpg".format(fname) 
        # vertical_path = save_path + "/{}_vertical.jpg".format(fname) 
        # intersections_path = save_path + "/{}_intersections.jpg".format(fname) 
        # intersections2_path = save_path + "/{}_intersections2.jpg".format(fname)
        # cv2.imwrite(horizontal_path, horizontal)
        # cv2.imwrite(vertical_path, vertical)
        # cv2.imwrite(intersections_path, intersections)
        # cv2.imwrite(intersections2_path, intersections2)


        # # 使用示例
        # points, intersection_result = find_intersections(enhanced2)
        # print(f"检测到交点坐标：{points}")


        blurred = cv2.medianBlur(enhanced, 3)  # 或 cv2.GaussianBlur()

        change_brt_path = change_brt_save_path + "/{}_change_brt.jpg".format(fname) 
        equalized_image_path = equalized_image_save_path + "/{}_enhanced.jpg".format(fname) 
        enhanced_path = enhanced_save_path + "/{}_enhanced.jpg".format(fname) 
        enhanced2_path = enhanced2_save_path + "/{}_enhanced2.jpg".format(fname) 
        enhanced3_path = enhanced3_save_path + "/{}_enhanced3.jpg".format(fname) 
        # enhanced_bgr_path = save_path + "/{}_enhanced_bgr.jpg".format(fname) 
        blurred_path = blurred_save_path + "/{}_blurred.jpg".format(fname) 
        # intersection_result_path = save_path + "/{}_intersection_result.jpg".format(fname) 
        cv2.imwrite(change_brt_path, changed_brt)
        cv2.imwrite(equalized_image_path, equalized_image)
        cv2.imwrite(enhanced_path, enhanced)
        cv2.imwrite(enhanced2_path, enhanced2)
        cv2.imwrite(enhanced3_path, enhanced3)
        # cv2.imwrite(enhanced_bgr_path, enhanced_bgr)
        cv2.imwrite(blurred_path, blurred)
        # cv2.imwrite(intersection_result_path, intersection_result)
        # print("========")



        

        # thresh = cv2.adaptiveThreshold(
        #     blurred, 255, 
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #     cv2.THRESH_BINARY_INV, 11, 2
        # )

        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 开运算去噪
        # cleaned_path = save_path + "/{}_cleaned.jpg".format(fname) 
        # cv2.imwrite(cleaned_path, cleaned)

        # # median_blurred = cv2.medianBlur(cleaned, 3)
        # # median_blurred_path = save_path + "/{}_median_blurred.jpg".format(fname) 
        # # cv2.imwrite(median_blurred_path, median_blurred)

        binary_path = binary_save_path + "/{}_binary.jpg".format(fname) 
        cv2.imwrite(binary_path, binary)

        try:
            masked, masked_enhanced, masked_enhanced2, masked_blurred, masked_cleaned = reduceSumMaskArea(img)

            masked_path = masked_save_path + "/{}_masked.jpg".format(fname) 
            masked_enhanced_path = masked_enhanced_save_path + "/{}_masked_enhanced.jpg".format(fname) 
            masked_enhanced2_path = masked_enhanced2_save_path + "/{}_masked_enhanced2.jpg".format(fname) 
            masked_blurred_path = masked_blurred_save_path + "/{}_masked_blurred.jpg".format(fname) 
            masked_cleaned_path = masked_cleaned_save_path + "/{}_masked_cleaned.jpg".format(fname) 
            cv2.imwrite(masked_path, masked)
            cv2.imwrite(masked_enhanced_path, masked_enhanced)
            cv2.imwrite(masked_enhanced2_path, masked_enhanced2)
            cv2.imwrite(masked_blurred_path, masked_blurred)
            cv2.imwrite(masked_cleaned_path, masked_cleaned)

            cd_res = corner_detect_v3(masked_enhanced2)
            cd_path = cd_save_path + "/{}_cd_res.jpg".format(fname) 
            cv2.imwrite(cd_path, cd_res)

            sharpen_kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
 
            # 应用滤波器核进行锐化
            sharpened_image = cv2.filter2D(masked_enhanced2, -1, sharpen_kernel)
            sharpened_path = sharpened_save_path + "/{}_sharpened_.jpg".format(fname) 
            cv2.imwrite(sharpened_path, sharpened_image)


        except Exception as e:
            print(e)


            
        # res = connected_components_analysis_20250401(masked_cleaned, area_thr=5000)
        # masked_cleaned_cca_path = save_path + "/{}_masked_cleaned_cca.jpg".format(fname) 
        # cv2.imwrite(masked_cleaned_cca_path, res[2])



        # bilateral_img = cv2.bilateralFilter(img, 9, 75, 75)
        # bilateral_enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        # bilateral_img_path = save_path + "/{}_bilateral_img.jpg".format(fname) 
        # bilateral_enhanced_path = save_path + "/{}_bilateral_enhanced.jpg".format(fname)
        # cv2.imwrite(bilateral_img_path, bilateral_img)
        # cv2.imwrite(bilateral_enhanced_path, bilateral_enhanced)

        print("========")


    
def corner_detect_v3(img):
    # =======================================================================================
    # https://www.cnblogs.com/GYH2003/articles/GYH-PythonOpenCV-FeatureDetection-Corner.html
    imgsz = img.shape
    if len(imgsz) == 3:
        img_vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)  # 转换为浮点型
    else:
        img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = np.float32(img)  # 转换为浮点型
    
    # dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)  # Harris 角点检测
    t1 = time.time()
    dst = cv2.cornerHarris(gray, blockSize=8, ksize=15, k=0.04)  # Harris 角点检测
    # dst = cv2.cornerHarris(gray, blockSize=16, ksize=7, k=0.04)  # Harris 角点检测
    # dst = cv2.cornerHarris(gray, blockSize=10, ksize=15, k=0.04)  # Harris 角点检测
    t2 = time.time()
    # print("Harris角点检测耗时：", t2 - t1)

    r, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)  # 二值化阈值处理
    dst = np.uint8(dst)  # 转换为整型
    # cv2.imshow('dst', dst)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # 不同的连通域赋予不同的颜色
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 256)
        output[:, :, 1][mask] = np.random.randint(0, 256)
        output[:, :, 2][mask] = np.random.randint(0, 256)

    # 标记角点
    dst = cv2.dilate(dst, None)  # 膨胀操作，增强角点标记
    # print(dst > 0.01 * dst.max())
    img_vis[dst > 0.01 * dst.max()] = [0, 0, 255]  # 将角点标记为红色

    # # 显示结果
    # cv2.imshow('Harris Corners', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img_vis
    

def test_20250401():
    src_path = r"E:\Gosion\data\006.Belt_Torn_Det\data\yitishi\pose_yolo_format\images"
    lbl_path = r"E:\Gosion\data\006.Belt_Torn_Det\data\yitishi\pose_yolo_format\labels"
    data_path = r"E:\Gosion\data\006.Belt_Torn_Det\data\yitishi\bgr\data\20250331\src_results"
    file_list = get_file_list(src_path)
    dir_list = get_dir_list(data_path)

    save_path = r"D:\Gosion\data\006.Belt_Torn_Det\data\pose\v4\v4_yitiji\000"

    for d in dir_list:
        d_path = os.path.join(data_path, d)

        d_img_save_path = save_path + "/{}/images".format(d)
        d_lbl_save_path = save_path + "/{}/labels".format(d)
        os.makedirs(d_img_save_path, exist_ok=True)
        os.makedirs(d_lbl_save_path, exist_ok=True)
       
        for f in file_list:
            fname = os.path.splitext(f)[0]
            
            if d == "equalized_image":
                f_abs_path = d_path + "/{}_{}.jpg".format(fname, "enhanced")
                f_dst_path = d_img_save_path + "/{}_{}.jpg".format(fname, "enhanced")
                shutil.copy(f_abs_path, f_dst_path)
                f_lbl_abs_path = lbl_path + "/{}.txt".format(fname)
                f_lbl_dst_path = d_lbl_save_path + "/{}_{}.txt".format(fname, "enhanced")
                shutil.copy(f_lbl_abs_path, f_lbl_dst_path)
            elif d == "sharpened":
                f_abs_path = d_path + "/{}_{}.jpg".format(fname, "sharpened_")
                f_dst_path = d_img_save_path + "/{}_{}.jpg".format(fname, "sharpened_")
                shutil.copy(f_abs_path, f_dst_path)
                f_lbl_abs_path = lbl_path + "/{}.txt".format(fname)
                f_lbl_dst_path = d_lbl_save_path + "/{}_{}.txt".format(fname, "sharpened_")
                shutil.copy(f_lbl_abs_path, f_lbl_dst_path)
            else:
                f_abs_path = d_path + "/{}_{}.jpg".format(fname, d)
                f_dst_path = d_img_save_path + "/{}_{}.jpg".format(fname, d)
                shutil.copy(f_abs_path, f_dst_path)
                f_lbl_abs_path = lbl_path + "/{}.txt".format(fname)
                f_lbl_dst_path = d_lbl_save_path + "/{}_{}.txt".format(fname, d)
                shutil.copy(f_lbl_abs_path, f_lbl_dst_path)

        
def extract_specific_color(img, lower, upper, color="green"):
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


def ransac_parabola(points, select_num=10, num_iterations=1000, threshold=15.0):
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
        # 随机选择三个点
        sample_indices = random.sample(range(n_points), select_num)
        sample_points = points[sample_indices]
        sx = sample_points[:, 0]
        sy = sample_points[:, 1]

        model = np.polyfit(sx, sy, 2)

        # 计算所有点的垂直距离
        x_all = points[:, 0]
        y_all = points[:, 1]
        y_pred = model[0] * x_all**2 + model[1] * x_all + model[2]
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
        model = np.polyfit(x_inliers, y_inliers, 2)
        best_model = model
    else:
        return None  # 无有效模型

    return best_model


def get_y_pred(model, x, n=1):
    if n == 1:
        # "y = {:.3f}x + {:.3f}"
        y_pred = model[0] * x + model[1]
        return y_pred
    elif n == 2:
        # "y = {:.3f}x² + {:.3f}x + {:.3f}"
        y_pred = model[0] * x**2 + model[1] * x + model[2]
        return y_pred
    elif n == 3:
        # "y = {:.3f}x³ + {:.3f}x² + {:.3f}x + {:.3f}"
        y_pred = model[0] * x**3 + model[1] * x**2 + model[2] * x + model[3]
        return y_pred


def ransac_fit_curve(points, n=1, select_num=10, num_iterations=1000, threshold=15.0):
    """
    RANSAC拟合曲线
    n: 曲线阶数
    """
    assert n in [1, 2, 3], "n must be 1, 2, or 3"

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

        model = np.polyfit(sx, sy, n)

        # 计算所有点的垂直距离
        x_all = points[:, 0]
        y_all = points[:, 1]
        # y_pred = model[0] * x_all**2 + model[1] * x_all + model[2]
        y_pred = get_y_pred(model, x_all, n)
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
        model = np.polyfit(x_inliers, y_inliers, n)
        best_model = model
    else:
        return None  # 无有效模型

    return best_model


def plot_fit_curve(img, model, n=1, color=(255, 0, 255)):
    imgsz = img.shape[:2]
    x = np.linspace(0, imgsz[1], imgsz[1] - 1)
    y = get_y_pred(model, x, n)
    for i in range(len(x)):
        cv2.circle(img, (int(x[i]), int(y[i])), 2, color, -1)

    return img


def plot_fit_parabola(img, model, color=(255, 0, 255)):
    """
    a, b, c = model
    print(f"拟合的抛物线方程: y = {a:.3f}x² + {b:.3f}x + {c:.3f}")
    """
    imgsz = img.shape[:2]
    x = np.linspace(0, imgsz[1], imgsz[1] - 1)
    y = model[0] * x**2 + model[1] * x + model[2]
    for i in range(len(x)):
        cv2.circle(img, (int(x[i]), int(y[i])), 2, color, -1)

    return img


def parabola(x, a, b, c):
    return a*x**2 + b*x + c


def contrast_stretch(img):
    assert len(img.shape) == 2, "len(img.shape) != 2"
    input_min, input_max = img.min(), img.max()
    out = (img - input_min) * (255 / (input_max - input_min))
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def stretch_opencv(img, alpha=1.5, beta=0):
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out



def test_20250407():
    data_path = r"E:\Gosion\data\006.Belt_Torn_Det\data\yitishi\bgr\data\20250331\src"
    save_path = make_save_path(data_path, ".", "results_20250407")
    enhanced_save_path = save_path + "/enhanced"
    enhanced2_save_path = save_path + "/enhanced2"
    contrast_stretch_save_path = save_path + "/contrast_stretch"
    orig_contrast_stretch_save_path = save_path + "/orig_contrast_stretch"
    enhanced2_inv_save_path = save_path + "/enhanced2_inv"

    blurred_save_path = save_path + "/blurred"
    binary_save_path = save_path + "/binary"
    orig_cd_save_path = save_path + "/orig_corner_detect"
    cd_save_path = save_path + "/corner_detect"
    sharpened_save_path = save_path + "/sharpened"
    sharpened_corner_detect_save_path = save_path + "/sharpened_corner_detect"
    os.makedirs(enhanced_save_path, exist_ok=True)
    os.makedirs(contrast_stretch_save_path, exist_ok=True)
    os.makedirs(orig_contrast_stretch_save_path, exist_ok=True)
    os.makedirs(enhanced2_save_path, exist_ok=True)
    os.makedirs(enhanced2_inv_save_path, exist_ok=True)
    os.makedirs(blurred_save_path, exist_ok=True)
    os.makedirs(binary_save_path, exist_ok=True)
    os.makedirs(orig_cd_save_path, exist_ok=True)
    os.makedirs(cd_save_path, exist_ok=True)
    os.makedirs(sharpened_save_path, exist_ok=True)
    os.makedirs(sharpened_corner_detect_save_path, exist_ok=True)


    file_list = get_file_list(data_path)

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)

        orig_cd_res = corner_detect_v3(img)
        orig_cd_path = orig_cd_save_path + "/{}_orig_cd_res.jpg".format(fname) 
        cv2.imwrite(orig_cd_path, orig_cd_res)

        img_vis = img.copy()

        changed_brt = change_brightness(img, random=True, p=1, value=(-50, 50))

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert the YUV image back to RGB format
        equalized_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced2 = clahe.apply(enhanced)

        blurred = cv2.medianBlur(enhanced, 3)  # 或 cv2.GaussianBlur()

        enhanced_path = enhanced_save_path + "/{}_enhanced.jpg".format(fname) 
        enhanced2_path = enhanced2_save_path + "/{}_enhanced2.jpg".format(fname) 
        # enhanced_bgr_path = save_path + "/{}_enhanced_bgr.jpg".format(fname) 
        blurred_path = blurred_save_path + "/{}_blurred.jpg".format(fname) 
        # intersection_result_path = save_path + "/{}_intersection_result.jpg".format(fname) 
        cv2.imwrite(enhanced_path, enhanced)
        cv2.imwrite(enhanced2_path, enhanced2)
        # cv2.imwrite(enhanced_bgr_path, enhanced_bgr)
        cv2.imwrite(blurred_path, blurred)
        # cv2.imwrite(intersection_result_path, intersection_result)
        # print("========")

        
        enhanced2_inv = 255 - enhanced2
        enhanced2_inv_path = enhanced2_inv_save_path + "/{}_enhanced2_inv.jpg".format(fname)
        cv2.imwrite(enhanced2_inv_path, enhanced2_inv)

        contrast_stretch_img = contrast_stretch(enhanced2)
        contrast_stretch_path = contrast_stretch_save_path + "/{}_contrast_stretch.jpg".format(fname)
        cv2.imwrite(contrast_stretch_path, contrast_stretch_img)



        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 开运算去噪
        # cleaned_path = save_path + "/{}_cleaned.jpg".format(fname) 
        # cv2.imwrite(cleaned_path, cleaned)

        # # median_blurred = cv2.medianBlur(cleaned, 3)
        # # median_blurred_path = save_path + "/{}_median_blurred.jpg".format(fname) 
        # # cv2.imwrite(median_blurred_path, median_blurred)

        binary_path = binary_save_path + "/{}_binary.jpg".format(fname) 
        cv2.imwrite(binary_path, binary)

        cd_res = corner_detect_v3(enhanced2)
        cd_path = cd_save_path + "/{}_cd_res.jpg".format(fname) 
        cv2.imwrite(cd_path, cd_res)

        sharpen_kernel = np.array([[-1,-1,-1], 
                        [-1, 9,-1],
                        [-1,-1,-1]])

        # 应用滤波器核进行锐化
        sharpened_image = cv2.filter2D(enhanced2, -1, sharpen_kernel)
        sharpened_path = sharpened_save_path + "/{}_sharpened_.jpg".format(fname) 
        cv2.imwrite(sharpened_path, sharpened_image)

        sp_cd_res = corner_detect_v3(sharpened_image)
        sp_cd_path = sharpened_corner_detect_save_path + "/{}_sp_cd_res.jpg".format(fname) 
        cv2.imwrite(sp_cd_path, sp_cd_res)

        print("========")


def high_reserve(img,ksize,sigm):
    img = img * 1.0
    gauss_out = cv2.GaussianBlur(img,(ksize,ksize),sigm)
    img_out = img - gauss_out + 128
    img_out = img_out/255.0
    # 饱和处理
    mask_1 = img_out  < 0
    mask_2 = img_out  > 1
    img_out = img_out * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2
    return img_out

def usm(img ,number):
    blur_img = cv2.GaussianBlur(img, (0, 0), number)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

    return usm


def Overlay(target, blend):
    mask = blend < 0.5
    img = 2 * target * blend * mask + (1 - mask) * (1 - 2 * (1 - target) * (1 - blend))
    return img


def merge_lines(lines, e0, e1):
    merged_lines = []
    # 遍历每条直线
    for i, line in enumerate(lines):
        if len(merged_lines) == 0:
            merged_lines.append(line)
        else:
            merged = False
            # 遍历已合并的直线
            for i, merged_line in enumerate(merged_lines):
                theta = line[1]
                distan= line[0]
                mtheta = merged_line[1]
                mdistan= merged_line[0]
                dtheta=abs(np.sin(theta))-abs(np.sin(mtheta))
                ds=abs(abs(mdistan)-abs(distan))
                # 判断两条直线是否可合并
                if dtheta < e0 and ds < e1 :
                    merged_lines[i] = [(line[0] + merged_line[0]) / 2, (line[1] + merged_line[1]) / 2]
                    merged = True
                    break
            # 如果不能合并，则将直线添加到已合并的直线列表中
            if not merged:
                merged_lines.append(line)
    return merged_lines


def extend_lines(lines, img_shape):
    extended_lines = []
    # 遍历每条直线
    for line in lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        extended_lines.append([(x1, y1), (x2, y2)])
    return extended_lines


def find_intersections(lines):
    intersections = []
    # 遍历每条直线的组合
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            x1, y1 = line1[0]
            x2, y2 = line1[1]
            x3, y3 = line2[0]
            x4, y4 = line2[1]
            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            # 判断两条直线是否相交
            if denominator != 0:
                x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
                y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
                intersections.append((int(x), int(y)))
    return intersections


def draw_lines(img, lines):
    for line in lines:
        cv2.line(img, line[0], line[1], (0, 0, 255), 2)


def draw_intersections(img, intersections):
    for intersection in intersections:
        cv2.circle(img, intersection, 5, (0, 255, 0), -1)


def detect_lines(img, e0, e1, rho, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, rho, np.pi/180, threshold)
    lines = lines[:, 0, :]
    merged_lines = merge_lines(lines, e0, e1)
    extended_lines = extend_lines(merged_lines, img.shape)
    intersections = find_intersections(extended_lines)
    draw_lines(img, extended_lines)
    draw_intersections(img, intersections)
    return img



def test_20250407_2():
    img_path = r"E:\Gosion\data\006.Belt_Torn_Det\data\yitishi\bgr\data\20250331\src\1743418859.9315038.jpg"
    fname = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    img_vis = img.copy()

    save_path = r"E:\Gosion\data\006.Belt_Torn_Det\data\yitishi\bgr\data\20250331\src_1743418859.9315038_results"
    os.makedirs(save_path, exist_ok=True)

    orig_cd_res = corner_detect_v3(img)
    orig_cd_path = save_path + "/{}_orig_cd_res.jpg".format(fname) 
    cv2.imwrite(orig_cd_path, orig_cd_res)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced2 = clahe.apply(enhanced)

    blurred = cv2.medianBlur(enhanced2, 7)  # 或 cv2.GaussianBlur()

    enhanced_path = save_path + "/{}_enhanced.jpg".format(fname)
    enhanced2_path = save_path + "/{}_enhanced2.jpg".format(fname)
    blurred_path = save_path + "/{}_blurred.jpg".format(fname)
    cv2.imwrite(enhanced_path, enhanced)
    cv2.imwrite(enhanced2_path, enhanced2)
    cv2.imwrite(blurred_path, blurred)

    
    enhanced2_inv = 255 - enhanced2
    enhanced2_inv_path = save_path + "/{}_enhanced2_inv.jpg".format(fname)
    cv2.imwrite(enhanced2_inv_path, enhanced2_inv)

    contrast_stretch_img = contrast_stretch(enhanced2)
    contrast_stretch_path = save_path + "/{}_contrast_stretch.jpg".format(fname)
    cv2.imwrite(contrast_stretch_path, contrast_stretch_img)


    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary_path = save_path + "/{}_binary.jpg".format(fname) 
    cv2.imwrite(binary_path, binary)

    cd_res = corner_detect_v3(enhanced2)

    cd_path = save_path + "/{}_cd_res.jpg".format(fname) 
    cv2.imwrite(cd_path, cd_res)

    sharpen_kernel = np.array([[-1,-1,-1], 
                    [-1, 9,-1],
                    [-1,-1,-1]])

    # 应用滤波器核进行锐化
    # sharpened_image = cv2.filter2D(enhanced2, -1, sharpen_kernel)
    sharpened_image = cv2.filter2D(img, -1, sharpen_kernel)
    sharpened_path = save_path + "/{}_sharpened.jpg".format(fname) 
    cv2.imwrite(sharpened_path, sharpened_image)

    sp_cd_res = corner_detect_v3(sharpened_image)
    sp_cd_path = save_path + "/{}_sp_cd_res.jpg".format(fname) 
    cv2.imwrite(sp_cd_path, sp_cd_res)


    enhanced2_inv_cd_res = corner_detect_v3(enhanced2_inv)
    enhanced2_inv_cd_path = save_path + "/{}_enhanced2_inv_cd_res.jpg".format(fname) 
    cv2.imwrite(enhanced2_inv_cd_path, enhanced2_inv_cd_res)


    sharpened_image_cd_res = corner_detect_v3(sharpened_image)
    sharpened_image_cd_path = save_path + "/{}_sharpened_image_cd_res.jpg".format(fname) 
    cv2.imwrite(sharpened_image_cd_path, sharpened_image_cd_res)


    blurred_image_cd_res = corner_detect_v3(blurred)
    blurred_image_cd_path = save_path + "/{}_blurred_image_cd_res.jpg".format(fname) 
    cv2.imwrite(blurred_image_cd_path, blurred_image_cd_res)




    img_gas = cv2.GaussianBlur(img, (3,3), 1.5)
    high = high_reserve(img_gas,11,5)
    usm1 = usm(high,11)
    add = (Overlay(img_gas/255,usm1)*255).astype(np.uint8)
    # add = cv2.medianBlur((add*255).astype(np.uint8),3)
    add_path = save_path + "/{}_add.jpg".format(fname)
    cv2.imwrite(add_path, add)


    # ----
    # laplacian = cv2.Laplacian(enhanced2, cv2.CV_64F)
    # sharp_img = cv2.subtract(enhanced2, laplacian)
    # sharp_img = cv2.convertScaleAbs(sharp_img)
    # sharp_img_path = save_path + "/{}_sharp_img.jpg".format(fname)
    # cv2.imwrite(sharp_img_path, sharp_img)

    # blurred = cv2.GaussianBlur(enhanced2, (5, 5), 0)
    # laplacian2 = cv2.Laplacian(blurred, cv2.CV_64F)
    # sharp2 = cv2.addWeighted(enhanced2, 1.5, laplacian2, -0.5, 0)
    # sharp2_path = save_path + "/{}_sharp2.jpg".format(fname)
    # cv2.imwrite(sharp2_path, sharp2)



    # e0 = 10 # 角度误差阈值
    # e1 = 10 # 距离误差阈值
    # rho = 2 # 像素偏差
    # threshold = 260 # 投票数阈值
    # lines_result = detect_lines(cv2.cvtColor(enhanced2_inv, cv2.COLOR_GRAY2BGR), np.sin(e0/180*np.pi/2), e1, rho, threshold)
    # lines_result_path = save_path + "/{}_lines_result.jpg".format(fname)
    # cv2.imwrite(lines_result_path, lines_result)



def main_test_20250408():
    data_path = r"E:\Gosion\data\006.Belt_Torn_Det\data\yitishi\bgr\data\20250331\src"
    save_path = make_save_path(data_path, ".", "results_20250408")
    enhanced2_save_path = save_path + "/enhanced2"
    blurred_save_path = save_path + "/blurred"
    orig_cd_save_path = save_path + "/orig_corner_detect"
    enhanced2_cd_save_path = save_path + "/enhanced2_corner_detect"
    blurred_cd_save_path = save_path + "/blurred_corner_detect"
    os.makedirs(enhanced2_save_path, exist_ok=True)
    os.makedirs(blurred_save_path, exist_ok=True)
    os.makedirs(orig_cd_save_path, exist_ok=True)
    os.makedirs(enhanced2_cd_save_path, exist_ok=True)
    os.makedirs(blurred_cd_save_path, exist_ok=True)

    file_list = get_file_list(data_path)

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)

        orig_cd_res = corner_detect_v3(img)
        orig_cd_path = orig_cd_save_path + "/{}_orig_cd_res.jpg".format(fname) 
        cv2.imwrite(orig_cd_path, orig_cd_res)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced2 = clahe.apply(enhanced)
        blurred = cv2.medianBlur(enhanced2, 7)

        enhanced2_path = enhanced2_save_path + "/{}_enhanced2.jpg".format(fname) 
        cv2.imwrite(enhanced2_path, enhanced2)

        blurred_path = blurred_save_path + "/{}_blurred.jpg".format(fname) 
        cv2.imwrite(blurred_path, blurred)

        enhanced2_cd_res = corner_detect_v3(enhanced2)
        enhanced2_cd_path = enhanced2_cd_save_path + "/{}_enhanced2_cd_res.jpg".format(fname) 
        cv2.imwrite(enhanced2_cd_path, enhanced2_cd_res)

        blurred_cd_res = corner_detect_v3(blurred)
        blurred_cd_path = blurred_cd_save_path + "/{}_blurred_cd_res.jpg".format(fname) 
        cv2.imwrite(blurred_cd_path, blurred_cd_res)


def select_specific_hw_images(data_path, hw=(1920, 1080)):
    """
    (h, w): (1920, 1080)
    """
    save_path = make_save_path(data_path, ".", "results_{}_{}".format(hw[0], hw[1]))
    img_save_path = save_path + "/images"
    jsn_save_path = save_path + "/jsons"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(jsn_save_path, exist_ok=True)

    img_path = data_path + "/images"
    jsn_path = data_path + "/jsons"

    file_list = get_file_list(img_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = img_path + "/{}".format(f)
        j_abs_path = jsn_path + "/{}.json".format(f)
        img = cv2.imread(f_abs_path)
        imgsz = img.shape
        if imgsz[0] == hw[0] and imgsz[1] == hw[1]:
            f_dst_path = img_save_path + "/{}".format(f)
            j_dst_path = jsn_save_path + "/{}.json".format(f)
            shutil.move(f_abs_path, f_dst_path)
            shutil.move(j_abs_path, j_dst_path)


def detect_horizontal_line(img, m=0):
    assert m in [0, 1], "m must be 0 or 1"
    imgsz = img.shape[:2]
    if m == 0:
        # CLAHE增强
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 横线检测
        k1 = np.array([
            [-1, -1, -1], 
            [0, 0, 0],
            [1, 1, 1]
        ])
        k1_filtered = cv2.filter2D(binary, -1, k1)

        # 连通域分析去除一些小点
        num_labels, labels, stats, centroids, output = connected_components_analysis(k1_filtered, connectivity=8, area_thr=25, h_thr=20, w_thr=50)
        _, labels_binary = cv2.threshold(np.uint8(labels), 0, 255, cv2.THRESH_BINARY)

        return labels_binary
    else:
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        himg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, hk, iterations=1)

        return himg



def compute_cross_correlation(img1, img2):
    # 转换为灰度图像
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 转换为浮点类型
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # 获取图像尺寸
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    # 计算FFT需要的填充尺寸
    fh = h1 + h2 - 1
    fw = w1 + w2 - 1
    
    # 零填充图像
    img1_pad = np.pad(img1, ((0, fh - h1), (0, fw - w1)), mode='constant')
    img2_pad = np.pad(img2, ((0, fh - h2), (0, fw - w2)), mode='constant')
    
    # 计算FFT
    fft1 = np.fft.fft2(img1_pad)
    fft2_conj = np.conj(np.fft.fft2(img2_pad))
    
    # 频域相乘并逆变换
    cross_corr = np.fft.ifft2(fft1 * fft2_conj).real
    
    # 移动零位移到中心
    cross_corr = np.fft.fftshift(cross_corr)
    
    return cross_corr


def Normalised_Cross_Correlation(roi, target):
    # Normalised Cross Correlation Equation
    cor=np.sum(roi*target)
    nor = np.sqrt((np.sum(roi**2)))*np.sqrt(np.sum(target**2))
    return cor / nor
    

def template_matching(img, target):
    # initial parameter
    height,width=img.shape
    tar_height,tar_width=target.shape
    (max_Y,max_X)=(0, 0)
    MaxValue = 0

    # Set image, target and result value matrix
    img=np.array(img, dtype="int")
    target=np.array(target, dtype="int")
    NccValue=np.zeros((height-tar_height,width-tar_width))

    # calculate value using filter-kind operation from top-left to bottom-right
    for y in range(0,height-tar_height):
        for x in range(0,width-tar_width):
            # image roi
            roi=img[y:y+tar_height,x:x+tar_width]
            # calculate ncc value
            NccValue[y,x] = Normalised_Cross_Correlation(roi,target)
            # find the most match area
            if NccValue[y,x]>MaxValue:
                MaxValue=NccValue[y,x]
                (max_Y,max_X) = (y,x)

    return (max_X,max_Y)



def ransac_fit_laser_curve(img, n=3, laser_color="green", select_num=1000, num_iterations=1000, threshold=50, color=(255, 0, 255)):
    mask, result, binary, binary_otsu = extract_specific_color(img, lower=None, upper=None, color=laser_color)
    # bitand = cv2.bitwise_and(binary, binary_otsu)
    # bitor = cv2.bitwise_or(binary, binary_otsu)

    points = np.where(binary_otsu == 255)
    points = np.hstack((points[1].reshape(-1, 1), points[0].reshape(-1, 1)))
    model = ransac_fit_curve(points, n, select_num, num_iterations, threshold)
    if model is not None:
        img = plot_fit_curve(img, model, n, color)
    else:
        print("拟合失败")

    return img


def main_fit_curve_20250417():
    # data_path = r"E:\Gosion\data\006.Belt_Torn_Det\data\yitishi\bgr\data\20250331\src"
    data_path = r"D:\Gosion\data\006.Belt_Torn_Det\data\video\video_frames\cropped\20250308_frames_merged"
    n = 3
    save_path = make_save_path(data_path, relative='.', add_str="ransac_fit_laser_curve_20250417_{}".format(n))

    file_list = get_file_list(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        out = ransac_fit_laser_curve(img, n, laser_color="green", select_num=1000, num_iterations=1000, threshold=50, color=(255, 0, 255))
        cv2.imwrite(save_path + "/{}".format(f), out)


# def check_false_alarm(img, pts):
#     imgsz = img.shape
#     if len(imgsz) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def fit_curve(f_abs_path, n=2):
    img = cv2.imread(f_abs_path)
    out = ransac_fit_laser_curve(img, n, laser_color="green", select_num=1000, num_iterations=1000, threshold=50, color=(255, 0, 255))
    cv2.imwrite("out.jpg", out)



# -----------------------
def fit_curve_test(n=1, select_num=2, num_iterations=100, threshold=5, color=(255, 0, 255)):
    # 生成测试数据
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    x = np.linspace(-250, 250, 500)
    # y_true = 0.5 * x**2 + 2 * x + 3
    y_true = 0.75 * x + 15
    y = y_true + np.random.normal(scale=3.0, size=x.shape)  # 添加噪声
    # 添加外点
    outlier_indices = np.random.choice(len(x), size=20, replace=False)
    y[outlier_indices] += np.random.normal(scale=30, size=20)
    points = np.column_stack((x, y))

    # 运行RANSAC
    model = ransac_fit_curve(points, n, select_num, num_iterations, threshold)
    if model is not None:
        img = plot_fit_curve(img, model, n, color)
    else:
        print("拟合失败")

    cv2.imwrite(r"D:\Gosion\data\006.Belt_Torn_Det\data\video\video_frames\cropped\ransac_curve.jpg", img)


    # 示例数据
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2, 5, 12, 8, 24, 14, 2, 20, 23, 26])
    
    # 使用numpy的polyfit函数进行多项式拟合，这里我们拟合一个一次多项式（即直线）
    model = np.polyfit(x, y, 1)
    
    # coefficients是一个数组，包含了拟合直线的斜率和截距
    a, b = model
    
    # 使用拟合的参数生成拟合直线的y值
    fitted_y = a * x + b
    
    # 绘制原始数据点和拟合直线
    plt.scatter(x, y)
    plt.plot(x, fitted_y, color='red')
    plt.savefig(r"D:\Gosion\data\006.Belt_Torn_Det\data\video\video_frames\cropped\polyfit.png")

    X = np.vstack([x, np.ones(len(x))]).T
    a2, b2 = np.linalg.lstsq(X, y, rcond=None)[0]
    # 使用拟合的参数生成拟合直线的y值
    fitted_y2 = a2 * x + b2
    
    # 绘制原始数据点和拟合直线
    plt.scatter(x, y)
    plt.plot(x, fitted_y2, color='red')
    plt.savefig(r"D:\Gosion\data\006.Belt_Torn_Det\data\video\video_frames\cropped\polyfit2.png")



def producer(q):
    for i in range(5):
        q.put(i)
    q.put(None)

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Consumed: {item}")


def corner_detect_v4(img, blockSize=15, ksize=15, k=0.06, vis=False):
    # =======================================================================================
    # https://www.cnblogs.com/GYH2003/articles/GYH-PythonOpenCV-FeatureDetection-Corner.html
    imgsz = img.shape
    if len(imgsz) == 3:
        img_vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        gray = np.float32(gray)  # 转换为浮点型
    else:
        img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        _, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        gray = np.float32(img)  # 转换为浮点型

    # 1. 计算 Harris 角点响应
    # dst = cv2.cornerHarris(gray, blockSize=15, ksize=15, k=0.06)  # Harris 角点检测
    dst = cv2.cornerHarris(gray, blockSize=blockSize, ksize=ksize, k=k)  # Harris 角点检测

    # 2. 设置阈值筛选角点
    threshold = 0.01 * dst.max()  # 通常取最大响应的 1%~5%
    corners = np.argwhere(dst > threshold)  # 获取符合条件的坐标

    # 3. 转换坐标格式（注意：np.argwhere 返回的是 (y, x) 格式）
    corners_xy = corners[:, [1, 0]]  # 转换为 (x, y) 格式

    # 4. 可视化角点（可选）
    if vis:
        for (x, y) in corners_xy:
            cv2.circle(img_vis, (x, y), 3, (255, 0, 255), -1)  # 在图像上绘制红色圆点

    return thresh_otsu, corners_xy, img_vis


# def corner_detect_filter_false_alarm(gray, blockSize=15, ksize=15, k=0.04, ccn_thr=5):
#     """
#     ccn_thr: 连通域数量阈值
#     """
#     gray = np.float32(gray)  # 转换为浮点型
#     dst = cv2.cornerHarris(gray, blockSize=blockSize, ksize=ksize, k=k)  # Harris 角点检测

#     # 2. 设置阈值筛选角点
#     threshold = 0.01 * dst.max()  # 通常取最大响应的 1%~5%
#     corners = np.argwhere(dst > threshold)  # 获取符合条件的坐标

#     # 3. 转换坐标格式（注意：np.argwhere 返回的是 (y, x) 格式）
#     corners_xy = corners[:, [1, 0]]  # 转换为 (x, y) 格式

#     # 4. 可视化角点（可选）
#     img_vis = np.zeros(shape=(gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
#     for (x, y) in corners_xy:
#         cv2.circle(img_vis, (x, y), 3, (255, 0, 255), -1)  # 在图像上绘制红色圆点

#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_vis)

#     if num_labels - 1 > ccn_thr:
#         return True
#     else:
#         return False



def corner_detect_test_20250507():
    data_path = r"G:\Gosion\data\001.Belt_Torn_Detection\data\data_51\data\20250507\src"
    save_path = r"G:\Gosion\data\001.Belt_Torn_Detection\data\data_51\data\20250507\src_corner_detect_test_20250507"
    os.makedirs(save_path, exist_ok=True)

    blockSizes = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    ksizes = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    ks = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21,0.22, 0.23, 0.24]

    file_list = get_file_list(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)

        for k in ks:
            for (blockSize, ksize) in zip(blockSizes, ksizes):
                save_path_i = save_path + "/{}_{}_{}".format(blockSize, ksize, k)
                os.makedirs(save_path_i, exist_ok=True)

                thresh_otsu, corners_xy, img_vis = corner_detect_v4(img, blockSize=blockSize, ksize=ksize, k=k, vis=True)
                cv2.imwrite(save_path_i + "/{}".format(f), img_vis)



def check_laser_conditions(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像")
        return
    
    # 转换到 HSV 颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义绿色范围（可根据实际情况调整）
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 形态学闭运算，连接断裂并去除小噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 统计每一行的白色像素数量
    row_white_counts = np.sum(mask, axis=1)
    # 设定阈值（根据激光线宽度调整，假设至少 3 像素宽）
    threshold = 3
    break_rows = np.where(row_white_counts < threshold)[0]
    
    # 检测结果
    if len(break_rows) > 0:
        print("检测到激光线断裂！断裂行：", break_rows)
    else:
        print("激光线连续，未检测到断裂")
    
    # 检查是否有异物（假设异物导致局部异常，可通过轮廓分析）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        print("检测到激光线上存在异物或异常！")
    else:
        print("激光线无明显异物")
    
    # 显示处理后的掩码图像（可选）
    cv2.imshow('Processed Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def corner_detect_filter_false_alarm(gray, blockSize=15, ksize=15, k=0.04, ccn_thr=5):
    """
    ccn_thr: 连通域数量阈值
    正常情况下, 就几个地方会有结果
    假如角点检测的结果非常多, 那么应该是误报.
    """
    gray = np.float32(gray)  # 转换为浮点型
    dst = cv2.cornerHarris(gray, blockSize=blockSize, ksize=ksize, k=k)  # Harris 角点检测

    # 2. 设置阈值筛选角点
    threshold = 0.01 * dst.max()  # 通常取最大响应的 1%~5%
    corners = np.argwhere(dst > threshold)  # 获取符合条件的坐标

    # 3. 转换坐标格式（注意：np.argwhere 返回的是 (y, x) 格式）
    corners_xy = corners[:, [1, 0]]  # 转换为 (x, y) 格式

    # 4. 可视化角点（可选）
    img_vis = np.zeros(shape=(gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    for (x, y) in corners_xy:
        cv2.circle(img_vis, (x, y), 3, (255, 0, 255), -1)  # 在图像上绘制红色圆点

    img_vis_gray = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)
    _, img_vis_thr = cv2.threshold(img_vis_gray, 0, 255, cv2.THRESH_BINARY)
    img_vis_thr = np.uint8(img_vis_thr)
    img_vis_thr = cv2.dilate(img_vis_thr, None)   
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_vis_thr)

    if num_labels - 1 > ccn_thr:
        return True, num_labels, labels, stats, centroids
    else:
        return False, num_labels, labels, stats, centroids


def corner_detect_crop_img(img, expand_pixels=25):
    imgsz = img.shape[:2]
    img_orig= img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    falseAlarm, num_labels, labels, stats, centroids = corner_detect_filter_false_alarm(gray, blockSize=15, ksize=15, k=0.04, ccn_thr=3)
    centroids_new = sorted(centroids[1:], key=lambda x: x[0])
    stats_new = sorted(stats[1:], key=lambda x: x[0])

    crops = []
    for i in range(len(stats_new)):
        bi = stats_new[i][:-1]
        bi_center = [bi[0] + bi[2] / 2, bi[1] + bi[3] / 2]
        bi_expand = expand_bbox(bi, imgsz, expand_pixels)

        cropped = img_orig[bi_expand[1]:bi_expand[3], bi_expand[0]:bi_expand[2]]
        crops.append(cropped)

    return crops


def corner_detect_crop_img_main():
    # data_path = r"D:\Gosion\data\006.Belt_Torn_Det\data\ningmei\data (1)"
    # save_path = r"G:\Gosion\data\001.Belt_Torn_Detection\data\corner_detect_cropped_52"
    # os.makedirs(save_path, exist_ok=True)

    # dir_list = os.listdir(data_path)
    # for d in dir_list:
    #     dir_path = data_path + "/{}/src".format(d)
    #     file_list = os.listdir(dir_path)
    #     for f in file_list:
    #         fname = os.path.splitext(f)[0]
    #         img_path = dir_path + "/{}".format(f)
    #         img = cv2.imread(img_path)
    #         crops = corner_detect_crop_img(img)
    #         for i, crop in enumerate(crops):
    #             cv2.imwrite(save_path + "/{}_{}_{}.jpg".format(d, fname, i), crop)


    save_path = r"G:\Gosion\data\006.Belt_Torn_Detection\data\data_51_20250422_vis"
    os.makedirs(save_path, exist_ok=True)

    data_path = r"G:\Gosion\data\006.Belt_Torn_Detection\data\data_51\data\20250422\src"
    file_list = os.listdir(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        img_path = data_path + "/{}".format(f)
        img = cv2.imread(img_path)
        crops = corner_detect_crop_img(img)
        for i, crop in enumerate(crops):
            thresh_otsu, corners_xy, img_vis = corner_detect_v4(crop, blockSize=15, ksize=15, k=0.06, vis=True)
            cv2.imwrite(save_path + "/{}_{}.jpg".format(fname, i), img_vis)
                

def resize_ico_image(input_path, output_path, size):
    img = Image.open(input_path)
    # img = img.resize((size, size), Image.ANTIALIAS)
    img = img.resize((size, size))
    img.save(output_path)


def extract_mnist():
    import os
    import torch
    import torchvision
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor
    from PIL import Image

    # 设置参数
    download_path = 'F:/downloads/MNIST/MNIST_data'  # 数据集下载路径
    save_path = 'F:/downloads/MNIST/MNIST_images'  # 图片保存路径
    batch_size = 64                 # 数据加载批大小

    # 创建保存目录
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # 下载训练集和测试集
    train_data = datasets.MNIST(
    root=download_path,
    train=True,
    transform=ToTensor(),
    download=True
    )

    test_data = datasets.MNIST(
    root=download_path,
    train=False,
    transform=ToTensor(),
    download=True
    )

    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 保存训练集图片
    for batch_idx, (images, labels) in enumerate(train_loader):
        for i in range(images.size(0)):
        # 生成唯一文件名：类型_批次序号_图片序号_标签.png
            filename = f"train_{batch_idx:04d}_{i:03d}_label{labels[i].item()}.png"
            img_path = os.path.join(save_path, filename)

            # 转换并保存图片（移除批次维度，转换为PIL图像）
            img = Image.fromarray(images[i].squeeze().numpy() * 255).convert('L')
            img.save(img_path)

        print(f"保存训练集批次: {batch_idx+1}/{len(train_loader)}")

    # 保存测试集图片
    for batch_idx, (images, labels) in enumerate(test_loader):
        for i in range(images.size(0)):
            filename = f"test_{batch_idx:04d}_{i:03d}_label{labels[i].item()}.png"
            img_path = os.path.join(save_path, filename)

            img = Image.fromarray(images[i].squeeze().numpy() * 255).convert('L')
            img.save(img_path)

        print(f"保存测试集批次: {batch_idx+1}/{len(test_loader)}")

    print("所有图片保存完成！")
    print(f"训练图片数量: {len(train_data)}")
    print(f"测试图片数量: {len(test_data)}")
    print(f"总图片数量: {len(train_data) + len(test_data)}")
    print(f"图片保存位置: {os.path.abspath(save_path)}")


def save_cifar_images(dataset, dataset_name, split='train'):
    """
    保存CIFAR数据集图片到本地文件夹
    
    参数:
        dataset: CIFAR数据集对象
        dataset_name: 数据集名称 ('cifar10' 或 'cifar100')
        split: 数据集分割类型 ('train' 或 'test')
    """
    # 创建保存目录
    save_dir = os.path.join(os.getcwd(), f"{dataset_name}_{split}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取类别名称
    if dataset_name == 'cifar10':
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck')
    else:  # cifar100
        # 使用细粒度标签名称
        classes = dataset.classes
    
    # 创建类别子目录
    for class_name in classes:
        class_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    print(f"正在保存 {dataset_name} {split} 数据集...")
    
    # 保存每张图片
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        
        # 获取类别名称
        class_name = classes[label]
        
        # 构建保存路径
        class_dir = os.path.join(save_dir, class_name)
        img_path = os.path.join(class_dir, f"{idx:05d}.png")
        
        # 保存图片
        image.save(img_path)
        
        # 每1000张打印一次进度
        if (idx + 1) % 1000 == 0:
            print(f"已保存 {idx+1}/{len(dataset)} 张图片")
    
    print(f"保存完成! 图片已存储在: {save_dir}\n")


def extract_cifar():
    from torchvision.datasets import CIFAR10, CIFAR100

    # 设置数据集下载路径
    data_root_CIFAR10 = "F:\downloads\CIFAR10"
    data_root_CIFAR100 = "F:\downloads\CIFAR100"
    os.makedirs(data_root_CIFAR10, exist_ok=True)
    os.makedirs(data_root_CIFAR100, exist_ok=True)
    
    # 下载并保存CIFAR-10
    for split in ['train', 'test']:
        cifar10_set = CIFAR10(root=data_root_CIFAR10, 
                              train=(split == 'train'), 
                              download=True)
        save_cifar_images(cifar10_set, 'cifar10', split)
    
    # 下载并保存CIFAR-100
    for split in ['train', 'test']:
        cifar100_set = CIFAR100(root=data_root_CIFAR100, 
                                train=(split == 'train'), 
                                download=True)
        save_cifar_images(cifar100_set, 'cifar100', split)
    
    print("所有数据集处理完成!")


def save_caltech_images(dataset, dataset_name, root_dir):
    """
    保存Caltech数据集图片到本地文件夹
    
    参数:
        dataset: Caltech数据集对象
        dataset_name: 数据集名称 ('caltech101' 或 'caltech256')
        root_dir: 根目录路径
    """
    # 创建保存目录
    save_dir = os.path.join(root_dir, f"{dataset_name}_images")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"正在保存 {dataset_name.upper()} 数据集图片...")
    print(f"总图片数量: {len(dataset)}")
    
    # 保存每张图片
    for idx in range(len(dataset)):
        if dataset_name == 'caltech101':
            # Caltech101返回(image, target)其中target是类别索引
            image, target = dataset[idx]
            # 获取类别名称
            class_name = dataset.categories[target]
        else:  # Caltech256
            # Caltech256返回(image, target)其中target是类别索引
            image, target = dataset[idx]
            # 类别名称从文件夹名获取
            class_name = f"{target:03d}"  # 格式化为3位数字
            
        # 创建类别子目录
        class_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # 构建保存路径
        img_path = os.path.join(class_dir, f"{dataset_name}_{idx:06d}.jpg")
        
        # 保存图片为JPEG格式
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image.save(img_path)
        
        # 每1000张打印一次进度
        if (idx + 1) % 1000 == 0:
            print(f"已保存 {idx+1}/{len(dataset)} 张图片")
    
    print(f"{dataset_name.upper()} 保存完成! 图片存储在: {save_dir}")
    print(f"类别数量: {len(os.listdir(save_dir))}\n")


def extract_caltech():
    from torchvision.datasets import Caltech101, Caltech256

    # 设置数据集下载和保存的根目录
    data_root = r"F:\downloads"
    os.makedirs(data_root, exist_ok=True)
    
    # 确保使用最新版本的torchvision（解决Caltech下载问题）
    print(f"torchvision版本: {torchvision.__version__}")
    if torchvision.__version__ < '0.17.1':
        print("警告: 推荐使用torchvision 0.17.1或更高版本来避免Caltech下载问题")
        print("请执行: pip install torchvision --upgrade")
    
    try:
        # 下载并保存Caltech101
        print("="*50)
        print("开始处理Caltech101数据集...")
        caltech101_set = Caltech101(root=data_root + "\Caltech101", 
                                   download=True, 
                                   target_type="category")
        save_caltech_images(caltech101_set, 'caltech101', data_root)
        
        # 下载并保存Caltech256
        print("="*50)
        print("开始处理Caltech256数据集...")
        caltech256_set = Caltech256(root=data_root + "\Caltech256", 
                                   download=True)
        save_caltech_images(caltech256_set, 'caltech256', data_root)
        
        print("="*50)
        print("所有数据集处理完成!")
        print(f"Caltech101图片总数: {len(caltech101_set)}")
        print(f"Caltech256图片总数: {len(caltech256_set)}")
        
    except RuntimeError as e:
        print(f"发生错误: {e}")
        print("可能的解决方案:")
        print("1. 确保使用最新版torchvision: pip install torchvision --upgrade")
        print("2. 安装gdown库: pip install gdown")
        print("3. 手动下载数据集:")
        print("   Caltech101: https://www.juhe.cn/market/product/id/10180")
        print("   Caltech256: 官方源或备用镜像")
        print("4. 将下载的文件放在对应目录后重新运行程序")


def save_fashion_mnist_images(dataset, save_dir):
    """保存数据集中的图片到指定目录"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存图片和标签文件
    for idx, (image, label) in enumerate(dataset):
        # 创建子文件夹：按类别存储
        class_dir = os.path.join(save_dir, str(label))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        # 生成唯一文件名
        img_path = os.path.join(class_dir, f"image_{idx:05d}.png")
        
        # 转换并保存图片
        image = image.squeeze().numpy()
        # image.save(img_path)
        image = cv2.cvtColor(image * 255, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(img_path, image)
        
        # 每1000张打印一次进度
        if idx % 1000 == 0:
            print(f"已保存 {idx}/{len(dataset)} 张图片")
    

def extract_fashion_mnist():
    from torchvision.datasets import FashionMNIST

    # 设置数据集路径
    data_root = "F:\downloads\FashionMNIST"
    os.makedirs(data_root, exist_ok=True)
    
    # 下载训练集和测试集
    train_set = FashionMNIST(
        root=data_root, 
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    test_set = FashionMNIST(
        root=data_root, 
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    print("数据集下载完成！")
    print(f"训练集大小: {len(train_set)} 张图片")
    print(f"测试集大小: {len(test_set)} 张图片")
    
    # 保存图片
    save_fashion_mnist_images(train_set, data_root + "/fashion_mnist_images/train")
    save_fashion_mnist_images(test_set, data_root + "/fashion_mnist_images/test")
    
    print("所有图片保存完成！")
    print(f"训练集保存路径: ./fashion_mnist_images/train")
    print(f"测试集保存路径: ./fashion_mnist_images/test")




if __name__ == '__main__':
    pass
    # iou = cal_iou(bbx1=[0, 0, 10, 10], bbx2=[2, 2, 12, 12])
    # extract_one_gif_frames(gif_path="")
    # extract_one_video_frames(video_path="", gap=5)
    # extract_videos_frames(base_path="", gap=5, save_path="")
    # convert_to_jpg_format(data_path="")
    # convert_to_png_format(data_path="")
    # convert_to_gray_image(data_path="")
    # convert_to_binary_image(data_path="", thr_low=88)
    # crop_image_according_labelbee_json(data_path="", crop_ratio=(1, 1.2, 1.5, ))
    # crop_ocr_rec_img_according_labelbee_det_json(data_path="")
    # crop_image_according_yolo_txt(data_path="", CLS=(0, ), crop_ratio=(1.0, ))  # 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0
    # random_crop_gen_cls_negative_samples(data_path="", random_size=(196, 224, 256, 288, 384), randint_low=1, randint_high=4, hw_dis=100, dst_num=1000)
    # seg_object_from_mask(base_path="")


    # ======== Object detection utils ========
    # labelbee2yolo(data_path="", copy_image=True)
    # labelbee2voc(data_path="")  # TODO
    # labelbee2coco(data_path="")  # TODO
    # yolo2labelbee(data_path="")
    # yolo2voc(data_path="")  # TODO
    # yolo2coco(data_path="")  # TODO
    # voc2labelbee(data_path="", classes=['dog', ], val_percent=0.1)
    # voc2yolo(data_path="", classes=['dog', ], val_percent=0.1)
    # voc2coco(data_path="", classes=['dog', ], val_percent=0.1)
    # coco2labelbee(data_path="")
    # coco2yolo(data_path="")
    # coco2voc(data_path="")
    # labelbee_kpt_to_yolo(data_path="", copy_image=False)
    # labelbee_kpt_to_dbnet(data_path="", copy_image=True)
    # labelbee_kpt_to_labelme_kpt(data_path="")
    # labelbee_kpt_to_labelme_kpt_multi_points(data_path="")
    # labelbee_seg_to_png(data_path="")

    # convert_Stanford_Dogs_Dataset_annotations_to_yolo_format(data_path="")
    # convert_WiderPerson_Dataset_annotations_to_yolo_format(data_path="")
    # convert_TinyPerson_Dataset_annotations_to_yolo_format(data_path="")
    # convert_AI_TOD_Dataset_to_yolo_format(data_path="")

    # random_select_yolo_images_and_labels(data_path="", select_num=500, move_or_copy="copy", select_mode=0)
    # vis_yolo_label(data_path="", print_flag=False, color_num=1000, rm_small_object=False, rm_size=32)  # TODO: 1.rm_small_object have bugs.
    # list_yolo_labels(label_path=r"G:\Gosion\data\007.PPE_Det\data\helmet\labels")
    # change_txt_content(txt_base_path="")
    # remove_yolo_txt_contain_specific_class(data_path="", rm_cls=(0, ))
    # remove_yolo_txt_small_bbx(data_path="", rm_cls=(0, ), rmsz=(48, 48))
    # select_yolo_txt_contain_specific_class(data_path="", select_cls=(3, ))
    # merge_txt(path1="", path2="")
    # merge_txt_files(data_path="")


    # ======== OCR ========
    # dbnet_aug_data(data_path="", bg_path="", maxnum=10000)
    # vis_dbnet_gt(data_path="")
    # warpPerspective_img_via_labelbee_kpt_json(data_path="")
    # alpha = read_ocr_lables(lbl_path="")  # alpha = ' ' + '0123456789' + '.:/\\-' + 'ABbC'
    # check_ocr_label(data_path="", label=alpha)
    # ocr_data_gen_train_txt_v2(data_path="", LABEL=alpha)
    # random_select_files_according_txt(data_path="", select_percent=0.25)
    # make_border_v7(img, (64, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # ocr_data_gen_train_txt(data_path="", LABEL=alpha)
    # ocr_data_gen_train_txt_v2(data_path="", LABEL=alpha)
    # ocr_data_merge_train_txt_files_v2(data_path="", LABEL=alpha)
    # random_select_files_according_txt(data_path="", select_percent=0.25)
    # random_select_files_from_txt(data_path="", n=2500)
    # convert_text_renderer_json_to_my_dataset_format(data_path="")
    # convert_Synthetic_Chinese_String_Dataset_labels(data_path="")
    # convert_mtwi_to_ocr_rec_data(data_path="")
    # convert_ShopSign_to_ocr_rec_data(data_path="")
    # ocr_train_txt_change_to_abs_path()
    # get_ocr_train_txt_alpha(data_path="")
    # check_ocr_train_txt(data_path="")
    # random_select_images_from_ocr_train_txt(data_path="", select_num= 5000)
    # ocr_train_txt_split_to_train_and_test(data_path="", train_percent=0.8)

    # byte_data = b""
    # img = byte2img(byte_data)
    # cv2.imwrite(r'G:\Gosion\data\000.Test_Data\images\test_res_20250611.jpg', img)

    # img_path = "G:\\Gosion\\data\\000.Test_Data\\images\\20250520142551.png"
    # byte_data = img2byte(img_path)
    # # print(byte_data)

    # with open(r'D:\Gosion\code\gitee\BeltTornDetection_C++\src\BeltTornDetection_C++\BeltTornDetection_C++\ZBase64_test_python.txt', 'w') as f:
    #     f.write(byte_data.decode('utf-8'))


    # txt = open(r"D:\Gosion\code\gitee\BeltTornDetection_C++\src\BeltTornDetection_C++\BeltTornDetection_C++\ZBase64_test2.txt", "r")
    # data = txt.readlines()
    # txt.close()

    # data_new = ""
    # for i in data:
    #     data_new += i.strip().replace("\n", "")

    # # byte_data = b"{}".format(data_new)
    # byte_data = bytes(data_new.encode("utf-8"))
    # img = byte2img(byte_data)
    # cv2.imwrite(r'G:\Gosion\data\000.Test_Data\images\test_res_20250611_2.jpg', img)



    # img = cv2.imread(r'D:\Gosion\Projects\data\202206070916487.png')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    # output, num_labels, labels, stats, centroids = connected_components_analysis(thresh, connectivity=8, area_thr=100, h_thr=8, w_thr=8)
    # cv2.imwrite(r'D:\Gosion\Projects\data\202206070916487_output2.jpg', output)

    # video_path = r"D:\GraceKafuu\Resources\vtest.avi"
    # # video_path = r"D:\Gosion\Projects\data\project_data\6870\270\192.168.45.192_01_20250109093741369.mp4"
    # # moving_object_detect(video_path=video_path, m=3, area_thresh=100, scale_r=(0.5, 0.5), time_watermark=[[0, 0.0488, 0.4370, 0.0651]], cca=True, flag_merge_bboxes=True, vis_result=True, save_path=None, debug=True)
    # moving_object_detect(video_path=video_path, m=3, area_thresh=100, scale_r=None, time_watermark=[[0, 0.0488, 0.4370, 0.0651]], cca=True, flag_merge_bboxes=True, vis_result=True, save_path=None, debug=True)

    # convertor = Labelme2YOLO(json_dir=r"D:\Gosion\Projects\002.Smoking_Det\001\jsons", to_seg=False)
    # convertor.convert(val_size=0.1)

    # yolo2labelme(data_path=r"D:\Gosion\Projects\002.Smoking_Det\002", out=None, skip=True)

    # change_txt_content(txt_path=r"G:\Gosion\data\007.PPE_Det\data\v1\fande_yolo_format\labels")
    # for i in range(10, 12):
    #     change_txt_content(txt_path=r"D:\Gosion\Projects\004.GuardArea_Det\data\v2_labelbee_format\{}_yolo_format\labels".format(i))

    # yolo_label_expand_bbox(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\001", classes=1, r=1.5)

    # yolo_to_labelbee(data_path=r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v2\New\zhanting_v2\train", copy_images=False)  # yolo_format 路径下是 images 和 labels
    # labelbee_to_yolo(data_path=r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v2\500_labelbee_format", copy_images=True)  # labelbee_format 路径下是 images 和 jsons

    # labelme_det_kpt_to_yolo_labels(data_path=r"D:\Gosion\Projects\006.Belt_Torn_Det\data\det_pose\v1\v1", class_list=["torn"], keypoint_list=["p1", "p2"])
    # labelbee_multi_step_det_kpt_to_yolo_labels(data_path=r"G:\Gosion\data\006.Belt_Torn_Detection\data\ningmei\nos", save_path="", copy_images=True, small_bbx_thresh=3, cls_plus=-1)
    # det_kpt_yolo_labels_to_labelbee_multi_step_json(data_path=r"G:\Gosion\data\006.Belt_Torn_Detection\data\kpt\src_selected\src", save_path="", copy_images=True, small_bbx_thresh=3, cls_plus=1, return_decimal=True)
    # labelbee_seg_json_to_yolo_txt(data_path=r"G:\Gosion\data\009.TuoGun_Det\obb\v1", cls_plus=-1)

    # voc_to_yolo(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\009", classes={"0": "smoke"})
    # voc_to_yolo(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\002", classes={"0": "smoking"})

    # random_select_yolo_images_and_labels(data_path=r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v1\train".replace("\\", "/"), select_num=2500, move_or_copy="move", select_mode=0)

    # ffmpeg_extract_video_frames(video_path=r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v2\pexels", fps=3)

    # crop_image_via_yolo_labels(data_path=r"D:\Gosion\Projects\001.Leaking_Liquid_Det\data\DET\v2\val", CLS=(0, 1), crop_ratio=(1, ))

    # vis_yolo_labels(data_path=r"G:\Gosion\data\007.PPE_Det\data\v1\val")

    # process_small_images(img_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\001_labelbee_format\images", size=256, mode=0)

    # remove_yolo_label_specific_class(data_path=r"G:\Gosion\data\007.PPE_Det\data\helmet", rm_cls=(2, ))

    # make_border_and_change_yolo_labels(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\v4_exp_make_border\train_base", dstsz=(1080 + 1920, 1920 + 1920))

    
    # p1 = np.array([0, 0])
    # p2 = np.array([1, 0])
    # p3 = np.array([1, 1])
    # n = cal_angle_via_vector_cross(p1, p2, p3)
    # print(n)

    # append_content_to_txt_test()
    # append_jitter_box_yolo_label(data_path=r"D:\Gosion\Projects\001.Leaking_Liquid_Det\data\20250219\v2\train\001_yolo_format", append_label=["0 0.5130208333333334 0.18055555555555555 0.20625 0.3277777777777778"], p=5)

    # jitter_bbox(data_path=r"D:\Gosion\Projects\004.GuardArea_Det\data\v2_new", classes=(1, ), p=3)

    # check_yolo_labels(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\v4\train")

    # delete_yolo_labels_high_iou_bbox(data_path=r"G:\Gosion\data\007.PPE_Det\data\v1\no_person", iou_thr=0.90, target_cls=(0, 1), del_cls=0)

    # select_specific_images_and_labels(data_path=r"D:\Gosion\Projects\003.Violated_Sitting_Det\data\v2\train")

    # process_corrupt_images(img_path=r"D:\Gosion\Projects\data\silie_data\train\tear", algorithm="cv2", flag="move")

    # remove_corrupt_image(data_path=r"D:\Gosion\Projects\data\silie_data\train\not_torn")

    # data_path = r"D:\Gosion\Projects\GuanWangLNG\20250318"
    # save_path = make_save_path(data_path, relative='.', add_str="result")
    # file_list = get_file_list(data_path)
    # ratio_list = []
    # for f in file_list:
    #     fname = os.path.splitext(f)[0]
    #     f_abs_path = os.path.join(data_path, f)
    #     print("{}: ".format(f_abs_path))
    #     img = cv2.imread(f_abs_path)
    #     # img = cv2.imdecode(np.fromfile(f_abs_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    #     res, color_area_ratio, rs = cal_area_ratio_of_sepecific_color(img, lower=(0, 0, 200), upper=(180, 30, 255), apply_mask=True)
    #     ratio_list.append(color_area_ratio)
    #     res_path = r"{}/{}_{}.jpg".format(save_path, fname, rs.replace("%", ""))
    #     cv2.imwrite(res_path, res)

    # mean_r = np.mean(ratio_list)
    # min_r = np.min(ratio_list)
    # max_r = np.max(ratio_list)
    # print("mean_r: {} %, min_r: {} %, max_r: {} %".format(mean_r, min_r, max_r))

    # --------------------------------------------------------------------------------------
    # extract_parabolic_curve_area(img_path=r"D:\Gosion\Projects\data\images\southeast.jpg")

    # data_path = r"D:\Gosion\Projects\006.Belt_Torn_Det\data\pose\v3\train_aug_0\images"
    # file_list = get_file_list(data_path)
    # for f in file_list:
    #     f_abs_path = data_path + "/{}".format(f)
    #     img = cv2.imread(f_abs_path)
    #     brightness = cal_brightness_v2(img)
    #     print("{}: {}".format(f_abs_path, brightness))
    # 


    # img_path = r"D:\Gosion\Projects\006.Belt_Torn_Det\data\video\video_frames\cropped\20250308_frames_merged\Video_2025_03_08_111948_1_output_000003979.jpg"
    # img = cv2.imread(img_path)
    # corner_detect(img)    

    # corner_detect_test()
    # polyfit_test()

    # classify_image_by_brightness(data_path=r"D:\Gosion\Projects\006.Belt_Torn_Det\data\cls\v5\train\1\000", show_brightness=False)

    # create_cls_negatives_via_random_crop(data_path=r"E:\wujiahu\coco\train2017", random_size=(64, 96, 100, 128, 160, 180, 200, 256), randint_low=2, randint_high=6, hw_dis=100, dst_num=5000)

    # seamless_clone(fg_path=r"D:\Gosion\Projects\006.Belt_Torn_Det\data\cls\v5\train\Random_Selected\2_random_selected_100", bg_path=r"E:\wujiahu\coco\Random_Selected\train2017_random_cropped_random_selected_500", dst_num=5000)

    # img_path = r"E:\wujiahu\006.Belt_Torn_Det\data\20250327\src\1743044234.5645235.jpg"
    # img = cv2.imread(img_path)
    # res = bilateral_filter(img)
    # save_path = img_path.replace(".jpg", "_bilateral_filter.jpg")
    # cv2.imwrite(save_path, res)

    # btd_test()

    # # 使用示例
    # result = detect_gap(r"D:\Gosion\Projects\006.Belt_Torn_Det\data\rare_samples\20250327203539000.png")
    # if result:
    #     print(f"缺口起始位置：{result[0]}，缺口宽度：{result[1]}")

    # image_path = r"D:\Gosion\Projects\006.Belt_Torn_Det\data\rare_samples\20250327203539000.png"
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # save_path = image_path.replace(".png", "_binary.png")
    # cv2.imwrite(save_path, binary)

    # image_path = r"E:\wujiahu\006.Belt_Torn_Det\data\A7300MG30_DE36009AAK00231_frames\cross_test\Video_2025_03_27_172528_4_output_0000000010000.jpg"
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # save_path = image_path.replace(".jpg", "_binary.jpg")
    # cv2.imwrite(save_path, binary)

    # image_path = r"E:\wujiahu\006.Belt_Torn_Det\data\A7300MG30_DE36009AAK00231_frames\cross_test\Video_2025_03_27_172528_4_output_0000000010000_binary.jpg"
    # img = cv2.imread(image_path)

    # # k1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    # # k2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    # k1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]) * 20
    # k2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]) * 20

    # res1 = cv2.filter2D(img, -1, k1)
    # res2 = cv2.filter2D(img, -1, k2)
    # cv2.imwrite(image_path.replace(".jpg", "_res1.jpg"), res1)
    # cv2.imwrite(image_path.replace(".jpg", "_res2.jpg"), res2)


    # image_path = r"E:\wujiahu\006.Belt_Torn_Det\data\A7300MG30_DE36009AAK00231_frames\cross_test\Video_2025_03_27_172528_4_output_0000000010000_binary.jpg"
    # img = cv2.imread(image_path)
    # img_vis, dst_bbx_pts, wides = corner_detect_v2(img, expand_pixels=25, brightness_thresh=2, dilate_pixels=2)
    # cv2.imwrite(image_path.replace(".jpg", "_corner_detect_v2.jpg"), img_vis)

    # file_path = r"E:\Gosion\data\006.Belt_Torn_Det\data\皮带撕裂检测日志"
    # stats_btd(file_path=file_path, start=5, end=168, baseline=[2, [2, 6]])

    # btd_test_20250401()
    # test_20250401()

    # # data_path = r"D:\Gosion\data\006.Belt_Torn_Det\data\video\video_frames\cropped\20250308_frames_merged"
    # data_path = r"E:\Gosion\data\006.Belt_Torn_Det\data\yitishi\bgr\data\20250331\src"
    # save_path = make_save_path(data_path, relative='.', add_str="extract_specific_color_result")

    # file_list = get_file_list(data_path)
    # for f in file_list:
    #     fname = os.path.splitext(f)[0]
    #     f_abs_path = data_path + "/{}".format(f)
    #     img = cv2.imread(f_abs_path)
    #     mask, result, binary, binary_otsu = extract_specific_color(img, lower=None, upper=None, color="green")
    #     bitand = cv2.bitwise_and(binary, binary_otsu)
    #     bitor = cv2.bitwise_or(binary, binary_otsu)

    #     points = np.where(binary_otsu == 255)
    #     points = np.hstack((points[1].reshape(-1, 1), points[0].reshape(-1, 1)))
    #     # 运行RANSAC
    #     model = ransac_parabola(points, select_num=1000, num_iterations=1000, threshold=50)
    #     if model is not None:
    #         a, b, c = model
    #         print(f"拟合的抛物线方程: y = {a:.12f}x² + {b:.12f}x + {c:.12f}")
            
    #         img = plot_fit_parabola(img, model, color=(255, 0, 255))

    #     else:
    #         print("拟合失败")

    #     # # a = 2.845889*1e-4
    #     # a = 3.905889*1e-4
    #     # b = -0.413223
    #     # # b = -0.453223
    #     # c = 200
    #     # model = (a, b, c)
    #     # img = plot_fit_parabola(img, model, color=(255, 0, 255))


    #     fit = np.polyfit(points[:, 0], points[:, 1], 2)
    #     f = np.poly1d(fit)

    #     img = plot_fit_parabola(img, fit, color=(255, 255, 0))

    #     # fit_params, pcov = scipy.optimize.curve_fit(parabola, points[:, 0], points[:, 1])
    #     # y_fit = parabola(points[:, 0], *fit_params)

    #     save_path_img_parabola = "{}/{}_img_parabola.jpg".format(save_path, fname)
    #     save_path_mask = "{}/{}_mask.jpg".format(save_path, fname)
    #     save_path_result = "{}/{}_result.jpg".format(save_path, fname)
    #     save_path_binary = "{}/{}_binary.jpg".format(save_path, fname)
    #     save_path_binary_otsu = "{}/{}_binary_otsu.jpg".format(save_path, fname)
    #     save_path_bitand = "{}/{}_bitand.jpg".format(save_path, fname)
    #     save_path_bitor = "{}/{}_bitor.jpg".format(save_path, fname)
    #     cv2.imwrite(save_path_img_parabola, img)
    #     cv2.imwrite(save_path_mask, mask)
    #     cv2.imwrite(save_path_result, result)
    #     cv2.imwrite(save_path_binary, binary)
    #     cv2.imwrite(save_path_binary_otsu, binary_otsu)
    #     cv2.imwrite(save_path_bitand, bitand)
    #     cv2.imwrite(save_path_bitor, bitor)


    # # 生成测试数据
    # np.random.seed(42)
    # img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # x = np.linspace(0, 1000, 1000 - 1)
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
    #     # y = -0.017x² + 7.754x + 578.284
    #     a = -0.017
    #     b = 7.754
    #     c = 578.284
    #     img = plot_fit_parabola(img, model, color=(255, 0, 255))
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     print("拟合失败")

    # test_20250407()
    # test_20250407_2()
    # main_test_20250408()

    # select_specific_hw_images(data_path=r"G:\Gosion\data\007.PPE_Det\data\v1\all", hw=(704, 576))


    # # 读取图像
    # img1 = cv2.imread(r"G:\Gosion\data\008.OilLevel_Det\data\xfeat\data_xfeat_matching_results_\192.168.31.101_01_20240618151241754_warped.jpg")
    # img2 = cv2.imread(r"G:\Gosion\data\008.OilLevel_Det\data\xfeat\data\192.168.31.101_01_20240618151241754.jpg")

    # if img1 is None or img2 is None:
    #     raise ValueError("无法读取图像文件")

    # # 计算互相关
    # correlation = compute_cross_correlation(img1, img2)

    # # 归一化结果以便显示
    # correlation_norm = (correlation - np.min(correlation)) / (np.max(correlation) - np.min(correlation))

    # # 显示结果
    # # cv2.imshow('Cross Correlation', correlation_norm)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # cv2.imwrite(r"G:\Gosion\data\008.OilLevel_Det\data\xfeat\data\192.168.31.101_01_20240618151241754_warped_correlation.jpg", correlation_norm * 255)



    # # 读取图像
    # img1_path = r"G:\Gosion\data\008.OilLevel_Det\data\xfeat\data_xfeat_matching_results_\192.168.31.101_01_20240618151241754_warped.jpg"
    # img2_path = r"G:\Gosion\data\008.OilLevel_Det\data\xfeat\data\192.168.31.101_01_20240618151241754.jpg"

    # img_rgb = cv2.imread(img2_path)
    # assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # template = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    # assert template is not None, "file could not be read, check with os.path.exists()"
    # w, h = template.shape[::-1]

    # res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # threshold = 0.35
    # loc = np.where( res >= threshold)
    # for pt in zip(*loc[::-1]):
    #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    # cv2.imwrite(r'G:\Gosion\data\008.OilLevel_Det\data\xfeat\res.png', img_rgb)


    # img1_path = r"G:\Gosion\data\008.OilLevel_Det\data\xfeat\data_xfeat_matching_results_\192.168.31.101_01_20240618151241754_warped.jpg"
    # img2_path = r"G:\Gosion\data\008.OilLevel_Det\data\xfeat\data\192.168.31.101_01_20240618151241754.jpg"


    # img = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    # img2 = img.copy()
    # template = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    # w, h = template.shape[::-1]

    # methods = ['TM_SQDIFF', 'TM_SQDIFF_NORMED', 'TM_CCOEFF', 
    #        'TM_CCOEFF_NORMED', 'TM_CCORR',  'TM_CCORR_NORMED']
    
    # for m in methods:
    #     img = img2.copy()
    #     method = getattr(cv2, m)

    #     res = cv2.matchTemplate(img, template, method)
    #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #         top_left = min_loc
    #     else:
    #         top_left = max_loc

    #     bottom_right = (top_left[0] + w, top_left[1] + h)
    #     cv2.rectangle(img, top_left, bottom_right, 255, 2)

    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    
    # from skimage.feature import match_template

    # img1_path = r"G:\Gosion\data\008.OilLevel_Det\data\xfeat\data_xfeat_matching_results_\192.168.31.101_01_20240618151241754_warped.jpg"
    # img2_path = r"G:\Gosion\data\008.OilLevel_Det\data\xfeat\data\192.168.31.101_01_20240618151241754.jpg"

    # # image = data.coins()
    # # coin = image[170:220, 75:130]

    # img1 = cv2.imread(img1_path)
    # img2 = cv2.imread(img2_path)

    # result = match_template(img2, img1)
    # ij = np.unravel_index(np.argmax(result), result.shape)
    # x, y = ij[::-1]

    # fig = plt.figure(figsize=(8, 3))
    # ax1 = plt.subplot(1, 3, 1)
    # ax2 = plt.subplot(1, 3, 2)
    # ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    # ax1.imshow(img1, cmap=plt.cm.gray)
    # ax1.set_axis_off()
    # ax1.set_title('template')

    # ax2.imshow(img2, cmap=plt.cm.gray)
    # ax2.set_axis_off()
    # ax2.set_title('image')
    # # highlight matched region
    # hcoin, wcoin = img1.shape
    # rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
    # ax2.add_patch(rect)

    # ax3.imshow(result)
    # ax3.set_axis_off()
    # ax3.set_title('`match_template`\nresult')
    # # highlight matched region
    # ax3.autoscale(False)
    # ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    # plt.show()

    # main_fit_curve_20250417()
    
    # fit_curve_test(n=1, select_num=2, num_iterations=100, threshold=5, color=(255, 0, 255))

    # import multiprocessing as mp
    # import queue
    # q = mp.Queue()
    # p1 = mp.Process(target=producer, args=(q,))
    # p2 = mp.Process(target=consumer, args=(q,))

    # p1.start()
    # p2.start()

    # while not q.empty():
    #     print(q.get())

    # # 通知消费者队列结束
    # q.put(None)

    # p1.join()
    # p2.join()

    # corner_detect_test_20250507()

    # img_path = r"G:\Gosion\data\001.Belt_Torn_Detection\data\data_51\data\20250507\src\1746601585.749494.jpg"
    # check_laser_conditions(img_path)

    # corner_detect_crop_img_main()

    # extract_mnist()
    # extract_cifar()
    # extract_caltech()
    extract_fashion_mnist()



    















    