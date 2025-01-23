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
            np.clip(v, 0, 255)
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
        np.clip(v, 0, 255)
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


def bbox_yolo_to_voc(imgsz, bbx):
    """
    YOLO --> VOC
    !!!!!! orig: (bbx, imgsz) 20230329 changed to (imgsz, bbx)
    :param bbx: yolo format bbx
    :param imgsz: [H, W]
    :return: [x_min, y_min, x_max, y_max]
    """
    bbx_ = (bbx[0] * imgsz[1], bbx[1] * imgsz[0], bbx[2] * imgsz[1], bbx[3] * imgsz[0])
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

    result = []
    for i in range(len(bbx)):
        result_dict = {}
        result_dict["x"] = bbx[i][0]
        result_dict["y"] = bbx[i][1]
        result_dict["width"] = bbx[i][2] - bbx[i][0]
        result_dict["height"] = bbx[i][3] - bbx[i][1]
        result_dict["attribute"] = "{}".format(bbx[i][4])
        result_dict["valid"] = True
        id_ = random.sample(chars, 8)
        result_dict["id"] = "".join(d for d in id_)
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

                        bb = bbox_voc_to_yolo((h, w), (x_min, y_min, x_max, y_max))
                        p_res = convert_points((w, h), xy_)

                        txt_content = "0" + " " + " ".join([str(a) for a in bb]) + " " + " ".join([str(c) for c in p_res]) + "\n"
                        fw.write(txt_content)
                else:
                    x_min, x_max = min(x_), max(x_)
                    y_min, y_max = min(y_), max(y_)

                    bb = bbox_voc_to_yolo((h, w), (x_min, y_min, x_max, y_max))
                    txt_content = "0" + " " + " ".join([str(a) for a in bb]) + "\n"
                    fw.write(txt_content)

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
                cls = int(l.strip().split(" ")[0])
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
    

def create_cls_negatives_via_random_crop(data_path, random_size=(96, 100, 128, 160), randint_low=10, randint_high=51, hw_dis=100, dst_num=20000):
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
            for line in txt_content:
                l = line.strip().split(" ")
                cls_new = int(l[0]) + 1
                l_new= str(cls_new) + " " + " ".join([str(a) for a in l[1:]]) + '\n'
                fw.write(l_new)


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


def ffmpeg_extract_video_frames(video_path):
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

        command = f"D:/installed/ffmpeg-7.0.2-essentials_build/bin/ffmpeg.exe -i {f_abs_path} -r 5 -q:v 1 -f image2 {f_dst_path}/{fname}_output_%09d.jpg"
        os.system(command)










if __name__ == '__main__':
    # pass
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
    # list_yolo_labels(label_path="")
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

    # byte_data = "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"
    # img = byte2img(byte_data)
    # cv2.imwrite(r'D:\Gosion\Projects\data\res.jpg', img)

    # img_path = r'D:\Gosion\Projects\data\res2.jpg'
    # byte_data = img2byte(img_path)
    # print(byte_data)

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

    # change_txt_content(txt_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\001\labels")
    # yolo_label_expand_bbox(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\001", classes=1, r=1.5)

    # yolo_to_labelbee(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\demo")  # yolo_format 路径下是 images 和 labels
    # labelbee_to_yolo(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\demo_labelbee_format")  # labelbee_format 路径下是 images 和 jsons

    # voc_to_yolo(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\009", classes={"0": "smoke"})
    # voc_to_yolo(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\002", classes={"0": "smoking"})

    # random_select_yolo_images_and_labels(data_path=r"D:\Gosion\Projects\001.Leaking_Det\data\v1\train".replace("\\", "/"), select_num=148, move_or_copy="move", select_mode=0)

    # ffmpeg_extract_video_frames(video_path=r"D:\Gosion\Projects\管网LNG\data\192.168.45.192_01_20250115163057108")

    crop_image_via_yolo_labels(data_path=r"D:\Gosion\Projects\001.Leaking_Det\data\DET\v1\val", CLS=(0, 1), crop_ratio=(1, ))

    # vis_yolo_labels(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\001")

    # process_small_images(img_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\001_labelbee_format\images", size=256, mode=0)

    # remove_yolo_label_specific_class(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\Add\Det\v4\001_labelbee_format_yolo_format", rm_cls=(0,))



    
    

    



















    