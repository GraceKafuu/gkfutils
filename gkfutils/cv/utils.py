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

from ..utils import *

import os
import re
import sys
import cv2
import json
import time
import math
import copy
import glob
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
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import skimage
import scipy
import torch
import torchvision
import onnxruntime
from torchvision import transforms
import matplotlib.pyplot as plt


# ========================================================================================================================================================================
# ========================================================================================================================================================================
# FILE PROCESS
def convertBboxVOC2YOLO(size, box):
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


def convertBboxYOLO2VOC(size, bbx):
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


def convertBboxXML2TXT(img_name, data_path, classes):
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
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        bb = convertBboxVOC2YOLO((h, w), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    return class_names


def yolo2labelbee(data_path):
    img_path = data_path + "/images"
    txt_path = data_path + "/labels"
    json_path = data_path + "/jsons"
    os.makedirs(json_path, exist_ok=True)

    txt_list = sorted(os.listdir(txt_path))

    for t in tqdm(txt_list):
        base_name = os.path.splitext(t)[0]
        txt_abs_path = txt_path + "/{}".format(t)
        try:
            img_abs_path, json_abs_path, size = get_img_json_path_and_size(img_path, json_path, base_name)

            bbx_for_json = []
            with open(txt_abs_path, "r", encoding="utf-8") as fo:
                lines = fo.readlines()
                for l in lines:
                    l_ = l.strip().split(" ")
                    # bbx_ = [float(l_[1]), float(l_[2]), float(l_[3]), float(l_[4])]
                    # VOC_bbx = convert_bbx_yolo_to_VOC(size, bbx_)
                    bbx_ = [float(l_[1]), float(l_[2]), float(l_[3]), float(l_[4])]
                    VOC_bbx = convertBboxYOLO2VOC(size, bbx_)
                    VOC_bbx = list(VOC_bbx)

                    w_, h_ = VOC_bbx[2] - VOC_bbx[0], VOC_bbx[3] - VOC_bbx[1]
                    if w_ < 3 or h_ < 3:
                        print("img_abs_path: ", img_abs_path)
                        print("txt_abs_path: ", txt_abs_path)

                    VOC_bbx.append(int(l_[0]) + 1)
                    bbx_for_json.append(VOC_bbx)

            with open(json_abs_path, "w", encoding="utf-8") as jfw:
                jfw.write(json.dumps(write_labelbee_det_json(bbx_for_json, size)))

            # print("labelbee json saved to --> {}".format(json_abs_path))

        except Exception as Error:
            print(Error)
            print("txt_abs_path: ", txt_abs_path)


def voc2yolo(data_path, classes, val_percent=0.1):
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


def labelbee2yolo(data_path, copy_image=False):
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
                    bb = convertBboxVOC2YOLO((h, w), (x_min, y_min, x_max, y_max))
                    txt_content = "{}".format(cls_id) + " " + " ".join([str(b) for b in bb]) + "\n"
                    # txt_content = "{}".format(cls_id - 1) + " " + " ".join([str(b) for b in bb]) + "\n"
                    # txt_content = "{}".format(cls_id + 1) + " " + " ".join([str(b) for b in bb]) + "\n"
                    fw.write(txt_content)

        except Exception as Error:
            print(Error)


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
            cv2img = cv2.imread(img_abs_path)
            h, w = cv2img.shape[:2]

            # txt_new_abs_path = txt_new_path + "/{}.txt".format(img_name)
            # txt_data_new = open(txt_new_abs_path, "w", encoding="utf-8")

            with open(txt_abs_path, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
                for l_orig in lines:
                    l = l_orig.strip()
                    cls = int(l.split(" ")[0])
                    l_ = [float(l.split(" ")[1]), float(l.split(" ")[2]), float(l.split(" ")[3]), float(l.split(" ")[4])]
                    # bbx_VOC_format = convert_bbx_yolo_to_VOC((h, w), l_)
                    bbx_VOC_format = convertBboxYOLO2VOC((h, w), l_)
                    # if rm_small_object:
                    #     ow, oh = bbx_VOC_format[2] - bbx_VOC_format[0], bbx_VOC_format[3] - bbx_VOC_format[1]
                    #     if ow >= rm_size and oh >= rm_size:
                    #         txt_data_new.write(l_orig)

                    cv2.rectangle(cv2img, (bbx_VOC_format[0], bbx_VOC_format[1]), (bbx_VOC_format[2], bbx_VOC_format[3]), (int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])), 2)
                    cv2.putText(cv2img, "{}".format(cls), (bbx_VOC_format[0], bbx_VOC_format[1] - 4), cv2.FONT_HERSHEY_PLAIN, 2, (int(colors[cls][0]), int(colors[cls][1]), int(colors[cls][2])))

                    cv2.imwrite("{}/{}".format(vis_path, img), cv2img)
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


def random_select_images_and_labels(data_path, select_num=1000, move_or_copy="copy", select_mode=0):
    orig_img_path = data_path + "/images"
    orig_lbl_path = data_path + "/labels"
    data_list = sorted(os.listdir(orig_img_path))

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
        bb = convertBboxVOC2YOLO((h, w), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    return class_names


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

            cv2img = cv2.imread(img_s_abs_path)
            imgsz = cv2img.shape[:2]

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
                        bbx_voc = convertBboxYOLO2VOC(imgsz, list(bbx[b]))
                        det_bbxs.append(bbx_voc)

            dst_lbl_path = save_lbl_path + "/{}.txt".format(fname)
            with open(dst_lbl_path, "w", encoding="utf-8") as fwdk:
                with open(kpt_s_abs_path, "r", encoding="utf-8") as frk:
                    kpt_lines = frk.readlines()

                    for detbbx in det_bbxs:
                        # detbbx_new = [detbbx[0], detbbx[2], detbbx[1], detbbx[3]]
                        # bbx_yolo = convert_bbx_VOC_to_yolo(imgsz, detbbx_new)
                        detbbx_new = [detbbx[0], detbbx[1], detbbx[2], detbbx[3]]
                        bbx_yolo = convertBboxVOC2YOLO(imgsz, detbbx_new)
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


def convert_labelbee_det_one_json_to_yolo_txt(json_path):
    save_path = os.path.abspath(os.path.join(json_path, "../..")) + "/{}".format("labels")
    image_move_path = save_path.replace("labels", "images")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(image_move_path, exist_ok=True)

    json_ = json.load(open(json_path, 'r', encoding='utf-8'))
    len_json = len(json_)
    for i in range(len_json):
        i_data = json_[i]
        url_ = i_data['url']
        fileName_ = i_data["fileName"].strip('\\')

        i_data_result = i_data["result"].replace("true", "True")
        i_data_eval = eval(i_data_result)  # Need to replace "true" --> "True"
        if not i_data_eval: continue
        if not i_data_eval["step_1"]["result"]: continue

        w, h = i_data_eval["width"], i_data_eval["height"]
        step_1_result_data = i_data_eval["step_1"]["result"]
        len_step_1_result_data = len(step_1_result_data)

        try:
            shutil.move(url_, image_move_path + "/{}".format(fileName_))
        except Exception as Error:
            print(Error)

        txt_save_path = save_path + "/{}".format(fileName_.replace(".jpg", ".txt"))
        with open(txt_save_path, "w", encoding="utf-8") as fw:
            for j in range(len_step_1_result_data):
                x_ = step_1_result_data[j]["x"]
                y_ = step_1_result_data[j]["y"]
                w_ = step_1_result_data[j]["width"]
                h_ = step_1_result_data[j]["height"]

                x_min = x_
                x_max = x_ + w_
                y_min = y_
                y_max = y_ + h_

                # bb = convert_bbx_VOC_to_yolo((h, w), (x_min, x_max, y_min, y_max))
                bb = convertBboxVOC2YOLO((h, w), (x_min, y_min, x_max, y_max))
                txt_content = "0" + " " + " ".join([str(b) for b in bb]) + "\n"
                fw.write(txt_content)


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


def convert_labelbee_seg_json_to_yolo_txt(data_path):
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
        # cv2img = cv2.imread(img_abs_path)
        # if cv2img is None: continue

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
                        bb = convertBboxVOC2YOLO((h, w), (x_min, y_min, x_max, y_max))
                        p_res = convert_points((w, h), xy_)

                        txt_content = "0" + " " + " ".join([str(a) for a in bb]) + " " + " ".join([str(c) for c in p_res]) + "\n"
                        fw.write(txt_content)
                else:
                    x_min, x_max = min(x_), max(x_)
                    y_min, y_max = min(y_), max(y_)

                    # bb = convert_bbx_VOC_to_yolo((h, w), (x_min, x_max, y_min, y_max))
                    bb = convertBboxVOC2YOLO((h, w), (x_min, y_min, x_max, y_max))
                    txt_content = "0" + " " + " ".join([str(a) for a in bb]) + "\n"
                    fw.write(txt_content)

            print("Saved --> {}".format(txt_save_path))


def convert_labelme_json_to_VOC_xml(data_path):
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


def coco2yolo(root):
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

    id_map = {}  # coco数据集的id不连续！重新映射一下再输出！
    with open(os.path.join(root, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(traindata['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i

    '''
    保存train txt
    '''
    # print(id_map)
    # 这里需要根据自己的需要，更改写入图像相对路径的文件位置。
    list_file = open(os.path.join(root, 'train2017.txt'), 'w')
    for img in tqdm(traindata['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(train_ana_txt_save_path, ana_txt_name), 'w')
        for ann in traindata['annotations']:
            if ann['image_id'] == img_id:
                # box = convert((img_width, img_height), ann["bbox"])
                # box = convert_bbx_VOC_to_yolo((img_height, img_width), ann["bbox"])
                ann_np = np.array([ann["bbox"]])
                ann_np = ann_np[:, [0, 2, 1, 3]]
                ann_list = list(ann_np[0])
                box = convertBboxVOC2YOLO((img_height, img_width), ann_list)
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
        # 将图片的相对路径写入train2017或val2017的路径
        list_file.write('./images/train2017/%s.jpg\n' % (head))
    list_file.close()

    '''
    保存val txt
    '''
    # print(id_map)
    # 这里需要根据自己的需要，更改写入图像相对路径的文件位置。
    list_file = open(os.path.join(root, 'val2017.txt'), 'w')
    for img in tqdm(valdata['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(val_ana_txt_save_path, ana_txt_name), 'w')
        for ann in valdata['annotations']:
            if ann['image_id'] == img_id:
                # box = convert((img_width, img_height), ann["bbox"])
                # box = convert_bbx_VOC_to_yolo((img_height, img_width), ann["bbox"])
                ann_np = np.array([ann["bbox"]])
                ann_np = ann_np[:, [0, 2, 1, 3]]
                ann_list = list(ann_np[0])
                box = convertBboxVOC2YOLO((img_height, img_width), ann_list)
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
        # 将图片的相对路径写入train2017或val2017的路径
        list_file.write('./images/val2017/%s.jpg\n' % (head))
    list_file.close()


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


# def get_file_list(path):
#     text_list = []
#     with open(path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             line_list = line.strip().split('\t')
#             text_list.append(line_list)
#     return text_list


def get_label_list(path):
    label_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line_list = line.strip().split(' ')[-1]
            if line_list == 'n':
                line_list = 'normal'
            label_list.append(line_list)
    return label_list


def generate_dict(text_list, label_list):
    content_dict = {}
    for test_part, label in zip(text_list, label_list):
        file_name = test_part[1].split('.')[0]
        if file_name not in content_dict:
            content_dict[file_name] = [test_part[2].split(',') + [label]]
        else:
            content_dict[file_name].append(test_part[2].split(',') + [label])
    return content_dict


def write_point(doc, root, label1, label2, value1, value2):
    root = root.appendChild(doc.createElement('points'))
    root.appendChild(doc.createElement(label1)).appendChild(doc.createTextNode(value1))
    root.appendChild(doc.createElement(label2)).appendChild(doc.createTextNode(value2))


def write_one(doc, root, label, value):
    root.appendChild(doc.createElement(label)).appendChild(doc.createTextNode(value))


def txt2xml(args):
    input_file_record_path = args.input_file_record_path
    input_label_checker_path = args.input_label_checker_path
    input_xml_file_path = args.input_xml_file_path
    output_folder_path = args.output_folder_path

    text_list = get_file_list(input_file_record_path)
    label_list = get_label_list(input_label_checker_path)
    content_dict = generate_dict(text_list, label_list)

    for key in content_dict.keys():
        file_name = key
        doc = minidom.Document()
        annotationlist = doc.createElement('annotation')
        doc.appendChild(annotationlist)

        # folder = doc.createElement('folder')
        # annotationlist.appendChild(folder)
        # folder_name = doc.createTextNode(sys.argv[0].strip().split('/')[-2])
        # folder.appendChild(folder_name)

        annotationlist.appendChild(doc.createElement('filename')).appendChild(doc.createTextNode(sys.argv[0]))

        xml_size = minidom.parse(os.path.join(input_xml_file_path, '{}.xml'.format(file_name)))
        width_value = xml_size.getElementsByTagName('width')
        width_value = width_value[0].firstChild.data
        height_value = xml_size.getElementsByTagName('height')
        height_value = height_value[0].firstChild.data
        depth_value = xml_size.getElementsByTagName('depth')
        depth_value = depth_value[0].firstChild.data

        size = doc.createElement('size')
        annotationlist.appendChild(size)
        write_one(doc, size, 'width', width_value)
        write_one(doc, size, 'height', height_value)
        write_one(doc, size, 'depth', depth_value)

        for i in range(len(content_dict[key])):
            x_min = content_dict[key][i][0]
            y_min = content_dict[key][i][1]
            x_max = content_dict[key][i][2]
            y_max = content_dict[key][i][3]
            label = content_dict[key][i][4]

            objectlist = doc.createElement('object')
            annotationlist.appendChild(objectlist)
            write_one(doc, objectlist, 'name', label)
            write_one(doc, objectlist, 'difficult', '0')
            write_one(doc, objectlist, 'truncated', '0')

            bndbox = doc.createElement('bndbox')
            objectlist.appendChild(bndbox)
            write_one(doc, bndbox, 'xmin', x_min)
            write_one(doc, bndbox, 'ymin', y_min)
            write_one(doc, bndbox, 'xmax', x_max)
            write_one(doc, bndbox, 'ymax', y_max)

            segmentation = doc.createElement('segmentation')
            objectlist.appendChild(segmentation)
            write_point(doc, segmentation, 'x', 'y', x_min, y_min)
            write_point(doc, segmentation, 'x', 'y', x_max, y_min)
            write_point(doc, segmentation, 'x', 'y', x_max, y_max)
            write_point(doc, segmentation, 'x', 'y', x_min, y_max)

            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            with open(os.path.join(output_folder_path, '{}.xml').format(file_name), 'w', encoding='UTF-8') as fh:
                doc.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = int(x) * dw
    w = int(w) * dw
    y = int(y) * dh
    h = int(h) * dh
    return (x, y, w, h)


def convert_annotation_(base_path, image_id):
    import xml.etree.ElementTree as ET

    in_file = open('{}/xmls/{}.xml'.format(base_path, image_id).replace("\\", "/"), encoding='utf-8')
    out_file = open('{}/labels/{}.txt'.format(base_path, image_id).replace("\\", "/"), 'w')

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
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def xml2txt_and_create_train_val(base_path, classes=["c1", "c2"]):
    val_percent = 0.1  # test set proportion of the total data set, the default 0.1, if the test set and the training set have been demarcated, the corresponding code is modified
    data_path = '{}/images/'.format(base_path).replace("\\", "/")  # darknet relative path folder, see description github, and they need to modify, according to note here the absolute path can also be used

    if not os.path.exists("{}/labels".format(base_path).replace("\\", "/")):
        os.makedirs("{}/labels".format(base_path).replace("\\", "/"))

    image_ids = [f for f in os.listdir('{}/images').format(base_path).replace("\\", "/")]  # XML data storage folder
    train_file = open('{}/train.txt', 'w')
    val_file = open('{}/val.txt', 'w')
    for i, image_id in enumerate(image_ids):
        if image_id[-3:] == "jpg":  # Sometimes jpg and xml files are placed in the same folder, so to determine what suffixes
            if i < (len(image_ids) * val_percent):
                val_file.write(data_path + '%s\n' % (image_id[:-3] + 'jpg'))
            else:
                train_file.write(data_path + '%s\n' % (image_id[:-3] + 'jpg'))
        convert_annotation_(image_id[:-4])
    train_file.close()
    val_file.close()
# ========================================================================================================================================================================
# ========================================================================================================================================================================




# ========================================================================================================================================================================
# ========================================================================================================================================================================
# Image Process
class ImageProcess():
    """
    1. Transformation based on geometric: m1.
    2. Transformation based on color space: m2.
    3. Transformation based on others: m3.
    """

    # ==================================================
    def rotate_cv2(self, img, center=(50, 50), angle=30, scale=1):
        imgsz = img.shape[:2]
        M = cv2.getROtationMatrix2D(center, angle, scale)
        rotated_img = cv2.warpAffine(img, M, imgsz[::-1])
        return rotated_img

    def rotate_pil(self, img, angle=30):
        img_pil = Image.fromarray(np.uint8(img))
        rotate_angle = np.random.randint(-angle, angle + 1)
        img_x = np.asarray(img_pil.rotate(rotate_angle, expand=True))
        return img_x

    def flip(self, img, d=0):
        """
        0：垂直翻转（沿x轴翻转）
        1：水平翻转（沿y轴翻转）
        -1：同时在水平和垂直方向翻转
        Parameters
        ----------
        img
        d

        Returns
        -------

        """
        assert d in [-1, 0, 1], "d(flip direction) should be one of [-1, 0, 1]"
        flipped_img = cv2.flip(img, d)
        return flipped_img

    def apply_random_crop(self, img, crop_size=(128, 128)):
        # crop_size: [H, W]
        imgsz = img.shape[:2]
        assert crop_size[0] >= 0 and crop_size[0] <= imgsz[0], "crop_size[0] < 0 or crop_size[0] > imgsz[0]"
        assert crop_size[1] >= 0 and crop_size[1] <= imgsz[1], "crop_size[1] < 0 or crop_size[1] > imgsz[1]"

        x = np.random.randint(0, imgsz[1])
        y = np.random.randint(0, imgsz[0])

        try:
            cropped_img = img[y:(y + crop_size[0]), x:(x + crop_size[1])]
            return cropped_img
        except Exception as Error:
            print(Error)
            return None

    def apply_random_scale(self, img, scale_factor_x=0.5, scale_factor_y=0.5):
        scaled_img = cv2.resize(img, None, fx=scale_factor_x, fy=scale_factor_y)
        return scaled_img

    def squeeze(self, img, degree=11):
        height, width, channels = img.shape
        center_x = width / 2
        center_y = height / 2
        new_data = img.copy()
        for i in range(width):
            for j in range(height):
                tx = i - center_x
                ty = j - center_y
                theta = math.atan2(ty, tx)
                # 半径
                radius = math.sqrt(tx ** 2 + ty ** 2)
                radius = math.sqrt(radius) * degree
                new_x = int(center_x + radius * math.cos(theta))
                new_y = int(center_y + radius * math.sin(theta))
                if new_x < 0:
                    new_x = 0
                if new_x >= width:
                    new_x = width - 1
                if new_y < 0:
                    new_y = 0
                if new_y >= height:
                    new_y = height - 1

                for channel in range(channels):
                    new_data[j][i][channel] = img[new_y][new_x][channel]
        return new_data

    def apply_haha_mirror(self, img, degree=4):
        height, width, n = img.shape
        center_x = width / 2
        center_y = height / 2
        randius = 40 * degree  # 直径
        real_randius = int(randius / 2)  # 半径
        new_data = img.copy()
        for i in range(width):
            for j in range(height):
                tx = i - center_x
                ty = j - center_y
                distance = tx ** 2 + tx ** 2
                # 为了保证选择的像素是图片上的像素
                if distance < randius ** 2:
                    new_x = tx / 2
                    new_y = ty / 2
                    # 图片的每个像素的坐标按照原来distance 之后的distance（real_randius**2）占比放大即可
                    new_x = int(new_x * math.sqrt(distance) / real_randius + center_x)
                    new_y = int(new_y * math.sqrt(distance) / real_randius + center_y)
                    # 当不超过new_data 的边界时候就可赋值
                    if new_x < width and new_y < height:
                        new_data[j][i][0] = img[new_y][new_x][0]
                        new_data[j][i][1] = img[new_y][new_x][1]
                        new_data[j][i][2] = img[new_y][new_x][2]
        return new_data

    def warp_img(self, img, degree=4):
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

    # ==================================================
    def equalize_hist(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # equalized_img = cv2.equalizeHist(img)
        # equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)

        b, g, r = cv2.split(img)
        equalized_b = cv2.equalizeHist(b)
        equalized_g = cv2.equalizeHist(g)
        equalized_r = cv2.equalizeHist(r)
        equalized = cv2.merge([equalized_b, equalized_g, equalized_r])

        return equalized

    def add_gaussian_noise(self, img, mu=0, sigma=0.1):
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
        # mu, sigma = 0, 0.5 ** 0.5
        gaussian = np.random.normal(mu, sigma, img.shape).astype('uint8')
        noisy_img = cv2.add(img, gaussian)
        return noisy_img

    def add_salt_pepper_noise(self, img, salt_prob=0.01, pepper_prob=0.01):
        """
        if noise_typ == "s&p":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            out[coords] = 0
            return out
        Parameters
        ----------
        self
        img
        salt_prob
        pepper_prob

        Returns
        -------

        """
        noisy_image = np.copy(img)
        total_pixels = img.shape[0] * img.shape[1]  # 计算图像的总像素数

        num_salt = int(total_pixels * salt_prob)  # 通过将总像素数与指定的椒盐噪声比例相乘，得到要添加的椒盐噪声的数量。
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
        noisy_image[salt_coords[0], salt_coords[1]] = 255

        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_image

    def add_poisson_noise(self, img):
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
        return noisy

    def apply_color_distortion(self, img, r=30):
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + r) % 180  # 在Hue通道上增加30
        result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return result_image

    def apply_random_mask(self, img, mask_size=(128, 128)):
        # 在图像上随机生成一个矩形遮挡，遮挡的位置和大小都是随机生成的。遮挡的颜色也是随机选择的
        # 生成随机遮挡位置和大小
        imgsz = img.shape[:2]
        assert mask_size[0] >= 0 and mask_size[0] <= imgsz[0], "mask_size[0] < 0 or mask_size[0] > imgsz[0]"
        assert mask_size[1] >= 0 and mask_size[1] <= imgsz[1], "mask_size[1] < 0 or mask_size[1] > imgsz[1]"

        mask_x = np.random.randint(0, imgsz[1])
        mask_y = np.random.randint(0, imgsz[0])

        # 生成随机颜色的遮挡
        mask_color = np.random.randint(0, 256, (1, 1, 3))
        img[mask_y:mask_y + mask_size[0], mask_x:mask_x + mask_size[1]] = mask_color
        return img

    def change_Contrast_and_Brightness(self, img, alpha=1.1, beta=30):
        # """使用公式f(x)=α.g(x)+β"""
        # #α调节对比度，β调节亮度
        blank = np.zeros(img.shape, img.dtype)  # 创建图片类型的零矩阵
        dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)  # 图像混合加权
        return dst

    def apply_CLAHE(self, img, clipLimit=2.0, tileGridSize=(8, 8)):
        """
        直方图适应均衡化
        该函数包含以下参数：
        clipLimit: 用于控制直方图均衡化的局部对比度，值越高，越容易出现失真和噪声。建议值为2-4，若使用默认值0则表示自动计算。
        tileGridSize: 表示每个块的大小，推荐16x16。
        tileGridSize.width: 块的宽度。
        tileGridSize.height: 块的高度。
        函数返回一个CLAHE对象，可以通过该对象调用apply函数来实现直方图均衡化。
        """
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        # res = clahe.apply(img)
        # res = cv2.merge([res, res, res])

        b, g, r = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        clahe_b = clahe.apply(b)
        clahe_g = clahe.apply(g)
        clahe_r = clahe.apply(r)
        res = cv2.merge([clahe_b, clahe_g, clahe_r])

        return res

    def change_gray_value(self, img, min_gray=0, max_gray=255):
        """
        灰度变换, 通过将像素值映射到新的范围来增强图像的灰度
        """
        gray_img_enhanced = cv2.convertScaleAbs(img, alpha=(max_gray - min_gray) / 255, beta=min_gray)
        return gray_img_enhanced

    def apply_homomorphic_filter(self, img):
        gray = cv2.bilateralFilter(img, 15, 75, 75)
        # 对数变换和傅里叶变换
        H, W = gray.shape
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

    def apply_contrast_stretching(self, img, alpha=0, beta=1):
        """
        对比拉伸
        """
        norm_img1 = cv2.normalize(img, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img = (255 * norm_img1).astype(np.uint8)
        return norm_img

    def apply_log_transformation(self, img):
        """
        对数变换
        """
        c = 255 / np.log(1 + np.max(img))
        log_image = c * (np.log(img + 1))
        # Specify the data type so that
        # float value will be converted to int
        log_image = np.array(log_image, dtype=np.uint8)
        return log_image

    def change_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def change_brightness_opencv_official(self, img, alpha=1.0, beta=0):
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

    def apply_gamma_transformation(self, img, gamma=0.8):
        # Apply Gamma=0.4 on the normalised image and then multiply by scaling constant (For 8 bit, c=255)
        gamma_res = np.array(255 * (img / 255) ** gamma, dtype='uint8')
        return gamma_res

    def gamma_correction(self, img, gamma=0.4):
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(img, lookUpTable)

        return res

    def gamma_correction_auto(self, img, method=2):
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

    def makeSunLightEffect(self, img, r=(50, 200), light_strength=150):
        imgsz = img.shape[:2]
        center = (np.random.randint(0, imgsz[1]), np.random.randint(0, imgsz[0]))
        effectR = np.random.randint(r[0], r[1])
        lightStrength = np.random.randint(light_strength // 4, light_strength)

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

    # ==============================================================================================================================
    # ==============================================================================================================================

    def get_color(self, specific_color_flag=True):
        global color
        color1 = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

        if specific_color_flag:
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

    def makeBorder_base(self, im, new_shape=(64, 256), r1=0.75, specific_color_flag=True):
        """
        :param im:
        :param new_shape: (H, W)
        :param r1:
        :param r2:
        :param sliding_window:
        :return:
        """
        color = self.get_color(specific_color_flag=specific_color_flag)

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

    def sliding_window_crop_v2(self, img, cropsz=(64, 256), gap=(0, 128), makeBorder=True, r1=0, specific_color_flag=True):
        cropped_imgs = []
        imgsz = img.shape[:2]

        if gap[0] == 0 and gap[1] > 0:
            cropsz = (imgsz[0], cropsz[1])
            for i in range(0, imgsz[1], gap[1]):
                if i + cropsz[1] > imgsz[1]:
                    cp_img = img[0:imgsz[0], i:imgsz[1]]
                    if makeBorder:
                        cp_img = self.makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                        cp_img = self.makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                                cp_img = self.makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                                cp_img = self.makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
                            cropped_imgs.append(cp_img)
                            break
                        else:
                            cp_img = img[j:j + cropsz[0], i:i + cropsz[1]]
                            cropped_imgs.append(cp_img)

        return cropped_imgs

    def makeBorder_v6(self, im, new_shape=(64, 256), r1=0.75, r2=0.25, sliding_window=False, specific_color_flag=True, gap_r=(0, 7 / 8), last_img_makeBorder=True):
        """
        :param im:
        :param new_shape: (H, W)
        :param r1:
        :param r2:
        :param sliding_window:
        :return:
        """
        color = self.get_color(specific_color_flag=specific_color_flag)

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
                    final_imgs = self.sliding_window_crop_v2(im, cropsz=new_shape, gap=(int(gap_r[0] * 0), int(gap_r[1] * new_shape[1])), makeBorder=last_img_makeBorder, r1=r1)
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
                    final_imgs = self.sliding_window_crop_v2(im, cropsz=new_shape, gap=(int(gap_r[0] * 0), int(gap_r[1] * new_shape[1])), makeBorder=last_img_makeBorder, r1=r1)
                    return final_imgs
                else:
                    im = cv2.resize(im, new_shape[::-1])

        return im

    # ==============================================================================================================================
    # ==============================================================================================================================

    # ======================================================================================================
    # ======================================================================================================
    def padding(self, img):
        res = math.sqrt(img.shape[0] * img.shape[0] + img.shape[1] * img.shape[1])
        pad_x = int(res - img.shape[1] * 0.5 + 1)
        pad_y = int(res - img.shape[0] * 0.5 + 1)
        img_pad = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_CONSTANT, value=0)
        return img_pad, (pad_x, pad_y)

    def resize(self, img, type="EASY", angle=30):
        img, crop_rect = self.padding(img)
        angle = random.uniform(-angle, angle + 1)
        rows, cols, _ = img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, affine_mat, (cols, rows))

        factor = random.uniform(0, 1.0)
        if type == "EASY":
            scale = factor * 0.25 + 0.8
        else:
            scale = factor * 0.1 + 0.2
        rows, cols, _ = img.shape
        dst = cv2.resize(dst, (int(cols * scale), int(rows * scale)))
        dst = cv2.resize(dst, (cols, rows))

        rows, cols, _ = dst.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360.0 - angle, 1)
        out_img = cv2.warpAffine(dst, affine_mat, (cols, rows))
        out_img = out_img[crop_rect[1]: out_img.shape[0] - crop_rect[1], crop_rect[0]: out_img.shape[1] - crop_rect[0], :]
        return out_img

    def blur(self, img, type="EASY", angle=30):
        img, crop_rect = self.padding(img)
        angle = random.uniform(-angle, angle + 1)
        rows, cols, _ = img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, affine_mat, (cols, rows))
        if type == "EASY":
            random_value = random.randint(0, 3)
            size = int(random_value / 2) * 2 + 1
        else:
            random_value = random.randint(5, 7)
            size = int(random_value / 2) * 2 + 3
        blur_img = cv2.blur(dst, (size, size))
        rows, cols, _ = blur_img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360.0 - angle, 1)
        out_img = cv2.warpAffine(blur_img, affine_mat, (cols, rows))
        out_img = out_img[crop_rect[1]: out_img.shape[0] - crop_rect[1], crop_rect[0]: out_img.shape[1] - crop_rect[0], :]
        return out_img

    def motion_blur(self, img, type="EASY", angle=30):
        img, crop_rect = self.padding(img)
        angle = random.uniform(-angle, angle + 1)
        rows, cols, _ = img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, affine_mat, (cols, rows))

        if type == "EASY":
            size = int(random.uniform(0.0, 3.0) + 2)
        else:
            size = int(random.uniform(5.0, 7.0) + 5)
        kernel = np.zeros((size, size), np.float32)
        h = (size - 1) // 2
        for i in range(size):
            kernel[h][i] = 1.0 / float(size)

        blur_img = cv2.filter2D(dst, -1, kernel)
        rows, cols, _ = blur_img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360.0 - angle, 1)
        out_img = cv2.warpAffine(blur_img, affine_mat, (cols, rows))

        out_img = out_img[crop_rect[1]: out_img.shape[0] - crop_rect[1], crop_rect[0]: out_img.shape[1] - crop_rect[0], :]
        return out_img

    def median_blur(self, img, type="EASY", angle=30):
        img, crop_rect = self.padding(img)
        angle = random.uniform(-angle, angle + 1)
        rows, cols, _ = img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, affine_mat, (cols, rows))
        if type == "EASY":
            random_value = random.randint(0, 3)
            size = int(random_value / 2) * 2 + 1
        else:
            random_value = random.randint(3, 7)
            size = int(random_value / 2) * 2 + 3
        blur_img = cv2.medianBlur(dst, size)
        rows, cols, _ = blur_img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360.0 - angle, 1)
        out_img = cv2.warpAffine(blur_img, affine_mat, (cols, rows))
        out_img = out_img[crop_rect[1]: out_img.shape[0] - crop_rect[1], crop_rect[0]: out_img.shape[1] - crop_rect[0], :]
        return out_img

    def gaussian_blur(self, img, type="EASY", angle=30):
        img, crop_rect = self.padding(img)
        angle = random.uniform(-angle, angle + 1)
        rows, cols, _ = img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, affine_mat, (cols, rows))
        if type == "EASY":
            random_value = random.randint(0, 2)
            size = int(random_value / 2) * 2 + 3
        else:
            random_value = random.randint(5, 7)
            size = int(random_value / 2) * 2 + 7
        blur_img = cv2.GaussianBlur(dst, (size, size), 0)
        rows, cols, _ = blur_img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360.0 - angle, 1)
        out_img = cv2.warpAffine(blur_img, affine_mat, (cols, rows))
        out_img = out_img[crop_rect[1]:out_img.shape[0] - crop_rect[1], crop_rect[0]:out_img.shape[1] - crop_rect[0], :]
        return out_img

    def get_trans_mat(self, center, degrees=0, translate=(0, 0), scale=1, shear=(0, 0), perspective=(0, 0)):
        C = np.eye(3)
        C[0, 2] = center[0]  # x translation (pixels)
        C[1, 2] = center[1]  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = perspective[0]  # x perspective (about y)
        P[2, 1] = perspective[1]  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = degrees
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = scale
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = shear[0]  # x shear (deg)
        S[1, 0] = shear[1]  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = translate[0]  # x translation (pixels)
        T[1, 2] = translate[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        return M

    def TransAffine(self, img, degrees=10, translate=0.1, scale=0.1, shear=0.1, perspective=0.1, border=(4, 4), prob=0.5):
        img = img  # results["img"]
        height = img.shape[0]
        width = img.shape[1]

        center_src = (-img.shape[1] / 2, -img.shape[0] / 2)
        perspective_src = (random.uniform(-perspective, perspective), random.uniform(-perspective, perspective))
        degrees_src = random.uniform(-degrees, degrees)
        scale_src = random.uniform(1 - 0.25, 1 + scale)
        shear_src = (math.tan(random.uniform(-shear, shear) * math.pi / 180), math.tan(random.uniform(-shear, shear) * math.pi / 180))
        translate_src = [random.uniform(0.5 - translate, 0.5 + translate) * width, random.uniform(0.5 - translate, 0.5 + translate) * height]

        M_src = self.get_trans_mat(center_src, degrees_src, translate_src, scale_src, shear_src, perspective_src)
        four_pt = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]])
        res_pt = M_src @ four_pt.T
        res_pt = res_pt.astype(np.int_).T
        res_pt = res_pt[:, :2]
        min_x = np.min(res_pt[:, 0])
        max_x = np.max(res_pt[:, 0])
        min_y = np.min(res_pt[:, 1])
        max_y = np.max(res_pt[:, 1])
        if (min_x < 0):
            translate_src[0] -= min_x
        if (min_y < 0):
            translate_src[1] -= min_y

        if (max_x - min_x > width):
            new_width = (max_x - min_x)
        else:
            new_width = width
        if (max_y - min_y > height):
            new_height = (max_y - min_y)
        else:
            new_height = height

        M = self.get_trans_mat((-width / 2, -height / 2), degrees_src, translate_src, scale_src, shear_src, perspective_src)

        border_color = (random.randint(220, 250), random.randint(220, 250), random.randint(220, 250))
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)
        return img

    def change_HSV(self, img, hgain=0.5, sgain=0.5, vgain=0.5):
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

    def translate(self, img, translate_xy=(20, 30), border_color=(114, 114, 114), dstsz=None):
        # M = np.array([[1, 0, np.random.randint(-30, 30)], [0, 1, np.random.randint(-30, 30)]], dtype=np.float32)
        M = np.array([[1, 0, translate_xy[0]], [0, 1, translate_xy[1]]], dtype=np.float32)
        img = cv2.warpAffine(img, M[:2], dsize=dstsz, borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)
        return img

    def perspective_transform(self, img, p1, p2, dstsz):
        """

        Parameters
        ----------
        img
        p1
        p2
        dstsz: [H, W]

        Returns
        -------

        """
        # p1 = np.array([[887, 530], [1069, 540], [886, 607], [1066, 617]], dtype=np.float32)
        # p2 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        # M = cv2.getPerspectiveTransform(p1, p2)
        # warped = cv2.warpPerspective(cv2img, M, (w, h))

        M = cv2.getPerspectiveTransform(p1, p2)
        warped = cv2.warpPerspective(img, M, dstsz[::-1])
        return warped


# ==============================================================================================================================
# ==============================================================================================================================

def get_color(specific_color_flag=True):
    global color
    color1 = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

    if specific_color_flag:
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


def makeBorder_base(im, new_shape=(64, 256), r1=0.75, specific_color_flag=True):
    """
    :param im:
    :param new_shape: (H, W)
    :param r1:
    :param r2:
    :param sliding_window:
    :return:
    """
    color = get_color(specific_color_flag=specific_color_flag)

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


def sliding_window_crop_v2(img, cropsz=(64, 256), gap=(0, 128), makeBorder=True, r1=0, specific_color_flag=True):
    cropped_imgs = []
    imgsz = img.shape[:2]

    if gap[0] == 0 and gap[1] > 0:
        cropsz = (imgsz[0], cropsz[1])
        for i in range(0, imgsz[1], gap[1]):
            if i + cropsz[1] > imgsz[1]:
                cp_img = img[0:imgsz[0], i:imgsz[1]]
                if makeBorder:
                    cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                    cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                            cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                            cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
                        cropped_imgs.append(cp_img)
                        break
                    else:
                        cp_img = img[j:j + cropsz[0], i:i + cropsz[1]]
                        cropped_imgs.append(cp_img)

    return cropped_imgs


def makeBorder_v6(im, new_shape=(64, 256), r1=0.75, r2=0.25, sliding_window=False, specific_color_flag=True, gap_r=(0, 7 / 8), last_img_makeBorder=True):
    """
    :param im:
    :param new_shape: (H, W)
    :param r1:
    :param r2:
    :param sliding_window:
    :return:
    """
    color = get_color(specific_color_flag=specific_color_flag)

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
                final_imgs = sliding_window_crop_v2(im, cropsz=new_shape, gap=(int(gap_r[0] * 0), int(gap_r[1] * new_shape[1])), makeBorder=last_img_makeBorder, r1=r1)
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
                final_imgs = sliding_window_crop_v2(im, cropsz=new_shape, gap=(int(gap_r[0] * 0), int(gap_r[1] * new_shape[1])), makeBorder=last_img_makeBorder, r1=r1)
                return final_imgs
            else:
                im = cv2.resize(im, new_shape[::-1])

    return im


def do_makeBorderv6(data_path):
    save_path = os.path.abspath(os.path.join(data_path, "..")) + "/makeBorder_Res"
    os.makedirs(save_path, exist_ok=True)

    file_list = sorted(os.listdir(data_path))
    for f in tqdm(file_list):
        try:
            f_abs_path = data_path + "/{}".format(f)
            f_name = os.path.splitext(f)[0]
            img = cv2.imread(f_abs_path)
            #img_cp = img.copy()
            makeBorderRes = makeBorder_v6(img, new_shape=(64, 256), r1=0, r2=0.25, sliding_window=False, specific_color_flag=True, gap_r=(0, 11 / 12), last_img_makeBorder=True)
            if isinstance(makeBorderRes, list):
                pass
            else:
                f_dst_path = save_path + "/{}".format(f)
                cv2.imwrite(f_dst_path, makeBorderRes)
        except Exception as Error:
            errinfo = "ImageProcess: LINE:{}, Error:{}".format(Error.__traceback__.tb_lineno, Error)
            print(errinfo)



# ==============================================================================================================================
# ==============================================================================================================================







class ResizeImages(object):
    def resize_images(self, img_path, size=(128, 128), n=8):
        img_list = os.listdir(img_path)
        save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_resize".format(img_path.split("/")[-1])
        os.makedirs(save_path, exist_ok=True)

        for img in img_list:
            img_abs_path = img_path + "/" + img
            img_name = os.path.splitext(img)[0]
            src_img = cv2.imread(img_abs_path)
            # resz_img = cv2.resize(src_img, size, interpolation=cv2.INTER_LINEAR)

            # gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            # ret, thr = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

            imgsz = src_img.shape[:2]
            # if imgsz[0] > 800 or imgsz[1] > 800:
            #     resz_img = cv2.resize(src_img, (imgsz[1] // n, imgsz[0] // n))
            #     # resz_img = cv2.resize(src_img, size)
            #     # resz_img = cv2.resize(thr, size)
            #     # resz_img = cv2.resize(src_img, size)
            #     cv2.imwrite("{}/{}.jpg".format(save_path, img_name), resz_img)
            # elif imgsz[0] > 400 and imgsz[1] > 400:
            #     resz_img = cv2.resize(src_img, (imgsz[1] * 2 // n, imgsz[0] * 2 // n))
            #     # resz_img = cv2.resize(src_img, size)
            #     # resz_img = cv2.resize(thr, size)
            #     # resz_img = cv2.resize(src_img, size)
            #     cv2.imwrite("{}/{}.jpg".format(save_path, img_name), resz_img)
            # else:
            #     resz_img = cv2.resize(src_img, size)
            #     cv2.imwrite("{}/{}.jpg".format(save_path, img_name), resz_img)

            resz_img = cv2.resize(src_img, size)
            cv2.imwrite("{}/{}.jpg".format(save_path, img_name), resz_img)



    def resize_cropped_images(self, img_path, small_or_big="small"):
        img_list = os.listdir(img_path)
        save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_resize".format(img_path.split("/")[-1])
        os.makedirs(save_path, exist_ok=True)

        for img in img_list:
            img_abs_path = img_path + "/" + img
            img_name = os.path.splitext(img)[0]
            src_img = cv2.imread(img_abs_path)
            oh, ow = src_img.shape[:2]
            if small_or_big == "big":
                if oh > 1600 or ow > 1600:
                    reszToOrig = cv2.resize(src_img, (ow // 8, oh // 8))
                    cv2.imwrite("{}/{}.jpg".format(save_path, img_name), reszToOrig)
                elif oh > 1024 or ow > 1024:
                    reszToOrig = cv2.resize(src_img, (ow // 6, oh // 6))
                    cv2.imwrite("{}/{}.jpg".format(save_path, img_name), reszToOrig)
                elif oh > 512 and ow > 512:
                    reszToOrig = cv2.resize(src_img, (ow // 4, oh // 4))
                    cv2.imwrite("{}/{}.jpg".format(save_path, img_name), reszToOrig)
                elif oh > 256 and ow > 256:
                    reszToOrig = cv2.resize(src_img, (ow // 2, oh // 2))
                    cv2.imwrite("{}/{}.jpg".format(save_path, img_name), reszToOrig)
                elif oh > 196 and ow > 196:
                    reszToOrig = cv2.resize(src_img, (ow // 4, oh // 4))
                    cv2.imwrite("{}/{}.jpg".format(save_path, img_name), reszToOrig)
                elif oh > 128 and ow > 128:
                    reszToOrig = cv2.resize(src_img, (ow // 2, oh // 2))
                    cv2.imwrite("{}/{}.jpg".format(save_path, img_name), reszToOrig)
                else:
                    print("Error!")
            else:
                if oh > 512 or ow > 512:
                    reszToOrig = cv2.resize(src_img, (ow // 8, oh // 8))
                    cv2.imwrite("{}/{}.jpg".format(save_path, img_name), reszToOrig)
                elif oh > 256 or ow > 256:
                    reszToOrig = cv2.resize(src_img, (ow // 6, oh // 6))
                    cv2.imwrite("{}/{}.jpg".format(save_path, img_name), reszToOrig)
                elif oh > 196 or ow > 196:
                    reszToOrig = cv2.resize(src_img, (ow // 4, oh // 4))
                    cv2.imwrite("{}/{}.jpg".format(save_path, img_name), reszToOrig)
                elif oh > 128 or ow > 128:
                    reszToOrig = cv2.resize(src_img, (ow // 2, oh // 2))
                    cv2.imwrite("{}/{}.jpg".format(save_path, img_name), reszToOrig)
                else:
                    print("Error!")

    def resize_seg_mask_images(self, img_path, size=(256, 256)):
        img_list = os.listdir(img_path)
        save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_resize".format(img_path.split("/")[-1])
        os.makedirs(save_path, exist_ok=True)

        for img in img_list:
            try:
                img_abs_path = img_path + "/" + img
                img_name = os.path.splitext(img)[0]
                src_img = cv2.imread(img_abs_path)
                reszToOrig = scale_uint16(src_img, size)
                cv2.imwrite("{}/{}.png".format(save_path, img_name), reszToOrig.astype('uint8'))  # scale_uint16 需要保存为 png 格式
            except Exception as Error:
                print(Error)

    def resize_base(self, list_i, img_path, save_path, size=(256, 256)):
        for i in range(len(list_i)):
            img_abs_path = img_path + "/" + list_i[i]
            img_name = os.path.splitext(list_i[i])[0]
            src_img = cv2.imread(img_abs_path)
            resz_img = cv2.resize(src_img, size)
            cv2.imwrite("{}/{}.jpg".format(save_path, img_name), resz_img)

    def resize_images_multithread(self, img_path, size=(256, 256), split_n=8):
        img_list = os.listdir(img_path)
        save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_resize".format(img_path.split("/")[-1])
        os.makedirs(save_path, exist_ok=True)

        len_ = len(img_list)

        img_lists = []
        for j in range(split_n):
            img_lists.append(img_list[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

        t_list = []
        for i in range(split_n):
            list_i = img_lists[i]
            t = threading.Thread(target=self.resize_base, args=(list_i, img_path, save_path, size,))
            t_list.append(t)

        for t in t_list:
            t.start()
        for t in t_list:
            t.join()


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


def cal_mean_std(imageDir, size=(64, 64)):
    img_h, img_w = size[0], size[1]  # 根据自己数据集适当调整，影响不大
    means, stdevs, vars_ = [], [], []
    img_list = []

    if os.path.exists(imageDir + "/Thumbs.db"):
        os.remove(imageDir + "/Thumbs.db")

    imgs_path_list = os.listdir(imageDir)

    len_ = len(imgs_path_list)
    i = 0
    for item in tqdm(imgs_path_list):
        img = cv2.imread(os.path.join(imageDir, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
        vars_.append(np.var(pixels))

    return means, stdevs, vars_


def cal_mean_std_dirs(imageDir, size=(64, 64)):
    img_h, img_w = size[0], size[1]  # 根据自己数据集适当调整，影响不大
    means, stdevs, vars_ = [], [], []
    img_list = []

    if os.path.exists(imageDir + "/Thumbs.db"):
        os.remove(imageDir + "/Thumbs.db")

    i = 0
    dir_list = os.listdir(imageDir)
    for d in dir_list:
        imgs_path_list = os.listdir(imageDir + "/{}".format(d))
        for item in tqdm(imgs_path_list):
            img = cv2.imread(os.path.join(imageDir + "/{}".format(d), item))
            img = cv2.resize(img, (img_w, img_h))
            img = img[:, :, :, np.newaxis]
            img_list.append(img)
            i += 1

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
        vars_.append(np.var(pixels))

    return means, stdevs, vars_


def convert_to_jpg_format(data_path):
    img_list = sorted(os.listdir(data_path))

    for img in img_list:
        img_name = os.path.splitext(img)[0]
        img_abs_path = data_path + "/{}".format(img)

        if img.endswith(".jpeg") or img.endswith(".png") or img.endswith(".bmp") or img.endswith(".JPG") or img.endswith(".JPEG") or img.endswith(".PNG") or img.endswith(".BMP"):
            cv2img = cv2.imread(img_abs_path)
            cv2.imwrite("{}/{}.jpg".format(data_path, img_name), cv2img)
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
                cv2img = cv2.imread(img_abs_path)
                cv2.imwrite("{}/{}.png".format(data_path, img_name), cv2img)
                os.remove(img_abs_path)
                print("write --> {}.png  |  remove --> {}".format(img_name, img))

            elif img.endswith(".png"):
                continue
            else:
                print(img)
        except Exception as Error:
            os.remove(img_abs_path)
            print("os.remove: {}".format(img_abs_path))


def HORIZON_quant_model_cal_mean_std(torchvision_mean, torchvision_std, print_flag=True):
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
    cv2img = cv2.imread(img_path)
    g_img = cv2.split(cv2img)[1]
    hsvimg = cv2.cvtColor(cv2img, cv2.COLOR_BGR2HSV)

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
        cv2img = cv2.imread(img_abs_path)
        cv2img_cp = cv2img.copy()

        # black_area = np.where((cv2img[:, :, 0] < 5) & (cv2img[:, :, 1] < 5) & (cv2img[:, :, 2] < 5))
        black_area = np.where((cv2img[:, :, 0] < 10) & (cv2img[:, :, 1] < 10) & (cv2img[:, :, 2] < 10))
        # black_area = np.where((cv2img[:, :, 0] < 20) & (cv2img[:, :, 1] < 20) & (cv2img[:, :, 2] < 20))
        # black_area = np.where((cv2img[:, :, 0] < 30) & (cv2img[:, :, 1] < 30) & (cv2img[:, :, 2] < 30))

        # bg_cv2img = bg_cv2img.copy()
        for x_b, y_b in zip(black_area[1], black_area[0]):
            try:
                cv2img_cp[y_b, x_b] = (255, 0, 255)
            except Exception as Error:
                print(Error)

        cv2.imwrite("{}/{}".format(save_path, img), cv2img_cp)


def perspective_transform_test(img_path):
    cv2img = cv2.imread(img_path)
    h, w = cv2img.shape[:2]

    p1 = np.array([[8, 26], [137, 44], [16, 162], [147, 209]], dtype=np.float32)
    p2 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(p1, p2)
    warped = cv2.warpPerspective(cv2img, M, (w, h))
    cv2.imwrite("{}".format(img_path.replace(".jpg", "_warpPerspective.jpg")), warped)


def click_event(event, x, y, flags, param):
    # xy = []
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        # xy.append((x, y))
        cv2.circle(cv2img, (x, y), 1, (255, 0, 255), -1)
        cv2.putText(cv2img, "({}, {})".format(x, y), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv2.imshow("cv2img", cv2img)
    # print(xy)


def do_perspective_transform(img_path):
    cv2img = cv2.imread(img_path)
    h, w = cv2img.shape[:2]

    # TODO return mouseClick xy to p1

    p1 = np.array([[8, 26], [137, 44], [16, 162], [147, 209]], dtype=np.float32)
    p2 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(p1, p2)
    warped = cv2.warpPerspective(cv2img, M, (w, h))
    cv2.imwrite("{}".format(img_path.replace(".jpg", "_warpPerspective.jpg")), warped)


def mv_or_rm_black_images(img_path, flag="mv", pixel_sum=100000):
    img_list = sorted(os.listdir(img_path))

    if flag == "mv":
        dir_name = os.path.basename(img_path)
        save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_moved_black_images_{}".format(dir_name, pixel_sum)
        os.makedirs(save_path, exist_ok=True)

    for img in img_list:
        if os.path.isdir(img): continue
        img_abs_path = img_path + "/{}".format(img)
        try:

            cv2img = cv2.imread(img_abs_path)
            cv2img = cv2.resize(cv2img, (128, 128))
            h, w = cv2img.shape[:2]
            sum_ = np.sum(cv2img[:, :, :])
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
        # cv2img = cv2.imread(img_abs_path)
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
        valid = buf.endswith(b'\xff\xd9') or buf.endswith(b'\xae\x82') or buf.endswith(b'\x00\x3B') or buf.endswith(b'\x60\x82')  # 检测jpg图片完整性， 检测png图片完整性
        buf.endswith(b'\x00\x00')
    # return valid or (imghdr.what(nm) =="webp")
    return valid


def remove_corrupt_images_pil_v2_main_thread(img_path, move_or_delete, img_list_i):
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
        t = threading.Thread(target=remove_corrupt_images_pil_v2_main_thread, args=(img_path, move_or_delete, img_list_i,))
        t_list.append(t)

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


def remove_corrupt_images_cv2_v2_main_thread(img_path, move_or_delete, img_list_i):
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
        t = threading.Thread(target=remove_corrupt_images_cv2_v2_main_thread, args=(img_path, move_or_delete, img_list_i,))
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
            # cv2img = cv2.imdecode(np.fromfile(img_abs_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2img = cv2.imread(img_abs_path)

            if cv2img is None:
                os.remove(img_abs_path)
                print("[img is None]: Removed --> {}".format(img_abs_path))
                continue

            cv2img_ = np.asarray(cv2img)

        except Exception as Error:
            print(Error)
            os.remove(img_abs_path)
            print("Removed --> {}".format(img_abs_path))


def ssim_move_or_remove_same_images(img_path, imgsz=(64, 64), move_or_remove="move"):
    from skimage.metrics import structural_similarity

    img_list = sorted(os.listdir(img_path))
    dir_name = os.path.basename(img_path)

    if move_or_remove == "move":
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
                    if move_or_remove == "remove" or move_or_remove == "delete":
                        os.remove(img_path_j)
                        print("{}, {} 两张图片相似度很高, ssim: {}  |  Removed: {}".format(img_list[i], img_list[j], ssim, img_path_j))
                    elif move_or_remove == "move":
                        shutil.move(img_path_j, move_path + "/{}".format(img_list[j]))
                        print("{}, {} 两张图片相似度很高, ssim: {}   |  {} --> {}/{}.".format(img_list[i], img_list[j], ssim, img_path_j, move_path, img_list[j]))
                    else:
                        print("'move_or_remove' should be one of [remove, delete, move]!")

            print(" ----------- {} ----------- ".format(i))

        except Exception as Error:
            print(Error, Error.__traceback__.tb_lineno)


def ssim_move_or_remove_base(img_path, imgsz, img_list, ii, img_i, img_list_i, move_or_remove, move_path):
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
            if move_or_remove == "remove" or move_or_remove == "delete":
                os.remove(img_path_j)
                print("{}, {} 两张图片相似度很高, ssim: {}  |  Removed: {}".format(img_list[ii], img_list_i[j], ssim, img_path_j))
            elif move_or_remove == "move":
                shutil.move(img_path_j, move_path + "/{}".format(img_list_i[j]))
                print("{}, {} 两张图片相似度很高, ssim: {}   |  {} --> {}/{}.".format(img_list[ii], img_list_i[j], ssim, img_path_j, move_path, img_list_i[j]))
            else:
                print("'move_or_remove' should be one of [remove, delete, move]!")

    print(" ----------- {} ----------- ".format(ii))


def ssim_move_or_remove_same_images_multithread(img_path, imgsz=(64, 64), move_or_remove="move"):
    img_list = sorted(os.listdir(img_path))
    dir_name = os.path.basename(img_path)

    if move_or_remove == "move":
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
                t = threading.Thread(target=ssim_move_or_remove_base, args=(img_path, imgsz, img_list, ii, img_i, img_list_i, move_or_remove, move_path,))
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
            cv2img = cv2.imread(img_abs_path)
            cv2img_gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(cv2img_gray.astype(np.uint8), 5, 255, cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                cx, cy, cw, ch = cv2.boundingRect(c)
                if cw * ch < 5000:
                    cv2img[cy:cy + ch, cx:cx + cw] = (0, 0, 0)
                    print("{}: {}_{}_{}_{}".format(img, cx, cy, cw, ch))

            cv2.imwrite("{}/{}".format(save_path, img), cv2img)

        except Exception as Error:
            print(Error)


def mv_or_remove_small_images(img_path, rmsz=48, mode=0):
    img_list = sorted(os.listdir(img_path))
    dir_name = os.path.basename(img_path)
    save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_small".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    for img in tqdm(img_list):
        if os.path.isdir(img): continue
        try:
            img_abs_path = img_path + "/{}".format(img)
            img_dst_path = save_path + "/{}".format(img)
            cv2img = cv2.imdecode(np.fromfile(img_abs_path, dtype=np.uint8), cv2.IMREAD_COLOR)

            h, w = cv2img.shape[:2]
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


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
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


def yolo_transform(img, final_shape=(64, 64)):
    img_src, ratio, (dw, dh) = letterbox(img, new_shape=final_shape)
    img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 1, 0))  # HWC -> CHW
    return img, img_src


def resize_image_keep_ratio(img_path):
    img_list = sorted(os.listdir(img_path))
    dir_name = os.path.basename(img_path)
    save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_resize_keep_ratio".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    for img in img_list:
        img_abs_path = img_path + "/{}".format(img)
        cv2img = cv2.imread(img_abs_path)

        img_, img_src = yolo_transform(cv2img, final_shape=(128, 128))
        cv2.imwrite("{}/{}".format(save_path, img), img_src)


def select_images_according_other_images():
    src_img_path = "/home/zengyifan/wujiahu/data/003.Cigar_Detection/others/optimize_cls_model/MPII_select_images/crop_images_/1"
    src_img_list = os.listdir(src_img_path)

    data_path = "/home/zengyifan/wujiahu/data/003.Cigar_Detection/others/optimize_cls_model/MPII_select_images/crop_images"
    data_path_1 = data_path + "/1"
    os.makedirs(data_path_1, exist_ok=True)

    for img in src_img_list:
        img_abs_path = data_path + "/{}".format(img)
        img_dst_path = data_path_1 + "/{}".format(img)

        shutil.move(img_abs_path, img_dst_path)


def select_images_according_txt_list(data_path, print_flag=True):
    txt_path = data_path + "/labels_new"
    img_path = data_path + "/images"
    txt_list = sorted(os.listdir(txt_path))

    save_path = data_path + "/images_new"
    os.makedirs(save_path, exist_ok=True)

    for t in txt_list:
        f_name = t.replace(".txt", ".jpg")
        img_abs_path = img_path + "/{}".format(f_name)
        img_dst_path = save_path + "/{}".format(f_name)
        shutil.copy(img_abs_path, img_dst_path)
        if print_flag:
            print("{} --> {}".format(img_abs_path, img_dst_path))


def select_images_according_txt_file(txt_file, img_path, save_path=None, cp_mv_del="copy"):
    img_dir_name = os.path.basename(img_path)

    with open(txt_file, "r", encoding="utf-8") as fo:
        lines = fo.readlines()

    if save_path is None:
        save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_selected_according_txt_file".format(img_dir_name)
        os.makedirs(save_path, exist_ok=True)
    else:
        os.makedirs(save_path, exist_ok=True)

    for line in lines:
        line = line.strip()
        # absolute path
        if "/" or "\\" in line:
            img_name = os.path.basename(line)
            img_src_path = img_path + "/{}".format(img_name)
            img_dst_path = save_path + "/{}".format(img_name)

            if cp_mv_del == "cp" or cp_mv_del == "copy":
                shutil.copy(img_src_path, img_dst_path)
            elif cp_mv_del == "mv" or cp_mv_del == "move":
                shutil.move(img_src_path, img_dst_path)
            elif cp_mv_del == "del" or cp_mv_del == "delete" or cp_mv_del == "rm" or cp_mv_del == "remove":
                os.remove(img_src_path)
            else:
                print("Error 'cp_mv_del' parameter!")
        # relative path
        else:
            # img_name = os.path.basename(line)
            img_src_path = img_path + "/{}".format(line)
            img_dst_path = save_path + "/{}".format(line)

            if cp_mv_del == "cp" or cp_mv_del == "copy":
                shutil.copy(img_src_path, img_dst_path)
            elif cp_mv_del == "mv" or cp_mv_del == "move":
                shutil.move(img_src_path, img_dst_path)
            elif cp_mv_del == "del" or cp_mv_del == "delete" or cp_mv_del == "rm" or cp_mv_del == "remove":
                os.remove(img_src_path)
            else:
                print("Error 'cp_mv_del' parameter!")


def select_images_according_yolo_label(data_path="", save_path=""):
    dir_name = os.path.basename(data_path)
    if save_path:
        save_path_images = save_path + "/images"
        save_path_labels = save_path + "/labels"
        os.makedirs(save_path_images, exist_ok=True)
        os.makedirs(save_path_labels, exist_ok=True)
    else:
        save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_selected_images_and_labels_according_yolo_label".format(dir_name)
        save_path_images = save_path + "/images"
        save_path_labels = save_path + "/labels"
        os.makedirs(save_path_images, exist_ok=True)
        os.makedirs(save_path_labels, exist_ok=True)

    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    lbl_list = sorted(os.listdir(lbl_path))

    for lbl in lbl_list:
        lbl_abs_path = lbl_path + "/{}".format(lbl)
        lbl_name = os.path.splitext(lbl)[0]
        img_abs_path = img_path + "/{}.jpg".format(lbl_name)

        lbl_sum = 0
        with open(lbl_abs_path, "r", encoding="utf-8") as fo:
            lines = fo.readlines()

            for l in lines:
                l = l.strip().split(" ")
                cls = int(l[0])
                if cls == 1:
                    lbl_sum += 1

        if lbl_sum >= 1:
            img_dst_path = save_path_images + "/{}.jpg".format(lbl_name)
            lbl_dst_path = save_path_labels + "/{}".format(lbl)

            shutil.copy(img_abs_path, img_dst_path)
            shutil.copy(lbl_abs_path, lbl_dst_path)


def apply_hog(img):
    from skimage import feature, exposure

    fd, hog_img = feature.hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    return hog_img_rescaled


def MinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    # 使用opencv的erode函数更高效

    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    ''''引导滤波，直接参考网上的matlab代码'''
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
    # 输入rgb图像，值范围[0,1]
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


def dehaze_test():
    m = deHaze(cv2.imread('/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data/20221111152824_8b46d8_75_0028505.jpg') / 255.0) * 255
    cv2.imwrite('/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data/20221111152824_8b46d8_75_0028505_defog.jpg', m)


def cal_saliency_map_FT(src):
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
        saliency_map = cal_saliency_map_FT(img) * 255
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
        # bg_cv2img_resz = cv2.resize(cv2img, (cA.shape[1], cA.shape[0]))
        bg_energy_gray = (bgcH ** 2 + bgcV ** 2 + bgcD ** 2).sum() / bg_roi.size
        print("E_bg_gray: ", bg_energy_gray)

        bg_roi = bg_b[y:y + h, x:x + w]
        bgcA, (bgcH, bgcV, bgcD) = dwt2(bg_roi, "haar")
        # bgcAH = np.hstack((bgcA, bgcH))
        # bgcVD = np.hstack((bgcV, bgcD))
        # bgcAHVD = np.vstack((bgcAH, bgcVD))
        # bg_cv2img_resz = cv2.resize(cv2img, (cA.shape[1], cA.shape[0]))
        bg_energy = (bgcH ** 2 + bgcV ** 2 + bgcD ** 2).sum() / bg_roi.size
        print("E_bg_b: ", bg_energy)

        b_roi = img_gray[y:y + h, x:x + w]
        cA, (cH, cV, cD) = dwt2(b_roi, "haar")
        cAH = np.hstack((cA, cH))
        cVD = np.hstack((cV, cD))
        cAHVD = np.vstack((cAH, cVD))
        cv2img_resz = cv2.resize(cAHVD, (W, H))
        cv2img_resz_merge = cv2.merge([cv2img_resz, cv2img_resz, cv2img_resz])
        energy_gray = (cH ** 2 + cV ** 2 + cD ** 2).sum() / b_roi.size
        print("E_gray: ", energy_gray)

        b_roi = b[y:y + h, x:x + w]
        cA, (cH, cV, cD) = dwt2(b_roi, "haar")
        cAH = np.hstack((cA, cH))
        cVD = np.hstack((cV, cD))
        cAHVD = np.vstack((cAH, cVD))
        cv2img_resz = cv2.resize(cAHVD, (W, H))
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
        out_img = np.hstack((img, saliency_map, b_bin_merge, cv2img_resz_merge, img_cp))
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
        cv2img = cv2.imread(img_abs_path, 0)
        cA, (cH, cV, cD) = dwt2(cv2img, "haar")

        cAH = np.hstack((cA, cH))
        cVD = np.hstack((cV, cD))
        cAHVD = np.vstack((cAH, cVD))
        cv2.imwrite("{}/{}_dwt2.jpg".format(save_path, img_name), cAHVD)

        cv2img_resz = cv2.resize(cv2img, (cA.shape[1], cA.shape[0]))
        img_cha = cv2.subtract(np.uint8(cv2.merge([cv2img_resz, cv2img_resz, cv2img_resz])), np.uint8(cv2.merge([cA, cA, cA])))
        # img_cha = cv2.subtract(cv2.merge([cv2img_resz, cv2img_resz, cv2img_resz]), cv2.merge([cA, cA, cA]))
        # img_cha = cv2.subtract(cv2.merge([cv2img_resz, cv2img_resz, cv2img_resz]), cv2.merge([cA, cA, cA]))
        print(img_cha.sum())
        cv2.imwrite("{}/{}_img_cha.jpg".format(save_path, img_name), img_cha)

        energy = (cH ** 2 + cV ** 2 + cD ** 2).sum() / cv2img.size
        print("E: ", energy)

        Es.append(energy)

    print("E mean: ", np.mean(Es))


def hog_test():
    data_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data"
    img_list = sorted(os.listdir(data_path))
    for img in img_list:
        img_name = os.path.splitext(img)[0]
        img_abs_path = data_path + "/{}".format(img)
        img = cv2.imread(img_abs_path, 0)
        hog_res = apply_hog(img)
        cv2.imwrite("{}/{}_hog.jpg".format(data_path, img_name), hog_res * 255)


def image_process_fire_smoke_experiment():
    data_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/train/train_fire_smoke/20230517_yolov5_experiment/train_5000"
    dir_name = os.path.basename(data_path)
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    img_list = sorted(os.listdir(img_path))
    # gammas = [0.4, 0.8, 1.2, 1.8]
    gammas = [0.8, 1.2]

    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_gamma_correction_output_0.8_1.2".format(dir_name)
    img_save_path = save_path + "/images"
    lbl_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    for img in img_list:
        img_name = os.path.splitext(img)[0]
        img_abs_path = img_path + "/{}".format(img)
        lbl_abs_path = lbl_path + "/{}.txt".format(img_name)
        img = cv2.imread(img_abs_path)
        for gamma in gammas:
            gamma_correction_res = gamma_correction(img, gamma=gamma)
            if len(gamma_correction_res.shape) == 2:
                gamma_correction_res_ = cv2.cvtColor(gamma_correction_res, cv2.COLOR_GRAY2BGR)
                cv2.imwrite("{}/{}_gamma_{}_correction_res.jpg".format(img_save_path, img_name, gamma), gamma_correction_res_)
                shutil.copy(lbl_abs_path, lbl_save_path + "/{}_gamma_{}_correction_res.txt".format(img_name, gamma))
            else:
                cv2.imwrite("{}/{}_gamma_{}_correction_res.jpg".format(img_save_path, img_name, gamma), gamma_correction_res)
                shutil.copy(lbl_abs_path, lbl_save_path + "/{}_gamma_{}_correction_res.txt".format(img_name, gamma))


def image_process_fire_smoke_experiment2():
    data_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/train/train_fire_smoke/20230517_yolov5_experiment/train_5000"
    dir_name = os.path.basename(data_path)
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    img_list = sorted(os.listdir(img_path))

    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_gamma_correction_output_auto".format(dir_name)
    img_save_path = save_path + "/images"
    lbl_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    for img in img_list:
        img_name = os.path.splitext(img)[0]
        img_abs_path = img_path + "/{}".format(img)
        lbl_abs_path = lbl_path + "/{}.txt".format(img_name)
        img = cv2.imread(img_abs_path)
        for i in range(2, 3):
            gamma_correction_res, gamma = gamma_correction_auto(img, method=i)
            if len(gamma_correction_res.shape) == 2:
                gamma_correction_res_ = cv2.cvtColor(gamma_correction_res, cv2.COLOR_GRAY2BGR)
                cv2.imwrite("{}/{}_gamma_{}_correction_res_method_{}.jpg".format(img_save_path, img_name, gamma, i), gamma_correction_res_)
                shutil.copy(lbl_abs_path, lbl_save_path + "/{}_gamma_{}_correction_res_method_{}.txt".format(img_name, gamma, i))
            else:
                cv2.imwrite("{}/{}_gamma_{}_correction_res_method_{}.jpg".format(img_save_path, img_name, gamma, i), gamma_correction_res)
                shutil.copy(lbl_abs_path, lbl_save_path + "/{}_gamma_{}_correction_res_method_{}.txt".format(img_name, gamma, i))


def convert_to_gray_image(data_path):
    img_list = sorted(os.listdir(data_path))
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/images_gray"
    os.makedirs(save_path, exist_ok=True)

    for i in img_list:
        img_abs_path = data_path + "/{}".format(i)
        img = cv2.imread(img_abs_path, 0)
        cv2.imwrite("{}/{}".format(save_path, i), img)


def crop_one_image(img_path, crop_area):
    cv2img = cv2.imread(img_path)
    par_path = os.path.abspath(os.path.join(img_path, "../.."))
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    cropped = cv2img[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3]]
    cv2.imwrite("{}/{}_cropped.jpg".format(par_path, img_name), cropped)


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
        cv2img = cv2.imread(img_abs_path)
        imgsz = cv2img.shape[:2]
        b, g, r = cv2.split(cv2img)
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


def rotate_img_any_angle(img_path):
    img_list = sorted(os.listdir(img_path))
    dir_name = os.path.basename(img_path)
    save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_ratated".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    angles = list(range(5, 180, 5))
    for img in img_list:
        img_abs_path = img_path + "/{}".format(img)
        img_name = os.path.splitext(img)[0]
        pilimg = Image.open(img_abs_path)
        for ang in angles:
            try:
                pilimg.rotate(ang, expand=True).save("{}/{}_{}.jpg".format(save_path, img_name, ang))
            except Exception as Error:
                print(Error, Error.__traceback__.tb_lineno)


def ssim_move_images(base_img_path, imgs_path, imgsz=(32, 32), ssim_thr=0.5):
    from skimage.metrics import structural_similarity

    img_list = sorted(os.listdir(imgs_path))
    base_img = cv2.imread(base_img_path)
    img_i = cv2.resize(base_img, imgsz)
    for j in tqdm(img_list):
        img_path_j = imgs_path + "/{}".format(j)
        img_j = cv2.imread(img_path_j)
        if img_j is None: continue
        img_j = cv2.resize(img_j, imgsz)

        ssim = structural_similarity(img_i, img_j, multichannel=True)
        print("ssim:", ssim)

        if ssim > ssim_thr:
            img_dst_path = "/home/zengyifan/wujiahu/data/004.Knife_Det/others/Robot_Test/Videos/C_Plus_Plus_det_output/20230711_video_frames_merged/crop_images/1/1.1" + "/{}".format(j)
            shutil.move(img_path_j, img_dst_path)
            print("{} -->\n{}".format(img_path_j, img_dst_path))


def canny_demo(image):
    t = 80
    canny_output = cv2.Canny(image, t, t * 2)

    return canny_output


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


def find_red_bbx_v2(img_path, cls):
    dir_name = os.path.basename(img_path)
    lbl_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_labels".format(dir_name)
    os.makedirs(lbl_path, exist_ok=True)

    img_list = sorted(os.listdir(img_path))
    for img in tqdm(img_list):
        img_name = os.path.splitext(img)[0]
        img_abs_path = img_path + "/{}".format(img)
        cv2img = cv2.imread(img_abs_path)
        imgsz = cv2img.shape[:2]

        with open(lbl_path + "/{}.txt".format(img_name), "w", encoding="utf-8") as fw:
            src = GetRed(cv2img)
            binary = canny_demo(src)
            k = np.ones((3, 3), dtype=np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, k)

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

                roi = cv2img[y1 + 5: y2 - 5, x1 + 5:x2 - 5]
                # print(width, height)
                # print(x1,y1,x2,y2)
                if width < 80 or height < 80:
                    continue

                # cv2.imshow("roi", roi)
                # cv2.waitKey(0)
                if len(roi):
                    # cv2.imwrite("{}/{}_{}.jpg".format(lbl_path, img_name, c), roi)
                    # bbx_voc = [x1, x2, y1, y2]
                    # bbx_yolo = convert_bbx_VOC_to_yolo(imgsz, bbx_voc)
                    bbx_voc = [x1, y1, y1, y2]
                    bbx_yolo = convertBboxVOC2YOLO(imgsz, bbx_voc)
                    txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bbx_yolo]) + "\n"
                    fw.write(txt_content)


def find_red_bbx(cv2img, expand_p=2):
    src = GetRed(cv2img)
    binary = canny_demo(src)
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

        roi = cv2img[y1 + expand_p: y2 - expand_p, x1 + expand_p:x2 - expand_p]
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


def seg_crop_object(cv2img, bgimg, maskimg):
    # imgsz = cv2img.shape
    outimg = np.zeros(cv2img.shape)
    # outimg2 = bgimg.copy()
    # roi = np.where(maskimg[:, :, 0] != 0 & maskimg[:, :, 1] != 0 & maskimg[:, :, 2] != 0)
    roi = np.where(maskimg[:, :, 0] != 0)
    outimg[roi] = cv2img[roi]
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
        cv2img = cv2.imread(f_abs_path)
        imgsz = cv2img.shape[:2]
        # cropped = cv2img[212:imgsz[0], :]
        cropped = cv2img[0:165, :]
        cv2.imwrite("{}/{}".format(save_path, f), cropped)


class BlurAug(object):
    def __init__(self, ratio=1.0, type="EASY"):  # easy hard
        self.ratio = ratio
        self.pre_rotate_angle = 135.0
        self.type = type

    def padding(self, img):
        res = math.sqrt(img.shape[0] * img.shape[0] + img.shape[1] * img.shape[1])
        pad_x = int(res - img.shape[1] * 0.5 + 1)
        pad_y = int(res - img.shape[0] * 0.5 + 1)
        img_pad = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_CONSTANT, value=0)
        return img_pad, (pad_x, pad_y)

    def aug_resize(self, img):
        img, crop_rect = self.padding(img)
        angle = random.uniform(-self.pre_rotate_angle, self.pre_rotate_angle)
        rows, cols, _ = img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, affine_mat, (cols, rows))

        factor = random.uniform(0, 1.0)
        if (self.type == "EASY"):
            scale = factor * 0.25 + 0.8
        else:
            scale = factor * 0.1 + 0.2
        rows, cols, _ = img.shape
        dst = cv2.resize(dst, (int(cols * scale), int(rows * scale)))
        dst = cv2.resize(dst, (cols, rows))

        rows, cols, _ = dst.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360.0 - angle, 1)
        out_img = cv2.warpAffine(dst, affine_mat, (cols, rows))
        out_img = out_img[crop_rect[1]: out_img.shape[0] - crop_rect[1], crop_rect[0]: out_img.shape[1] - crop_rect[0], :]
        return out_img

    def aug_blur(self, img):
        img, crop_rect = self.padding(img)
        angle = random.uniform(-self.pre_rotate_angle, self.pre_rotate_angle)
        rows, cols, _ = img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, affine_mat, (cols, rows))
        if (self.type == "EASY"):
            random_value = random.randint(0, 3)
            size = int(random_value / 2) * 2 + 1
        else:
            random_value = random.randint(5, 7)
            size = int(random_value / 2) * 2 + 3
        blur_img = cv2.blur(dst, (size, size))
        rows, cols, _ = blur_img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360.0 - angle, 1)
        out_img = cv2.warpAffine(blur_img, affine_mat, (cols, rows))
        out_img = out_img[crop_rect[1]: out_img.shape[0] - crop_rect[1], crop_rect[0]: out_img.shape[1] - crop_rect[0], :]
        return out_img

    def aug_motion_blur(self, img):
        img, crop_rect = self.padding(img)
        angle = random.uniform(-self.pre_rotate_angle, self.pre_rotate_angle)
        rows, cols, _ = img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, affine_mat, (cols, rows))

        if self.type == "EASY":
            size = int(random.uniform(0.0, 3.0) + 2)
        else:
            size = int(random.uniform(5.0, 7.0) + 5)
        kernel = np.zeros((size, size), np.float32)
        h = (size - 1) // 2
        for i in range(size):
            kernel[h][i] = 1.0 / float(size)

        blur_img = cv2.filter2D(dst, -1, kernel)
        rows, cols, _ = blur_img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360.0 - angle, 1)
        out_img = cv2.warpAffine(blur_img, affine_mat, (cols, rows))

        out_img = out_img[crop_rect[1]: out_img.shape[0] - crop_rect[1], crop_rect[0]: out_img.shape[1] - crop_rect[0], :]
        return out_img

    def aug_medianblur(self, img):
        img, crop_rect = self.padding(img)
        angle = random.uniform(-self.pre_rotate_angle, self.pre_rotate_angle)
        rows, cols, _ = img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, affine_mat, (cols, rows))
        if (self.type == "EASY"):
            random_value = random.randint(0, 3)
            size = int(random_value / 2) * 2 + 1
        else:
            random_value = random.randint(3, 7)
            size = int(random_value / 2) * 2 + 3
        blur_img = cv2.medianBlur(dst, size)
        rows, cols, _ = blur_img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360.0 - angle, 1)
        out_img = cv2.warpAffine(blur_img, affine_mat, (cols, rows))
        out_img = out_img[crop_rect[1]: out_img.shape[0] - crop_rect[1], crop_rect[0]: out_img.shape[1] - crop_rect[0], :]
        return out_img

    def aug_gaussblur(self, img):
        img, crop_rect = self.padding(img)
        angle = random.uniform(-self.pre_rotate_angle, self.pre_rotate_angle)
        rows, cols, _ = img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, affine_mat, (cols, rows))
        if (self.type == "EASY"):
            random_value = random.randint(0, 2)
            size = int(random_value / 2) * 2 + 3
        else:
            random_value = random.randint(5, 7)
            size = int(random_value / 2) * 2 + 7
        blur_img = cv2.GaussianBlur(dst, (size, size), 0)
        rows, cols, _ = blur_img.shape
        affine_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 360.0 - angle, 1)
        out_img = cv2.warpAffine(blur_img, affine_mat, (cols, rows))
        out_img = out_img[crop_rect[1]: out_img.shape[0] - crop_rect[1], crop_rect[0]: out_img.shape[1] - crop_rect[0], :]
        return out_img

    def __call__(self, img):
        if (np.random.rand() < self.ratio):
            img = img.astype(np.uint8)
            select_id = random.choice([1, 2, 4])
            # select_id = random.choice( [0] )
            if (select_id == 0):
                img = self.aug_resize(img)
            elif (select_id == 1):
                img = self.aug_blur(img)
            elif (select_id == 2):
                img = self.aug_motion_blur(img)
            elif (select_id == 3):
                img = self.aug_medianblur(img)
            else:
                img = self.aug_gaussblur(img)
            # print ("blur type : " , select_id)
            img = img.astype(np.float32)

        # bbox_mosaic = results["gt_bboxes"]
        # img_mosaic = results["img"].astype(np.uint8)
        # for k in range(bbox_mosaic.shape[0]):
        #     bbox = bbox_mosaic[k].astype(np.int)
        #     cv2.rectangle(img_mosaic, (bbox[0], bbox[1]), (bbox[2], bbox[3]) , (0,0,255), 2)
        # if (img_mosaic.shape[0] > 1000 or img_mosaic.shape[1] > 1000):
        #     img_mosaic = cv2.resize(img_mosaic, (img_mosaic.shape[1] // 4, img_mosaic.shape[0] // 4 ) )
        # cv2.imshow("blur", img_mosaic)
        # cv2.waitKey(-1)

        return img


class NoiseAug(object):
    def __init__(self, ratio=0.9):
        self.ratio = ratio

    # sault and peper noise
    def sp_noise(self, image, prob):
        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    def gasuss_noise(self, image, mean=0, var=0.001):
        image = np.array(image / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        out = np.clip(out, 0.0, 1.0)
        out = np.uint8(out * 255)
        # cv.imshow("gasuss", out)
        return out

    def __call__(self, img):
        if (np.random.rand() < self.ratio):
            img = img.astype(np.uint8)

            select_id = random.choice([0, 1])
            if (select_id == 0):
                img = self.sp_noise(img, 0.01)
            elif (select_id == 1):
                img = self.gasuss_noise(img, mean=0, var=0.005)
            # img = self.gasuss_noise(img)

            img = img.astype(np.float32)

        # bbox_mosaic = results["gt_bboxes"].astype(np.int)
        # img_mosaic = results["img"].astype(np.uint8)
        # for k in range(bbox_mosaic.shape[0]):
        #     bbox = bbox_mosaic[k]
        #     cv2.rectangle(img_mosaic, (bbox[0], bbox[1]), (bbox[2], bbox[3]) , (0,0,255), 2)
        # cv2.imshow("noise", img_mosaic)
        # cv2.waitKey(-1)

        return img


def get_trans_mat(center, degrees=0, translate=(0, 0), scale=1, shear=(0, 0), perspective=(0, 0)):
    C = np.eye(3)
    C[0, 2] = center[0]  # x translation (pixels)
    C[1, 2] = center[1]  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = perspective[0]  # x perspective (about y)
    P[2, 1] = perspective[1]  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = degrees
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = scale
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = shear[0]  # x shear (deg)
    S[1, 0] = shear[1]  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = translate[0]  # x translation (pixels)
    T[1, 2] = translate[1]  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    return M


def TransAffine(img, degrees=10, translate=0.1, scale=0.1, shear=0.1, perspective=0.1, border=(4, 4), prob=0.5):
    if (random.random() < prob):
        img = img  # results["img"]
        height = img.shape[0]
        width = img.shape[1]

        center_src = (-img.shape[1] / 2, -img.shape[0] / 2)
        perspective_src = (random.uniform(-perspective, perspective), random.uniform(-perspective, perspective))
        degrees_src = random.uniform(-degrees, degrees)
        scale_src = random.uniform(1 - 0.25, 1 + scale)
        shear_src = (math.tan(random.uniform(-shear, shear) * math.pi / 180), math.tan(random.uniform(-shear, shear) * math.pi / 180))
        translate_src = [random.uniform(0.5 - translate, 0.5 + translate) * width, random.uniform(0.5 - translate, 0.5 + translate) * height]

        M_src = get_trans_mat(center_src, degrees_src, translate_src, scale_src, shear_src, perspective_src)
        four_pt = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]])
        res_pt = M_src @ four_pt.T
        res_pt = res_pt.astype(np.int).T
        res_pt = res_pt[:, :2]
        min_x = np.min(res_pt[:, 0])
        max_x = np.max(res_pt[:, 0])
        min_y = np.min(res_pt[:, 1])
        max_y = np.max(res_pt[:, 1])
        if (min_x < 0):
            translate_src[0] -= min_x
        if (min_y < 0):
            translate_src[1] -= min_y

        if (max_x - min_x > width):
            new_width = (max_x - min_x)
        else:
            new_width = width
        if (max_y - min_y > height):
            new_height = (max_y - min_y)
        else:
            new_height = height

        M = get_trans_mat((-width / 2, -height / 2), degrees_src, translate_src, scale_src, shear_src, perspective_src)

        border_color = (random.randint(220, 250), random.randint(220, 250), random.randint(220, 250))
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)
        return img
    else:
        return img


class HSVAug(object):
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5, ratio=0.95):
        self.ratio = ratio
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img):
        if (np.random.rand() < self.ratio):
            img = img.astype(np.uint8)
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
            # Histogram equalization
            # if random.random() < 0.2:
            #     for i in range(3):
            #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])
            img = img.astype(np.float32)

        # bbox_mosaic = results["gt_bboxes"].astype(np.int)
        # img_mosaic = results["img"].astype(np.uint8)
        # for k in range(bbox_mosaic.shape[0]):
        #     bbox = bbox_mosaic[k]
        #     cv2.rectangle(img_mosaic, (bbox[0], bbox[1]), (bbox[2], bbox[3]) , (0,0,255), 2)
        # cv2.imshow("hsv_img", img_mosaic)
        # cv2.waitKey(-1)

        return img


def doing_aug(img, use_trans_affine=True):
    if (use_trans_affine):
        # border_width = random.randint(2,4)
        # border_height = random.randint(2,4)
        border_width = 0
        border_height = 0
        # img = TransAffine(img, degrees=8, translate=0.0, scale=0.2, shear=0, perspective=0, border=(2,border_width), prob=0.95)
        img = TransAffine(img, degrees=3, translate=0.00025, scale=0.1, shear=3, perspective=0.0005, border=(border_height, border_width), prob=1.0)

        # TODO tiling  resize x or y resize !
    img = img.astype(np.uint8)
    return img


def do_aug(data_path):
    dir_name = os.path.basename(data_path)
    file_list = sorted(os.listdir(data_path))

    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_aug".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    datetime = get_strftime()

    for f in tqdm(file_list):
        try:
            f_abs_path = data_path + "/{}".format(f)
            f_dst_path = save_path + "/{}".format(f)
            fname = os.path.basename(f)
            img_name, suffix = os.path.splitext(fname)[0], os.path.splitext(fname)[1]
            img_name0 = img_name.split("=")[0]
            label = img_name.split("=")[1]
            cv2img = cv2.imread(f_abs_path)

            noise_aug = NoiseAug(ratio=0.9)
            blur_rdm = np.random.random()
            if blur_rdm < 0.5:
                blur_aug = BlurAug(type="EASY", ratio=0.9)
            else:
                blur_aug = BlurAug(type="HARD", ratio=0.9)
            hsv_aug = HSVAug(hgain=0.2, sgain=0.7, vgain=0.5, ratio=0.9)

            cv2img = noise_aug(cv2img)
            cv2img = blur_aug(cv2img)
            cv2img = hsv_aug(cv2img)
            cv2img_aug = doing_aug(cv2img)

            fname_rdm = np.random.random()
            cv2.imwrite("{}/{}_aug_{}_{}={}.jpg".format(save_path, datetime, str(fname_rdm).replace(".", ""), img_name0, label), cv2img_aug)
        except Exception as Error:
            print(Error)


def do_aug_base(file_list_i, data_path, save_path):
    # dir_name = os.path.basename(data_path)
    # file_list = sorted(os.listdir(data_path))

    # save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_aug".format(dir_name)
    # os.makedirs(save_path, exist_ok=True)

    datetime = get_strftime()

    for f in tqdm(file_list_i):
        try:
            f_abs_path = data_path + "/{}".format(f)
            f_dst_path = save_path + "/{}".format(f)
            fname = os.path.basename(f)
            img_name, suffix = os.path.splitext(fname)[0], os.path.splitext(fname)[1]
            img_name0 = img_name.split("=")[0]
            label = img_name.split("=")[1]
            cv2img = cv2.imread(f_abs_path)

            noise_aug = NoiseAug(ratio=0.9)
            blur_rdm = np.random.random()
            if blur_rdm < 0.5:
                blur_aug = BlurAug(type="EASY", ratio=0.9)
            else:
                blur_aug = BlurAug(type="HARD", ratio=0.9)
            hsv_aug = HSVAug(hgain=0.2, sgain=0.7, vgain=0.5, ratio=0.9)

            cv2img = noise_aug(cv2img)
            cv2img = blur_aug(cv2img)
            cv2img = hsv_aug(cv2img)
            cv2img_aug = doing_aug(cv2img)

            fname_rdm = np.random.random()
            cv2.imwrite("{}/{}_aug_{}_{}={}.jpg".format(save_path, datetime, str(fname_rdm).replace(".", ""), img_name0, label), cv2img_aug)
        except Exception as Error:
            print(Error)


def do_aug_multithreading(data_path, split_n=8):
    dir_name = os.path.basename(data_path)
    file_list = sorted(os.listdir(data_path))

    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_aug".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    len_ = len(file_list)

    img_lists = []
    for j in range(split_n):
        img_lists.append(file_list[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

    t_list = []
    for i in range(split_n):
        list_i = img_lists[i]
        t = threading.Thread(target=do_aug_base, args=(list_i, data_path, save_path,))
        t_list.append(t)

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


def sliding_window_crop(img, cropsz=(96, 256), gap=(0, 128)):
    cropped_imgs = []
    imgsz = img.shape[:2]

    if gap[0] == 0 and gap[1] > 0:
        cropsz = (imgsz[0], cropsz[1])
        for i in range(0, imgsz[1], gap[1]):
            if i + cropsz[1] >= imgsz[1]:
                i_last = imgsz[1] - cropsz[1]

                cp_img = img[0:imgsz[0], i_last:i_last + cropsz[1], :]
                cropped_imgs.append(cp_img)
                break
            else:
                if i + cropsz[1] > imgsz[1]:
                    break
                cp_img = img[0:imgsz[0], i:i + cropsz[1], :]
                cropped_imgs.append(cp_img)
    elif gap[0] > 0 and gap[1] == 0:
        cropsz = (cropsz[0], imgsz[1])
        for j in range(0, imgsz[0], gap[0]):
            if j + cropsz[0] >= imgsz[0]:
                j_last = imgsz[0] - cropsz[0]

                cp_img = img[j_last:j_last + cropsz[0], 0:imgsz[1], :]
                cropped_imgs.append(cp_img)
                break
            else:
                if j + cropsz[0] > imgsz[0]:
                    break
                cp_img = img[j:j + cropsz[0], 0:imgsz[1], :]
                cropped_imgs.append(cp_img)
    elif gap[0] == 0 and gap[1] == 0:
        print("Error! gap[0] == 0 and gap[1] == 0!")
    else:
        for j in range(0, imgsz[0], gap[0]):
            if j + cropsz[0] >= imgsz[0]:
                j_last = imgsz[0] - cropsz[0]

                for i in range(0, imgsz[1], gap[1]):
                    if i + cropsz[1] >= imgsz[1]:
                        i_last = imgsz[1] - cropsz[1]

                        cp_img = img[j_last:j_last + cropsz[0], i_last:i_last + cropsz[1], :]
                        cropped_imgs.append(cp_img)
                        break
                    else:
                        if i + cropsz[1] > imgsz[1]:
                            break
                        cp_img = img[j_last:j_last + cropsz[0], i:i + cropsz[1], :]
                        cropped_imgs.append(cp_img)
                break

            else:
                for i in range(0, imgsz[1], gap[1]):
                    if j + cropsz[0] > imgsz[0] or i + cropsz[1] > imgsz[1]:
                        break
                    cp_img = img[j:j + cropsz[0], i:i + cropsz[1], :]
                    cropped_imgs.append(cp_img)

    return cropped_imgs


def sliding_window_crop_test2():
    img_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/sliding_window_test/images/bg_natural_images_0000000.jpg"
    cv2img = cv2.imread(img_path)
    imgsz = cv2img.shape[:2]

    cropped_imgs = sliding_window_crop(cv2img, cropsz=(96, 256), gap=(48, 128))
    for idx, ci in enumerate(cropped_imgs):
        cv2.imwrite("/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/sliding_window_test/images_sliding_window_crop/{}.jpg".format(idx), ci)


def sliding_window_crop_test():
    ratio = 0.7  # 像素占比

    # img_path = "/home/zengyifan/wujiahu/data/000.Bg/bg_natural_images_21781/images/bg_natural_images_0000000.jpg"
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # h, w, _ = img.shape
    # crop_w, crop_h = 200, 100  # 定义裁剪图像尺寸
    # gap_w, gap_h = 200, 100  # 定义滑动间隔
    # gp_w, gp_h = 100, 50
    # cp_w, cp_h = 100, 50
    cropsz = (560, 96)
    gap = (cropsz[0] // 2, cropsz[1] // 2)

    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/sliding_window_test/images"
    dir_name = get_dir_name(data_path)
    file_list = get_file_list(data_path)
    save_path = make_save_path(data_path, dir_name_add_str="sliding_window_crop")

    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        base_name, file_name, suffix = get_baseName_fileName_suffix(f_abs_path)
        cv2img = cv2.imread(f_abs_path)
        imgsz = cv2img.shape[:2]

        num = 0
        for j in range(0, imgsz[0], gap[0]):
            if j + cropsz[0] > imgsz[0]:
                j_last = imgsz[0] - cropsz[0]

                for i in range(0, imgsz[1], gap[1]):
                    if i + cropsz[1] > imgsz[1]:
                        i_last = imgsz[1] - cropsz[1]

                        print("+" * 200)
                        print(j_last, j_last + cropsz[0], i_last, i_last + cropsz[1])
                        cp_img = cv2img[j_last:j_last + cropsz[0], i_last:i_last + cropsz[1], :]
                        cv2.imwrite(os.path.join(save_path, base_name.replace('.jpg', f'_{num}.jpg')), cp_img)

                        num += 1


                    else:
                        print("&" * 200)
                        print(j_last, j_last + cropsz[0], i, i + cropsz[1])
                        cp_img = cv2img[j_last:j_last + cropsz[0], i:i + cropsz[1], :]
                        cv2.imwrite(os.path.join(save_path, base_name.replace('.jpg', f'_{num}.jpg')), cp_img)

                        num += 1
            else:
                for i in range(0, imgsz[1], gap[1]):
                    if i + cropsz[1] > imgsz[1]:
                        i = imgsz[1] - cropsz[1]

                    print(j, j + cropsz[0], i, i + cropsz[1])

                    if j + cropsz[0] > imgsz[0]:
                        j_last = imgsz[0] - cropsz[0]
                    if i + cropsz[1] > imgsz[1]:
                        i_last = imgsz[1] - cropsz[1]

                    cp_img = cv2img[j:j + cropsz[0], i:i + cropsz[1], :]
                    cv2.imwrite(os.path.join(save_path, base_name.replace('.jpg', f'_{num}.jpg')), cp_img)

                    num += 1


def crop_red_bbx_area(data_path, expand_p=5):
    file_list = get_file_list(data_path)
    save_path = make_save_path(data_path, dir_name_add_str="crop_red_bbx")
    # expand_p = 10

    for f in tqdm(file_list):
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        cv2img = cv2.imread(f_abs_path)
        results = find_red_bbx(cv2img, expand_p=expand_p)
        for ri, r in enumerate(results):
            try:
                f_ri_dst_path = save_path + "/{}_{}_{}_cropped.jpg".format(fname, expand_p, ri)
                cropped = cv2img[r[2]:r[3], r[0]:r[1]]
                cv2.imwrite(f_ri_dst_path, cropped)
            except Exception as Error:
                print(Error)


def gray_img_or_not(cv2img, dstsz=(64, 64), mean_thr=5):
    cv2img = cv2.resize(cv2img, dstsz)
    imgsz = cv2img.shape[:2]
    psum = []
    for hi in range(imgsz[0]):
        for wi in range(imgsz[1]):
            pgap = (abs(cv2img[hi, wi, 0] - cv2img[hi, wi, 1]) + abs(cv2img[hi, wi, 1] - cv2img[hi, wi, 2]) + abs(cv2img[hi, wi, 0] - cv2img[hi, wi, 2])) / 3
            psum.append(pgap)

    pmean = np.mean(psum)

    if pmean < mean_thr:
        return "Gray"
    else:
        return "Color"


def classify_color_and_gray_images(data_path):
    file_list = get_file_list(data_path)
    save_path0 = make_save_path(data_path, "output/0")
    save_path1 = make_save_path(data_path, "output/1")

    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        cv2img = cv2.imread(f_abs_path)
        gray_or_color = gray_img_or_not(cv2img, dstsz=(64, 64), mean_thr=5)

        if gray_or_color.upper() == "GRAY":
            f_dst_path = save_path0 + "/{}".format(f)
            shutil.move(f_abs_path, f_dst_path)
        else:
            f_dst_path = save_path1 + "/{}".format(f)
            shutil.move(f_abs_path, f_dst_path)


def makeSunLightEffect(img, r=(50, 200), light_strength=300):
    imgsz = img.shape[:2]
    center = (np.random.randint(0, imgsz[1]), np.random.randint(0, imgsz[0]))
    effectR = np.random.randint(r[0], r[1])
    lightStrength = np.random.randint(light_strength // 4, light_strength)

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


def get_color(specific_color_flag=True):
    global color
    color1 = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

    if specific_color_flag:
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


def makeBorder_base(im, new_shape=(64, 256), r1=0.75, specific_color_flag=True):
    """
    :param im:
    :param new_shape: (H, W)
    :param r1:
    :param r2:
    :param sliding_window:
    :return:
    """
    color = get_color(specific_color_flag=specific_color_flag)

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


def sliding_window_crop_v2(img, cropsz=(64, 256), gap=(0, 128), makeBorder=True, r1=0, r2=0.25, specific_color_flag=True):
    cropped_imgs = []
    imgsz = img.shape[:2]

    if gap[0] == 0 and gap[1] > 0:
        cropsz = (imgsz[0], cropsz[1])
        for i in range(0, imgsz[1], gap[1]):
            if i + cropsz[1] > imgsz[1]:
                cp_img = img[0:imgsz[0], i:imgsz[1]]
                if makeBorder:
                    cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                    cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                            cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                            cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
                        cropped_imgs.append(cp_img)
                        break
                    else:
                        cp_img = img[j:j + cropsz[0], i:i + cropsz[1]]
                        cropped_imgs.append(cp_img)

    return cropped_imgs


def makeBorder_v6(im, new_shape=(64, 256), r1=0.75, r2=0.25, sliding_window=False, specific_color_flag=True, gap_r=(0, 7 / 8), last_img_makeBorder=True):
    """
    :param im:
    :param new_shape: (H, W)
    :param r1:
    :param r2:
    :param sliding_window:
    :return:
    """
    color = get_color(specific_color_flag=specific_color_flag)

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
    basic_color = np.array(colors.to_rgb(color)) * 255
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


class CVRandomRotation(object):
    def __init__(self, degrees=15):
        assert isinstance(degrees, numbers.Number), "degree should be a single number."
        assert degrees >= 0, "degree must be positive."
        self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        return sample_sym(degrees)

    def __call__(self, img):
        angle = self.get_params(self.degrees)
        src_h, src_w = img.shape[:2]
        M = cv2.getRotationMatrix2D(center=(src_w / 2, src_h / 2), angle=angle, scale=1.0)
        abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
        dst_w = int(src_h * abs_sin + src_w * abs_cos)
        dst_h = int(src_h * abs_cos + src_w * abs_sin)
        M[0, 2] += (dst_w - src_w) / 2
        M[1, 2] += (dst_h - src_h) / 2

        flags = get_interpolation()
        return cv2.warpAffine(img, M, (dst_w, dst_h), flags=flags, borderMode=cv2.BORDER_REPLICATE)


class CVRandomAffine(object):
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        assert isinstance(degrees, numbers.Number), "degree should be a single number."
        assert degrees >= 0, "degree must be positive."
        self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = [shear]
            else:
                assert isinstance(shear, (tuple, list)) and (len(shear) == 2), \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

    def _get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        # https://github.com/pytorch/vision/blob/v0.4.0/torchvision/transforms/functional.py#L717
        from numpy import sin, cos, tan

        if isinstance(shear, numbers.Number):
            shear = [shear, 0]

        if not isinstance(shear, (tuple, list)) and len(shear) == 2:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))

        rot = math.radians(angle)
        sx, sy = [math.radians(s) for s in shear]

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = cos(rot - sy) / cos(sy)
        b = -cos(rot - sy) * tan(sx) / cos(sy) - sin(rot)
        c = sin(rot - sy) / cos(sy)
        d = -sin(rot - sy) * tan(sx) / cos(sy) + cos(rot)

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        M = [d, -b, 0,
             -c, a, 0]
        M = [x / scale for x in M]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
        M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        M[2] += cx
        M[5] += cy
        return M

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, height):
        angle = sample_sym(degrees)
        if translate is not None:
            max_dx = translate[0] * height
            max_dy = translate[1] * height
            translations = (np.round(sample_sym(max_dx)), np.round(sample_sym(max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = sample_uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 1:
                shear = [sample_sym(shears[0]), 0.]
            elif len(shears) == 2:
                shear = [sample_sym(shears[0]), sample_sym(shears[1])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        src_h, src_w = img.shape[:2]
        angle, translate, scale, shear = self.get_params(
            self.degrees, self.translate, self.scale, self.shear, src_h)

        M = self._get_inverse_affine_matrix((src_w / 2, src_h / 2), angle, (0, 0), scale, shear)
        M = np.array(M).reshape(2, 3)

        startpoints = [(0, 0), (src_w - 1, 0), (src_w - 1, src_h - 1), (0, src_h - 1)]
        project = lambda x, y, a, b, c: int(a * x + b * y + c)
        endpoints = [(project(x, y, *M[0]), project(x, y, *M[1])) for x, y in startpoints]

        rect = cv2.minAreaRect(np.array(endpoints))
        bbox = cv2.boxPoints(rect).astype(dtype=np.int)
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()

        dst_w = int(max_x - min_x)
        dst_h = int(max_y - min_y)
        M[0, 2] += (dst_w - src_w) / 2
        M[1, 2] += (dst_h - src_h) / 2

        # add translate
        dst_w += int(abs(translate[0]))
        dst_h += int(abs(translate[1]))
        if translate[0] < 0: M[0, 2] += abs(translate[0])
        if translate[1] < 0: M[1, 2] += abs(translate[1])

        flags = get_interpolation()
        return cv2.warpAffine(img, M, (dst_w, dst_h), flags=flags, borderMode=cv2.BORDER_REPLICATE)


class CVRandomPerspective(object):
    def __init__(self, distortion=0.5):
        self.distortion = distortion

    def get_params(self, width, height, distortion):
        offset_h = sample_asym(distortion * height / 2, size=4).astype(dtype=np.int)
        offset_w = sample_asym(distortion * width / 2, size=4).astype(dtype=np.int)
        topleft = (offset_w[0], offset_h[0])
        topright = (width - 1 - offset_w[1], offset_h[1])
        botright = (width - 1 - offset_w[2], height - 1 - offset_h[2])
        botleft = (offset_w[3], height - 1 - offset_h[3])

        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return np.array(startpoints, dtype=np.float32), np.array(endpoints, dtype=np.float32)

    def __call__(self, img):
        height, width = img.shape[:2]
        startpoints, endpoints = self.get_params(width, height, self.distortion)
        M = cv2.getPerspectiveTransform(startpoints, endpoints)

        # TODO: more robust way to crop image
        rect = cv2.minAreaRect(endpoints)
        bbox = cv2.boxPoints(rect).astype(dtype=np.int)
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()
        min_x, min_y = max(min_x, 0), max(min_y, 0)

        flags = get_interpolation()
        img = cv2.warpPerspective(img, M, (max_x, max_y), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        img = img[min_y:, min_x:]
        return img


class CVRescale(object):

    def __init__(self, factor=4, base_size=(128, 512)):
        """ Define image scales using gaussian pyramid and rescale image to target scale.

        Args:
            factor: the decayed factor from base size, factor=4 keeps target scale by default.
            base_size: base size the build the bottom layer of pyramid
        """
        if isinstance(factor, numbers.Number):
            self.factor = round(sample_uniform(0, factor))
        elif isinstance(factor, (tuple, list)) and len(factor) == 2:
            self.factor = round(sample_uniform(factor[0], factor[1]))
        else:
            raise Exception('factor must be number or list with length 2')
        # assert factor is valid
        self.base_h, self.base_w = base_size[:2]

    def __call__(self, img):
        if self.factor == 0: return img
        src_h, src_w = img.shape[:2]
        cur_w, cur_h = self.base_w, self.base_h
        scale_img = cv2.resize(img, (cur_w, cur_h), interpolation=get_interpolation())
        for _ in range(self.factor):
            scale_img = cv2.pyrDown(scale_img)
        scale_img = cv2.resize(scale_img, (src_w, src_h), interpolation=get_interpolation())
        return scale_img


class CVGaussianNoise(object):
    def __init__(self, mean=0, var=20):
        self.mean = mean
        if isinstance(var, numbers.Number):
            self.var = max(int(sample_asym(var)), 1)
        elif isinstance(var, (tuple, list)) and len(var) == 2:
            self.var = int(sample_uniform(var[0], var[1]))
        else:
            raise Exception('degree must be number or list with length 2')

    def __call__(self, img):
        noise = np.random.normal(self.mean, self.var ** 0.5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img


class CVMotionBlur(object):
    def __init__(self, degrees=12, angle=90):
        if isinstance(degrees, numbers.Number):
            self.degree = max(int(sample_asym(degrees)), 1)
        elif isinstance(degrees, (tuple, list)) and len(degrees) == 2:
            self.degree = int(sample_uniform(degrees[0], degrees[1]))
        else:
            raise Exception('degree must be number or list with length 2')
        self.angle = sample_uniform(-angle, angle)

    def __call__(self, img):
        M = cv2.getRotationMatrix2D((self.degree // 2, self.degree // 2), self.angle, 1)
        motion_blur_kernel = np.zeros((self.degree, self.degree))
        motion_blur_kernel[self.degree // 2, :] = 1
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        img = cv2.filter2D(img, -1, motion_blur_kernel)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class CVGeometry(object):
    def __init__(self, degrees=15, translate=(0.3, 0.3), scale=(0.5, 2.),
                 shear=(45, 15), distortion=0.5, p=0.5):
        self.p = p
        type_p = random.random()
        if type_p < 0.33:
            self.transforms = CVRandomRotation(degrees=degrees)
        elif type_p < 0.66:
            self.transforms = CVRandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)
        else:
            self.transforms = CVRandomPerspective(distortion=distortion)

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            return Image.fromarray(self.transforms(img))
        else:
            return img


class CVDeterioration(object):
    def __init__(self, var, degrees, factor, p=0.5):
        self.p = p
        transforms = []
        if var is not None:
            transforms.append(CVGaussianNoise(var=var))
        if degrees is not None:
            transforms.append(CVMotionBlur(degrees=degrees))
        if factor is not None:
            transforms.append(CVRescale(factor=factor))

        random.shuffle(transforms)
        transforms = Compose(transforms)
        self.transforms = transforms

    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            return Image.fromarray(self.transforms(img))
        else:
            return img


class CVColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5):
        self.p = p
        self.transforms = transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                                 saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transforms(img)
        else:
            return img


def apply_CLAHE(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3, (4, 4))
    dst = clahe.apply(gray)

    return dst


def bmp2jpg(bmp_abs_path):
    base_name = os.path.basename(bmp_abs_path)
    save_name = base_name.replace(".bmp", ".jpg")
    cv2_img = cv2.imread(bmp_abs_path)

    return cv2_img, save_name


def dcm2jpg(dcm_path):
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


def cal_mean_std(imageDir):

    img_h, img_w = 64, 64  # 根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []

    if os.path.exists(imageDir + "\\Thumbs.db"):
        os.remove(imageDir + "\\Thumbs.db")
    imgs_path_list = os.listdir(imageDir)

    len_ = len(imgs_path_list)
    i = 0
    for item in imgs_path_list:
        img = cv2.imread(os.path.join(imageDir, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        print(i, '/', len_)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means = means.reverse()
    stdevs = stdevs.reverse()

    return means, stdevs


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


def cv2ImgAddText(img, text, left, top, font_path="simsun.ttc", textColor=(0, 255, 0), textSize=20):
    from PIL import ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageFile

    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(font_path, textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

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


def find_specific_color(img):
    """
    https://stackoverflow.com/questions/42592234/python-opencv-morphologyex-remove-specific-color
    Parameters
    ----------
    img

    Returns
    -------

    """
    # lower = np.array([10, 10, 120])  # -- Lower range --
    # upper = np.array([60, 60, 245])  # -- Upper range --
    lower = np.array([0, 0, 100])  # -- Lower range --
    upper = np.array([80, 80, 255])  # -- Upper range --
    mask = cv2.inRange(img, lower, upper)
    # mask = 255 - mask
    res = cv2.bitwise_and(img, img, mask=mask)  # -- Contains pixels having the gray color--

    return res


def remove_specific_color(image):
    H, W, _ = image.shape
    newimg = image.copy()

    for i in range(H):
        for j in range(W):
            if image[:, :, 0][i, j] < 200 and image[:, :, 1][i, j] < 200 and image[:, :, 2][i, j] > 200:
            # if abs(image[:, :, 0][i, j] - image[:, :, 1][i, j]) < 20 and abs(image[:, :, 2][i, j] - image[:, :, 0][i, j]) > 50:
                newimg[:, :, 0][i, j] = 0
                newimg[:, :, 1][i, j] = 0
                newimg[:, :, 2][i, j] = 0

    return newimg


def remove_specific_color_v2(cv2img):
    hsv = cv2.cvtColor(cv2img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    thresh1 = cv2.threshold(s, 92, 255, cv2.THRESH_BINARY)[1]
    thresh2 = cv2.threshold(v, 10, 255, cv2.THRESH_BINARY)[1]
    thresh2 = 255 - thresh2
    mask = cv2.add(thresh1, thresh2)

    H, W, _ = cv2img.shape
    newimg = cv2img.copy()

    for i in range(H):
        for j in range(W):
            if mask[i, j] != 0:
                newimg[i, j] = cv2img[i - 12, j - 12]

    return newimg


def resize_images(imgPath, savePath, size_=(256, 32)):
    imgList = sorted(os.listdir(imgPath))
    for i in range(len(imgList)):
        imgAbsPath = imgPath + "\\" + imgList[i]
        img_name = os.path.basename(imgAbsPath).split(".")[0]
        ends = os.path.splitext(imgList[i])[1]
        cv2_img = cv2.imread(imgAbsPath)
        resized_img = cv2.resize(cv2_img, size_)
        cv2.imwrite(savePath + "\\{}{}".format(img_name, ends), resized_img)

    print("Resized!")


def retrieve_image_from_video(video_path, frame_save_path):
    video_name = os.path.basename(video_path).split(".")[0]

    if not os.path.exists(frame_save_path):
        os.makedirs(frame_save_path)

    cap = cv2.VideoCapture(video_path)
    count = 0
    while(True):
        success, frame = cap.read()
        if success:
            cv2.imwrite(frame_save_path + "/{}_{}.jpg".format(video_name, count), frame)
        elif not success:
            break
        count += 1
        print("Processing...")


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


def makeBorderH(image, H, W, W_need=128, H_need=32):
    """
    Horizental
    :param image:
    :param H:
    :param W:
    :param W_need:
    :param H_need:
    :return:
    """
    top_size, bottom_size, left_size, right_size = 0, 0, 0, 0
    if W < W_need:
        lr_pixel_need = W_need - W
        left_size = lr_pixel_need // 2
        right_size = lr_pixel_need - left_size
    if H < H_need:
        tb_pixel_need = H_need - H
        top_size = tb_pixel_need // 2
        bottom_size = tb_pixel_need - top_size

    if top_size != 0 or bottom_size != 0 or left_size != 0 or right_size != 0:
        replicate = cv2.copyMakeBorder(image, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
        replicateResized = cv2.resize(replicate, (W_need, H_need))
        return replicateResized
    else:
        resized = cv2.resize(image, (W_need, H_need))
        return resized


def makeBorderV(image, H, W, W_need=32, H_need=128):
    """
    Vertical
    :param image:
    :param H:
    :param W:
    :param W_need:
    :param H_need:
    :return:
    """
    top_size, bottom_size, left_size, right_size = 0, 0, 0, 0
    if W < W_need:
        lr_pixel_need = W_need - W
        left_size = lr_pixel_need // 2
        right_size = lr_pixel_need - left_size
    if H < H_need:
        tb_pixel_need = H_need - H
        top_size = tb_pixel_need // 2
        bottom_size = tb_pixel_need - top_size

    if top_size != 0 or bottom_size != 0 or left_size != 0 or right_size != 0:
        replicate = cv2.copyMakeBorder(image, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
        replicateResized = cv2.resize(replicate, (W_need, H_need))
        return replicateResized
    else:
        resized = cv2.resize(image, (W_need, H_need))
        return resized


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
        cv2img = cv2.imread(img_abs_path)
        RES = identify_colors(cv2img, COLORS, THRESHOLD=60, N_COLORS=5, SIZE=(32, 16))
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


def rotate_image_90(img_path):
    dir_name = os.path.basename(img_path)
    save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/{}_rotated".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    img_list = sorted(os.listdir(img_path))
    for img in img_list:
        img_abs_path = img_path + "/{}".format(img)
        cv2img = cv2.imread(img_abs_path)
        img90 = np.rot90(cv2img, 1)
        cv2.imwrite("{}/{}".format(save_path, img), img90)


def seg_object_from_mask(base_path):
    img_path = base_path + "/images"
    mask_path = base_path + "/masks"

    save_path = base_path + "/output"
    os.makedirs(save_path, exist_ok=True)

    img_list = sorted(os.listdir(img_path))
    for img in img_list:
        img_abs_path = img_path + "/{}".format(img)
        mask_abs_path = mask_path + "/{}".format(img.replace(".jpg", ".png"))

        cv2img = cv2.imread(img_abs_path)
        maskimg = cv2.imread(mask_abs_path)
        zeros = np.zeros(shape=cv2img.shape)

        object_area = np.where((maskimg[:, :, 0] != 0) & (maskimg[:, :, 1] != 0) & (maskimg[:, :, 2] != 0))
        x, y = object_area[1], object_area[0]
        for i in range(len(x)):
            zeros[y[i], x[i], :] = cv2img[y[i], x[i], :]

        cv2.imwrite("{}/{}".format(save_path, img), zeros)


def select_h_or_v_images(src_path1, src_path2):
    img_list1 = os.listdir(src_path1)
    img_list2 = os.listdir(src_path2)

    save_path1_h = src_path1 + "/h"
    save_path1_v = src_path1 + "/v"
    save_path2_h = src_path2 + "/h"
    save_path2_v = src_path2 + "/v"
    os.makedirs(save_path1_h, exist_ok=True)
    os.makedirs(save_path1_v, exist_ok=True)
    os.makedirs(save_path2_h, exist_ok=True)
    os.makedirs(save_path2_v, exist_ok=True)

    for i in img_list1:
        img_abs_path = src_path1 + "/{}".format(i)
        cv2img = cv2.imread(img_abs_path)
        h, w = cv2img.shape[:2]
        if w >= h:
            shutil.move(img_abs_path, save_path1_h)
        else:
            shutil.move(img_abs_path, save_path1_v)

    for i in img_list2:
        img_abs_path = src_path2 + "/{}".format(i)
        cv2img = cv2.imread(img_abs_path)
        h, w = cv2img.shape[:2]
        if w >= h:
            shutil.move(img_abs_path, save_path2_h)
        else:
            shutil.move(img_abs_path, save_path2_v)


def cal_images_mean_h_w(img_path):
    img_list = sorted(img_path)

    hs, ws = [], []

    for img in img_list:
        img_abs_path = img_path + "/{}".format(img)
        cv2img = cv2.imread(img_abs_path)
        h, w = cv2img.shape[:2]
        hs.append(h)
        ws.append(w)

    h_mean = np.mean(hs)
    w_mean = np.mean(ws)

    print(h_mean)  # 511.35578569681155
    print(w_mean)  # 478.03767430481935

    return h_mean, w_mean
# ========================================= Color Identification =========================================
# ========================================================================================================================================================================
# ========================================================================================================================================================================





# ========================================================================================================================================================================
# ========================================================================================================================================================================
# DET
class YOLOv5_ONNX(object):
    """
    onnx_path = "/home/zengyifan/wujiahu/data/003.Cigar_Detection/weights/smoke/best.onnx"
    img_path = "/home/zengyifan/wujiahu/data/003.Cigar_Detection/test/test_20221206/1/20221115_baidudisk_00000007.jpg"

    model = YOLOv5_ONNX(onnx_path)
    model_input_size = (384, 384)
    img0, img, src_size = model.pre_process(img_path, img_size=model_input_size)
    print("src_size: ", src_size)
    pred = model.inference(img)
    out_bbx = model.post_process(pred, src_size, img_size=model_input_size)
    print("out_bbx: ", out_bbx)
    for b in out_bbx:
        cv2.rectangle(img0, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
    cv2.imshow("test", img0)
    cv2.waitKey(0)
    """
    def __init__(self, onnx_path):
        cuda = torch.cuda.is_available()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.input_names = self.session.get_inputs()[0].name
        self.output_names = self.session.get_outputs()[0].name

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_area(self, box):
        # box = xyxy(4,n)
        return (box[2] - box[0]) * (box[3] - box[1])

    def box_iou(self, box1, box2, eps=1e-7):
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

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / (self.box_area(box1.T)[:, None] + self.box_area(box2.T) - inter + eps)

    def non_max_suppression(self, prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.3 + 0.03 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        # output = [torch.zeros((0, 6), device=prediction.device)] * bs
        output = [torch.zeros((0, 6))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                # v = torch.zeros((len(lb), nc + 5), device=x.device)
                v = torch.zeros((len(lb), nc + 5), device=x)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                # conf, j = x[:, 5:].max(1, keepdim=True)
                conf, j = torch.tensor(x[:, 5:]).float().max(1, keepdim=True)
                # x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                x = torch.cat((torch.tensor(box), conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                # x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
                x = x[(x[:, 5:6] == torch.tensor(classes)).any(1)]

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
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                # x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            # if (time.time() - t) > time_limit:
            #     LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            #     break  # time limit exceeded

        return output

    def clip_coords(self, boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def pre_process(self, img_path, img_size=(640, 640), stride=32):
        img0 = cv2.imread(img_path)
        src_size = img0.shape[:2]
        img = self.letterbox(img0, img_size, stride=stride, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img0, img, src_size

    def inference(self, img):
        # im = img.cpu().numpy()  # torch to numpy
        pred = self.session.run([self.output_names], {self.input_names: img})[0]
        return pred

    def post_process(self, pred, src_size, img_size):
        output = self.non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, agnostic=False)
        out_bbx = []
        for i, det in enumerate(output):  # detections per image
            if len(det):
                det[:, :4] = self.scale_coords(img_size, det[:, :4], src_size).round()
                for *xyxy, conf, cls in reversed(det):
                    x1y1x2y2_VOC = [int(round(ci)) for ci in torch.tensor(xyxy).view(1, 4).view(-1).tolist()]
                    out_bbx.append(x1y1x2y2_VOC)

        return out_bbx


class YOLOv8_ONNX(object):
    def __init__(self, onnx_path):
        cuda = torch.cuda.is_available()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.input_names = self.session.get_inputs()[0].name
        self.output_names = self.session.get_outputs()[0].name

        # output_names = [x.name for x in self.session.get_outputs()]
        # metadata = self.session.get_modelmeta().custom_metadata_map  # metadata

        # Load external metadata YAML
        # if isinstance(metadata, (str, Path)) and Path(metadata).exists():
        #     metadata = yaml_load(metadata)
        # if metadata:
        #     for k, v in metadata.items():
        #         if k in ('stride', 'batch'):
        #             metadata[k] = int(v)
        #         elif k in ('imgsz', 'names') and isinstance(v, str):
        #             metadata[k] = eval(v)
        #     stride = metadata['stride']
        #     task = metadata['task']
        #     batch = metadata['batch']
        #     imgsz = metadata['imgsz']
        #     names = metadata['names']

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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

    def xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
        Returns:
            y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def clip_boxes(self, boxes, shape):
        """
        It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
        shape

        Args:
          boxes (torch.Tensor): the bounding boxes to clip
          shape (tuple): the shape of the image
        """
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        """
        Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
        (img1_shape) to the shape of a different image (img0_shape).

        Args:
          img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
          boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
          img0_shape (tuple): the shape of the target image, in the format of (height, width).
          ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                             calculated based on the size difference between the two images.

        Returns:
          boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes

    def box_iou(self, box1, box2, eps=1e-7):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
            eps

        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300,
                            nc=0,  # number of classes (optional)
                            max_time_img=0.05,
                            max_nms=30000,
                            max_wh=7680,
        ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Arguments:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        prediction = torch.Tensor(prediction)

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + max_time_img * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x.transpose(0, -1)[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)
            box = self.xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            if multi_label:
                i, j = (cls > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

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
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            # if mps:
            #     output[xi] = output[xi].to(device)
            # if (time.time() - t) > time_limit:
            #     LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            #     break  # time limit exceeded

        return output

    def pre_process(self, img_path, img_size=(640, 640), stride=32):
        # img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        # img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        # img /= 255  # 0 - 255 to 0.0 - 1.0
        # return img

        img0 = cv2.imread(img_path)
        src_size = img0.shape[:2]
        img = self.letterbox(img0, img_size, stride=stride, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img0, img, src_size

    def inference(self, img):
        # im = im.cpu().numpy()  # torch to numpy
        pred = self.session.run([self.output_names], {self.input_names: img})[0]
        return pred

    def post_process(self, preds, src_size, img_size, conf_thres=0.25, iou_thres=0.45):
        preds = self.non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        agnostic=False,
                                        max_det=300,
                                        classes=None)

        # results = []
        # for i, pred in enumerate(preds):
        #     # orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        #     if not isinstance(orig_imgs, torch.Tensor):
        #         pred[:, :4] = self.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        #     path, _, _, _, _ = self.batch
        #     img_path = path[i] if isinstance(path, list) else path
        #     results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        # return results
        out_bbx = []
        for i, det in enumerate(preds):  # detections per image
            if len(det):
                det[:, :4] = self.scale_boxes(img_size, det[:, :4], src_size).round()
                for *xyxy, conf, cls in reversed(det):
                    x1y1x2y2_VOC = [int(round(ci)) for ci in torch.tensor(xyxy).view(1, 4).view(-1).tolist()]
                    out_bbx.append(x1y1x2y2_VOC)

        return out_bbx


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


def seamlessclone_mixed_output(bg_img, obj_img):
    bh, bw = bg_img.shape[:2]
    oh, ow = obj_img.shape[:2]
    if oh > bh and ow > bw:
        obj_img = cv2.resize(obj_img, (bw, bh))

        mask = 255 * np.ones(obj_img.shape, obj_img.dtype)
        center = (int(round(bw / 2)), int(round(bh / 2)))
        output_mixed = cv2.seamlessClone(obj_img, bg_img, mask, center, cv2.MIXED_CLONE)  # cv::NORMAL_CLONE, cv::MIXED_CLONE or cv::MONOCHROME_TRANSFER

        # gen obj bbx
        VOC_bbxes, thresh = gen_VOC_bbx_from_image(obj_img)  # [xmin, xmax, ymin, ymax]

        # gen new bbx
        yolo_bbxes_new = []
        for lbl in VOC_bbxes:
            # lbl_yolo_new = convert_bbx_VOC_to_yolo((bh, bw), lbl)
            lbl_np = np.array([lbl])
            lbl_np = lbl_np[:, [0, 2, 1, 3]]
            lbl_list = list(lbl_np[0])
            lbl_yolo_new = convertBboxVOC2YOLO((bh, bw), lbl_list)
            yolo_bbxes_new.append(lbl_yolo_new)

        return output_mixed, yolo_bbxes_new, VOC_bbxes, thresh

    else:
        if (oh > bh and ow <= bw) or (oh <= bh and ow > bw):
            obj_img = cv2.resize(obj_img, (int(round(ow / 2)), bh), int(round(oh / 2)))

        mask = 255 * np.ones(obj_img.shape, obj_img.dtype)
        center = (int(round(bw / 2)), int(round(bh / 2)))
        output_mixed = cv2.seamlessClone(obj_img, bg_img, mask, center, cv2.MIXED_CLONE)

        # gen obj bbx
        VOC_bbxes, thresh = gen_VOC_bbx_from_image(obj_img)  # [xmin, xmax, ymin, ymax]
        # gen new bbx
        yolo_bbxes_new = []
        for lbl in VOC_bbxes:
            # lbl_VOC_new = [int(round(bw / 2 - ow / 2 + lbl[0])), int(round(bw / 2 - ow / 2 + lbl[0] + (lbl[1] - lbl[0]))),
            #                int(round(bh / 2 - oh / 2 + lbl[2])), int(round(bh / 2 - oh / 2 + lbl[2] + (lbl[3] - lbl[2])))]
            # lbl_yolo_new = convert_bbx_VOC_to_yolo((bh, bw), lbl_VOC_new)
            lbl_VOC_new = [int(round(bw / 2 - ow / 2 + lbl[0])), int(round(bh / 2 - oh / 2 + lbl[2])),
                           int(round(bw / 2 - ow / 2 + lbl[0] + (lbl[1] - lbl[0]))), int(round(bh / 2 - oh / 2 + lbl[2] + (lbl[3] - lbl[2])))]
            lbl_yolo_new = convertBboxVOC2YOLO((bh, bw), lbl_VOC_new)
            yolo_bbxes_new.append(lbl_yolo_new)

        return output_mixed, yolo_bbxes_new, VOC_bbxes, thresh


def aug_img_with_seamless_clone(bg_path, obj_path, save_path, aug_n=10, label=0, fname_add_content=""):
    bg_list = sorted(os.listdir(bg_path))
    obj_list = sorted(os.listdir(obj_path))

    # save_path = os.path.abspath(os.path.join())
    # save_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/seamless_clone_aug"
    save_img_path = save_path + "/images"
    save_lbl_path = save_path + "/labels"
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_lbl_path, exist_ok=True)

    for bg in bg_list:
        bg_name = os.path.splitext(bg)[0]
        bg_abs_path = bg_path + "/{}".format(bg)
        rdm_obj = random.sample(obj_list, aug_n)
        for i, obj in enumerate(rdm_obj):
            try:
                obj_name = os.path.splitext(obj)[0]
                obj_abs_path = obj_path + "/{}".format(obj)
                bg_cv2img = cv2.imread(bg_abs_path)
                obj_cv2img = cv2.imread(obj_abs_path)

                output_mixed, yolo_bbxes_new, VOC_bbxes, thresh = seamlessclone_mixed_output(bg_cv2img, obj_cv2img)
                new_name = "{}_{}_{}_MIXED_CLONE".format(bg_name, obj_name, fname_add_content)

                #
                # cv2.rectangle(obj_img, (VOC_bbxes[0][0], VOC_bbxes[0][2]), (VOC_bbxes[0][1], VOC_bbxes[0][3]), (255, 0, 0), 2)
                # cv2.imwrite("{}/{}_{}_thresh.jpg".format(save_img_path, new_name, i), thresh)
                # cv2.imwrite("{}/{}_{}_rect.jpg".format(save_img_path, new_name, i), obj_img)
                #

                # save img & lbl
                cv2.imwrite("{}/{}_{}.jpg".format(save_img_path, new_name, i), output_mixed)
                save_lbl_abs_path = "{}/{}_{}.txt".format(save_lbl_path, new_name, i)
                with open(save_lbl_abs_path, "w", encoding="utf-8") as fw:
                    for bb in yolo_bbxes_new:
                        txt_content = "{}".format(label) + " " + " ".join([str(b) for b in bb]) + "\n"
                        fw.write(txt_content)
            except Exception as Error:
                print(Error)


def aug_img_with_seamless_clone_main(bg_list, obj_list, bg_path, obj_path, save_path, aug_n=10, label=0, fname_add_content=""):
    save_img_path = save_path + "/images"
    save_lbl_path = save_path + "/labels"
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_lbl_path, exist_ok=True)

    for bg in tqdm(bg_list):
        bg_name = os.path.splitext(bg)[0]
        bg_abs_path = bg_path + "/{}".format(bg)
        rdm_obj = random.sample(obj_list, aug_n)
        for i, obj in enumerate(rdm_obj):
            try:
                obj_name = os.path.splitext(obj)[0]
                obj_abs_path = obj_path + "/{}".format(obj)
                bg_cv2img = cv2.imread(bg_abs_path)
                obj_cv2img = cv2.imread(obj_abs_path)

                output_mixed, yolo_bbxes_new, VOC_bbxes, thresh = seamlessclone_mixed_output(bg_cv2img, obj_cv2img)
                new_name = "{}_{}_{}_Seamless_Clone".format(bg_name, obj_name, fname_add_content)

                #
                # cv2.rectangle(obj_img, (VOC_bbxes[0][0], VOC_bbxes[0][2]), (VOC_bbxes[0][1], VOC_bbxes[0][3]), (255, 0, 0), 2)
                # cv2.imwrite("{}/{}_{}_thresh.jpg".format(save_img_path, new_name, i), thresh)
                # cv2.imwrite("{}/{}_{}_rect.jpg".format(save_img_path, new_name, i), obj_img)
                #

                # save img & lbl
                cv2.imwrite("{}/{}_{}.jpg".format(save_img_path, new_name, i), output_mixed)
                save_lbl_abs_path = "{}/{}_{}.txt".format(save_lbl_path, new_name, i)
                with open(save_lbl_abs_path, "w", encoding="utf-8") as fw:
                    for bb in yolo_bbxes_new:
                        txt_content = "{}".format(label) + " " + " ".join([str(b) for b in bb]) + "\n"
                        fw.write(txt_content)
            except Exception as Error:
                print(Error)


def aug_img_with_seamless_clone_multi_thread(bg_path, obj_path, save_path, aug_n=3, label=0, fname_add_content=""):
    bg_list = sorted(os.listdir(bg_path))
    obj_list = sorted(os.listdir(obj_path))

    len_ = len(bg_list)
    bg_lists = []
    split_n = 8
    for j in range(split_n):
        bg_lists.append(bg_list[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

    t_list = []
    for i in range(split_n):
        bg_list_i = bg_lists[i]
        t = threading.Thread(target=aug_img_with_seamless_clone_main, args=(bg_list_i, obj_list, bg_path, obj_path, save_path, aug_n, label, fname_add_content, ))
        t_list.append(t)

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


def gen_random_pos(paste_num, bg_size, obj_size, dis_thresh=50, scatter_num=5):
    paste_poses = []
    last_pos = (0, 0)  # try to scatter the bbxs.
    for ii in range(scatter_num):
        for k in range(paste_num):
            # paste_pos_k = [np.random.randint(0, bg_size[1]), np.random.randint(0, bg_size[0]), obj_size[1], obj_size[0]]
            paste_pos_k = [np.random.randint(0, (bg_size[1] - obj_size[1])), np.random.randint(0, (bg_size[0] - obj_size[1])) + obj_size[1],
                           np.random.randint(0, (bg_size[1] - obj_size[0])), np.random.randint(0, (bg_size[1] - obj_size[0])) + obj_size[0]]
            if last_pos != (0, 0):
                if np.sqrt((paste_pos_k[0] - last_pos[0]) ** 2 + (paste_pos_k[2] - last_pos[2]) ** 2) < dis_thresh:
                    continue
                else:
                    paste_poses.append(paste_pos_k)
            last_pos = paste_pos_k

    return paste_poses


def pil_paste_img(bg_img, obj_img, paste_num=1):
    bh, bw = bg_img.shape[:2]
    oh, ow = obj_img.shape[:2]
    if oh > bh and ow > bw:
        obj_img = cv2.resize(obj_img, (bw, bh))

        pil_bg_img = PIL_paste_image_on_bg(obj_img, bg_img, [[0, 0]])
        array_bg_img = np.asarray(pil_bg_img)

        # gen obj bbx
        VOC_bbxes = gen_VOC_bbx_from_image(obj_img)  # [xmin, xmax, ymin, ymax]
        # gen new bbx
        yolo_bbxes_new = []
        for lbl in VOC_bbxes:
            # lbl_yolo_new = convert_bbx_VOC_to_yolo((bh, bw), lbl)
            lbl_np = np.array([lbl])
            lbl_np = lbl_np[:, [0, 2, 1, 3]]
            lbl_list = list(lbl_np[0])
            lbl_yolo_new = convertBboxVOC2YOLO((bh, bw), lbl_list)
            yolo_bbxes_new.append(lbl_yolo_new)

        return array_bg_img, yolo_bbxes_new


    else:
        if (oh > bh and ow <= bw) or (oh <= bh and ow > bw):
            obj_img = cv2.resize(obj_img, (int(round(ow / 2)), bh), int(round(oh / 2)))

        oh, ow = obj_img.shape[:2]

        paste_poses = gen_random_pos(paste_num, (bh, bw), (oh, ow), dis_thresh=50, scatter_num=5)
        paste_poses_selected = random.sample(paste_poses, paste_num)
        pil_bg_img = PIL_paste_image_on_bg(obj_img, bg_img, paste_poses_selected)
        array_bg_img = np.asarray(pil_bg_img)

        # gen obj bbx
        VOC_bbxes = gen_VOC_bbx_from_image(obj_img)  # [xmin, xmax, ymin, ymax]
        # gen new bbx
        yolo_bbxes_new = []
        for i, lbl in enumerate(VOC_bbxes):
            # lbl_VOC_new = [int(round(lbl[0] + paste_poses_selected[i][0])), int(round(lbl[1] + paste_poses_selected[i][0])),
            #                int(round(lbl[2] + paste_poses_selected[i][1])), int(round(lbl[3] + paste_poses_selected[i][1]))]
            # lbl_yolo_new = convert_bbx_VOC_to_yolo((bh, bw), lbl_VOC_new)
            lbl_VOC_new = [int(round(lbl[0] + paste_poses_selected[i][0])), int(round(lbl[2] + paste_poses_selected[i][1])),
                           int(round(lbl[1] + paste_poses_selected[i][0])), int(round(lbl[3] + paste_poses_selected[i][1]))]
            lbl_yolo_new = convertBboxVOC2YOLO((bh, bw), lbl_VOC_new)
            yolo_bbxes_new.append(lbl_yolo_new)

        return array_bg_img, yolo_bbxes_new


def PIL_paste_image_on_bg(paste_imgs, bg_img, paste_poses_selected):
    pil_bg_img = Image.fromarray(np.uint8(bg_img)).convert("RGBA")

    for i, pos in enumerate(paste_poses_selected):
        pil_img = Image.fromarray(np.uint8(paste_imgs)).convert("RGBA")
        pil_img_alpha = pil_img.split()[-1]
        pil_bg_img.paste(pil_img, (pos[0], pos[1]), mask=pil_img_alpha)

    pil_bg_img = pil_bg_img.convert("RGB")
    return pil_bg_img


def aug_img_with_pil_image_paste(bg_path, obj_path, save_path, aug_n=10):
    bg_list = sorted(os.listdir(bg_path))
    obj_list = sorted(os.listdir(obj_path))

    # save_path = os.path.abspath(os.path.join())
    # save_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/seamless_clone_aug"
    save_img_path = save_path + "/images"
    save_lbl_path = save_path + "/labels"
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_lbl_path, exist_ok=True)

    for bg in bg_list:
        bg_name = os.path.splitext(bg)[0]
        bg_abs_path = bg_path + "/{}".format(bg)
        rdm_obj = random.sample(obj_list, aug_n)
        for i, obj in enumerate(rdm_obj):
            try:
                obj_name = os.path.splitext(obj)[0]
                obj_abs_path = obj_path + "/{}".format(obj)
                bg_cv2img = cv2.imread(bg_abs_path)
                obj_cv2img = cv2.imread(obj_abs_path)

                pasted_out, yolo_bbxes_new = pil_paste_img(bg_cv2img, obj_cv2img)

                new_name = "{}_{}_pil_image_paste".format(bg_name, obj_name)

                # save img & lbl
                cv2.imwrite("{}/{}_{}.jpg".format(save_img_path, new_name, i), pasted_out)
                save_lbl_abs_path = "{}/{}_{}.txt".format(save_lbl_path, new_name, i)
                with open(save_lbl_abs_path, "w", encoding="utf-8") as fw:
                    for bb in yolo_bbxes_new:
                        txt_content = "{}".format(0) + " " + " ".join([str(b) for b in bb]) + "\n"
                        fw.write(txt_content)

            except Exception as Error:
                print(Error)


def draw_label(size=(384, 384, 3), polygon_list=None):
    image = np.zeros(size, np.uint8)
    img_vis = cv2.fillPoly(image, polygon_list, (128, 128, 128))
    img_vis = Image.fromarray(img_vis)
    img = cv2.fillPoly(image, polygon_list, (1, 1, 1))
    img = Image.fromarray(img)

    return img_vis, img


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


def labels_to_split_images(n_labels, labels, size):
    res_arr = []
    for i in range(1, n_labels):
        # res_arr.append(np.zeros(size, dtype=np.int32))
        img_to_write = np.zeros(size, dtype=np.int32)
        target = np.where(labels[:, :] == i)
        img_to_write[target] = i
        res_arr.append(img_to_write)

    return res_arr


def image_gen_yolo_bbx(res_arr, size):
    bboxes = []
    for img in res_arr:
        cnts, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # bboxes.append((x, y, w, h))

            x_min = x
            x_max = x + w
            y_min = y
            y_max = y + h

            # bb = convert_bbx_VOC_to_yolo((size[0], size[1]), (x_min, x_max, y_min, y_max))
            bb = convertBboxVOC2YOLO((size[0], size[1]), (x_min, y_min, x_max, y_max))
            bboxes.append(bb)

    return bboxes


def image_gen_VOC_bbx(res_arr):
    bboxes = []
    for img in res_arr:
        cnts, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)

            if w < 10 and h < 10:
                continue

            x_min = x
            x_max = x + w
            y_min = y
            y_max = y + h

            bboxes.append([x_min, x_max, y_min, y_max])

    return bboxes


def get_img_json_path_and_size(img_path, json_path, base_name):
    if os.path.exists(img_path + "/{}.jpg".format(base_name)):
        img_abs_path = img_path + "/{}.jpg".format(base_name)
        json_abs_path = json_path + "/{}.jpg.json".format(base_name)
        img_ = cv2.imread(img_abs_path)
        size = img_.shape[:2]
        return img_abs_path, json_abs_path, size
    elif os.path.exists(img_path + "/{}.png".format(base_name)):
        img_abs_path = img_path + "/{}.png".format(base_name)
        json_abs_path = json_path + "/{}.png.json".format(base_name)
        img_ = cv2.imread(img_abs_path)
        size = img_.shape[:2]
        return img_abs_path, json_abs_path.replace(".png.json", ".jpg.json"), size
    elif os.path.exists(img_path + "/{}.jpeg".format(base_name)):
        img_abs_path = img_path + "/{}.jpeg".format(base_name)
        json_abs_path = json_path + "/{}.jpeg.json".format(base_name)
        img_ = cv2.imread(img_abs_path)
        size = img_.shape[:2]
        return img_abs_path, json_abs_path, size
    elif os.path.exists(img_path + "/{}.bmp".format(base_name)):
        img_abs_path = img_path + "/{}.bmp".format(base_name)
        json_abs_path = json_path + "/{}.bmp.json".format(base_name)
        img_ = cv2.imread(img_abs_path)
        size = img_.shape[:2]
        return img_abs_path, json_abs_path, size
    elif os.path.exists(img_path + "/{}.JPG".format(base_name)):
        img_abs_path = img_path + "/{}.JPG".format(base_name)
        json_abs_path = json_path + "/{}.JPG.json".format(base_name)
        img_ = cv2.imread(img_abs_path)
        size = img_.shape[:2]
        return img_abs_path, json_abs_path, size
    elif os.path.exists(img_path + "/{}.PNG".format(base_name)):
        img_abs_path = img_path + "/{}.PNG".format(base_name)
        json_abs_path = json_path + "/{}.PNG.json".format(base_name)
        img_ = cv2.imread(img_abs_path)
        size = img_.shape[:2]
        return img_abs_path, json_abs_path, size
    else:
        print("Error!")
        return None, None, None


def gen_yolo_txt_from_mask_image(img_path):
    txt_save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/masks_vis_generated_labels"
    os.makedirs(txt_save_path, exist_ok=True)
    img_list = os.listdir(img_path)

    for img in img_list:
        img_name = img.strip("_vis.png")
        img_abs_path = img_path + "/{}".format(img)

        cv2img = cv2.imread(img_abs_path)
        h, w = cv2img.shape[:2]
        size = (h, w)
        gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        txt_save_path_ = txt_save_path + "/{}.txt".format(img_name)
        with open(txt_save_path_, "w", encoding="utf-8") as fw:
            res_arr = labels_to_split_images(n_labels, labels, size)
            bboxes = image_gen_yolo_bbx(res_arr, size)
            for b in bboxes:
                txt_content = "0" + " " + " ".join([str(a) for a in b]) + "\n"
                fw.write(txt_content)


def gen_VOC_bbx_from_image(img):
    # imgsz = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    # res_arr = labels_to_split_images(n_labels, labels, imgsz)

    # thresh_filtered = cv2.medianBlur(thresh, 3)
    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # dilated = cv2.dilate(thresh, k)

    bboxes = []  # bboxes: [x_min, x_max, y_min, y_max]
    cnts, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_cnts = max(cnts, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(max_cnts)
    x_min = x
    x_max = x + w
    y_min = y
    y_max = y + h

    bboxes.append([x_min, x_max, y_min, y_max])

    # for c in cnts:
    #     x, y, w, h = cv2.boundingRect(c)
    #
    #     if w < 10 and h < 10:
    #         continue
    #
    #     x_min = x
    #     x_max = x + w
    #     y_min = y
    #     y_max = y + h
    #
    #     bboxes.append([x_min, x_max, y_min, y_max])

    return bboxes, thresh


def copy_or_move_specific_image_and_label_by_name(base_path, key_words, copy_or_move="copy"):
    """

    :param base_path:
    :param key_words: ["skating", "Skating"]
    :param copy_or_move:
    :return:
    """
    img_path = base_path + "/images"
    lbl_path = base_path + "/labels"

    save_path = os.path.abspath(os.path.join(img_path, "../..")) + "/selected_specific_data"
    save_img_path = save_path + "/images"
    save_lbl_path = save_path + "/labels"
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_lbl_path, exist_ok=True)

    img_list = os.listdir(img_path)
    lbl_list = os.listdir(lbl_path)

    for img in img_list:
        img_name = os.path.splitext(img)[0]
        if key_words[0] in img or key_words[1] in img:
            img_abs_path = img_path + "/{}".format(img)
            img_dst_path = save_img_path + "/{}".format(img)

            lbl_abs_path = lbl_path + "/{}.txt".format(img_name)
            lbl_dst_path = save_lbl_path + "/{}.txt".format(img_name)

            if copy_or_move == "copy":
                shutil.copy(img_abs_path, img_dst_path)
                shutil.copy(lbl_abs_path, lbl_dst_path)
            elif copy_or_move == "move":
                shutil.move(img_abs_path, img_dst_path)
                shutil.move(lbl_abs_path, lbl_dst_path)
            else:
                print("Error! copy_or_move")


def crop_img_expand_n_times_v2(img, bbx, size, n=1.0):
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
        if x1 - expand_x >= 0 and y1 - expand_y >= 0 and x2 + expand_x <= size[1] and y2 + expand_y <= size[0]:
            x1_new = x1 - expand_x
            y1_new = y1 - expand_y
            x2_new = x2 + expand_x
            y2_new = y2 + expand_y
        elif x1 - expand_x_half >= 0 and y1 - expand_y_half >= 0 and x2 + expand_x_half <= size[1] and y2 + expand_y_half <= size[0]:
            x1_new = x1 - expand_x_half
            y1_new = y1 - expand_y_half
            x2_new = x2 + expand_x_half
            y2_new = y2 + expand_y_half
        else:
            x1_new = x1
            y1_new = y1
            x2_new = x2
            y2_new = y2

        cropped = img[y1_new:y2_new, x1_new:x2_new]
        return cropped


def crop_img_expand_n_times_v3(img, bbx, size, n=1.0):
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


def crop_image_according_labelbee_json(data_path, crop_ratio=(1, 1.5, 2, 2.5, 3)):
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
        cv2img = cv2.imread(img_abs_path)
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
                        cropped_img = crop_img_expand_n_times_v2(cv2img, [x_min, y_min, x_max, y_max], [h, w], nx)
                        cropped_nx_path = cropped_path + "/{}".format(nx)
                        os.makedirs(cropped_nx_path, exist_ok=True)
                        cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, i, nx), cropped_img)
                    except Exception as Error:
                        print(Error)
                        # cropped_img = crop_img_expand_n_times_v2(cv2img, [x_min, y_min, x_max, y_max], [h, w], 1)
                        # cropped_nx_path = cropped_path + "/{}".format(nx)
                        # os.makedirs(cropped_nx_path, exist_ok=True)
                        # cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, i, nx), cropped_img)

                # bb = convert_bbx_VOC_to_yolo((h, w), (x_min, x_max, y_min, y_max))
                bb = convertBboxVOC2YOLO((h, w), (x_min, y_min, x_max, y_max))
                txt_content = "0" + " " + " ".join([str(b) for b in bb]) + "\n"
                fw.write(txt_content)


def crop_image_according_yolo_txt(data_path, CLS=(1, 2), crop_ratio=(1, 1.5, 2, 2.5, 3)):
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
            cv2img = cv2.imread(img_abs_path)
            if cv2img is None: continue
            h, w = cv2img.shape[:2]

            txt_o = open(txt_abs_path, "r", encoding="utf-8")
            lines = txt_o.readlines()
            txt_o.close()

            for i, l in enumerate(lines):
                l_s = l.strip().split(" ")
                cls = int(l_s[0])
                if cls in CLS:
                    bbx_yolo = list(map(float, l_s[1:]))
                    # bbx_voc = convert_bbx_yolo_to_VOC([h, w], bbx_yolo)
                    bbx_voc = convertBboxYOLO2VOC([h, w], bbx_yolo)

                    # crop_ratio_rdm = np.random.randint(20, 31)
                    # crop_ratio_ = [crop_ratio_rdm * 0.1]
                    for nx in crop_ratio:
                        try:
                            cropped_img = crop_img_expand_n_times_v3(cv2img, bbx_voc, [h, w], nx)
                            cropped_nx_path = cropped_path + "/{}/{}".format(cls, nx)
                            os.makedirs(cropped_nx_path, exist_ok=True)
                            cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, i, nx), cropped_img)
                        except Exception as Error:
                            print(Error)
        except Exception as Error:
            print(Error)


def remove_special_yolo_label(data_path):
    moved_img_path = data_path + "_moved/images"
    moved_lbl_path = data_path + "_moved/labels"
    os.makedirs(moved_img_path, exist_ok=True)
    os.makedirs(moved_lbl_path, exist_ok=True)

    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"

    lbl_list = sorted(os.listdir(lbl_path))
    for lbl in lbl_list:
        lbl_abs_path = lbl_path + "/{}".format(lbl)
        with open(lbl_abs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) > 2:
                lbl_dst_path = moved_lbl_path + '/{}'.format(lbl)
                img_abs_path = img_path + "/{}.jpg".format(lbl.split(".")[0])
                img_dst_path = moved_img_path + "/{}.jpg".format(lbl.split(".")[0])
                shutil.move(img_abs_path, img_dst_path)
                shutil.move(lbl_abs_path, lbl_dst_path)


def select_images_according_C_Plus_Plus_det_cls_output(txt_path, save_path_flag="current", save_path="", save_no_det_res_img=False, n_classes=3, crop_expand_ratio=1.5):
    if save_path_flag == "current":
        save_base_path = os.path.abspath(os.path.join(txt_path, "../.."))
    else:
        save_base_path = save_path

    dataset_name = os.path.basename(txt_path).split("_list_res")[0]

    save_path_ = {}
    for i in range(n_classes):
        save_path_src_i = "save_path_src_{}".format(i)
        save_path_vis_i = "save_path_vis_{}".format(i)
        save_path_crop_i = "save_path_crop_{}".format(i)
        save_path_crop_src = "save_path_crop_src".format(i)

        save_path_[save_path_src_i] = save_base_path + "/C_Plus_Plus_det_output/{}/src_images/{}".format(dataset_name, i)
        save_path_[save_path_vis_i] = save_base_path + "/C_Plus_Plus_det_output/{}/vis_images/{}".format(dataset_name, i)
        save_path_[save_path_crop_i] = save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/{}/{}".format(dataset_name, i, crop_expand_ratio)
        save_path_[save_path_crop_src] = save_base_path + "/C_Plus_Plus_det_output/{}/crop_images_src".format(dataset_name)
        os.makedirs(save_path_[save_path_src_i], exist_ok=True)
        os.makedirs(save_path_[save_path_vis_i], exist_ok=True)
        os.makedirs(save_path_[save_path_crop_i], exist_ok=True)
        os.makedirs(save_path_[save_path_crop_src], exist_ok=True)

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

            if len(ff[1:]) != 0:
                shutil.copy(fpath, "{}/{}".format(save_path_["save_path_crop_src"], fname))

            res = list(map(float, ff[1:]))
            np_res = np.asarray(res).reshape(-1, 7)

            cv2img = cv2.imread(fpath)
            cv2img_cp = cv2img.copy()
            h, w = cv2img.shape[:2]

            sum_ = 0
            for i in range(len(np_res)):
                x1y1x2y2_VOC = [int(np_res[i][0]), int(np_res[i][1]), int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])]
                cropped_img = crop_img_expand_n_times_v2(cv2img_cp, x1y1x2y2_VOC, [h, w], crop_expand_ratio)

                pred_label = int(np_res[i][4])
                cls_np_res = int(np_res[i][6])
                for j in range(n_classes):
                    if j == cls_np_res:
                        cv2.imwrite("{}/{}_{}_{}.jpg".format(save_path_["save_path_crop_{}".format(j)], fname.split(".")[0], i, crop_expand_ratio), cropped_img)

                        if cls_np_res == 0:
                            cv2.rectangle(cv2img, (int(np_res[i][0]), int(np_res[i][1])), (int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])), (0, 0, 255), 2)
                            cv2.putText(cv2img, "{}: {}".format(int(np_res[i][6]), np_res[i][5]), (int(np_res[i][0]), int(np_res[i][1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        elif cls_np_res == 1:
                            cv2.rectangle(cv2img, (int(np_res[i][0]), int(np_res[i][1])), (int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])), (0, 255, 0), 2)
                            cv2.putText(cv2img, "{}: {}".format(int(np_res[i][6]), np_res[i][5]), (int(np_res[i][0]), int(np_res[i][1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                        else:
                            print(cls_np_res)

                # if pred_label == 0 and cls_np_res == 1:  # pred_label == 0: cigar, cls_np_res == 1: true cigar
                #     sum_ += 1
                if pred_label == 43 and cls_np_res == 1:  # pred_label == 0: cigar, cls_np_res == 1: true cigar
                    sum_ += 1

            # if sum_ == 0:
            #     cv2.imwrite("{}/{}".format(save_path_["save_path_vis_0"], fname), cv2img)
            #     shutil.copy(fpath, "{}/{}".format(save_path_["save_path_src_0"], fname))
            # else:
            #     cv2.imwrite("{}/{}".format(save_path_["save_path_vis_1"], fname), cv2img)
            #     shutil.copy(fpath, "{}/{}".format(save_path_["save_path_src_1"], fname))
            if sum_ > 0:
                cv2.imwrite("{}/{}".format(save_path_["save_path_vis_0"], fname), cv2img)
                shutil.copy(fpath, "{}/{}".format(save_path_["save_path_src_0"], fname))


def select_images_according_C_Plus_Plus_det_cls_output_two_classes(txt_path, save_path_flag="current", save_path="", save_no_det_res_img=False, save_crop_img=True, save_src_img=False, save_vis_img=False, crop_expand_ratio=1.5):
    if save_path_flag == "current":
        save_base_path = os.path.abspath(os.path.join(txt_path, "../.."))
    else:
        save_base_path = save_path

    dataset_name = os.path.basename(txt_path).split("_list_res")[0]

    if save_src_img:
        save_path_cls_0_src_0 = save_base_path + "/C_Plus_Plus_det_output/{}/src_images/cls_0/0".format(dataset_name)
        save_path_cls_0_src_1 = save_base_path + "/C_Plus_Plus_det_output/{}/src_images/cls_0/1".format(dataset_name)
        save_path_cls_1_src_0 = save_base_path + "/C_Plus_Plus_det_output/{}/src_images/cls_1/0".format(dataset_name)
        save_path_cls_1_src_1 = save_base_path + "/C_Plus_Plus_det_output/{}/src_images/cls_1/1".format(dataset_name)
        os.makedirs(save_path_cls_0_src_0, exist_ok=True)
        os.makedirs(save_path_cls_0_src_1, exist_ok=True)
        os.makedirs(save_path_cls_1_src_0, exist_ok=True)
        os.makedirs(save_path_cls_1_src_1, exist_ok=True)

    if save_vis_img:
        save_path_cls_0_vis_0 = save_base_path + "/C_Plus_Plus_det_output/{}/vis_images/cls_0/0".format(dataset_name)
        save_path_cls_0_vis_1 = save_base_path + "/C_Plus_Plus_det_output/{}/vis_images/cls_0/1".format(dataset_name)
        save_path_cls_1_vis_0 = save_base_path + "/C_Plus_Plus_det_output/{}/vis_images/cls_1/0".format(dataset_name)
        save_path_cls_1_vis_1 = save_base_path + "/C_Plus_Plus_det_output/{}/vis_images/cls_1/1".format(dataset_name)
        os.makedirs(save_path_cls_0_vis_0, exist_ok=True)
        os.makedirs(save_path_cls_0_vis_1, exist_ok=True)
        os.makedirs(save_path_cls_1_vis_0, exist_ok=True)
        os.makedirs(save_path_cls_1_vis_1, exist_ok=True)

    if save_crop_img:
        save_path_cls_0_crop_0 = save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/cls_0/0/{}".format(dataset_name, crop_expand_ratio)
        save_path_cls_0_crop_1 = save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/cls_0/1/{}".format(dataset_name, crop_expand_ratio)
        save_path_cls_1_crop_0 = save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/cls_1/0/{}".format(dataset_name, crop_expand_ratio)
        save_path_cls_1_crop_1 = save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/cls_1/1/{}".format(dataset_name, crop_expand_ratio)
        os.makedirs(save_path_cls_0_crop_0, exist_ok=True)
        os.makedirs(save_path_cls_0_crop_1, exist_ok=True)
        os.makedirs(save_path_cls_1_crop_0, exist_ok=True)
        os.makedirs(save_path_cls_1_crop_1, exist_ok=True)

    if save_no_det_res_img:
        save_path_no_det_res = save_base_path + "/C_Plus_Plus_det_output/{}/no_det_res".format(dataset_name)
        os.makedirs(save_path_no_det_res, exist_ok=True)

    with open(txt_path, "r", encoding="utf-8") as fo:
        lines = fo.readlines()
        for l in tqdm(lines):
            ff = l.strip().split(" ")
            fpath = ff[0]
            fname = os.path.basename(fpath)
            if not os.path.exists(fpath): continue

            if len(ff) <= 1:
                if save_no_det_res_img:
                    shutil.copy(fpath, "{}/{}".format(save_path_no_det_res, fname))
                continue

            res = list(map(float, ff[1:]))
            np_res = np.asarray(res).reshape(-1, 7)

            cv2img = cv2.imread(fpath)
            cv2img_cp = cv2img.copy()
            h, w = cv2img.shape[:2]

            label_0_sum_ = 0
            label_1_sum_ = 0
            label_0_flag = False
            label_1_flag = False
            for i in range(len(np_res)):
                pred_label = int(np_res[i][4])
                if pred_label == 0:
                    label_0_flag = True
                    x1y1x2y2_VOC = [int(np_res[i][0]), int(np_res[i][1]), int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])]
                    cropped_img = crop_img_expand_n_times_v2(cv2img_cp, x1y1x2y2_VOC, [h, w], crop_expand_ratio)

                    cls_np_res = int(np_res[i][6])
                    if cls_np_res == 0:
                        if save_crop_img:
                            cv2.imwrite("{}/{}_{}_{}.jpg".format(save_path_cls_0_crop_0, fname.split(".")[0], i, crop_expand_ratio), cropped_img)
                        if save_vis_img:
                            cv2.rectangle(cv2img, (int(np_res[i][0]), int(np_res[i][1])), (int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])), (0, 0, 255), 2)
                            cv2.putText(cv2img, "{}: {}".format(int(np_res[i][4]), np_res[i][5]), (int(np_res[i][0]), int(np_res[i][1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                    else:
                        if save_crop_img:
                            cv2.imwrite("{}/{}_{}_{}.jpg".format(save_path_cls_0_crop_1, fname.split(".")[0], i, crop_expand_ratio), cropped_img)
                        if save_vis_img:
                            cv2.rectangle(cv2img, (int(np_res[i][0]), int(np_res[i][1])), (int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])), (0, 255, 0), 2)
                            cv2.putText(cv2img, "{}: {}".format(int(np_res[i][4]), np_res[i][5]), (int(np_res[i][0]), int(np_res[i][1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                        label_0_sum_ += 1

                elif pred_label == 1:
                    label_1_flag = True
                    x1y1x2y2_VOC = [int(np_res[i][0]), int(np_res[i][1]), int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])]
                    cropped_img = crop_img_expand_n_times_v2(cv2img_cp, x1y1x2y2_VOC, [h, w], crop_expand_ratio)

                    cls_np_res = int(np_res[i][6])
                    if cls_np_res == 0:
                        if save_crop_img:
                            cv2.imwrite("{}/{}_{}_{}.jpg".format(save_path_cls_1_crop_0, fname.split(".")[0], i, crop_expand_ratio), cropped_img)
                        if save_vis_img:
                            cv2.rectangle(cv2img, (int(np_res[i][0]), int(np_res[i][1])), (int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])), (0, 0, 255), 2)
                            cv2.putText(cv2img, "{}: {}".format(int(np_res[i][4]), np_res[i][5]), (int(np_res[i][0]), int(np_res[i][1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    else:
                        if save_crop_img:
                            cv2.imwrite("{}/{}_{}_{}.jpg".format(save_path_cls_1_crop_1, fname.split(".")[0], i, crop_expand_ratio), cropped_img)
                        if save_vis_img:
                            cv2.rectangle(cv2img, (int(np_res[i][0]), int(np_res[i][1])), (int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])), (0, 255, 0), 2)
                            cv2.putText(cv2img, "{}: {}".format(int(np_res[i][4]), np_res[i][5]), (int(np_res[i][0]), int(np_res[i][1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                        label_1_sum_ += 1

            if label_0_flag and not label_1_flag:
                if label_0_sum_ == 0:
                    if save_vis_img:
                        cv2.imwrite("{}/{}".format(save_path_cls_0_vis_0, fname), cv2img)
                    if save_src_img:
                        shutil.copy(fpath, "{}/{}".format(save_path_cls_0_src_0, fname))
                else:
                    if save_vis_img:
                        cv2.imwrite("{}/{}".format(save_path_cls_0_vis_1, fname), cv2img)
                    if save_src_img:
                        shutil.copy(fpath, "{}/{}".format(save_path_cls_0_src_1, fname))
            elif not label_0_flag and label_1_flag:
                if label_1_sum_ == 0:
                    if save_vis_img:
                        cv2.imwrite("{}/{}".format(save_path_cls_1_vis_0, fname), cv2img)
                    if save_src_img:
                        shutil.copy(fpath, "{}/{}".format(save_path_cls_1_src_0, fname))
                else:
                    if save_vis_img:
                        cv2.imwrite("{}/{}".format(save_path_cls_1_vis_1, fname), cv2img)
                    if save_src_img:
                        shutil.copy(fpath, "{}/{}".format(save_path_cls_1_src_1, fname))
            elif label_0_flag and label_1_flag:
                if label_0_sum_ == 0:
                    if save_vis_img:
                        cv2.imwrite("{}/{}".format(save_path_cls_0_vis_0, fname), cv2img)
                    if save_src_img:
                        shutil.copy(fpath, "{}/{}".format(save_path_cls_0_src_0, fname))
                else:
                    if save_vis_img:
                        cv2.imwrite("{}/{}".format(save_path_cls_0_vis_1, fname), cv2img)
                    if save_src_img:
                        shutil.copy(fpath, "{}/{}".format(save_path_cls_0_src_1, fname))
                if label_1_sum_ == 0:
                    if save_vis_img:
                        cv2.imwrite("{}/{}".format(save_path_cls_1_vis_0, fname), cv2img)
                    if save_src_img:
                        shutil.copy(fpath, "{}/{}".format(save_path_cls_1_src_0, fname))
                else:
                    if save_vis_img:
                        cv2.imwrite("{}/{}".format(save_path_cls_1_vis_1, fname), cv2img)
                    if save_src_img:
                        shutil.copy(fpath, "{}/{}".format(save_path_cls_1_src_1, fname))
            else:
                print("Error!")


def select_images_according_C_Plus_Plus_det_cls_output_n_classes(txt_path, save_path_flag="current", save_path="", save_no_det_res_img=False, save_crop_img=True, save_src_img=False, save_vis_img=False, crop_expand_ratio=1.5, n_cls=4):
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

            cv2img = cv2.imread(fpath)
            cv2img_cp = cv2img.copy()
            h, w = cv2img.shape[:2]

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
                        cropped_img = crop_img_expand_n_times_v2(cv2img_cp, x1y1x2y2_VOC, [h, w], crop_expand_ratio)

                        cls_np_res = int(np_res[i][6])
                        if cls_np_res == 0:
                            if save_crop_img:
                                cv2.imwrite("{}/{}_{}_{}.jpg".format(save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/cls_{}/0/{}".format(dataset_name, n, crop_expand_ratio), fname.split(".")[0], i, crop_expand_ratio), cropped_img)
                            if save_vis_img:
                                cv2.rectangle(cv2img, (int(np_res[i][0]), int(np_res[i][1])), (int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])), (0, 0, 255), 2)
                                cv2.putText(cv2img, "{}: {}".format(int(np_res[i][4]), np_res[i][5]), (int(np_res[i][0]), int(np_res[i][1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                        else:
                            if save_crop_img:
                                cv2.imwrite("{}/{}_{}_{}.jpg".format(save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/cls_{}/1/{}".format(dataset_name, n, crop_expand_ratio), fname.split(".")[0], i, crop_expand_ratio), cropped_img)
                            if save_vis_img:
                                cv2.rectangle(cv2img, (int(np_res[i][0]), int(np_res[i][1])), (int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])), (0, 255, 0), 2)
                                cv2.putText(cv2img, "{}: {}".format(int(np_res[i][4]), np_res[i][5]), (int(np_res[i][0]), int(np_res[i][1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                            label_i_sum_["label_{}_sum_".format(n)] += 1

            for n in range(n_cls):
                if label_i_flag["label_{}_flag".format(n)]:
                    if label_i_sum_["label_{}_sum_".format(n)] == 0:
                        if save_vis_img:
                            cv2.imwrite("{}/{}".format(save_base_path + "/C_Plus_Plus_det_output/{}/vis_images/cls_{}/0".format(dataset_name, n), fname), cv2img)
                        if save_src_img:
                            shutil.copy(fpath, "{}/{}".format(save_base_path + "/C_Plus_Plus_det_output/{}/src_images/cls_{}/0".format(dataset_name, n), fname))
                    else:
                        if save_vis_img:
                            cv2.imwrite("{}/{}".format(save_base_path + "/C_Plus_Plus_det_output/{}/vis_images/cls_{}/1".format(dataset_name, n), fname), cv2img)
                        if save_src_img:
                            shutil.copy(fpath, "{}/{}".format(save_base_path + "/C_Plus_Plus_det_output/{}/src_images/cls_{}/1".format(dataset_name, n), fname))


def select_images_according_C_Plus_Plus_det_output(txt_path, save_path_flag="current", save_path="", save_no_det_res_img=False, crop_expand_ratio=1.0):
    if save_path_flag == "current":
        save_base_path = os.path.abspath(os.path.join(txt_path, "../.."))
    else:
        save_base_path = save_path

    dataset_name = os.path.basename(txt_path).split("_list_res")[0]
    save_path_src = save_base_path + "/C_Plus_Plus_det_output/{}/src_images".format(dataset_name)
    save_path_vis = save_base_path + "/C_Plus_Plus_det_output/{}/vis_images".format(dataset_name)
    save_path_crop = save_base_path + "/C_Plus_Plus_det_output/{}/crop_images/{}".format(dataset_name, crop_expand_ratio)
    os.makedirs(save_path_src, exist_ok=True)
    os.makedirs(save_path_vis, exist_ok=True)
    os.makedirs(save_path_crop, exist_ok=True)
    if save_no_det_res_img:
        save_path_no_det_res = save_base_path + "/C_Plus_Plus_det_output/{}/no_det_res".format(dataset_name)
        os.makedirs(save_path_no_det_res, exist_ok=True)

    # classes = ["roller_skating", "board_skating"]
    # classes = ["knife"]
    classes = ["smoke", "fire"]

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
            np_res = np.asarray(res).reshape(-1, 6)

            cv2img = cv2.imread(fpath)
            cv2img_cp = cv2img.copy()
            h, w = cv2img.shape[:2]

            for i in range(len(np_res)):
                x1y1x2y2_VOC = [int(np_res[i][0]), int(np_res[i][1]), int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])]
                cropped_img = crop_img_expand_n_times_v2(cv2img_cp, x1y1x2y2_VOC, [h, w], crop_expand_ratio)
                cv2.imwrite("{}/{}_{}_{}.jpg".format(save_path_crop, fname.split(".")[0], i, crop_expand_ratio), cropped_img)
                cv2.rectangle(cv2img, (int(np_res[i][0]), int(np_res[i][1])), (int(np_res[i][0] + np_res[i][2]), int(np_res[i][1] + np_res[i][3])), (0, 255, 0), 2)
                cv2.putText(cv2img, "{}: {}".format(classes[int(np_res[i][4])], np_res[i][5]), (int(np_res[i][0]), int(np_res[i][1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

            cv2.imwrite("{}/{}".format(save_path_vis, fname), cv2img)
            shutil.copy(fpath, "{}/{}".format(save_path_src, fname))


def select_images_according_python_det_output():
    selected_dir = "person"
    # coco_data_path = "/home/zengyifan/wujiahu/data/003.Cigar_Detection/others/person".format(selected_dir)
    # labels_path = "/home/zengyifan/wujiahu/yolo/yolov5-6.2/runs/detect/smoke_detect/labels".format(selected_dir)
    # save_path = "/home/zengyifan/wujiahu/data/003.Cigar_Detection/others/cropped_person_smoking_det_data/20221119".format(selected_dir)
    coco_data_path = "/home/zengyifan/wujiahu/yolo/yolov5-6.2/runs/detect/person"
    labels_path = "/home/zengyifan/wujiahu/yolo/yolov5-6.2/runs/detect/smoking_person_detect/labels"
    save_path = "/home/zengyifan/wujiahu/data/003.Cigar_Detection/others/cropped_person_smoking_det_data/20221121"
    os.makedirs(save_path, exist_ok=True)

    new_images_path = save_path + "/images"
    new_labels_path = save_path + "/labels"
    # os.makedirs(save_path, exist_ok=True)
    os.makedirs(new_images_path, exist_ok=True)

    labels_list = os.listdir(labels_path)
    for img in labels_list:
        # img = img.split("_")[-1]
        try:
            img_src_path = coco_data_path + "/{}.jpg".format(img.split(".")[0])
            img_dst_path = new_images_path + "/{}.jpg".format(img.split(".")[0])
            shutil.copy(img_src_path, img_dst_path)
            # shutil.move(img_src_path, img_dst_path)
            # print("{} --> {}".format(img_src_path, img_dst_path))
        except Exception as Error:
            print(Error)

    shutil.copytree(labels_path, new_labels_path)


def select_images_and_write_yolo_label_according_C_Plus_Plus_det_cls_output(img_path, txt_path):
    img_list = sorted(os.listdir(img_path))

    save_base_path_0 = os.path.abspath(os.path.join(img_path, "../.."))
    save_base_path_1 = os.path.abspath(os.path.join(save_base_path_0, "../.."))
    save_base_path_2 = os.path.abspath(os.path.join(save_base_path_1, "../.."))
    save_img_path = save_base_path_2 + "/selected_images_and_labels/images"
    save_lbl_path = save_base_path_2 + "/selected_images_and_labels/labels"
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_lbl_path, exist_ok=True)

    with open(txt_path, "r", encoding="utf-8") as fo:
        lines = fo.readlines()
        for l in tqdm(lines):
            for img_name in img_list:
                ll = l.strip().split(" ")
                if len(ll) <= 1: continue
                ll_path = ll[0]
                ll_np_res = np.array(ll[1:]).reshape(-1, 7)
                if img_name == os.path.basename(ll_path):
                    img_dst_path = save_img_path + "/{}".format(img_name)
                    shutil.copy(ll_path, img_dst_path)

                    cv2img = cv2.imread(ll_path)
                    imgsz = cv2img.shape[:2]
                    lbl_dst_path = save_lbl_path + "/{}.txt".format(os.path.splitext(img_name)[0])
                    with open(lbl_dst_path, "w", encoding="utf-8") as fw:
                        for i in range(ll_np_res.shape[0]):
                            bbx = list(map(float, ll_np_res[i][:4]))
                            lbl = ll_np_res[i][4]
                            xmin = bbx[0]
                            xmax = bbx[0] + bbx[2]
                            ymin = bbx[1]
                            ymax = bbx[1] + bbx[3]

                            # yolo_bbx = convert_bbx_VOC_to_yolo(imgsz, (xmin, xmax, ymin, ymax))
                            yolo_bbx = convertBboxVOC2YOLO(imgsz, (xmin, ymin, xmax, ymax))
                            txt_content = "{}".format(lbl) + " " + " ".join([str(b) for b in yolo_bbx]) + "\n"
                            fw.write(txt_content)


def select_images_and_write_yolo_label_according_C_Plus_Plus_det_output(img_path, txt_path):
    img_list = sorted(os.listdir(img_path))

    save_base_path_0 = os.path.abspath(os.path.join(img_path, "../.."))
    # save_base_path_1 = os.path.abspath(os.path.join(save_base_path_0, ".."))
    # save_base_path_2 = os.path.abspath(os.path.join(save_base_path_1, ".."))
    save_img_path = save_base_path_0 + "/selected_images_and_labels/images"
    save_lbl_path = save_base_path_0 + "/selected_images_and_labels/labels"
    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_lbl_path, exist_ok=True)

    with open(txt_path, "r", encoding="utf-8") as fo:
        lines = fo.readlines()
        for l in tqdm(lines):
            for img_name in img_list:
                ll = l.strip().split(" ")
                if len(ll) <= 1: continue
                ll_path = ll[0]
                ll_np_res = np.array(ll[1:]).reshape(-1, 6)
                if img_name == os.path.basename(ll_path):
                    img_dst_path = save_img_path + "/{}".format(img_name)
                    shutil.copy(ll_path, img_dst_path)

                    cv2img = cv2.imread(ll_path)
                    imgsz = cv2img.shape[:2]
                    lbl_dst_path = save_lbl_path + "/{}.txt".format(os.path.splitext(img_name)[0])
                    with open(lbl_dst_path, "w", encoding="utf-8") as fw:
                        for i in range(ll_np_res.shape[0]):
                            bbx = list(map(float, ll_np_res[i][:4]))
                            lbl = ll_np_res[i][4]
                            xmin = bbx[0]
                            xmax = bbx[0] + bbx[2]
                            ymin = bbx[1]
                            ymax = bbx[1] + bbx[3]

                            # yolo_bbx = convert_bbx_VOC_to_yolo(imgsz, (xmin, xmax, ymin, ymax))
                            yolo_bbx = convertBboxVOC2YOLO(imgsz, (xmin, ymin, xmax, ymax))
                            txt_content = "{}".format(lbl) + " " + " ".join([str(b) for b in yolo_bbx]) + "\n"
                            fw.write(txt_content)


class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.25, max_rotation_angle=13,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.35,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5,
                 is_addNoise=True, is_changeLight=True, is_cutout=False, is_rotate_img_bbox=True,
                 is_crop_img_bboxes=True, is_shift_pic_bboxes=True, is_filp_pic_bboxes=True):

        # 配置各个操作的属性
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

        # 是否使用某种增强方式
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_cutout = is_cutout
        self.is_rotate_img_bbox = is_rotate_img_bbox
        self.is_crop_img_bboxes = is_crop_img_bboxes
        self.is_shift_pic_bboxes = is_shift_pic_bboxes
        self.is_filp_pic_bboxes = is_filp_pic_bboxes

    # 加噪声
    def _addNoise(self, img):
        from skimage.util import random_noise
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # random.seed(int(time.time()))
        return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True) * 255
        # return random_noise(img, mode='gaussian', clip=True)

    # 调整亮度
    def _changeLight(self, img):
        flag = random.uniform(0.6, 1.3)
        blank = np.zeros(img.shape, img.dtype)
        alpha = beta = flag
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)

    # cutout
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''

        def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            '''
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            # return the intersection over union value
            return iou

        # 得到h和w
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            _, h, w, c = img.shape
        mask = np.ones((h, w, c), np.float32)
        for n in range(n_holes):
            chongdie = True  # 看切割的区域是否与box重叠太多
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0,
                             h)  # numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        chongdie = True
                        break

            mask[y1: y2, x1: x2, :] = 0.

        # mask = np.expand_dims(mask, axis=0)
        img = img * mask

        return img

    # 旋转
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        # ---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_img, rot_bboxes

    def _rotate_img_bbox_v2(self, img, bboxes, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        # ---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            # point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            # point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            # point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            # point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # # 合并np.array
            # concat = np.vstack((point1, point2, point3, point4))
            # # 改变array类型
            # concat = concat.astype(np.int32)
            # # 得到旋转后的坐标
            # rx, ry, rw, rh = cv2.boundingRect(concat)
            # rx_min = rx
            # ry_min = ry
            # rx_max = rx + rw
            # ry_max = ry + rh
            # # 加入list中
            # rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

            p1 = np.array([xmin, ymin])
            p2 = np.array([xmax, ymin])
            p3 = np.array([xmax, ymax])
            p4 = np.array([xmin, ymax])
            pts = np.vstack([p1, p2, p3, p4])
            pts = np.float32(pts)
            pts = np.hstack([pts, np.ones([len(pts), 1])]).T
            rotated_box = np.dot(rot_mat, pts)
            rotated_box = [[rotated_box[0][x], rotated_box[1][x]] for x in range(len(rotated_box[0]))]
            rotated_xmin = int(round(np.min(np.array(rotated_box)[:, 0])))
            rotated_ymin = int(round(np.min(np.array(rotated_box)[:, 1])))
            rotated_xmax = int(round(np.max(np.array(rotated_box)[:, 0])))
            rotated_ymax = int(round(np.max(np.array(rotated_box)[:, 1])))
            rot_bboxes.append([rotated_xmin, rotated_ymin, rotated_xmax, rotated_ymax])

        return rot_img, rot_bboxes

    # 裁剪
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        # ---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes

    # 平移
    def _shift_pic_bboxes(self, img, bboxes):
        '''
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

        return shift_img, shift_bboxes

    # 镜像
    def _filp_pic_bboxes(self, img, bboxes):
        '''
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 翻转图像 ----------------------

        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:  # 0.5的概率水平翻转，0.5的概率垂直翻转
            horizon = True
        else:
            horizon = False
        h, w, _ = img.shape
        if horizon:  # 水平翻转
            flip_img = cv2.flip(flip_img, 1)  # 1是水平，-1是水平垂直
        else:
            flip_img = cv2.flip(flip_img, 0)

        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
            else:
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])

        return flip_img, flip_bboxes

    # 图像增强方法
    def dataAugment(self, img, bboxes):
        """
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        :param img:
        :param bboxes:
        :return:
        """

        change_num = 0  # 改变的次数
        while change_num < 1:  # 默认至少有一种数据增强生效

            if self.is_rotate_img_bbox:
                if random.random() > self.rotation_rate:  # 旋转
                    # print('旋转')
                    change_num += 1
                    # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                    angle = random.uniform(0, self.max_rotation_angle)
                    scale = random.uniform(0.7, 0.8)
                    img, bboxes = self._rotate_img_bbox_v2(img, bboxes, angle, scale)

            if self.is_shift_pic_bboxes:
                if random.random() < self.shift_rate:  # 平移
                    change_num += 1
                    img, bboxes = self._shift_pic_bboxes(img, bboxes)

            if self.is_changeLight:
                if random.random() > self.change_light_rate:  # 改变亮度
                    change_num += 1
                    img = self._changeLight(img)

            if self.is_addNoise:
                if random.random() < self.add_noise_rate:  # 加噪声
                    change_num += 1
                    img = self._addNoise(img)
            if self.is_cutout:
                if random.random() < self.cutout_rate:  # cutout
                    print('cutout')
                    change_num += 1
                    img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                       threshold=self.cut_out_threshold)
            if self.is_filp_pic_bboxes:
                if random.random() < self.flip_rate:  # 翻转
                    change_num += 1
                    img, bboxes = self._filp_pic_bboxes(img, bboxes)

        return img, bboxes


class ToolHelper():
    # def parse_xml(self, path):
    #     import xml.etree.ElementTree as ET
    #     '''
    #     输入：
    #         xml_path: xml的文件路径
    #     输出：
    #         从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    #     '''
    #     tree = ET.parse(path)
    #     root = tree.getroot()
    #     objs = root.findall('object')
    #     coords = list()
    #     for ix, obj in enumerate(objs):
    #         name = obj.find('name').text
    #         box = obj.find('bndbox')
    #         x_min = int(box[0].text)
    #         y_min = int(box[1].text)
    #         x_max = int(box[2].text)
    #         y_max = int(box[3].text)
    #         coords.append([x_min, y_min, x_max, y_max, name])
    #     return coords

    def parse_txt(self, lbl_path, img_size):
        txt_fo = open(lbl_path, "r", encoding="utf-8")
        txt_data = txt_fo.readlines()
        coords = list()

        for lbl in txt_data:
            lbl_ = lbl.strip().split(" ")
            cls = lbl_[0]
            b = list(map(float, lbl_[1:]))
            # bbx_VOC = convert_bbx_yolo_to_VOC(img_size, b)
            bbx_VOC = convertBboxYOLO2VOC(img_size, b)
            bbx_VOC.append(cls)
            coords.append(bbx_VOC)

        return coords

    # 保存图片结果
    def save_img(self, img_save_path, img):
        cv2.imwrite(img_save_path, img)

    # # 保持xml结果
    # def save_xml(self, file_name, save_folder, img_info, height, width, channel, bboxs_info):
    #     from lxml import etree, objectify
    #     '''
    #     :param file_name:文件名
    #     :param save_folder:#保存的xml文件的结果
    #     :param height:图片的信息
    #     :param width:图片的宽度
    #     :param channel:通道
    #     :return:
    #     '''
    #     folder_name, img_name = img_info  # 得到图片的信息
    #
    #     E = objectify.ElementMaker(annotate=False)
    #
    #     anno_tree = E.annotation(
    #         E.folder(folder_name),
    #         E.filename(img_name),
    #         E.path(os.path.join(folder_name, img_name)),
    #         E.source(
    #             E.database('Unknown'),
    #         ),
    #         E.size(
    #             E.width(width),
    #             E.height(height),
    #             E.depth(channel)
    #         ),
    #         E.segmented(0),
    #     )
    #
    #     labels, bboxs = bboxs_info  # 得到边框和标签信息
    #     for label, box in zip(labels, bboxs):
    #         anno_tree.append(
    #             E.object(
    #                 E.name(label),
    #                 E.pose('Unspecified'),
    #                 E.truncated('0'),
    #                 E.difficult('0'),
    #                 E.bndbox(
    #                     E.xmin(box[0]),
    #                     E.ymin(box[1]),
    #                     E.xmax(box[2]),
    #                     E.ymax(box[3])
    #                 )
    #             ))
    #
    #     etree.ElementTree(anno_tree).write(os.path.join(save_folder, file_name), pretty_print=True)

    def save_txt(self, lbl_save_path, auged_img_size, bboxs_info):
        txt_fw = open(lbl_save_path, "w", encoding="utf-8")

        labels, bboxs = bboxs_info
        for label, box in zip(labels, bboxs):
            # yolo_bbx = convert_bbx_VOC_to_yolo(auged_img_size, box)
            box_np = np.array([box])
            box_np = box_np[:, [0, 2, 1, 3]]
            box_list = list(box_np[0])
            yolo_bbx = convertBboxVOC2YOLO(auged_img_size, box_list)
            txt_content = "{}".format(label) + " " + " ".join(str(bb) for bb in yolo_bbx) + "\n"
            txt_fw.write(txt_content)


def det_data_aug_main(data_path, aug_num=10):
    # data_path = "/home/zengyifan/wujiahu/myutils/data/test_aug/20230516"
    data_dir_name = os.path.basename(data_path)
    source_pic_root_path = data_path + "/images"
    source_txt_root_path = data_path + "/labels"

    save_pic_folder = os.path.join(os.path.abspath(os.path.join(data_path, '../..')), '{}_AUG/images'.format(data_dir_name))
    save_txt_folder = os.path.join(os.path.abspath(os.path.join(data_path, '../..')), '{}_AUG/labels'.format(data_dir_name))

    if not os.path.exists(save_pic_folder):
        os.makedirs(save_pic_folder)
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)

    dataAug = DataAugmentForObjectDetection()
    toolhelper = ToolHelper()

    for parent, _, files in os.walk(source_pic_root_path):
        for file in files:
            file_name, file_suffix = os.path.splitext(file)[0], os.path.splitext(file)[1]
            cnt = 0
            pic_path = os.path.join(parent, file)
            img = cv2.imread(pic_path)
            img_size = img.shape[:2]
            txt_path = os.path.join(source_txt_root_path, "{}.txt".format(file_name))
            values = toolhelper.parse_txt(txt_path, img_size)

            if values == []:
                continue

            coords = [v[:4] for v in values]  # 得到框
            labels = [v[-1] for v in values]  # 对象的标签

            while cnt < aug_num:  # 继续增强
                auged_img, auged_bboxes = dataAug.dataAugment(img, coords)
                auged_bboxes_int = np.array(auged_bboxes).astype(np.int32)  # [xmin, ymin, xmax, ymax]
                auged_bboxes_int = auged_bboxes_int[:, [0, 2, 1, 3]]  # [xmin, xmax, ymin, ymax]

                auged_img_shape = auged_img.shape[:2]  # 得到图片的属性
                img_save_path = "{}/{}_{}{}".format(save_pic_folder, file_name, cnt + 1, file_suffix)
                toolhelper.save_img(img_save_path, auged_img)  # 保存增强图片

                lbl_save_path = "{}/{}_{}.txt".format(save_txt_folder, file_name, cnt + 1)
                toolhelper.save_txt(lbl_save_path, auged_img_shape, (labels, auged_bboxes_int))  # 保存xml文件
                cnt += 1  # 继续增强下一张

    print("\n#################### Successful ######################\n")


def gen_random_pos_cropped_object_aug_data_v2(cropped_imgs, random_N, scatter_bbxs_num, bg_size, bg_yolov5_false_positive_labels_path, img_name, dis_thresh):
    """

    :param random_N:
    :param bg_size: (h, w)
    :param bg_yolov5_false_positive_labels_path:
    :param img_name:
    :return:
    """
    try:
        paste_poses = []
        last_pos = (0, 0)  # try to scatter the bbxs.
        for ii in range(scatter_bbxs_num):
            for k in range(random_N):
                cropped_k_size = cropped_imgs[k].shape[:2]
                paste_pos_k = (np.random.randint(0, (bg_size[1] - cropped_k_size[1])), np.random.randint(0, (bg_size[0] - cropped_k_size[0])))

                # yolov5 false positive labels
                bg_labels_path = bg_yolov5_false_positive_labels_path + "/{}.txt".format(img_name)
                with open(bg_labels_path, "r", encoding="utf-8") as lfo:
                    bg_bbx_lines = lfo.readlines()
                    for l in bg_bbx_lines:
                        l = l.strip()
                        l_ = [float(l.split(" ")[1]), float(l.split(" ")[2]), float(l.split(" ")[3]), float(l.split(" ")[4])]
                        # bbx_VOC_format = convert_bbx_yolo_to_VOC(bg_size, l_)
                        bbx_VOC_format = convertBboxYOLO2VOC(bg_size, l_)

                        # if in yolov5 false positive detections bbx, is not our desired results
                        if (paste_pos_k[0] >= bbx_VOC_format[0] and paste_pos_k[0] <= bbx_VOC_format[2]) and (paste_pos_k[1] >= bbx_VOC_format[1] and paste_pos_k[1] <= bbx_VOC_format[3]):
                            continue
                        elif np.sqrt((paste_pos_k[0] - bbx_VOC_format[0]) ** 2 + (paste_pos_k[1] - bbx_VOC_format[1]) ** 2) < dis_thresh:
                            continue
                        elif last_pos != (0, 0):
                            if np.sqrt((paste_pos_k[0] - last_pos[0]) ** 2 + (paste_pos_k[1] - last_pos[1]) ** 2) < dis_thresh:
                                continue
                            else:
                                paste_poses.append(paste_pos_k)
                        else:
                            paste_poses.append(paste_pos_k)

                last_pos = paste_pos_k
        return paste_poses

    except Exception as Error:
        print(Error, Error.__traceback__.tb_lineno)


def paste_on_bg_designated_pos_cropped_object_aug_data_v2(bg_cv2img, bg_size, paste_img, paste_pos):
    """

    :param bg_cv2img:
    :param bg_size: (h, w)
    :param out:
    :param thresh:
    :param bbx:
    :param paste_pos:
    :return:
    """
    try:
        h_bg, w_bg = bg_size[0], bg_size[1]
        h_p, w_p = paste_img.shape[:2]

        added_res_bbx = [paste_pos[0], paste_pos[1], w_p, h_p]

        new_1 = bg_cv2img[0:paste_pos[1], 0:w_bg]
        new_21 = bg_cv2img[paste_pos[1]:paste_pos[1] + h_p, 0:paste_pos[0]]
        new_22 = paste_img
        new_23 = bg_cv2img[paste_pos[1]:paste_pos[1] + h_p, paste_pos[0] + w_p:w_bg]
        new_3 = bg_cv2img[paste_pos[1] + h_p:h_bg, 0:w_bg]

        new_mid = np.hstack((new_21, new_22, new_23))
        pasted = np.vstack((new_1, new_mid, new_3))

        added_res = pasted

        return added_res, added_res_bbx, paste_pos

    except Exception as Error:
        print(Error, Error.__traceback__.tb_lineno)


def draw_rectangle_on_added_res_cropped_object_aug_data(rectangle_flag, added_res, added_res_bbx):
    if rectangle_flag:
        for bb in added_res_bbx:
            cv2.rectangle(added_res, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (225, 225, 0), 2)

    return added_res


def scale_down_bbx(bbx, scale_ratio=0.02):
    scale_h_one_side = bbx[3] * scale_ratio
    scale_w_one_side = bbx[2] * scale_ratio

    x_new = bbx[0] + scale_w_one_side
    y_new = bbx[1] + scale_h_one_side
    w_new = bbx[2] - 2 * scale_w_one_side
    h_new = bbx[3] - 2 * scale_h_one_side
    bbx_new = [x_new, y_new, w_new, h_new]
    return bbx_new


def scale_down_bbx_v2(bbx, scale_ratio=0.5):
    scale_h_one_side = bbx[3] * (scale_ratio / 2)
    scale_w_one_side = bbx[2] * (scale_ratio / 2)

    x_new = bbx[0] + scale_w_one_side
    y_new = bbx[1] + scale_h_one_side
    w_new = bbx[2] - 2 * scale_w_one_side
    h_new = bbx[3] - 2 * scale_h_one_side
    bbx_new = [x_new, y_new, w_new, h_new]
    return bbx_new


def write_yolo_label_cropped_object_aug_data_v2(labels_save_path, added_res_bbx, bg_size, img_name, affine_style, dis_thresh=200, scale_flag=False, scale_type=1, scale_ratio=0.04, cls=0, add_rename_str=""):
    """
    :param labels_save_path:
    :param added_res_bbx:
    :param bg_size: (h_bg, w_bg)
    :param img_name:
    :param affine_style:
    :param i:
    :param dis_thresh:
    :return:
    """

    # 1. remove special bbx
    poses = added_res_bbx
    # for bb_ in added_res_bbx:
    #     poses.append([bb_[0], bb_[1], bb_[2], bb_[3]])

    for bi in range(len(poses) - 2, -1, -1):
        bi_p0 = (poses[bi][0], poses[bi][1])
        for bj in range(len(poses) - 1, bi, -1):
            # 1. bbxes very close, very small distance.
            bj_p0 = (poses[bj][0], poses[bj][1])
            bj_p1 = (poses[bj][0] + poses[bj][2], poses[bj][1])
            bj_p2 = (poses[bj][0] + poses[bj][2], poses[bj][1] + poses[bj][3])
            bj_p3 = (poses[bj][0], poses[bj][1] + poses[bj][3])

            dis_bi_p0_bj_p0 = np.sqrt((bi_p0[0] - bj_p0[0]) ** 2 + (bi_p0[1] - bj_p0[1]) ** 2)
            dis_bi_p0_bj_p1 = np.sqrt((bi_p0[0] - bj_p1[0]) ** 2 + (bi_p0[1] - bj_p1[1]) ** 2)
            dis_bi_p0_bj_p2 = np.sqrt((bi_p0[0] - bj_p2[0]) ** 2 + (bi_p0[1] - bj_p2[1]) ** 2)
            dis_bi_p0_bj_p3 = np.sqrt((bi_p0[0] - bj_p3[0]) ** 2 + (bi_p0[1] - bj_p3[1]) ** 2)

            if dis_bi_p0_bj_p0 < dis_thresh or dis_bi_p0_bj_p1 < dis_thresh or dis_bi_p0_bj_p2 < dis_thresh or dis_bi_p0_bj_p3 < dis_thresh:
                poses.remove(poses[bi])
                print("======================== S1 ========================")
                continue

            # 2. small bbx in big bbx.
            # 2.1 bi contain bj
            if poses[bi][0] < poses[bj][0] and poses[bi][1] < poses[bj][1] and poses[bi][0] + poses[bi][2] > poses[bj][0] + poses[bj][2] and poses[bi][1] + poses[bi][3] > poses[bj][1] + poses[bj][3]:
                poses.remove(poses[bi])
                print("======================== S2.1 ========================")
                continue
            # 2.2 bi in bj
            if poses[bi][0] > poses[bj][0] and poses[bi][1] > poses[bj][1] and poses[bi][0] + poses[bi][2] < poses[bj][0] + poses[bj][2] and poses[bi][1] + poses[bi][3] < poses[bj][1] + poses[bj][3]:
                poses.remove(poses[bi])
                print("======================== S2.2 ========================")
                continue

            # 3. cal iou
            iou = cal_iou(poses[bi], poses[bj])
            if iou > 0.10:
                poses.remove(poses[bi])
                print("======================== S3 ========================")
                continue

    # 2. write bbx
    txt_save_path_added_res = "{}/{}_{}_v5_{}.txt".format(labels_save_path, img_name, affine_style, add_rename_str)
    with open(txt_save_path_added_res, "w", encoding="utf-8") as fw:
        for bb_ in poses:
            if scale_flag:
                if scale_type == 1:
                    bbx_new = scale_down_bbx(bb_, scale_ratio=scale_ratio)
                elif scale_type == 2:
                    bbx_new = scale_down_bbx_v2(bb_, scale_ratio=scale_ratio)
                # bb = convert_bbx_VOC_to_yolo(bg_size, [bbx_new[0], bbx_new[0] + bbx_new[2], bbx_new[1], bbx_new[1] + bbx_new[3]])
                bb = convertBboxVOC2YOLO(bg_size, [bbx_new[0], bbx_new[1], bbx_new[0] + bbx_new[2], bbx_new[1] + bbx_new[3]])
                txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bb]) + "\n"
                fw.write(txt_content)
            else:
                # bb = convert_bbx_VOC_to_yolo(bg_size, [bb_[0], bb_[0] + bb_[2], bb_[1], bb_[1] + bb_[3]])
                bb = convertBboxVOC2YOLO(bg_size, [bb_[0], bb_[1], bb_[0] + bb_[2], bb_[1] + bb_[3]])
                txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bb]) + "\n"
                fw.write(txt_content)


def apply_paste_cropped_object_aug_data(cropped_imgs, random_N, scatter_bbxs_num, bg_size, bg_labels_path, img_name, bg_cv2img, bg_cv2img_cp, save_path, dis_thresh, scale_flag, scale_type, scale_ratio, cls, add_rename_str):
    """

    :param cropped_imgs:
    :param random_N:
    :param scatter_bbxs_num:
    :param bg_size: (h, w)
    :param bg_yolov5_false_positive_labels_path:
    :param img_name:
    :param bg_cv2img:
    :param bg_cv2img_cp:
    :return:
    """
    try:
        aug_type = "paste"
        images_save_path = save_path + "/images"
        labels_save_path = save_path + "/labels"
        os.makedirs(images_save_path, exist_ok=True)
        os.makedirs(labels_save_path, exist_ok=True)

        paste_poses = gen_random_pos_cropped_object_aug_data_v2(cropped_imgs, random_N, scatter_bbxs_num, bg_size, bg_labels_path, img_name, dis_thresh)
        # print(paste_poses)
        paste_pos_final = random.sample(paste_poses, random_N)

        added_res_bbx_final = []
        pasted_pos_final = []
        for p in range(random_N):
            added_res, added_res_bbx, paste_pos = paste_on_bg_designated_pos_cropped_object_aug_data_v2(bg_cv2img, bg_size, cropped_imgs[p], paste_pos_final[p])
            added_res_bbx_final.append(added_res_bbx)
            pasted_pos_final.append(paste_pos)
            added_res = draw_rectangle_on_added_res_cropped_object_aug_data(rectangle_flag=False, added_res=added_res, added_res_bbx=added_res_bbx)
            bg_cv2img = added_res

        assert len(added_res_bbx_final) == len(pasted_pos_final), "len(added_res_bbx_final) != len(pasted_pos_final)"
        flag = True
        for ii in range(len(added_res_bbx_final)):
            if added_res_bbx_final[ii][0] != pasted_pos_final[ii][0] and added_res_bbx_final[ii][1] != pasted_pos_final[ii][1]:
                flag = False
                print("flag == False !!!!!!")

        if flag:
            bbx_n = len(added_res_bbx_final)

            if len(added_res_bbx_final) == random_N:
                for bb_ in added_res_bbx_final:
                    if bb_[2] < 50 and bb_[3] < 50:
                        bbx_n -= 1
                if bbx_n == random_N:
                    cv2.imwrite("{}/{}_{}_v5_{}.jpg".format(images_save_path, img_name, aug_type, add_rename_str), bg_cv2img)
                    write_yolo_label_cropped_object_aug_data_v2(labels_save_path, added_res_bbx_final, bg_size, img_name, aug_type, dis_thresh=dis_thresh, scale_flag=scale_flag, scale_type=scale_type, scale_ratio=scale_ratio, cls=cls, add_rename_str=add_rename_str)

                    bg_cv2img = bg_cv2img_cp

    except Exception as Error:
        print(Error, Error.__traceback__.tb_lineno)


def timeit_paste_cropped_object_aug_data(func):
    def wrapper(bg_list, bg_images_path, bg_labels_path, object_list, object_path, random_N, scatter_bbxs_num, save_path, dis_thresh, scale_flag, scale_type, scale_ratio, cls, add_rename_str):
        t1 = time.time()
        func(bg_list, bg_images_path, bg_labels_path, object_list, object_path, random_N, scatter_bbxs_num, save_path, dis_thresh, scale_flag, scale_type, scale_ratio, cls, add_rename_str)
        t2 = time.time()
        print(t2 - t1)

    return wrapper


@timeit_paste_cropped_object_aug_data
def main_thread_paste_cropped_object_aug_data(bg_list, bg_images_path, bg_labels_path, object_list, object_path, random_N, scatter_bbxs_num, save_path, dis_thresh, scale_flag, scale_type, scale_ratio, cls, add_rename_str):
    for bg in tqdm(bg_list):
        try:
            bg_abs_path = bg_images_path + "/{}".format(bg)
            img_name = os.path.splitext(bg)[0]
            bg_cv2img = cv2.imread(bg_abs_path)
            bg_cv2img_cp = bg_cv2img.copy()
            h_bg, w_bg = bg_cv2img.shape[:2]
            bg_size = (h_bg, w_bg)  # [H, w]

            random_num = np.random.randint(1, random_N + 1)  # paste random (less than random_num(including)) objects
            random_samples = random.sample(object_list, random_num)

            cropped_imgs = []
            for s in random_samples:
                s_abs_path = object_path + "/{}".format(s)
                cropped_cv2img = cv2.imread(s_abs_path)
                cropped_imgs.append(cropped_cv2img)

            apply_paste_cropped_object_aug_data(cropped_imgs, random_num, scatter_bbxs_num, bg_size, bg_labels_path, img_name, bg_cv2img, bg_cv2img_cp, save_path, dis_thresh, scale_flag, scale_type, scale_ratio, cls, add_rename_str)

        except Exception as Error:
            print(Error, Error.__traceback__.tb_lineno)


def paste_cropped_object_for_det_aug_data_train_negative_samples_multi_thread_v5_main(bg_path, bg_images_dir_name, bg_labels_dir_name, cropped_object_path, save_path, random_N=1, scatter_bbxs_num=3, dis_thresh=50, scale_flag=True, scale_type=2, scale_ratio=0.02, cls=0, add_rename_str="lock_20230327"):
    bg_images_path = bg_path + "/{}".format(bg_images_dir_name)
    bg_labels_path = bg_path + "/{}".format(bg_labels_dir_name)

    images_save_path = save_path + "/images"
    labels_save_path = save_path + "/labels"
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(labels_save_path, exist_ok=True)

    cropped_object_list = os.listdir(cropped_object_path)
    bg_list = os.listdir(bg_images_path)

    len_ = len(bg_list)
    bg_lists = []
    split_n = 8
    for j in range(split_n):
        bg_lists.append(bg_list[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

    t_list = []
    for i in range(split_n):
        bg_list_i = bg_lists[i]
        t = threading.Thread(target=main_thread_paste_cropped_object_aug_data, args=(bg_list_i, bg_images_path, bg_labels_path, cropped_object_list, cropped_object_path, random_N, scatter_bbxs_num, save_path, dis_thresh, scale_flag, scale_type, scale_ratio, cls, add_rename_str,))
        t_list.append(t)

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


# ======================================================================================================================================
# ================================== Paste cropped object for det train negative samples multi thread ==================================
# ======================================================================================================================================


# ======================================================================================================================================
# ================================== PIL paste cropped object for det train negative samples multi thread ==================================
# ======================================================================================================================================

def get_lbl_bbx_pil_paste_cropped_object_aug_data(bg_lbl_abs_path, img_size):
    """

    :param bg_lbl_abs_path:
    :param img_size: (h, w)
    :return:
    """
    bbxes = []
    with open(bg_lbl_abs_path, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
        for l in lines:
            l_ = [float(l.split(" ")[1]), float(l.split(" ")[2]), float(l.split(" ")[3]), float(l.split(" ")[4])]
            # bbx_VOC_format = convert_bbx_yolo_to_VOC(l_, img_size)
            bbx_VOC_format = convertBboxYOLO2VOC(img_size, l_)
            bbxes.append(bbx_VOC_format)
    return bbxes


def write_yolo_label_pil_paste_cropped_object_aug_data(labels_save_path, img_name, pasted_poses, bg_size, add_rename_str="pasted", scale_flag=False, scale_ratio=0.04, cls=0):
    txt_save_path_added_res = "{}/{}_{}_v6.txt".format(labels_save_path, img_name, add_rename_str)
    with open(txt_save_path_added_res, "w", encoding="utf-8") as fw:
        for bb_ in pasted_poses:
            if scale_flag:
                bbx_new = scale_down_bbx(bb_, scale_ratio=scale_ratio)
                # bb = convert_bbx_VOC_to_yolo((bbx_new[0], bbx_new[0] + bbx_new[2], bbx_new[1], bbx_new[1] + bbx_new[3]), bg_size)
                bb = convertBboxVOC2YOLO((bbx_new[0], bbx_new[1], bbx_new[0] + bbx_new[2], bbx_new[1] + bbx_new[3]), bg_size)
                txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bb]) + "\n"
                fw.write(txt_content)
            else:
                # bb = convert_bbx_VOC_to_yolo((bb_[0], bb_[0] + bb_[2], bb_[1], bb_[1] + bb_[3]), bg_size)
                bb = convertBboxVOC2YOLO((bb_[0], bb_[1], bb_[0] + bb_[2], bb_[1] + bb_[3]), bg_size)
                txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bb]) + "\n"
                fw.write(txt_content)


def gen_random_pos_pil_paste_cropped_object_aug_data(bbxes, paste_num, cropped_imgs, bg_size, dis_thresh=50, scatter_bbxs_num=3):
    paste_poses = []
    last_pos = (0, 0)  # try to scatter the bbxs.
    for ii in range(scatter_bbxs_num):
        for k in range(paste_num):
            cropped_k_size = cropped_imgs[k].shape[:2]
            if bg_size[1] - cropped_k_size[1] <= 0 or bg_size[0] - cropped_k_size[0] <= 0:
                continue
            paste_pos_k = [np.random.randint(0, (bg_size[1] - cropped_k_size[1])), np.random.randint(0, (bg_size[0] - cropped_k_size[0])), cropped_k_size[1], cropped_k_size[0]]

            for bb in bbxes:
                iou = cal_iou(bb, paste_pos_k)
                # if in yolov5 false positive detections bbx, is not our desired results
                if (paste_pos_k[0] >= bb[0] and paste_pos_k[0] <= bb[2]) and (paste_pos_k[1] >= bb[1] and paste_pos_k[1] <= bb[3]):
                    continue
                elif iou > 0.10:
                    continue
                elif np.sqrt((paste_pos_k[0] - bb[0]) ** 2 + (paste_pos_k[1] - bb[1]) ** 2) < dis_thresh:
                    continue
                elif last_pos != (0, 0):
                    if np.sqrt((paste_pos_k[0] - last_pos[0]) ** 2 + (paste_pos_k[1] - last_pos[1]) ** 2) < dis_thresh:
                        continue
                    else:
                        paste_poses.append(paste_pos_k)
                else:
                    paste_poses.append(paste_pos_k)
            last_pos = paste_pos_k

    # 1. remove special bbx
    poses = copy.copy(paste_poses)

    if poses:
        if len(poses) >= 2:
            for bi in range(len(poses) - 2, -1, -1):
                bi_p0 = (poses[bi][0], poses[bi][1])
                for bj in range(len(poses) - 1, bi, -1):
                    # 1. bbxes very close, very small distance.
                    bj_p0 = (poses[bj][0], poses[bj][1])
                    bj_p1 = (poses[bj][0] + poses[bj][2], poses[bj][1])
                    bj_p2 = (poses[bj][0] + poses[bj][2], poses[bj][1] + poses[bj][3])
                    bj_p3 = (poses[bj][0], poses[bj][1] + poses[bj][3])

                    dis_bi_p0_bj_p0 = np.sqrt((bi_p0[0] - bj_p0[0]) ** 2 + (bi_p0[1] - bj_p0[1]) ** 2)
                    dis_bi_p0_bj_p1 = np.sqrt((bi_p0[0] - bj_p1[0]) ** 2 + (bi_p0[1] - bj_p1[1]) ** 2)
                    dis_bi_p0_bj_p2 = np.sqrt((bi_p0[0] - bj_p2[0]) ** 2 + (bi_p0[1] - bj_p2[1]) ** 2)
                    dis_bi_p0_bj_p3 = np.sqrt((bi_p0[0] - bj_p3[0]) ** 2 + (bi_p0[1] - bj_p3[1]) ** 2)

                    if dis_bi_p0_bj_p0 < dis_thresh or dis_bi_p0_bj_p1 < dis_thresh or dis_bi_p0_bj_p2 < dis_thresh or dis_bi_p0_bj_p3 < dis_thresh:
                        poses.remove(poses[bi])
                        # print("======================== S1 ========================")
                        continue

                    if poses[bi][0] < poses[bj][0] and poses[bi][1] < poses[bj][1] and poses[bi][0] + poses[bi][2] > poses[bj][0] + poses[bj][2] and poses[bi][1] + poses[bi][3] > poses[bj][1] + poses[bj][3]:
                        poses.remove(poses[bi])
                        # print("======================== S2.1 ========================")
                        continue
                    # 2.2 bi in bj
                    if poses[bi][0] > poses[bj][0] and poses[bi][1] > poses[bj][1] and poses[bi][0] + poses[bi][2] < poses[bj][0] + poses[bj][2] and poses[bi][1] + poses[bi][3] < poses[bj][1] + poses[bj][3]:
                        poses.remove(poses[bi])
                        # print("======================== S2.2 ========================")
                        continue

                    # 3. cal iou
                    iou = cal_iou(poses[bi], poses[bj])
                    if iou > 0.10:
                        poses.remove(poses[bi])
                        # print("======================== S3 ========================")
                        continue

    return poses


def PIL_paste_image_on_bg_pil_paste_cropped_object_aug_data(paste_imgs, bg_img, paste_poses_selected):
    pil_bg_img = Image.fromarray(np.uint8(bg_img)).convert("RGBA")

    for i, img in enumerate(paste_imgs):
        pil_img = Image.fromarray(np.uint8(img)).convert("RGBA")
        pil_img_alpha = pil_img.split()[-1]
        pil_bg_img.paste(pil_img, (paste_poses_selected[i][0], paste_poses_selected[i][1]), mask=pil_img_alpha)

    pil_bg_img = pil_bg_img.convert("RGB")
    return pil_bg_img


def main_thread_pil_paste_cropped_object_aug_data(bg_list_i, bg_images_path, bg_labels_path, cropped_object_list, cropped_object_path, save_path, paste_largest_num, add_rename_str, scale_flag, scale_ratio, cls, dis_thresh, scatter_bbxs_num):
    paste_num = np.random.randint(1, paste_largest_num + 1)

    images_save_path = save_path + "/images"
    labels_save_path = save_path + "/labels"
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(labels_save_path, exist_ok=True)

    for img in bg_list_i:
        try:
            img_name = os.path.splitext(img)[0]
            bg_img_abs_path = bg_images_path + "/{}".format(img)
            bg_lbl_abs_path = bg_labels_path + "/{}.txt".format(img_name)

            bg_img = cv2.imread(bg_img_abs_path)
            bg_size = bg_img.shape[:2]

            cropped_random_samples = random.sample(cropped_object_list, paste_num)
            cropped_imgs = []
            for s in cropped_random_samples:
                s_abs_path = cropped_object_path + "/{}".format(s)
                cropped_cv2img = cv2.imread(s_abs_path)
                cropped_imgs.append(cropped_cv2img)

            bbxes = get_lbl_bbx_pil_paste_cropped_object_aug_data(bg_lbl_abs_path, bg_size)
            paste_poses = gen_random_pos_pil_paste_cropped_object_aug_data(bbxes, paste_num, cropped_imgs, bg_size, dis_thresh=dis_thresh, scatter_bbxs_num=scatter_bbxs_num)
            if paste_poses:
                if len(paste_poses) < paste_num:
                    continue
            if not paste_poses:
                continue

            paste_poses_selected = random.sample(paste_poses, paste_num)
            pil_bg_img = PIL_paste_image_on_bg_pil_paste_cropped_object_aug_data(cropped_imgs, bg_img, paste_poses_selected)

            # save image and yolo label
            # pil_bg_img.save("{}/{}_{}.jpg".format(save_img_path, img_name, "pasted"))
            array_bg_img = np.asarray(pil_bg_img)
            cv2.imwrite("{}/{}_{}_v6.jpg".format(images_save_path, img_name, add_rename_str), array_bg_img)
            write_yolo_label_pil_paste_cropped_object_aug_data(labels_save_path, img_name, paste_poses_selected, bg_size, add_rename_str=add_rename_str, scale_flag=scale_flag, scale_ratio=scale_ratio, cls=cls)
        except Exception as Error:
            print(Error, Error.__traceback__.tb_lineno)


def pil_paste_cropped_object_for_det_aug_data_train_negative_samples_multi_thread_v6_main(bg_path, bg_images_dir_name, bg_labels_dir_name, cropped_object_path, save_path, paste_largest_num=1, add_rename_str="pasted", scale_flag=True, scale_ratio=0.02, cls=0, dis_thresh=50, scatter_bbxs_num=5):
    bg_images_path = bg_path + "/{}".format(bg_images_dir_name)
    bg_labels_path = bg_path + "/{}".format(bg_labels_dir_name)

    cropped_object_list = os.listdir(cropped_object_path)
    bg_list = os.listdir(bg_images_path)

    len_ = len(bg_list)
    bg_lists = []
    split_n = 8
    for j in range(split_n):
        bg_lists.append(bg_list[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

    t_list = []
    for i in range(split_n):
        bg_list_i = bg_lists[i]
        t = threading.Thread(target=main_thread_pil_paste_cropped_object_aug_data, args=(bg_list_i, bg_images_path, bg_labels_path, cropped_object_list, cropped_object_path, save_path, paste_largest_num, add_rename_str, scale_flag, scale_ratio, cls, dis_thresh, scatter_bbxs_num,))
        t_list.append(t)

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


# ======================================================================================================================================
# ================================== PIL paste cropped object for det train negative samples multi thread ==================================
# ======================================================================================================================================


# ======================================================================================================================================
# ============================== Add black bg images(e.g. seg output image) for det aug data multi thread ==============================
# ======================================================================================================================================
def image_gen_bbx_add_black_bg_object_aug_data(res_arr, size):
    bboxes = []
    for img in res_arr:
        cnts, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # bboxes.append((x, y, w, h))

            if w > 50 and h > 50:
                x_min = x
                x_max = x + w
                y_min = y
                y_max = y + h

                # bb = convert_bbx_VOC_to_yolo(size, (x_min, x_max, y_min, y_max))
                bb = convertBboxVOC2YOLO(size, (x_min, y_min, x_max, y_max))
                bboxes.append(bb)

    return bboxes


def timeit_add_black_bg_object_aug_data(func):
    def wrapper(bg_list, bg_path, seg_object_list, seg_object_path, data_path, random_N, save_image_path, save_txt_path, cls, rename_add_str):
        t1 = time.time()
        func(bg_list, bg_path, seg_object_list, seg_object_path, data_path, random_N, save_image_path, save_txt_path, cls, rename_add_str)
        t2 = time.time()
        print(t2 - t1)

    return wrapper


@timeit_add_black_bg_object_aug_data
def main_thread_add_black_bg_object_aug_data(bg_list, bg_path, seg_object_list, seg_object_path, data_path, random_N, save_image_path, save_txt_path, cls, rename_add_str):
    image_path = data_path + "/images"
    mask_path = data_path + "/masks"
    mask_list = os.listdir(mask_path)

    for bg in tqdm(bg_list):
        try:
            bg_abs_path = bg_path + "/{}".format(bg)
            bg_img_name = os.path.splitext(bg)[0]
            bg_img_pil = Image.open(bg_abs_path)
            bg_img_array = np.asarray(bg_img_pil)
            w, h = bg_img_pil.size

            random_num = np.random.randint(1, random_N + 1)  # paste random (less than random_num(including)) objects
            # seg_object_random_sample = random.sample(seg_object_list, random_num)
            seg_object_random_sample = random.sample(mask_list, random_num)

            for j, s in enumerate(seg_object_random_sample):
                s_name = os.path.splitext(s)[0]
                s_abs_path = mask_path + "/{}".format(s)
                seg_object_img_pil = Image.open(s_abs_path)
                object_array = np.asarray(seg_object_img_pil)
                resized_object = scale_uint16(object_array, (w, h))
                resized_object = cv2.cvtColor(resized_object, cv2.COLOR_RGB2BGR)
                resized_object_gray = cv2.cvtColor(resized_object, cv2.COLOR_BGR2GRAY)
                bg_img_array_BGR = cv2.cvtColor(bg_img_array, cv2.COLOR_RGB2BGR)
                bg_img_array_cp = bg_img_array_BGR.copy()

                image_abs_path = image_path + "/{}.jpg".format(s_name)
                mask_abs_path = mask_path + "/{}".format(s)
                cv2img = cv2.imread(image_abs_path)
                maskimg = cv2.imread(mask_abs_path)

                resized_cv2img = scale_uint16(cv2img, (w, h))

                resized_mask = scale_uint16(maskimg, (w, h))
                zeros = np.zeros(shape=resized_mask.shape)
                object_area = np.where((resized_mask[:, :, 0] != 0) & (resized_mask[:, :, 1] != 0) & (resized_mask[:, :, 2] != 0))
                x, y = object_area[1], object_area[0]
                for i in range(len(x)):
                    zeros[y[i], x[i], :] = resized_cv2img[y[i], x[i], :]
                    bg_img_array_cp[y[i], x[i], :] = (0, 0, 0)

                # object_area = np.where((resized_object[:, :, 0] > 0) & (resized_object[:, :, 1] > 0) & (resized_object[:, :, 2] > 0))
                # bg_img_array_cp = bg_img_array_BGR.copy()
                # for x_, y_ in zip(object_area[1], object_area[0]):
                #     try:
                #         bg_img_array_cp[y_, x_] = (0, 0, 0)
                #     except Exception as Error:
                #         print(Error)

                # added_res = bg_img_array_cp + resized_object
                added_res = bg_img_array_cp + zeros
                # open_ = cv2.morphologyEx(added_res, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

                # gen yolo txt
                resized_mask_0 = resized_mask[:, :, 0]
                cnts, hierarchy = cv2.findContours(resized_mask_0.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                sortedcnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
                x_, y_, w_, h_ = cv2.boundingRect(sortedcnts[0])
                x_min, x_max, y_min, y_max = x_, x_ + w_, y_, y_ + h_
                # bb = convert_bbx_VOC_to_yolo((h, w), (x_min, x_max, y_min, y_max))
                bb = convertBboxVOC2YOLO((h, w), (x_min, y_min, x_max, y_max))

                cv2.imwrite("{}/{}_added_{}_{}.jpg".format(save_image_path, bg_img_name, j, rename_add_str), added_res)
                txt_save_path_added = save_txt_path + "/{}_added_{}_{}.txt".format(bg_img_name, j, rename_add_str)
                with open(txt_save_path_added, "w", encoding="utf-8") as fw:
                    txt_content = "{}".format(cls) + " " + " ".join([str(a) for a in bb]) + "\n"
                    fw.write(txt_content)

                # ret, thresh = cv2.threshold(resized_object_gray.astype(np.uint8), 5, 255, cv2.THRESH_BINARY)
                # thresh_filtered = cv2.medianBlur(thresh, 7)
                # n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_filtered)
                # txt_save_path_added = save_txt_path + "/{}_added_{}_{}.txt".format(bg_img_name, j, rename_add_str)
                # txt_save_path_open = save_txt_path + "/{}_added_open_{}_{}.txt".format(bg_img_name, j, rename_add_str)
                #
                # res_arr = labels_to_split_images(n_labels, labels, (h, w))
                # bboxes = image_gen_bbx_add_black_bg_object_aug_data(res_arr, (h, w))
                # if len(bboxes) == n_labels - 1:
                #     cv2.imwrite("{}/{}_added_{}_{}.jpg".format(save_image_path, bg_img_name, j, rename_add_str), added_res)
                #     cv2.imwrite("{}/{}_added_open_{}_{}.jpg".format(save_image_path, bg_img_name, j, rename_add_str), open_)
                #
                #     with open(txt_save_path_added, "w", encoding="utf-8") as fw:
                #         for b in bboxes:
                #             txt_content = "{}".format(cls) + " " + " ".join([str(a) for a in b]) + "\n"
                #             fw.write(txt_content)
                #
                #     with open(txt_save_path_open, "w", encoding="utf-8") as fw:
                #         for b in bboxes:
                #             txt_content = "{}".format(cls) + " " + " ".join([str(a) for a in b]) + "\n"
                #             fw.write(txt_content)

        except Exception as Error:
            print(Error, Error.__traceback__.tb_lineno)


def add_black_bg_object_for_det_aug_data_multi_thread_main(bg_path, seg_object_path, data_path, save_data_path, random_N=2, cls=0, rename_add_str="moisture_absorber_20230426"):
    save_image_path = save_data_path + "/images"
    save_txt_path = save_data_path + "/labels"
    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_txt_path, exist_ok=True)

    bg_list = os.listdir(bg_path)
    seg_object_list = os.listdir(seg_object_path)

    len_ = len(bg_list)
    bg_lists = []
    split_n = 8
    for j in range(split_n):
        bg_lists.append(bg_list[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

    t_list = []
    for i in range(split_n):
        bg_list_i = bg_lists[i]
        t = threading.Thread(target=main_thread_add_black_bg_object_aug_data, args=(bg_list_i, bg_path, seg_object_list, seg_object_path, data_path, random_N, save_image_path, save_txt_path, cls, rename_add_str,))
        t_list.append(t)

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


# ======================================================================================================================================
# ============================== Add black bg images(e.g. seg output image) for det aug data multi thread ==============================
# ======================================================================================================================================


# ======================================================================================================================================
# ============================= Paste object like opencv seamless clone for det aug data multi thread v6 ===============================
# ======================================================================================================================================
def gen_translate_M_seamless_paste_v6(affine_num=2):
    """
    :param n:
    :return:
    """
    Ms = []
    for i in range(affine_num):
        M = np.array([[1, 0, np.random.randint(-30, 30)], [0, 1, np.random.randint(-30, 30)]], dtype=np.float32)
        Ms.append(M)

    return Ms


def gen_rotate_M_seamless_paste_v6(affine_num=2):
    Ms = []
    theta_list = [np.pi / 180, np.pi / 170, np.pi / 160, np.pi / 150, np.pi / 145, np.pi / 90]
    theta_list_select = random.sample(theta_list, affine_num)
    for theta in theta_list_select:
        M = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0]], dtype=np.float32)
        Ms.append(M)

    return Ms


def gen_perspective_tran_M_seamless_paste_v6(size, affine_num=2):
    h, w = size[0], size[1]
    Ms = []
    p_list = [2, 5, 8]
    if affine_num == 1:
        p_list_select = random.sample(p_list, 1)
    else:
        p_list_select = random.sample(p_list, affine_num // 2)
    for i in p_list_select:
        m_src = np.array([[0, 0], [w, 0], [w, h]], dtype=np.float32)
        m_dst = np.array([[0, 0], [w - i, i], [w - i, h - i]], dtype=np.float32)
        M = cv2.getAffineTransform(m_src, m_dst)
        Ms.append(M)

    for j in p_list_select:
        m_src = np.array([[0, 0], [0, h], [w, 0]], dtype=np.float32)
        m_dst = np.array([[j, j], [j, h - j], [w, 0]], dtype=np.float32)
        M = cv2.getAffineTransform(m_src, m_dst)
        Ms.append(M)

    return Ms


def write_yolo_label_seamless_paste_v6(labels_save_path, final_yolo_bbxes, bg_img_name, i, random_obj_num, cls=0, rename_add_str="lock_20230327", affine_type="affine_type"):
    lbl_save_abs_path = labels_save_path + "/{}_affine_{}_obj_{}_{}_{}.txt".format(bg_img_name, i, random_obj_num, rename_add_str, affine_type)
    with open(lbl_save_abs_path, "w", encoding="utf-8") as fw:
        for bb in final_yolo_bbxes:
            txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bb]) + "\n"
            fw.write(txt_content)


def timeit_seamless_paste_v6(func):
    def wrapper(bg_list_i, bg_path, bg_img_dir_name, bg_lbl_dir_name, object_path, save_path, obj_num, affine_num, threshold_min_thr, medianblur_k, pixel_thr, iou_thr, bbx_thr, cls, rename_add_str, random_scale_flag, adaptiveThreshold):
        t1 = time.time()
        func(bg_list_i, bg_path, bg_img_dir_name, bg_lbl_dir_name, object_path, save_path, obj_num, affine_num, threshold_min_thr, medianblur_k, pixel_thr, iou_thr, bbx_thr, cls, rename_add_str, random_scale_flag, adaptiveThreshold)
        t2 = time.time()
        print(t2 - t1)

    return wrapper


@timeit_seamless_paste_v6
def seamless_paste_main_thread_v6(bg_list_i, bg_path, bg_img_dir_name, bg_lbl_dir_name, object_path, save_path, obj_num, affine_num, threshold_min_thr, medianblur_k, pixel_thr, iou_thr, bbx_thr, cls, rename_add_str, random_scale_flag, adaptiveThreshold):
    bg_images_path = bg_path + "/{}".format(bg_img_dir_name)
    bg_labels_path = bg_path + "/{}".format(bg_lbl_dir_name)
    bg_list = os.listdir(bg_images_path)

    images_save_path = save_path + "/images"
    labels_save_path = save_path + "/labels"
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(labels_save_path, exist_ok=True)

    object_list = sorted(os.listdir(object_path))

    for bg in tqdm(bg_list_i):
        try:
            bg_abs_path = bg_images_path + "/{}".format(bg)
            bg_img_name = os.path.splitext(bg)[0]
            bg_lbl_abs_path = bg_labels_path + "/{}.txt".format(bg_img_name)
            bg_lbl_data = open(bg_lbl_abs_path, "r", encoding="utf-8")
            bg_lbl_data_lines = bg_lbl_data.readlines()
            bg_lbl_data.close()

            bg_cv2img = cv2.imread(bg_abs_path)
            bg_cv2img_cp = bg_cv2img.copy()
            bg_cv2img_cp2 = bg_cv2img.copy()
            bg_size = bg_cv2img.shape[:2]

            random_obj_num = np.random.randint(1, obj_num + 1)  # paste random (less than obj_num(including)) objects
            object_random_sample = random.sample(object_list, random_obj_num)

            translate_Ms = gen_translate_M_seamless_paste_v6(affine_num)
            rotate_Ms = gen_rotate_M_seamless_paste_v6(affine_num)

            # ========================================= translate =========================================
            affine_type = "translate"
            for idx in range(affine_num):
                pasted_bg_img = None
                final_yolo_bbxes = []
                bg_cv2img_for_paste = bg_cv2img_cp2

                obj_img_names = ""
                for o in object_random_sample:
                    o_abs_path = object_path + "/{}".format(o)
                    obj_img_name = os.path.splitext(o)[0]
                    obj_img_names += obj_img_name + "_"
                    cv2img = cv2.imread(o_abs_path)
                    img_size = cv2img.shape[:2]

                    # perspective_Ms = gen_perspective_tran_M_seamless_paste(img_size, affine_num)

                    out = cv2.warpAffine(cv2img, translate_Ms[idx], img_size[::-1])
                    out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    # ret, thresh = cv2.threshold(out_gray, threshold_min_thr, 255, cv2.THRESH_BINARY)
                    ret, thresh = thresh_img(out_gray, threshold_min_thr=threshold_min_thr, adaptiveThreshold=adaptiveThreshold)
                    thresh_filtered = cv2.medianBlur(thresh, medianblur_k)
                    cnts, hierarchy = cv2.findContours(thresh_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    sortedcnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
                    x_, y_, w_, h_ = cv2.boundingRect(sortedcnts[0])
                    bbx = []
                    if w_ > pixel_thr and h_ > pixel_thr:
                        bbx.append([x_, y_, w_, h_])

                    # print("img_size, out_size, [x_, y_, w_, h_]: {} {} {}".format(img_size, out.shape[:2], [x_, y_, w_, h_]))
                    # cv2.rectangle(cv2img, (x_, y_), (x_ + w_, y_ + h_), (255, 255, 0))
                    # cv2.imwrite("{}/bg_{}_obj_{}_cv2img.jpg".format(images_save_path, bg_img_name, obj_img_name), cv2img)
                    # cv2.rectangle(out, (x_, y_), (x_ + w_, y_ + h_), (255, 255, 0))
                    # cv2.imwrite("{}/bg_{}_obj_{}_affineout.jpg".format(images_save_path, bg_img_name, obj_img_name), out)
                    # cv2.imwrite("{}/bg_{}_obj_{}_thresh.jpg".format(images_save_path, bg_img_name, obj_img_name), thresh)
                    # cv2.imwrite("{}/bg_{}_obj_{}_thresh_filtered.jpg".format(images_save_path, bg_img_name, obj_img_name), thresh_filtered)

                    # gen random pos --> bbx
                    poses = []

                    while True:
                        paste_k_pos = (np.random.randint(0, (bg_size[1] - img_size[1])), np.random.randint(0, (bg_size[0] - img_size[0])))
                        paste_k_VOC_bbx = (paste_k_pos[0], paste_k_pos[1], paste_k_pos[0] + img_size[1], paste_k_pos[1] + img_size[0])
                        for l in bg_lbl_data_lines:
                            gb_yolo_bbx = list(map(float, l.strip().split(" ")[1:]))
                            # gb_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, gb_yolo_bbx)
                            gb_VOC_bbx = convertBboxYOLO2VOC(bg_size, gb_yolo_bbx)
                            iou = cal_iou(paste_k_VOC_bbx, gb_VOC_bbx)

                            if iou < iou_thr:
                                poses.append(paste_k_VOC_bbx)

                        if len(poses) >= 1:
                            break

                    select_one_pos = random.sample(poses, 1)
                    thresh_3c = cv2.merge([thresh, thresh, thresh])
                    bg_mask1 = np.zeros((select_one_pos[0][1], bg_size[1], 3), dtype=np.uint8)
                    bg_mask2 = np.zeros(((select_one_pos[0][3] - select_one_pos[0][1]), select_one_pos[0][0], 3), dtype=np.uint8)
                    bg_mask4 = np.zeros(((select_one_pos[0][3] - select_one_pos[0][1]), bg_size[1] - select_one_pos[0][0] - (select_one_pos[0][2] - select_one_pos[0][0]), 3), dtype=np.uint8)
                    bg_mask5 = np.zeros((bg_size[0] - select_one_pos[0][1] - (select_one_pos[0][3] - select_one_pos[0][1]), bg_size[1], 3), dtype=np.uint8)

                    bg_mask_mid = np.hstack((bg_mask2, thresh_3c, bg_mask4))
                    bg_mask = np.vstack((bg_mask1, bg_mask_mid, bg_mask5))

                    object_formed_mid = np.hstack((bg_mask2, out, bg_mask4))
                    object_formed = np.vstack((bg_mask1, object_formed_mid, bg_mask5))

                    bg_cv2img_for_paste = bg_cv2img_for_paste.copy()
                    object_area = np.where((bg_mask[:, :, 0] >= pixel_thr) & (bg_mask[:, :, 1] >= pixel_thr) & (bg_mask[:, :, 2] >= pixel_thr))
                    for x_b, y_b in zip(object_area[1], object_area[0]):
                        try:
                            bg_cv2img_for_paste[y_b, x_b] = (0, 0, 0)
                        except Exception as Error:
                            print(Error)

                    pasted_bg_img = bg_cv2img_for_paste + object_formed

                    # cv2.rectangle(pasted_bg_img, (select_one_pos[0][0], select_one_pos[0][1]), (select_one_pos[0][2], select_one_pos[0][3]), (255, 0, 255), 5)

                    final_VOC_bbx = [select_one_pos[0][0] + x_, select_one_pos[0][1] + y_, select_one_pos[0][0] + x_ + w_, select_one_pos[0][1] + y_ + h_]
                    final_yolo_bbx = convertBboxVOC2YOLO(bg_size, final_VOC_bbx)
                    assert final_yolo_bbx[0] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[1] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[2] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[3] > 0, "bbx should > 0!"

                    # assert h_ >= img_size[0] * bbx_thr, "May have some problems!"
                    # assert w_ >= img_size[1] * bbx_thr, "May have some problems!"

                    final_yolo_bbxes.append(final_yolo_bbx)

                    bg_cv2img_for_paste = pasted_bg_img

                # # bg_cv2img = pasted_bg_img
                # if random_obj_num >= 2:

                assert len(final_yolo_bbxes) == random_obj_num, "bbx length should be same as random_obj_num!"

                # remove overlapped bbx through iou
                overlap_flag = False
                for bi in range(len(final_yolo_bbxes)):
                    for bj in range(bi + 1, len(final_yolo_bbxes)):
                        # bi_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, final_yolo_bbxes[bi])
                        # bj_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, final_yolo_bbxes[bj])
                        bi_VOC_bbx = convertBboxYOLO2VOC(bg_size, final_yolo_bbxes[bi])
                        bj_VOC_bbx = convertBboxYOLO2VOC(bg_size, final_yolo_bbxes[bj])

                        iou_bi_bj = cal_iou(bi_VOC_bbx, bj_VOC_bbx)
                        if iou_bi_bj > 0:
                            overlap_flag = True
                            break

                if overlap_flag:
                    print("There are some bbxes overlapped!")
                    continue

                # write image and label
                cv2.imwrite("{}/{}_affine_{}_obj_{}_{}_{}.jpg".format(images_save_path, bg_img_name, idx, random_obj_num, rename_add_str, affine_type), pasted_bg_img)
                write_yolo_label_seamless_paste_v6(labels_save_path, final_yolo_bbxes, bg_img_name, idx, random_obj_num, cls=cls, rename_add_str=rename_add_str, affine_type=affine_type)

            # =========================================    rotate    =========================================
            affine_type = "rotate"
            for idx in range(affine_num):
                pasted_bg_img = None
                final_yolo_bbxes = []
                bg_cv2img_for_paste = bg_cv2img_cp2

                obj_img_names = ""
                for o in object_random_sample:
                    o_abs_path = object_path + "/{}".format(o)
                    obj_img_name = os.path.splitext(o)[0]
                    obj_img_names += obj_img_name + "_"
                    cv2img = cv2.imread(o_abs_path)
                    img_size = cv2img.shape[:2]

                    # perspective_Ms = gen_perspective_tran_M_seamless_paste(img_size, affine_num)

                    out = cv2.warpAffine(cv2img, rotate_Ms[idx], img_size[::-1])
                    out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    # ret, thresh = cv2.threshold(out_gray, threshold_min_thr, 255, cv2.THRESH_BINARY)
                    ret, thresh = thresh_img(out_gray, threshold_min_thr=threshold_min_thr, adaptiveThreshold=adaptiveThreshold)
                    thresh_filtered = cv2.medianBlur(thresh, medianblur_k)
                    cnts, hierarchy = cv2.findContours(thresh_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    sortedcnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
                    x_, y_, w_, h_ = cv2.boundingRect(sortedcnts[0])
                    bbx = []
                    if w_ > pixel_thr and h_ > pixel_thr:
                        bbx.append([x_, y_, w_, h_])

                    # print("img_size, out_size, [x_, y_, w_, h_]: {} {} {}".format(img_size, out.shape[:2], [x_, y_, w_, h_]))
                    # cv2.rectangle(cv2img, (x_, y_), (x_ + w_, y_ + h_), (255, 255, 0))
                    # cv2.imwrite("{}/bg_{}_obj_{}_cv2img.jpg".format(images_save_path, bg_img_name, obj_img_name), cv2img)
                    # cv2.rectangle(out, (x_, y_), (x_ + w_, y_ + h_), (255, 255, 0))
                    # cv2.imwrite("{}/bg_{}_obj_{}_affineout.jpg".format(images_save_path, bg_img_name, obj_img_name), out)
                    # cv2.imwrite("{}/bg_{}_obj_{}_thresh.jpg".format(images_save_path, bg_img_name, obj_img_name), thresh)
                    # cv2.imwrite("{}/bg_{}_obj_{}_thresh_filtered.jpg".format(images_save_path, bg_img_name, obj_img_name), thresh_filtered)

                    # gen random pos --> bbx
                    poses = []

                    while True:
                        paste_k_pos = (np.random.randint(0, (bg_size[1] - img_size[1])), np.random.randint(0, (bg_size[0] - img_size[0])))
                        paste_k_VOC_bbx = (paste_k_pos[0], paste_k_pos[1], paste_k_pos[0] + img_size[1], paste_k_pos[1] + img_size[0])
                        for l in bg_lbl_data_lines:
                            gb_yolo_bbx = list(map(float, l.strip().split(" ")[1:]))
                            # gb_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, gb_yolo_bbx)
                            gb_VOC_bbx = convertBboxYOLO2VOC(bg_size, gb_yolo_bbx)
                            iou = cal_iou(paste_k_VOC_bbx, gb_VOC_bbx)

                            if iou < iou_thr:
                                poses.append(paste_k_VOC_bbx)

                        if len(poses) >= 1:
                            break

                    select_one_pos = random.sample(poses, 1)
                    thresh_3c = cv2.merge([thresh, thresh, thresh])
                    bg_mask1 = np.zeros((select_one_pos[0][1], bg_size[1], 3), dtype=np.uint8)
                    bg_mask2 = np.zeros(((select_one_pos[0][3] - select_one_pos[0][1]), select_one_pos[0][0], 3), dtype=np.uint8)
                    bg_mask4 = np.zeros(((select_one_pos[0][3] - select_one_pos[0][1]), bg_size[1] - select_one_pos[0][0] - (select_one_pos[0][2] - select_one_pos[0][0]), 3), dtype=np.uint8)
                    bg_mask5 = np.zeros((bg_size[0] - select_one_pos[0][1] - (select_one_pos[0][3] - select_one_pos[0][1]), bg_size[1], 3), dtype=np.uint8)

                    bg_mask_mid = np.hstack((bg_mask2, thresh_3c, bg_mask4))
                    bg_mask = np.vstack((bg_mask1, bg_mask_mid, bg_mask5))

                    object_formed_mid = np.hstack((bg_mask2, out, bg_mask4))
                    object_formed = np.vstack((bg_mask1, object_formed_mid, bg_mask5))

                    bg_cv2img_for_paste = bg_cv2img_for_paste.copy()
                    object_area = np.where((bg_mask[:, :, 0] >= pixel_thr) & (bg_mask[:, :, 1] >= pixel_thr) & (bg_mask[:, :, 2] >= pixel_thr))
                    for x_b, y_b in zip(object_area[1], object_area[0]):
                        try:
                            bg_cv2img_for_paste[y_b, x_b] = (0, 0, 0)
                        except Exception as Error:
                            print(Error)

                    pasted_bg_img = bg_cv2img_for_paste + object_formed

                    # cv2.rectangle(pasted_bg_img, (select_one_pos[0][0], select_one_pos[0][1]), (select_one_pos[0][2], select_one_pos[0][3]), (255, 0, 255), 5)

                    final_VOC_bbx = [select_one_pos[0][0] + x_, select_one_pos[0][1] + y_, select_one_pos[0][0] + x_ + w_, select_one_pos[0][1] + y_ + h_]
                    final_yolo_bbx = convertBboxVOC2YOLO(bg_size, final_VOC_bbx)
                    assert final_yolo_bbx[0] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[1] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[2] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[3] > 0, "bbx should > 0!"

                    # assert h_ >= img_size[0] * bbx_thr, "May have some problems!"
                    # assert w_ >= img_size[1] * bbx_thr, "May have some problems!"

                    final_yolo_bbxes.append(final_yolo_bbx)

                    bg_cv2img_for_paste = pasted_bg_img

                # # bg_cv2img = pasted_bg_img
                # if random_obj_num >= 2:

                assert len(final_yolo_bbxes) == random_obj_num, "bbx length should be same as random_obj_num!"

                # remove overlapped bbx through iou
                overlap_flag = False
                for bi in range(len(final_yolo_bbxes)):
                    for bj in range(bi + 1, len(final_yolo_bbxes)):
                        # bi_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, final_yolo_bbxes[bi])
                        # bj_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, final_yolo_bbxes[bj])
                        bi_VOC_bbx = convertBboxYOLO2VOC(bg_size, final_yolo_bbxes[bi])
                        bj_VOC_bbx = convertBboxYOLO2VOC(bg_size, final_yolo_bbxes[bj])

                        iou_bi_bj = cal_iou(bi_VOC_bbx, bj_VOC_bbx)
                        if iou_bi_bj > 0:
                            overlap_flag = True
                            break

                if overlap_flag:
                    print("There are some bbxes overlapped!")
                    continue

                # write image and label
                cv2.imwrite("{}/{}_affine_{}_obj_{}_{}_{}.jpg".format(images_save_path, bg_img_name, idx, random_obj_num, rename_add_str, affine_type), pasted_bg_img)
                write_yolo_label_seamless_paste_v6(labels_save_path, final_yolo_bbxes, bg_img_name, idx, random_obj_num, cls=cls, rename_add_str=rename_add_str, affine_type=affine_type)

            # =========================================  perspective =========================================
            affine_type = "perspective"
            for idx in range(affine_num):
                pasted_bg_img = None
                final_yolo_bbxes = []
                bg_cv2img_for_paste = bg_cv2img_cp2

                obj_img_names = ""
                for o in object_random_sample:
                    o_abs_path = object_path + "/{}".format(o)
                    obj_img_name = os.path.splitext(o)[0]
                    obj_img_names += obj_img_name + "_"
                    cv2img = cv2.imread(o_abs_path)
                    img_size = cv2img.shape[:2]

                    perspective_Ms = gen_perspective_tran_M_seamless_paste_v6(img_size, 1)

                    out = cv2.warpAffine(cv2img, perspective_Ms[idx], img_size[::-1])
                    out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    # ret, thresh = cv2.threshold(out_gray, threshold_min_thr, 255, cv2.THRESH_BINARY)
                    ret, thresh = thresh_img(out_gray, threshold_min_thr=threshold_min_thr, adaptiveThreshold=adaptiveThreshold)
                    thresh_filtered = cv2.medianBlur(thresh, medianblur_k)
                    cnts, hierarchy = cv2.findContours(thresh_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    sortedcnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
                    x_, y_, w_, h_ = cv2.boundingRect(sortedcnts[0])
                    bbx = []
                    if w_ > pixel_thr and h_ > pixel_thr:
                        bbx.append([x_, y_, w_, h_])

                    # print("img_size, out_size, [x_, y_, w_, h_]: {} {} {}".format(img_size, out.shape[:2], [x_, y_, w_, h_]))
                    # cv2.rectangle(cv2img, (x_, y_), (x_ + w_, y_ + h_), (255, 255, 0))
                    # cv2.imwrite("{}/bg_{}_obj_{}_cv2img.jpg".format(images_save_path, bg_img_name, obj_img_name), cv2img)
                    # cv2.rectangle(out, (x_, y_), (x_ + w_, y_ + h_), (255, 255, 0))
                    # cv2.imwrite("{}/bg_{}_obj_{}_affineout.jpg".format(images_save_path, bg_img_name, obj_img_name), out)
                    # cv2.imwrite("{}/bg_{}_obj_{}_thresh.jpg".format(images_save_path, bg_img_name, obj_img_name), thresh)
                    # cv2.imwrite("{}/bg_{}_obj_{}_thresh_filtered.jpg".format(images_save_path, bg_img_name, obj_img_name), thresh_filtered)

                    # gen random pos --> bbx
                    poses = []

                    while True:
                        paste_k_pos = (np.random.randint(0, (bg_size[1] - img_size[1])), np.random.randint(0, (bg_size[0] - img_size[0])))
                        paste_k_VOC_bbx = (paste_k_pos[0], paste_k_pos[1], paste_k_pos[0] + img_size[1], paste_k_pos[1] + img_size[0])
                        for l in bg_lbl_data_lines:
                            gb_yolo_bbx = list(map(float, l.strip().split(" ")[1:]))
                            # gb_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, gb_yolo_bbx)
                            gb_VOC_bbx = convertBboxYOLO2VOC(bg_size, gb_yolo_bbx)
                            iou = cal_iou(paste_k_VOC_bbx, gb_VOC_bbx)

                            if iou < iou_thr:
                                poses.append(paste_k_VOC_bbx)

                        if len(poses) >= 1:
                            break

                    select_one_pos = random.sample(poses, 1)
                    thresh_3c = cv2.merge([thresh, thresh, thresh])
                    bg_mask1 = np.zeros((select_one_pos[0][1], bg_size[1], 3), dtype=np.uint8)
                    bg_mask2 = np.zeros(((select_one_pos[0][3] - select_one_pos[0][1]), select_one_pos[0][0], 3), dtype=np.uint8)
                    bg_mask4 = np.zeros(((select_one_pos[0][3] - select_one_pos[0][1]), bg_size[1] - select_one_pos[0][0] - (select_one_pos[0][2] - select_one_pos[0][0]), 3), dtype=np.uint8)
                    bg_mask5 = np.zeros((bg_size[0] - select_one_pos[0][1] - (select_one_pos[0][3] - select_one_pos[0][1]), bg_size[1], 3), dtype=np.uint8)

                    bg_mask_mid = np.hstack((bg_mask2, thresh_3c, bg_mask4))
                    bg_mask = np.vstack((bg_mask1, bg_mask_mid, bg_mask5))

                    object_formed_mid = np.hstack((bg_mask2, out, bg_mask4))
                    object_formed = np.vstack((bg_mask1, object_formed_mid, bg_mask5))

                    bg_cv2img_for_paste = bg_cv2img_for_paste.copy()
                    object_area = np.where((bg_mask[:, :, 0] >= pixel_thr) & (bg_mask[:, :, 1] >= pixel_thr) & (bg_mask[:, :, 2] >= pixel_thr))
                    for x_b, y_b in zip(object_area[1], object_area[0]):
                        try:
                            bg_cv2img_for_paste[y_b, x_b] = (0, 0, 0)
                        except Exception as Error:
                            print(Error)

                    pasted_bg_img = bg_cv2img_for_paste + object_formed

                    # cv2.rectangle(pasted_bg_img, (select_one_pos[0][0], select_one_pos[0][1]), (select_one_pos[0][2], select_one_pos[0][3]), (255, 0, 255), 5)

                    final_VOC_bbx = [select_one_pos[0][0] + x_, select_one_pos[0][1] + y_, select_one_pos[0][0] + x_ + w_, select_one_pos[0][1] + y_ + h_]
                    final_yolo_bbx = convertBboxVOC2YOLO(bg_size, final_VOC_bbx)
                    assert final_yolo_bbx[0] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[1] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[2] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[3] > 0, "bbx should > 0!"

                    # assert h_ >= img_size[0] * bbx_thr, "May have some problems!"
                    # assert w_ >= img_size[1] * bbx_thr, "May have some problems!"

                    final_yolo_bbxes.append(final_yolo_bbx)

                    bg_cv2img_for_paste = pasted_bg_img

                # # bg_cv2img = pasted_bg_img
                # if random_obj_num >= 2:

                assert len(final_yolo_bbxes) == random_obj_num, "bbx length should be same as random_obj_num!"

                # remove overlapped bbx through iou
                overlap_flag = False
                for bi in range(len(final_yolo_bbxes)):
                    for bj in range(bi + 1, len(final_yolo_bbxes)):
                        # bi_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, final_yolo_bbxes[bi])
                        # bj_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, final_yolo_bbxes[bj])
                        bi_VOC_bbx = convertBboxYOLO2VOC(bg_size, final_yolo_bbxes[bi])
                        bj_VOC_bbx = convertBboxYOLO2VOC(bg_size, final_yolo_bbxes[bj])

                        iou_bi_bj = cal_iou(bi_VOC_bbx, bj_VOC_bbx)
                        if iou_bi_bj > 0:
                            overlap_flag = True
                            break

                if overlap_flag:
                    print("There are some bbxes overlapped!")
                    continue

                # write image and label
                cv2.imwrite("{}/{}_affine_{}_obj_{}_{}_{}.jpg".format(images_save_path, bg_img_name, idx, random_obj_num, rename_add_str, affine_type), pasted_bg_img)
                write_yolo_label_seamless_paste_v6(labels_save_path, final_yolo_bbxes, bg_img_name, idx, random_obj_num, cls=cls, rename_add_str=rename_add_str, affine_type=affine_type)

            # ========================================= random scale =========================================
            affine_type = "random_scale"
            for idx in range(affine_num):
                pasted_bg_img = None
                final_yolo_bbxes = []
                bg_cv2img_for_paste = bg_cv2img_cp2

                obj_img_names = ""
                for o in object_random_sample:
                    o_abs_path = object_path + "/{}".format(o)
                    obj_img_name = os.path.splitext(o)[0]
                    obj_img_names += obj_img_name + "_"
                    cv2img = cv2.imread(o_abs_path)
                    img_size = cv2img.shape[:2]

                    # perspective_Ms = gen_perspective_tran_M_seamless_paste(img_size, 1)
                    #
                    # out = cv2.warpAffine(cv2img, perspective_Ms[idx], img_size[::-1])
                    # out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    # ret, thresh = cv2.threshold(out_gray, threshold_min_thr, 255, cv2.THRESH_BINARY)
                    # thresh_filtered = cv2.medianBlur(thresh, medianblur_k)
                    # cnts, hierarchy = cv2.findContours(thresh_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    cv2img = np.asarray(cv2img)[:, :, ::-1]
                    scale_cut_size = [1.2, 1.4, 1.6, 1.8, 2, 2.5, 5, 6, 7, 8, 10, 12, 15]
                    if random_scale_flag == "small_images":
                        scale_cut_size = scale_cut_size[:int(len(scale_cut_size) / 2)]
                    elif random_scale_flag == "big_images":
                        scale_cut_size = scale_cut_size[:int(len(scale_cut_size) * 2 / 3)]
                    scale_cut_size_choose = random.sample(scale_cut_size, 1)

                    target_size = (int(img_size[1] / scale_cut_size_choose[0]), int(img_size[0] / scale_cut_size_choose[0]))
                    scale_img = cv2.resize(cv2img, target_size)
                    scale_pil_img = Image.fromarray(np.uint8(scale_img))
                    new_img = Image.new("RGB", img_size[::-1], (0, 0, 0))
                    pos = (np.random.randint(0, (img_size[1] - target_size[0])), np.random.randint(0, (img_size[0] - target_size[1])))
                    new_img.paste(scale_pil_img, pos)

                    new_img_cv2 = np.asarray(new_img)[:, :, ::-1]
                    out_gray = cv2.cvtColor(new_img_cv2, cv2.COLOR_BGR2GRAY)
                    # ret, thresh = cv2.threshold(out_gray, threshold_min_thr, 255, cv2.THRESH_BINARY)
                    ret, thresh = thresh_img(out_gray, threshold_min_thr=threshold_min_thr, adaptiveThreshold=adaptiveThreshold)
                    thresh_filtered = cv2.medianBlur(thresh, medianblur_k)
                    cnts, hierarchy = cv2.findContours(thresh_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    sortedcnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
                    x_, y_, w_, h_ = cv2.boundingRect(sortedcnts[0])
                    bbx = []
                    if w_ > pixel_thr and h_ > pixel_thr:
                        bbx.append([x_, y_, w_, h_])

                    # print("img_size, out_size, [x_, y_, w_, h_]: {} {} {}".format(img_size, out.shape[:2], [x_, y_, w_, h_]))
                    # cv2.rectangle(cv2img, (x_, y_), (x_ + w_, y_ + h_), (255, 255, 0))
                    # cv2.imwrite("{}/bg_{}_obj_{}_cv2img.jpg".format(images_save_path, bg_img_name, obj_img_name), cv2img)
                    # cv2.rectangle(out, (x_, y_), (x_ + w_, y_ + h_), (255, 255, 0))
                    # cv2.imwrite("{}/bg_{}_obj_{}_affineout.jpg".format(images_save_path, bg_img_name, obj_img_name), out)
                    # cv2.imwrite("{}/bg_{}_obj_{}_thresh.jpg".format(images_save_path, bg_img_name, obj_img_name), thresh)
                    # cv2.imwrite("{}/bg_{}_obj_{}_thresh_filtered.jpg".format(images_save_path, bg_img_name, obj_img_name), thresh_filtered)

                    # gen random pos --> bbx
                    poses = []

                    while True:
                        paste_k_pos = (np.random.randint(0, (bg_size[1] - img_size[1])), np.random.randint(0, (bg_size[0] - img_size[0])))
                        paste_k_VOC_bbx = (paste_k_pos[0], paste_k_pos[1], paste_k_pos[0] + img_size[1], paste_k_pos[1] + img_size[0])
                        for l in bg_lbl_data_lines:
                            gb_yolo_bbx = list(map(float, l.strip().split(" ")[1:]))
                            # gb_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, gb_yolo_bbx)
                            gb_VOC_bbx = convertBboxYOLO2VOC(bg_size, gb_yolo_bbx)
                            iou = cal_iou(paste_k_VOC_bbx, gb_VOC_bbx)

                            if iou < iou_thr:
                                poses.append(paste_k_VOC_bbx)

                        if len(poses) >= 1:
                            break

                    select_one_pos = random.sample(poses, 1)
                    thresh_3c = cv2.merge([thresh, thresh, thresh])
                    bg_mask1 = np.zeros((select_one_pos[0][1], bg_size[1], 3), dtype=np.uint8)
                    bg_mask2 = np.zeros(((select_one_pos[0][3] - select_one_pos[0][1]), select_one_pos[0][0], 3), dtype=np.uint8)
                    bg_mask4 = np.zeros(((select_one_pos[0][3] - select_one_pos[0][1]), bg_size[1] - select_one_pos[0][0] - (select_one_pos[0][2] - select_one_pos[0][0]), 3), dtype=np.uint8)
                    bg_mask5 = np.zeros((bg_size[0] - select_one_pos[0][1] - (select_one_pos[0][3] - select_one_pos[0][1]), bg_size[1], 3), dtype=np.uint8)

                    bg_mask_mid = np.hstack((bg_mask2, thresh_3c, bg_mask4))
                    bg_mask = np.vstack((bg_mask1, bg_mask_mid, bg_mask5))

                    object_formed_mid = np.hstack((bg_mask2, new_img_cv2, bg_mask4))
                    object_formed = np.vstack((bg_mask1, object_formed_mid, bg_mask5))

                    bg_cv2img_for_paste = bg_cv2img_for_paste.copy()
                    object_area = np.where((bg_mask[:, :, 0] >= pixel_thr) & (bg_mask[:, :, 1] >= pixel_thr) & (bg_mask[:, :, 2] >= pixel_thr))
                    for x_b, y_b in zip(object_area[1], object_area[0]):
                        try:
                            bg_cv2img_for_paste[y_b, x_b] = (0, 0, 0)
                        except Exception as Error:
                            print(Error)

                    pasted_bg_img = bg_cv2img_for_paste + object_formed

                    # cv2.rectangle(pasted_bg_img, (select_one_pos[0][0], select_one_pos[0][1]), (select_one_pos[0][2], select_one_pos[0][3]), (255, 0, 255), 5)

                    final_VOC_bbx = [select_one_pos[0][0] + x_, select_one_pos[0][1] + y_, select_one_pos[0][0] + x_ + w_, select_one_pos[0][1] + y_ + h_]
                    final_yolo_bbx = convertBboxVOC2YOLO(bg_size, final_VOC_bbx)
                    assert final_yolo_bbx[0] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[1] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[2] > 0, "bbx should > 0!"
                    assert final_yolo_bbx[3] > 0, "bbx should > 0!"

                    # assert h_ >= img_size[0] * bbx_thr, "May have some problems!"
                    # assert w_ >= img_size[1] * bbx_thr, "May have some problems!"

                    final_yolo_bbxes.append(final_yolo_bbx)

                    bg_cv2img_for_paste = pasted_bg_img

                # # bg_cv2img = pasted_bg_img
                # if random_obj_num >= 2:

                assert len(final_yolo_bbxes) == random_obj_num, "bbx length should be same as random_obj_num!"

                # remove overlapped bbx through iou
                overlap_flag = False
                for bi in range(len(final_yolo_bbxes)):
                    for bj in range(bi + 1, len(final_yolo_bbxes)):
                        # bi_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, final_yolo_bbxes[bi])
                        # bj_VOC_bbx = convert_bbx_yolo_to_VOC(bg_size, final_yolo_bbxes[bj])
                        bi_VOC_bbx = convertBboxYOLO2VOC(bg_size, final_yolo_bbxes[bi])
                        bj_VOC_bbx = convertBboxYOLO2VOC(bg_size, final_yolo_bbxes[bj])

                        iou_bi_bj = cal_iou(bi_VOC_bbx, bj_VOC_bbx)
                        if iou_bi_bj > 0:
                            overlap_flag = True
                            break

                if overlap_flag:
                    print("There are some bbxes overlapped!")
                    continue

                # write image and label
                cv2.imwrite("{}/{}_affine_{}_obj_{}_{}_{}.jpg".format(images_save_path, bg_img_name, idx, random_obj_num, rename_add_str, affine_type), pasted_bg_img)
                write_yolo_label_seamless_paste_v6(labels_save_path, final_yolo_bbxes, bg_img_name, idx, random_obj_num, cls=cls, rename_add_str=rename_add_str, affine_type=affine_type)

        except Exception as Error:
            print("Line: {} Error: {}".format(Error.__traceback__.tb_lineno, Error))


def seamless_paste_main_v6(bg_path, bg_img_dir_name, bg_lbl_dir_name, object_path, save_path, obj_num=2, affine_num=2, threshold_min_thr=10, medianblur_k=5, pixel_thr=10, iou_thr=0.05, bbx_thr=0.80, cls=0, rename_add_str="exit_light_20230411", random_scale_flag="small_images", adaptiveThreshold=True):
    bg_images_path = bg_path + "/{}".format(bg_img_dir_name)
    bg_labels_path = bg_path + "/{}".format(bg_lbl_dir_name)

    bg_list = os.listdir(bg_images_path)

    len_ = len(bg_list)
    bg_lists = []
    split_n = 8
    for j in range(split_n):
        bg_lists.append(bg_list[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

    t_list = []
    for i in range(split_n):
        bg_list_i = bg_lists[i]
        t = threading.Thread(target=seamless_paste_main_thread_v6, args=(bg_list_i, bg_path, bg_img_dir_name, bg_lbl_dir_name, object_path, save_path, obj_num, affine_num, threshold_min_thr, medianblur_k, pixel_thr, iou_thr, bbx_thr, cls, rename_add_str, random_scale_flag, adaptiveThreshold,))
        t_list.append(t)

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()

# ======================================================================================================================================
# ============================= Paste object like opencv seamless clone for det aug data multi thread v6 ===============================
# ======================================================================================================================================
def remove_yolo_txt_contain_specific_class(data_path, rm_cls=(1, 2,)):
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


def convert_Stanford_Dogs_Dataset_annotations_to_yolo_format(data_path):
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
                    bb = convertBboxVOC2YOLO((h, w), b)
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

                in_file.close()
                out_file.close()

            except Exception as Error:
                print("Error: {} {}".format(Error, f_name))


def convert_WiderPerson_Dataset_annotations_to_yolo_format(data_path):
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
        cv2img = cv2.imread(img_abs_path)
        img_shape = cv2img.shape[:2]

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
            yolo_bb = convertBboxVOC2YOLO(img_shape, VOC_bb)
            txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in yolo_bb]) + "\n"
            new_lbl.write(txt_content)

        orig_lbl.close()
        new_lbl.close()


def convert_TinyPerson_Dataset_annotations_to_yolo_format(data_path):
    data_type = ["train", "test"]
    dense_or_not = ["", "dense"]
    for dt in data_type:
        for d in dense_or_not:
            save_path = data_path + "/yolo_format/{}/labels_{}".format(dt, d)
            os.makedirs(save_path, exist_ok=True)

            json_data = return_json_data(data_path, dt, d)

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
                    yolo_bbx = convertBboxVOC2YOLO(img_shape, VOC_bbx)

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


def convert_AI_TOD_Dataset_to_yolo_format(data_path):
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

            cv2img = cv2.imread(img_abs_path)
            img_shape = cv2img.shape[:2]

            txt_fo = open(lbl_abs_path, "r", encoding="utf-8")
            txt_data = txt_fo.readlines()
            txt_fw = open(lbl_dst_path, "w", encoding="utf-8")

            for line in txt_data:
                l = line.strip().split(" ")
                cls = classes.index(l[-1])
                bbx = list(map(float, l[:4]))
                # bbx = list(np.array([bbx])[:, [0, 2, 1, 3]][0])
                # bbx_yolo = convert_bbx_VOC_to_yolo(img_shape, bbx)
                bbx_yolo = convertBboxVOC2YOLO(img_shape, bbx)

                txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bbx_yolo]) + "\n"
                txt_fw.write(txt_content)

            txt_fw.close()


def check_yolo_dataset_classes(data_path):
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"

    lbl_list = sorted(os.listdir(lbl_path))

    classes = []

    for lbl in lbl_list:
        lbl_abs_path = lbl_path + "/{}".format(lbl)
        txt_fo = open(lbl_abs_path, "r", encoding="utf-8")
        txt_data = txt_fo.readlines()
        txt_fo.close()

        for l in txt_data:
            l_ = l.strip().split(" ")
            cls = int(l_[0])
            if cls not in classes:
                classes.append(cls)

    classes.sort()

    print("labels: {} len(labels): {}".format(classes, len(classes)))


def gen_coco_unlabel_json(data_path):
    # For mmdetection SSOD
    # from mmengine.fileio import load, dump

    img_list = sorted(os.listdir(data_path))

    data = {"info": {"description": "COCO 2017 Dataset", "url": "http://cocodataset.org", "version": "1.0", "year": 2017, "contributor": "COCO Consortium", "date_created": "2017/09/01"},
            "images": [],
            "licenses": [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}, {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2, "name": "Attribution-NonCommercial License"}, {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 3, "name": "Attribution-NonCommercial-NoDerivs License"},
                         {"url": "http://creativecommons.org/licenses/by/2.0/", "id": 4, "name": "Attribution License"}, {"url": "http://creativecommons.org/licenses/by-sa/2.0/", "id": 5, "name": "Attribution-ShareAlike License"}, {"url": "http://creativecommons.org/licenses/by-nd/2.0/", "id": 6, "name": "Attribution-NoDerivs License"}, {"url": "http://flickr.com/commons/usage/", "id": 7, "name": "No known copyright restrictions"},
                         {"url": "http://www.usa.gov/copyright.shtml", "id": 8, "name": "United States Government Work"}],
            "categories": [{"supercategory": "smoke", "id": 1, "name": "smoke"}, {"supercategory": "fire", "id": 2, "name": "fire"}]
            }

    pbar = tqdm(enumerate(img_list), total=len(img_list), ncols=100, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    for idx, i in pbar:
        img_abs_path = data_path + "/{}".format(i)
        img_abs_path_10026 = "/home/wujiahu/data/006.Fire_Smoke_Det/train/smoke_fire/SSOD/train/not_labeled_34349" + "/{}".format(i)
        cv2img = cv2.imread(img_abs_path)
        h, w = cv2img.shape[:2]

        datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

        data["images"].append({
            "license": 1,
            "file_name": i,
            "coco_url": img_abs_path_10026,
            "height": h,
            "width": w,
            "date_captured": datetime,
            "flickr_url": "http://farm3.staticflickr.com/2567/test.jpg",
            "id": idx,
        })

    json_save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/unlabel.json"
    with open(json_save_path, "w") as fw:
        json.dump(data, fw)
        print("Saved --> {}".format(json_save_path))


def gen_yolo_others_label(data_path, others_class=2, fixnum=True, others_num=1, hw_thr=64, r=0.5, wait_time=7):
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    # new_lbl_path = data_path + "/labels_new"
    # os.makedirs(new_lbl_path, exist_ok=True)

    img_list = sorted(os.listdir(img_path))
    for i in tqdm(img_list):
        iname = os.path.splitext(i)[0]
        img_abs_path = img_path + "/{}".format(i)
        lbl_abs_path = lbl_path + "/{}.txt".format(iname)
        # lbl_abs_path_new = new_lbl_path + "/{}.txt".format(iname)

        cv2img = cv2.imread(img_abs_path)
        imgsz = cv2img.shape[:2]

        with open(lbl_abs_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            bbxs_voc = []
            for l in lines:
                l = l.strip().split(" ")
                bbx_yolo = list(map(float, l[1:]))
                # bbx_voc = convert_bbx_yolo_to_VOC(imgsz, bbx_yolo)
                bbx_voc = convertBboxYOLO2VOC(imgsz, bbx_yolo)
                bbxs_voc.append(bbx_voc)

        t1 = time.time()
        with open(lbl_abs_path, "a+", encoding="utf-8") as fw:
            iou = 1
            num_ = 0
            ious = 0
            while (iou > 0):
                x_ = np.random.randint(0, imgsz[1])
                y_ = np.random.randint(0, imgsz[0])
                w_ = imgsz[1] * np.random.random()
                h_ = imgsz[0] * np.random.random()

                while w_ > imgsz[1] * r:
                    w_ = w_ * np.random.random()
                while h_ > imgsz[0] * r:
                    h_ = h_ * np.random.random()

                if h_ < hw_thr:
                    h_ = random.sample([96, 128], 1)[0]
                if w_ < hw_thr:
                    w_ = random.sample([96, 128], 1)[0]

                if x_ + w_ > imgsz[1] or y_ + h_ > imgsz[0]:
                    continue

                random_bbx = [x_, y_, x_ + w_, y_ + h_]
                for b in bbxs_voc:
                    iou_ = cal_iou(random_bbx, b)
                    ious += iou_

                iou = ious

                if ious <= 0:
                    if fixnum:
                        if num_ == others_num:
                            break
                    else:
                        if num_ + np.random.randint(0, others_num) > others_num:
                            break

                    # random_bbx = list(np.array(random_bbx).reshape(1, -1)[:, [0, 2, 1, 3]][0, :])
                    # bbx_yolo_new = convert_bbx_VOC_to_yolo(imgsz, random_bbx)
                    random_bbx = list(np.array(random_bbx).reshape(1, -1)[0, :])
                    bbx_yolo_new = convertBboxVOC2YOLO(imgsz, random_bbx)
                    l_new = str(others_class) + " " + " ".join([str(a) for a in bbx_yolo_new]) + "\n"
                    fw.writelines(l_new)

                    num_ += 1

                t2 = time.time()
                if t2 - t1 > wait_time:
                    break

                ious = 0


def vis_coco_pose_data_test():
    img_path = "/home/zengyifan/wujiahu/data/000.Open_Dataset/coco/train2017/000000000036.jpg"
    label_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/coco_kpts/labels/train2017/000000000036.txt"

    cv2img = cv2.imread(img_path)
    imgsz = cv2img.shape[:2]

    with open(label_path, "r", encoding="utf-8") as fo:
        lines = fo.readlines()
        for l in lines:
            l = l.strip().split(" ")
            cls = int(l[0])
            bbx = list(map(float, l[1:5]))
            # bbx_voc = convert_bbx_yolo_to_VOC(imgsz, bbx)
            bbx_voc = convertBboxYOLO2VOC(imgsz, bbx)
            cv2.rectangle(cv2img, (bbx_voc[0], bbx_voc[1]), (bbx_voc[2], bbx_voc[3]), (255, 255, 0))

            points = np.asarray(list(map(float, l[5:]))).reshape(-1, 3)
            points_x = points[:, 0] * imgsz[1]
            points_y = points[:, 1] * imgsz[0]
            for i in range(points_x.shape[0]):
                if points_x[i] == 0 and points_y[i] == 0:
                    continue
                cv2.circle(cv2img, (int(round(points_x[i])), int(round(points_y[i]))), 3, (255, 0, 255), 2)

    cv2.imshow("test", cv2img)
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


def write_label_txt(fpath, bboxes, cls):
    with open(fpath, "w", encoding="utf-8") as fw:
        for bb in bboxes:
            txt_content = "{} ".format(cls) + " ".join([str(bi) for bi in bb]) + "\n"
            fw.write(txt_content)


def according_yolov8_pose_gen_head_bbx(data_path, cls=2):
    from ultralytics import YOLO

    model = YOLO("/home/zengyifan/wujiahu/yolo/ultralytics-main/yolov8s-pose.pt")

    dir_name = get_dir_name(data_path)
    file_list = get_file_list(data_path)
    save_path = make_save_path(data_path, dir_name_add_str="labels_head")

    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        base_name, file_name, suffix = get_baseName_fileName_suffix(f_abs_path)
        cv2img = cv2.imread(f_abs_path)
        imgsz = cv2img.shape[:2]
        bboxes = []

        results = model(f_abs_path)
        for r in results:
            keypoints = r.keypoints
            kpt_np = keypoints.xy.cpu().numpy()
            for pi in kpt_np:
                # for k in pi[:5]:
                # cv2.circle(cv2img, (int(k[0]), int(k[1])), 2, (255, 0, 255))
                if len(pi[:5]) < 5: continue
                bbx, area = get_bbx(pi[:5], imgsz, r=0.68)
                if area < 500:
                    continue

                # bbx_yolo = convert_bbx_VOC_to_yolo(imgsz, bbx)
                bbx_yolo = convertBboxVOC2YOLO(imgsz, bbx)
                bboxes.append(bbx_yolo)

        txt_save_path = "{}/{}.txt".format(save_path, file_name)
        write_label_txt(txt_save_path, bboxes, cls=cls)


def rm_iou_larger_than_zero(data_path):
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"

    save_path = make_save_path(data_path, "moved")

    file_list = get_file_list(lbl_path)
    for f in tqdm(file_list):
        try:
            fname = os.path.splitext(f)[0]
            f_abs_path = lbl_path + "/{}".format(f)
            f_ = open(f_abs_path, "r", encoding="utf-8")
            lines = f_.readlines()
            f_.close()

            img_abs_path = img_path + "/{}.jpg".format(fname)
            cv2img = cv2.imread(img_abs_path)
            imgsz = cv2img.shape[:2]

            bbxes = []
            for l_ in lines:
                l = l_.strip()
                l = l.split(" ")
                bbx_ = list(map(float, l[1:]))
                # bbx_voc = convert_bbx_yolo_to_VOC(imgsz, bbx_)
                bbx_voc = convertBboxYOLO2VOC(imgsz, bbx_)
                bbxes.append(bbx_voc)

            if len(bbxes) > 1:
                for i in range(len(bbxes)):
                    for j in range(i + 1, len(bbxes)):
                        iou_ij = cal_iou(bbxes[i], bbxes[j])
                        if iou_ij > 0:
                            # os.remove(f_abs_path)
                            # os.remove(img_abs_path)
                            f_dst_path = save_path + "/{}".format(f)
                            img_dst_path = save_path + "/{}.jpg".format(fname)
                            shutil.move(f_abs_path, f_dst_path)
                            shutil.move(img_abs_path, img_dst_path)
        except Exception as Error:
            print(Error)


def makeBorderAndChangeYoloBbx(data_path, n=3):
    save_img_path = make_save_path(data_path, "makeBorderAndChangeYoloBbx/images")
    save_lbl_path = make_save_path(data_path, "makeBorderAndChangeYoloBbx/labels")
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    file_list = get_file_list(img_path)
    for f in tqdm(file_list):
        fname = os.path.splitext(f)[0]
        img_abs_path = img_path + "/{}".format(f)
        lbl_abs_path = lbl_path + "/{}.txt".format(fname)

        cv2img = cv2.imread(img_abs_path)
        imgsz = cv2img.shape[:2]

        for ni in range(n):
            makeBorderSize = np.random.randint(200, 350)
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            res = cv2.copyMakeBorder(cv2img, makeBorderSize, makeBorderSize, makeBorderSize, makeBorderSize, borderType=cv2.BORDER_CONSTANT, value=color)

            lblf = open(lbl_abs_path, "r")
            lbl_data = lblf.readlines()
            lblf.close()

            img_dst_path = save_img_path + "/{}_{}.jpg".format(fname, ni)
            lbl_dst_path = save_lbl_path + "/{}_{}.txt".format(fname, ni)
            lblw = open(lbl_dst_path, "w", encoding="utf-8")

            for line in lbl_data:
                l = line.strip()
                cls = int(l.split(" ")[0])
                bbx = list(map(float, l.split(" ")[1:]))
                # bbx_voc = convert_bbx_yolo_to_VOC(imgsz, bbx)
                bbx_voc = convertBboxYOLO2VOC(imgsz, bbx)
                bbx_voc_new = [bbx_voc[0] + makeBorderSize, bbx_voc[1] + makeBorderSize, bbx_voc[2] + makeBorderSize, bbx_voc[3] + makeBorderSize]

                imgsz_new = [imgsz[0] + 2 * makeBorderSize, imgsz[1] + 2 * makeBorderSize]
                bbx_new = convertBboxVOC2YOLO(imgsz_new, bbx_voc_new)
                lbl_new_i = "{}".format(cls) + " " + " ".join([str(bb) for bb in bbx_new]) + "\n"
                lblw.write(lbl_new_i)

            cv2.imwrite(img_dst_path, res)
            lblw.close()


def moveYoloLabelNumGreaterThanN(data_path, cls=0, N=2):
    save_img_path = make_save_path(data_path, "makeBorderAndChangeYoloBbx/images")
    save_lbl_path = make_save_path(data_path, "makeBorderAndChangeYoloBbx/labels")
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    file_list = get_file_list(img_path)
    for f in tqdm(file_list):
        fname = os.path.splitext(f)[0]
        img_abs_path = img_path + "/{}".format(f)
        lbl_abs_path = lbl_path + "/{}.txt".format(fname)
        img_dst_path = save_img_path + "/{}".format(f)
        lbl_dst_path = save_lbl_path + "/{}.txt".format(fname)

        cv2img = cv2.imread(img_abs_path)
        imgsz = cv2img.shape[:2]

        lblf = open(lbl_abs_path, "r")
        lbl_data = lblf.readlines()
        lblf.close()

        clsn = 0
        for l in lbl_data:
            cls_ = int(l.strip().split(" ")[0])
            if cls_ == cls:
                clsn += 1

        if clsn > N:
            shutil.move(img_abs_path, img_dst_path)
            shutil.move(lbl_abs_path, lbl_dst_path)


def get_min_max_xy(bbx):
    """
    [[595, 582], [621, 598], [619, 620], [593, 607]]
    Parameters
    ----------
    bbx

    Returns
    -------

    """

    minx, miny = 1e12, 1e12
    maxx, maxy = -1e12, -1e12

    for bi in bbx:
        if bi[0] <= minx:
            minx = bi[0]
        if bi[1] <= miny:
            miny = bi[1]

        if bi[0] >= maxx:
            maxx = bi[0]
        if bi[1] >= maxy:
            maxy = bi[1]

    return [minx, miny, maxx, maxy]


def get_resize_hw(bbx):
    dis01 = np.sqrt((bbx[0][0] - bbx[1][0]) ** 2 + (bbx[0][1] - bbx[1][1]) ** 2)
    dis12 = np.sqrt((bbx[1][0] - bbx[2][0]) ** 2 + (bbx[1][1] - bbx[2][1]) ** 2)
    dis23 = np.sqrt((bbx[2][0] - bbx[3][0]) ** 2 + (bbx[2][1] - bbx[3][1]) ** 2)
    dis30 = np.sqrt((bbx[3][0] - bbx[0][0]) ** 2 + (bbx[3][1] - bbx[0][1]) ** 2)

    reszH = int(round(max(dis12, dis30)))
    reszW = int(round(max(dis01, dis23)))

    return (reszH, reszW)


def coco_names():
    names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}
    return names


# ===========================================================
# Detection dataset augmentatioin
# Aug with xmls
# ===========================================================
# 显示图片
def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像均为cv2读取
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=13,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5,
                 is_addNoise=True, is_changeLight=True, is_cutout=False, is_rotate_img_bbox=True,
                 is_crop_img_bboxes=True, is_shift_pic_bboxes=True, is_filp_pic_bboxes=True):

        # 配置各个操作的属性
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

        # 是否使用某种增强方式
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_cutout = is_cutout
        self.is_rotate_img_bbox = is_rotate_img_bbox
        self.is_crop_img_bboxes = is_crop_img_bboxes
        self.is_shift_pic_bboxes = is_shift_pic_bboxes
        self.is_filp_pic_bboxes = is_filp_pic_bboxes

    # 加噪声
    def _addNoise(self, img):
        from skimage.util import random_noise

        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # random.seed(int(time.time()))
        return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True) * 255
        # return random_noise(img, mode='gaussian', clip=True)

    # 调整亮度
    def _changeLight(self, img):
        flag = random.uniform(0.6, 1.3)
        blank = np.zeros(img.shape, img.dtype)
        alpha = beta = flag
        return cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)

    # cutout
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        '''
        原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : 框的坐标
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''

        def cal_iou(boxA, boxB):
            '''
            boxA, boxB为两个框，返回iou
            boxB为bouding box
            '''
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            # return the intersection over union value
            return iou

        # 得到h和w
        if img.ndim == 3:
            h, w, c = img.shape
        else:
            _, h, w, c = img.shape
        mask = np.ones((h, w, c), np.float32)
        for n in range(n_holes):
            chongdie = True  # 看切割的区域是否与box重叠太多
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0,
                             h)  # numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > threshold:
                        chongdie = True
                        break

            mask[y1: y2, x1: x2, :] = 0.

        # mask = np.expand_dims(mask, axis=0)
        img = img * mask

        return img

    # 旋转
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        # ---------------------- 旋转图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_img, rot_bboxes

    # 裁剪
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        # ---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min])

        return crop_img, crop_bboxes

    # 平移
    def _shift_pic_bboxes(self, img, bboxes):
        '''
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        d_to_left = x_min  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

        return shift_img, shift_bboxes

    # 镜像
    def _filp_pic_bboxes(self, img, bboxes):
        '''
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 翻转图像 ----------------------

        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:  # 0.5的概率水平翻转，0.5的概率垂直翻转
            horizon = True
        else:
            horizon = False
        h, w, _ = img.shape
        if horizon:  # 水平翻转
            flip_img = cv2.flip(flip_img, 1)  # 1是水平，-1是水平垂直
        else:
            flip_img = cv2.flip(flip_img, 0)

        # ---------------------- 调整boundingbox ----------------------
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
            else:
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])

        return flip_img, flip_bboxes

    # 图像增强方法
    def dataAugment(self, img, bboxes):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        change_num = 0  # 改变的次数
        # print('------')
        while change_num < 1:  # 默认至少有一种数据增强生效

            if self.is_rotate_img_bbox:
                if random.random() > self.rotation_rate:  # 旋转
                    # print('旋转')
                    change_num += 1
                    angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                    scale = random.uniform(0.7, 0.8)
                    img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)

            if self.is_shift_pic_bboxes:
                if random.random() < self.shift_rate:  # 平移
                    change_num += 1
                    img, bboxes = self._shift_pic_bboxes(img, bboxes)

            if self.is_changeLight:
                if random.random() > self.change_light_rate:  # 改变亮度
                    change_num += 1
                    img = self._changeLight(img)

            if self.is_addNoise:
                if random.random() < self.add_noise_rate:  # 加噪声
                    change_num += 1
                    img = self._addNoise(img)
            if self.is_cutout:
                if random.random() < self.cutout_rate:  # cutout
                    print('cutout')
                    change_num += 1
                    img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes,
                                       threshold=self.cut_out_threshold)
            if self.is_filp_pic_bboxes:
                if random.random() < self.flip_rate:  # 翻转
                    change_num += 1
                    img, bboxes = self._filp_pic_bboxes(img, bboxes)

        return img, bboxes


# xml解析工具
class ToolHelper():
    # 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    def parse_xml(self, path):
        import xml.etree.ElementTree as ET

        '''
        输入：
            xml_path: xml的文件路径
        输出：
            从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
        '''
        tree = ET.parse(path)
        root = tree.getroot()
        objs = root.findall('object')
        coords = list()
        for ix, obj in enumerate(objs):
            name = obj.find('name').text
            box = obj.find('bndbox')
            x_min = int(box[0].text)
            y_min = int(box[1].text)
            x_max = int(box[2].text)
            y_max = int(box[3].text)
            coords.append([x_min, y_min, x_max, y_max, name])
        return coords

    # 保存图片结果
    def save_img(self, file_name, save_folder, img):
        cv2.imwrite(os.path.join(save_folder, file_name), img)

    # 保持xml结果
    def save_xml(self, file_name, save_folder, img_info, height, width, channel, bboxs_info):
        from lxml import etree, objectify

        '''
        :param file_name:文件名
        :param save_folder:#保存的xml文件的结果
        :param height:图片的信息
        :param width:图片的宽度
        :param channel:通道
        :return:
        '''
        folder_name, img_name = img_info  # 得到图片的信息

        E = objectify.ElementMaker(annotate=False)

        anno_tree = E.annotation(
            E.folder(folder_name),
            E.filename(img_name),
            E.path(os.path.join(folder_name, img_name)),
            E.source(
                E.database('Unknown'),
            ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(channel)
            ),
            E.segmented(0),
        )

        labels, bboxs = bboxs_info  # 得到边框和标签信息
        for label, box in zip(labels, bboxs):
            anno_tree.append(
                E.object(
                    E.name(label),
                    E.pose('Unspecified'),
                    E.truncated('0'),
                    E.difficult('0'),
                    E.bndbox(
                        E.xmin(box[0]),
                        E.ymin(box[1]),
                        E.xmax(box[2]),
                        E.ymax(box[3])
                    )
                ))

        etree.ElementTree(anno_tree).write(os.path.join(save_folder, file_name), pretty_print=True)


def aug_det_dataset_with_xmls(img_path, xml_path):
    # source_pic_root_path = args.img_path
    # source_xml_root_path = args.xml_path

    save_pic_folder = os.path.join(os.path.abspath(os.path.join(img_path, '../../..')), 'Aug_JPEGImages')
    save_xml_folder = os.path.join(os.path.abspath(os.path.join(xml_path, '../../..')), 'Aug_Annotations')

    if not os.path.exists(save_pic_folder):
        os.mkdir(save_pic_folder)
    if not os.path.exists(save_xml_folder):
        os.mkdir(save_xml_folder)

    need_aug_num = 10  # 每张图片需要增强的次数

    is_endwidth_dot = True  # 文件是否以.jpg或者png结尾

    dataAug = DataAugmentForObjectDetection()  # 数据增强工具类

    toolhelper = ToolHelper()  # 工具

    for parent, _, files in os.walk(source_pic_root_path):
        for file in files:
            try:
                cnt = 0
                pic_path = os.path.join(parent, file)
                xml_path = os.path.join(source_xml_root_path, file[:-4] + '.xml')
                values = toolhelper.parse_xml(xml_path)  # 解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
                coords = [v[:4] for v in values]  # 得到框
                labels = [v[-1] for v in values]  # 对象的标签

                # 如果图片是有后缀的
                if is_endwidth_dot:
                    # 找到文件的最后名字
                    dot_index = file.rfind('.')
                    _file_prefix = file[:dot_index]  # 文件名的前缀
                    _file_suffix = file[dot_index:]  # 文件名的后缀
                img = cv2.imread(pic_path)

                # show_pic(img, coords)  # 显示原图
                while cnt < need_aug_num:  # 继续增强
                    auged_img, auged_bboxes = dataAug.dataAugment(img, coords)
                    auged_bboxes_int = np.array(auged_bboxes).astype(np.int32)
                    height, width, channel = auged_img.shape  # 得到图片的属性
                    img_name = '{}_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  # 图片保存的信息
                    toolhelper.save_img(img_name, save_pic_folder,
                                        auged_img)  # 保存增强图片

                    toolhelper.save_xml('{}_{}.xml'.format(_file_prefix, cnt + 1),
                                        save_xml_folder, (save_pic_folder, img_name), height, width, channel,
                                        (labels, auged_bboxes_int))  # 保存xml文件
                    # show_pic(auged_img, auged_bboxes)  # 强化后的图
                    cnt += 1  # 继续增强下一张
            except Exception as Error:
                print(Error)
    print("\n#################### Successful ######################\n")


# 读取出图像中的目标框
def read_xml_annotation(root, image_id):
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


# 将xml文件中的旧坐标值替换成新坐标值，并保存，这个程序里面没有使用
# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml_annotation(root, image_id, new_target):
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


# 仅仅是替换，并没有新建
def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
    import xml.etree.ElementTree as ET

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 读取原来的xml文件
    tree = ET.parse(in_file)  # 读取xml文件
    xmlroot = tree.getroot()
    index = 0
    # 将bbox中原来的坐标值换成新生成的坐标值
    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        # xmin = int(bndbox.find('xmin').text)
        # xmax = int(bndbox.find('xmax').text)
        # ymin = int(bndbox.find('ymin').text)
        # ymax = int(bndbox.find('ymax').text)

        # 注意new_target原本保存为高维数组
        ### 要是更换数据集的话这里需要改一下
        for i in range(4):
            if new_target[index][i] < 0:
                new_target[index][i] = 0
            if new_target[index][i] > 500:
                new_target[index][i] = 500

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str(image_id) + "_aug_" + str(id) + '.xml'))
    # tree.write(os.path.join(saveroot, str(image_id) + '.xml'))


def imgaug_aug_det_dataset_with_xmls(img_path, xml_path):
    import imgaug as ia

    ia.seed(1)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--img_path')
    # parser.add_argument('--xml_path')

    # args = parser.parse_args()
    # IMG_DIR = args.img_path
    # XML_DIR = args.xml_path

    # 存储增强后的影像文件夹路径
    AUG_IMG_DIR = os.path.join(os.path.abspath(os.path.join(img_path, '../../..')), 'Aug_JPEGImages_imgaug')
    AUG_XML_DIR = os.path.join(os.path.abspath(os.path.join(xml_path, '../../..')), 'Aug_Annotations_imgaug')
    os.makedirs(AUG_IMG_DIR, exist_ok=True)
    os.makedirs(AUG_XML_DIR, exist_ok=True)

    AUGLOOP = 10  # 每张影像增强的数量

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    sometimes = lambda aug: ia.augmenters.Sometimes(0.25, aug)
    seq = ia.augmenters.Sequential([
        ia.augmenters.Flipud(1),
        # sometimes(ia.augmenters.Multiply((0.7, 1.3))),
        sometimes(ia.augmenters.GaussianBlur(sigma=(0, 3.0))),
        sometimes(ia.augmenters.Cutout(nb_iterations=(1, 5), size=0.1, squared=False)),
        sometimes(ia.augmenters.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
            rotate=(-30, 30)
        ))
    ])

    # 得到当前运行的目录和目录当中的文件，其中sub_folders可以为空
    for root, sub_folders, files in os.walk(XML_DIR):
        # 遍历没一张图片
        for name in files:

            bndbox = read_xml_annotation(XML_DIR, name)

            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机

                # 读取图片
                img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                img = np.array(img)

                # bndbox 坐标增强，依次处理所有的bbox
                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                    ], shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    boxes_img_aug_list.append(bbs_aug)

                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    new_bndbox_list.append([int(bbs_aug.bounding_boxes[0].x1),
                                            int(bbs_aug.bounding_boxes[0].y1),
                                            int(bbs_aug.bounding_boxes[0].x2),
                                            int(bbs_aug.bounding_boxes[0].y2)])
                # 存储变化后的图片
                image_aug = seq_det.augment_images([img])[0]
                path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                # path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + '.jpg')
                # image_auged = bbs.draw_on_image(image_aug, thickness=0)
                Image.fromarray(image_aug).save(path)

                # 存储变化后的XML
                change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR, epoch)
                # print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                new_bndbox_list = []


# ========================================================================================================================================================================
# ========================================================================================================================================================================




# ========================================================================================================================================================================
# ========================================================================================================================================================================
# SEG
def convert_0_255_to_0_classes(image):
    """
    根据实际进行修改
    """
    image_to_write = np.zeros(image.shape)

    red = np.where((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 128))

    image_to_write[red] = (1, 1, 1)

    # yellow = np.where((image_to_write[:, :, 0] != 255) & (image_to_write[:, :, 1] != 255) & (image_to_write[:, :, 2] != 255))
    # image_to_write[yellow] = (0, 0, 0)
    #
    # green = np.where((image_to_write[:, :, 0] == 0) & (image_to_write[:, :, 1] == 128) & (image_to_write[:, :, 2] == 0))
    # image_to_write[green] = (3, 3, 3)

    return image_to_write


def convert_0_255_to_0_classes_1(image):
    """
    之前的版本生成的是3通道的mask图, 这个版本生成单通道的mask图
    Parameters
    ----------
    image

    Returns
    -------

    """
    image_to_write = np.zeros((image.shape[:2]), dtype=np.int32)
    red = np.where((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 128))
    image_to_write[red] = 1

    return image_to_write


def create_Camvid_trainval_txt(base_path):
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


class DataAugmentation:
    from PIL import ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageFile
    """
    包含数据增强的八种方式
    """

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, label, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

    # 暂时未使用这个函数
    @staticmethod
    def randomCrop(image, label):
        """
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
        :param image: PIL的图像image
        :return: 剪切之后的图像
        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_size = np.random.randint(40, 68)
        random_region = (
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
        return image.crop(random_region), label

    @staticmethod
    def randomColor(image, label):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label  # 调整图像锐度

    @staticmethod
    def randomGaussian(image, label, mean=0.3, sigma=0.5):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.3, sigma=0.5):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img = np.require(img, dtype='f4', requirements=['O', 'W'])
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label

    @staticmethod
    def saveImage(image, path):
        image.save(path)


def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception as e:
        print(str(e))
        return -2


def imageOps(func_name, image, label, img_des_path, label_des_path, img_file_name, label_file_name, times=5):
    funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomCrop": DataAugmentation.randomCrop,
               "randomColor": DataAugmentation.randomColor,
               "randomGaussian": DataAugmentation.randomGaussian
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1

    for _i in range(0, times, 1):
        new_image, new_label = funcMap[func_name](image, label)
        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, func_name + str(_i) + img_file_name))
        DataAugmentation.saveImage(new_label, os.path.join(label_des_path, func_name + str(_i) + label_file_name))


opsList = {"randomRotation", "randomColor", "randomGaussian"}


# opsList = {"randomGaussian"}

def threadOPS(img_path, new_img_path, label_path, new_label_path):
    """
    多线程处理事务
    :param src_path: 资源文件
    :param des_path: 目的地文件
    :return:
    """
    # img path
    if os.path.isdir(img_path):
        img_names = os.listdir(img_path)
    else:
        img_names = [img_path]

    # label path
    if os.path.isdir(label_path):
        label_names = os.listdir(label_path)
    else:
        label_names = [label_path]

    img_num = 0
    label_num = 0

    # img num
    for img_name in img_names:
        tmp_img_name = os.path.join(img_path, img_name)
        if os.path.isdir(tmp_img_name):
            print('contain file folder')
            exit()
        else:
            img_num = img_num + 1
    # label num
    for label_name in label_names:
        tmp_label_name = os.path.join(label_path, label_name)
        if os.path.isdir(tmp_label_name):
            print('contain file folder')
            exit()
        else:
            label_num = label_num + 1

    if img_num != label_num:
        print('the num of img and label is not equl')
        exit()
    else:
        num = img_num

    for i in range(num):
        img_name = img_names[i]
        print(img_name)
        label_name = label_names[i]
        print(label_name)

        tmp_img_name = os.path.join(img_path, img_name)
        tmp_label_name = os.path.join(label_path, label_name)

        # 读取文件并进行操作
        image = DataAugmentation.openImage(tmp_img_name)
        label = DataAugmentation.openImage(tmp_label_name)

        threadImage = [0] * 5
        _index = 0
        for ops_name in opsList:
            threadImage[_index] = threading.Thread(target=imageOps,
                                                   args=(ops_name, image, label, new_img_path, new_label_path, img_name, label_name))
            threadImage[_index].start()
            _index += 1
            time.sleep(0.2)


def aug_seg_dataset_with_masks(img_path, mask_path):
    """
    数据增强:
    1. 翻转变换 flip
    2. 随机修剪 random crop
    3. 色彩抖动 color jittering
    4. 平移变换 shift
    5. 尺度变换 scale
    6. 对比度变换 contrast
    7. 噪声扰动 noise
    8. 旋转变换/反射变换 Rotation/reflection
    """

    aug_save_img_path = os.path.join(os.path.abspath(os.path.join(img_path, '../../..')), 'aug_images')
    aug_save_mask_path = os.path.join(os.path.abspath(os.path.join(mask_path, '../../..')), 'aug_masks')

    threadOPS("{}".format(img_path), "{}".format(aug_save_img_path), "{}".format(mask_path), "{}".format(aug_save_mask_path))


class ImageAugmentation(object):
    def __init__(self, image_aug_dir, segmentationClass_aug_dir, image_start_num=1):
        self.image_aug_dir = image_aug_dir
        self.segmentationClass_aug_dir = segmentationClass_aug_dir
        self.image_start_num = image_start_num  # 增强后图片的起始编号
        self.seed_set()
        if not os.path.exists(self.image_aug_dir):
            os.mkdir(self.image_aug_dir)
        if not os.path.exists(self.segmentationClass_aug_dir):
            os.mkdir(self.segmentationClass_aug_dir)

    def seed_set(self, seed=1):
        import imgaug as ia

        np.random.seed(seed)
        random.seed(seed)
        ia.seed(seed)

    def array2p_mode(self, alpha_channel):
        """alpha_channel is a binary image."""
        # assert set(alpha_channel.flatten().tolist()) == {0, 1}, "alpha_channel is a binary image."
        alpha_channel[alpha_channel == 1] = 128
        h, w = alpha_channel.shape
        image_arr = np.zeros((h, w, 3))
        image_arr[:, :, 0] = alpha_channel
        img = Image.fromarray(np.uint8(image_arr))
        img_p = img.convert("P")
        return img_p

    def augmentor(self, image):
        # height, width, _ = image.shape
        height, width = image.shape
        resize = ia.augmenters.Sequential([
            ia.augmenters.Resize({"height": int(height / 2), "width": int(width / 2)}),
        ])  # 缩放

        fliplr_flipud = ia.augmenters.Sequential([
            ia.augmenters.Fliplr(),
            ia.augmenters.Flipud(),
        ])  # 左右+上下翻转

        rotate = ia.augmenters.Sequential([
            ia.augmenters.Affine(rotate=(-15, 15))
        ])  # 旋转

        translate = ia.augmenters.Sequential([
            ia.augmenters.Affine(translate_percent=(0.2, 0.35))
        ])  # 平移

        crop_and_pad = ia.augmenters.Sequential([
            ia.augmenters.CropAndPad(percent=(-0.25, 0), keep_size=False),
        ])  # 裁剪

        rotate_and_crop = ia.augmenters.Sequential([
            ia.augmenters.Affine(rotate=15),
            ia.augmenters.CropAndPad(percent=(-0.25, 0), keep_size=False)
        ])  # 旋转 + 裁剪

        guassian_blur = ia.augmenters.Sequential([
            ia.augmenters.GaussianBlur(sigma=(2, 3)),
        ])  # 增加高斯噪声

        ops = [resize, fliplr_flipud, rotate, translate, crop_and_pad, rotate_and_crop, guassian_blur]
        #        缩放、   镜像+上下翻转、   旋转、    xy平移、      裁剪、        旋转 + 裁剪、   高斯平滑
        return ops

    def augment_img(self, image_name, segmap_name):
        from imgaug.augmentables.segmaps import SegmentationMapsOnImage

        # 1.Load an image.
        image = Image.open(image_name)  # RGB
        segmap = Image.open(segmap_name)  # P

        image_name = os.path.basename(image_name).split(".")[0]

        name = f"{self.image_start_num:04d}"
        image.save(self.image_aug_dir + "\\{}_{}.jpg".format(image_name, name))
        segmap.save(self.segmentationClass_aug_dir + "\\{}_{}.png".format(image_name, name))
        self.image_start_num += 1

        image = np.array(image)
        segmap = SegmentationMapsOnImage(np.array(segmap), shape=image.shape)

        # 2. define the ops
        ops = self.augmentor(image)

        # 3.execute ths ops
        for _, op in enumerate(ops):
            name = f"{self.image_start_num:04d}"
            print(f"当前增强了{self.image_start_num:04d}张数据...")
            images_aug_i, segmaps_aug_i = op(image=image, segmentation_maps=segmap)
            images_aug_i = Image.fromarray(images_aug_i)
            images_aug_i.save(self.image_aug_dir + "\\{}_{}.jpg".format(image_name, name))
            segmaps_aug_i_ = segmaps_aug_i.get_arr()
            segmaps_aug_i_[segmaps_aug_i_ > 0] = 1
            segmaps_aug_i_ = self.array2p_mode(segmaps_aug_i_)
            segmaps_aug_i_.save(self.segmentationClass_aug_dir + "\\{}_{}.png".format(image_name, name))
            self.image_start_num += 1

    def augment_images(self, image_dir, segmap_dir):
        # image_names = sorted(glob.glob(image_dir + "*"))
        # segmap_names = sorted(glob.glob(segmap_dir + "*"))
        image_names = sorted(os.listdir(image_dir))
        segmap_names = sorted(os.listdir(segmap_dir))
        image_names_, segmap_names_ = [], []
        for img in image_names:
            image_names_.append(image_dir + "\\" + img)
        for jsv in segmap_names:
            segmap_names_.append(segmap_dir + "\\" + jsv)

        image_num = len(image_names)
        count = 1
        for image_name, segmap_name in zip(image_names_, segmap_names_):
            print("*" * 30, f"正在增强第【{count:04d}/{image_num:04d}】张图片...", "*" * 30)
            self.augment_img(image_name, segmap_name)
            count += 1


def imgaug_aug_seg_dataset_with_masks(img_path, mask_path):
    # args = parser.parse_args()
    # IMG_DIR = args.img_path
    # JSONVIS_DIR = args.jsonvis_path

    # 存储增强后的影像文件夹路径
    AUG_IMG_DIR = os.path.join(os.path.abspath(os.path.join(img_path, '../../..')), 'aug_images_imgaug')
    AUG_JSONVIS_DIR = os.path.join(os.path.abspath(os.path.join(mask_path, '../../..')), 'aug_masks_imgaug')
    os.makedirs(AUG_IMG_DIR, exist_ok=True)
    os.makedirs(AUG_JSONVIS_DIR, exist_ok=True)

    image_start_num = 1
    image_augmentation = ImageAugmentation(AUG_IMG_DIR, AUG_JSONVIS_DIR, image_start_num)
    image_augmentation.augment_images(IMG_DIR, JSONVIS_DIR)

# ========================================================================================================================================================================
# ========================================================================================================================================================================




# ========================================================================================================================================================================
# ========================================================================================================================================================================
# KPT
def warpPerspective_img_via_labelbee_kpt_json(data_path):
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
            cv2img = cv2.imread(img_abs_path)

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
            cropped_base = cv2img[y1:y2, x1:x2]
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
                    warpped = cv2.warpPerspective(cv2img, M, (h // 2, w // 2))
                    cv2.imwrite("{}/{}_{}.jpg".format(save_path, fname, ki), warpped)
                else:
                    src_points = np.float32([[kpts[ki][0], kpts[ki][1]], [kpts[ki][2], kpts[ki][3]], [kpts[ki][6], kpts[ki][7]], [kpts[ki][4], kpts[ki][5]]])
                    dst_points = np.float32([[0, 0], [w // 2, 0], [0, h // 2], [w // 2, h // 2]])
                    M = cv2.getPerspectiveTransform(src_points, dst_points)
                    warpped = cv2.warpPerspective(cv2img, M, (w // 2, h // 2))
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
            cv2img = cv2.imread(img_abs_path)
            imgsz = cv2img.shape[:2]

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
            json_labelme["imageData"] = labelme.utils.img_arr_to_b64(cv2img).strip()
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
            cv2img = cv2.imread(img_abs_path)
            imgsz = cv2img.shape[:2]

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
                cropped_base = cv2img[y1:y2, x1:x2]
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
                    cropped = cv2img[iy1:iy2, ix1:ix2]
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
        if (pts[i][0] - expandP >= 0):
            if (i == 0 or i == 3):
                pts[i][0] = pts[i][0] - expandP
            else:
                pts[i][0] = pts[i][0] + expandP
        elif (pts[i][0] - expandP_half >= 0):
            if (i == 0 or i == 3):
                pts[i][0] = pts[i][0] - expandP_half
            else:
                pts[i][0] = pts[i][0] + expandP_half
        elif (pts[i][0] - expandP_quarter >= 0):
            if (i == 0 or i == 3):
                pts[i][0] = pts[i][0] - expandP_quarter
            else:
                pts[i][0] = pts[i][0] + expandP_quarter
        elif (pts[i][0] - expandP_one_sixth >= 0):
            if (i == 0 or i == 3):
                pts[i][0] = pts[i][0] - expandP_one_sixth
            else:
                pts[i][0] = pts[i][0] + expandP_one_sixth
        elif (pts[i][0] - expandP_one_eighth >= 0):
            if (i == 0 or i == 3):
                pts[i][0] = pts[i][0] - expandP_one_eighth
            else:
                pts[i][0] = pts[i][0] + expandP_one_eighth
        else:
            pts[i][0] = pts[i][0]

        if (pts[i][1] - expandP >= 0):
            if (i == 0 or i == 1):
                pts[i][1] = pts[i][1] - expandP
            else:
                pts[i][1] = pts[i][1] + expandP
        elif (pts[i][1] - expandP_half >= 0):
            if (i == 0 or i == 1):
                pts[i][1] = pts[i][1] - expandP_half
            else:
                pts[i][1] = pts[i][1] + expandP_half
        elif (pts[i][1] - expandP_quarter >= 0):
            if (i == 0 or i == 1):
                pts[i][1] = pts[i][1] - expandP_quarter
            else:
                pts[i][1] = pts[i][1] + expandP_quarter
        elif (pts[i][1] - expandP_one_sixth >= 0):
            if (i == 0 or i == 1):
                pts[i][1] = pts[i][1] - expandP_one_sixth
            else:
                pts[i][1] = pts[i][1] + expandP_one_sixth
        elif (pts[i][1] - expandP_one_eighth >= 0):
            if (i == 0 or i == 1):
                pts[i][1] = pts[i][1] - expandP_one_eighth
            else:
                pts[i][1] = pts[i][1] + expandP_one_eighth
        else:
            pts[i][1] = pts[i][1]

    return pts


def doPerspectiveTransformByLabelmeJson(data_path, r=0.10):
    save_path = data_path + "/images_perspective_transform"
    os.makedirs(save_path, exist_ok=True)

    img_path = data_path + "/images"
    json_path = data_path + "/jsons"

    file_list = get_file_list(img_path)
    for f in tqdm(file_list):
        fname = os.path.splitext(f)[0]
        img_abs_path = img_path + "/{}".format(f)
        json_abs_path = json_path + "/{}.json".format(fname)

        cv2img = cv2.imread(img_abs_path)
        imgsz = cv2img.shape[:2]

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
        warped = cv2.warpPerspective(cv2img, M, (dstW, dstH))
        cv2.imwrite("{}/{}".format(save_path, f), warped)


def expand_kpt_(kpt, imgsz, r):
    minSide = min(imgsz[0], imgsz[1])
    expandP = round(minSide * r)
    expandP_half = round(minSide * r / 2)
    expandP_quarter = round(minSide * r / 4)
    expandP_one_sixth = round(minSide * r / 6)
    expandP_one_eighth = round(minSide * r / 8)

    for i in range(len(kpt)):
        if (kpt[i][0] - expandP) >= 0:
            if i == 0 or i == 3:
                kpt[i][0] = kpt[i][0] - expandP
            else:
                kpt[i][0] = kpt[i][0] + expandP
        elif (kpt[i][0] - expandP_half) >= 0:
            if i == 0 or i == 3:
                kpt[i][0] = kpt[i][0] - expandP_half
            else:
                kpt[i][0] = kpt[i][0] + expandP_half
        elif (kpt[i][0] - expandP_quarter) >= 0:
            if i == 0 or i == 3:
                kpt[i][0] = kpt[i][0] - expandP_quarter
            else:
                kpt[i][0] = kpt[i][0] + expandP_quarter
        elif (kpt[i][0] - expandP_one_sixth) >= 0:
            if i == 0 or i == 3:
                kpt[i][0] = kpt[i][0] - expandP_one_sixth
            else:
                kpt[i][0] = kpt[i][0] + expandP_one_sixth
        elif (kpt[i][0] - expandP_one_eighth) >= 0:
            if i == 0 or i == 3:
                kpt[i][0] = kpt[i][0] - expandP_one_eighth
            else:
                kpt[i][0] = kpt[i][0] + expandP_one_eighth
        else:
            kpt[i][0] = kpt[i][0]

        if (kpt[i][1] - expandP) >= 0:
            if i == 0 or i == 1:
                kpt[i][1] = kpt[i][1] - expandP
            else:
                kpt[i][1] = kpt[i][1] + expandP
        elif (kpt[i][1] - expandP_half) >= 0:
            if i == 0 or i == 1:
                kpt[i][1] = kpt[i][1] - expandP_half
            else:
                kpt[i][1] = kpt[i][1] + expandP_half
        elif (kpt[i][1] - expandP_quarter) >= 0:
            if i == 0 or i == 1:
                kpt[i][1] = kpt[i][1] - expandP_quarter
            else:
                kpt[i][1] = kpt[i][1] + expandP_quarter
        elif (kpt[i][1] - expandP_one_sixth) >= 0:
            if i == 0 or i == 1:
                kpt[i][1] = kpt[i][1] - expandP_one_sixth
            else:
                kpt[i][1] = kpt[i][1] + expandP_one_sixth
        elif (kpt[i][1] - expandP_one_eighth) >= 0:
            if i == 0 or i == 1:
                kpt[i][1] = kpt[i][1] - expandP_one_eighth
            else:
                kpt[i][1] = kpt[i][1] + expandP_one_eighth
        else:
            kpt[i][1] = kpt[i][1]

    return kpt
# ========================================================================================================================================================================
# ========================================================================================================================================================================




# ========================================================================================================================================================================
# ========================================================================================================================================================================
# OCR
def gen_dbnet_torch_train_test_txt(data_path, data_type="test"):
    img_path = data_path + "/{}images".format(data_type)
    gt_path = data_path + "/{}gts".format(data_type)

    img_list = sorted(os.listdir(img_path))
    gt_list = sorted(os.listdir(gt_path))

    # same_list = list(set(img_list) & set(gt_list))

    with open(data_path + "/{}images_list.txt".format(data_type), "w", encoding="utf-8") as fw:
        for s in tqdm(img_list):
            sname = os.path.splitext(s)[0]
            s_img_path = img_path + "/{}".format(s)
            s_gt_path = gt_path + "/{}.gt".format(sname)

            # s_img_path = "/{}".format(sname)
            # s_gt_path = "/{}.gt".format(sname)
            l = "{}\t{}\n".format(s_img_path, s_gt_path)
            fw.write(l)


def dbnet_aug_data(data_path, bg_path, maxnum=20000):
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

                cv2img = cv2.imread(img_abs_path)
                cv2imgsz = cv2img.shape[:2]
                maskimg = cv2.imread(mask_abs_path)

                rdmnum = np.random.random()
                if cv2imgsz[0] > 3000 and cv2imgsz[1] > 3000:
                    if rdmnum < 0.25:
                        cv2img = cv2.resize(cv2img, (cv2imgsz[1] // 2, cv2imgsz[0] // 2))
                        maskimg = cv2.resize(maskimg, (cv2imgsz[1] // 2, cv2imgsz[0] // 2))
                    elif rdmnum > 0.75:
                        cv2img = cv2.resize(cv2img, (cv2imgsz[1] // 4, cv2imgsz[0] // 4))
                        maskimg = cv2.resize(maskimg, (cv2imgsz[1] // 4, cv2imgsz[0] // 4))
                else:
                    if rdmnum < 0.45:
                        cv2img = cv2.resize(cv2img, (cv2imgsz[1] // 2, cv2imgsz[0] // 2))
                        maskimg = cv2.resize(maskimg, (cv2imgsz[1] // 2, cv2imgsz[0] // 2))

                outimg_crop, bbox, relative_roi = seg_crop_object(cv2img, bgimg, maskimg)

                with open(gt_abs_path, "r", encoding="utf-8") as fo:
                    lines = fo.readlines()
                    assert len(lines) == 1, "{}: lines > 1!".format(gt_abs_path)
                    for line in lines:
                        # line = line.strip().split(", ")[:8]
                        # line = list(map(float, line))
                        # relative_points_x = np.array(line[::2]) - bbox[0]
                        # relative_points_y = np.array(line[1::2]) - bbox[1]

                        if cv2imgsz[0] > 3000 and cv2imgsz[1] > 3000:
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

        cv2img = cv2.imread(img_abs_path)

        with open(gt_abs_path, "r", encoding="utf-8") as fo:
            lines = fo.readlines()
            for line in lines:
                line = line.strip().split(", ")[:8]
                line = list(map(int, map(round, map(float, line))))
                for j in range(0, 8, 2):
                    cv2.circle(cv2img, (line[j], line[j + 1]), 4, (255, 0, 255), 2)

        cv2.imwrite("{}/{}.jpg".format(vis_path, gt_name), cv2img)


def crop_ocr_rec_img_according_labelbee_det_json(data_path):
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
        cv2img = cv2.imread(img_abs_path)
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
                cropped_img0 = cv2img[y_min:y_max, x_min:x_max]
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
                cropped_img1 = cv2img[y_min - np.random.randint(0, 4):y_max + np.random.randint(0, 4), x_min - np.random.randint(0, 4):x_max + np.random.randint(0, 4)]
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
                cropped_img2 = cv2img[y_min - np.random.randint(0, 4):y_max - np.random.randint(0, 4), x_min - np.random.randint(0, 4):x_max - np.random.randint(0, 4)]
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
                cropped_img3 = cv2img[y_min + np.random.randint(0, 4):y_max - np.random.randint(0, 4), x_min + np.random.randint(0, 4):x_max - np.random.randint(0, 4)]
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
                cropped_img4 = cv2img[y_min + np.random.randint(0, 4):y_max + np.random.randint(0, 4), x_min + np.random.randint(0, 4):x_max + np.random.randint(0, 4)]
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

            # try:
            #     cropped_img0 = cv2img[y_min:y_max, x_min:x_max]
            #     cropped_img1 = cv2img[y_min - np.random.randint(0, 4):y_max + np.random.randint(0, 4), x_min - np.random.randint(0, 4):x_max + np.random.randint(0, 4)]
            #     cropped_img2 = cv2img[y_min - np.random.randint(0, 4):y_max - np.random.randint(0, 4), x_min - np.random.randint(0, 4):x_max - np.random.randint(0, 4)]
            #     cropped_img3 = cv2img[y_min + np.random.randint(0, 4):y_max - np.random.randint(0, 4), x_min + np.random.randint(0, 4):x_max - np.random.randint(0, 4)]
            #     cropped_img4 = cv2img[y_min + np.random.randint(0, 4):y_max + np.random.randint(0, 4), x_min + np.random.randint(0, 4):x_max + np.random.randint(0, 4)]
            #
            #     if "A" in label or "b" in label or "C" in label:
            #         rdm_w = np.random.randint(55, 76)
            #         alpha0 = cropped_img0[0:cropped_img0.shape[0], 0:rdm_w]
            #         digital0 = cropped_img0[0:cropped_img0.shape[0], rdm_w:cropped_img0.shape[1]]
            #         alpha0_label = label[0]
            #         digital0_label = label[1:]
            #         cv2.imwrite("{}/{}_{}_{}_alpha={}.jpg".format(cropped_path, img_name, i, 0, alpha0_label), alpha0)
            #         cv2.imwrite("{}/{}_{}_{}_digital={}.jpg".format(cropped_path, img_name, i, 0, digital0_label), digital0)
            #
            #     cv2.imwrite("{}/{}_{}_{}={}.jpg".format(cropped_path, img_name, i, 0, label), cropped_img0)
            #     cv2.imwrite("{}/{}_{}_{}={}.jpg".format(cropped_path, img_name, i, 1, label), cropped_img1)
            #     cv2.imwrite("{}/{}_{}_{}={}.jpg".format(cropped_path, img_name, i, 2, label), cropped_img2)
            #     cv2.imwrite("{}/{}_{}_{}={}.jpg".format(cropped_path, img_name, i, 3, label), cropped_img3)
            #     cv2.imwrite("{}/{}_{}_{}={}.jpg".format(cropped_path, img_name, i, 4, label), cropped_img4)
            # except Exception as Error:
            #     print(Error)


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


def to_unicode(glyph):
    return json.loads(f'"{glyph}"')


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

        char = to_unicode(glyph)
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

        # 如果下个字符超出图像，则退出
        if x_shift + 1.5 * char_w > imgsz[1]:
            break

    return text[:i]


def gen_img(imgsz=(64, 128), font=None, alpha="0123456789.AbC", target_len=1):
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


def main_010_aug_test(save_path):
    os.makedirs(save_path, exist_ok=True)

    # img_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/dbnet_PS/output_warp_test/VID_20230726_155204_0000409_ps_0.jpg"
    # cv2img = cv2.imread(img_path)
    # imgsz = cv2img.shape[:2]
    #
    # mask = np.zeros(shape=cv2img.shape, dtype=np.uint8)
    #
    # # for i in range(99 - 13, 204 - 13):
    # #     for j in range(347 - 95, 560 - 95):
    # #         mask[j, i] = (255, 255, 255)
    #
    # # for i in range(261 - 13, 368 - 13):
    # #     for j in range(347 - 95, 560 - 95):
    # #         mask[j, i] = (255, 255, 255)
    # #
    # # for i in range(426 - 13, 534 - 13):
    # #     for j in range(347 - 95, 560 - 95):
    # #         mask[j, i] = (255, 255, 255)
    # #
    # # for i in range(587 - 13, 695 - 13):
    # #     for j in range(347 - 95, 560 - 95):
    # #         mask[j, i] = (255, 255, 255)
    # #
    # for i in range(752 - 13, 858 - 13):
    #     for j in range(305 - 95, 615 - 95):
    #         mask[j, i] = (255, 255, 255)
    #
    # cv2.imwrite("/home/zengyifan/wujiahu/data/010.Digital_Rec/others/dbnet_PS/output_warp_test/VID_20230726_155204_0000409_ps_0_mask_5.jpg", mask)

    # ==============================================================================================================================================
    bg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/dbnet_PS/New Folder/VID_20230726_155204_0000409_ps_0.jpg"
    fg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/llj_0-9/llj_0-9_new_THR_INV"
    fg_list = sorted(os.listdir(fg_path))

    last_number_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/llj_0-9/0-9_output_ud_stack"
    last_list = sorted(os.listdir(last_number_path))

    bg_img = cv2.imread(bg_path)
    bgsz = bg_img.shape[:2]

    label_str = ""
    for i in range(4):
        fgi = random.sample(fg_list, 1)[0]
        fgi_name = os.path.splitext(fgi)[0]
        label_str += fgi_name

        fg_abs_path = fg_path + "/{}".format(fgi)
        fg_img = cv2.imread(fg_abs_path)
        fg_img = cv2.resize(fg_img, (204 - 99, 560 - 347))
        # fgsz = fg_img.shape[:2]

        mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
        out = cv2.seamlessClone(fg_img, bg_img, mask_img, (99 + (204 - 99) // 2 - 13 + 163 * i, 347 + (560 - 347) // 2 - 95), cv2.MIXED_CLONE)
        bg_img = out

    lasti = random.sample(last_list, 1)[0]
    lasti_name = os.path.splitext(lasti)[0]
    lasti_label = lasti_name.split("=")[1]
    label_str += "."
    label_str += lasti_label

    lasti_abs_path = last_number_path + "/{}".format(lasti)
    lasti_img = cv2.imread(lasti_abs_path)
    lasti_img = cv2.resize(lasti_img, (858 - 752, 615 - 310))
    mask_img = 255 * np.ones(shape=lasti_img.shape, dtype=np.uint8)
    out = cv2.seamlessClone(lasti_img, bg_img, mask_img, (752 + (858 - 752) // 2 - 13, 310 + (615 - 310) // 2 - 95), cv2.MIXED_CLONE)

    cv2.imwrite("{}/llj_stack_20230918={}.jpg".format(save_path, label_str), out)


def main_010_aug_test_AbC(save_path):
    os.makedirs(save_path, exist_ok=True)

    # ==============================================================================================================================================
    bg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/AbC/bg/1.jpg"
    fg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/AbC/0-9_AbC"
    fg_list = sorted(os.listdir(fg_path))

    bg_img = cv2.imread(bg_path)
    bgsz = bg_img.shape[:2]

    # for fgi in fg_list:
    fgi = random.sample(fg_list, 1)[0]
    label_str = ""
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
    else:
        print("Error!")

    fg_abs_path = fg_path + "/{}".format(fgi)
    fg_img = cv2.imread(fg_abs_path)
    fg_img = cv2.resize(fg_img, (54, 72))
    fgsz = fg_img.shape[:2]

    rdm = random.random()

    mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
    out = cv2.seamlessClone(fg_img, bg_img, mask_img, (bgsz[1] // 2, bgsz[0] // 2), cv2.MIXED_CLONE)
    # bg_img = out

    cv2.imwrite("{}/20230905_{}_{}={}.jpg".format(save_path, str(rdm).replace(".", ""), fgi_name, label_str), out)

    # # ==============================================================================================================================================
    # bg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/AbC/bg/2.jpg"
    # fg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/AbC/0-9_AbC"
    # fg_list = sorted(os.listdir(fg_path))
    #
    # bg_img = cv2.imread(bg_path)
    # bgsz = bg_img.shape[:2]
    #
    # rdm = random.random()
    #
    # label_str = ""
    # for i in range(4):
    #     fgi = random.sample(fg_list, 1)[0]
    #     fgi_name = os.path.splitext(fgi)[0]
    #     if "AN" in fgi_name:
    #         label_str += "A"
    #     elif "bN" in fgi_name:
    #         label_str += "b"
    #     elif "CN" in fgi_name:
    #         label_str += "C"
    #     elif "0N" in fgi_name:
    #         label_str += "0"
    #     elif "1N" in fgi_name:
    #         label_str += "1"
    #     elif "2N" in fgi_name:
    #         label_str += "2"
    #     elif "3N" in fgi_name:
    #         label_str += "3"
    #     elif "4N" in fgi_name:
    #         label_str += "4"
    #     elif "5N" in fgi_name:
    #         label_str += "5"
    #     elif "6N" in fgi_name:
    #         label_str += "6"
    #     elif "7N" in fgi_name:
    #         label_str += "7"
    #     elif "8N" in fgi_name:
    #         label_str += "8"
    #     elif "9N" in fgi_name:
    #         label_str += "9"
    #     elif "0.N" in fgi_name:
    #         label_str += "0."
    #     elif "1.N" in fgi_name:
    #         label_str += "1."
    #     elif "2.N" in fgi_name:
    #         label_str += "2."
    #     elif "3.N" in fgi_name:
    #         label_str += "3."
    #     elif "4.N" in fgi_name:
    #         label_str += "4."
    #     elif "5.N" in fgi_name:
    #         label_str += "5."
    #     elif "6.N" in fgi_name:
    #         label_str += "6."
    #     elif "7.N" in fgi_name:
    #         label_str += "7."
    #     elif "8.N" in fgi_name:
    #         label_str += "8."
    #     elif "9.N" in fgi_name:
    #         label_str += "9."
    #     else:
    #         print("Error!")
    #     # label_str += fgi_name
    #
    #     fg_abs_path = fg_path + "/{}".format(fgi)
    #     fg_img = cv2.imread(fg_abs_path)
    #     fg_img = cv2.resize(fg_img, (48, 70))
    #     fgsz = fg_img.shape[:2]
    #
    #     mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
    #     out = cv2.seamlessClone(fg_img, bg_img, mask_img, (30 + 56 * i, 36), cv2.MIXED_CLONE)
    #     bg_img = out
    #
    # cv2.imwrite("{}/20230905_{}_{}={}.jpg".format(save_path, str(rdm).replace(".", ""), fgi_name, label_str), out)

    # # ==============================================================================================================================================
    # bg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/AbC/bg/3.jpg"
    # fg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/AbC/0-9_AbC"
    # fg_list = sorted(os.listdir(fg_path))
    #
    # bg_img = cv2.imread(bg_path)
    # bgsz = bg_img.shape[:2]
    #
    # rdm = random.random()
    #
    # label_str = ""
    # for i in range(5):
    #     fgi = random.sample(fg_list, 1)[0]
    #     fgi_name = os.path.splitext(fgi)[0]
    #     if "AN" in fgi_name:
    #         label_str += "A"
    #     elif "bN" in fgi_name:
    #         label_str += "b"
    #     elif "CN" in fgi_name:
    #         label_str += "C"
    #     elif "0N" in fgi_name:
    #         label_str += "0"
    #     elif "1N" in fgi_name:
    #         label_str += "1"
    #     elif "2N" in fgi_name:
    #         label_str += "2"
    #     elif "3N" in fgi_name:
    #         label_str += "3"
    #     elif "4N" in fgi_name:
    #         label_str += "4"
    #     elif "5N" in fgi_name:
    #         label_str += "5"
    #     elif "6N" in fgi_name:
    #         label_str += "6"
    #     elif "7N" in fgi_name:
    #         label_str += "7"
    #     elif "8N" in fgi_name:
    #         label_str += "8"
    #     elif "9N" in fgi_name:
    #         label_str += "9"
    #     elif "0.N" in fgi_name:
    #         label_str += "0."
    #     elif "1.N" in fgi_name:
    #         label_str += "1."
    #     elif "2.N" in fgi_name:
    #         label_str += "2."
    #     elif "3.N" in fgi_name:
    #         label_str += "3."
    #     elif "4.N" in fgi_name:
    #         label_str += "4."
    #     elif "5.N" in fgi_name:
    #         label_str += "5."
    #     elif "6.N" in fgi_name:
    #         label_str += "6."
    #     elif "7.N" in fgi_name:
    #         label_str += "7."
    #     elif "8.N" in fgi_name:
    #         label_str += "8."
    #     elif "9.N" in fgi_name:
    #         label_str += "9."
    #     else:
    #         print("Error!")
    #     # label_str += fgi_name
    #
    #     fg_abs_path = fg_path + "/{}".format(fgi)
    #     fg_img = cv2.imread(fg_abs_path)
    #     fg_img = cv2.resize(fg_img, (48, 70))
    #     fgsz = fg_img.shape[:2]
    #
    #     mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
    #
    #     if i == 0:
    #         out = cv2.seamlessClone(fg_img, bg_img, mask_img, (30, 36), cv2.MIXED_CLONE)
    #     else:
    #         out = cv2.seamlessClone(fg_img, bg_img, mask_img, (68 + 30 + 56 * (i - 1), 36), cv2.MIXED_CLONE)
    #     bg_img = out
    #
    # cv2.imwrite("{}/20230905_{}_{}={}.jpg".format(save_path, str(rdm).replace(".", ""), fgi_name, label_str), out)


def main_010_aug_test_AbC_v2(save_path):
    os.makedirs(save_path, exist_ok=True)

    # # ==============================================================================================================================================
    # bg_path = "/home/wujiahu/code/gen_fake/gen_AbC/bg/2.jpg"
    # fg_path = "/home/wujiahu/code/gen_fake/gen_AbC/0-9_AbC_new"
    # fg_list = sorted(os.listdir(fg_path))
    #
    # bg_img = cv2.imread(bg_path)
    # bgsz = bg_img.shape[:2]
    #
    # # for fgi in fg_list:
    # fgi = random.sample(fg_list, 1)[0]
    # label_str = ""
    # fgi_name = os.path.splitext(fgi)[0]
    # if "AN" in fgi_name:
    #     label_str += "A"
    # elif "bN" in fgi_name:
    #     label_str += "b"
    # elif "CN" in fgi_name:
    #     label_str += "C"
    # elif "0N" in fgi_name:
    #     label_str += "0"
    # elif "1N" in fgi_name:
    #     label_str += "1"
    # elif "2N" in fgi_name:
    #     label_str += "2"
    # elif "3N" in fgi_name:
    #     label_str += "3"
    # elif "4N" in fgi_name:
    #     label_str += "4"
    # elif "5N" in fgi_name:
    #     label_str += "5"
    # elif "6N" in fgi_name:
    #     label_str += "6"
    # elif "7N" in fgi_name:
    #     label_str += "7"
    # elif "8N" in fgi_name:
    #     label_str += "8"
    # elif "9N" in fgi_name:
    #     label_str += "9"
    # elif "0.N" in fgi_name:
    #     label_str += "0."
    # elif "1.N" in fgi_name:
    #     label_str += "1."
    # elif "2.N" in fgi_name:
    #     label_str += "2."
    # elif "3.N" in fgi_name:
    #     label_str += "3."
    # elif "4.N" in fgi_name:
    #     label_str += "4."
    # elif "5.N" in fgi_name:
    #     label_str += "5."
    # elif "6.N" in fgi_name:
    #     label_str += "6."
    # elif "7.N" in fgi_name:
    #     label_str += "7."
    # elif "8.N" in fgi_name:
    #     label_str += "8."
    # elif "9.N" in fgi_name:
    #     label_str += "9."
    # elif "space" in fgi_name:
    #     label_str += ""
    # else:
    #     print("Error!")
    #
    # fg_abs_path = fg_path + "/{}".format(fgi)
    # fg_img = cv2.imread(fg_abs_path)
    # fg_img = cv2.resize(fg_img, (54, 72))
    # fgsz = fg_img.shape[:2]
    #
    # rdm = random.random()
    #
    # mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
    # out = cv2.seamlessClone(fg_img, bg_img, mask_img, (bgsz[1] // 2, bgsz[0] // 2), cv2.MIXED_CLONE)
    # # bg_img = out
    #
    # cv2.imwrite("{}/20230905_{}_{}={}.jpg".format(save_path, str(rdm).replace(".", ""), fgi_name, label_str), out)

    # ==============================================================================================================================================
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
        # label_str += fgi_name

        fg_abs_path = fg_path + "/{}".format(fgi)
        fg_img = cv2.imread(fg_abs_path)
        fg_img = cv2.resize(fg_img, (48, 70))
        fgsz = fg_img.shape[:2]

        mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
        out = cv2.seamlessClone(fg_img, bg_img, mask_img, (30 + 56 * i, 36), cv2.MIXED_CLONE)
        bg_img = out

    cv2.imwrite("{}/20231008_{}_{}={}.jpg".format(save_path, str(rdm).replace(".", ""), fgi_name, label_str), out)

    # # ==============================================================================================================================================
    # bg_path = "/home/wujiahu/code/gen_fake/gen_AbC/bg/2.jpg"
    # fg_path = "/home/wujiahu/code/gen_fake/gen_AbC/0-9_AbC_new"
    # fg_list = sorted(os.listdir(fg_path))
    #
    # bg_img = cv2.imread(bg_path)
    # bgsz = bg_img.shape[:2]
    #
    # rdm = random.random()
    #
    # label_str = ""
    # for i in range(5):
    #     fgi = random.sample(fg_list, 1)[0]
    #     fgi_name = os.path.splitext(fgi)[0]
    #     if "AN" in fgi_name:
    #         label_str += "A"
    #     elif "bN" in fgi_name:
    #         label_str += "b"
    #     elif "CN" in fgi_name:
    #         label_str += "C"
    #     elif "0N" in fgi_name:
    #         label_str += "0"
    #     elif "1N" in fgi_name:
    #         label_str += "1"
    #     elif "2N" in fgi_name:
    #         label_str += "2"
    #     elif "3N" in fgi_name:
    #         label_str += "3"
    #     elif "4N" in fgi_name:
    #         label_str += "4"
    #     elif "5N" in fgi_name:
    #         label_str += "5"
    #     elif "6N" in fgi_name:
    #         label_str += "6"
    #     elif "7N" in fgi_name:
    #         label_str += "7"
    #     elif "8N" in fgi_name:
    #         label_str += "8"
    #     elif "9N" in fgi_name:
    #         label_str += "9"
    #     elif "0.N" in fgi_name:
    #         label_str += "0."
    #     elif "1.N" in fgi_name:
    #         label_str += "1."
    #     elif "2.N" in fgi_name:
    #         label_str += "2."
    #     elif "3.N" in fgi_name:
    #         label_str += "3."
    #     elif "4.N" in fgi_name:
    #         label_str += "4."
    #     elif "5.N" in fgi_name:
    #         label_str += "5."
    #     elif "6.N" in fgi_name:
    #         label_str += "6."
    #     elif "7.N" in fgi_name:
    #         label_str += "7."
    #     elif "8.N" in fgi_name:
    #         label_str += "8."
    #     elif "9.N" in fgi_name:
    #         label_str += "9."
    #     elif "space" in fgi_name:
    #         label_str += ""
    #     else:
    #         print("Error!")
    #     # label_str += fgi_name
    #
    #     fg_abs_path = fg_path + "/{}".format(fgi)
    #     fg_img = cv2.imread(fg_abs_path)
    #     fg_img = cv2.resize(fg_img, (48, 70))
    #     fgsz = fg_img.shape[:2]
    #
    #     mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
    #
    #     if i == 0:
    #         out = cv2.seamlessClone(fg_img, bg_img, mask_img, (30, 36), cv2.MIXED_CLONE)
    #     else:
    #         out = cv2.seamlessClone(fg_img, bg_img, mask_img, (68 + 30 + 56 * (i - 1), 36), cv2.MIXED_CLONE)
    #     bg_img = out
    #
    # cv2.imwrite("{}/20230905_{}_{}={}.jpg".format(save_path, str(rdm).replace(".", ""), fgi_name, label_str), out)


def find_OCR_labels():
    # data_path = "/home/disk/disk7/010.Digital_Rec/crnn/train/v5_20231020_cp_bk/64_256"
    # save_path = make_save_path(data_path, "selected")
    # dir_list = get_sub_dir_list(data_path)
    # for d in dir_list:
    #     file_list = get_file_list(d)
    #     for f in tqdm(file_list):
    #         f_abs_path = d + "/{}".format(f)
    #         fname = os.path.splitext(f)[0]
    #         label = fname.split("=")[1]
    #         if label == "" or label[0] == "." or label[-1] == ".":
    #             f_dst_path = save_path + "/{}".format(f)
    #             shutil.move(f_abs_path, f_dst_path)

    # data_path = "/home/disk/disk7/010.Digital_Rec/crnn/train/v5_20231020/gen_on_bg"
    # save_path = make_save_path(data_path, "selected")
    # dir_list = get_sub_dir_list(data_path)
    # for d in dir_list:
    #     sub_dir_list = get_sub_dir_list(d)
    #     for sd in sub_dir_list:
    #         file_list = get_file_list(sd)
    #         for f in tqdm(file_list):
    #             f_abs_path = sd + "/{}".format(f)
    #             fname = os.path.splitext(f)[0]
    #             label = fname.split("=")[1]
    #             if label == "" or label[0] == "." or label[-1] == ".":
    #                 f_dst_path = save_path + "/{}".format(f)
    #                 shutil.move(f_abs_path, f_dst_path)

    data_path = "/home/wujiahu/data/000.OCR/CRNN/data/train/v1"
    save_path = make_save_path(data_path, "selected")
    dir_list = get_sub_dir_list(data_path)

    LABELS = ""

    for d in dir_list:
        file_list = get_file_list(d)
        for f in tqdm(file_list):
            f_abs_path = d + "/{}".format(f)
            fname = os.path.splitext(f)[0]
            deng_idx = fname.index("=")
            label = fname[deng_idx + 1:]
            for l in label:
                if l not in LABELS:
                    LABELS += l
    print(LABELS)
    print(len(LABELS))


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True, map_size=5073741824):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=map_size)
    cache = {}
    cnt = 1

    datalist = open(gtFile, 'r', encoding='utf-8').read().strip().split('\n')

    print(len(datalist))
    for i, sample in tqdm(enumerate(datalist)):
        try:
            imagePath, label = sample.split('\t')
            if len(label) < 51:
                imagePath = os.path.join(inputPath, imagePath)

                # # only use alphanumeric data
                # if re.search('[^a-zA-Z0-9]', label):
                #     continue

                if not os.path.exists(imagePath):
                    print('%s does not exist' % imagePath)
                    continue
                with open(imagePath, 'rb') as f:
                    imageBin = f.read()
                if checkValid:
                    try:
                        if not checkImageIsValid(imageBin):
                            print('%s is not a valid image' % imagePath)
                            continue
                    except:
                        print('error occured', i)
                        with open(outputPath + '/error_image_log.txt', 'a') as log:
                            log.write('%s-th image data occured error\n' % str(i))
                        continue

                imageKey = 'image-%09d'.encode() % cnt
                labelKey = 'label-%09d'.encode() % cnt
                cache[imageKey] = imageBin
                cache[labelKey] = label.strip().encode()

                if cnt % 1000 == 0:
                    writeCache(env, cache)
                    cache = {}
                    print('Written %d / %d' % (cnt, i))
                cnt += 1
        except Exception as e:
            print(sample, e)
    i = cnt - 1
    cache['num-samples'.encode()] = str(i).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % i)


def createDataset_v2(data_path, checkValid=True, map_size=5073741824):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """

    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/lmdb"
    os.makedirs(save_path, exist_ok=True)

    env = lmdb.open(save_path, map_size=map_size)
    cache = {}
    cnt = 1

    # datalist = open(gtFile, 'r', encoding='utf-8').read().strip().split('\n')

    fr = open(data_path, 'r', encoding='utf-8')
    datalist = fr.readlines()
    fr.close()
    len_d = len(datalist)
    print(len_d)

    for i, sample in tqdm(enumerate(datalist)):
        try:
            imagePath, label = sample.split(' ')
            # if len(label) < 51:
            # imagePath = os.path.join(inputPath, imagePath)

            # # only use alphanumeric data
            # if re.search('[^a-zA-Z0-9]', label):
            #     continue

            if not os.path.exists(imagePath):
                print('%s does not exist' % imagePath)
                continue

            with open(imagePath, 'rb') as f:
                imageBin = f.read()

            if checkValid:
                try:
                    if not checkImageIsValid(imageBin):
                        print('%s is not a valid image' % imagePath)
                        continue
                except:
                    print('error occured', i)
                    with open(save_path + '/error_image_log.txt', 'a') as log:
                        log.write('%s-th image data occured error\n' % str(i))
                    continue

            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label.strip().encode()

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, len_d))
            cnt += 1

        except Exception as e:
            print(sample, e)

    i = cnt - 1
    cache['num-samples'.encode()] = str(i).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % i)


# class ImageDataset(Dataset):
    # "`ImageDataset` read data from LMDB database."

    # def __init__(self,
                 # path: PathOrStr,
                 # is_training: bool = True,
                 # img_h: int = 32,
                 # img_w: int = 100,
                 # max_length: int = 25,
                 # check_length: bool = True,
                 # case_sensitive: bool = False,
                 # charset_path: str = 'data/charset_vn_with_space.txt',
                 # convert_mode: str = 'RGB',
                 # data_aug: bool = True,
                 # deteriorate_ratio: float = 0.,
                 # multiscales: bool = True,
                 # one_hot_y: bool = True,
                 # return_idx: bool = False,
                 # return_raw: bool = False,
                 # **kwargs):
        # self.path, self.name = Path(path), Path(path).name
        # assert self.path.is_dir() and self.path.exists(), f"{path} is not a valid directory."
        # self.convert_mode, self.check_length = convert_mode, check_length
        # self.img_h, self.img_w = img_h, img_w
        # self.max_length, self.one_hot_y = max_length, one_hot_y
        # self.return_idx, self.return_raw = return_idx, return_raw
        # self.case_sensitive, self.is_training = case_sensitive, is_training
        # self.data_aug, self.multiscales = data_aug, multiscales
        # self.charset = CharsetMapper(charset_path, max_length=max_length + 1)
        # self.character = self.charset.label_to_char.values()
        # self.c = self.charset.num_classes

        # self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        # assert self.env, f'Cannot open LMDB dataset from {path}.'
        # with self.env.begin(write=False) as txn:
            # self.length = int(txn.get('num-samples'.encode()))

        # if self.is_training and self.data_aug:
            # self.augment_tfs = transforms.Compose([
                # CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                # CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                # CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            # ])
        # self.totensor = transforms.ToTensor()

    # def __len__(self):
        # return self.length

    # def _next_image(self, index):
        # next_index = random.randint(0, len(self) - 1)
        # return self.get(next_index)

    # def _check_image(self, x, pixels=6):
        # if x.size[0] <= pixels or x.size[1] <= pixels:
            # return False
        # else:
            # return True

    # def resize_multiscales(self, img, borderType=cv2.BORDER_CONSTANT):
        # def _resize_ratio(img, ratio, fix_h=True):
            # if ratio * self.img_w < self.img_h:
                # if fix_h:
                    # trg_h = self.img_h
                # else:
                    # trg_h = int(ratio * self.img_w)
                # trg_w = self.img_w
            # else:
                # trg_h, trg_w = self.img_h, int(self.img_h / ratio)
            # img = cv2.resize(img, (trg_w, trg_h))
            # pad_h, pad_w = (self.img_h - trg_h) / 2, (self.img_w - trg_w) / 2
            # top, bottom = math.ceil(pad_h), math.floor(pad_h)
            # left, right = math.ceil(pad_w), math.floor(pad_w)
            # img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
            # return img

        # if self.is_training:
            # if random.random() < 0.5:
                # base, maxh, maxw = self.img_h, self.img_h, self.img_w
                # h, w = random.randint(base, maxh), random.randint(base, maxw)
                # return _resize_ratio(img, h / w)
            # else:
                # return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio
        # else:
            # return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio

    # def resize(self, img):
        # if self.multiscales:
            # return self.resize_multiscales(img, cv2.BORDER_REPLICATE)
        # else:
            # return cv2.resize(img, (self.img_w, self.img_h))

    # def get(self, idx):
        # with self.env.begin(write=False) as txn:
            # image_key, label_key = f'image-{idx + 1:09d}', f'label-{idx + 1:09d}'
            # try:
                # label = str(txn.get(label_key.encode()), 'utf-8').strip()  # label
                # if not set(label).issubset(self.character):
                    # return self._next_image(idx)
                # label = re.sub('[^0-9a-zA-Z]+', '', label)
                # if self.check_length and self.max_length > 0:
                    # if len(label) > self.max_length or len(label) <= 0:
                        # logging.info(f'Long or short text image is found: {self.name}, {idx}, {label}, {len(label)}')
                        # return self._next_image(idx)
                # label = label[:self.max_length]

                # imgbuf = txn.get(image_key.encode())  # image
                # buf = six.BytesIO()
                # buf.write(imgbuf)
                # buf.seek(0)
                # with warnings.catch_warnings():
                    # warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
                    # image = PIL.Image.open(buf).convert(self.convert_mode)
                # if self.is_training and not self._check_image(image):
                    # logging.info(f'Invalid image is found: {self.name}, {idx}, {label}, {len(label)}')
                    # return self._next_image(idx)
            # except:
                # import traceback
                # traceback.print_exc()
                # logging.info(f'Corrupted image is found: {self.name}, {idx}, {label}, {len(label)}')
                # return self._next_image(idx)
            # return image, label, idx

    # def _process_training(self, image):
        # if self.data_aug: image = self.augment_tfs(image)
        # image = self.resize(np.array(image))
        # return image

    # def _process_test(self, image):
        # return self.resize(np.array(image))  # TODO:move is_training to here

    # def __getitem__(self, idx):
        # image, text, idx_new = self.get(idx)
        # print(image, text, idx_new, idx)
        # if not self.is_training: assert idx == idx_new, f'idx {idx} != idx_new {idx_new} during testing.'

        # if self.is_training:
            # image = self._process_training(image)
        # else:
            # image = self._process_test(image)
        # if self.return_raw: return image, text
        # image = self.totensor(image)

        # length = tensor(len(text) + 1).to(dtype=torch.long)  # one for end token
        # label = self.charset.get_labels(text, case_sensitive=self.case_sensitive)
        # label = tensor(label).to(dtype=torch.long)
        # if self.one_hot_y: label = onehot(label, self.charset.num_classes)

        # if self.return_idx:
            # y = [label, length, idx_new]
        # else:
            # y = [label, length]
        # return image, y


# class TextDataset(Dataset):
    # def __init__(self,
                 # path: PathOrStr,
                 # delimiter: str = '\t',
                 # max_length: int = 25,
                 # charset_path: str = 'data/charset_vn.txt',
                 # case_sensitive=False,
                 # one_hot_x=True,
                 # one_hot_y=True,
                 # is_training=True,
                 # smooth_label=False,
                 # smooth_factor=0.2,
                 # use_sm=False,
                 # **kwargs):
        # self.path = Path(path)
        # self.case_sensitive, self.use_sm = case_sensitive, use_sm
        # self.smooth_factor, self.smooth_label = smooth_factor, smooth_label
        # self.charset = CharsetMapper(charset_path, max_length=max_length + 1)
        # self.one_hot_x, self.one_hot_y, self.is_training = one_hot_x, one_hot_y, is_training
        # if self.is_training and self.use_sm: self.sm = SpellingMutation(charset=self.charset)

        # dtype = {'inp': str, 'gt': str}
        # self.df = pd.read_csv(self.path, dtype=dtype, delimiter=delimiter, na_filter=False)
        # self.inp_col, self.gt_col = 0, 1

    # def __len__(self):
        # return len(self.df)

    # def __getitem__(self, idx):
        # text_x = self.df.iloc[idx, self.inp_col].strip()
        # text_x = re.sub('[^0-9a-zA-Z]+', '', text_x)
        # if not self.case_sensitive: text_x = text_x.lower()
        # if self.is_training and self.use_sm: text_x = self.sm(text_x)

        # length_x = tensor(len(text_x) + 1).to(dtype=torch.long)  # one for end token
        # label_x = self.charset.get_labels(text_x, case_sensitive=self.case_sensitive)
        # label_x = tensor(label_x)
        # if self.one_hot_x:
            # label_x = onehot(label_x, self.charset.num_classes)
            # if self.is_training and self.smooth_label:
                # label_x = torch.stack([self.prob_smooth_label(l) for l in label_x])
        # x = [label_x, length_x]

        # text_y = self.df.iloc[idx, self.gt_col]
        # text_y = re.sub('[^0-9a-zA-Z]+', '', text_y)
        # if not self.case_sensitive: text_y = text_y.lower()
        # length_y = tensor(len(text_y) + 1).to(dtype=torch.long)  # one for end token
        # label_y = self.charset.get_labels(text_y, case_sensitive=self.case_sensitive)
        # label_y = tensor(label_y)
        # if self.one_hot_y: label_y = onehot(label_y, self.charset.num_classes)
        # y = [label_y, length_y]

        # return x, y

    # def prob_smooth_label(self, one_hot):
        # one_hot = one_hot.float()
        # delta = torch.rand([]) * self.smooth_factor
        # num_classes = len(one_hot)
        # noise = torch.rand(num_classes)
        # noise = noise / noise.sum() * delta
        # one_hot = one_hot * (1 - delta) + noise
        # return one_hot


# class SpellingMutation(object):
    # def __init__(self, pn0=0.7, pn1=0.85, pn2=0.95, pt0=0.7, pt1=0.85, charset=None):
        # """
        # Args:
            # pn0: the prob of not modifying characters is (pn0)
            # pn1: the prob of modifying one characters is (pn1 - pn0)
            # pn2: the prob of modifying two characters is (pn2 - pn1),
                 # and three (1 - pn2)
            # pt0: the prob of replacing operation is pt0.
            # pt1: the prob of inserting operation is (pt1 - pt0),
                 # and deleting operation is (1 - pt1)
        # """
        # super().__init__()
        # self.pn0, self.pn1, self.pn2 = pn0, pn1, pn2
        # self.pt0, self.pt1 = pt0, pt1
        # self.charset = charset
        # logging.info(f'the probs: pn0={self.pn0}, pn1={self.pn1} ' +
                     # f'pn2={self.pn2}, pt0={self.pt0}, pt1={self.pt1}')

    # def is_digit(self, text, ratio=0.5):
        # length = max(len(text), 1)
        # digit_num = sum([t in self.charset.digits for t in text])
        # if digit_num / length < ratio: return False
        # return True

    # def is_unk_char(self, char):
        # return char == self.charset.unk_char
        # return (char not in self.charset.digits) and (char not in self.charset.alphabets)

    # def get_num_to_modify(self, length):
        # prob = random.random()
        # if prob < self.pn0:
            # num_to_modify = 0
        # elif prob < self.pn1:
            # num_to_modify = 1
        # elif prob < self.pn2:
            # num_to_modify = 2
        # else:
            # num_to_modify = 3

        # if length <= 1:
            # num_to_modify = 0
        # elif length >= 2 and length <= 4:
            # num_to_modify = min(num_to_modify, 1)
        # else:
            # num_to_modify = min(num_to_modify, length // 2)  # smaller than length // 2
        # return num_to_modify

    # def __call__(self, text, debug=False):
        # if self.is_digit(text): return text
        # length = len(text)
        # num_to_modify = self.get_num_to_modify(length)
        # if num_to_modify <= 0: return text

        # chars = []
        # index = np.arange(0, length)
        # random.shuffle(index)
        # index = index[: num_to_modify]
        # if debug: self.index = index
        # for i, t in enumerate(text):
            # if i not in index:
                # chars.append(t)
            # elif self.is_unk_char(t):
                # chars.append(t)
            # else:
                # prob = random.random()
                # if prob < self.pt0:  # replace
                    # chars.append(random.choice(self.charset.alphabets))
                # elif prob < self.pt1:  # insert
                    # chars.append(random.choice(self.charset.alphabets))
                    # chars.append(t)
                # else:  # delete
                    # continue
        # new_text = ''.join(chars[: self.charset.max_length - 1])
        # return new_text if len(new_text) >= 1 else text


# class LMDBImageDataset_v2(Dataset):
    # "`ImageDataset` read data from LMDB database."

    # def __init__(self,
                 # path: PathOrStr,
                 # is_training: bool = True,
                 # img_h: int = 64,
                 # img_w: int = 256,
                 # max_length: int = 30,
                 # check_length: bool = True,
                 # case_sensitive: bool = True,
                 # charset_path: str = 'data/charset_vn_with_space.txt',
                 # convert_mode: str = 'RGB',
                 # data_aug: bool = True,
                 # deteriorate_ratio: float = 0.,
                 # multiscales: bool = True,
                 # one_hot_y: bool = True,
                 # return_idx: bool = False,
                 # return_raw: bool = False,
                 # imgsz=(64, 256), input_len=64, mean=np.array([0.5, 0.5, 0.5]), std=np.array([0.5, 0.5, 0.5]), r1=0, r2=0.25,
                 # rotate=True, rotate_angle=45, aug_ratio=0.90, change_brightness=True, gamma_correction=True, add_noise=True, add_sunlight_effect=True, train_Chinese_flag=True,
                 # **kwargs):
        # self.path, self.name = Path(path), Path(path).name
        # assert self.path.is_dir() and self.path.exists(), f"{path} is not a valid directory."
        # self.convert_mode, self.check_length = convert_mode, check_length
        # self.img_h, self.img_w = img_h, img_w
        # self.max_length, self.one_hot_y = max_length, one_hot_y
        # self.return_idx, self.return_raw = return_idx, return_raw
        # self.case_sensitive, self.is_training = case_sensitive, is_training
        # self.data_aug, self.multiscales = data_aug, multiscales
        # self.charset = CharsetMapper(charset_path, max_length=max_length + 1)
        # self.character = self.charset.label_to_char.values()
        # self.c = self.charset.num_classes

        # ============================= Chinese ============================
        # CH_SIM_CHARS = ' ' + '0123456789.' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        # CH_SIM_CHARS += ',;~!@#$%^&*()_+-={}:"<>?-=[]/|\\' + "'"
        # CH_SIM_CHARS += '、。┅《》「」【】¥®πи‰℃№Ⅱ←↑→↓①②③④▪☆❤'
        # ch_sim_chars = open("/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/ch_sim_char.txt", "r", encoding="utf-8")
        # lines = ch_sim_chars.readlines()
        # for l in lines:
            # CH_SIM_CHARS += l.strip()
        # alpha = CH_SIM_CHARS  # len = 6738
        # args.alpha = alpha
        # self.character = alpha
        # ============================= Chinese ============================

        # self.alpha = alpha
        # self.preprocess = True
        # self.eval = False

        # self.im_h = imgsz[0]
        # self.im_w = imgsz[1]
        # self.input_len = input_len

        # self.mean_ = mean
        # self.std_ = std
        # self.r1 = r1
        # self.r2 = r2
        # self.rotate = rotate
        # self.rotate_angle = rotate_angle
        # self.aug_ratio = aug_ratio
        # self.change_brightness = change_brightness
        # self.gamma_correction = gamma_correction
        # self.add_noise = add_noise
        # self.add_sunlight_effect = add_sunlight_effect

        # self.noise_aug = NoiseAug(ratio=1.0)
        # self.blur_aug = BlurAug(type="EASY", ratio=1.0)
        # self.hsv_aug = HSVAug(hgain=0.2, sgain=0.7, vgain=0.5, ratio=1.0)

        # self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        # assert self.env, f'Cannot open LMDB dataset from {path}.'
        # with self.env.begin(write=False) as txn:
            # self.length = int(txn.get('num-samples'.encode()))

        # if self.is_training and self.data_aug:
            # self.augment_tfs = transforms.Compose([
                # CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                # CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                # CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            # ])
        # self.totensor = transforms.ToTensor()

    # def __len__(self):
        # return self.length

    # def _next_image(self, index):
        # return self.get(index + 1)

    # def _check_image(self, x, pixels=6):
        # if x.size[0] <= pixels or x.size[1] <= pixels:
            # return False
        # else:
            # return True

    # def get(self, idx):
        # with self.env.begin(write=False) as txn:
            # image_key, label_key = f'image-{idx + 1:09d}', f'label-{idx + 1:09d}'
            # try:
                # label = str(txn.get(label_key.encode()), 'utf-8').strip()  # label
                # if not set(label).issubset(self.character):
                    # return self._next_image(idx)
                # label = re.sub('[^0-9a-zA-Z]+', '', label)
                # if self.check_length and self.max_length > 0:
                    # if len(label) > self.max_length or len(label) <= 0:
                        # logging.info(f'Long or short text image is found: {self.name}, {idx}, {label}, {len(label)}')
                        # return self._next_image(idx)
                # label = label[:self.max_length]

                # imgbuf = txn.get(image_key.encode())  # image
                # buf = six.BytesIO()
                # buf.write(imgbuf)
                # buf.seek(0)
                # with warnings.catch_warnings():
                    # warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
                    # image = PIL.Image.open(buf).convert(self.convert_mode)
                # if self.is_training and not self._check_image(image):
                    # logging.info(f'Invalid image is found: {self.name}, {idx}, {label}, {len(label)}')
                    # return self._next_image(idx)
            # except:
                # import traceback
                # traceback.print_exc()
                # logging.info(f'Corrupted image is found: {self.name}, {idx}, {label}, {len(label)}')
                # return self._next_image(idx)
            # return image, label, idx

    # def _process_training(self, image):
        # if self.data_aug: image = self.augment_tfs(image)
        # image = self.resize(np.array(image))
        # return image
    
    # def _process_test(self, image):
        # return self.resize(np.array(image))  # TODO:move is_training to here

    # def changeBrightness(self, img, value=30):
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)
        # v = cv2.add(v, value)
        # v[v > 255] = 255
        # v[v < 0] = 0
        # final_hsv = cv2.merge((h, s, v))
        # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        # return img

    # def gammaCorrection(self, img, gamma=0.4):
        # lookUpTable = np.empty((1, 256), np.uint8)
        # for i in range(256):
            # lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        # res = cv2.LUT(img, lookUpTable)
        # return res

    # def addNoise_v2(self, img, noise_typ):
        # """
        # Parameters
        # ----------
        # image : ndarray
            # Input image data. Will be converted to float.
        # mode : str
        # One of the following strings, selecting the type of noise to add:

        # 'gauss'     Gaussian-distributed additive noise.
        # 'poisson'   Poisson-distributed noise generated from the data.
        # 's&p'       Replaces random pixels with 0 or 1.
        # 'speckle'   Multiplicative noise using out = image + n*image,where
                    # n is uniform noise with specified mean & variance.

        # """
        # if noise_typ == "gaussian":
            # """
            # Examples
                # --------
                # Draw samples from the distribution:
                
                # >>> mu, sigma = 0, 0.1 # mean and standard deviation
                # >>> s = np.random.normal(mu, sigma, 1000)
                
                # Verify the mean and the variance:
                
                # >>> abs(mu - np.mean(s))
                # 0.0  # may vary
                
                # >>> abs(sigma - np.std(s, ddof=1))
                # 0.1  # may vary
                
                # Display the histogram of the samples, along with
                # the probability density function:
                
                # >>> import matplotlib.pyplot as plt
                # >>> count, bins, ignored = plt.hist(s, 30, density=True)
                # >>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                # ...                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                # ...          linewidth=2, color='r')
                # >>> plt.show()
            # Parameters
            # ----------
            # img

            # Returns
            # -------

            # """
            # 生成高斯噪声
            # mu, sigma = 0, 0.5 ** 0.5
            # gaussian = np.random.normal(mu, sigma, img.shape).astype('uint8')
            # noisy_img = cv2.add(img, gaussian)
            # return noisy_img

        # elif noise_typ == "s&p":
            # row, col, ch = img.shape
            # s_vs_p = 0.5
            # amount = 0.004
            # out = np.copy(img)
            # Salt mode
            # num_salt = np.ceil(amount * img.size * s_vs_p)
            # coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            # out[coords] = 1
            # coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape]
            # for ii in range(len(coords[0])):
                # out[coords[0][ii], coords[1][ii], coords[2][ii]] = 1

            # Pepper mode
            # num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            # coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            # out[coords] = 0
            # coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape]
            # for ii in range(len(coords[0])):
                # out[coords[0][ii], coords[1][ii], coords[2][ii]] = 0
            # return out

        # elif noise_typ == "s&p_v2":
            # salt_prob = 0.01
            # pepper_prob = 0.01
            # noisy_image = np.copy(img)
            # total_pixels = img.shape[0] * img.shape[1]  # 计算图像的总像素数

            # num_salt = int(total_pixels * salt_prob)  # 通过将总像素数与指定的椒盐噪声比例相乘，得到要添加的椒盐噪声的数量。
            # salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
            # noisy_image[salt_coords[0], salt_coords[1]] = 255

            # num_pepper = int(total_pixels * pepper_prob)
            # pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
            # noisy_image[pepper_coords[0], pepper_coords[1]] = 0

            # return noisy_image

        # elif noise_typ == "poisson":
            # vals = len(np.unique(img))
            # vals = 2 ** np.ceil(np.log2(vals))
            # noisy = np.random.poisson(img * vals) / float(vals)
            # return noisy

        # else:
            # raise NotImplementedError("noise_typ error: not found {}".format(noise_typ))

    # def makeSunLightEffect(self, img, r=(50, 200), light_strength=150):
        # imgsz = img.shape[:2]
        # center = (np.random.randint(0, imgsz[1]), np.random.randint(0, imgsz[0]))
        # effectR = np.random.randint(r[0], r[1])
        # lightStrength = np.random.randint(light_strength // 4, light_strength)

        # dst = np.zeros(shape=img.shape, dtype=np.uint8)

        # for i in range(imgsz[0]):
            # for j in range(imgsz[1]):
                # dis = (center[0] - j) ** 2 + (center[1] - i) ** 2
                # B, G, R = img[i, j][0], img[i, j][1], img[i, j][2]
                # if dis < effectR * effectR:
                    # result = int(lightStrength * (1.0 - np.sqrt(dis) / effectR))
                    # B += result
                    # G += result
                    # R += result

                    # B, G, R = min(max(0, B), 255), min(max(0, G), 255), min(max(0, R), 255)
                    # dst[i, j] = np.uint8((B, G, R))
                # else:
                    # dst[i, j] = np.uint8((B, G, R))
        # return dst

    # -----------------------------------------
    # def apply_color_distortion(self, img, r=30):
        # hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hsv_image[:, :, 0] = (hsv_image[:, :, 0] + r) % 180  # 在Hue通道上增加30
        # result_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        # return result_image

    # def change_Contrast_and_Brightness(self, img, alpha=1.1, beta=30):
        # """使用公式f(x)=α.g(x)+β"""
        # α调节对比度，β调节亮度
        # blank = np.zeros(img.shape, img.dtype)  # 创建图片类型的零矩阵
        # dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)  # 图像混合加权
        # return dst

    # def apply_CLAHE(self, img, clipLimit=2.0, tileGridSize=(8, 8)):
        # """
        # 直方图适应均衡化
        # 该函数包含以下参数：
        # clipLimit: 用于控制直方图均衡化的局部对比度，值越高，越容易出现失真和噪声。建议值为2-4，若使用默认值0则表示自动计算。
        # tileGridSize: 表示每个块的大小，推荐16x16。
        # tileGridSize.width: 块的宽度。
        # tileGridSize.height: 块的高度。
        # 函数返回一个CLAHE对象，可以通过该对象调用apply函数来实现直方图均衡化。
        # """
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        # res = clahe.apply(img)
        # res = cv2.merge([res, res, res])

        # b, g, r = cv2.split(img)
        # clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        # clahe_b = clahe.apply(b)
        # clahe_g = clahe.apply(g)
        # clahe_r = clahe.apply(r)
        # res = cv2.merge([clahe_b, clahe_g, clahe_r])

        # return res

    # def change_HSV(self, img, hgain=0.5, sgain=0.5, vgain=0.5):
        # img = img.astype(np.uint8)
        # r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        # hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        # dtype = img.dtype  # uint8

        # x = np.arange(0, 256, dtype=np.int16)
        # lut_hue = ((x * r[0]) % 180).astype(dtype)
        # lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        # lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        # cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        # img = img.astype(np.float32)
        # return img

    # def do_aug_v2(self, img_x):
        # aug_rdm = np.random.rand()

        # if aug_rdm < 0.20:
            # img_x_pil = Image.fromarray(np.uint8(img_x))
            # rotate_angle = np.random.randint(-self.rotate_angle, self.rotate_angle + 1)
            # img_x = np.asarray(img_x_pil.rotate(rotate_angle, expand=True))
        # elif 0.20 <= aug_rdm < 0.30:
            # brightness = np.random.randint(-20, 21)
            # img_x = self.changeBrightness(img_x, value=brightness)
        # elif 0.30 <= aug_rdm < 0.40:
            # gammav = 0.1 * np.random.randint(4, 17)
            # img_x = self.gammaCorrection(img_x, gamma=gammav)
        # elif 0.40 <= aug_rdm < 0.50:
            # mthd = ["gaussian", "s&p", "s&p_v2", "poisson"]
            # img_x = self.addNoise_v2(img_x, np.random.choice(mthd, 1)[0])
        # elif 0.50 <= aug_rdm < 0.60:
            # hue_v = np.random.randint(0, 50)
            # img_x = self.apply_color_distortion(img_x, r=hue_v)
        # elif 0.60 <= aug_rdm < 0.70:
            # img_x = self.apply_CLAHE(img_x, clipLimit=2.0, tileGridSize=(8, 8))
        # elif 0.70 <= aug_rdm < 0.80:
            # img_x = self.change_HSV(img_x)
        # elif 0.80 <= aug_rdm < 0.90:
            # img_x = self.noise_aug(img_x)
            # img_x = self.blur_aug(img_x)
            # img_x = self.hsv_aug(img_x)
            # img = TransAffine(img, degrees=8, translate=0.0, scale=0.2, shear=0, perspective=0, border=(2,border_width), prob=0.95)
            # img = TransAffine(img, degrees=3, translate=0.00025, scale=0.1, shear=3, perspective=0.0005, border=(border_height, border_width), prob=1.0)

            # angle_v = np.random.randint(1, 8)
            # translate_v = 0.001 * np.random.randint(0, 30)
            # scale_v = 0.01 * np.random.randint(0, 15)
            # shear_v = 0.1 * np.random.randint(0, 15)
            # perspective_v = 0.00001 * np.random.randint(0, 30)

            # img_x = TransAffine(img_x, degrees=angle_v, translate=translate_v, scale=scale_v, shear=shear_v, perspective=perspective_v, border=(0, 0), prob=1.0)
            # img_x = img_x.astype(np.uint8)

        # else:
            # img_x = self.makeSunLightEffect(img_x, r=(50, 200), light_strength=150)

        # return img_x

    # def __getitem__(self, idx):
        # image, label, idx_new = self.get(idx)

        # print(image, text, idx_new, idx)
        # if not self.is_training: assert idx == idx_new, f'idx {idx} != idx_new {idx_new} during testing.'
        
        # if self.is_training:
            # image = self._process_training(image)
        # else:
            # image = self._process_test(image)
        # if self.return_raw: return image, text
        # image = self.totensor(image)
        
        # length = tensor(len(text) + 1).to(dtype=torch.long)  # one for end token
        # label = self.charset.get_labels(text, case_sensitive=self.case_sensitive)
        # label = tensor(label).to(dtype=torch.long)
        # if self.one_hot_y: label = onehot(label, self.charset.num_classes)
        
        # if self.return_idx:
            # y = [label, length, idx_new]
        # else:
            # y = [label, length]
        # return image, y

        # target_len = len(label)
        # try:
        # indices = np.array([self.alpha.index(c) for c in label])
        # except Exception as Error:
            # print(Error)
            # print(x_path)
        # target = np.zeros(shape=(self.input_len,), dtype=np.int64)
        # target[:target_len] = indices

        # rdm = np.random.rand()

        # img_x = np.asarray(image, dtype=np.uint8)

        # if self.preprocess:
            # img_x = cv2.imread(x_path)
            # img_x = cv2.imdecode(np.fromfile(x_path, dtype=np.uint8), cv2.IMREAD_COLOR)

            # if not self.eval:
                # img_x = self.do_aug(img_x)
                # img_x = self.do_aug_v2(img_x)
                # if rdm > 0.5:
                    # img_x = self.do_aug_v2(img_x)
            # img_x = makeBorder_v5(img_x, new_shape=(self.im_h, self.im_w), r1=self.r1, r2=self.r2, sliding_window=False)
            # img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
            # img_x = (img_x / 255. - self.mean_) / self.std_
        # else:
            # img_x = cv2.imread(x_path)
            # img_x = cv2.imdecode(np.fromfile(x_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            # if not self.eval:
                # img_x = self.do_aug(img_x)
                # img_x = self.do_aug_v2(img_x)
                # if rdm > 0.5:
                    # img_x = self.do_aug_v2(img_x)
            # img_x = makeBorder_v5(img_x, new_shape=(self.im_h, self.im_w), r1=self.r1, r2=self.r2, sliding_window=False)
            # img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)

        # img_x = img_x.transpose(2, 0, 1)
        # img_x = img_x.copy()
        # img_x = torch.from_numpy(img_x).type(torch.FloatTensor)

        # if self.eval:
            # return img_x, target, self.input_len, target_len, label
        # else:
            # return img_x, target, self.input_len, target_len


def convertBaiduChineseOCRDatasetToCustomDatasetFormat(data_path):
    train_images_path = data_path + "/train_images"
    train_list_path = data_path + "/train.list"
    img_list = sorted(os.listdir(train_images_path))

    save_Chinese_path = make_save_path(train_images_path, "isAllChinese")
    save_digits_path = make_save_path(train_images_path, "isAllDigits")

    with open(train_list_path, "r", encoding="utf-8") as fo:
        lines = fo.readlines()
        for line in tqdm(lines):
            try:
                line = line.strip()
                img_name = line.split("\t")[2]
                label = line.split("\t")[3]

                res1 = isAllChinese(label)
                res2 = isAllDigits(label)

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


def convert_label(lb):
    labels = "０１２３４５６７８９"
    labels_new = "0123456789"

    lb_new = ""
    for l in lb:
        idx = labels.find(l)
        lb_new += labels_new[idx]

    return lb_new
# ========================================================================================================================================================================
# ========================================================================================================================================================================




# ========================================================================================================================================================================
# ========================================================================================================================================================================
# CLS
def random_crop_gen_cls_negative_samples(data_path, random_size=(96, 100, 128, 160), randint_low=10, randint_high=51, hw_dis=100, dst_num=20000):
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
            cv2img = cv2.imread(img_abs_path)
            h, w = cv2img.shape[:2]
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
                    random_cropped = cv2img[random_pos[1]:(random_pos[1] + size_i[0][0]), random_pos[0]:(random_pos[0] + size_i[1][0])]
                    cv2.imwrite("{}/{}_{}_{}_{}.jpg".format(save_path, img_name, size_i[0][0], size_i[1][0], i), random_cropped)

                    total_num += 1

                except Exception as Error:
                    print(Error, Error.__traceback__.tb_lineno)
        except Exception as Error:
            print(Error, Error.__traceback__.tb_lineno)


class ClsModel():
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
        self.ort_session = onnxruntime.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', "CPUExecutionProvider"])

    def keep_ratio(self, pilimg, flag=True, shape=(128, 128)):
        if flag:
            cv2img = np.array(np.uint8(pilimg))
            img_src, ratio, (dw, dh) = letterbox(cv2img, new_shape=shape)
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


def random_erasing_aug_cls_data(data_path):
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


def random_paste_four_corner_aug_cls_data(positive_img_path, negative_img_path):
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
# ========================================================================================================================================================================
# ========================================================================================================================================================================


# ========================================================================================================================================================================
# ========================================================================================================================================================================
# PCL
def mat2pcl(img, K):
    f_x = K[0, 0]
    f_y = K[1, 1]
    s = K[0, 1]
    p_x = K[0, 2]
    p_y = K[1, 2]

    y_values, x_values = np.where(img > 300)
    z_values = img[y_values, x_values]

    Y_values = (y_values - p_y) * z_values / f_y
    X_values = (x_values - p_x - (y_values - p_y) * s / f_y) * z_values / f_x

    points_3d = np.array([X_values, Y_values, z_values]).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    return pcd


def main_pcl():
    fs = cv2.FileStorage(r"data/HXZY/matrix.yml", cv2.FileStorage_READ)
    K = fs.getNode("M").mat()
    fs.release()

    image_path = r"data/HXZY/2111043208/1-1067.data"
    h_image = cv2.imread(image_path, 2)

    # h_image = (h_image/65535)*900+200

    pcd = mat2pcl(h_image, K)
    o3d.io.write_point_cloud(image_path.replace(".data", ".ply"), pcd)

    return 0
# ========================================================================================================================================================================
# ========================================================================================================================================================================


def select_horizontal_images(data_path, flag="move"):
    file_list = sorted(os.listdir(data_path))
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_selected_horizontal_images".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        imgsz = img.shape[:2]
        if imgsz[0] > imgsz[1]:
            f_dst_path = save_path + "/{}".format(f)
            if flag == "copy" or flag == "cp":
                shutil.copy(f_abs_path, f_dst_path)
            elif flag == "move" or flag == "mv":
                shutil.move(f_abs_path, f_dst_path)
            elif flag == "delete" or flag == "del":
                os.remove(f_abs_path)


# ==============================================================================================================================
# ==============================================================================================================================

def get_color(specific_color_flag=True):
    global color
    color1 = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

    if specific_color_flag:
        color2, color3, color4 = (0, 0, 0), (114, 114, 114), (255, 255, 255)
        color_rdm = np.random.rand()
        # if color_rdm <= 0.85:
        #     color = color1
        # elif color_rdm > 0.85 and color_rdm <= 0.90:
        #     color = color2
        # elif color_rdm > 0.90 and color_rdm <= 0.95:
        #     color = color3
        # else:
        #     color = color4

        if color_rdm <= 0.50:
            color = color2
        elif color_rdm > 0.50 and color_rdm <= 0.75:
            color = color3
        else:
            color = color4

        # color = color3
    else:
        color = color1

    return color


def makeBorder_base(im, new_shape=(64, 256), r1=0.75, specific_color_flag=True):
    """
    :param im:
    :param new_shape: (H, W)
    :param r1:
    :param r2:
    :param sliding_window:
    :return:
    """
    color = get_color(specific_color_flag=specific_color_flag)

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


def sliding_window_crop_v2(img, cropsz=(64, 256), gap=(0, 128), makeBorder=True, r1=0, specific_color_flag=True):
    cropped_imgs = []
    imgsz = img.shape[:2]

    if gap[0] == 0 and gap[1] > 0:
        cropsz = (imgsz[0], cropsz[1])
        for i in range(0, imgsz[1], gap[1]):
            if i + cropsz[1] > imgsz[1]:
                cp_img = img[0:imgsz[0], i:imgsz[1]]
                if makeBorder:
                    cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                    cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                            cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
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
                            cp_img = makeBorder_base(cp_img, new_shape=cropsz, r1=r1, specific_color_flag=specific_color_flag)
                        cropped_imgs.append(cp_img)
                        break
                    else:
                        cp_img = img[j:j + cropsz[0], i:i + cropsz[1]]
                        cropped_imgs.append(cp_img)

    return cropped_imgs


def makeBorder_v6(im, new_shape=(64, 256), r1=0.75, r2=0.25, sliding_window=False, specific_color_flag=True, gap_r=(0, 7 / 8), last_img_makeBorder=True):
    """
    :param im:
    :param new_shape: (H, W)
    :param r1:
    :param r2:
    :param sliding_window:
    :return:
    """
    color = get_color(specific_color_flag=specific_color_flag)

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
                final_imgs = sliding_window_crop_v2(im, cropsz=new_shape, gap=(int(gap_r[0] * 0), int(gap_r[1] * new_shape[1])), makeBorder=last_img_makeBorder, r1=r1)
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
                final_imgs = sliding_window_crop_v2(im, cropsz=new_shape, gap=(int(gap_r[0] * 0), int(gap_r[1] * new_shape[1])), makeBorder=last_img_makeBorder, r1=r1)
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

# ==============================================================================================================================
# ==============================================================================================================================

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


def select_horizontal_images(data_path, flag="move"):
    file_list = sorted(os.listdir(data_path))
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_selected_horizontal_images".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        imgsz = img.shape[:2]
        # if imgsz[0] > imgsz[1]:
        if imgsz[0] * 1.2 < imgsz[1]:
            f_dst_path = save_path + "/{}".format(f)
            if flag == "copy" or flag == "cp":
                shutil.copy(f_abs_path, f_dst_path)
            elif flag == "move" or flag == "mv":
                shutil.move(f_abs_path, f_dst_path)
            elif flag == "delete" or flag == "del":
                os.remove(f_abs_path)


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


def perspective_transform(p1, dstsz, img):
    p1 = np.array([p1[0], p1[1], p1[3], p1[2]], dtype=np.float32)
    p2 = np.array([[0, 0], [dstsz[1], 0], [0, dstsz[0]], [dstsz[1], dstsz[0]]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(p1, p2)
    warped = cv2.warpPerspective(img, M, dstsz[::-1])
    return warped


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)


def median_blur(img, k=3):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.medianBlur(img, k)
    return dst


def clahe(img, clipLimit=40):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    dst1 = clahe.apply(gray)
    dst = cv2.merge([dst1, dst1, dst1])
    return dst


def isAllDigits(string):
    pattern = r'^\d+$'
    if re.match(pattern, string):
        return True
    else:
        return False


def hasDigits(string):
    pattern = r'^\d'
    if re.match(pattern, string):
        return True
    else:
        return False


def hasChinese(string):
    pattern = '[\u4e00-\u9fa5]'
    if re.match(pattern, string):
        return True
    else:
        return False


def containChinese(string):
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


def convert_to_ocr_rec_data_mtwi(data_path):
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


def convert_to_ocr_rec_data_ShopSign1(data_path):
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

        txt_abs_path = txt_path + "/{}.txt".format(fname)

        with open(txt_abs_path, "r", encoding="utf-8") as fr:
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


def convert_to_ocr_rec_data_ShopSign2(data_path):
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


                










if __name__ == '__main__':
    pass






















