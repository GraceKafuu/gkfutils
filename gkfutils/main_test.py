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


from cv.utils import *
from utils.utils import *



import os
import re
import cv2
import numpy as np
import json
import pandas as pd
import random
import shutil
import time
import math
import scipy
import scipy.misc
import copy
import codecs
import onnxruntime
import threading
import skimage
import skimage.io
import imghdr
import struct
import pickle
import hashlib
from tqdm import tqdm
from glob import glob
import torch
import torchvision
from torchvision import transforms


def main_test():
    contain_ciagr_dir = "/home/zengyifan/wujiahu/data/003.Cigar_Det/others/ROBOT_DATA/wubao_wanda_20230128/images/contain_cigar"
    random_cropped_dir = "/home/zengyifan/wujiahu/data/003.Cigar_Det/others/ROBOT_DATA/wubao_wanda_20230128/images_random_cropped"
    cigar_list = os.listdir(contain_ciagr_dir)
    random_cropped_list = os.listdir(random_cropped_dir)

    save_dir = os.path.abspath(os.path.join(contain_ciagr_dir, "../..")) + "/selected_cropped"
    os.makedirs(save_dir, exist_ok=True)

    for img in cigar_list:
        img_name = os.path.splitext(img)[0]
        for cropped_img in random_cropped_list:
            if img_name in cropped_img:
                cropped_img_src_path = random_cropped_dir + "/{}".format(cropped_img)
                cropped_img_dst_path = save_dir + "/{}".format(cropped_img)
                shutil.move(cropped_img_src_path, cropped_img_dst_path)
def main_test2():
    txt_path = "/home/zengyifan/wujiahu/data/001.Banner_Det/train/20230215/train_32395/s.txt"
    data = open(txt_path, "r", encoding="utf-8")
    lines = data.readlines()
    data.close()

    save_path = "/home/zengyifan/wujiahu/data/001.Banner_Det/train/20230215/train_32395/selected_20230223"
    img_save_path = save_path + "/images"
    lbl_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    lines = list(set(lines))
    for l in lines:
        f_name = os.path.basename(l.strip().split(":  ")[1])
        if "img_abs_path" in l:
            img_dst_path = img_save_path + "/{}".format(f_name)
            shutil.copy(l.strip().split(":  ")[1], img_dst_path)
        elif "txt_abs_path" in l:
            lbl_dst_path = lbl_save_path + "/{}".format(f_name)
            shutil.copy(l.strip().split(":  ")[1], lbl_dst_path)
def main_test3():
    data_path1 = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/train/train_fire/20230223/train_14798"
    data_path2 = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/train/train_fire_smoke/20230311/train_smoke_fire_need_to_check_label"

    data_path1_lbl = data_path1 + "/labels"
    data_path2_lbl = data_path2 + "/labels"

    save_path = data_path2 + "/labels_new"
    os.makedirs(save_path, exist_ok=True)

    lbl_list1 = sorted(os.listdir(data_path1_lbl))
    lbl_list2 = sorted(os.listdir(data_path2_lbl))

    same_files = set(lbl_list1) & set(lbl_list2)
    not_same_files = set(lbl_list1) ^ set(lbl_list2)

    for s in same_files:
        s_path1 = data_path1_lbl + "/{}".format(s)
        s_path2 = data_path2_lbl + "/{}".format(s)

        s1_o = open(s_path1, "r", encoding="utf-8")
        s2_o = open(s_path2, "r", encoding="utf-8")
        s1_lines = s1_o.readlines()
        s2_lines = s2_o.readlines()
        s1_o.close()
        s2_o.close()

        s_new_w = open(save_path + "/{}".format(s), "w", encoding="utf-8")
        for l2 in s2_lines:
            cls = l2.strip().split(" ")[0]
            if int(cls) == 0:
                s_new_w.write(l2)

        for l1 in s1_lines:
            cls = l1.strip().split(" ")[0]
            if int(cls) == 0:
                s_new_w.write("1" + l1[1:])

        s_new_w.close()
        print("write --> {}".format(save_path + "/{}".format(s)))
def main_test4():
    # data_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/train/train_fire_smoke/20230311/train_smoke_fire_need_to_check_label/jsons"
    # l = sorted(os.listdir(data_path))
    #
    # for f in l:
    #     if ".avi_" in f:
    #         f_abs_path = data_path + "/{}".format(f)
    #         f_new_name = f.replace(".avi", "")
    #         os.rename(f_abs_path,  data_path + "/{}".format(f_new_name))

    all_files = get_sub_dir_file_list(base_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/test/test_all")
    for f in all_files:
        if ".avi_" in f:
            # f_abs_path = data_path + "/{}".format(f)
            f_new_name = f.replace(".avi", "")
            os.rename(f, f_new_name)
            print("{} --> {}".format(f, f_new_name))
def main_test5():
    img_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/seg/for_paste/smoke_for_paste_0000003.jpg"
    cv2img = cv2.imread(img_path)
    cropped = cv2img[:373, :]
    cv2.imwrite("/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/seg/for_paste/smoke_for_paste_0000003.jpg", cropped)
def main_test6():
    img_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/HK_Prison/prison_data/filter_prison_img"
    bg_txt_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/HK_Prison/prison_data/list_cus_train.txt.bg"
    save_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/HK_Prison/HK_prison_bg"

    with open(bg_txt_path, "r", encoding="utf-8") as fo:
        lines = fo.readlines()
        for l in lines:
            img_curr_abs_path = img_path + "/{}".format(l.strip().split("/")[-1])
            img_curr_dst_path = save_path + "/{}".format(l.strip().split("/")[-1])
            shutil.copy(img_curr_abs_path, img_curr_dst_path)
def main_test7():
    all_files = get_sub_dir_file_list("/home/zengyifan/wujiahu/data/003.Cigar_Det/test/test_all")
    for img in all_files:
        # img_abs_path = img_path + "/{}".format(img)
        try:
            cv2img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)

            if cv2img is None:
                os.remove(img)
                print("[img is None]: Removed --> {}".format(img))
                continue

        except Exception as Error:
            print(Error)
def main_test8():
    all_dirs = get_sub_dir_list("/home/zengyifan/wujiahu/data/003.Cigar_Det/test/test_all")
    for d in all_dirs:
        convert_to_jpg_format(d)
def main_test9():
    img_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data/fire_smoke_20230203_0001538_hog.jpg"
    cv2img = cv2.imread(img_path)
    cv2img = np.transpose(cv2img, (2, 0, 1))
    img_tensor = torch.from_numpy(cv2img).float()
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    pool_out = torch.nn.MaxPool2d(3, 2, 0)(img_tensor)
    pool_out = pool_out.numpy()
    pool_out = np.squeeze(pool_out, axis=0)
    pool_out = np.transpose(pool_out, (1, 2, 0))
    pool_out = np.uint8(pool_out)
    cv2.imwrite("/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/wt/data/fire_smoke_20230203_0001538_hog_maxpool.jpg", pool_out)
def main_test10():
    """
    paper images. 20230414
    :return:
    """
    img_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/paper/1"
    save_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/paper/1_sz"
    os.makedirs(save_path, exist_ok=True)

    img_list = sorted(os.listdir(img_path))

    merged = np.empty(shape=(256, 0, 3))
    for i, img in enumerate(img_list):
        img_abs_path = img_path + "/{}".format(img)
        cv2img = cv2.imread(img_abs_path)

        cv2img = cv2.resize(cv2img, (256, 256))
        if i < 3:
            wht = np.ones(shape=(256, 10, 3)) * 255
            cv2img_cat = np.hstack((merged, cv2img, wht))
            merged = cv2img_cat
        else:
            merged = np.hstack((merged, cv2img))

    cv2.imwrite(save_path + "/merged.jpg", merged)
def main_test11():
    # img_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/paper/test_seg_output/unet_fire/bitwise_and_fire096_resz_croppped_2_filtered.jpg"
    # cv2img = cv2.imread(img_path)
    # cv2img_rsz = cv2.resize(cv2img, (214, 158))
    # cv2.imwrite("/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/paper/test_seg_output/unet_fire/bitwise_and_fire096_resz_croppped_2_filtered_rsz.jpg", cv2img_rsz)

    img_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/paper/test_seg_output/unet_fire/bitwise_and_fire096_resz_croppped_2_filtered_rsz.jpg"
    cv2img = cv2.imread(img_path)
    # sum_fire = 0
    fire_area = np.where((cv2img[:, :, 0] == 255) & (cv2img[:, :, 1] == 255) & (cv2img[:, :, 2] == 255))
    print(len(fire_area[0]))
def main_test12():
    data_path = "E:/Gosuncn/Patents/004-videos_video_frames_merged"
    image_path = data_path + "/images"
    json_path = data_path + "/jsons"
    save_path = data_path + "/output"
    os.makedirs(save_path, exist_ok=True)

    copy_image = True
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
              [255, 0, 255], [0, 255, 255], [128, 128, 0], [128, 0, 128], ]
    # for i in range(20):
    #     c = list(np.random.choice(range(256), size=3))
    #     if c not in colors:
    #         colors.append(c)

    img_list = os.listdir(image_path)
    for img in img_list:
        try:
            img_abs_path = image_path + "/{}".format(img)
            json_abs_path = json_path + "/{}.json".format(img)
            cv2img = cv2.imread(img_abs_path)
            json_ = json.load(open(json_abs_path, "r", encoding="utf-8"))
            if not json_: continue
            w, h = json_["width"], json_["height"]
            result_ = json_["step_1"]["result"]
            if not result_: continue

            # if copy_image:

            kpt_sorted = []

            len_result = len(result_)
            for i in range(len_result):
                for j in range(8):
                    x_ = int(round(result_[j]["x"]))
                    y_ = int(round(result_[j]["y"]))
                    # x_n = x_ / w
                    # y_n = y_ / h
                    attr_ = result_[j]["attribute"]

                    if int(attr_) == i + 1:
                        kpt_sorted.append([x_, y_])
                        break

            bbx_x1 = min(kpt_sorted[0][0], kpt_sorted[1][0])
            bbx_y1 = min(kpt_sorted[0][1], kpt_sorted[3][1])
            bbx_x2 = max(kpt_sorted[2][0], kpt_sorted[3][0])
            bbx_y2 = max(kpt_sorted[1][1], kpt_sorted[2][1])

            cv2.rectangle(cv2img, (bbx_x1 - 5, bbx_y1 - 5), (bbx_x2 + 5, bbx_y2 + 5), (255, 0, 255), 2)
            cv2.putText(cv2img, "{}: {}".format("door", 0.95), (bbx_x1, bbx_y1 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            for i in range(len_result):
                cv2.circle(cv2img, (kpt_sorted[i][0], kpt_sorted[i][1]), 2, (int(colors[i][0]), int(colors[i][1]), int(colors[i][2])), 15)

            # draw angle
            v15 = (kpt_sorted[4][0] - kpt_sorted[0][0], kpt_sorted[4][1] - kpt_sorted[0][1])
            v14 = (kpt_sorted[3][0] - kpt_sorted[0][0], kpt_sorted[3][1] - kpt_sorted[0][1])
            v26 = (kpt_sorted[5][0] - kpt_sorted[1][0], kpt_sorted[5][1] - kpt_sorted[1][1])
            v23 = (kpt_sorted[2][0] - kpt_sorted[1][0], kpt_sorted[2][1] - kpt_sorted[1][1])
            v47 = (kpt_sorted[6][0] - kpt_sorted[3][0], kpt_sorted[6][1] - kpt_sorted[3][1])
            v41 = (kpt_sorted[0][0] - kpt_sorted[3][0], kpt_sorted[0][1] - kpt_sorted[3][1])
            v38 = (kpt_sorted[7][0] - kpt_sorted[2][0], kpt_sorted[7][1] - kpt_sorted[2][1])
            v32 = (kpt_sorted[1][0] - kpt_sorted[2][0], kpt_sorted[1][1] - kpt_sorted[2][1])

            r15 = np.sqrt((kpt_sorted[4][0] - kpt_sorted[0][0]) ** 2 + (kpt_sorted[4][1] - kpt_sorted[0][1]) ** 2)
            r14 = np.sqrt((kpt_sorted[3][0] - kpt_sorted[0][0]) ** 2 + (kpt_sorted[3][1] - kpt_sorted[0][1]) ** 2)
            r26 = np.sqrt((kpt_sorted[5][0] - kpt_sorted[1][0]) ** 2 + (kpt_sorted[5][1] - kpt_sorted[1][1]) ** 2)
            r23 = np.sqrt((kpt_sorted[2][0] - kpt_sorted[1][0]) ** 2 + (kpt_sorted[2][1] - kpt_sorted[1][1]) ** 2)
            r47 = np.sqrt((kpt_sorted[6][0] - kpt_sorted[3][0]) ** 2 + (kpt_sorted[6][1] - kpt_sorted[3][1]) ** 2)
            r41 = np.sqrt((kpt_sorted[0][0] - kpt_sorted[3][0]) ** 2 + (kpt_sorted[0][1] - kpt_sorted[3][1]) ** 2)
            r38 = np.sqrt((kpt_sorted[7][0] - kpt_sorted[2][0]) ** 2 + (kpt_sorted[7][1] - kpt_sorted[2][1]) ** 2)
            r32 = np.sqrt((kpt_sorted[1][0] - kpt_sorted[2][0]) ** 2 + (kpt_sorted[1][1] - kpt_sorted[2][1]) ** 2)

            theta1514 = np.arccos((v15[0] * v14[0] + v15[1] * v14[1]) / (r15 * r14))
            theta2623 = np.arccos((v26[0] * v23[0] + v26[1] * v23[1]) / (r26 * r23))
            theta4741 = np.arccos((v47[0] * v41[0] + v47[1] * v41[1]) / (r47 * r41))
            theta3832 = np.arccos((v38[0] * v32[0] + v38[1] * v32[1]) / (r38 * r32))

            # cal line k
            k15 = (kpt_sorted[4][1] - kpt_sorted[0][1]) / (kpt_sorted[4][0] - kpt_sorted[0][0])
            k14 = (kpt_sorted[3][1] - kpt_sorted[0][1]) / (kpt_sorted[3][0] - kpt_sorted[0][0])
            k26 = (kpt_sorted[5][1] - kpt_sorted[1][1]) / (kpt_sorted[5][0] - kpt_sorted[1][0])
            k23 = (kpt_sorted[2][1] - kpt_sorted[1][1]) / (kpt_sorted[2][0] - kpt_sorted[1][0])
            k47 = (kpt_sorted[6][1] - kpt_sorted[3][1]) / (kpt_sorted[6][0] - kpt_sorted[3][0])
            k41 = (kpt_sorted[0][1] - kpt_sorted[3][1]) / (kpt_sorted[0][0] - kpt_sorted[3][0])
            k38 = (kpt_sorted[7][1] - kpt_sorted[2][1]) / (kpt_sorted[7][0] - kpt_sorted[2][0])
            k32 = (kpt_sorted[1][1] - kpt_sorted[2][1]) / (kpt_sorted[1][0] - kpt_sorted[2][0])

            b15 = kpt_sorted[4][1] - k15 * kpt_sorted[4][0]
            b14 = kpt_sorted[3][1] - k14 * kpt_sorted[3][0]
            b26 = kpt_sorted[5][1] - k26 * kpt_sorted[5][0]
            b23 = kpt_sorted[2][1] - k23 * kpt_sorted[2][0]
            b47 = kpt_sorted[6][1] - k47 * kpt_sorted[6][0]
            b41 = kpt_sorted[0][1] - k41 * kpt_sorted[0][0]
            b38 = kpt_sorted[7][1] - k38 * kpt_sorted[7][0]
            b32 = kpt_sorted[1][1] - k32 * kpt_sorted[1][0]

            expand_pixel = 200
            exp_p151 = (kpt_sorted[0][0] - expand_pixel, int(round(k15 * (kpt_sorted[0][0] - expand_pixel) + b15)))
            exp_p152 = (kpt_sorted[4][0] + expand_pixel, int(round(k15 * (kpt_sorted[4][0] + expand_pixel) + b15)))
            exp_p141 = (kpt_sorted[0][0] - expand_pixel, int(round(k14 * (kpt_sorted[0][0] - expand_pixel) + b14)))
            exp_p142 = (kpt_sorted[3][0] + expand_pixel, int(round(k14 * (kpt_sorted[3][0] + expand_pixel) + b14)))

            exp_p261 = (kpt_sorted[1][0] - expand_pixel, int(round(k26 * (kpt_sorted[1][0] - expand_pixel) + b26)))
            exp_p262 = (kpt_sorted[5][0] + expand_pixel, int(round(k26 * (kpt_sorted[5][0] + expand_pixel) + b26)))
            exp_p231 = (kpt_sorted[1][0] - expand_pixel, int(round(k23 * (kpt_sorted[1][0] - expand_pixel) + b23)))
            exp_p232 = (kpt_sorted[2][0] + expand_pixel, int(round(k23 * (kpt_sorted[2][0] + expand_pixel) + b23)))

            exp_p471 = (kpt_sorted[3][0] - expand_pixel, int(round(k47 * (kpt_sorted[3][0] - expand_pixel) + b47)))
            exp_p472 = (kpt_sorted[6][0] + expand_pixel, int(round(k47 * (kpt_sorted[6][0] + expand_pixel) + b47)))
            exp_p411 = (kpt_sorted[3][0] - expand_pixel, int(round(k41 * (kpt_sorted[3][0] - expand_pixel) + b41)))
            exp_p412 = (kpt_sorted[0][0] + expand_pixel, int(round(k41 * (kpt_sorted[0][0] + expand_pixel) + b41)))

            exp_p381 = (kpt_sorted[2][0] - expand_pixel, int(round(k38 * (kpt_sorted[2][0] - expand_pixel) + b38)))
            exp_p382 = (kpt_sorted[7][0] + expand_pixel, int(round(k38 * (kpt_sorted[7][0] + expand_pixel) + b38)))
            exp_p321 = (kpt_sorted[2][0] - expand_pixel, int(round(k32 * (kpt_sorted[2][0] - expand_pixel) + b32)))
            exp_p322 = (kpt_sorted[1][0] + expand_pixel, int(round(k32 * (kpt_sorted[1][0] + expand_pixel) + b32)))

            # cv2.line(cv2img, (kpt_sorted[0][0], kpt_sorted[0][1]), (kpt_sorted[4][0], kpt_sorted[4][1]), (128, 128, 0), 2)
            # cv2.line(cv2img, (kpt_sorted[0][0], kpt_sorted[0][1]), (kpt_sorted[3][0], kpt_sorted[3][1]), (128, 128, 0), 2)
            # cv2.line(cv2img, (kpt_sorted[1][0], kpt_sorted[1][1]), (kpt_sorted[5][0], kpt_sorted[5][1]), (128, 128, 0), 2)
            # cv2.line(cv2img, (kpt_sorted[1][0], kpt_sorted[1][1]), (kpt_sorted[2][0], kpt_sorted[2][1]), (128, 128, 0), 2)
            # cv2.line(cv2img, (kpt_sorted[3][0], kpt_sorted[3][1]), (kpt_sorted[6][0], kpt_sorted[6][1]), (128, 128, 0), 2)
            # cv2.line(cv2img, (kpt_sorted[3][0], kpt_sorted[3][1]), (kpt_sorted[0][0], kpt_sorted[0][1]), (128, 128, 0), 2)
            # cv2.line(cv2img, (kpt_sorted[2][0], kpt_sorted[2][1]), (kpt_sorted[7][0], kpt_sorted[7][1]), (128, 128, 0), 2)
            # cv2.line(cv2img, (kpt_sorted[2][0], kpt_sorted[2][1]), (kpt_sorted[1][0], kpt_sorted[1][1]), (128, 128, 0), 2)

            cv2.line(cv2img, (exp_p151[0], exp_p151[1]), (exp_p152[0], exp_p152[1]), (128, 128, 0), 4)
            cv2.line(cv2img, (exp_p141[0], exp_p141[1]), (exp_p142[0], exp_p142[1]), (128, 128, 0), 4)

            cv2.line(cv2img, (exp_p261[0], exp_p261[1]), (exp_p262[0], exp_p262[1]), (128, 128, 0), 4)
            cv2.line(cv2img, (exp_p231[0], exp_p231[1]), (exp_p232[0], exp_p232[1]), (128, 128, 0), 4)

            cv2.line(cv2img, (exp_p471[0], exp_p471[1]), (exp_p472[0], exp_p472[1]), (128, 128, 0), 4)
            cv2.line(cv2img, (exp_p411[0], exp_p411[1]), (exp_p412[0], exp_p412[1]), (128, 128, 0), 4)

            cv2.line(cv2img, (exp_p381[0], exp_p381[1]), (exp_p382[0], exp_p382[1]), (128, 128, 0), 4)
            cv2.line(cv2img, (exp_p321[0], exp_p321[1]), (exp_p322[0], exp_p322[1]), (128, 128, 0), 4)

            cv2.putText(cv2img, "{}: {:.2f}".format("thetaAEAD", theta1514 * 180 / np.pi), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(cv2img, "{}: {:.2f}".format("thetaBFBC", theta2623 * 180 / np.pi), (20, 50 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(cv2img, "{}: {:.2f}".format("thetaDGDA", theta4741 * 180 / np.pi), (20, 50 + 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(cv2img, "{}: {:.2f}".format("thetaCHCB", theta3832 * 180 / np.pi), (20, 50 + 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(cv2img, "{}: {}".format("Door Open State", int((2 * theta1514 * 180 / np.pi + theta2623 * 180 / np.pi) / 2)), (20, 50 + 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            cv2.imwrite("{}/{}".format(save_path, img), cv2img)

        except Exception as Error:
            print(Error)
def main_test13():
    img_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/Others/HK_Prison/Videos_Frames/Gray_videos_frames_20230531/images/Gray_Frames_Bg_0000000.jpg"
    cv2img = cv2.imread(img_path)
    b, g, r = cv2.split(cv2img)
    print(b == g)
    print(b == r)
    print(g == r)
    print(cv2img)
def main_test14(arg1: float = 2.3, arg2: int = 100) -> float:
    return arg1 * arg2
def main_test15():
    txt_file = "/home/zengyifan/wujiahu/data/003.Cigar_Det/cls/cls_train/train_base_v8.8/1_output/false.txt"
    data = open(txt_file, "r", encoding="utf-8")
    lines = data.readlines()
    for l in lines:
        os.remove(l.strip())
def main_test16():
    data_path = "/home/zengyifan/wujiahu/data/000.HK/20230330-20230803"
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/dst"
    os.makedirs(save_path, exist_ok=True)

    dirs = []
    dirs_datetime = os.listdir(data_path)
    for d in tqdm(dirs_datetime):
        dirs_ = os.listdir(data_path + "/{}".format(d))
        for dd in dirs_:
            if dd not in dirs:
                dirs.append(dd)
                os.makedirs(save_path + "/{}".format(dd), exist_ok=True)

    for d in tqdm(dirs_datetime):
        dirs_ = os.listdir(data_path + "/{}".format(d))
        for dd in dirs_:
            # if dd == "461" or dd == "312" or dd == "500" or dd == "501":
            if dd != "461" and dd != "312" and dd != "500" and dd != "501":

                dd_abs_path = data_path + "/{}/{}".format(d, dd)
                dd_dst_path = save_path + "/{}".format(dd)
                files = glob(dd_abs_path + "/*/*", recursive=True)
                for f in files:
                    if os.path.isfile(f):
                        fname = os.path.basename(f)
                        TD = f.split("/")[-2]
                        # if os.path.exists(dd_dst_path + "/{}".format(fname)):
                        #     rdn = np.random.rand()
                        #     print(rdn)
                        #     shutil.copy(f, dd_dst_path + "/{:.6f}_{}".format(rdn, fname))
                        # else:
                        #     shutil.copy(f, dd_dst_path + "/{}".format(fname))
                        shutil.copy(f, dd_dst_path + "/{}_{}_{}".format(d, TD, fname))
def main_test17():
    img_path = "/home/zengyifan/wujiahu/data/000.HK/dst/211"
    img_list = sorted(os.listdir(img_path))
    for img in tqdm(img_list):
        img_abs_path = img_path + "/{}".format(img)
        cv2img = cv2.imread(img_abs_path)
        imgsz = cv2img.shape[:2]
        if imgsz[0] > imgsz[1]:
            os.remove(img_abs_path)
def main_test18():
    # img_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/test/test_fire_smoke/test/20221111150039_8b46c5_62_0015550.jpg"
    img_path = "/home/zengyifan/wujiahu/data/000.HK/dst/461_res/C_Plus_Plus_det_output/461_images/crop_images/2/1.1/20230524_23_186278_0_1.1.jpg"
    cv2img = cv2.imread(img_path)

    roi = cv2img
    # roi = cv2img[100:1000, 1200:]
    # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/test.jpg", roi)

    hsvimg = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # cv::inRange(cropped_hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 50, 255), tmp_white);
    # cv::inRange(cropped_hsv, cv::Scalar(0, 0, 50), cv::Scalar(180, 50, 200), tmp_black);
    # cv::inRange(cropped_hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 50), tmp_gray);
    # cv::inRange(cropped_hsv, cv::Scalar(78, 43, 46), cv::Scalar(99, 255, 255), tmp_cyan);
    # cv::inRange(cropped_hsv, cv::Scalar(100, 43, 46), cv::Scalar(124, 255, 255), tmp_blue);
    # cv::inRange(cropped_hsv, cv::Scalar(95, 80, 160), cv::Scalar(125, 120, 255), tmp_HK_prison);
    tmp_white = cv2.inRange(hsvimg, (0, 0, 200), (180, 50, 255))
    tmp_black = cv2.inRange(hsvimg, (0, 0, 50), (180, 50, 200))
    tmp_gray = cv2.inRange(hsvimg, (0, 0, 0), (180, 255, 50))
    tmp_cyan = cv2.inRange(hsvimg, (78, 43, 46), (99, 255, 255))
    tmp_blue = cv2.inRange(hsvimg, (100, 43, 46), (124, 255, 255))
    tmp_HK_prison = cv2.inRange(hsvimg, (95, 80, 160), (125, 120, 255))
    # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res1.jpg", res1)
    # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res2.jpg", res2)
    # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res3.jpg", res3)
    # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res4.jpg", res4)
    # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res5.jpg", res5)
    # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res6.jpg", res6)

    mean_ = cv2.mean(hsvimg)
    print("cv2.mean(hsvimg): ", mean_)

    mean_tmp_white = cv2.mean(tmp_white)
    print("mean_tmp_white: ", mean_tmp_white)
    mean_tmp_black = cv2.mean(tmp_black)
    print("mean_tmp_black: ", mean_tmp_black)
    mean_tmp_gray = cv2.mean(tmp_gray)
    print("mean_tmp_gray: ", mean_tmp_gray)
    mean_tmp_cyan = cv2.mean(tmp_cyan)
    print("mean_tmp_cyan: ", mean_tmp_cyan)
    mean_tmp_blue = cv2.mean(tmp_blue)
    print("mean_tmp_blue: ", mean_tmp_blue)
    mean_tmp_HK_prison = cv2.mean(tmp_HK_prison)
    print("mean_tmp_HK_prison: ", mean_tmp_HK_prison)

    add1 = cv2.addWeighted(tmp_white, 1, tmp_black, 0.7, 0)
    add2 = cv2.addWeighted(add1, 1, tmp_gray, 0.7, 0)
    add3 = cv2.addWeighted(tmp_cyan, 0.25, tmp_blue, 0.25, 0)
    add4 = cv2.addWeighted(add2, 1, add3, 0.1, 0)
    add5 = cv2.addWeighted(add4, 1, tmp_HK_prison, 1, 0)
    mean_add5 = cv2.mean(add5)
    print("cv2.addWeighted------mean_add5: ", mean_add5)

    add1 = cv2.add(mean_tmp_white, mean_tmp_black)
    add2 = cv2.add(add1, mean_tmp_gray)
    add3 = cv2.add(add2, mean_tmp_cyan)
    add4 = cv2.add(add3, mean_tmp_blue)
    add5 = cv2.add(add4, mean_tmp_HK_prison)
    mean_add5 = cv2.mean(add1)
    print("===mean_add5: ", mean_add5)

    add2 = cv2.addWeighted(res1, 1, res2, 1, 0)
    cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/add2.jpg", add2)

    mean_add1 = cv2.mean(add1)
    print(mean_add1)
    mean_add2 = cv2.mean(add2)
    print(mean_add2)

    img_path = "/home/zengyifan/wujiahu/data/000.HK/dst/461_res/C_Plus_Plus_det_output/461_images/crop_images/2/1.1"
    img_list = os.listdir(img_path)
    for img in img_list:
        img_name = os.path.splitext(img)[0]
        img_abs_path = img_path + "/{}".format(img)
        cv2_img = cv2.imread(img_abs_path)

        hsvimg = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)

        # cv::inRange(cropped_hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 50, 255), tmp_white);
        # cv::inRange(cropped_hsv, cv::Scalar(0, 0, 50), cv::Scalar(180, 50, 200), tmp_black);
        # cv::inRange(cropped_hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 50), tmp_gray);
        # cv::inRange(cropped_hsv, cv::Scalar(78, 43, 46), cv::Scalar(99, 255, 255), tmp_cyan);
        # cv::inRange(cropped_hsv, cv::Scalar(100, 43, 46), cv::Scalar(124, 255, 255), tmp_blue);
        # cv::inRange(cropped_hsv, cv::Scalar(95, 80, 160), cv::Scalar(125, 120, 255), tmp_HK_prison);
        tmp_white = cv2.inRange(hsvimg, (0, 0, 200), (180, 50, 255))
        tmp_black = cv2.inRange(hsvimg, (0, 0, 50), (180, 50, 200))
        tmp_gray = cv2.inRange(hsvimg, (0, 0, 0), (180, 255, 50))
        tmp_cyan = cv2.inRange(hsvimg, (78, 43, 46), (99, 255, 255))
        tmp_blue = cv2.inRange(hsvimg, (100, 43, 46), (124, 255, 255))
        tmp_HK_prison = cv2.inRange(hsvimg, (95, 80, 160), (125, 120, 255))
        # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res1.jpg", res1)
        # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res2.jpg", res2)
        # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res3.jpg", res3)
        # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res4.jpg", res4)
        # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res5.jpg", res5)
        # cv2.imwrite("/home/zengyifan/wujiahu/data/000.HK/dst/461_res/res6.jpg", res6)

        add1 = cv2.addWeighted(tmp_white, 1, tmp_black, 0.45, 0)
        add2 = cv2.addWeighted(add1, 1, tmp_gray, 0.45, 0)
        add3 = cv2.addWeighted(tmp_cyan, 0.25, tmp_blue, 0.25, 0)
        add4 = cv2.addWeighted(add2, 1, add3, 0.1, 0)
        add5 = cv2.addWeighted(add4, 1, tmp_HK_prison, 1, 0)

        mean_add5 = cv2.mean(add5)
        print(img_abs_path, mean_add5)
def main_test19():
    img_path = "/home/zengyifan/wujiahu/data/000.HK/dst/461_res/C_Plus_Plus_det_output/461_images/crop_images/2/1.1"
    img_list = os.listdir(img_path)
    for img in img_list:
        img_name = os.path.splitext(img)[0]
        print(img_name)
        img_abs_path = img_path + "/{}".format(img)
        cv2_img = cv2.imread(img_abs_path)

        hsvimg = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)

        tmp_white = cv2.inRange(hsvimg, (0, 0, 220), (180, 30, 255))
        tmp_black = cv2.inRange(hsvimg, (0, 0, 0), (180, 255, 46))
        tmp_gray = cv2.inRange(hsvimg, (0, 0, 46), (180, 43, 220))
        tmp_cyan = cv2.inRange(hsvimg, (78, 43, 46), (99, 255, 255))
        tmp_blue = cv2.inRange(hsvimg, (100, 43, 46), (124, 255, 255))
        tmp_HK_prison1 = cv2.inRange(hsvimg, (95, 80, 160), (125, 120, 255))
        tmp_HK_prison2 = cv2.inRange(hsvimg, (30, 0, 120), (60, 40, 180))

        # cv::inRange(cropped_hsv, cv::Scalar(0, 0, 220), cv::Scalar(180, 30, 255), tmp_white);
        # cv::inRange(cropped_hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 46), tmp_black);
        # cv::inRange(cropped_hsv, cv::Scalar(0, 0, 46), cv::Scalar(180, 43, 220), tmp_gray);
        # cv::inRange(cropped_hsv, cv::Scalar(78, 43, 46), cv::Scalar(99, 255, 255), tmp_cyan);
        # cv::inRange(cropped_hsv, cv::Scalar(100, 43, 46), cv::Scalar(124, 255, 255), tmp_blue);
        # cv::inRange(cropped_hsv, cv::Scalar(95, 80, 160), cv::Scalar(125, 120, 255), tmp_HK_prison1);
        # cv::inRange(cropped_hsv, cv::Scalar(30, 0, 120), cv::Scalar(60, 40, 180), tmp_HK_prison2);
        # cv::addWeighted(tmp_white, 1.0, tmp_black, 0.45, 0.0, tmp_merge1);
        # cv::addWeighted(tmp_merge1, 1.0, tmp_gray, 0.45, 0.0, tmp_merge2);
        # cv::addWeighted(tmp_cyan, 0.25, tmp_blue, 0.25, 0.0, tmp_merge3);
        # cv::addWeighted(tmp_merge2, 1.0, tmp_merge3, 0.1, 0.0, tmp_merge4);
        # cv::addWeighted(tmp_HK_prison1, 1.0, tmp_HK_prison2, 0.0, 0.0, tmp_merge5);
        # cv::addWeighted(tmp_merge4, 1.0, tmp_merge5, 1.0, 0.0, tmp_merge);

        mean_ = cv2.mean(hsvimg)
        print("cv2.mean(hsvimg): ", mean_)

        mean_tmp_white = cv2.mean(tmp_white)
        print("mean_tmp_white: ", mean_tmp_white)
        mean_tmp_black = cv2.mean(tmp_black)
        print("mean_tmp_black: ", mean_tmp_black)
        mean_tmp_gray = cv2.mean(tmp_gray)
        print("mean_tmp_gray: ", mean_tmp_gray)
        mean_tmp_cyan = cv2.mean(tmp_cyan)
        print("mean_tmp_cyan: ", mean_tmp_cyan)
        mean_tmp_blue = cv2.mean(tmp_blue)
        print("mean_tmp_blue: ", mean_tmp_blue)
        mean_tmp_HK_prison1 = cv2.mean(tmp_HK_prison1)
        print("mean_tmp_HK_prison1: ", mean_tmp_HK_prison1)
        mean_tmp_HK_prison2 = cv2.mean(tmp_HK_prison2)
        print("mean_tmp_HK_prison2: ", mean_tmp_HK_prison2)

        add1 = cv2.addWeighted(tmp_white, 1, tmp_black, 0.45, 0)
        add2 = cv2.addWeighted(add1, 1, tmp_gray, 0.15, 0)
        add3 = cv2.addWeighted(tmp_cyan, 0.25, tmp_blue, 0.25, 0)
        add4 = cv2.addWeighted(add2, 1, add3, 0.1, 0)
        add5 = cv2.addWeighted(tmp_HK_prison1, 1, tmp_HK_prison2, 0.0, 0)
        add6 = cv2.addWeighted(add4, 1, add5, 1.0, 0)
        mean_add6 = cv2.mean(add6)
        print("cv2.addWeighted------mean_add6: ", mean_add6)

        add1 = cv2.add(mean_tmp_white, mean_tmp_black)
        add2 = cv2.add(add1, mean_tmp_gray)
        add3 = cv2.add(add2, mean_tmp_cyan)
        add4 = cv2.add(add3, mean_tmp_blue)
        add5 = cv2.add(add4, mean_tmp_HK_prison1)
        # add6 = cv2.add(add5, mean_tmp_HK_prison2)
        # mean_add6 = cv2.mean(add6)
        # print("===mean_add6: ", mean_add6)

        mean_add5 = cv2.mean(add5)
        print("===mean_add5: ", mean_add5)
        print("\n")
def main_test20():
    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/0-9_20230831"
    merged = np.empty(shape=(169, 0, 3))
    white = 255 * np.ones(shape=(169, 18, 3))
    for i in range(10):
        imgi_path = data_path + "/{}.png".format(i)
        imgi = cv2.imread(imgi_path)
        if imgi.shape[:2] != (169, 89):
            imgi = cv2.resize(imgi, (89, 169))

        imgi_ = 255 - imgi
        cv2.imwrite("{}/{}_.png".format(data_path, i), imgi_)

        merged = np.hstack((merged, imgi))
        if i != 9:
            merged = np.hstack((merged, white))

    merged2 = 255 - merged
    cv2.imwrite("{}/0-9.png".format(data_path), merged)
    cv2.imwrite("{}/0-9_2.png".format(data_path), merged2)
def main_test21():
    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/0-9_20230831"
    for i in range(10):
        imgi_path = data_path + "/{}_.png".format(i)
        imgi = cv2.imread(imgi_path)
        gray = cv2.cvtColor(imgi, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        imgsz = imgi.shape[:2]

        thr = cv2.resize(thr, (8, 14))
        thrsz = thr.shape[:2]

        sums = []
        for ci in range(thrsz[1]):
            sumi = 0
            for ri in range(thrsz[0]):
                # for ci in range(imgsz[1]):
                sumi += thr[ri, ci]

            sumi = sumi / 2500
            if sumi > 1: sumi = 1.0
            sums.append("{:.1f}".format(sumi))

        print("{}: {}".format(i, sums))
def main_test22():
    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/0-9_20230831"
    merged = np.empty(shape=(169, 0, 3))
    white = np.zeros(shape=(169, 23, 3))
    for idx, i in enumerate([9, 8, 5, 6, 2, 7, 4, 4]):
        imgi_path = data_path + "/{}_.png".format(i)
        imgi = cv2.imread(imgi_path)
        if imgi.shape[:2] != (169, 89):
            imgi = cv2.resize(imgi, (89, 169))

        # imgi_ = 255 - imgi
        # cv2.imwrite("{}/{}_.png".format(data_path, i), imgi_)

        merged = np.hstack((merged, imgi))
        if idx != 7:
            merged = np.hstack((merged, white))

    cv2.imwrite("{}/98562744.png".format(data_path), merged)
def main_test23():
    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/0-9_20230831"
    for i in [2, 5]:
        imgi_path = data_path + "/{}_.png".format(i)
        imgi = cv2.imread(imgi_path)
        gray = cv2.cvtColor(imgi, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        imgsz = imgi.shape[:2]

        thr = cv2.resize(thr, (8, 14))
        thr = thr[0:7, :]

        thrsz = thr.shape[:2]

        sums = []
        for ci in range(thrsz[1]):
            sumi = 0
            for ri in range(thrsz[0]):
                # for ci in range(imgsz[1]):
                sumi += thr[ri, ci]

            sumi = sumi / 2500
            if sumi > 1: sumi = 1.0
            sums.append("{:.1f}".format(sumi))

        print("{}: {}".format(i, sums))
def main_test24():
    label = {"0": ['1.0', '1.0', '0.4', '0.4', '0.4', '0.4', '1.0', '1.0'],
             "1": ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.8', '1.0'],
             "2": ['0.6', '0.7', '0.6', '0.6', '0.6', '0.6', '0.7', '0.5'],
             "3": ['0.0', '0.3', '0.6', '0.6', '0.6', '0.6', '1.0', '1.0'],
             "4": ['0.5', '0.4', '0.2', '0.2', '0.2', '0.2', '0.9', '1.0'],
             "5": ['0.5', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.5'],
             "6": ['1.0', '1.0', '0.6', '0.6', '0.6', '0.6', '0.6', '0.5'],
             "7": ['0.0', '0.1', '0.2', '0.2', '0.2', '0.2', '1.0', '1.0'],
             "8": ['1.0', '1.0', '0.6', '0.6', '0.6', '0.6', '1.0', '1.0'],
             "9": ['0.5', '0.6', '0.6', '0.6', '0.6', '0.6', '1.0', '1.0'],
             "2h": ['0.0', '0.1', '0.3', '0.3', '0.3', '0.3', '0.6', '0.5'],
             "5h": ['0.5', '0.5', '0.3', '0.3', '0.3', '0.3', '0.1', '0.0']
             }

    img_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/0-9_20230831/98562744.png"
    cv2img = cv2.imread(img_path)

    t1 = time.time()

    cv2img = cv2.resize(cv2img, (8 * 8 + 2 * 7, 14))
    gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    thrsz = thr.shape[:2]

    sums = []
    sums_2500 = []
    for ci in range(thrsz[1]):
        sumi = 0
        for ri in range(thrsz[0]):
            sumi += thr[ri, ci]
        sums_2500.append(sumi)
        sumi = sumi / 2500
        if sumi > 1: sumi = 1.0
        sums.append("{:.1f}".format(sumi))

    # print("{}: {}".format("98562744", sums))

    sums_2500_arr = np.array(sums_2500[:-8]).reshape(-1, 10)[:, :-2].astype(np.float32)
    sums_2500_arr = np.vstack((sums_2500_arr, np.array(sums_2500[-8:]))).astype(np.float32)
    sums_arr = np.array(sums[:-8]).reshape(-1, 10)[:, :-2].astype(np.float32)
    sums_arr = np.vstack((sums_arr, np.array(sums[-8:]))).astype(np.float32)
    print(sums_2500_arr)
    print(sums_arr)

    results = ""
    for i in range(sums_arr.shape[0]):
        minus_sum = []
        for j in range(10):
            resi = sums_arr[i] - np.array(label[str(j)], dtype=np.float32)
            # resi_abs_sum = sum([abs(ii) for ii in resi])
            resi_abs_sum = sum([ii ** 2 for ii in resi])
            minus_sum.append(resi_abs_sum)

        print("minus_sum: ", minus_sum)

        index = minus_sum.index(min(minus_sum))
        results += str(index)

    print(results)

    for idx, ri in enumerate(results):
        if ri == "2" or ri == "5":
            if idx != len(results) - 1:
                crop_img = thr[0:7, 8 * idx + 2 * idx:(8 * idx + 2 * idx + 8)]
                thrsz = crop_img.shape[:2]
                sums_2500 = []
                sums = []
                for ci in range(thrsz[1]):
                    sumi = 0
                    for ri in range(thrsz[0]):
                        sumi += crop_img[ri, ci]
                    sums_2500.append(sumi)
                    sumi = sumi / 2500
                    if sumi > 1: sumi = 1.0
                    sums.append("{:.1f}".format(sumi))

                print("sums_2500 -> ", sums_2500)
                print("sums -> ", sums)

                minus_sum_ = []
                for jh in ["2h", "5h"]:
                    resi = np.array(sums, dtype=np.float32) - np.array(label[jh], dtype=np.float32)
                    # resi_abs_sum = sum([abs(ii) for ii in resi])
                    resi_abs_sum = sum([ii ** 2 for ii in resi])
                    minus_sum_.append(resi_abs_sum)

                print("minus_sum_: ", minus_sum_)
                index_ = minus_sum_.index(min(minus_sum_))
                if index_ == 0: results = results[:idx] + "2" + results[idx + 1:]
                if index_ == 1: results = results[:idx] + "5" + results[idx + 1:]

    print(results)

    t2 = time.time()
    print(t2 - t1)
def main_test25():
    label = {"0": ['1.0', '1.0', '0.4', '0.4', '0.4', '0.4', '1.0', '1.0'],
             "1": ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.8', '1.0'],
             "2": ['0.6', '0.7', '0.6', '0.6', '0.6', '0.6', '0.7', '0.5'],
             "3": ['0.0', '0.3', '0.6', '0.6', '0.6', '0.6', '1.0', '1.0'],
             "4": ['0.5', '0.4', '0.2', '0.2', '0.2', '0.2', '0.9', '1.0'],
             "5": ['0.5', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.5'],
             "6": ['1.0', '1.0', '0.6', '0.6', '0.6', '0.6', '0.6', '0.5'],
             "7": ['0.0', '0.1', '0.2', '0.2', '0.2', '0.2', '1.0', '1.0'],
             "8": ['1.0', '1.0', '0.6', '0.6', '0.6', '0.6', '1.0', '1.0'],
             "9": ['0.5', '0.6', '0.6', '0.6', '0.6', '0.6', '1.0', '1.0'],
             "2h": ['0.0', '0.1', '0.3', '0.3', '0.3', '0.3', '0.6', '0.5'],
             "5h": ['0.5', '0.5', '0.3', '0.3', '0.3', '0.3', '0.1', '0.0']
             }

    img_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/0-9_20230831/6_.png"
    cv2img = cv2.imread(img_path)
    cv2img_cp = cv2img.copy()
    cv2imgsz = cv2img.shape[:2]

    cv2img = cv2.resize(cv2img, (8, 14))
    gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    thrsz = thr.shape[:2]

    sums = []
    sums_2500 = []
    for ci in range(thrsz[1]):
        sumi = 0
        for ri in range(thrsz[0]):
            sumi += thr[ri, ci]

        sums_2500.append(sumi)
        sumi = sumi / 2500
        if sumi > 1: sumi = 1.0
        sums.append("{:.1f}".format(sumi))

    sums_arr = np.array([sums], dtype=np.float32)
    print(sums_2500)
    print(sums_arr)

    results = ""
    for i in range(sums_arr.shape[0]):
        minus_sum = []
        for j in range(10):
            resi = sums_arr[i] - np.array(label[str(j)], dtype=np.float32)
            resi_abs_sum = sum([abs(ii) for ii in resi])
            minus_sum.append(resi_abs_sum)

        index = minus_sum.index(min(minus_sum))
        results += str(index)

    print(results)

    for ii in range(9):
        cv2.line(cv2img_cp, (cv2imgsz[1] // 8 * ii, 0), (cv2imgsz[1] // 8 * ii, cv2imgsz[0]), (255, 0, 255), 1)

    cv2.imwrite("/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/0-9_20230831/6_line.png", cv2img_cp)
def main_test26():
    # data_path ="/home/zengyifan/wujiahu/data/010.Digital_Rec/test/changchun"
    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/llj_0-9/AUG/aug_20230918"
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_cropped".format(dir_name)
    os.makedirs(save_path, exist_ok=True)
    img_list = sorted(os.listdir(data_path))
    for img in tqdm(img_list):
        img_name = os.path.splitext(img)[0]
        img_abs_path = data_path + "/{}".format(img)
        cv2img = cv2.imread(img_abs_path)
        imgsz = cv2img.shape[:2]
        # imgnew = cv2img[340:430, 605:915]
        imgnew = cv2img[int(round(0.4 * imgsz[0])):, :]
        # imgnew2 = cv2img[340:430, 605 + 68:915]
        cv2.imwrite("{}/{}.jpg".format(save_path, img_name), imgnew)
        # cv2.imwrite("{}/{}_2.jpg".format(save_path, img_name), imgnew2)
def main_test27():
    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/AbC"
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_white".format(dir_name)
    os.makedirs(save_path, exist_ok=True)
    img_list = sorted(os.listdir(data_path))
    for img in tqdm(img_list):
        img_name = os.path.splitext(img)[0]
        img_abs_path = data_path + "/{}".format(img)
        cv2img = cv2.imread(img_abs_path)
        gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        thr = cv2.resize(thr, (89, 169))
        cv2.imwrite("{}/{}".format(save_path, img), thr)
def main_test28():
    # img_A_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/AbC/20230904/1691982698.0490382.jpg"
    # img_b_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/AbC/20230904/1691982689.9716394.jpg"
    # img_C_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/AbC/20230904/1691982690.069378.jpg"

    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/AbC/2_merged"
    img_list = sorted(os.listdir(data_path))
    for img in tqdm(img_list):
        img_name = os.path.splitext(img)[0]
        img_abs_path = data_path + "/{}".format(img)

        cv2img = cv2.imread(img_abs_path)
        hsvimg = cv2.cvtColor(cv2img, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsvimg, (156, 43, 46), (180, 255, 255))
        # red = cv2.inRange(hsvimg, (), ())
        # cv2.imwrite(img_A_path.replace(".jpg", "_red.jpg"), red)

        conts, hierarchy = cv2.findContours(red.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # maxc = max(conts, key=cv2.contourArea)
        cnts = sorted(conts, key=cv2.contourArea, reverse=True)

        for i, c in enumerate(cnts):
            bbox = cv2.boundingRect(c)
            outimg_crop = cv2img[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
            area = bbox[2] * bbox[3]
            if area > 3000 and area < 3800:
                cv2.imwrite("{}/{}_{}.jpg".format(data_path, img_name, i), outimg_crop)
def main_test29():
    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/AbC/0-9_AbC_cp"
    img_list = sorted(os.listdir(data_path))
    for img in tqdm(img_list):
        img_name = os.path.splitext(img)[0]
        img_abs_path = data_path + "/{}".format(img)
        label_str = ""
        if "AN" in img_name:
            label_str = "A"
        elif "bN" in img_name:
            label_str = "b"
        elif "CN" in img_name:
            label_str = "C"
        elif "0N" in img_name:
            label_str = "0"
        elif "1N" in img_name:
            label_str = "1"
        elif "2N" in img_name:
            label_str = "2"
        elif "3N" in img_name:
            label_str = "3"
        elif "4N" in img_name:
            label_str = "4"
        elif "5N" in img_name:
            label_str = "5"
        elif "6N" in img_name:
            label_str = "6"
        elif "7N" in img_name:
            label_str = "7"
        elif "8N" in img_name:
            label_str = "8"
        elif "9N" in img_name:
            label_str = "9"
        elif "0.N" in img_name:
            label_str = "0."
        elif "1.N" in img_name:
            label_str = "1."
        elif "2.N" in img_name:
            label_str = "2."
        elif "3.N" in img_name:
            label_str = "3."
        elif "4.N" in img_name:
            label_str = "4."
        elif "5.N" in img_name:
            label_str = "5."
        elif "6.N" in img_name:
            label_str = "6."
        elif "7.N" in img_name:
            label_str = "7."
        elif "8.N" in img_name:
            label_str = "8."
        elif "9.N" in img_name:
            label_str = "9."
        else:
            print("Error!")

        rdmnum = np.random.random()
        new_name = "0-9_AbC_20230906_{}={}.jpg".format(str(rdmnum).replace(".", ""), label_str)
        img_dst_path = data_path + "/{}".format(new_name)
        os.rename(img_abs_path, img_dst_path)
def main_test30(data_path):
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_renamed".format(dir_name)
    os.makedirs(save_path, exist_ok=True)
    file_list = sorted(os.listdir(data_path))
    for f in tqdm(file_list):
        try:
            f_abs_path = data_path + "/{}".format(f)
            # # cv2img = cv2.imread(f_abs_path)
            # if "=" in f:
            #     f_name = os.path.splitext(f)[0].split("=")[1]
            f_new_name = data_path + "/{}".format(f.replace(" ", ""))
            os.rename(f_abs_path, f_new_name)

        except Exception as Error:
            print(Error)
def main_test31(use_glob=True, n_subdir=3):
    syn90k_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/CRNN_OpenDataset/Syn90k/Syn90k"

    if use_glob:
        dir_list = glob(syn90k_path + "{}/*".format(n_subdir * "/*"), recursive=True)
        for f in dir_list:
            if os.path.isfile(f):
                fname = os.path.basename(f)
                img_name, suffix = os.path.splitext(fname)[0], os.path.splitext(fname)[1]
                new_name = "{}_{}={}{}".format(img_name.split("_")[0], img_name.split("_")[2], img_name.split("_")[1], suffix)
                f_dst_path = f.replace(fname, new_name)
                os.rename(f, f_dst_path)
def main_test32(data_path):
    dir_name = os.path.basename(data_path)
    file_list = sorted(os.listdir(data_path))
    labels = "0123456789.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_moved".format(dir_name)
    os.makedirs(save_path, exist_ok=True)
    file_list = sorted(os.listdir(data_path))

    for f in tqdm(file_list):
        try:
            f_abs_path = data_path + "/{}".format(f)
            f_dst_path = save_path + "/{}".format(f)
            fname = os.path.basename(f)
            img_name, suffix = os.path.splitext(fname)[0], os.path.splitext(fname)[1]
            label = img_name.split("=")[1]
            for l in label:
                if l not in labels:
                    shutil.move(f_abs_path, f_dst_path)
                    break

            cv2img = cv2.imread(f_abs_path)
            imgsz = cv2img.shape[:2]
            if imgsz[0] > imgsz[1]:
                shutil.move(f_abs_path, f_dst_path)
        except Exception as Error:
            print(Error)
def main_test33():
    syn90k_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/test/crnn/test"

    dir_list = glob(syn90k_path + "/*", recursive=True)
    for f in dir_list:
        if os.path.isfile(f):
            fname = os.path.basename(f)
            # img_name, suffix = os.path.splitext(fname)[0], os.path.splitext(fname)[1]
            # name0, label = img_name.split("=")[0], img_name.split("=")[1]
            # new_name = name
            f_dst_path = f.replace(fname, fname.replace(" ", "").replace("R", "A"))
            os.rename(f, f_dst_path)
def main_test34(bg_img_path, fg_path, save_path, dstsz=(28, 45), aug_flag=True, m_ratio=0.025, max_num=9):
    """
    :param bg_img_path:
    :param fg_path:
    :param save_path:
    :param dstsz: [w, h]
    :param aug_flag:
    :param m_ratio:
    :param max_num:
    :return:
    """
    normal_save_path = "{}/normal".format(save_path)
    aug_save_path = "{}/aug".format(save_path)
    os.makedirs(normal_save_path, exist_ok=True)
    os.makedirs(aug_save_path, exist_ok=True)

    datetime = time.strftime("%Y%m%d", time.localtime(time.time()))

    color_db = [
        np.array([0, 0, 250], dtype=np.float32),  # red
        np.array([0, 250, 0], dtype=np.float32),  # green
        np.array([250, 250, 250], dtype=np.float32),  # white
        np.array([10, 10, 10], dtype=np.float32)  # black
    ]
    number_color = random.choice(color_db)
    # ==============================================================================================================================================
    fname_rdm = np.random.random()
    digital_num = np.random.randint(1, max_num + 1)
    x_space = np.random.randint(1, dstsz[0] + (9 - digital_num) * dstsz[0])

    fg_num_path = fg_path + "/num"
    fg_numdot_path = fg_path + "/numdot"
    fg_num_list = sorted(os.listdir(fg_num_path))
    fg_numdot_list = sorted(os.listdir(fg_numdot_path))

    selected_numdot_list = ["{}/{}".format(fg_numdot_path, random.sample(fg_numdot_list, 1)[0])]
    selected_num_list = []
    for si in fg_num_list:
        selected_num_list.append("{}/{}".format(fg_num_path, si))
    selected_all_abs_path = selected_num_list + selected_numdot_list
    selected_all = random.sample(selected_all_abs_path, digital_num)

    bg_img = cv2.imread(bg_img_path)
    bgsz = bg_img.shape[:2]

    label_str = ""
    for i in range(digital_num):
        fgi_abs_path = random.sample(selected_all, 1)[0]

        if "0.N" in fgi_abs_path or "1.N" in fgi_abs_path or "2.N" in fgi_abs_path or "3.N" in fgi_abs_path or "4.N" in fgi_abs_path or "5.N" in fgi_abs_path or "6.N" in fgi_abs_path or "7.N" in fgi_abs_path or "8.N" in fgi_abs_path or "9.N" in fgi_abs_path:
            selected_all.remove(fgi_abs_path)

        fgi_base_name = os.path.basename(fgi_abs_path)
        fgi_name = os.path.splitext(fgi_base_name)[0]
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
        # label_str += fgi_name

        # fg_abs_path = fg_path + "/{}".format(fgi)
        # fg_img = cv2.imread(fg_abs_path)
        # fg_img_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
        # ret, thr = cv2.threshold(fg_img_gray, 80, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("{}/gray_20230912_{}.jpg".format(save_path, fgi), thr)

        fg_img = cv2.imread(fgi_abs_path)
        fgsz = fg_img.shape[:2]
        r = dstsz[1] / fgsz[0]
        # fg_img = cv2.resize(fg_img, (int(round(fgsz[1] * r)), 45))
        # dstsz = (28, 45)
        fg_img = cv2.resize(fg_img, dstsz)
        fgsz = fg_img.shape[:2]
        fg_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(fg_gray, 180, 255, cv2.THRESH_BINARY_INV)

        # mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
        # out = cv2.seamlessClone(fg_img, bg_img, mask_img, (30 + fgsz[1] * i, 45 // 2), cv2.NORMAL_CLONE)  #

        # for ii in range(dstsz[0]):
        #     for jj in range(dstsz[1]):
        #         if thr[jj, ii] == 255:
        #             try:
        #                 fg_img[jj, ii] = fg_img[jj, ii] + number_color
        #                 fg_img[jj, ii] = np.clip(fg_img[jj, ii], 0, 255)
        #             except Exception as Error:
        #                 print(Error)

        upper_loew_space = (bgsz[0] - dstsz[1]) // 2
        bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])] = bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])] + fg_img

        for ii in range(dstsz[0]):
            for jj in range(dstsz[1]):
                if thr[jj, ii] == 255:
                    bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])][jj, ii] = 0 + np.random.randint(0, 50)

    aug_rdm = np.random.random()
    if aug_rdm > 0.5:
        noise_aug = NoiseAug(ratio=0.9)
        blur_rdm = np.random.random()
        if blur_rdm < 0.5:
            blur_aug = BlurAug(type="EASY", ratio=0.9)
        else:
            blur_aug = BlurAug(type="HARD", ratio=0.9)
        hsv_aug = HSVAug(hgain=0.2, sgain=0.7, vgain=0.5, ratio=0.9)
        bg_img = noise_aug(bg_img)
        bg_img = blur_aug(bg_img)
        bg_img = hsv_aug(bg_img)
        bg_img_aug = doing_aug(bg_img)

        cv2.imwrite("{}/{}_aug_{}_{}={}.jpg".format(aug_save_path, datetime, str(fname_rdm).replace(".", ""), fgi_name, label_str), bg_img_aug)
    else:
        cv2.imwrite("{}/{}_{}_{}={}.jpg".format(normal_save_path, datetime, str(fname_rdm).replace(".", ""), fgi_name, label_str), bg_img)
def main_test35():
    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/crnn_labelbee_data/20230912/crnn_labelbee_data_cropped_random_selected_2824"
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_selected".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    dir_list = glob(data_path + "/*", recursive=True)
    for f in dir_list:
        if os.path.isfile(f):
            fname = os.path.basename(f)
            img_name, suffix = os.path.splitext(fname)[0], os.path.splitext(fname)[1]
            name0, label = img_name.split("=")[0], img_name.split("=")[1]
            # new_name = name
            # f_dst_path = f.replace(fname, fname.replace(" ", "").replace("R", "A"))
            f_dst_path = save_path + "/{}".format(fname)
            # os.rename(f, f_dst_path)
            if len(label) == 0:
                shutil.move(f, f_dst_path)
def main_test36():
    # img_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/changchundianli_and_shiyougongye_data/shiyougongyexunjian/test/010_20230821_0000009=0.0514.jpg"
    # cv2img = cv2.imread(img_path)
    # imgsz = cv2img.shape[:2]
    # res = cv2img[:int(0.58 * imgsz[0]), :int(1.0 * imgsz[1])]
    # cv2.imwrite(img_path.replace(".jpg", "_cropped.jpg"), res)
    # # cv2.imwrite(img_path, res)

    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/fake_number/old/big_images/llj_m3_20000"
    save_path = make_save_path(data_path, dir_name_add_str="cropped")
    file_list = get_file_list(data_path)
    for f in tqdm(file_list):
        base_name, file_name, suffix = get_baseName_fileName_suffix(f)
        f_abs_path = data_path + "/{}".format(f)
        if "=" in file_name:
            name0, label = file_name.split("=")[0], file_name.split("=")[1]
            cv2img = cv2.imread(f_abs_path)
            imgsz = cv2img.shape[:2]
            res = cv2img[int(0.63 * imgsz[0]):, :int(1.0 * imgsz[1])]
            cv2.imwrite("{}/20230916_{}_{}={}.jpg".format(save_path, str(np.random.random()).replace(".", ""), name0, label), res)
        else:
            print(f)

    # data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/changchundianli_and_shiyougongye_data/changchundianlixunjian/train/changchun_orig_cropped"
    # save_path = make_save_path(data_path, dir_name_add_str="selected")
    # file_list = get_file_list(data_path)
    # for f in tqdm(file_list):
    #     base_name, file_name, suffix = get_baseName_fileName_suffix(f)
    #     f_abs_path = data_path + "/{}".format(f)
    #     # if "=" in file_name:
    #     #     pass
    #         # name0, label = file_name.split("=")[0], file_name.split("=")[1]
    #         # cv2img = cv2.imread(f_abs_path)
    #         # imgsz = cv2img.shape[:2]
    #         # res = cv2img[int(0.43 * imgsz[0]):, :int(1.0 * imgsz[1])]
    #         # cv2.imwrite("{}/20230916_{}_{}={}.jpg".format(save_path, str(np.random.random()).replace(".", ""), name0, label), res)
    #     # else:
    #     if "=" not in file_name:
    #         print(f)
    #         f_dst_path = save_path + "/{}".format(f)
    #         shutil.move(f_abs_path, f_dst_path)
def main_test38():
    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/llj_0-9/llj_0-9_new"
    save_path = make_save_path(data_path, dir_name_add_str="THR_INV")
    file_list = get_file_list(data_path)
    for f in tqdm(file_list):
        base_name, file_name, suffix = get_baseName_fileName_suffix(f)
        f_abs_path = data_path + "/{}".format(f)
        cv2img = cv2.imread(f_abs_path)
        gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
        ret, thr = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        f_dst_path = save_path + "/{}".format(f)
        cv2.imwrite(f_dst_path, thr)
def main_test39(bg_img_path, fg_path, save_path, dstsz=(28, 45), aug_flag=True, m_ratio=0.025, max_num=9):
    """
    :param bg_img_path:
    :param fg_path:
    :param save_path:
    :param dstsz: [w, h]
    :param aug_flag:
    :param m_ratio:
    :param max_num:
    :return:
    """
    normal_save_path = "{}/normal".format(save_path)
    aug_save_path = "{}/aug".format(save_path)
    os.makedirs(normal_save_path, exist_ok=True)
    os.makedirs(aug_save_path, exist_ok=True)

    datetime = time.strftime("%Y%m%d", time.localtime(time.time()))

    color_db = [
        np.array([0, 0, 250], dtype=np.float32),  # red
        np.array([0, 250, 0], dtype=np.float32),  # green
        np.array([250, 250, 250], dtype=np.float32),  # white
        np.array([10, 10, 10], dtype=np.float32)  # black
    ]
    number_color = random.choice(color_db)
    # ==============================================================================================================================================
    fname_rdm = np.random.random()
    digital_num = np.random.randint(1, max_num + 1)
    x_space = np.random.randint(1, dstsz[0] + (9 - digital_num) * dstsz[0])

    fg_num_path = fg_path + "/num"
    # fg_numdot_path = fg_path + "/numdot"
    fg_num_list = sorted(os.listdir(fg_num_path))
    # fg_numdot_list = sorted(os.listdir(fg_numdot_path))

    # selected_numdot_list = ["{}/{}".format(fg_numdot_path, random.sample(fg_numdot_list, 1)[0])]
    selected_num_list = []
    for si in fg_num_list:
        selected_num_list.append("{}/{}".format(fg_num_path, si))
    # selected_all_abs_path = selected_num_list + selected_numdot_list
    # selected_all = random.sample(selected_all_abs_path, digital_num)

    selected_all = random.sample(selected_num_list, digital_num)

    bg_img = cv2.imread(bg_img_path)
    bgsz = bg_img.shape[:2]

    label_str = ""
    for i in range(digital_num):
        fgi_abs_path = random.sample(selected_all, 1)[0]

        # if "0.N" in fgi_abs_path or "1.N" in fgi_abs_path or "2.N" in fgi_abs_path or "3.N" in fgi_abs_path or "4.N" in fgi_abs_path or "5.N" in fgi_abs_path or "6.N" in fgi_abs_path or "7.N" in fgi_abs_path or "8.N" in fgi_abs_path or "9.N" in fgi_abs_path:
        #     selected_all.remove(fgi_abs_path)

        fgi_base_name = os.path.basename(fgi_abs_path)
        fgi_name = os.path.splitext(fgi_base_name)[0]
        if "AN" in fgi_name:
            label_str += "A"
        elif "bN" in fgi_name:
            label_str += "b"
        elif "CN" in fgi_name:
            label_str += "C"
        elif "0" in fgi_name:
            label_str += "0"
        elif "1" in fgi_name:
            label_str += "1"
        elif "2" in fgi_name:
            label_str += "2"
        elif "3" in fgi_name:
            label_str += "3"
        elif "4" in fgi_name:
            label_str += "4"
        elif "5" in fgi_name:
            label_str += "5"
        elif "6" in fgi_name:
            label_str += "6"
        elif "7" in fgi_name:
            label_str += "7"
        elif "8" in fgi_name:
            label_str += "8"
        elif "9" in fgi_name:
            label_str += "9"
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
        # label_str += fgi_name

        # fg_abs_path = fg_path + "/{}".format(fgi)
        # fg_img = cv2.imread(fg_abs_path)
        # fg_img_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
        # ret, thr = cv2.threshold(fg_img_gray, 80, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("{}/gray_20230912_{}.jpg".format(save_path, fgi), thr)

        fg_img = cv2.imread(fgi_abs_path)
        fgsz = fg_img.shape[:2]
        r = dstsz[1] / fgsz[0]
        # fg_img = cv2.resize(fg_img, (int(round(fgsz[1] * r)), 45))
        # dstsz = (28, 45)
        fg_img = cv2.resize(fg_img, dstsz)
        fgsz = fg_img.shape[:2]
        # fg_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
        # ret, thr = cv2.threshold(fg_gray, 180, 255, cv2.THRESH_BINARY_INV)

        # mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
        # out = cv2.seamlessClone(fg_img, bg_img, mask_img, (30 + fgsz[1] * i, 45 // 2), cv2.NORMAL_CLONE)  #

        # for ii in range(dstsz[0]):
        #     for jj in range(dstsz[1]):
        #         if thr[jj, ii] == 255:
        #             try:
        #                 fg_img[jj, ii] = fg_img[jj, ii] + number_color
        #                 fg_img[jj, ii] = np.clip(fg_img[jj, ii], 0, 255)
        #             except Exception as Error:
        #                 print(Error)

        upper_loew_space = (bgsz[0] - dstsz[1]) // 2
        bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])] = bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])] + fg_img

        for ii in range(dstsz[0]):
            for jj in range(dstsz[1]):
                if thr[jj, ii] == 255:
                    bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])][jj, ii] = 0 + np.random.randint(0, 50)

    aug_rdm = np.random.random()
    if aug_rdm > 0.5:
        noise_aug = NoiseAug(ratio=0.9)
        blur_rdm = np.random.random()
        if blur_rdm < 0.5:
            blur_aug = BlurAug(type="EASY", ratio=0.9)
        else:
            blur_aug = BlurAug(type="HARD", ratio=0.9)
        hsv_aug = HSVAug(hgain=0.2, sgain=0.7, vgain=0.5, ratio=0.9)
        bg_img = noise_aug(bg_img)
        bg_img = blur_aug(bg_img)
        bg_img = hsv_aug(bg_img)
        bg_img_aug = doing_aug(bg_img)

        cv2.imwrite("{}/{}_aug_{}_{}={}.jpg".format(aug_save_path, datetime, str(fname_rdm).replace(".", ""), fgi_name, label_str), bg_img_aug)
    else:
        cv2.imwrite("{}/{}_{}_{}={}.jpg".format(normal_save_path, datetime, str(fname_rdm).replace(".", ""), fgi_name, label_str), bg_img)
def main_test39_(bg_img_path, fg_path, save_path, dstsz=(28, 45), aug_flag=True, m_ratio=0.025, max_num=9):
    """
    :param bg_img_path:
    :param fg_path:
    :param save_path:
    :param dstsz: [w, h]
    :param aug_flag:
    :param m_ratio:
    :param max_num:
    :return:
    """
    normal_save_path = "{}/normal".format(save_path)
    aug_save_path = "{}/aug".format(save_path)
    os.makedirs(normal_save_path, exist_ok=True)
    os.makedirs(aug_save_path, exist_ok=True)

    datetime = time.strftime("%Y%m%d", time.localtime(time.time()))

    color_db = [
        np.array([0, 0, 250], dtype=np.float32),  # red
        np.array([0, 250, 0], dtype=np.float32),  #
        np.array([250, 0, 0], dtype=np.float32),  # green
        # np.array([250, 250, 250], dtype=np.float32),  # white
        np.array([250, 250, 0], dtype=np.float32),  # green
        np.array([250, 0, 250], dtype=np.float32),  # green
        np.array([0, 250, 250], dtype=np.float32),  # green
        # np.array([10, 10, 10], dtype=np.float32)  # black
    ]
    number_color = random.choice(color_db)
    # ==============================================================================================================================================
    fname_rdm = np.random.random()
    digital_num = np.random.randint(1, max_num + 1)
    x_space = np.random.randint(1, dstsz[0] + (9 - digital_num) * dstsz[0])

    # fg_num_path = fg_path + "/num"
    fg_num_path = fg_path
    # fg_numdot_path = fg_path + "/numdot"
    fg_num_list = sorted(os.listdir(fg_num_path))
    # fg_numdot_list = sorted(os.listdir(fg_numdot_path))

    # selected_numdot_list = ["{}/{}".format(fg_numdot_path, random.sample(fg_numdot_list, 1)[0])]
    selected_num_list = []
    for si in fg_num_list:
        selected_num_list.append("{}/{}".format(fg_num_path, si))
    # selected_all_abs_path = selected_num_list + selected_numdot_list
    # selected_all = random.sample(selected_all_abs_path, digital_num)

    selected_all = random.sample(selected_num_list, digital_num)

    # ===============================================
    # # fg_num_path = fg_path + "/num"
    # fg_num_path = fg_path
    # fg_numdot_path = os.path.abspath(os.path.join(fg_path, "..")) + "/0-9_output_ud_stack"
    # fg_num_list = sorted(os.listdir(fg_num_path))
    # fg_numdot_list = sorted(os.listdir(fg_numdot_path))
    #
    # selected_numdot_list = ["{}/{}".format(fg_numdot_path, random.sample(fg_numdot_list, 1)[0])]
    # selected_num_list = []
    # for si in fg_num_list:
    #     selected_num_list.append("{}/{}".format(fg_num_path, si))
    # selected_all_abs_path = selected_num_list + selected_numdot_list
    # selected_all = random.sample(selected_all_abs_path, digital_num)


    bg_img = cv2.imread(bg_img_path)
    bgsz = bg_img.shape[:2]

    label_str = ""
    for i in range(digital_num):
        fgi_abs_path = random.sample(selected_all, 1)[0]

        # if "0.N" in fgi_abs_path or "1.N" in fgi_abs_path or "2.N" in fgi_abs_path or "3.N" in fgi_abs_path or "4.N" in fgi_abs_path or "5.N" in fgi_abs_path or "6.N" in fgi_abs_path or "7.N" in fgi_abs_path or "8.N" in fgi_abs_path or "9.N" in fgi_abs_path:
        #     selected_all.remove(fgi_abs_path)
        # if "=0" in fgi_abs_path or "=1" in fgi_abs_path or "=2" in fgi_abs_path or "=3" in fgi_abs_path or "=4" in fgi_abs_path or "=5" in fgi_abs_path or "=6" in fgi_abs_path or "=7" in fgi_abs_path or "=8" in fgi_abs_path or "=9" in fgi_abs_path:
        #     selected_all.remove(fgi_abs_path)

        fgi_base_name = os.path.basename(fgi_abs_path)
        fgi_name = os.path.splitext(fgi_base_name)[0]
        if "AN" in fgi_name:
            label_str += "A"
        elif "bN" in fgi_name:
            label_str += "b"
        elif "CN" in fgi_name:
            label_str += "C"
        # elif "0" in fgi_name:
        #     label_str += "0"
        # elif "1" in fgi_name:
        #     label_str += "1"
        # elif "2" in fgi_name:
        #     label_str += "2"
        # elif "3" in fgi_name:
        #     label_str += "3"
        # elif "4" in fgi_name:
        #     label_str += "4"
        # elif "5" in fgi_name:
        #     label_str += "5"
        # elif "6" in fgi_name:
        #     label_str += "6"
        # elif "7" in fgi_name:
        #     label_str += "7"
        # elif "8" in fgi_name:
        #     label_str += "8"
        # elif "9" in fgi_name:
        #     label_str += "9"
        elif "=0" in fgi_name:
            label_str += "0"
        elif "=1" in fgi_name:
            label_str += "1"
        elif "=2" in fgi_name:
            label_str += "2"
        elif "=3" in fgi_name:
            label_str += "3"
        elif "=4" in fgi_name:
            label_str += "4"
        elif "=5" in fgi_name:
            label_str += "5"
        elif "=6" in fgi_name:
            label_str += "6"
        elif "=7" in fgi_name:
            label_str += "7"
        elif "=8" in fgi_name:
            label_str += "8"
        elif "=9" in fgi_name:
            label_str += "9"
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
        # label_str += fgi_name

        # fg_abs_path = fg_path + "/{}".format(fgi)
        # fg_img = cv2.imread(fg_abs_path)
        # fg_img_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
        # ret, thr = cv2.threshold(fg_img_gray, 80, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("{}/gray_20230912_{}.jpg".format(save_path, fgi), thr)

        fg_img = cv2.imread(fgi_abs_path)
        fgsz = fg_img.shape[:2]
        r = dstsz[1] / fgsz[0]
        # fg_img = cv2.resize(fg_img, (int(round(fgsz[1] * r)), 45))
        # dstsz = (28, 45)
        fg_img = cv2.resize(fg_img, dstsz)
        fgsz = fg_img.shape[:2]
        fg_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
        # ret, thr = cv2.threshold(fg_gray, 180, 255, cv2.THRESH_BINARY_INV)
        ret, thr = cv2.threshold(fg_gray, 128, 255, cv2.THRESH_BINARY)

        thr_merge = cv2.merge([thr, thr, thr])

        # mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
        # out = cv2.seamlessClone(fg_img, bg_img, mask_img, (30 + fgsz[1] * i, 45 // 2), cv2.NORMAL_CLONE)  #

        for ii in range(dstsz[0]):
            for jj in range(dstsz[1]):
                if thr[jj, ii] == 255:
                    try:
                        # fg_img[jj, ii] = fg_img[jj, ii] + number_color
                        fg_img[jj, ii] = number_color
                        fg_img[jj, ii] = np.clip(fg_img[jj, ii], 0, 255)
                    except Exception as Error:
                        print(Error)

        upper_loew_space = (bgsz[0] - dstsz[1]) // 2
        # bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])] = bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])] + fg_img
        bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])] = fg_img

        # for ii in range(dstsz[0]):
        #     for jj in range(dstsz[1]):
        #         if thr[jj, ii] == 255:
        #         # if fg_img[jj, ii] == (255, 255, 255):
        #         #     bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])][jj, ii] = 0
        #             bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])][jj, ii] += 0 + np.random.randint(127, 255)

    aug_rdm = np.random.random()
    if aug_rdm > 0.5:
        noise_aug = NoiseAug(ratio=0.9)
        blur_rdm = np.random.random()
        if blur_rdm < 0.5:
            blur_aug = BlurAug(type="EASY", ratio=0.9)
        else:
            blur_aug = BlurAug(type="HARD", ratio=0.9)
        hsv_aug = HSVAug(hgain=0.2, sgain=0.7, vgain=0.5, ratio=0.9)
        bg_img = noise_aug(bg_img)
        bg_img = blur_aug(bg_img)
        bg_img = hsv_aug(bg_img)
        bg_img_aug = doing_aug(bg_img)

        # cv2.imwrite("{}/{}_aug_{}_{}={}.jpg".format(aug_save_path, datetime, str(fname_rdm).replace(".", ""), fgi_name, label_str), bg_img_aug)
        cv2.imwrite("{}/{}_aug_{}_{}={}.jpg".format(aug_save_path, datetime, str(fname_rdm).replace(".", ""), fgi_name.replace("=", ""), label_str), bg_img_aug)
    else:
        # cv2.imwrite("{}/{}_{}_{}={}.jpg".format(normal_save_path, datetime, str(fname_rdm).replace(".", ""), fgi_name, label_str), bg_img)
        cv2.imwrite("{}/{}_{}_{}={}.jpg".format(normal_save_path, datetime, str(fname_rdm).replace(".", ""), fgi_name.replace("=", ""), label_str), bg_img)
def main_test39__(bg_img_path, fg_path, save_path, dstsz=(28, 45), aug_flag=True, m_ratio=0.025, max_num=9):
    """
    :param bg_img_path:
    :param fg_path:
    :param save_path:
    :param dstsz: [w, h]
    :param aug_flag:
    :param m_ratio:
    :param max_num:
    :return:
    """
    normal_save_path = "{}/normal".format(save_path)
    aug_save_path = "{}/aug".format(save_path)
    os.makedirs(normal_save_path, exist_ok=True)
    os.makedirs(aug_save_path, exist_ok=True)

    datetime = time.strftime("%Y%m%d", time.localtime(time.time()))

    color_db = [
        np.array([0, 0, 250], dtype=np.float32),  # red
        np.array([0, 250, 0], dtype=np.float32),  # green
        np.array([250, 250, 250], dtype=np.float32),  # white
        np.array([10, 10, 10], dtype=np.float32)  # black
    ]
    number_color = random.choice(color_db)
    # ==============================================================================================================================================
    fname_rdm = np.random.random()
    digital_num = np.random.randint(1, max_num + 1)
    x_space = np.random.randint(1, dstsz[0] + (9 - digital_num) * dstsz[0])

    # # fg_num_path = fg_path + "/num"
    # fg_num_path = fg_path
    # # fg_numdot_path = fg_path + "/numdot"
    # fg_num_list = sorted(os.listdir(fg_num_path))
    # # fg_numdot_list = sorted(os.listdir(fg_numdot_path))
    #
    # # selected_numdot_list = ["{}/{}".format(fg_numdot_path, random.sample(fg_numdot_list, 1)[0])]
    # selected_num_list = []
    # for si in fg_num_list:
    #     selected_num_list.append("{}/{}".format(fg_num_path, si))
    # # selected_all_abs_path = selected_num_list + selected_numdot_list
    # # selected_all = random.sample(selected_all_abs_path, digital_num)
    #
    # selected_all = random.sample(selected_num_list, digital_num)

    # ===============================================
    # fg_num_path = fg_path + "/num"
    fg_num_path = fg_path
    fg_numdot_path = os.path.abspath(os.path.join(fg_path, "..")) + "/0-9_output_ud_stack"
    fg_num_list = sorted(os.listdir(fg_num_path))
    fg_numdot_list = sorted(os.listdir(fg_numdot_path))

    selected_numdot_list = ["{}/{}".format(fg_numdot_path, random.sample(fg_numdot_list, 1)[0])]
    selected_num_list = []
    for si in fg_num_list:
        selected_num_list.append("{}/{}".format(fg_num_path, si))
    selected_all_abs_path = selected_num_list + selected_numdot_list
    selected_all = random.sample(selected_all_abs_path, digital_num)


    bg_img = cv2.imread(bg_img_path)
    bgsz = bg_img.shape[:2]

    label_str = ""
    for i in range(digital_num):
        fgi_abs_path = random.sample(selected_all, 1)[0]

        # if "0.N" in fgi_abs_path or "1.N" in fgi_abs_path or "2.N" in fgi_abs_path or "3.N" in fgi_abs_path or "4.N" in fgi_abs_path or "5.N" in fgi_abs_path or "6.N" in fgi_abs_path or "7.N" in fgi_abs_path or "8.N" in fgi_abs_path or "9.N" in fgi_abs_path:
        #     selected_all.remove(fgi_abs_path)
        # if "=0" in fgi_abs_path or "=1" in fgi_abs_path or "=2" in fgi_abs_path or "=3" in fgi_abs_path or "=4" in fgi_abs_path or "=5" in fgi_abs_path or "=6" in fgi_abs_path or "=7" in fgi_abs_path or "=8" in fgi_abs_path or "=9" in fgi_abs_path:
        #     selected_all.remove(fgi_abs_path)

        fgi_base_name = os.path.basename(fgi_abs_path)
        fgi_name = os.path.splitext(fgi_base_name)[0]
        if "AN" in fgi_name:
            label_str += "A"
        elif "bN" in fgi_name:
            label_str += "b"
        elif "CN" in fgi_name:
            label_str += "C"
        # elif "0" in fgi_name:
        #     label_str += "0"
        # elif "1" in fgi_name:
        #     label_str += "1"
        # elif "2" in fgi_name:
        #     label_str += "2"
        # elif "3" in fgi_name:
        #     label_str += "3"
        # elif "4" in fgi_name:
        #     label_str += "4"
        # elif "5" in fgi_name:
        #     label_str += "5"
        # elif "6" in fgi_name:
        #     label_str += "6"
        # elif "7" in fgi_name:
        #     label_str += "7"
        # elif "8" in fgi_name:
        #     label_str += "8"
        # elif "9" in fgi_name:
        #     label_str += "9"
        elif "=0" in fgi_name:
            label_str += "0"
        elif "=1" in fgi_name:
            label_str += "1"
        elif "=2" in fgi_name:
            label_str += "2"
        elif "=3" in fgi_name:
            label_str += "3"
        elif "=4" in fgi_name:
            label_str += "4"
        elif "=5" in fgi_name:
            label_str += "5"
        elif "=6" in fgi_name:
            label_str += "6"
        elif "=7" in fgi_name:
            label_str += "7"
        elif "=8" in fgi_name:
            label_str += "8"
        elif "=9" in fgi_name:
            label_str += "9"
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
        # label_str += fgi_name

        # fg_abs_path = fg_path + "/{}".format(fgi)
        # fg_img = cv2.imread(fg_abs_path)
        # fg_img_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
        # ret, thr = cv2.threshold(fg_img_gray, 80, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("{}/gray_20230912_{}.jpg".format(save_path, fgi), thr)

        fg_img = cv2.imread(fgi_abs_path)
        fgsz = fg_img.shape[:2]
        r = dstsz[1] / fgsz[0]
        # fg_img = cv2.resize(fg_img, (int(round(fgsz[1] * r)), 45))
        # dstsz = (28, 45)
        fg_img = cv2.resize(fg_img, dstsz)
        fgsz = fg_img.shape[:2]
        fg_gray = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)
        # ret, thr = cv2.threshold(fg_gray, 180, 255, cv2.THRESH_BINARY_INV)
        ret, thr = cv2.threshold(fg_gray, 128, 255, cv2.THRESH_BINARY)

        thr_merge = cv2.merge([thr, thr, thr])

        # mask_img = 255 * np.ones(shape=fg_img.shape, dtype=np.uint8)
        # out = cv2.seamlessClone(fg_img, bg_img, mask_img, (30 + fgsz[1] * i, 45 // 2), cv2.NORMAL_CLONE)  #

        # for ii in range(dstsz[0]):
        #     for jj in range(dstsz[1]):
        #         if thr[jj, ii] == 255:
        #             try:
        #                 fg_img[jj, ii] = fg_img[jj, ii] + number_color
        #                 fg_img[jj, ii] = np.clip(fg_img[jj, ii], 0, 255)
        #             except Exception as Error:
        #                 print(Error)

        upper_loew_space = (bgsz[0] - dstsz[1]) // 2
        bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])] = bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])] + thr_merge

        for ii in range(dstsz[0]):
            for jj in range(dstsz[1]):
                if thr[jj, ii] == 255:
                # if fg_img[jj, ii] == (255, 255, 255):
                    bg_img[upper_loew_space:(upper_loew_space + dstsz[1]), (x_space + fgsz[1] * i):(x_space + fgsz[1] * i + dstsz[0])][jj, ii] = 0 + np.random.randint(127, 255)

    aug_rdm = np.random.random()
    if aug_rdm > 0.5:
        noise_aug = NoiseAug(ratio=0.9)
        blur_rdm = np.random.random()
        if blur_rdm < 0.5:
            blur_aug = BlurAug(type="EASY", ratio=0.9)
        else:
            blur_aug = BlurAug(type="HARD", ratio=0.9)
        hsv_aug = HSVAug(hgain=0.2, sgain=0.7, vgain=0.5, ratio=0.9)
        bg_img = noise_aug(bg_img)
        bg_img = blur_aug(bg_img)
        bg_img = hsv_aug(bg_img)
        bg_img_aug = doing_aug(bg_img)

        cv2.imwrite("{}/{}_aug_{}_{}={}.jpg".format(aug_save_path, datetime, str(fname_rdm).replace(".", ""), fgi_name, label_str), bg_img_aug)
    else:
        cv2.imwrite("{}/{}_{}_{}={}.jpg".format(normal_save_path, datetime, str(fname_rdm).replace(".", ""), fgi_name, label_str), bg_img)
def main_test40():
    bg = np.zeros(shape=(320, 960))
    cv2.imwrite("/home/zengyifan/wujiahu/data/010.Digital_Rec/others/dbnet_PS/New Folder/bg_20230918.jpg", bg)
def main_test41():
    data_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/Others/HK_Prison/cropped_20230921/smoke"
    file_list = get_file_list(data_path)
    h, s, v = [], [], []
    for f in tqdm(file_list):
        hsv_ = os.path.splitext(f)[0].split("=")[1]
        h_, s_, v_ = hsv_.split("_")[0], hsv_.split("_")[1], hsv_.split("_")[2]
        h.append(float(h_))
        s.append(float(s_))
        v.append(float(v_))

    print(np.mean(h), np.mean(s), np.mean(v))
    print(min(h), max(h))
    print(min(s), max(s))
    print(min(v), max(v))
def main_test42():
    data_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/gen_fake/output/font_char_img/bg2_OK"
    file_list = get_file_list(data_path)
    fonts = []
    for f in tqdm(file_list):
        if f.count("_") == 2:
            font_name = f.split("_")[0]
        else:
            font_name = f.replace("_{}_{}".format(f.split("_")[2], f.split("_")[3]), "")
        if font_name not in fonts:
            fonts.append(font_name)

    for ft in fonts:
        for f in tqdm(file_list):
            if f.count("_") == 2:
                font_name = f.split("_")[0]
            else:
                font_name = f.replace("_{}_{}".format(f.split("_")[2], f.split("_")[3]), "")
            if font_name == ft:
                save_path = make_save_path(data_path, "New")
                save_path = save_path + '/{}'.format(ft)
                os.makedirs(save_path, exist_ok=True)
                f_abs_path = data_path + "/{}".format(f)
                f_dst_path = save_path + "/{}".format(f)
                shutil.copy(f_abs_path, f_dst_path)
def main_test43():
    data_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/v3_4_cls_20230921/NewFolder2/20231031/jsons"
    file_list = get_file_list(data_path)
    save_path = make_save_path(data_path, "selected")

    file_list2 = file_list[:3036]
    for f in file_list2:
        f_abs_path = data_path + "/{}".format(f)
        f_dst_path = save_path + "/{}".format(f)
        shutil.copy(f_abs_path, f_dst_path)
def main_test44():
    data_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/v4_5_cls_20231122/NEW/New_2"
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels"
    save_path = make_save_path(data_path, "selected_wrong")
    img_save_path = save_path + "/images"
    lbl_save_path = save_path + "/labels"
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(lbl_save_path, exist_ok=True)

    txt_path = data_path + "/wrong.txt"
    fo = open(txt_path, "r", encoding="utf-8")
    txt_data = fo.readlines()
    fo.close()

    wrong_imgs = []
    for l in tqdm(txt_data):
        l_ = l.strip()
        img_name_ = os.path.splitext(os.path.basename(l_))[0]
        img_name = img_name_.replace(img_name_[-6:], "")
        if img_name[-1] == "_": img_name = img_name[:-1]
        if img_name not in wrong_imgs and img_name != "":
            wrong_imgs.append(img_name)

    for i in tqdm(wrong_imgs):
        img_abs_path = img_path + "/{}.jpg".format(i)
        txt_abs_path = lbl_path + "/{}.txt".format(i)

        if not os.path.exists(img_abs_path):
            print(img_abs_path)

        img_dst_path = img_save_path + "/{}.jpg".format(i)
        lbl_dst_path = lbl_save_path + "/{}.txt".format(i)
        shutil.copy(img_abs_path, img_dst_path)
        shutil.copy(txt_abs_path, lbl_dst_path)
def main_test45():
    img_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/test/crnn/others/20231109/lQLPJxxvcZKlfzTNAxDNBJewmiOkGNpISIIFPg0Ark4sAA_1175_784.png"
    img = cv2.imread(img_path)
    imgnew = img[275:310, 245:325]
    cv2.imwrite("/home/zengyifan/wujiahu/data/010.Digital_Rec/test/crnn/others/20231109/lQLPJxxvcZKlfzTNAxDNBJewmiOkGNpISIIFPg0Ark4sAA_1175_784_cropped.jpg", imgnew)
def main_test46():
    data_path1 = "/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/v4_5_cls_20231122/20231122_Pexels/mouth/mouth_cropped/2/3.0_random_selected_2500"
    data_path2 = "/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/v4_5_cls_20231122/20231122_Pexels/mouth/mouth_cropped/2/New Folder"

    file_list = get_file_list(data_path1)

    dirs = get_dir_list(data_path2)
    for d in tqdm(dirs):
        d_path = data_path2 + "/{}".format(d)
        save_path = make_save_path(data_path2, "/{}".format(d))
        for f in file_list:
            fname = os.path.splitext(f)[0].split("_")[0]
            for i in range(100):
                img_abs_path = d_path + "/{}_{}_{}.jpg".format(fname, i, d)
                if os.path.exists(img_abs_path):
                    img_dst_path = save_path + "/{}_{}_{}.jpg".format(fname, i, d)
                    shutil.move(img_abs_path, img_dst_path)
                    break
def main_test47():
    data_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/20231206/AUG_20231207/AUG_20231207_OK/labels_"
    save_path = make_save_path(data_path, "rename_002")
    file_list = get_file_list(data_path)
    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        fname = os.path.splitext(f)[0]
        f_dst_path = save_path + "/{}_{}.txt".format(fname, "paste_v5_003_aug_20231208_cup_002")
        os.rename(f_abs_path, f_dst_path)
def main_test48():
    data_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_4_cls_20231225/train-cigar_person_cup_mobile-phone"
    img_path = data_path + "/{}".format("images")
    txt_path = data_path + "/{}".format("labels_new_0_1_2_3")

    save_path = make_save_path(data_path, "cropped_for_cls_20240102")
    save_path_person_no_cigar = save_path + "/{}".format("no_cigar")
    save_path_person_with_cigar = save_path + "/{}".format("with_cigar")
    os.makedirs(save_path_person_no_cigar, exist_ok=True)
    os.makedirs(save_path_person_with_cigar, exist_ok=True)

    crop_ratio = [1.1, 1.2, 1.5, 1.8]
    img_list = os.listdir(img_path)

    for j in tqdm(img_list):
        try:
            img_name = os.path.splitext(j)[0]
            txt_abs_path = txt_path + "/{}.txt".format(img_name)
            img_abs_path = img_path + "/{}".format(j)
            cv2img = cv2.imread(img_abs_path)
            if cv2img is None: continue
            h, w = cv2img.shape[:2]
            imgsz = (h, w)

            txt_o = open(txt_abs_path, "r", encoding="utf-8")
            lines = txt_o.readlines()
            txt_o.close()

            # for i, l in enumerate(lines):
            #     l_s = l.strip().split(" ")
            #     cls = int(l_s[0])
            #     bbx_yolo = list(map(float, l_s[1:]))
            #     bbx_voc = convert_bbx_yolo_to_VOC([h, w], bbx_yolo)

            # ====================================================
            # ====================================================
            bbxes = []
            for l_ in lines:
                l = l_.strip().split(" ")
                cls = int(l[0])
                bbx_ = list(map(float, l[1:]))
                if cls == 0 or cls == 1:
                    bbx_voc = convert_bbx_yolo_to_VOC(imgsz, bbx_)
                    bbx_voc.append(cls)
                    bbxes.append(bbx_voc)

            if len(bbxes) == 1 and bbxes[0][-1] == 1:
                for nx in crop_ratio:
                    try:
                        cropped_img = crop_img_expand_n_times_v2(cv2img, bbxes[0][:4], [h, w], nx)
                        cropped_nx_path = save_path_person_no_cigar + "/{}/{}".format(bbxes[0][-1], nx)
                        os.makedirs(cropped_nx_path, exist_ok=True)
                        cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, str(np.random.randn()).replace(".", ""), nx), cropped_img)
                    except Exception as Error:
                        print(Error, Error.__traceback__.tb_lineno)
            elif len(bbxes) > 1:
                for i in range(len(bbxes)):
                    for j in range(i + 1, len(bbxes)):
                        iou_ij = cal_iou(bbxes[i][:4], bbxes[j][:4])
                        if iou_ij > 0 and bbxes[i][-1] != bbxes[j][-1]:
                            if bbxes[i][-1] == 1 and bbxes[j][-1] == 0:
                                for nx in crop_ratio:
                                    try:
                                        cropped_img = crop_img_expand_n_times_v2(cv2img, bbxes[i][:4], [h, w], nx)
                                        cropped_nx_path = save_path_person_with_cigar + "/{}/{}".format(bbxes[i][-1], nx)
                                        os.makedirs(cropped_nx_path, exist_ok=True)
                                        cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, str(np.random.randn()).replace(".", ""), nx), cropped_img)
                                    except Exception as Error:
                                        print(Error, Error.__traceback__.tb_lineno)
                            elif bbxes[i][-1] == 0 and bbxes[j][-1] == 1:
                                for nx in crop_ratio:
                                    try:
                                        cropped_img = crop_img_expand_n_times_v2(cv2img, bbxes[j][:4], [h, w], nx)
                                        cropped_nx_path = save_path_person_with_cigar + "/{}/{}".format(bbxes[j][-1], nx)
                                        os.makedirs(cropped_nx_path, exist_ok=True)
                                        cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, str(np.random.randn()).replace(".", ""), nx), cropped_img)
                                    except Exception as Error:
                                        print(Error, Error.__traceback__.tb_lineno)
                        elif iou_ij <= 0 and bbxes[i][-1] != bbxes[j][-1]:
                            if bbxes[i][-1] == 1 and bbxes[j][-1] == 0:
                                for nx in crop_ratio:
                                    try:
                                        cropped_img = crop_img_expand_n_times_v2(cv2img, bbxes[i][:4], [h, w], nx)
                                        cropped_nx_path = save_path_person_no_cigar + "/{}/{}".format(bbxes[i][-1], nx)
                                        os.makedirs(cropped_nx_path, exist_ok=True)
                                        cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, str(np.random.randn()).replace(".", ""), nx), cropped_img)
                                    except Exception as Error:
                                        print(Error, Error.__traceback__.tb_lineno)
                            elif bbxes[i][-1] == 0 and bbxes[j][-1] == 1:
                                for nx in crop_ratio:
                                    try:
                                        cropped_img = crop_img_expand_n_times_v2(cv2img, bbxes[j][:4], [h, w], nx)
                                        cropped_nx_path = save_path_person_no_cigar + "/{}/{}".format(bbxes[j][-1], nx)
                                        os.makedirs(cropped_nx_path, exist_ok=True)
                                        cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, str(np.random.randn()).replace(".", ""), nx), cropped_img)
                                    except Exception as Error:
                                        print(Error, Error.__traceback__.tb_lineno)
                        elif bbxes[i][-1] == 1 and bbxes[j][-1] == 1:
                            for nx in crop_ratio:
                                try:
                                    cropped_img = crop_img_expand_n_times_v2(cv2img, bbxes[i][:4], [h, w], nx)
                                    cropped_nx_path = save_path_person_no_cigar + "/{}/{}".format(bbxes[i][-1], nx)
                                    os.makedirs(cropped_nx_path, exist_ok=True)
                                    cv2.imwrite("{}/{}_{}_{}.jpg".format(cropped_nx_path, img_name, str(np.random.randn()).replace(".", ""), nx), cropped_img)
                                except Exception as Error:
                                    print(Error, Error.__traceback__.tb_lineno)
                        else:
                            print(bbxes[i], bbxes[j])

        except Exception as Error:
            print(Error, Error.__traceback__.tb_lineno)
def main_test49():
    data_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_4_cls_20231225/yolov5_with_cls/2"
    cls = os.path.basename(data_path)

    file_list = get_file_list(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        fname_new = fname + "={}".format(cls)

        f_abs_path = data_path + "/{}".format(f)
        f_dst_path = data_path + "/{}.jpg".format(fname_new)

        os.rename(f_abs_path, f_dst_path)
def main_test50():
    data_path = "/home/zengyifan/wujiahu/data/000.Open_Dataset/coco/train2017/train2017_cropped"
    dir_list = get_dir_list(data_path)
    for d in dir_list:
        if d == "0" or d == "67": continue
        d_path = data_path + "/{}/1.1".format(d)
        try:
            random_select_files(d_path, select_num=60, move_or_copy="move")
        except Exception as Error:
            print(Error)
def main_test51():
    data_path = "/home/zengyifan/wujiahu/data/000.Open_Dataset/coco/train2017/train2017_cropped/0/1.1"
    save_path = make_save_path(data_path, dir_name_add_str="selected_20240109")

    file_list = get_file_list(data_path)
    for f in tqdm(file_list):
        try:
            f_bas_path = data_path + "/{}".format(f)
            cv2img = cv2.imread(f_bas_path)
            imgsz = cv2img.shape[:2]
            if imgsz[0] > 256 and imgsz[1] > 256:
                f_dst_path = save_path + "/{}".format(f)
                shutil.move(f_bas_path, f_dst_path)
        except Exception as Error:
            print(Error)
def main_test52():
    dirs = "0123"
    src_dirs = ["1.1", "1.2", "1.5", "1.8"]
    src_path1 = "/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_4_cls_20231225/train-cigar_person_cup_mobile-phone_cropped_for_cls_20240102/with_cigar/1"
    src_path2 = "/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_4_cls_20231225/train-cigar_person_cup_mobile-phone_cropped_for_cls_20240102/no_cigar/1"

    save_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_4_cls_20231225/train-cigar_person_cup_mobile-phone_cropped_for_cls_20240102/dataset_20240108/selected_moved"
    for d in dirs:
        data_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_4_cls_20231225/train-cigar_person_cup_mobile-phone_cropped_for_cls_20240102/dataset_20240108/{}".format(d)
        file_list = get_file_list(data_path)
        for f in tqdm(file_list):
            # fname = f[:-7]
            # f_abs_path = data_path + "/{}".format(f)

            fname = os.path.splitext(f)[0]
            fname_ = fname.split("_")[:-2]

            fname_new = ""
            for idx, fi in enumerate(fname_):
                fname_new += fi + "_"
            # fname_new += "0" + "_{}.jpg".format(sd)

            for sd in src_dirs[1:]:
                for nn in range(25):
                    f_src_path1 = src_path1 + "/{}/{}{}_{}.jpg".format(sd, fname_new, nn, sd)
                    f_src_path2 = src_path2 + "/{}/{}{}_{}.jpg".format(sd, fname_new, nn, sd)

                    os.makedirs(save_path + "/{}".format(d), exist_ok=True)
                    f_dst_path = save_path + "/{}/{}{}_{}.jpg".format(d, fname_new, nn, sd)

                    if os.path.exists(f_src_path1):
                        shutil.move(f_src_path1, f_dst_path)
                    elif os.path.exists(f_src_path2):
                        shutil.move(f_src_path2, f_dst_path)
def main_test53():
    src_dirs = ["1.1", "1.2", "1.5", "1.8"]

    for sd in src_dirs[1:]:
        data_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_4_cls_20231225/train-cigar_person_cup_mobile-phone_cropped_for_cls_20240102/with_cigar/1/{}".format(sd)
        file_list = get_file_list(data_path)

        imgi = 0
        for f in file_list:
            f_abs_path = data_path + "/{}".format(f)
            fname = os.path.splitext(f)[0]
            fname_ = fname.split("_")[:-2]

            fname_new = ""
            for idx, fi in enumerate(fname_):
                fname_new += fi + "_"
            fname_new += str(imgi) + "_{}.jpg".format(sd)
            f_dst_path = data_path + "/{}".format(fname_new)

            if os.path.exists(f_dst_path):
                imgi += 1
                fname_new = ""
                for idx, fi in enumerate(fname_):
                    fname_new += fi + "_"
                fname_new += str(imgi) + "_{}.jpg".format(sd)
                f_dst_path = data_path + "/{}".format(fname_new)

                os.rename(f_abs_path, f_dst_path)

            else:
                os.rename(f_abs_path, f_dst_path)

            imgi = 0
def easyocr_test():
    import easyocr
    # TODO
def main_test54():
    data_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/others/handpose/1.5_2_res"
    save_path = make_save_path(data_path, "pose_img")
    src_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_4_cls_20231225/train-cigar_hand_face_cup_mobile-phone/train-cigar_hand_face_cup_mobile-phone_cropped/1/1.5_output_black"
    dirs = [0, 1, 2]
    for d in dirs:
        d_path = data_path + "/{}".format(d)
        save_path_d = save_path + "/{}".format(d)
        os.makedirs(save_path_d, exist_ok=True)
        file_list = get_file_list(d_path)
        for f in file_list:
            # f_abs_path = d_path + "/{}".format(f)
            f_src_path = src_path + "/{}".format(f)
            f_dst_path = save_path_d + "/{}".format(f)
            shutil.copy(f_src_path, f_dst_path)
def main_test55():
    file_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/others/handpose/res12.csv"
    res0, res1, res2 = [], [], []
    res0_, res1_, res2_ = [], [], []
    data = read_csv(file_path)
    data_ = data[:, -9:]
    # print(data_[0])

    for i in range(len(data_)):
        datai, datai_label = data_[i][:-1], data_[i][-1]
        datai = list(datai)
        if datai not in res0 and datai_label == 0:
            res0.append(datai)
        elif datai not in res1 and datai_label == 1:
            res1.append(datai)
        elif datai not in res2 and datai_label == 2:
            res2.append(datai)

        if datai_label == 0:
            res0_.append(datai)
        elif datai_label == 1:
            res1_.append(datai)
        elif datai_label == 2:
            res2_.append(datai)

    # print(res0, len(res0))
    # print(res1, len(res1))
    # print(res2, len(res2))

    final_res0, final_res1, final_res2 = {}, {}, {}
    max_common0, max_common1, max_common2 = {}, {}, {}
    max_common0_num, max_common1_num, max_common2_num = [], [], []

    for i in range(len(res0)):
        # print("res0: {}, {}".format(res0[i], res0_.count(res0[i])))
        final_res0["{}".format(res0[i])] = res0_.count(res0[i])
        # max_common0["{}".format(res0[i])] = sorted(res0[i][-15:], key=lambda item: (res0[i][-15:].count(item), item), reverse=True)[0]
        # max_common0_num.append(sorted(res0[i][-15:], key=lambda item: (res0[i][-15:].count(item), item), reverse=True)[0])

    for i in range(len(res1)):
        # print("res1: {}, {}".format(res1[i], res1_.count(res1[i])))
        final_res1["{}".format(res1[i])] = res1_.count(res1[i])
        # max_common1["{}".format(res1[i])] = sorted(res1[i][-15:], key=lambda item: (res1[i][-15:].count(item), item), reverse=True)[0]
        # max_common1_num.append(sorted(res1[i][-15:], key=lambda item: (res1[i][-15:].count(item), item), reverse=True)[0])
    for i in range(len(res2)):
        # print("res2: {}, {}".format(res2[i], res2_.count(res2[i])))
        final_res2["{}".format(res2[i])] = res2_.count(res2[i])
        # max_common2["{}".format(res2[i])] = sorted(res2[i][-15:], key=lambda item: (res2[i][-15:].count(item), item), reverse=True)[0]
        # max_common2_num.append(sorted(res2[i][-15:], key=lambda item: (res2[i][-15:].count(item), item), reverse=True)[0])

    # print("final_res0: ", sorted(final_res0.items(), key=lambda x: x[1], reverse=True)[:100])
    # print("final_res1: ", sorted(final_res1.items(), key=lambda x: x[1], reverse=True)[:100])
    # print("final_res2: ", sorted(final_res2.items(), key=lambda x: x[1], reverse=True)[:100])

    print("final_res0: ", sorted(final_res0.items(), key=lambda x: x[1], reverse=True)[:100])
    print("final_res1: ", sorted(final_res1.items(), key=lambda x: x[1], reverse=True)[:100])
    print("final_res2: ", sorted(final_res2.items(), key=lambda x: x[1], reverse=True)[:100])

    # # print("max_common0: ", max_common0)
    # # print("max_common1: ", max_common1)
    # # print("max_common2: ", max_common2)
    #
    # print("max_common0_num: ", max_common0_num)
    # print("max_common2_num: ", max_common1_num)
    # print("max_common2_num: ", max_common2_num)
    #
    # max_common0_num_max_i = sorted(max_common0_num, key=lambda item: (max_common0_num.count(item), item), reverse=True)
    # max_common1_num_max_i = sorted(max_common1_num, key=lambda item: (max_common1_num.count(item), item), reverse=True)
    # max_common2_num_max_i = sorted(max_common2_num, key=lambda item: (max_common2_num.count(item), item), reverse=True)
    #
    # print("max_common0_num_max_i: ", max_common0_num_max_i)
    # print("max_common1_num_max_i: ", max_common1_num_max_i)
    # print("max_common2_num_max_i: ", max_common2_num_max_i)
def main_test56():
    img_path = "/home/zengyifan/wujiahu/data/002.Exit_Light_Det/test/test_robot/robot_test_20230224/002_robot_test_20230224_0000.jpg"
    cv2img = cv2.imread(img_path)
    imgsz = cv2img.shape[:2]
    cropped = cv2img[:int(imgsz[0] * 0.65), :]
    cv2.imwrite("/home/zengyifan/wujiahu/data/002.Exit_Light_Det/test/test_robot/robot_test_20230224/002_robot_test_20230224_0000_cropped.jpg", cropped)
def main_test57():
    data_path1 = "/home/zengyifan/wujiahu/data/010.Digital_Rec/test/dbnet/v1_add_20240203/output/crnn_lite_lstm_output/digits"
    data_path2 = "/home/zengyifan/wujiahu/data/010.Digital_Rec/test/dbnet/v1_add_20240203/output/20240202_merged_output_output_v5_/orig_img_no_label"

    file_list1_ = sorted(os.listdir(data_path1))
    file_list2_ = sorted(os.listdir(data_path2))

    file_list1 = []
    file_list2 = []
    for f in file_list1_:
        sp = f.split("=")[:-1]
        fname = "=".join([ii for ii in sp])
        file_list1.append(fname)
    for f in file_list2_:
        sp = f.split("=")[:-1]
        fname = "=".join([ii for ii in sp])
        file_list2.append(fname)

    unexpected = list(set(file_list1) ^ set(file_list2))
    print(len(unexpected))
    # for u in unexpected:
    #     print(u)
def main_test58():
    data_path1 = "/home/zengyifan/wujiahu/data/010.Digital_Rec/test/dbnet/v1_add_20240203/output/20240202_merged_output_output_v5_/crnn_lite_lstm_output/Chinese"
    data_path2 = "/home/zengyifan/wujiahu/data/010.Digital_Rec/test/dbnet/v1_add_20240203/output/20240202_merged_output_output_v5_/orig_img_no_label"

    file_list1_ = sorted(os.listdir(data_path1))
    file_list2_ = sorted(os.listdir(data_path2))

    file_list1 = []
    file_list2 = []
    for f in file_list1_:
        sp = f.split("=")[:-2]
        fname = "=".join([ii for ii in sp])
        pred_label = os.path.splitext(f.split("=")[-1])[0]
        if len(pred_label) > 3 and isAllChinese(pred_label):
            file_list1.append(fname)
    for f in file_list2_:
        sp = f.split("=")[:-1]
        fname = "=".join([ii for ii in sp])
        file_list2.append(fname)

    unexpected = list(set(file_list1) ^ set(file_list2))
    print(len(unexpected))

    save_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/test/dbnet/v1_add_20240203/output/20240202_merged_output_output_v5_/orig_img_no_label_Selected_20240205"
    os.makedirs(save_path, exist_ok=True)
    for u in unexpected:
        for imgi in file_list2_:
            if u in imgi:
                try:
                    imgi_abs_path = data_path2 + "/{}".format(imgi)
                    imgi_dst_path = save_path + "/{}".format(imgi)
                    shutil.move(imgi_abs_path, imgi_dst_path)
                except Exception as Error:
                    print(Error)
def main_test59(data_path, copy_image=True):
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

            len_result = len(result_)

            if len_result == 4:
                if copy_image:
                    img_name = os.path.splitext(j)[0]
                    # img_abs_path = img_path + "/{}".format(j.replace(".json", ""))
                    # # shutil.move(img_path, det_images_path + "/{}".format(j.replace(".json", "")))
                    # shutil.copy(img_abs_path, kpt_images_path + "/{}".format(j.replace(".json", "")))
                    img_abs_path = img_path + "/{}".format(img_name)

                    if not os.path.exists(img_abs_path):
                        img_abs_path = img_path + "/{}".format(img_name.replace(".jpg", ".jpeg"))

                    # shutil.move(img_path, det_images_path + "/{}".format(j.replace(".json", "")))
                    shutil.copy(img_abs_path, kpt_images_path + "/{}".format(img_name))

                txt_save_path = kpt_labels_path + "/{}.gt".format(j.replace(".json", "").split(".")[0])
                with open(txt_save_path, "w", encoding="utf-8") as fw:
                    kpts = []
                    attributes = []
                    for i in range(len_result):
                        x_ = result_[i]["x"]
                        y_ = result_[i]["y"]
                        attribute_ = result_[i]["attribute"]
                        # x_normalized = x_ / w
                        # y_normalized = y_ / h

                        # visible = True
                        # if visible:
                        #     kpts.append([x_normalized, y_normalized, 2])
                        kpts.append([x_, y_])
                        attributes.append(attribute_)

                    assert attributes == ["1", "2", "3", "4"] or attributes == ["tl", "tr", "br", "bl"], "attributes error"

                    kpts = np.asarray(kpts).reshape(-1, 8)
                    for ki in range(kpts.shape[0]):
                        txt_content = ", ".join([str(k) for k in kpts[ki]]) + ", 0\n"
                        fw.write(txt_content)

        except Exception as Error:
            print(Error)
def main_test60(base_path, dir1_name="images", dir2_name="labels", labelbee_json_label=False, move_or_delete="delete", dir="dir2"):
    """
    :param dir1:
    :param dir2:
    :param move_or_delete: "move" or "delete"
    :param dir: files in which dir will be move or delete
    :return:
    """
    dir1 = base_path + "/{}".format(dir1_name)
    dir2 = base_path + "/{}".format(dir2_name)
    file1_list = [os.path.splitext(i)[0] for i in os.listdir(dir1)]
    dir1_file_ends = os.path.splitext(os.listdir(dir1)[0])[1]
    dir2_file_ends = os.path.splitext(os.listdir(dir2)[0])[1]

    if labelbee_json_label:
        file2_list = [os.path.splitext(os.path.splitext(i)[0])[0] for i in os.listdir(dir2)]
    else:
        file2_list = [os.path.splitext(i)[0] for i in os.listdir(dir2)]

    unexpected = list(set(file1_list) & set(file2_list))

    if move_or_delete == "move":
        unexpected_path = os.path.abspath(os.path.join(dir1, "../..")) + "/unexpected"
        os.makedirs(unexpected_path, exist_ok=True)

    if move_or_delete == "delete":
        if dir == "dir1":
            for j in tqdm(unexpected):
                if labelbee_json_label:
                    file_abs_path = dir1 + "/{}.jpeg{}".format(j, dir1_file_ends)
                    try:
                        os.remove(file_abs_path)
                        print("os.remove: --> {}".format(file_abs_path))
                    except Exception as Error:
                        print(Error)
                else:
                    file_abs_path = dir1 + "/{}{}".format(j, dir1_file_ends)
                    try:
                        os.remove(file_abs_path)
                        print("os.remove: --> {}".format(file_abs_path))
                    except Exception as Error:
                        print(Error)
        else:
            for j in tqdm(unexpected):
                if labelbee_json_label:
                    file_abs_path = dir2 + "/{}.jpeg{}".format(j, dir2_file_ends)
                    try:
                        os.remove(file_abs_path)
                        print("os.remove: --> {}".format(file_abs_path))
                    except Exception as Error:
                        print(Error)
                else:
                    file_abs_path = dir2 + "/{}{}".format(j, dir2_file_ends)
                    try:
                        os.remove(file_abs_path)
                        print("os.remove: --> {}".format(file_abs_path))
                    except Exception as Error:
                        print(Error)
    # move
    else:
        if dir == "dir1":
            for j in tqdm(unexpected):
                if labelbee_json_label:
                    file_abs_path = dir1 + "/{}{}".format(j, dir1_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir1_file_ends)
                    try:
                        shutil.move(file_abs_path, file_dst_path)
                        print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))
                    except Exception as Error:
                        print(Error)
                else:
                    file_abs_path = dir1 + "/{}{}".format(j, dir1_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir1_file_ends)
                    try:
                        shutil.move(file_abs_path, file_dst_path)
                        print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))
                    except Exception as Error:
                        print(Error)

                # file_abs_path = dir1 + "/{}.PNG".format(j, dir1_file_ends)
                # file_dst_path = unexpected_path + "/{}.PNG".format(j, dir1_file_ends)
                # try:
                #     shutil.move(file_abs_path, file_dst_path)
                #     print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))
                # except Exception as Error:
                #     print(Error)
        else:
            for j in tqdm(unexpected):
                if labelbee_json_label:
                    file_abs_path = dir2 + "/{}.jpeg{}".format(j, dir2_file_ends)
                    file_dst_path = unexpected_path + "/{}.jpeg{}".format(j, dir2_file_ends)
                    try:
                        shutil.move(file_abs_path, file_dst_path)
                        print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))
                    except Exception as Error:
                        print(Error)
                else:
                    file_abs_path = dir2 + "/{}{}".format(j, dir2_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir2_file_ends)
                    try:
                        shutil.move(file_abs_path, file_dst_path)
                        print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))
                    except Exception as Error:
                        print(Error)
def main_test61():
    data_path = "/home/zengyifan/wujiahu/data/004.Knife_Det/train/v4_labels-hand_need_to_check/labels_merged_person0_knife43_hand80"
    save_path = make_save_path(data_path, "new_labels")
    img_path = os.path.abspath(os.path.join(data_path, "../..")) + "/images"

    file_list = get_file_list(data_path)
    for f in tqdm(file_list):
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        fo = open(f_abs_path, "r", encoding="utf-8")
        lines = fo.readlines()
        fo.close()

        f_dst_path = save_path + "/{}".format(f)
        fw = open(f_dst_path, "w", encoding="utf-8")

        try:
            img_abs_path1 = img_path + "/{}.jpg".format(fname)
            img_abs_path2 = img_path + "/{}.JPG".format(fname)
            img_abs_path3 = img_path + "/{}.png".format(fname)
            img_abs_path4 = img_path + "/{}.jpeg".format(fname)
            if not os.path.exists(img_abs_path1):
                img_abs_path = img_abs_path2
            else:
                img_abs_path = img_abs_path1

            cv2img = cv2.imread(img_abs_path)
            imgsz = cv2img.shape[:2]
        except Exception as Error:
            print(Error, fname, img_abs_path)

        len_ = len(lines)

        if len_ == 1:
            fw.write(lines[0])
            fw.close()
            continue

        for i in range(len_):
            li_split = lines[i].strip().split(" ")
            clsi = int(li_split[0])
            bbxi_yolo = list(map(float, [ii for ii in li_split[1:]]))
            bbxi_voc = convert_bbx_yolo_to_VOC(imgsz, bbxi_yolo)
            if clsi == 0:
                for j in range(i + 1, len_):
                    lj_split = lines[j].strip().split(" ")
                    clsj = int(lj_split[0])
                    bbxj_yolo = list(map(float, [jj for jj in lj_split[1:]]))
                    bbxj_voc = convert_bbx_yolo_to_VOC(imgsz, bbxj_yolo)
                    if clsj == 80:
                        iou = cal_iou(bbxi_voc, bbxj_voc)
                        if iou > 0.50:
                            fw.write(lines[j])
                        else:
                            fw.write(lines[i])
                            fw.write(lines[j])
                    else:
                        fw.write(lines[j])
            elif clsi == 80:
                for j in range(i + 1, len_):
                    lj_split = lines[j].strip().split(" ")
                    clsj = int(lj_split[0])
                    bbxj_yolo = list(map(float, [jj for jj in lj_split[1:]]))
                    bbxj_voc = convert_bbx_yolo_to_VOC(imgsz, bbxj_yolo)
                    if clsj == 0:
                        iou = cal_iou(bbxi_voc, bbxj_voc)
                        if iou > 0.50:
                            fw.write(lines[i])
                        else:
                            fw.write(lines[i])
                            fw.write(lines[j])
                    else:
                        fw.write(lines[j])
            else:
                fw.write(lines[i])

        fw.close()
def main_test_62():
    data_path = "/home/disk/disk7/data/000.XiaoFangXiang/train/v1/labels"
    file_list = sorted(os.listdir(data_path))

    save_path = "/home/disk/disk7/data/000.XiaoFangXiang/train/v1/labels_"
    os.makedirs(save_path, exist_ok=True)

    for f in file_list:
        f_abs_path = data_path + "/{}".format(f)
        data_ = open(f_abs_path, "r", encoding="utf-8")
        lines = data_.readlines()

        cls_ = []
        for line in lines:
            line = line.strip().split(" ")
            clsi = int(line[0])
            cls_.append(clsi)

        if (max(cls_)) > 4:
            print(f_abs_path)
            f_dst_path = save_path + "/{}".format(f)
            shutil.copy(f_abs_path, f_dst_path)
def main_test_10026_30():
    img_path = "/home/disk/disk7/010.Digital_Rec/rec/v4/letterbox_makeBorder_1691982864.5237536_0_4=R 42.2.jpg"
    dst_path = "/home/disk/disk7/010.Digital_Rec/rec/letterbox_makeBorder_1691982864.5237536_0_4=R 42.2.jpg"
    shutil.move(img_path, dst_path)
def main_test_10026_31():
    img_path = "/home/disk/disk7/010.Digital_Rec/rec/32_128/v1/20230912/crnn_labelbee_data_cropped"
    img_list = os.listdir(img_path)
    for img in tqdm(img_list):
        # img_name = os.path.splitext(img)[0]
        # print(img_name)
        img_abs_path = img_path + "/{}".format(img)
        # cv2_img = cv2.imread(img_abs_path)

        img_dst_path = "/home/disk/disk7/010.Digital_Rec/rec/32_128/v1/orig_merged" + "/{}".format(img)

        shutil.move(img_abs_path, img_dst_path)
def main_test_10026_32():
    res = []
    img_path = "/home/disk/disk7/010.Digital_Rec/rec/32_128/v1/orig_merged"
    img_list = os.listdir(img_path)
    for img in tqdm(img_list):
        img_name = os.path.splitext(img)[0]
        label = img_name.split("=")[1]
        for l in label:
            if l not in res:
                res.append(l)

    print(res, len(res))
def main_test_10026_33():
    img_path = "/home/disk/disk7/010.Digital_Rec/test/crnn/test_v2"
    save_path = "/home/disk/disk7/010.Digital_Rec/test/crnn/test_v2_selected"
    os.makedirs(save_path, exist_ok=True)

    img_list = os.listdir(img_path)
    for img in tqdm(img_list):
        img_name = os.path.splitext(img)[0]
        label = img_name.split("=")[1]
        img_abs_path = img_path + "/{}".format(img)
        img_dst_path = save_path + "/{}".format(img)
        if label == "":
            shutil.move(img_abs_path, img_dst_path)
def main_test_10026_34():
    img_path = "/home/disk/disk7/010.Digital_Rec/rec/Syn90k_merged"
    save_path = "/home/disk/disk7/010.Digital_Rec/rec/Syn90k_merged_selected"
    os.makedirs(save_path, exist_ok=True)

    img_list = os.listdir(img_path)
    for img in tqdm(img_list):
        img_name = os.path.splitext(img)[0]
        label = img_name.split("=")[1]
        img_abs_path = img_path + "/{}".format(img)
        img_dst_path = save_path + "/{}".format(img)
        # if label == "":
        #     shutil.move(img_abs_path, img_dst_path)

        try:
            cv2img = cv2.imread(img_abs_path)
        # if cv2img is None:
        except Exception as Error:
            print(Error)
            shutil.move(img_abs_path, img_dst_path)
            print(img_dst_path)
def main_test_10026_35():
    img_path = "/home/disk/disk7/010.Digital_Rec/rec/64_256_merged"
    save_path = "/home/disk/disk7/010.Digital_Rec/rec/64_256_merged_selected"
    os.makedirs(save_path, exist_ok=True)

    img_list = os.listdir(img_path)
    for img in tqdm(img_list):
        img_name = os.path.splitext(img)[0]
        if "horizontal" in img:
            name_0, name_1 = img_name.split("=")[0].split("_")[0], img_name.split("=")[0].split("_")[1]
            label = img_name.split("=")[1]
            img_abs_path = img_path + "/{}".format(img)
            img_dst_path = save_path + "/{}".format(img)

            # if name_0 == "horizontal" and (int(name_1) >= 0 and int(name_1) <= 50000) and len(name_1) == 8:
            #     shutil.move(img_abs_path, img_dst_path)

            shutil.move(img_abs_path, img_dst_path)
def main_test_10026_36():
    from trdg.generators import (
        GeneratorFromDict,
        GeneratorFromRandom,
        GeneratorFromStrings,
        GeneratorFromWikipedia
    )

    generator = GeneratorFromStrings(["test1", "test2", "test3"], blur=2, random_blur=True)
    for img, lbl in generator:
        cv2.imwrite("/home/disk/disk7/010.Digital_Rec/rec/64_256/trdg_test/test={}.jpg".format(lbl))
def main_test_10026_37():
    img_path = "/home/disk/disk7/010.Digital_Rec/rec/64_256_v4/train_random_selected_720000"
    save_path = "/home/disk/disk7/010.Digital_Rec/rec/64_256_v4/train_random_selected_720000_selected"
    os.makedirs(save_path, exist_ok=True)

    img_list = os.listdir(img_path)
    for img in tqdm(img_list):
        # img_name = os.path.splitext(img)[0]
        # name_0, name_1 = img_name.split("=")[0].split("_")[0], img_name.split("=")[0].split("_")[1]
        # label = img_name.split("=")[1]
        img_abs_path = img_path + "/{}".format(img)
        img_dst_path = save_path + "/{}".format(img)

        # if name_0 == "horizontal" and (int(name_1) >= 0 and int(name_1) <= 50000) and len(name_1) == 8:
        #     shutil.move(img_abs_path, img_dst_path)

        cv2img = cv2.imread(img_abs_path)
        if cv2img is None:
            shutil.move(img_abs_path, img_dst_path)
def main_test_10026_38():
    img_path = "/home/disk/disk7/010.Digital_Rec/rec/64_256_v4/train"

    img_list = os.listdir(img_path)
    for img in tqdm(img_list):
        # img_name = os.path.splitext(img)[0]
        # name_0, name_1 = img_name.split("=")[0].split("_")[0], img_name.split("=")[0].split("_")[1]
        # label = img_name.split("=")[1]
        if "m3.jpg" in img:
            img_abs_path = img_path + "/{}".format(img)
            img_dst_path = img_path + "/{}".format(img.replace("m3.jpg", ".jpg"))
            os.rename(img_abs_path, img_dst_path)
def main_test_10026_39():
    file_path = "/home/disk/disk7/010.Digital_Rec/Others/20231008/data.txt"
    with open(file_path, "r", encoding="utf-8") as fr:
        lines = fr.readlines()

    file_list = get_file_list("/home/disk/disk7/010.Digital_Rec/crnn/test/Syn90k_test")
    save_path = make_save_path(data_path="/home/disk/disk7/010.Digital_Rec/crnn/test/Syn90k_test", dir_name_add_str="moved")

    for f in tqdm(file_list):

        for line in lines:
            line = os.path.basename(line.strip())
            label = line.split("_")[1]
            if label in f:
                f_abs_path = "/home/disk/disk7/010.Digital_Rec/crnn/test/Syn90k_test" + "/{}".format(f)
                cv2img = cv2.imread(f_abs_path)
                if cv2img is None:
                    f_dst_path = save_path + "/{}".format(f)
                    shutil.move(f_abs_path, f_dst_path)
def main_test_10026_40():
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/from_lzx/fg/0-9._new/New Folder"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/from_lzx/fg/0-9._new/new2/led16sgmnt2-Regular"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/llj/fg/normal/0-9"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/qianyinbainyaqi/fg/num"
    data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/llj/bg/new"
    file_list = get_file_list(data_path)
    save_path = make_save_path(data_path, "new")

    global x10

    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        cv2img = cv2.imread(f_abs_path)
        h, w = cv2img.shape[:2]
        # gray = 255 - cv2img[:, :, 0]
        # cnts, hierarchy = cv2.findContours(gray.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnt_max = max(cnts, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(cnt_max)
        # x_min = x
        # x_max = x + w
        # y_min = y
        # y_max = y + h
        #
        # if "0" in f:
        #     x10 = x_min
        #
        # if "1" in f:
        #     imgnew = cv2img[y_min - 2:y_max + 1, x10:x_max + 2]
        # else:
        #     imgnew = cv2img[y_min - 2:y_max + 1, x_min - 2:x_max + 2]

        # imgnew = cv2img[11:h - 11, 4:w - 4]
        # if "=4" in f or "=5" in f or "=7" in f:
        #     imgnew = cv2img[7:h - 10, 7:w - 2]
        # else:
        #     imgnew = cv2img[7:h - 10, 6:w - 2]
        # imgnew = cv2img[7:h - 10, 6:w - 2]

        # if "1" in f:
        #     imgnew = cv2img[10:h - 12, :]
        # elif "2" in f:
        #     imgnew = cv2img[11:h - 12, :]
        # elif "3" in f:
        #     imgnew = cv2img[11:h - 12, :]
        # elif "4" in f:
        #     imgnew = cv2img[16:h - 10, :]
        # elif "5" in f:
        #     imgnew = cv2img[13:h - 11, :]
        # elif "6" in f:
        #     imgnew = cv2img[12:h - 13, :]
        # elif "8" in f:
        #     imgnew = cv2img[9:h - 13, :]
        #
        # else:
        #     imgnew = cv2img[13:h - 8, :]

        # if "1" in f:
        #     imgnew = cv2img[10:h - 12, :]
        # if "2" in f:
        #     imgnew = cv2img[4:h - 4, 3:w - 3]
        # # elif "3" in f:
        # #     imgnew = cv2img[3:h - 3, 2:w - 2]
        # # elif "4" in f:
        # #     imgnew = cv2img[3:h - 3, 2:w - 2]
        # # elif "5" in f:
        # #     imgnew = cv2img[3:h - 3, 2:w - 2]
        # elif "6" in f:
        #     imgnew = cv2img[7:h - 6, 4:w - 4]
        # # elif "8" in f:
        # #     imgnew = cv2img[3:h - 3, 2:w - 2]
        # else:
        #     imgnew = cv2img[3:h - 3, 2:w - 2]

        imgnew = cv2img[h // 2 - 65:, :]
        cv2.imwrite("{}/{}".format(save_path, f), imgnew)
def main_test_10026_41():
    img_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/from_lzx/fg/0-9._new/new/LiquidCrystal_bg1_=..jpg"
    cv2img = cv2.imread(img_path)
    h, w = cv2img.shape[:2]
    # imgnew = cv2img[:, 19:w - 14]
    # imgnew = cv2img[2:, :]
    imgnew = cv2img[46:h - 3, 5:w - 3]
    gray = cv2.cvtColor(imgnew, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thresh_merge = cv2.merge([thresh, thresh, thresh])
    imgnew = cv2.resize(thresh_merge, (50, 50))
    cv2.imwrite("/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/from_lzx/fg/0-9._new/new/dot_s.png", imgnew)
def main_test_10026_42():
    data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/no_space_New/0-9_AbC_new_no_space"
    file_list = get_file_list(data_path)
    num_save_path = make_save_path(data_path, dir_name_add_str="num")
    num_dot_save_path = make_save_path(data_path, dir_name_add_str="num_dot")

    for f in file_list:
        f_abs_path = data_path + "/{}".format(f)
        if ".N" in f:
            f_dst_path = num_dot_save_path + "/{}".format(f)
            shutil.copy(f_abs_path, f_dst_path)
        else:
            f_dst_path = num_save_path + "/{}".format(f)
            shutil.copy(f_abs_path, f_dst_path)
def main_test_10026_43():
    # img_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/bg/010_bg_rename_20231019_0000013.png"
    img_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/bg/010_bg_rename_20231019_0000014.png"
    cv2img = cv2.imread(img_path)
    h, w = cv2img.shape[:2]

    # p1 = np.array([[4, 3], [198, 17], [1, 59], [197, 72]], dtype=np.float32)
    # p2 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)

    p1 = np.array([[17, 7], [211, 20], [14, 63], [210, 76]], dtype=np.float32)
    p2 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(p1, p2)
    warped = cv2.warpPerspective(cv2img, M, (w, h))
    cv2.imwrite("{}".format(img_path.replace(".png", "_warpPerspective.png")), warped)
def main_test_10026_44():
    data_path = "/home/disk/disk7/010.Digital_Rec/crnn/train/v5_20231020/AbC_seamlessClone/no_space/007"
    file_list = get_file_list(data_path)
    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        cv2img = cv2.imread(f_abs_path)
        imgsz = cv2img.shape[:2]
        if imgsz == (83, 218):
            os.remove(f_abs_path)
def main_test_10026_45():
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

    data_path = "/home/wujiahu/data/000.OCR/CRNN/data/train/train_v1"
    save_path = make_save_path(data_path, "selected")
    dir_list = get_sub_dir_list(data_path)
    for d in dir_list:
        file_list = get_file_list(d)
        for f in tqdm(file_list):
            f_abs_path = d + "/{}".format(f)
            fname = os.path.splitext(f)[0]
            deng_idx = fname.index("=")
            label = fname[deng_idx + 1:]

            if label == "" or label[0] == "." or label[-1] == "." or " " in label:
                f_dst_path = save_path + "/{}".format(f)
                shutil.move(f_abs_path, f_dst_path)
def main_test_10026_46():
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/DigifaceWide"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/DS-Digital"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/FX-LED"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/LCD"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/LCD2"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/LCD-BQ"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/led16sgmnt2-Italic"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/led16sgmnt2-Regular"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/led16sgmnt-Italic"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/led16sgmnt-Regular"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/led_8x6"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/ledbdrev"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/ledboard"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/led_board"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/led_board-7"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/led_counter-7"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/ledfont"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/ledfonth"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/LESLIE"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/Lets-go-Digital"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/LiquidCrystal"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/MTC-7-Segment"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/PUTHIAfont"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/YournameD7CentralNarrow"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/YournameD7GeneralHalf"
    # data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/YournameD7GeneralNarrow"
    data_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cp/YournameD7HomeHalf"
    save_path = make_save_path(data_path, "New")
    file_list = get_file_list(data_path)

    for f in file_list:
        f_abs_path = data_path + "/{}".format(f)
        cv2img = cv2.imread(f_abs_path)
        h, w = cv2img.shape[:2]
        # if "=1" in f:
        #     img_new = cv2img[12:h - 3, :w - 11]
        # elif "=3" in f:
        #     img_new = cv2img[12:h - 3, :w - 6]
        # elif "=7" in f:
        #     img_new = cv2img[12:h - 3, :w - 7]
        # else:
        #     img_new = cv2img[12:h - 3, 5:w - 3]

        # if "=1" in f:
        #     img_new = cv2img[12:h - 9, :w - 2]
        # elif "=7" in f:
        #     img_new = cv2img[12:h - 9, 7:w - 2]
        # else:
        #     img_new = cv2img[12:h - 9, 6:w - 2]

        # if "=1" in f:
        #     img_new = cv2img[12:h - 9, 6:w - 3]
        # # elif "=7" in f:
        # #     img_new = cv2img[12:h - 9, 7:w - 3]
        # else:
        #     img_new = cv2img[12:h - 9, 6:w - 3]

        # if "=1" in f:
        #     img_new = cv2img[7:h - 9, :w - 6]
        # elif "=3" in f or "=4" in f or "=9" in f or "=A" in f:
        #     img_new = cv2img[7:h - 9, 4:w - 3]
        # else:
        #     img_new = cv2img[7:h - 9, 3:w - 4]

        # if "=1" in f:
        #     img_new = cv2img[7:h - 9, :w - 5]
        # elif "=3" in f or "=4" in f or "=9" in f or "=A" in f:
        #     img_new = cv2img[7:h - 9, 4:w - 2]
        # else:
        #     img_new = cv2img[7:h - 9, 3:w - 3]

        # if "=1" in f:
        #     img_new = cv2img[10:h - 7, :w - 2]
        # elif "=3" in f:
        #     img_new = cv2img[10:h - 7, 2:w - 4]
        # elif "=4" in f or "=5" in f or "=9" in f:
        #     img_new = cv2img[10:h - 7, 5:w - 2]
        # else:
        #     img_new = cv2img[10:h - 7, 2:w - 2]

        # if "=1" in f:
        #     img_new = cv2img[5:h - 7, 3:w - 1]
        # elif "=3" in f:
        #     img_new = cv2img[5:h - 7, 3:w - 1]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[5:h - 7, 4:w - 1]
        # else:
        #     img_new = cv2img[5:h - 7, 3:w - 1]

        # if "=1" in f:
        #     img_new = cv2img[5:h - 7, 3:w - 1]
        # elif "=3" in f:
        #     img_new = cv2img[5:h - 7, 3:w - 1]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[5:h - 7, 4:w - 1]
        # else:
        #     img_new = cv2img[5:h - 7, 3:w - 1]

        # if "=1" in f:
        #     img_new = cv2img[3:h - 7, 3:w - 1]
        # elif "=3" in f:
        #     img_new = cv2img[3:h - 7, 5:w - 1]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[3:h - 7, 5:w - 1]
        # else:
        #     img_new = cv2img[3:h - 7, 3:w - 1]

        # if "=1" in f:
        #     img_new = cv2img[3:h - 7, 3:w - 1]
        # elif "=3" in f:
        #     img_new = cv2img[3:h - 7, 4:w - 1]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[3:h - 7, 3:w - 1]
        # else:
        #     img_new = cv2img[3:h - 7, 3:w - 1]

        # if "=1" in f:
        #     img_new = cv2img[0:h - 4, 2:w - 5]
        # elif "=3" in f:
        #     img_new = cv2img[0:h - 4, 2:w - 5]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[0:h - 4, 2:w - 5]
        # else:
        #     img_new = cv2img[0:h - 4, 2:w - 5]

        # if "=1" in f:
        #     img_new = cv2img[0:h - 4, 2:w - 2]
        # elif "=3" in f:
        #     img_new = cv2img[0:h - 4, 2:w - 2]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[0:h - 4, 2:w - 2]
        # else:
        #     img_new = cv2img[0:h - 4, 2:w - 2]

        # if "=1" in f:
        #     img_new = cv2img[0:h - 4, 4:w - 2]
        # elif "=3" in f:
        #     img_new = cv2img[0:h - 4, 4:w - 2]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[0:h - 4, 4:w - 2]
        # else:
        #     img_new = cv2img[0:h - 4, 4:w - 2]

        # if "=1" in f:
        #     img_new = cv2img[11:h - 12, 5:w - 4]
        # elif "=2" in f:
        #     img_new = cv2img[13:h - 12, 5:w - 4]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[11:h - 12, 5:w - 4]
        # else:
        #     img_new = cv2img[11:h - 12, 5:w - 4]

        # if "=1" in f:
        #     img_new = cv2img[1:h - 9, 2:w - 6]
        # elif "=2" in f:
        #     img_new = cv2img[1:h - 9, 2:w - 6]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[1:h - 9, 2:w - 6]
        # else:
        #     img_new = cv2img[1:h - 9, 2:w - 6]

        # if "=1" in f:
        #     img_new = cv2img[6:h - 8, 2:w - 5]
        # elif "=2" in f:
        #     img_new = cv2img[6:h - 8, 2:w - 5]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[6:h - 8, 2:w - 5]
        # else:
        #     img_new = cv2img[6:h - 8, 2:w - 5]

        # if "=1" in f:
        #     img_new = cv2img[8:h - 7, 0:w - 2]
        # elif "=2" in f:
        #     img_new = cv2img[8:h - 7, 3:w - 2]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[8:h - 7, 3:w - 2]
        # else:
        #     img_new = cv2img[8:h - 7, 3:w - 2]

        # if "=1" in f:
        #     img_new = cv2img[8:h - 7, 0:w - 2]
        # elif "=2" in f:
        #     img_new = cv2img[8:h - 7, 3:w - 2]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[8:h - 7, 3:w - 2]
        # else:
        #     img_new = cv2img[8:h - 7, 3:w - 2]

        # img_new = cv2img[8:h - 9, 5:w - 3]

        # if "=1" in f:
        #     img_new = cv2img[11:h - 9, 0:w - 2]
        # elif "=3" in f:
        #     img_new = cv2img[11:h - 9, 6:w - 2]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[11:h - 9, 6:w - 2]
        # else:
        #     img_new = cv2img[11:h - 9, 5:w - 2]

        # if "=1" in f:
        #     img_new = cv2img[13:h - 2, 0:w - 2]
        # elif "=3" in f:
        #     img_new = cv2img[13:h - 2, 3:w - 2]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[13:h - 2, 3:w - 2]
        # else:
        #     img_new = cv2img[13:h - 2, 3:w - 2]

        # if "=1" in f:
        #     img_new = cv2img[15:h - 3, 6:w - 2]
        # elif "=3" in f:
        #     img_new = cv2img[15:h - 3, 6:w - 2]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[15:h - 3, 6:w - 2]
        # else:
        #     img_new = cv2img[15:h - 3, 6:w - 2]

        # if "=1" in f:
        #     img_new = cv2img[9:h - 8, 4:w - 2]
        # elif "=3" in f:
        #     img_new = cv2img[9:h - 8, 5:w - 2]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[9:h - 8, 5:w - 2]
        # else:
        #     img_new = cv2img[9:h - 8, 4:w - 2]

        # if "=1" in f:
        #     img_new = cv2img[8:h - 10, 2:w - 2]
        # elif "=3" in f:
        #     img_new = cv2img[8:h - 10, 2:w - 2]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[8:h - 10, 2:w - 2]
        # else:
        #     img_new = cv2img[8:h - 10, 2:w - 2]

        # if "=1" in f:
        #     img_new = cv2img[9:h - 8, 2:w - 2]
        # elif "=3" in f:
        #     img_new = cv2img[9:h - 8, 2:w - 2]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[9:h - 8, 2:w - 2]
        # else:
        #     img_new = cv2img[9:h - 8, 2:w - 2]

        # if "=1" in f:
        #     img_new = cv2img[10:h - 8, 3:w - 2]
        # elif "=3" in f:
        #     img_new = cv2img[10:h - 8, 3:w - 2]
        # elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
        #     img_new = cv2img[10:h - 8, 3:w - 2]
        # else:
        #     img_new = cv2img[10:h - 8, 3:w - 2]

        if "=1" in f:
            img_new = cv2img[10:h - 8, 3:w - 2]
        elif "=3" in f:
            img_new = cv2img[10:h - 8, 3:w - 2]
        elif "=4" in f or "=5" in f or "=7" in f or "=9" in f:
            img_new = cv2img[10:h - 8, 3:w - 2]
        else:
            img_new = cv2img[10:h - 8, 3:w - 2]

        f_dst_path = save_path + "/{}".format(f)
        cv2.imwrite("{}/{}".format(save_path, f), img_new)
def main_test_10026_47():
    base_path = "/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/AbC/fg/bg2_OK_New_cropped"
    save_path1 = make_save_path(base_path, "Blue")
    dir_list = sorted(os.listdir(base_path))
    for d in dir_list:
        d_path = base_path + "/{}".format(d)
        save_path2 = save_path1 + "/{}_{}".format(d, "Blue")
        os.makedirs(save_path2, exist_ok=True)
        file_list = get_file_list(d_path)
        for f in file_list:
            f_abs_path = d_path + "/{}".format(f)
            cv2img = cv2.imread(f_abs_path)
            cv2img_cp = cv2img.copy()
            gray = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            imgsz = cv2img.shape[:2]
            white = np.where(thresh[:, :] == 255)
            for ri, ci in zip(white[0], white[1]):
                cv2img_cp[ri, ci] = (255, 0, 0)
            cv2.imwrite("{}/{}".format(save_path2, f), cv2img_cp)
def main_test_10026_48():
    # data_path = "/home/disk/disk7/010.Digital_Rec/crnn/train/v5_20231020/fonts_seamlessClone/010_bg_rename_20231019_0000013_warpPerspective"
    # save_path = make_save_path(data_path, "moved")
    # dir_list = os.listdir(data_path)
    # for d in dir_list:
    #     d_path = data_path + "/{}".format(d)
    #     sbd_list = os.listdir(d_path)
    #     for sbd in sbd_list:
    #         sbd_path = d_path + "/{}".format(sbd)
    #         file_list = get_file_list(sbd_path)
    #         for f in file_list:
    #             f_abs_path = sbd_path + "/{}".format(f)
    #             f_dst_path = save_path + "/{}".format(f)
    #             label = os.path.splitext(f)[0].split("=")[1]
    #             if label == "" or label[0] == "." or label[-1] == ".":
    #                 shutil.move(f_abs_path, f_dst_path)

    # data_path = "/home/disk/disk7/010.Digital_Rec/crnn/train/v5_20231020/fonts_seamlessClone/010_bg_rename_20231019_0000000_Red"
    # save_path = make_save_path(data_path, "moved")
    # dir_list = os.listdir(data_path)
    # for d in dir_list:
    #     d_path = data_path + "/{}".format(d)
    #     # sbd_path = d_path + "/{}".format(sbd)
    #     file_list = get_file_list(d_path)
    #     for f in file_list:
    #         f_abs_path = d_path + "/{}".format(f)
    #         f_dst_path = save_path + "/{}".format(f)
    #         label = os.path.splitext(f)[0].split("=")[1]
    #         if label == "" or label[0] == "." or label[-1] == ".":
    #             shutil.move(f_abs_path, f_dst_path)

    # data_path = "/home/disk/disk7/010.Digital_Rec/crnn/train/v6_20231122_merged"
    # save_path = make_save_path(data_path, "moved")
    # # dir_list = os.listdir(data_path)
    # # for d in dir_list:
    # #     d_path = data_path + "/{}".format(d)
    # #     # sbd_path = d_path + "/{}".format(sbd)
    # file_list = get_file_list(data_path)
    # for f in file_list:
    #     f_abs_path = data_path + "/{}".format(f)
    #     f_dst_path = save_path + "/{}".format(f)
    #     try:
    #         label = os.path.splitext(f)[0].split("=")[1]
    #         if label == "" or label[0] == "." or label[-1] == ".":
    #             shutil.move(f_abs_path, f_dst_path)
    #     except Exception as Error:
    #         shutil.move(f_abs_path, f_dst_path)
    #         print(Error)

    data_path = "/home/disk/disk7/010.Digital_Rec/crnn/train/15_cls_mini_v2/v6/v6_20231122_mini"
    labels = '0123456789.' + 'AbC'

    dir_list = os.listdir(data_path)
    for d in dir_list:
        d_path = data_path + "/{}".format(d)
        # sbd_path = d_path + "/{}".format(sbd)
        save_path = make_save_path(d_path, "moved")
        file_list = get_file_list(d_path)
        for f in file_list:
            f_abs_path = d_path + "/{}".format(f)
            f_dst_path = save_path + "/{}".format(f)
            try:
                label = os.path.splitext(f)[0].split("=")[-1]
                # if label == "" or label[0] == "." or label[-1] == ".":
                #     shutil.move(f_abs_path, f_dst_path)

                for l in label:
                    if l not in labels:
                        shutil.move(f_abs_path, f_dst_path)
            except Exception as Error:
                shutil.move(f_abs_path, f_dst_path)
                print(Error)
def main_test_10026_49():
    data_path = "/home/disk/disk7/003.Cigar_Det/cls/train/train_base_v8.8/1"
    save_path = "/home/disk/disk7/003.Cigar_Det/cls/train/v8.11_add/1"
    file_list = get_file_list(data_path)
    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        if os.path.isdir(f_abs_path):
            f_dst_path = save_path + "/{}".format(f)
            shutil.move(f_abs_path, f_dst_path)
def main_test_10026_50(root):
    # imgs = []
    save_path = make_save_path(root, "Selected_20231222")
    dirs = sorted(os.listdir(root))
    for d in dirs:
        d_path = root + "/{}".format(d)
        sbdirs = sorted(os.listdir(d_path))
        for sbd in sbdirs:
            sbd_path = d_path + "/{}".format(sbd)
            ssbdirs = sorted(os.listdir(sbd_path))
            for ssbd in ssbdirs:
                ssbd_path = sbd_path + "/{}".format(ssbd)
                file_list = sorted(os.listdir(ssbd_path))
                # file_list_selected = random_select_files(ssbd_path, select_num=10, move_or_copy="copy", select_mode=0)
                selected = random.sample(file_list, 500)
                for f in selected:
                    f_abs_path = ssbd_path + "/{}".format(f)
                    # if "=" in f:
                    #     imgs.append(f_abs_path)
                    save_path_ = ssbd_path.replace(root, save_path)
                    os.makedirs(save_path_, exist_ok=True)
                    f_dst_path = save_path_ + "/{}".format(f)
                    shutil.copy(f_abs_path, f_dst_path)
def main_test_10026_51():
    data_path = "/home/wujiahu/data/000.OCR/CRNN/data/train/train_v1_add/004/out_512_color"
    file_list = get_file_list(data_path)
    for f_ in tqdm(file_list):
        f = f_.replace(" ", "")
        fname, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
        f_abs_path = data_path + "/{}".format(f_)
        fname_new = "20240117_512_color_{}={}{}".format(fname.split("_")[1], fname.split("_")[0], suffix)
        f_dst_path = data_path + "/{}".format(fname_new)
        os.rename(f_abs_path, f_dst_path)

        # trdg -l cn -c 25000 -w 10 -k 5 -rk  -d 1 -do 0 -b -r -f 128 -tc '#FFFF00,#D8D8BF'
def main_test_10026_52():
    dirname = "15_cls_20240117_add"
    root = "/home/disk/disk7/010.Digital_Rec/crnn/train/15_cls/{}".format(dirname)
    save_path = "/home/disk/disk7/010.Digital_Rec/crnn/train/15_cls_mini_v2/{}".format(dirname)
    dirs = sorted(os.listdir(root))
    for d in dirs:
        d_path = root + "/{}".format(d)
        sbdirs = sorted(os.listdir(d_path))
        for sbd in sbdirs:
            sbd_path = d_path + "/{}".format(sbd)
            # ssbdirs = sorted(os.listdir(sbd_path))
            # for ssbd in ssbdirs:
            #     ssbd_path = sbd_path + "/{}".format(ssbd)
            #     file_list = sorted(os.listdir(ssbd_path))

            file_list = sorted(os.listdir(sbd_path))
            selectNum = 0
            if len(file_list) <= 5000:
                selectNum = len(file_list) * 0.50
            elif 5000 < len(file_list) <= 10000:
                selectNum = len(file_list) * 0.30
            elif 10000 < len(file_list) <= 25000:
                selectNum = len(file_list) * 0.15
            elif 25000 < len(file_list) <= 50000:
                selectNum = len(file_list) * 0.075
            else:
                selectNum = len(file_list) * 0.025

            save_pathi = save_path + "/{}/{}".format(d, sbd)
            os.makedirs(save_pathi, exist_ok=True)

            selected = random.sample(file_list, int(selectNum))

            for s in tqdm(selected):
                f_src_path = sbd_path + "/{}".format(s)
                f_dst_path = save_pathi + "/{}".format(s)

                shutil.copy(f_src_path, f_dst_path)
def main_test_10026_53():
    data_path = "/home/disk/disk7/10021_bk/data/000.OCR/CRNN/data/train_Chinese/BaiduOCR_Chinese_Dataset/train_images"
    img_list = sorted(os.listdir(data_path))
    for f in tqdm(img_list):
        img_orig_name, suffix = os.path.splitext(f)[0].split("=")[0], os.path.splitext(f)[1]
        img_src_path = data_path + "/{}".format(f)
        img_dst_path = data_path + "/{}{}".format(img_orig_name, suffix)
        os.rename(img_src_path, img_dst_path)
def main_test_10026_54():
    data_path = "/home/disk/disk7/000.ChineseOCR/CRNN/data/train_Chinese/BaiduOCR_Chinese_Dataset"
    train_images_path = data_path + "/train_images"
    train_list_path = data_path + "/train.list"
    img_list = sorted(os.listdir(train_images_path))

    pattern = '[\u4e00-\u9fa5]+'
    labels = ""
    chinese_labels = ""

    with open(train_list_path, "r", encoding="utf-8") as fo:
        lines = fo.readlines()
        for line in tqdm(lines):
            try:
                line = line.strip()
                img_name = line.split("\t")[2]
                label = line.split("\t")[3]
                for lb in label:
                    if lb not in labels:
                        labels += lb
                    res = re.match(pattern, lb)
                    if res:
                        chinese_labels += lb

                # res = isAllChinese(label)
                # if res:
                #     chinese_labels += lb

            except Exception as Error:
                print(Error)

    print(len(labels), labels)
    print(len(chinese_labels), chinese_labels)

    alphabets = set(list(labels)) ^ set(list(chinese_labels))
    print(sorted(list(alphabets)))
def main_test_10026_55():
    data_path = "/home/disk/disk7/10021_bk/data/000.OCR/CRNN/data/train_Chinese/BaiduOCR_Chinese_Dataset/train_images_isAllDigits"
    img_list = sorted(os.listdir(data_path))
    # labels = ""
    #
    # for f in img_list:
    #     f_base_name, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
    #     imgname, label = f_base_name.split("=")[0], f_base_name.split("=")[1]
    #     for lb in label:
    #         if lb not in labels:
    #             labels += lb
    #
    # sorted(labels)

    for f in img_list:
        f_base_name, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
        imgname, label = f_base_name.split("=")[0], f_base_name.split("=")[1]
        lb_new = convert_label(label)
        f_src_path = data_path + "/{}".format(f)
        f_dst_path = data_path + "/{}={}".format(imgname, lb_new, suffix)
        os.rename(f_src_path, f_dst_path)
def main_test_10026_56():
    data_path = "/home/disk/disk7/000.ChineseOCR/CRNN/data/train_Chinese"
    img_list = sorted(os.listdir(data_path))
    for f in img_list:
        f_abs_path = data_path + "/{}".format(f)
        if os.path.isfile(f_abs_path) and f_abs_path.endswith(".jpg"):
            # print(f_abs_path)
            os.remove(f_abs_path)
def main_test_10026_57():
    data_path = "/home/disk/disk7/000.ChineseOCR/CRNN/data/train_Chinese/Synthetic_Chinese_String_Dataset"
    train_save_path = data_path + "/train_images"
    test_save_path = data_path + "/test_images"
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

    img_path = data_path + "/images"
    train_labels_path = data_path + "/Synthetic_Chinese_String_Dataset_labels/train.txt"
    test_labels_path = data_path + "/Synthetic_Chinese_String_Dataset_labels/test.txt"

    train_labels_fr = open(train_labels_path, "r", encoding="utf-8")
    test_labels_fr = open(test_labels_path, "r", encoding="utf-8")
    train_labels_data = train_labels_fr.readlines()
    test_labels_data = test_labels_fr.readlines()
    train_labels_fr.close()
    test_labels_fr.close()

    for line in train_labels_data:
        try:
            len_space = sum(1 for s in line if s == " ")
            assert len_space == 1, "len_space != 1"

            fname, label = line.strip().split(" ")[0], line.strip().split(" ")[1]
            f_base_name, suffix = os.path.splitext(fname)[0], os.path.splitext(fname)[1]
            img_abs_path = img_path + "/{}".format(fname)
            img_dst_path = train_save_path + "/{}={}{}".format(f_base_name, label, suffix)
            os.rename(img_abs_path, img_dst_path)
        except Exception as Error:
            print(Error)

    for line in test_labels_data:
        try:
            len_space = sum(1 for s in line if s == " ")
            assert len_space == 1, "len_space != 1"

            fname, label = line.strip().split(" ")[0], line.strip().split(" ")[1]
            f_base_name, suffix = os.path.splitext(fname)[0], os.path.splitext(fname)[1]
            img_abs_path = img_path + "/{}".format(fname)
            img_dst_path = test_save_path + "/{}={}{}".format(f_base_name, label, suffix)
            os.rename(img_abs_path, img_dst_path)
        except Exception as Error:
            print(Error)
def main_test_10026_58():
    data_path = "/home/disk/disk7/000.ChineseOCR/CRNN/data/train_Chinese/benchmarking-chinese-text-recognition/benchmark_dataset/document/NewFolder"
    dirs = os.listdir(data_path)
    for d in dirs:
        d_path = data_path + "/{}".format(d)
        if os.path.isdir(d_path):
            vertical_save_path = data_path + "/{}_vertical_images".format(d)
            # error_img_save_path = data_path + "/{}_error_images".format(d)
            os.makedirs(vertical_save_path, exist_ok=True)
            # os.makedirs(error_img_save_path, exist_ok=True)

            file_list = os.listdir(d_path)
            for f in file_list:
                f_abs_path = d_path + "/{}".format(f)
                try:
                    cv2img = cv2.imread(f_abs_path)
                    imgsz = cv2img.shape[:2]

                    if imgsz[0] > imgsz[1]:
                        f_dst_path = vertical_save_path + "/{}".format(f)
                        shutil.move(f_abs_path, f_dst_path)
                except Exception as Error:
                    print(Error)
                    # f_dst_path = error_img_save_path + "/{}".format(f)
                    # shutil.move(f_abs_path, f_dst_path)
def main_test_10026_59():
    data_path = "/home/disk/disk7/000.ChineseOCR/CRNN/data/train_Chinese/Datasets/Synthetic_Chinese_String_Dataset/Synthetic_Chinese_String_Dataset"
    img_path = data_path + "/images"
    train_label_path = data_path + "/Synthetic_Chinese_String_Dataset_labels/train.txt"
    test_label_path = data_path + "/Synthetic_Chinese_String_Dataset_labels/test.txt"

    train_img_save_path = data_path + "/New/train/train_images"
    test_img_save_path = data_path + "/New/test/test_images"
    os.makedirs(train_img_save_path, exist_ok=True)
    os.makedirs(test_img_save_path, exist_ok=True)
    train_label_save_path = data_path + "/New/train/train_labels.txt"
    test_label_save_path = data_path + "/New/test/test_labels.txt"

    train_fo = open(train_label_path, "r", encoding="utf-8")
    test_fo = open(test_label_path, "r", encoding="utf-8")
    train_data = train_fo.readlines()
    test_data = test_fo.readlines()
    train_fo.close()
    test_fo.close()

    train_fw = open(train_label_save_path, "w", encoding="utf-8")
    test_fw = open(test_label_save_path, "w", encoding="utf-8")

    for line in tqdm(train_data):
        try:
            l = line.strip()
            img_name, label = l.split(" ")[0], l.split(" ")[1]
            if "/" in label or "\\" in label:
                img_abs_path = img_path + "/{}".format(img_name)
                img_dst_path = train_img_save_path + "/{}".format(img_name)
                shutil.copy(img_abs_path, img_dst_path)

                train_fw.write(line)
        except Exception as Error:
            print(Error)

    for line in tqdm(test_data):
        try:
            l = line.strip()
            img_name, label = l.split(" ")[0], l.split(" ")[1]
            if "/" in label or "\\" in label:
                img_abs_path = img_path + "/{}".format(img_name)
                img_dst_path = test_img_save_path + "/{}".format(img_name)
                shutil.copy(img_abs_path, img_dst_path)

                test_fw.write(line)
        except Exception as Error:
            print(Error)

    train_fw.close()
    test_fw.close()
def main_test_10026_60():
    data_path = "/home/disk/disk7/000.ChineseOCR/CRNN/data/train_Chinese/BaiduOCR_Chinese_Dataset"
    img_path = data_path + "/train_images"
    train_label_path = data_path + "/train.list"

    train_img_save_path = data_path + "/New/train/train_images"
    os.makedirs(train_img_save_path, exist_ok=True)
    train_label_save_path = data_path + "/New/train/train_labels.txt"

    train_fo = open(train_label_path, "r", encoding="utf-8")
    train_data = train_fo.readlines()
    train_fo.close()

    train_fw = open(train_label_save_path, "w", encoding="utf-8")

    labels_unexpected = ['¥', '®', '°', '²', '³', '·', '×', 'í', 'Ƨ', 'ʌ', 'Λ', 'π', 'и', '—', '‘', '’', '“', '”', '•', '…', '‰', '′', '℃', '№', 'Ⅱ', '←', '↑', '→', '↓', '∶', '①', '②', '③', '④', '┅', '▪', '☆', '❤', '\u3000', '、', '。', '〇', '〈', '〉', '《', '》', '「', '」', '【', '】', 'の', 'サ', 'シ', 'ジ', 'マ', '㸃', '凉', '￥']
    labels_need_to_convert = ['！', '＂', '＃', '％', '＆', '＇', '（', '）', '＊', '＋', '，', '－', '．', '／', '０', '１', '２', '３', '４', '５', '６', '７', '８', '９', '：', '；', '＜', '＝', '＞', '？', '＠', 'Ａ', 'Ｂ', 'Ｃ', 'Ｄ', 'Ｅ', 'Ｆ', 'Ｇ', 'Ｈ', 'Ｉ', 'Ｊ', 'Ｋ', 'Ｌ', 'Ｍ', 'Ｎ', 'Ｏ', 'Ｐ', 'Ｑ', 'Ｒ', 'Ｓ', 'Ｔ', 'Ｕ', 'Ｖ', 'Ｗ', 'Ｘ', 'Ｙ', 'Ｚ', '［', '＼', '］', '＾', '＿', '｀', 'ａ', 'ｂ', 'ｃ', 'ｄ', 'ｅ', 'ｆ', 'ｇ', 'ｈ', 'ｉ', 'ｊ', 'ｋ', 'ｌ', 'ｍ', 'ｎ', 'ｏ', 'ｐ', 'ｑ', 'ｒ', 'ｓ', 'ｔ', 'ｕ', 'ｖ', 'ｗ', 'ｘ', 'ｙ', 'ｚ', '｛',
                              '｜', '｝', '～']
    labels_converted = ['!', '"', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|',
                        '}', '~']

    for line in tqdm(train_data):
        try:
            l = line.strip()
            img_name, label = l.split("\t")[2], l.split("\t")[3]

            lbun = 0
            for lb in label:
                if lb in labels_unexpected:
                    lbun += 1

            if lbun > 0:
                continue

            for i, lb in enumerate(label):
                if lb in labels_need_to_convert:
                    idx = labels_need_to_convert.index(lb)
                    # label[idx] = labels_converted[idx]
                    print(label[i], labels_converted[idx])
                    label = label.replace(label[i], labels_converted[idx])

            if "/" in label or "\\" in label:
                img_abs_path = img_path + "/{}".format(img_name)
                img_dst_path = train_img_save_path + "/{}".format(img_name)
                shutil.copy(img_abs_path, img_dst_path)

                line_new = "{} {}\n".format(img_name, label)
                train_fw.write(line_new)
        except Exception as Error:
            print(Error)

    train_fw.close()
def main_test_10026_61():
    """
    generate Chinese CRNN train.txt
    Returns
    -------

    """
    # data_path = "/home/disk/disk7/000.ChineseOCR/CRNN/data/train_Chinese/test/no_slash"
    # data_path = "/home/disk/disk7/data/000.ChineseOCR/CRNN/data/train_Chinese/test/no_slash"
    data_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash"
    dirs = sorted(os.listdir(data_path))

    save_label_path = data_path + "/no_slash_train.txt"

    # fw = open(save_label_path, "w", encoding="utf-8")
    with open(save_label_path, "w", encoding="utf-8") as fw:
        for d in dirs:
            d_path = data_path + "/{}".format(d)
            subdirs = sorted(os.listdir(d_path))
            for sbd in subdirs:
                sbd_path = d_path + "/{}".format(sbd)
                ssbdirs = sorted(os.listdir(sbd_path))
                for ssbd in ssbdirs:
                    ssbd_path = sbd_path + "/{}".format(ssbd)
                    file_list = sorted(os.listdir(ssbd_path))
                    for f in file_list:
                        f_abs_path = ssbd_path + "/{}".format(f)
                        f_base_name = os.path.splitext(f)[0]
                        first_idx = f_base_name.find("=")
                        label = f_base_name[first_idx + 1:]
                        content = "{} {}\n".format(f_abs_path, label)
                        fw.write(content)

    print("================")
    # fw.close()
def main_test_10026_62():
    """
    generate Chinese CRNN train.txt
    Returns
    -------

    """
    data_path = "/home/disk/disk7/000.ChineseOCR/CRNN/data/train_Chinese/test/slash"
    dirs = sorted(os.listdir(data_path))

    save_label_path = data_path + "/slash_test.txt"

    # fw = open(save_label_path, "w", encoding="utf-8")
    with open(save_label_path, "w", encoding="utf-8") as fw:
        for d in dirs:
            d_path = data_path + "/{}".format(d)
            d_train_labels_txt_path = d_path + "/test_labels.txt"
            img_path = d_path + "/test_images"
            with open(d_train_labels_txt_path, "r", encoding="utf-8") as fo:
                lines = fo.readlines()
                for line in lines:
                    line_new = img_path + "/{}".format(line)
                    fw.write(line_new)

        # for d in dirs:
        #     d_path = data_path + "/{}".format(d)
        #     subdirs = sorted(os.listdir(d_path))
        #     for sbd in subdirs:
        #         sbd_path = d_path + "/{}".format(sbd)
        #         ssbdirs = sorted(os.listdir(sbd_path))
        #         for ssbd in ssbdirs:
        #             ssbd_path = sbd_path + "/{}".format(ssbd)
        #             file_list = sorted(os.listdir(ssbd_path))
        #             for f in file_list:
        #                 f_abs_path = ssbd_path + "/{}".format(f)
        #                 f_base_name = os.path.splitext(f)[0]
        #                 first_idx = f_base_name.find("=")
        #                 label = f_base_name[first_idx + 1:]
        #                 content = "{} {}\n".format(f_abs_path, label)
        #                 fw.write(content)

    print("================")
    # fw.close()
def main_test_10026_63():
    CH_SIM_CHARS = ' ' + '0123456789.' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    CH_SIM_CHARS += ',;~!@#$%^&*()_+-={}:"<>?-=[]/|\\' + "'"
    CH_SIM_CHARS += '、。┅《》「」【】¥®πи‰℃№Ⅱ←↑→↓①②③④▪☆❤'
    ch_sim_chars = open("/home/wujiahu/code/crnn.pytorch-2023.10.20/utils/gen_fake/words/ch_sim_char.txt", "r", encoding="utf-8")
    lines = ch_sim_chars.readlines()
    for l in lines:
        CH_SIM_CHARS += l.strip()
    alpha = CH_SIM_CHARS
    print(len(alpha))
def main_test_10026_64():
    # root = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/15_cls_mini_v2"
    root = "/home/disk/disk7/data/000.ChineseOCR/data/train_experiment/no_slash"
    dirs = sorted(os.listdir(root))
    save_label_path = "/home/disk/disk7/data/000.ChineseOCR/data/train_experiment/no_slash/no_slash_train_experiment.txt"
    with open(save_label_path, "w", encoding="utf-8") as fw:
        for d in dirs:
            d_path = root + "/{}".format(d)
            sbdirs = sorted(os.listdir(d_path))
            for sbd in sbdirs:
                sbd_path = d_path + "/{}".format(sbd)
                ssbdirs = sorted(os.listdir(sbd_path))
                for ssbd in ssbdirs:
                    ssbd_path = sbd_path + "/{}".format(ssbd)
                    file_list = sorted(os.listdir(ssbd_path))
                    for f in file_list:
                        f_abs_path = ssbd_path + "/{}".format(f)
                        f_base_name = os.path.splitext(f)[0]
                        if "=" in f:
                            first_space_idx = f_base_name.find("=")
                            # x_path = f_base_name[:first_space_idx]
                            label = f_base_name[first_space_idx + 1:]

                            content = "{} {}\n".format(f_abs_path, label)
                            fw.write(content)

    print("OK")
def main_test_10026_65():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/CRNN/data/train_Chinese/train/no_slash/no_slash_train.txt"
    save_path = data_path.replace(".txt", "_new.txt")

    with open(save_path, "w", encoding="utf-8") as fw:
        with open(data_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.replace("disk7", "disk7/data")
                fw.write(line)

    print("OK!")
def main_test_10026_66():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/CRNN/data/train_Chinese/test/no_slash/no_slash_test.txt"
    save_path = data_path.replace(".txt", "_new.txt")

    with open(save_path, "w", encoding="utf-8") as fw:
        with open(data_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            for line in lines:
                if "BaiduOCR_Chinese_Dataset" not in line:
                    fw.write(line)

    print("OK!")
def main_test_10026_67_001_ICDAR2019_LSVT():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/001_ICDAR2019-LSVT"
    img_path0 = data_path + "/train_full_images_0"
    img_path1 = data_path + "/train_full_images_1"
    json_path = data_path + "/train_full_labels.json"

    save_path0 = data_path + "/cropped/0"
    save_path1 = data_path + "/cropped/1"
    os.makedirs(save_path0, exist_ok=True)
    os.makedirs(save_path1, exist_ok=True)

    json_fr = open(json_path, "r", encoding='utf-8')
    json_data = json.load(json_fr)
    # print(json_data)

    for idx in range(30000):
        fname = "gt_{}".format(idx)
        img_name = img_path0 + "/{}.jpg".format(fname)
        cv2img = cv2.imread(img_name)
        imgsz = cv2img.shape[:2]

        datai = json_data[fname]
        # print(datai)
        # print("============")
        for ii in range(len(datai)):
            try:
                points_idx_ii = datai[ii]['points']
                transcription_idx_ii = datai[ii]['transcription']
                minxy_maxxy = get_min_max_xy(points_idx_ii)
                reszHw = get_resize_hw(points_idx_ii)

                if "#" in transcription_idx_ii or "/" in transcription_idx_ii or "\\" in transcription_idx_ii: continue

                cropped0 = cv2img[minxy_maxxy[1]:minxy_maxxy[3], minxy_maxxy[0]:minxy_maxxy[2]]
                cv2.imwrite("{}/{}_{}_cropped0={}.jpg".format(save_path0, fname, ii, transcription_idx_ii), cropped0)

                p1 = np.array([points_idx_ii[0], points_idx_ii[1], points_idx_ii[3], points_idx_ii[2]], dtype=np.float32)
                p2 = np.array([[0, 0], [reszHw[1], 0], [0, reszHw[0]], [reszHw[1], reszHw[0]]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(p1, p2)
                warped = cv2.warpPerspective(cv2img, M, reszHw[::-1])
                cv2.imwrite("{}/{}_{}_warped={}.jpg".format(save_path1, fname, ii, transcription_idx_ii), warped)

            except Exception as Error:
                print(Error)
def main_test_10026_68_002_ICDAR2017_RCTW_17():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/002_ICDAR2017-RCTW-17/RCTW"
    img_path = data_path + "/train_images"
    lbl_path = data_path + "/train_gts"

    save_path0 = data_path + "/cropped/0"
    save_path1 = data_path + "/cropped/1"
    os.makedirs(save_path0, exist_ok=True)
    os.makedirs(save_path1, exist_ok=True)

    file_list = sorted(os.listdir(img_path))
    for f in tqdm(file_list):
        fname = os.path.splitext(f)[0]
        img_abs_path = img_path + "/{}".format(f)
        cv2img = cv2.imread(img_abs_path)
        imgsz = cv2img.shape[:2]

        lbl_abs_path = lbl_path + "/{}.txt".format(fname)
        fr = open(lbl_abs_path, "r", encoding="utf-8")
        lines = fr.readlines()
        fr.close()

        for line in lines:
            try:
                line = line.strip().split(",")
                points = []
                for ii in range(0, 8, 2):
                    points.append([int(line[ii]), int(line[ii + 1])])
                label = line[-1].replace('"', '')
                if "#" in label or "/" in label or "\\" in label: continue

                minxy_maxxy = get_min_max_xy(points)
                reszHw = get_resize_hw(points)

                cropped0 = cv2img[minxy_maxxy[1]:minxy_maxxy[3], minxy_maxxy[0]:minxy_maxxy[2]]
                cv2.imwrite("{}/{}_{}_cropped0={}.jpg".format(save_path0, fname, ii, label), cropped0)

                p1 = np.array([points[0], points[1], points[3], points[2]], dtype=np.float32)
                p2 = np.array([[0, 0], [reszHw[1], 0], [0, reszHw[0]], [reszHw[1], reszHw[0]]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(p1, p2)
                warped = cv2.warpPerspective(cv2img, M, reszHw[::-1])
                cv2.imwrite("{}/{}_{}_warped={}.jpg".format(save_path1, fname, ii, label), warped)
            except Exception as Error:
                print(Error)
def main_test_10026_68_005_ICDAR2019_ArT():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/005_ICDAR2019-ArT"
    img_path = data_path + "/train_task2_images"
    lbl_path = data_path + "/train_task2_labels.json"

    save_path = data_path + "/img_with_label_in_fname"
    os.makedirs(save_path, exist_ok=True)

    json_fr = open(lbl_path, "r", encoding="utf-8")
    json_data = json.load(json_fr)

    file_list = os.listdir(img_path)

    for f in tqdm(file_list):
        fname = os.path.splitext(f)[0]
        img_name = img_path + "/{}.jpg".format(fname)
        cv2img = cv2.imread(img_name)
        imgsz = cv2img.shape[:2]

        datai = json_data[fname]
        # print(datai)
        # print("============")
        for ii in range(len(datai)):
            try:
                points_idx_ii = datai[ii]['points']
                transcription_idx_ii = datai[ii]['transcription']
                minxy_maxxy = get_min_max_xy(points_idx_ii)
                reszHw = get_resize_hw(points_idx_ii)

                if "#" in transcription_idx_ii or "/" in transcription_idx_ii or "\\" in transcription_idx_ii: continue

                cv2.imwrite("{}/{}={}.jpg".format(save_path, fname, transcription_idx_ii), cv2img)
            except Exception as Error:
                print(Error)
def main_test_10026_68_004_Chinese_Document_Text_Recognition():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/004_Chinese_Document_Text_Recognition"
    img_path = data_path + "/images"
    lbl_path = data_path + "/labels.txt"

    save_path = "/home/disk/disk7/data/000.ChineseOCR/train_v1_add_20240315/004_Chinese_Document_Text_Recognition/images"
    os.makedirs(save_path, exist_ok=True)

    labels_fr = open(lbl_path, "r", encoding="utf-8")
    labels_data = labels_fr.readlines()
    labels_fr.close()

    for line in tqdm(labels_data):
        try:
            space_idx = line.find(" ")
            img_name = line[:space_idx]
            label = line[space_idx:].strip()
            if "###" in label or "/" in label or "\\" in label: continue

            img_abs_path = img_path + "/{}".format(img_name)
            cv2img = cv2.imread(img_abs_path)
            img_dst_path = save_path + "/{}={}.jpg".format(img_name, label)
            cv2.imwrite(img_dst_path, cv2img)
        except Exception as Error:
            print(Error)
def main_test_10026_69():
    # ============================= Chinese ============================
    CH_SIM_CHARS = ' ' + '0123456789.' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    CH_SIM_CHARS += ',;~!@#$%^&*()_+-={}:"<>?-=[]/|\\' + "'"
    CH_SIM_CHARS += '、。┅《》「」【】¥®πи‰℃№Ⅱ←↑→↓①②③④▪☆❤'
    ch_sim_chars = open("/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/ch_sim_char.txt", "r", encoding="utf-8")
    lines = ch_sim_chars.readlines()
    for l in lines:
        CH_SIM_CHARS += l.strip()
    alpha = CH_SIM_CHARS  # len = 6738
    # ============================= Chinese ============================

    labels_unexpected = ['¥', '®', '°', '²', '³', '·', '×', 'í', 'Ƨ', 'ʌ', 'Λ', 'π', 'и', '—', '‘', '’', '“', '”', '•', '…', '‰', '′', '℃', '№', 'Ⅱ', '←', '↑', '→', '↓', '∶', '①', '②', '③', '④', '┅', '▪', '☆', '❤', '\u3000', '、', '。', '〇', '〈', '〉', '《', '》', '「', '」', '【', '】', 'の', 'サ', 'シ', 'ジ', 'マ', '㸃', '凉', '￥']
    labels_need_to_convert = ['！', '＂', '＃', '％', '＆', '＇', '（', '）', '＊', '＋', '，', '－', '．', '／', '０', '１', '２', '３', '４', '５', '６', '７', '８', '９', '：', '；', '＜', '＝', '＞', '？', '＠', 'Ａ', 'Ｂ', 'Ｃ', 'Ｄ', 'Ｅ', 'Ｆ', 'Ｇ', 'Ｈ', 'Ｉ', 'Ｊ', 'Ｋ', 'Ｌ', 'Ｍ', 'Ｎ', 'Ｏ', 'Ｐ', 'Ｑ', 'Ｒ', 'Ｓ', 'Ｔ', 'Ｕ', 'Ｖ', 'Ｗ', 'Ｘ', 'Ｙ', 'Ｚ', '［', '＼', '］', '＾', '＿', '｀', 'ａ', 'ｂ', 'ｃ', 'ｄ', 'ｅ', 'ｆ', 'ｇ', 'ｈ', 'ｉ', 'ｊ', 'ｋ', 'ｌ', 'ｍ', 'ｎ', 'ｏ', 'ｐ', 'ｑ', 'ｒ', 'ｓ', 'ｔ', 'ｕ', 'ｖ', 'ｗ', 'ｘ', 'ｙ', 'ｚ', '｛',
                              '｜', '｝', '～']
    labels_converted = ['!', '"', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|',
                        '}', '~']

    train_chars = ""
    train_chars2 = ""

    root = "/home/disk/disk7/data/000.ChineseOCR/data/train"
    dirs = sorted(os.listdir(root))
    dir_name = os.path.basename(root)
    for d in dirs:
        if d == "no_label": continue
        train_data_path = root + "/{}/{}_{}_8602.txt".format(d, d, dir_name)
        train_data_path_new = root + "/{}/{}_{}_new_with_space.txt".format(d, d, dir_name)
        fw = open(train_data_path_new, "w", encoding="utf-8")
        train_data_path_new2 = root + "/{}/{}_{}_new2_no_space.txt".format(d, d, dir_name)
        fw2 = open(train_data_path_new2, "w", encoding="utf-8")

        with open(train_data_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip("\n")
                first_space_idx = line.find(" ")
                x_path = line[:first_space_idx]
                label = line[first_space_idx + 1:]

                # -------------------
                for lb in label:
                    if lb not in train_chars:
                        train_chars += lb

                num_un = 0
                num_not_in_alpha = 0
                for lb in label:
                    if lb in labels_unexpected:
                        num_un += 1
                    if lb not in alpha:
                        num_not_in_alpha += 1

                if num_un > 0 or num_not_in_alpha > 0:
                    continue

                for lb in label:
                    if lb not in train_chars2:
                        train_chars2 += lb

                for i, lb in enumerate(label):
                    if lb in labels_need_to_convert:
                        idx = labels_need_to_convert.index(lb)
                        # # label[idx] = labels_converted[idx]
                        # print(label[i], labels_converted[idx])
                        label = label.replace(label[i], labels_converted[idx])

                if not x_path.endswith(".jpg"):
                    print(x_path)
                    x_path = x_path + ".jpg"

                content_new = "{} {}\n".format(x_path, label)
                fw.write(content_new)

                content_new2 = "{} {}\n".format(x_path, label.replace(" ", ""))
                fw2.write(content_new2)

                # if len(label) > 30: continue
                #
                # lb_abnormal_num = 0
                # for lb in label:
                #     if lb not in alpha:
                #         lb_abnormal_num += 1
                # if lb_abnormal_num > 0:
                #     continue

        fw.close()
        fw2.close()

    print(train_chars)
    print(len(train_chars))
    print("\n")
    print(train_chars2)
    print(len(train_chars2))
def main_test_10026_70():
    """
    generate Chinese CRNN train.txt
    Returns
    -------

    """
    # data_path = "/home/disk/disk7/000.ChineseOCR/CRNN/data/train_Chinese/test/no_slash"
    # data_path = "/home/disk/disk7/data/000.ChineseOCR/CRNN/data/train_Chinese/test/no_slash"
    data_path = "/home/disk/disk7/data/000.ChineseOCR/data/test/slash"
    dirs = sorted(os.listdir(data_path))

    # save_label_path = data_path + "/no_slash_train.txt"

    # fw = open(save_label_path, "w", encoding="utf-8")
    # with open(save_label_path, "w", encoding="utf-8") as fw:

    for d in dirs:
        d_path = data_path + "/{}".format(d)
        if os.path.isdir(d_path):
            subdirs = sorted(os.listdir(d_path))
            for sbd in subdirs:
                sbd_path = d_path + "/{}".format(sbd)
                if os.path.isdir(sbd_path):
                    ssbdirs = sorted(os.listdir(sbd_path))
                    for ssbd in ssbdirs:
                        ssbd_path = sbd_path + "/{}".format(ssbd)
                        if os.path.isdir(ssbd_path):
                            file_list = sorted(os.listdir(ssbd_path))
                            for f in file_list:
                                f_abs_path = ssbd_path + "/{}".format(f)
                                if " " in f:
                                    print(f_abs_path)
                                    f_dst_path = ssbd_path + "/{}".format(f.replace(" ", ""))
                                    os.rename(f_abs_path, f_dst_path)
def main_test_10026_71():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/train_v1_add_20240315"
    dirs = os.listdir(data_path)

    chars = ""

    for d in dirs:
        d_path = data_path + "/{}".format(d)
        sdirs = os.listdir(d_path)
        for sd in sdirs:
            sd_path = d_path + "/{}".format(sd)
            file_list = sorted(os.listdir(sd_path))
            for f in file_list:
                f_abs_path = sd_path + "/{}".format(f)
                print(f_abs_path)

                f = os.path.splitext(f)[0]
                f_ = f.split("=")
                label = f_[1]

                for lb in label:
                    if lb not in chars:
                        chars += lb

    print(chars)
    print(len(chars))
def main_test_10026_72():
    """
    generate Chinese CRNN train.txt
    Returns
    -------

    """
    root = "/home/disk/disk7/data/000.ChineseOCR/data/test"
    dirs = sorted(os.listdir(root))
    dir_name = os.path.basename(root)
    for d in dirs:
        if d == "no_label": continue
        train_data_path = root + "/{}/{}_{}.txt".format(d, d, dir_name)
        # train_data_path_new = root + "/{}/{}_{}_new_with_space.txt".format(d, d, dir_name)
        # fw = open(train_data_path_new, "w", encoding="utf-8")
        train_data_path_new2 = root + "/{}/{}_{}_new2_no_space_exists.txt".format(d, d, dir_name)
        fw2 = open(train_data_path_new2, "w", encoding="utf-8")

        # num_ = 0

        with open(train_data_path, "r", encoding="utf-8") as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip("\n")
                first_space_idx = line.find(" ")
                x_path = line[:first_space_idx]

                pdir = os.path.abspath(os.path.join(x_path, "../.."))
                ppdir = os.path.abspath(os.path.join(pdir, "../.."))
                pdir_dirname = os.path.basename(pdir)
                ppdir_dirname = os.path.basename(ppdir)

                # if not os.path.exists(x_path):
                #     # print(x_path)
                #     if ppdir_dirname == "002_ICDAR2017-RCTW-17_cropped" and pdir_dirname == "1":

                #         num_ += 1

                label = line[first_space_idx + 1:]

                if os.path.exists(x_path) and label != "":
                    content_new2 = "{} {}\n".format(x_path, label.replace(" ", ""))
                    fw2.write(content_new2)

        # print("num_: ", num_)
        fw2.close()
def main_test_10026_73():
    data_path = "/home/wujiahu/myutils/data/test_20240318/images"
    file_list = sorted(os.listdir(data_path))

    save_path = make_save_path(data_path, "dst")

    image_process = ImageProcess()

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_src_path = data_path + "/{}".format(f)

        an = 0
        for attr_name in dir(image_process):
            attr = getattr(image_process, attr_name)
            if callable(attr) and not attr_name.startswith("__"):
                # print(attr_name)
                an += 1
                try:
                    img = cv2.imread(f_src_path)
                    out = attr(img)
                    f_dst_path = save_path + "/{}_{}.jpg".format(fname, attr_name)
                    cv2.imwrite(f_dst_path, out)
                except Exception as Error:
                    print(Error)
        print("an: ", an)
def main_test_10026_74():
    data_path = "/home/disk/disk7/data/011.GatePole_Det/cls/train/v1_cp/4"
    file_list = sorted(os.listdir(data_path))

    save_path = make_save_path(data_path, "dst")

    image_process = ImageProcess()

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_src_path = data_path + "/{}".format(f)
        img = cv2.imread(f_src_path)
        out = image_process.makeBorder(img, new_shape=(128, 128), r1=1, r2=0.25)
        f_dst_path = save_path + "/{}_{}.jpg".format(fname, "makeBorder")
        cv2.imwrite(f_dst_path, out)
def main_test_10026_75():
    data_path = "/home/wujiahu/myutils/data/test_20240318/images"
    f = "011_GatePole_20240312_0002214.jpg"
    fname = os.path.splitext(f)[0]
    f_src_path = data_path + "/{}".format(f)
    save_path = make_save_path(data_path, "dst")

    image_process = ImageProcess()

    for attr_name in sorted(dir(image_process)):
        attr = getattr(image_process, attr_name)
        if callable(attr) and not attr_name.startswith("__"):
            print(attr_name)
            try:
                img = cv2.imread(f_src_path)
                out = attr(img)
                f_dst_path = save_path + "/{}_{}.jpg".format(fname, attr_name)
                cv2.imwrite(f_dst_path, out)
            except Exception as Error:
                print(Error)
def main_test_10026_76():
    noise_aug = NoiseAug(ratio=1.0)
    blur_aug = BlurAug(type="EASY", ratio=1.0)
    hsv_aug = HSVAug(hgain=0.2, sgain=0.7, vgain=0.5, ratio=1.0)

    data_path = "/home/disk/disk7/data/000.ChineseOCR/data/test/no_label/BaiduOCR_Chinese_Dataset/test_images"
    file_list = sorted(os.listdir(data_path))

    save_path = make_save_path(data_path, "dst")

    image_process = ImageProcess()

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_src_path = data_path + "/{}".format(f)
        img_x = cv2.imread(f_src_path)

        img_x = noise_aug(img_x)
        img_x = blur_aug(img_x)
        img_x = hsv_aug(img_x)
        # img = TransAffine(img, degrees=8, translate=0.0, scale=0.2, shear=0, perspective=0, border=(2,border_width), prob=0.95)
        # img = TransAffine(img, degrees=3, translate=0.00025, scale=0.1, shear=3, perspective=0.0005, border=(border_height, border_width), prob=1.0)

        angle_v = np.random.randint(1, 8)
        translate_v = 0.001 * np.random.randint(0, 30)
        scale_v = 0.01 * np.random.randint(0, 15)
        shear_v = 0.1 * np.random.randint(0, 15)
        perspective_v = 0.00001 * np.random.randint(0, 30)

        img_x = TransAffine(img_x, degrees=angle_v, translate=translate_v, scale=scale_v, shear=shear_v, perspective=perspective_v, border=(0, 0), prob=1.0)
        img_x = img_x.astype(np.uint8)
        img_dst_path = "{}/{}".format(save_path, f)
        cv2.imwrite(img_dst_path, img_x)
def main_test_10026_77():
    data_path = "/home/disk/disk7/data/001.Banner_Det/kw.txt"
    res = []

    fr = open(data_path, "r", encoding="utf-8")
    lines = fr.readlines()
    fr.close()

    for line in lines:
        res.append(line.strip())

    print(res)
def main_test_10026_78():
    data_path = "/home/disk/disk7/data/001.Banner_Det/OCR/hf_ocr/train"
    file_list = sorted(os.listdir(data_path))
    for f in file_list:
        f_abs_path = data_path + "/{}".format(f)
        if os.path.isfile(f_abs_path):
            img = cv2.imread(f_abs_path)
            imgsz = img.shape[:2]
            if imgsz[0] >= imgsz[1]:
                f_dst_path = data_path + "/vertical/{}".format(f)
                shutil.move(f_abs_path, f_dst_path)
            else:
                f_dst_path = data_path + "/horizontal/{}".format(f)
                shutil.move(f_abs_path, f_dst_path)
def main_test_10026_79():
    data_path = "/home/disk/disk7/data/001.Banner_Det/OCR/hf_ocr/train/horizontal"
    file_list = sorted(os.listdir(data_path))

    # ============================= Chinese ============================
    CH_SIM_CHARS = ' ' + '0123456789.' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    CH_SIM_CHARS += ',;~!@#$%^&*()_+-={}:"<>?-=[]/|\\' + "'"
    CH_SIM_CHARS += '、。┅《》「」【】¥®πи‰℃№Ⅱ←↑→↓①②③④▪☆❤'
    print(len(CH_SIM_CHARS))

    ch_sim_chars = open("/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/ch_sim_char.txt", "r", encoding="utf-8")
    lines = ch_sim_chars.readlines()
    for l in lines:
        CH_SIM_CHARS += l.strip()
    alpha = CH_SIM_CHARS  # len = 6738

    CHARS = ""
    CHARS_N = ""
    for f in file_list:
        f_abs_path = data_path + "/{}".format(f)
        fname = os.path.splitext(f)[0]
        first_idx = fname.find("=")
        label = fname[first_idx + 1:]

        for lb in label:
            if lb not in CHARS:
                CHARS += lb
            if lb not in CHARS_N and lb not in alpha:
                CHARS_N += lb

    print("len(CHARS): {} \tCHARS: {}".format(len(CHARS), CHARS))
    print("len(CHARS_N): {} \tCHARS_N: {}".format(len(CHARS_N), CHARS_N))
def main_test_10026_80():
    data_path = "/home/disk/disk7/data/012.FastDeploy/others/v1/labels"
    file_list = sorted(os.listdir(data_path))
    save_path = make_save_path(data_path, "-1")

    for f in file_list:
        f_abs_path = data_path + "/{}".format(f)
        fr = open(f_abs_path, "r", encoding="utf-8")
        lines = fr.readlines()
        fr.close()

        for line in lines:
            line = line.strip().split(" ")
            cls = int(line[0])
            if cls == -1:
                f_dst_path = save_path + "/{}".format(f)
                shutil.copy(f_abs_path, f_dst_path)
def main_test_10026_81():
    data_path = "/home/disk/disk7/data/001.Banner_Det/OCR/hf_ocr/train/horizontal/horizontal_train_rename__rename"
    save_path = make_save_path(data_path, "rename")
    file_list = sorted(os.listdir(data_path))

    # ============================= Chinese ============================
    CH_SIM_CHARS = ' ' + '0123456789.' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    CH_SIM_CHARS += ',;~!@#$%^&*()_+-={}:"<>?-=[]/|\\' + "'"
    CH_SIM_CHARS += '、。┅《》「」【】¥®πи‰℃№Ⅱ←↑→↓①②③④▪☆❤'
    print(len(CH_SIM_CHARS))

    ch_sim_chars = open("/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/ch_sim_char.txt", "r", encoding="utf-8")
    lines = ch_sim_chars.readlines()
    for l in lines:
        CH_SIM_CHARS += l.strip()
    alpha = CH_SIM_CHARS  # len = 6738

    # CHARS = ""
    # CHARS_N = ""
    # for f in file_list:
    #     f_abs_path = data_path + "/{}".format(f)
    #     fname = os.path.splitext(f)[0]
    #     first_idx = fname.find("=")
    #     label = fname[first_idx + 1:]
    #
    #     for lb in label:
    #         if lb not in CHARS:
    #             CHARS += lb
    #         if lb not in CHARS_N and lb not in alpha:
    #             CHARS_N += lb
    #
    # print("len(CHARS): {} \tCHARS: {}".format(len(CHARS), CHARS))
    # print("len(CHARS_N): {} \tCHARS_N: {}".format(len(CHARS_N), CHARS_N))

    # CHARS_N = "，“”！；（）—：·贛對員衞園？險陸～峯灀媖郵電琊屌們‘撚樂隊"
    # CHARS_N = "贛對員衞園險陸峯灀媖郵電琊屌們撚樂隊"

    # for f in file_list:
    #     f_abs_path = data_path + "/{}".format(f)
    #     fname = os.path.splitext(f)[0]
    #     first_idx = fname.find("=")
    #     label = fname[first_idx + 1:]
    #
    #     for lb in label:
    #         if lb in CHARS_N:
    #             f_dst_path = save_path + "/{}".format(f)
    #             shutil.move(f_abs_path, f_dst_path)
    #             break

    for f in file_list:
        try:
            f_abs_path = data_path + "/{}".format(f)
            fname, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
            first_idx = fname.find("=")
            # label = fname[first_idx + 1:]
            second_idx = fname[first_idx + 1:].find("=")
            label = fname[first_idx + 1:][second_idx + 1:]
            fname_new = fname[:first_idx]

            # label = label.replace(label[first_idx:second_idx], "")

            # label = label.replace("，", ",").replace('“', '"').replace("！", "!").replace("；", ";").replace("（", "(").replace("）", ")").replace("—", "-").replace("：", ":").replace("·", ".").replace('？', '?').replace("～", "~").replace("‘", "'").replace('”', '"')
            f_dst_path = save_path + "/{}={}{}".format(fname_new, label, suffix)
            os.rename(f_abs_path, f_dst_path)
        except Exception as Error:
            print(Error)
    #
    # # print("len(CHARS): {} \tCHARS: {}".format(len(CHARS), CHARS))
    # # print("len(CHARS_N): {} \tCHARS_N: {}".format(len(CHARS_N), CHARS_N))
def main_test_10026_82():
    # root = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/15_cls_mini_v2"
    root = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash"
    dirs = sorted(os.listdir(root))
    save_label_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/no_slash_banner.txt"
    with open(save_label_path, "w", encoding="utf-8") as fw:
        for d in dirs:
            d_path = root + "/{}".format(d)
            if os.path.isdir(d_path):
                if d == "train_v1_add_20240326":
                    sbdirs = sorted(os.listdir(d_path))
                    for sbd in sbdirs:
                        sbd_path = d_path + "/{}".format(sbd)
                        ssbdirs = sorted(os.listdir(sbd_path))
                        for ssbd in ssbdirs:
                            ssbd_path = sbd_path + "/{}".format(ssbd)
                            file_list = sorted(os.listdir(ssbd_path))
                            for f in file_list:
                                f_abs_path = ssbd_path + "/{}".format(f)
                                f_base_name = os.path.splitext(f)[0]
                                if "=" in f:
                                    first_space_idx = f_base_name.find("=")
                                    # x_path = f_base_name[:first_space_idx]
                                    label = f_base_name[first_space_idx + 1:]

                                    content = "{} {}\n".format(f_abs_path, label)
                                    fw.write(content)

    print("OK")
def main_test_10026_83():
    path1 = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/no_slash_train.txt"
    path2 = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/no_slash_banner.txt"

    fa = open(path1, "a", encoding="utf-8")
    fr = open(path2, "r", encoding="utf-8")
    banner_lines = fr.readlines()
    fr.close()

    print("1: ", banner_lines[-1])

    for line in banner_lines:
        fa.write(line)

    fa.close()

    print("2:", banner_lines[-1])

    # path1 = "/home/disk/disk7/data/000.ChineseOCR/data/train_experiment/no_slash/no_slash_train_experiment.txt"
    # path2 = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/no_slash_banner.txt"
    #
    # fa = open(path1, "a", encoding="utf-8")
    # fr = open(path2, "r", encoding="utf-8")
    #
    # # fa_lines = fa.readlines()
    # # print("0: ", fa_lines[-1])
    #
    # banner_lines = fr.readlines()
    # fr.close()
    # print("1: ", banner_lines[-1])
    #
    # for line in banner_lines:
    #     fa.write(line)
    #
    # # fa_lines_ = fa.readlines()
    # # print("2:", fa_lines_[-1])
    # fa.close()
def main_test_10026_84():
    # path1 = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/no_slash_train_20240326.txt"
    # path2 = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/no_slash_banner.txt"
    path1 = "/home/disk/disk7/data/000.ChineseOCR/data/train_experiment/no_slash/no_slash_train_experiment (copy 1).txt"
    path2 = "/home/disk/disk7/data/000.ChineseOCR/data/train_experiment/no_slash/no_slash_train_experiment.txt"

    fr1 = open(path1, "r", encoding="utf-8")
    fr2 = open(path2, "r", encoding="utf-8")
    lines1 = fr1.readlines()
    lines2 = fr2.readlines()
    fr1.close()
    fr2.close()

    print("1: ", lines1[-1])
    print("2: ", lines2[-1])
def main_test_10026_85():
    path1 = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/no_slash_train.txt"
    save_path = "/home/disk/disk7/data/000.ChineseOCR/data/train_experiment/no_slash/no_slash_train_experiment.txt"

    fr1 = open(path1, "r", encoding="utf-8")
    lines1 = fr1.readlines()
    fr1.close()

    r = 0.005
    f2 = open(save_path, "w", encoding="utf-8")
    lines2 = random.sample(lines1, int(len(lines1) * r))
    for line in lines2:
        f2.write(line)
    f2.close()
def main_test_10026_86():
    path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/no_slash_banner.txt"
    save_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/no_slash_banner_new.txt"

    fr1 = open(path, "r", encoding="utf-8")
    lines1 = fr1.readlines()
    fr1.close()

    fw = open(save_path, "w", encoding="utf-8")
    for line in lines1:
        l = line.strip().split(" ")
        fpath = l[0]
        if os.path.exists(fpath):
            fw.write(line)

    fw.close()
def main_test_10026_87():
    data_path = "/home/disk/disk7/data/001.Banner_Det/kw3.txt"
    kws = []
    fr = open(data_path, "r", encoding="utf-8")
    lines = fr.readlines()
    for line in lines:
        l = line.strip()
        kws.append(l)

    print(kws)
def main_test_10026_88():
    data = [
        "恭喜", "祝贺", "热烈", "欢迎", "莅临", "指导", "珍惜", "爱护", "文明", "安全", "旗帜", "小康", "伟大", "胜利", "风斗",
        "努力", "佳绩", "精彩", "辉煌", "携手", "共赢", "领导", "振兴", "中华", "中国", "团结", "团队", "梦想", "超越", "共创",
        "创造", "未来", "明天", "前景", "美好", "平安", "教育", "安全", "溺水", "火灾", "预防",
        "永无止境", "超越自我", "共创辉煌", "一路向前", "风雨同舟", "同心协力", "金榜题名", "雷锋精神", "圆满成功", "社会主义",

        "欺骗", "欺诈", "骗子", "业主", "还钱", "欠债", "欠钱", "老板", "跑路", "打倒", "你妈", "他妈", "你妹", "垃圾", "滚出",
        "沙雕", "傻吊", "傻屌", "傻叼", "傻D", "傻d", "傻逼", "煞笔", "傻比", "煞笔", "傻B", "傻b", "SB", "sb", "CNM", "cnm",
        "Sb", "sB", "蠢货", "愚蠢", "鲁莽", "粗莽", "禽兽", "下流", "卑鄙", "无耻", "色情", "黄色", "泼妇", "骂人", "走狗",
        "贱人", "贱货", "村霸", "霸王", "美国", "日本", "西方", "坑害", "滚出", "逾期", "造假", "退钱", "还钱", "欠钱", "退车",
        "恶意", "打击", "欺压", "欺负", "赔偿", "严厉", "声讨", "公道", "公平", "工资", "杀人", "偿命", "黑心", "无德", "缺德",
        "违规", "违约", "规避", "非法", "无视", "维权", "侵权", "祸害", "讨薪", "骗子", "请愿", "尊严", "公道", "远离", "骗钱",
        "贷款", "偷税", "逃税", "漏税", "赔偿", "补偿", "歪曲", "侵犯", "权益", "权力", "权利", "欺压", "腐败", "贪污", "讨要",
        "投资", "合法", "费用", "通奸", "房款", "房贷", "贷款", "汇款", "刷单", "裸聊", "被骗", "欢迎", "骗财", "抵制", "伤害",
        "补偿", "赔偿", "金钱", "坑爹", "退款", "高价", "质量", "横幅", "庆祝", "韭菜", "拒绝", "责任", "担责", "利益", "群众",
        "烂尾", "诉求", "蒙骗", "诚信", "造谣", "信谣", "汗水", "无良", "黑恶", "严禁", "爱国", "违纪", "高额", "被骗", "转账",

        "受害者", "血汗钱", "开发商", "烂尾楼", "豆腐渣", "法西斯", "不交房", "消费者", "消费者", "开发商", "幸苦钱", "老百姓",
        "房产证", "不合格", "养老金", "不作为", "王八蛋",

        "帝国主义", "狼心狗肺", "衣冠禽兽", "猪狗不如", "厚颜无耻", "狗仗人势", "偷工减料", "丧心病狂", "丧尽天良", "狼狈为奸",
        "卑鄙无耻", "禽兽不如", "忘恩负义", "吃里扒外", "为虎作伥", "祸国殃民", "恩将仇报", "恬不知耻", "污言秽语", "血口喷人",
        "蛇蝎心肠", "掩耳盗铃", "无恶不作", "好吃懒做", "遗臭万年", "惨无人道", "两面三刀", "人面兽心", "损人利己", "魑魅魍魉",
        "害群之马", "伤天害理", "无法无天", "草菅人命", "无理取闹", "不讲诚信", "惩奸除恶", "饱受折磨", "无法维持", "虚假宣传",
        "良心何在", "天理难容", "强烈要求", "合理合法", "天理难容", "坚决反对", "珍爱生命", "漫天要价", "垃圾工程", "扫黑除恶",
        "血本无归", "血债血还", "瞒天过海", "凝心聚力",
    ]

    dd = sorted(list(set(data)))
    # print(dd)

    print(len(data))
    print(len(dd))

    uu = []

    unique = {}.fromkeys(data).keys()
    print(unique)
    print(len(unique))

    ddd = [
        "恭喜", "祝贺", "热烈", "欢迎", "莅临", "指导", "珍惜", "爱护", "文明", "安全", "旗帜", "小康", "伟大", "胜利", "奋斗", "努力", "佳绩", "精彩", "辉煌", "携手",
        "共赢", "领导", "振兴", "中华", "中国", "团结", "团队", "梦想", "超越", "共创", "创造", "未来", "明天", "前景", "美好", "平安", "教育", "溺水", "火灾", "预防",
        "永无止境", "超越自我", "共创辉煌", "一路向前", "风雨同舟", "同心协力", "凝心聚力", "金榜题名", "雷锋精神", "圆满成功", "社会主义", "有限公司",
        "欺骗", "欺诈", "骗子", "业主", "还钱", "欠债", "欠钱", "老板", "跑路", "打倒", "你妈", "他妈", "你妹", "垃圾", "滚出", "沙雕", "傻吊", "傻屌", "傻叼", "傻D",
        "傻d", "傻逼", "煞笔", "傻比", "傻B", "傻b", "SB", "sb", "CNM", "cnm", "Sb", "sB", "蠢货", "愚蠢", "鲁莽", "粗莽", "禽兽", "下流", "卑鄙", "无耻", "色情",
        "黄色", "泼妇", "骂人", "走狗", "贱人", "贱货", "村霸", "霸王", "美国", "日本", "西方", "坑害", "逾期", "造假", "退钱", "退车", "恶意", "打击", "欺压", "欺负",
        "赔偿", "严厉", "声讨", "公道", "公平", "工资", "杀人", "偿命", "黑心", "无德", "缺德", "违规", "违约", "规避", "非法", "无视", "维权", "侵权", "祸害", "讨薪",
        "请愿", "尊严", "远离", "骗钱", "贷款", "偷税", "逃税", "漏税", "补偿", "歪曲", "侵犯", "权益", "权力", "权利", "腐败", "贪污", "讨要", "投资", "合法", "费用",
        "通奸", "房款", "房贷", "汇款", "刷单", "裸聊", "被骗", "骗财", "抵制", "伤害", "金钱", "坑爹", "退款", "高价", "质量", "横幅", "庆祝", "韭菜", "拒绝", "责任",
        "担责", "利益", "群众", "烂尾", "诉求", "蒙骗", "诚信", "造谣", "信谣", "汗水", "无良", "黑恶", "严禁", "爱国", "违纪", "高额", "转账",
        "受害者", "血汗钱", "开发商", "烂尾楼", "豆腐渣", "法西斯", "不交房", "消费者", "幸苦钱", "老百姓", "房产证", "不合格", "养老金", "不作为", "王八蛋",
        "帝国主义", "狼心狗肺", "衣冠禽兽", "猪狗不如", "厚颜无耻", "狗仗人势", "偷工减料", "丧心病狂", "丧尽天良", "狼狈为奸", "卑鄙无耻", "禽兽不如", "忘恩负义", "吃里扒外",
        "为虎作伥", "祸国殃民", "恩将仇报", "恬不知耻", "污言秽语", "血口喷人", "蛇蝎心肠", "掩耳盗铃", "无恶不作", "好吃懒做", "遗臭万年", "惨无人道", "两面三刀", "人面兽心",
        "损人利己", "魑魅魍魉", "害群之马", "伤天害理", "无法无天", "草菅人命", "无理取闹", "不讲诚信", "惩奸除恶", "饱受折磨", "无法维持", "虚假宣传", "良心何在", "天理难容",
        "强烈要求", "合理合法", "坚决反对", "珍爱生命", "漫天要价", "垃圾工程", "扫黑除恶", "血本无归", "血债血还", "瞒天过海"
    ]
def main_test_10026_89():
    path1 = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash_train.txt"
    path2 = "/home/disk/disk7/data/000.ChineseOCR/data/train/digits_train.txt"
    path3 = "/home/disk/disk7/data/000.ChineseOCR/data/train/slash_train.txt"

    fa = open(path1, "a", encoding="utf-8")
    fr = open(path2, "r", encoding="utf-8")
    fr3 = open(path3, "r", encoding="utf-8")

    # fa_lines = fa.readlines()
    # print("0: ", fa_lines[-1])

    lines2 = fr.readlines()
    fr.close()
    print("1: ", lines2[-1])

    lines3 = fr3.readlines()
    fr3.close()
    print("1: ", lines3[-1])

    for line in lines2:
        fa.write(line)
    for line in lines3:
        fa.write(line)

    # fa_lines_ = fa.readlines()
    # print("2:", fa_lines_[-1])
    fa.close()
def main_test_10026_90():
    """
    sliding_window_crop test
    Returns
    -------

    """
    img_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/train_v1_add_20240326/banner/horizontal_train/band_00000763_1_1.1=人生因梦想而绚丽多彩生活因语文而乐趣无穷.jpg"
    # img = cv2.imread(img_path)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # final_imgs = sliding_window_crop(img, cropsz=new_shape, gap=(0, new_shape[1] // 8))
    makeBorderRes = makeBorder_v5(img, new_shape=(64, 256), r1=0, r2=0.25, sliding_window=True)

    if isinstance(makeBorderRes, list):
        if len(makeBorderRes) > 1:
            print("len(makeBorders) > 1")
        for mi, mb in enumerate(makeBorderRes):
            cv2.imwrite("/home/wujiahu/myutils/data/test_20240327/{}.jpg".format(mi), mb)
    else:
        cv2.imwrite("/home/wujiahu/myutils/data/test_20240327/makeBorderRes.jpg", makeBorderRes)
def main_test_10026_91():
    x = torch.randn(1, 1024, 2, 64)
    y = torch.chunk(x, chunks=2, dim=2)
    print(y[0].shape, y[1].shape)

    x2np = np.array(list(range(12))).reshape(1, 4, 3)
    x2 = torch.tensor(x2np)
    print(x2)

    y2 = torch.chunk(x2, chunks=2, dim=1)
    print(y2[0], y2[1])
    print(y2[0].shape, y2[1].shape)

    y3 = torch.cat((y2[0], y2[1]), dim=2)
    print(y3, y3.shape)
def main_test_10026_92():
    img_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/train_v1_add_20240326/banner/horizontal_train/band_00000763_1_1.1=人生因梦想而绚丽多彩生活因语文而乐趣无穷.jpg"
    # img = cv2.imread(img_path)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # final_imgs = sliding_window_crop(img, cropsz=new_shape, gap=(0, new_shape[1] // 8))
    # makeBorderRes = makeBorder_v5(img, new_shape=(64, 256), r1=0, r2=0.25, sliding_window=True)
    makeBorderRes = makeBorder_v6(img, new_shape=(64, 256), r1=0, r2=0.25, sliding_window=True, specific_color_flag=True, gap_r=(0, 11 / 12), last_img_makeBorder=True)

    if isinstance(makeBorderRes, list):
        for mi, mb in enumerate(makeBorderRes):
            cv2.imwrite("/home/wujiahu/myutils/data/test_20240327/{}.jpg".format(mi), mb)
    else:
        cv2.imwrite("/home/wujiahu/myutils/data/test_20240327/makeBorderRes.jpg", makeBorderRes)
def main_test_ip30_93():
    # rec_predictor.m_CHARS_KeyWords = {
    #     "恭喜", "祝贺", "热烈", "欢迎", "莅临", "指导", "珍惜", "爱护", "文明", "安全", "旗帜", "小康", "伟大", "胜利", "奋斗", "努力", "佳绩", "精彩", "辉煌", "携手",
    #     "共赢", "领导", "振兴", "中华", "中国", "团结", "团队", "梦想", "超越", "共创", "创造", "未来", "明天", "前景", "美好", "平安", "教育", "溺水", "火灾", "预防",
    #     "永无止境", "超越自我", "共创辉煌", "一路向前", "风雨同舟", "同心协力", "凝心聚力", "金榜题名", "雷锋精神", "圆满成功", "社会主义", "有限公司",
    #     "欺骗", "欺诈", "骗子", "业主", "还钱", "欠债", "欠钱", "老板", "跑路", "打倒", "你妈", "他妈", "你妹", "垃圾", "滚出", "沙雕", "傻吊", "傻屌", "傻叼", "傻D",
    #     "傻d", "傻逼", "煞笔", "傻比", "傻B", "傻b", "SB", "sb", "CNM", "cnm", "Sb", "sB", "蠢货", "愚蠢", "鲁莽", "粗莽", "禽兽", "下流", "卑鄙", "无耻", "色情",
    #     "黄色", "泼妇", "骂人", "走狗", "贱人", "贱货", "村霸", "霸王", "美国", "日本", "西方", "坑害", "逾期", "造假", "退钱", "退车", "恶意", "打击", "欺压", "欺负",
    #     "赔偿", "严厉", "声讨", "公道", "公平", "工资", "杀人", "偿命", "黑心", "无德", "缺德", "违规", "违约", "规避", "非法", "无视", "维权", "侵权", "祸害", "讨薪",
    #     "请愿", "尊严", "远离", "骗钱", "贷款", "偷税", "逃税", "漏税", "补偿", "歪曲", "侵犯", "权益", "权力", "权利", "腐败", "贪污", "讨要", "投资", "合法", "费用",
    #     "通奸", "房款", "房贷", "汇款", "刷单", "裸聊", "被骗", "骗财", "抵制", "伤害", "金钱", "坑爹", "退款", "高价", "质量", "横幅", "庆祝", "韭菜", "拒绝", "责任",
    #     "担责", "利益", "群众", "烂尾", "诉求", "蒙骗", "诚信", "造谣", "信谣", "汗水", "无良", "黑恶", "严禁", "爱国", "违纪", "高额", "转账",
    #     "受害者", "血汗钱", "开发商", "烂尾楼", "豆腐渣", "法西斯", "不交房", "消费者", "幸苦钱", "老百姓", "房产证", "不合格", "养老金", "不作为", "王八蛋",
    #     "帝国主义", "狼心狗肺", "衣冠禽兽", "猪狗不如", "厚颜无耻", "狗仗人势", "偷工减料", "丧心病狂", "丧尽天良", "狼狈为奸", "卑鄙无耻", "禽兽不如", "忘恩负义", "吃里扒外",
    #     "为虎作伥", "祸国殃民", "恩将仇报", "恬不知耻", "污言秽语", "血口喷人", "蛇蝎心肠", "掩耳盗铃", "无恶不作", "好吃懒做", "遗臭万年", "惨无人道", "两面三刀", "人面兽心",
    #     "损人利己", "魑魅魍魉", "害群之马", "伤天害理", "无法无天", "草菅人命", "无理取闹", "不讲诚信", "惩奸除恶", "饱受折磨", "无法维持", "虚假宣传", "良心何在", "天理难容",
    #     "强烈要求", "合理合法", "坚决反对", "珍爱生命", "漫天要价", "垃圾工程", "扫黑除恶", "血本无归", "血债血还", "瞒天过海"
    # };
    pass
def main_test_ip30_94():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/train_v1_add_20240326/banner/horizontal_train"
    file_list = get_file_list(data_path)
    save_path = make_save_path(data_path, "sliding_window_crop2")
    gap_r = (0, 11 / 12)

    for f in file_list:
        fname = os.path.splitext(f)[0]
        first_space_idx = fname.find("=")
        img_name = fname[:first_space_idx]
        label = fname[first_space_idx + 1:]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        # imgs = sliding_window_crop_v2(img, cropsz=(64, 256), gap=(0, int(256 * gap_r[1])), makeBorder=False, r1=0, r2=0.25, specific_color_flag=True)
        imgs = makeBorder_v6(img, new_shape=(64, 256), r1=0, r2=0.25, sliding_window=True, specific_color_flag=True, gap_r=gap_r, last_img_makeBorder=False)

        if isinstance(imgs, list):
            for i, imgi in enumerate(imgs):
                imgi_dst_path = save_path + "/{}_{}={}.jpg".format(img_name, i, label)
                cv2.imwrite(imgi_dst_path, imgi)
        else:
            imgi_dst_path = save_path + "/{}={}.jpg".format(img_name, label)
            cv2.imwrite(imgi_dst_path, imgs)


def main_test_ip30_95():
    # root = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/15_cls"
    root = "/home/disk/disk7/data/010.Digital_Rec/crnn/test/15_cls"
    # root = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash"
    dirs = sorted(os.listdir(root))
    save_label_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/test/15_cls/15_cls_test.txt"
    with open(save_label_path, "w", encoding="utf-8") as fw:
        for d in dirs:
            d_path = root + "/{}".format(d)
            if os.path.isdir(d_path):
                file_list = sorted(os.listdir(d_path))
                for f in file_list:
                    f_abs_path = d_path + "/{}".format(f)
                    f_base_name = os.path.splitext(f)[0]
                    if "=" in f:
                        first_space_idx = f_base_name.find("=")
                        # x_path = f_base_name[:first_space_idx]
                        label = f_base_name[first_space_idx + 1:]

                        content = "{} {}\n".format(f_abs_path, label)
                        fw.write(content)




                # if d == "train_v1_add_20240326":
                # sbdirs = sorted(os.listdir(d_path))
                # for sbd in sbdirs:
                #     sbd_path = d_path + "/{}".format(sbd)
                #     ssbdirs = sorted(os.listdir(sbd_path))
                #     for ssbd in ssbdirs:
                #         ssbd_path = sbd_path + "/{}".format(ssbd)
                #         file_list = sorted(os.listdir(ssbd_path))
                #         for f in file_list:
                #             f_abs_path = ssbd_path + "/{}".format(f)
                #             f_base_name = os.path.splitext(f)[0]
                #             if "=" in f:
                #                 first_space_idx = f_base_name.find("=")
                #                 # x_path = f_base_name[:first_space_idx]
                #                 label = f_base_name[first_space_idx + 1:]
                #
                #                 content = "{} {}\n".format(f_abs_path, label)
                #                 fw.write(content)

    print("OK")


def main_test_ip30_96():
    # data_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/train_v1/train_v1"
    # dirs = get_dir_list(data_path, abspath=False)
    #
    # save_path = "/home/disk/disk7/docker/Projects/000_ChineseOCR/ChineseOCR_images"
    # os.makedirs(save_path, exist_ok=True)
    #
    # for d in dirs:
    #     d_path = data_path + "/{}".format(d)
    #     file_list = get_file_list(d_path, abspath=False)
    #     selected_files = random.sample(file_list, 15)
    #     for f in selected_files:
    #         f_src_path = d_path + "/{}".format(f)
    #         f_dst_path = save_path + "/{}".format(f)
    #         shutil.copy(f_src_path, f_dst_path)

    # ============================================================================================
    data_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/train_v1_add_20240326"
    dirs = get_dir_list(data_path, abspath=False)

    save_path = "/home/disk/disk7/docker/Projects/000_ChineseOCR/ChineseOCR_images_train_v1_add_20240326"
    os.makedirs(save_path, exist_ok=True)

    for d in dirs:
        d_path = data_path + "/{}".format(d)
        sbdirs = get_dir_list(d_path, abspath=False)
        for sbd in sbdirs:
            sbd_path = d_path + "/{}".format(sbd)
            file_list = get_file_list(sbd_path, abspath=False)
            selected_files = random.sample(file_list, 600)
            for f in selected_files:
                f_src_path = sbd_path + "/{}".format(f)
                f_dst_path = save_path + "/{}".format(f)
                shutil.copy(f_src_path, f_dst_path)
            print(sbd_path)
        print(d_path)


def main_test_ip30_97():
    data_path = "/home/disk/disk7/data/012.FastDeploy/train/v2/digital_pointer_meter"
    lables_path = data_path + "/labels"
    selected_path = data_path + "/selected_labels"
    os.makedirs(selected_path, exist_ok=True)

    file_list = get_file_list(lables_path, abspath=False)
    for f in file_list:
        f_abs_path = lables_path + "/{}".format(f)
        fr = open(f_abs_path, "r", encoding="utf-8")
        lines = fr.readlines()

        num = 0
        for line in lines:
            line = line.strip().split(" ")
            cls = int(line[0])
            if cls == -1:
                num += 1
        if num > 0:
            f_dst_path = selected_path + "/{}".format(f)
            shutil.copy(f_abs_path, f_dst_path)


def main_test_ip30_98():
    # ============================= Chinese ============================
    # chars0 = ' ' + '0123456789' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    chars0 = '0123456789' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    chars1 = ",./;'[]\\`-=" + '<>?:"{}|~!@#$%^&*()_+'
    chars2 = "，。、；’【】·" + "《》？：”｛｝｜～！＠＃￥％…（）" + "║〈〉［］〔〕「」︵︷︿︹︽﹁︻︶︸﹀︺︾﹂︼"
    chars3 = "⓪①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳❶❷❸❹❺❻❼❽❾❿⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴㊀㊁㊂㊃㊄㊅㊆㊇㊈㊉㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹ"
    chars4 = "ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵"
    chars5 = "㊎㊍㊌㊋㊏㊚㊛㊐㊊㊣㊤㊥㊦㊧㊨㊒㊫㊑㊓㊔㊕㊖㊗㊘㊜㊝㊞㊟㊠㊡㊢㊩㊪㊬㊭㊮㊯㊰"
    chars6 = "€£Ұ₴$₰¢₤¥₳₲₪₵₣₱฿¤₡₮₭₩ރ円₢₥₫₦zł﷼₠₧₯₨Kčर₹ƒ₸￠"
    chars7 = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζνξοπρσηθικλμτυφχψω"
    chars8 = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    chars9 = "āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜüㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩ"
    chars10 = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんゔゕゖァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶヷヸヹヺ・ーヽヾヿ゠ㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"
    chars11 = "°℃℉㎡㏕㎜㎝㎞㏎㎎㎏㏄○¹²³"
    chars12 = "↑↓←→↖↗↘↙↔↕▶◀«»➤⇐⇑⇒⇓⇔⇖⇗⇘⇙↰↱↲↳↴↵↶↷↺↻"
    chars13 = "÷±≌∽≦≧≒≈≡≠≤≥<>≮≯∷∶∫∮∝∞∧∨∑∏∪∩∈∵∴⊥∥∠⌒∟⊿㏒㏑%‰≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟≠≡≢≣≨≩⊰⊱⋛⋚∫∬∭∮∯∰∱∲∳℅‱øØπ"
    chars14 = "卍卐囍㍿❤❥♥❣♠♣⌘▸◂▴▾√×☢★☯❄♂♀☭☻♬♪♩♭§╬♫◣◢◥◤‼⁉™©®℗◆※№¶"

    txt_path0 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/ch_sim_char.txt"
    ch_sim_chars = ""
    fr0 = open(txt_path0, "r", encoding="utf-8")
    lines0 = fr0.readlines()
    for line in lines0:
        line = line.strip()
        if line not in ch_sim_chars:
            ch_sim_chars += line
    print(len(ch_sim_chars))

    save_path = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/chinese_simple_with_special_chars.txt"
    fw = open(save_path, "w", encoding="utf-8")

    charslist = []
    for i in range(2, 15):
        charslist.append("chars{}".format(i))

    CHARS = ""
    CHARS += chars0
    CHARS += chars1
    CHARS += ch_sim_chars

    for ci in charslist:
        for c in eval(ci):
            if c not in CHARS:
                CHARS += c
    # print(CHARS)

    CHARS_NEW = ""
    for cc in CHARS:
        if cc not in CHARS_NEW:
            CHARS_NEW += cc

    for n in CHARS_NEW:
        fw.write("{}\n".format(n))

    fw.close()

    print(len(CHARS_NEW))










    # chars9_ = ""
    # charslist = []
    # for i in range(9):
    #     charslist.append("chars{}".format(i))
    # for i in range(10, 18):
    #     charslist.append("chars{}".format(i))
    #
    # CHARS = ""
    # for ci in charslist:
    #     for c in eval(ci):
    #         if c not in CHARS:
    #             CHARS += c
    # # print(CHARS)
    #
    # for cc in chars9:
    #     if cc not in CHARS:
    #         if cc not in chars9_:
    #             chars9_ += cc
    # print(chars9_)


    # chars910_ = ""
    # for c in chars10:
    #     if c not in chars910_:
    #         chars910_ += c
    # for c in chars9:
    #     if c not in chars910_:
    #         chars910_ += c
    # print(chars910_)

    # chars18 = "❤❥웃유♋☮✌☏☢☠✔☑♚▲♪✈✞÷↑↓◆◇⊙■□△▽¿─│♥❣♂♀☿Ⓐ✍✉☣☤✘☒♛▼♫⌘☪≈←→◈◎☉★☆⊿※¡━┃♡ღツ☼☁❅♒✎©®™Σ✪✯☭➳卐√↖↗●◐Θ◤◥︻〖〗┄┆℃℉°✿ϟ☃☂✄¢€£∞✫★½✡×↙↘○◑⊕◣◢︼【】┅┇☽☾✚〓▂▃▄▅▆▇█▉▊▋▌▍▎▏↔↕☽☾の•▸◂▴▾┈┊①②③④⑤⑥⑦⑧⑨⑩ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ㍿▓♨♛❖♓☪✙┉┋☹☺☻ ヅツッシÜϡﭢ™℠℗©®♥❤❥❣❦❧♡ 웃유ღ♋♂♀☿☼☀☁☂☄☾☽❄☃☈⊙☉℃℉❅✺ϟ☇♤♧♡♢♠♣♥♦☜☞☝✍☚☛☟✌✽✾✿❁❃❋❀⚘☑✓✔√☐☒✗✘ㄨ✕✖✖⋆✢✣✤✥❋✦✧✩✰✪✫✬✭✮✯❂✡★✱✲✳✴✵✶✷✸✹✺✻✼❄❅❆❇❈❉❊†☨✞✝☥☦☓☩☯☧☬☸✡♁✙♆。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑•¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼☩☨☦✞✛✜✝✙✠✚†‡◉○◌◍◎●◐◑◒◓◔◕◖◗❂☢⊗⊙◘◙◍⅟½⅓⅕⅙⅛⅔⅖⅚⅜¾⅗⅝⅞⅘≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟≠≡≢≣≤≥≦≧≨≩⊰⊱⋛⋚∫∬∭∮∯∰∱∲∳%℅‰‱㊣㊎㊍㊌㊋㊏㊐㊊㊚㊛㊤㊥㊦㊧㊨㊒㊞㊑㊒㊓㊔㊕㊖㊗㊘㊜㊝㊟㊠㊡㊢㊩㊪㊫㊬㊭㊮㊯㊰㊙㉿囍♔♕♖♗♘♙♚♛♜♝♞♟ℂℍℕℙℚℝℤℬℰℯℱℊℋℎℐℒℓℳℴ℘ℛℭ℮ℌℑℜℨ♪♫♩♬♭♮♯°øⒶ☮✌☪✡☭✯卐✐✎✏✑✒✍✉✁✂✃✄✆✉☎☏➟➡➢➣➤➥➦➧➨➚➘➙➛➜➝➞➸♐➲➳⏎➴➵➶➷➸➹➺➻➼➽←↑→↓↔↕↖↗↘↙↚↛↜↝↞↟↠↡↢↣↤↥↦↧↨➫➬➩➪➭➮➯➱↩↪↫↬↭↮↯↰↱↲↳↴↵↶↷↸↹↺↻↼↽↾↿⇀⇁⇂⇃⇄⇅⇆⇇⇈⇉⇊⇋⇌⇍⇎⇏⇐⇑⇒⇓⇔⇕⇖⇗⇘⇙⇚⇛⇜⇝⇞⇟⇠⇡⇢⇣⇤⇥⇦⇧⇨⇩⇪➀➁➂➃➄➅➆➇➈➉➊➋➌➍➎➏➐➑➒➓㊀㊁㊂㊃㊄㊅㊆㊇㊈㊉ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅬⅭⅮⅯⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹⅺⅻⅼⅽⅾⅿ┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬◤◥◄►▶◀◣◢▲▼◥▸◂▴▾△▽▷◁⊿▻◅▵▿▹◃❏❐❑❒▀▁▂▃▄▅▆▇▉▊▋█▌▍▎▏▐░▒▓▔▕■□▢▣▤▥▦▧▨▩▪▫▬▭▮▯㋀㋁㋂㋃㋄㋅㋆㋇㋈㋉㋊㋋㏠㏡㏢㏣㏤㏥㏦㏧㏨㏩㏪㏫㏬㏭㏮㏯㏰㏱㏲㏳㏴㏵㏶㏷㏸㏹㏺㏻㏼㏽㏾㍙㍚㍛㍜㍝㍞㍟㍠㍡㍢㍣㍤㍥㍦㍧㍨㍩㍪㍫㍬㍭㍮㍯㍰㍘☰☲☱☴☵☶☳☷☯"
    # chars18_ = ""
    # charslist = []
    # for i in range(18):
    #     charslist.append("chars{}".format(i))
    #
    # CHARS = ""
    # for ci in charslist:
    #     for c in eval(ci):
    #         if c not in CHARS:
    #             CHARS += c
    # # print(CHARS)

    # for cc in chars18:
    #     if cc not in CHARS:
    #         if cc not in chars18_:
    #             chars18_ += cc
    # print(chars18_)





    # # print(len(CH_SIM_CHARS))
    #
    # txt_path0 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/ch_sim_char.txt"
    # ch_sim_chars = ""
    # fr0 = open(txt_path0, "r", encoding="utf-8")
    # lines0 = fr0.readlines()
    # for line in lines0:
    #     line = line.strip()
    #     if line not in ch_sim_chars:
    #         ch_sim_chars += line
    # print(len(ch_sim_chars))
    #
    # txt_path1 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/dict/chinese_cht_dict.txt"
    # txt_path2 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/all_words.txt"
    # txt_path3 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/chinese_word.txt"
    # txt_path4 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/dict90.txt"
    # txt_path5 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/digit_alphabet.txt"
    # txt_path6 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/ppocr_keys_v1.txt"
    # txt_path7 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/simple_chinese_word.txt"
    # txt_path8 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/Special_CHARS.txt"
    #
    # paths = []
    # for i in range(8):
    #     paths.append("txt_path{}".format(i + 1))
    #
    # other_alpha = ""
    # other_chn = ""
    #
    # for p in paths:
    #     fr = open(eval(p), "r", encoding="utf-8")
    #     linesp = fr.readlines()
    #     for line in linesp:
    #         line = line.strip()
    #         res = isAllChinese(line)
    #         # if line not in CH_SIM_CHARS:
    #         #         other_alpha += line
    #         if res:
    #             if line not in ch_sim_chars:
    #                 other_chn += line
    #
    # # print(other_alpha)
    # # print(len(other_alpha))
    # print(other_chn)
    # print(len(other_chn))
    #
    #
    # # aa_ = ""
    # # for a in aa:
    # #     if a not in aa_:
    # #         aa_ += a
    # # print(aa_)

    # for cc in cahrs17:
    #     if cc not in ch_sim_chars:
    #         print(cc)


def main_test_ip30_99():
    # ============================= Chinese ============================
    # chars0 = ' ' + '0123456789' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    # chars0 = '0123456789' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    # chars1 = ",./;'[]\\`-=" + '<>?:"{}|~!@#$%^&*()_+'
    # chars2 = "，。、；’【】·" + "《》？：”｛｝｜～！＠＃￥％…（）" + "║〈〉［］〔〕「」︵︷︿︹︽﹁︻︶︸﹀︺︾﹂︼"
    # chars3 = "⓪①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㊀㊁㊂㊃㊄㊅㊆㊇㊈㊉㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ"
    # chars4 = "ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵"
    # chars5 = "㊎㊍㊌㊋㊏㊚㊛㊐㊊㊣㊤㊥㊦㊧㊨㊒㊫㊑㊓㊔㊕㊖㊗㊘㊜㊝㊞㊟㊠㊡㊢㊩㊪㊬㊭㊮㊯㊰"
    # chars6 = "€£Ұ₴$₰₤¥₳₲₪₵₣₱฿₡₮₭₩円₢₥₫₦złƒ₸"
    # chars7 = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
    # chars8 = "āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜü"
    # chars9 = "°℃℉㎡㏕㎜㎝㎞㏎㎎㎏¹²³"
    # chars10 = "↑↓←→↖↗↘↙"
    # chars11 = "√×№π∞÷±∽≈≅≠≤≥∪∩∈∵∴∷⊥∥∠㏒㏑‰∫∬∭∮∯∰"
    # chars12 = "卍卐㍿❤☢☯♂♀♫♪©®™"

    # chars8 = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    # chars10 = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんゔゕゖァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶヷヸヹヺ・ーヽヾヿ゠ㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"

    chars0 = '0123456789' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    chars1 = ",./;'[]\\`-=" + '<>?:"{}|~!@#$%^&*()_+'
    chars2 = "，。、；’【】·" + "《》？：”｛｝｜～！＠％…（）" + "〈〉［］〔〕「」︵︷︿︹︽﹁︻︶︸﹀︺︾﹂︼"
    chars3 = "①②③④⑤⑥⑦⑧⑨⑩㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ"
    chars4 = "€£$¥"
    chars5 = "αβγδεζηθικλμνξοπρστυφχψω"
    chars6 = "°℃℉㎡㏕㎜㎝㎞㏎㎎㎏²³"  # "°℃℉㎡㏕㎜㎝㎞㏎㎎㎏²³"
    chars7 = "↑↓←→↖↗↘↙"
    chars8 = "√×№π∞÷±≈≠≤≥∪∩∈∵∴⊥∠㏒㏑‰"
    chars9 = "❤"  # H means Heart , actually is ❤
    chars10 = "♂♀"

    txt_path0 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/ch_sim_char.txt"
    ch_sim_chars = ""
    fr0 = open(txt_path0, "r", encoding="utf-8")
    lines0 = fr0.readlines()
    for line in lines0:
        line = line.strip()
        if line not in ch_sim_chars:
            ch_sim_chars += line
    print(len(ch_sim_chars))

    save_path = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/chinese_simple_with_special_chars.txt"
    fw = open(save_path, "w", encoding="utf-8")

    charslist = []
    for i in range(2, 11):
        charslist.append("chars{}".format(i))

    CHARS = ""
    CHARS += chars0
    CHARS += chars1
    CHARS += ch_sim_chars

    for ci in charslist:
        for c in eval(ci):
            if c not in CHARS:
                CHARS += c
    # print(CHARS)

    CHARS_NEW = ""
    for cc in CHARS:
        if cc not in CHARS_NEW:
            CHARS_NEW += cc

    for n in CHARS_NEW:
        fw.write("{}\n".format(n))

    fw.close()

    print(len(CHARS_NEW))  # 6866


def main_test_ip30_100_ChineseOCR_gen_no_slash_train_txt():
    root = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash"
    dirs = sorted(os.listdir(root))
    save_label_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/no_slash_train_with_banner.txt"
    with open(save_label_path, "w", encoding="utf-8") as fw:
        for d in dirs:
            if d != "others":
                d_path = root + "/{}".format(d)
                sbdirs = sorted(os.listdir(d_path))
                for sbd in sbdirs:
                    sbd_path = d_path + "/{}".format(sbd)
                    ssbdirs = sorted(os.listdir(sbd_path))
                    for ssbd in ssbdirs:
                        ssbd_path = sbd_path + "/{}".format(ssbd)
                        file_list = sorted(os.listdir(ssbd_path))
                        for f in file_list:
                            f_abs_path = ssbd_path + "/{}".format(f)
                            f_base_name = os.path.splitext(f)[0]
                            if "=" in f:
                                first_equal_idx = f_base_name.find("=")
                                # x_path = f_base_name[:first_space_idx]
                                label = f_base_name[first_equal_idx + 1:]

                                if os.path.exists(f_abs_path):
                                    content = "{} {}\n".format(f_abs_path, label)
                                    fw.write(content)
    print("OK")


def main_test_ip30_101_ChineseOCR_gen_slash_train_txt():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/slash/BaiduOCR_Chinese_Dataset/train_images"
    label_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/slash/BaiduOCR_Chinese_Dataset/train_labels.txt"
    label_new_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/slash/BaiduOCR_Chinese_Dataset/train_labels_new.txt"

    fr = open(label_path, "r", encoding="utf-8")
    lines = fr.readlines()
    fr.close()

    fw = open(label_new_path, "w", encoding="utf-8")
    for line in lines:

        if "\\t" in line:
            print(line)
        line = line.replace("\\t", "")
        if "##" in line:
            continue

        fw.write(line)

    fw.close()


def main_test_ip30_102():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/slash/Synthetic_Chinese_String_Dataset/train_images"
    label_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/slash/Synthetic_Chinese_String_Dataset/train_labels.txt"
    save_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/slash/Synthetic_Chinese_String_Dataset/train_labels_abs_path.txt"

    fw = open(save_path, "w", encoding="utf-8")
    fr = open(label_path, "r", encoding="utf-8")
    lines = fr.readlines()
    fr.close()

    for line in lines:
        new_content = data_path + "/" + line
        fw.write(new_content)

    fw.close()


def main_test_ip30_103_ChineseOCR_gen_experiment_train_test_txt():
    root = "/home/disk/disk7/data/000.ChineseOCR/data/experiment/test_v2/horizontal"
    dirs = sorted(os.listdir(root))
    save_label_path = root + "/test.txt"
    CHARS = ""

    with open(save_label_path, "w", encoding="utf-8") as fw:
        for d in dirs:

            d_path = root + "/{}".format(d)
            sbdirs = sorted(os.listdir(d_path))
            for sbd in sbdirs:
                sbd_path = d_path + "/{}".format(sbd)
                # ssbdirs = sorted(os.listdir(sbd_path))
                # for ssbd in ssbdirs:
                #     ssbd_path = sbd_path + "/{}".format(ssbd)
                file_list = sorted(os.listdir(sbd_path))
                for f in file_list:
                    f_abs_path = sbd_path + "/{}".format(f)
                    f_base_name = os.path.splitext(f)[0]
                    if "=" in f:
                        first_equal_idx = f_base_name.find("=")
                        # x_path = f_base_name[:first_space_idx]
                        label = f_base_name[first_equal_idx + 1:]

                        for l in label:
                            if l not in CHARS:
                                CHARS += l

                        if os.path.exists(f_abs_path):
                            content = "{} {}\n".format(f_abs_path, label)
                            fw.write(content)
    print("OK")
    print(len(CHARS), CHARS)





def main_test_ip30_104():
    txt_path0 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/chinese_simple_with_special_chars.txt"
    # txt_path0 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/chinese_word.txt"
    ch_sim_chars = ""
    fr0 = open(txt_path0, "r", encoding="utf-8")
    lines0 = fr0.readlines()
    for line in lines0:
        line = line.strip()
        if line not in ch_sim_chars:
            ch_sim_chars += line
    print(len(ch_sim_chars))



    # un = ""
    # chars_ = test_chars + train_chars
    # for c in chars_:
    #     if c not in ch_sim_chars:
    #         un += c
    #
    # print(un, len(un))


def main_test_ip30_105():
    txt_path0 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/chinese_simple_with_special_chars.txt"
    ch_sim_chars = ""
    fr0 = open(txt_path0, "r", encoding="utf-8")
    lines0 = fr0.readlines()
    for line in lines0:
        line = line.strip()
        if line not in ch_sim_chars:
            ch_sim_chars += line
    print(len(ch_sim_chars))

    txt_path2 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/chinese_word.txt"
    ch_sim_chars2 = ""
    fr2 = open(txt_path2, "r", encoding="utf-8")
    lines2 = fr2.readlines()
    for line in lines2:
        line = line.strip()
        if line not in ch_sim_chars2:
            ch_sim_chars2 += line
    print(len(ch_sim_chars2))

    un = ""
    for c in ch_sim_chars2:
        if c not in ch_sim_chars:
            un += c

    print(un, len(un))


def isAllChinese(string):
    pattern = '[\u4e00-\u9fa5]+'
    if re.match(pattern, string) and len(string) == len(set(string)):
        return True
    else:
        return False


def main_test_ip30_106():
    txt_path0 = "/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/chinese_simple_with_special_chars.txt"
    ch_sim_chars = ""
    fr0 = open(txt_path0, "r", encoding="utf-8")
    lines0 = fr0.readlines()
    for line in lines0:
        line = line.strip()
        if line not in ch_sim_chars:
            ch_sim_chars += line
    print(len(ch_sim_chars))

    root = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash"
    dirs = sorted(os.listdir(root))
    CHARS = ""

    for d in dirs:
        if d != "others":
            d_path = root + "/{}".format(d)
            if os.path.isdir(d_path):
                sbdirs = sorted(os.listdir(d_path))
                for sbd in sbdirs:
                    sbd_path = d_path + "/{}".format(sbd)
                    ssbdirs = sorted(os.listdir(sbd_path))
                    for ssbd in ssbdirs:
                        ssbd_path = sbd_path + "/{}".format(ssbd)
                        file_list = sorted(os.listdir(ssbd_path))
                        for f in file_list:
                            f_abs_path = ssbd_path + "/{}".format(f)
                            f_base_name = os.path.splitext(f)[0]
                            if "=" in f:
                                first_equal_idx = f_base_name.find("=")
                                # x_path = f_base_name[:first_space_idx]
                                label = f_base_name[first_equal_idx + 1:]

                                for l in label:
                                    if l not in CHARS:
                                        CHARS += l

    print("OK")
    print(len(CHARS), CHARS)

    un = ""
    for c in CHARS:
        if c not in ch_sim_chars and isAllChinese(c):
            un += c

    print(un, len(un))


def main_test_ip30_107():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash_train_merged_20240422.txt"
    save_path = data_path.replace(".txt", "_random_selected_25_percent.txt")

    fr = open(data_path, "r", encoding="utf-8")
    lines = fr.readlines()
    fr.close()

    fw = open(save_path, "w", encoding="utf-8")
    lines_20 = random.sample(lines, int(len(lines) * 0.25))
    for line in lines_20:
        # line = line.replace("\t", " ")
        fw.write(line)
    fw.close()


def main_test_ip30_108():
    # data_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/15_cls/sygy_v4/llb/sygy_v4_llb_m3_fake1"
    data_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/v1_v2/v1_15_cls_v6/bg_03"
    file_list = get_file_list(data_path, abspath=False)

    for f in tqdm(file_list):
        fname, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
        f_abs_path = data_path + "/{}".format(f)
        equal_num = fname.count("=")
        if equal_num > 1:
            print(f_abs_path)

        # fname_new = fname + "m³"  # m³ MPa
        fname_new = fname[:-1]
        f_dst_path = data_path + "/{}{}".format(fname_new, suffix)
        os.rename(f_abs_path, f_dst_path)


def main_test_ip30_109():
    data_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/15_cls/sygy_v4/llb/sygy_v4_llb_m3_fake1"
    file_list = get_file_list(data_path, abspath=False)
    save_path = make_save_path(data_path)

    for f in tqdm(file_list):
        fname, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
        f_abs_path = data_path + "/{}".format(f)
        f_dst_path = save_path + "/{}".format(f)

        img = cv2.imread(f_abs_path)
        imgsz = img.shape[:2]

        cropped = img[:, int(round(imgsz[1] * 0.81856)):]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        v = np.mean(gray)
        print(v)

        # cv2.imwrite("/home/disk/disk7/data/010.Digital_Rec/crnn/train/15_cls/sygy_v4/llb/sygy_v4_llb_m3_fake1_cropped/{}-meanv={}{}".format(fname, int(v), suffix), cropped)
        # cv2.imwrite("/home/disk/disk7/data/010.Digital_Rec/crnn/train/15_cls/sygy_v4/llb/sygy_v4_llb_m3_fake1_cropped/{}_gray-meanv={}{}".format(fname, int(v), suffix), gray)
        #
        # cv2.imwrite("/home/disk/disk7/data/010.Digital_Rec/crnn/train/15_cls/sygy_v4/llb/sygy_v4_llb_m3_fake1_cropped2/meanv={}-{}{}".format(int(v), fname, suffix), cropped)
        # cv2.imwrite("/home/disk/disk7/data/010.Digital_Rec/crnn/train/15_cls/sygy_v4/llb/sygy_v4_llb_m3_fake1_cropped2/meanv={}-{}_gray{}".format(int(v), fname, suffix), gray)

        if int(v) > 190:
            shutil.move(f_abs_path, f_dst_path)


def main_test_ip30_110():
    data_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/v2/v2_add_20240425/bg_10_no_dot/fake"
    file_list = get_file_list(data_path)

    # loc1 = 96 / 578
    # loc2 = 193 / 578

    dd = ["top", "bottom", "top_and_bottom"]

    for d in dd:
        save_path = make_save_path(data_path, "crop_border_{}".format(d))
        for f in tqdm(file_list):
            fname, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
            f_abs_path = data_path + "/{}".format(f)
            equal_num = fname.count("=")
            if equal_num > 1: print(f_abs_path)
            label = fname.split("=")[1]

            img = cv2.imread(f_abs_path)
            imgsz = img.shape[:2]

            # # bg_01
            # crop_loc = 0.01 * np.random.randint(15, 36)
            # if "1" in label or "7" in label or "6" in label:
            #     crop_loc = 0.01 * np.random.randint(13, 17)

            # # bg_02
            # crop_loc = 0.01 * np.random.randint(16, 37)
            # if "1" in label or "7" in label or "6" in label:
            #     crop_loc = 0.01 * np.random.randint(14, 18)

            # # bg_04
            # crop_loc = 0.01 * np.random.randint(16, 37)
            # if "1" in label or "7" in label or "6" in label:
            #     crop_loc = 0.01 * np.random.randint(14, 18)

            # # bg_05
            # crop_loc = 0.01 * np.random.randint(27, 41)
            # if "1" in label or "7" in label or "6" in label:
            #     crop_loc = 0.01 * np.random.randint(23, 28)

            # # bg_06
            # crop_loc = 0.01 * np.random.randint(18, 38)
            # if "1" in label or "7" in label or "6" in label:
            #     crop_loc = 0.01 * np.random.randint(13, 18)

            # # bg_07
            # crop_loc = 0.01 * np.random.randint(15, 33)
            # if "1" in label or "7" in label or "6" in label:
            #     crop_loc = 0.01 * np.random.randint(12, 17)

            # # bg_08
            # crop_loc = 0.01 * np.random.randint(31, 43)
            # if "1" in label or "7" in label or "6" in label:
            #     crop_loc = 0.01 * np.random.randint(25, 32)

            # # bg_09
            # crop_loc = 0.01 * np.random.randint(22, 35)
            # if "1" in label or "7" in label or "6" in label:
            #     crop_loc = 0.01 * np.random.randint(17, 23)

            # # bg_10, bg_11
            # crop_loc = 0.01 * np.random.randint(18, 35)

            # # ===============================================
            # # # bg_01
            # if d == "top":
            #     crop_loc = 0.01 * np.random.randint(15, 36)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc = 0.01 * np.random.randint(13, 17)
            #
            #     cropped = img[int(round(crop_loc * imgsz[0])):, :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # elif d == "bottom":
            #     crop_loc = 0.01 * np.random.randint(63, 85)
            #     # if "1" in label or "7" in label or "6" in label:
            #     #     crop_loc = 0.01 * np.random.randint(13, 17)
            #
            #     cropped = img[:int(round(crop_loc * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # else:
            #     crop_loc_top = 0.01 * np.random.randint(15, 36)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc_top = 0.01 * np.random.randint(13, 17)
            #
            #     crop_loc_bottom = 0.01 * np.random.randint(63, 85)
            #
            #     cropped = img[int(round(crop_loc_top * imgsz[0])):int(round(crop_loc_bottom * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)

            # # ===============================================
            # # # bg_02
            # if d == "top":
            #     crop_loc = 0.01 * np.random.randint(16, 37)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc = 0.01 * np.random.randint(14, 18)
            #
            #     cropped = img[int(round(crop_loc * imgsz[0])):, :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # elif d == "bottom":
            #     crop_loc = 0.01 * np.random.randint(66, 85)
            #     # if "1" in label or "7" in label or "6" in label:
            #     #     crop_loc = 0.01 * np.random.randint(13, 17)
            #
            #     cropped = img[:int(round(crop_loc * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # else:
            #     crop_loc_top = 0.01 * np.random.randint(16, 37)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc_top = 0.01 * np.random.randint(14, 18)
            #
            #     crop_loc_bottom = 0.01 * np.random.randint(66, 85)
            #
            #     cropped = img[int(round(crop_loc_top * imgsz[0])):int(round(crop_loc_bottom * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)

            # # ===============================================
            # # # bg_04
            # if d == "top":
            #     crop_loc = 0.01 * np.random.randint(16, 37)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc = 0.01 * np.random.randint(14, 18)
            #
            #     cropped = img[int(round(crop_loc * imgsz[0])):, :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # elif d == "bottom":
            #     crop_loc = 0.01 * np.random.randint(62, 80)
            #     # if "1" in label or "7" in label or "6" in label:
            #     #     crop_loc = 0.01 * np.random.randint(13, 17)
            #
            #     cropped = img[:int(round(crop_loc * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # else:
            #     crop_loc_top = 0.01 * np.random.randint(16, 37)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc_top = 0.01 * np.random.randint(14, 18)
            #
            #     crop_loc_bottom = 0.01 * np.random.randint(62, 80)
            #
            #     cropped = img[int(round(crop_loc_top * imgsz[0])):int(round(crop_loc_bottom * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)

            # # ===============================================
            # # # bg_05
            # if d == "top":
            #     crop_loc = 0.01 * np.random.randint(27, 41)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc = 0.01 * np.random.randint(23, 28)
            #
            #     cropped = img[int(round(crop_loc * imgsz[0])):, :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # elif d == "bottom":
            #     crop_loc = 0.01 * np.random.randint(64, 80)
            #     # if "1" in label or "7" in label or "6" in label:
            #     #     crop_loc = 0.01 * np.random.randint(13, 17)
            #
            #     cropped = img[:int(round(crop_loc * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # else:
            #     crop_loc_top = 0.01 * np.random.randint(27, 41)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc_top = 0.01 * np.random.randint(23, 28)
            #
            #     crop_loc_bottom = 0.01 * np.random.randint(64, 80)
            #
            #     cropped = img[int(round(crop_loc_top * imgsz[0])):int(round(crop_loc_bottom * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)

            # # ===============================================
            # # # bg_06
            # if d == "top":
            #     crop_loc = 0.01 * np.random.randint(18, 38)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc = 0.01 * np.random.randint(13, 18)
            #
            #     cropped = img[int(round(crop_loc * imgsz[0])):, :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # elif d == "bottom":
            #     crop_loc = 0.01 * np.random.randint(78, 100)
            #     # if "1" in label or "7" in label or "6" in label:
            #     #     crop_loc = 0.01 * np.random.randint(13, 17)
            #
            #     cropped = img[:int(round(crop_loc * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # else:
            #     crop_loc_top = 0.01 * np.random.randint(18, 38)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc_top = 0.01 * np.random.randint(13, 18)
            #
            #     crop_loc_bottom = 0.01 * np.random.randint(78, 100)
            #
            #     cropped = img[int(round(crop_loc_top * imgsz[0])):int(round(crop_loc_bottom * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)

            # # ===============================================
            # # # bg_07
            # if d == "top":
            #     crop_loc = 0.01 * np.random.randint(15, 33)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc = 0.01 * np.random.randint(12, 17)
            #
            #     cropped = img[int(round(crop_loc * imgsz[0])):, :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # elif d == "bottom":
            #     crop_loc = 0.01 * np.random.randint(60, 80)
            #     # if "1" in label or "7" in label or "6" in label:
            #     #     crop_loc = 0.01 * np.random.randint(13, 17)
            #
            #     cropped = img[:int(round(crop_loc * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # else:
            #     crop_loc_top = 0.01 * np.random.randint(15, 33)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc_top = 0.01 * np.random.randint(12, 17)
            #
            #     crop_loc_bottom = 0.01 * np.random.randint(60, 80)
            #
            #     cropped = img[int(round(crop_loc_top * imgsz[0])):int(round(crop_loc_bottom * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)

            # # ===============================================
            # # # bg_08
            # if d == "top":
            #     crop_loc = 0.01 * np.random.randint(31, 43)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc = 0.01 * np.random.randint(25, 32)
            #
            #     cropped = img[int(round(crop_loc * imgsz[0])):, :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # elif d == "bottom":
            #     crop_loc = 0.01 * np.random.randint(62, 76)
            #     # if "1" in label or "7" in label or "6" in label:
            #     #     crop_loc = 0.01 * np.random.randint(13, 17)
            #
            #     cropped = img[:int(round(crop_loc * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # else:
            #     crop_loc_top = 0.01 * np.random.randint(31, 43)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc_top = 0.01 * np.random.randint(25, 32)
            #
            #     crop_loc_bottom = 0.01 * np.random.randint(62, 76)
            #
            #     cropped = img[int(round(crop_loc_top * imgsz[0])):int(round(crop_loc_bottom * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)

            # # ===============================================
            # # # bg_09
            # if d == "top":
            #     crop_loc = 0.01 * np.random.randint(22, 35)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc = 0.01 * np.random.randint(17, 23)
            #
            #     cropped = img[int(round(crop_loc * imgsz[0])):, :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # elif d == "bottom":
            #     crop_loc = 0.01 * np.random.randint(53, 68)
            #     # if "1" in label or "7" in label or "6" in label:
            #     #     crop_loc = 0.01 * np.random.randint(13, 17)
            #
            #     cropped = img[:int(round(crop_loc * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)
            # else:
            #     crop_loc_top = 0.01 * np.random.randint(22, 35)
            #     if "1" in label or "7" in label or "6" in label:
            #         crop_loc_top = 0.01 * np.random.randint(17, 23)
            #
            #     crop_loc_bottom = 0.01 * np.random.randint(53, 68)
            #
            #     cropped = img[int(round(crop_loc_top * imgsz[0])):int(round(crop_loc_bottom * imgsz[0])), :]
            #     f_dst_path = save_path + "/{}{}".format(fname, suffix)
            #     cv2.imwrite(f_dst_path, cropped)

            # ===============================================
            # # bg_10, bg_11
            if d == "top":
                crop_loc = 0.01 * np.random.randint(18, 35)

                cropped = img[int(round(crop_loc * imgsz[0])):, :]
                f_dst_path = save_path + "/{}{}".format(fname, suffix)
                cv2.imwrite(f_dst_path, cropped)
            elif d == "bottom":
                crop_loc = 0.01 * np.random.randint(60, 85)

                cropped = img[:int(round(crop_loc * imgsz[0])), :]
                f_dst_path = save_path + "/{}{}".format(fname, suffix)
                cv2.imwrite(f_dst_path, cropped)
            else:
                crop_loc_top = 0.01 * np.random.randint(18, 35)
                crop_loc_bottom = 0.01 * np.random.randint(60, 85)

                cropped = img[int(round(crop_loc_top * imgsz[0])):int(round(crop_loc_bottom * imgsz[0])), :]
                f_dst_path = save_path + "/{}{}".format(fname, suffix)
                cv2.imwrite(f_dst_path, cropped)







def main_test_ip30_111():
    data_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/v1_v2/v1_15_cls_v6/20231026"
    save_path = make_save_path(data_path, "selected_1")

    file_list = get_file_list(data_path)
    for f in tqdm(file_list):
        fname, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
        f_abs_path = data_path + "/{}".format(f)

        label = fname.split("=")[1]
        if "1" in label:
            f_dst_path = save_path + "/{}".format(f)
            shutil.move(f_abs_path, f_dst_path)


def main_test_ip30_112():
    # data_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/v1_v2"
    # save_path = make_save_path(data_path, "many_equals")
    #
    # dirs = sorted(os.listdir(data_path))
    # for d in dirs:
    #     d_path = data_path + "/{}".format(d)
    #     ddirs = sorted(os.listdir(d_path))
    #     for dd in ddirs:
    #         dd_path = d_path + "/{}".format(dd)
    #         dddirs = sorted(os.listdir(dd_path))
    #         for ddd in dddirs:
    #             ddd_path = dd_path + "/{}".format(ddd)
    #             if os.path.isfile(ddd_path): continue
    #             file_list = get_file_list(ddd_path)
    #             for f in tqdm(file_list):
    #                 if f.count("=") > 1:
    #                     f_src_path = ddd_path + "/{}".format(f)
    #                     save_path_ = save_path + "/{}/{}/{}".format(d, dd, ddd)
    #                     os.makedirs(save_path_, exist_ok=True)
    #                     f_dst_path = save_path_ + "/{}".format(f)
    #                     shutil.move(f_src_path, f_dst_path)

    # data_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/v2/v1_15_cls_20240117_add"
    # save_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/v2/v1_15_cls_20240117_add.txt"
    # fw = open(save_path, "w", encoding="utf-8")

    # dirs = sorted(os.listdir(data_path))
    # for d in dirs:
    #     d_path = data_path + "/{}".format(d)
    #     ddirs = sorted(os.listdir(d_path))
    #     for dd in ddirs:
    #         dd_path = d_path + "/{}".format(dd)
    #         dddirs = sorted(os.listdir(dd_path))
    #         for ddd in dddirs:
    #             ddd_path = dd_path + "/{}".format(ddd)
    #             if os.path.isfile(ddd_path): continue
    #             file_list = get_file_list(ddd_path)
    #             for f in tqdm(file_list):
                    # if f.count("=") > 1:
                    #     f_src_path = ddd_path + "/{}".format(f)
                    #     save_path_ = save_path + "/{}/{}/{}".format(d, dd, ddd)
                    #     os.makedirs(save_path_, exist_ok=True)
                    #     f_dst_path = save_path_ + "/{}".format(f)
                    #     shutil.move(f_src_path, f_dst_path)
                    # f_src_path = ddd_path + "/{}".format(f)
                    # fname, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
                    # label = fname.split("=")[1]
                    # equal_first_idx = fname.find("=")
                    # equal_last_idx = len(fname) - (fname[::-1].find("=") + 1)
                    # fname_new = fname.replace(fname[equal_first_idx:equal_last_idx], "")
                    # f_dst_path = ddd_path + "/{}{}".format(fname_new, suffix)
                    # os.rename(f_src_path, f_dst_path)

    data_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/v2/v1_15_cls_v6"
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
                fname, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
                label = fname.split("=")[1]
                assert "=" in fname, "= not in {}".format(fname)
                assert fname.count("=") == 1, '{}.count("=") != 1'.format(fname)
                txt_content = "{} {}\n".format(f_src_path, label)
                fw.write(txt_content)
    fw.close()

    # data_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/v2/v1_15_cls_v6"
    # save_path = make_save_path(data_path, "unexpected_data")
    #
    # dirs = sorted(os.listdir(data_path))
    # for d in dirs:
    #     d_path = data_path + "/{}".format(d)
    #     ddirs = sorted(os.listdir(d_path))
    #     for dd in ddirs:
    #         dd_path = d_path + "/{}".format(dd)
    #         if os.path.isfile(dd_path): continue
    #         file_list = get_file_list(dd_path)
    #         for f in tqdm(file_list):
    #             f_src_path = dd_path + "/{}".format(f)
    #             fname, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
    #             label = fname.split("=")[1]
    #             if label[0] == "." or label[-1] == ".":
    #                 save_path_ = save_path + "/{}/{}".format(d, dd)
    #                 os.makedirs(save_path_, exist_ok=True)
    #                 f_dst_path = save_path_ + "/{}".format(f)
    #                 shutil.move(f_src_path, f_dst_path)


def main_test_ip30_113():
    data_path = "/home/disk/disk7/data/010.Digital_Rec/crnn/train/v2/v1_15_cls_20240117_add/0123456789_dot_new"
    save_path = make_save_path(data_path, "selected")
    # save_path_abnormal = make_save_path(data_path, "selected_abnormal")

    dirs = sorted(os.listdir(data_path))
    for d in dirs:
        d_path = data_path + "/{}".format(d)
        print(d_path)
        if os.path.isfile(d_path): continue
        file_list = get_file_list(d_path)
        for f in tqdm(file_list):
            f_src_path = d_path + "/{}".format(f)
            try:
                fname, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
                if "=" in fname:
                    label = fname.split("=")[1]
                    img = cv2.imread(f_src_path)
                    imgsz = img.shape[:2]
                    # print(len(label))

                    if imgsz[1] < 96:
                        # if len(label) > int(imgsz[1] / 48 * 3):
                        if len(label) > 4:
                            save_path_ = save_path + "/{}".format(d)
                            os.makedirs(save_path_, exist_ok=True)
                            f_dst_path = save_path_ + "/{}".format(f)
                            shutil.move(f_src_path, f_dst_path)
                    if imgsz[1] >= 96 and imgsz[1] < 128:
                        # if len(label) > int(imgsz[1] / 48 * 3):
                        if len(label) > 8:
                            save_path_ = save_path + "/{}".format(d)
                            os.makedirs(save_path_, exist_ok=True)
                            f_dst_path = save_path_ + "/{}".format(f)
                            shutil.move(f_src_path, f_dst_path)
                    if imgsz[1] >= 128 and imgsz[1] < 256:
                        # if len(label) > int(imgsz[1] / 48 * 3):
                        if len(label) > 15:
                            save_path_ = save_path + "/{}".format(d)
                            os.makedirs(save_path_, exist_ok=True)
                            f_dst_path = save_path_ + "/{}".format(f)
                            shutil.move(f_src_path, f_dst_path)
                    if imgsz[1] >= 256:
                        # if len(label) > int(imgsz[1] / 48 * 3):
                        if len(label) > 20:
                            save_path_ = save_path + "/{}".format(d)
                            os.makedirs(save_path_, exist_ok=True)
                            f_dst_path = save_path_ + "/{}".format(f)
                            shutil.move(f_src_path, f_dst_path)
            except Exception as Error:
                print("{} {}".format(Error, Error.__traceback__.tb_lineno))
                print(f_src_path)
                # f_dst_path = save_path_abnormal + "/{}".format(f)
                # shutil.move(f_src_path, f_dst_path)

def isAllDigits(string):
    pattern = r'^\d+$'
    if re.match(pattern, string):
        return True
    else:
        return False


def main_test_ip30_114():
    data_path = "/home/disk/disk7/data/000.ChineseOCR/data/train/merged.txt"
    save_path = data_path.replace(".txt", "_new.txt")

    fr = open(data_path, "r", encoding="utf-8")
    lines = fr.readlines()
    fr.close()

    fw = open(save_path, "w", encoding="utf-8")

    for line in lines:
        try:
            if line.count("=") > 1: continue

            line_ = line.split(" ")
            fname, label = line_[0], line_[1].strip()
            if label == "": continue

            if isAllDigits(label[:-1]) and label[-1] == ".":
                print('isAllDigits(label[:-1]) and label[-1] == ".": {}'.format(line))
                continue

            if isAllDigits(label[1:]) and label[1] == ".":
                print('isAllDigits(label[1:]) and label[1] == ".": {}'.format(line))
                continue

            # if label[0] == "." or label[0] == "³":
            #     print(line)
            #     continue

            if label[0] == "³":
                print('label[0] == "³": {}'.format(line))
                continue

            else:
                fw.write(line)
        except Exception as Error:
            print(Error)
            print("--------- line: {}".format(line))
    fw.close()


def main_test_ip31_115():
    linear = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 26),
    )

    for name, value in linear.named_parameters():
        print(name, value)







if __name__ == '__main__':
    # main_test_10026_82()
    # main_test_10026_83()
    # main_test_ip30_94()
    # main_test_ip30_95()
    # main_test_ip30_96()
    # main_test_ip30_97()
    # main_test_ip30_98()
    # main_test_ip30_99()
    # main_test_ip30_100_ChineseOCR_gen_no_slash_train_txt()
    # main_test_ip30_101_ChineseOCR_gen_slash_train_txt()
    # main_test_ip30_102()
    # main_test_ip30_103_ChineseOCR_gen_experiment_train_test_txt()
    # main_test_ip30_104()
    # main_test_ip30_105()
    # main_test_ip30_106()
    # main_test_ip30_107()
    # main_test_ip30_108()
    # main_test_ip30_109()
    # main_test_ip30_110()
    # main_test_ip30_111()
    # main_test_ip30_112()
    # main_test_ip30_113()
    # main_test_ip30_114()
    # main_test_ip31_115()


    # # alpha = read_ocr_lables(lbl_path="/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/chinese_simple_with_special_chars.txt")
    # alpha = ' ' + '0123456789' + '.:/\\-' + 'AbC' + '℃' + 'MPa' + '㎡m³'
    # # print(len(alpha))
    # # # ocr_data_gen_train_txt(data_path="/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/train_v1", LABEL=alpha)
    # ocr_data_gen_train_txt_v2(data_path="/home/disk/disk7/data/010.Digital_Rec/crnn/train/v2/v2_add_20240425", LABEL=alpha)
    # check_ocr_label(data_path="/home/disk/disk7/data/000.ChineseOCR/data/train/merged.txt", label=alpha)
    # random_select_files_according_txt(data_path="/home/disk/disk7/data/010.Digital_Rec/crnn/train/v2/v1_15_cls_20240117_add.txt", select_percent=0.25)

    # =====================================================================================================================
    # ==================================================== Simple Test ====================================================
    # main()
    # main_test()
    # main_test2()
    # main_test3()
    # main_test4()
    # main_test5()
    # main_test6()
    # wt_test()
    # hog_test()
    # main_test7()
    # main_test8()
    # main_test9()
    # dehaze_test()
    # saliency_map_ft_test()
    # main_test10()

    # b1 = [0, 0, 10, 10]
    # b2 = [2, 2, 12, 12]
    # iou = cal_iou(b1, b2)
    # print(iou)

    # image_process_fire_smoke_experiment()
    # image_process_fire_smoke_experiment2()

    # main_test11()
    # pass

    # rename_test()
    # convert_Stanford_Dogs_Dataset_annotations_to_yolo_format(data_path="/home/zengyifan/wujiahu/data/Open_Dataset/Stanford Dogs Dataset")
    # convert_WiderPerson_Dataset_annotations_to_yolo_format(data_path="/home/zengyifan/wujiahu/data/Open_Dataset/WiderPerson/yolo_format")
    # convert_TinyPerson_Dataset_annotations_to_yolo_format(data_path="/home/zengyifan/wujiahu/data/Open_Dataset/TinyPerson")
    # convert_AI_TOD_Dataset_to_yolo_format(data_path="/home/zengyifan/wujiahu/data/Open_Dataset/AI-TOD")

    # check_yolo_dataset_classes(data_path="/home/zengyifan/wujiahu/data/Open_Dataset/WiderPerson/yolo_format_random_selected_500")
    # crop_one_image(img_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/seg_crop/for_seamless_clone/20230530/smoke/006_20230530_0001501.jpg", crop_area=[0, 1080, 0, 1920 // 3 * 2 - 20])
    # create_pure_images(save_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/Others/create_image_test", max_pixel_value=10, save_num=50000 * 2, p=0.7)
    # classify_images_via_bgr_values(img_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/cls/1.1")
    # main_test13()
    # main_test15()
    # rotate_img_any_angle(img_path="/home/zengyifan/wujiahu/data/007.Lock_Det/others/Robot_Test/Images/20230609_merged")
    # random_erasing_aug_cls_data(data_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/cls/cls_train/cls_v8.9_add_20230613_random_erasing/1_random_selected_100000")
    # random_paste_four_corner_aug_cls_data(positive_img_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/cls/cls_train/cls_v8.9_add_20230615/random_erase/1_random_selected_100000", negative_img_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/cls/cls_train/cls_v8.8_add_20230607/0")
    # split_dir(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/CRNN_OpenDataset/Syn90k/Syn90k_merged", split_n=8)

    # res = get_file_type(file_name="/home/zengyifan/anaconda3/envs/pth181/bin/lzdiff", max_len=16)
    # print(res)  # 23212F686F6D652F6C69757A68656E78
    # change_Linux_conda_envs_bin_special_files_content(conda_envs_path="/home/zengyifan/anaconda3/envs/pth212_cpu/bin")
    # dict_save_to_file(data_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/cls/cls_train/train_base_v8.8/1", flag="json")
    # dict_data = load_saved_dict_file(file_path="10010_list_dict.json")
    # compare_two_dicts(file_path1="", file_path2="")

    # gen_coco_unlabel_json(data_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/train/smoke_fire/SSOD/train/not_label_34349/images")
    # ssim_move_images(base_img_path="/home/zengyifan/wujiahu/data/004.Knife_Det/others/Robot_Test/Videos/C_Plus_Plus_det_output/20230711_video_frames_merged/crop_images/1/1.1/20230710180728-c1_0010099_0_1.1.jpg",
    #                  imgs_path="/home/zengyifan/wujiahu/data/004.Knife_Det/others/Robot_Test/Videos/C_Plus_Plus_det_output/20230711_video_frames_merged/crop_images/New Folder/0", imgsz=(32, 32), ssim_thr=0.325)

    # gen_yolo_others_label(data_path="/home/zengyifan/wujiahu/data/000.HK/dst/test", others_class=2, fixnum=True, others_num=1, hw_thr=64, wait_time=2)

    # vis_coco_pose_data_test()
    # gen_dbnet_torch_train_test_txt(data_path="/home/disk/disk7/code/OCR/DBNet_PyTorch/datasets/digital_meter_DBNet", data_type="")
    # find_red_bbx_v2(img_path="/home/zengyifan/wujiahu/data/000.HK/dst/312", cls=1)

    # main_test16()
    # main_test17()
    # main_test18()
    # main_test19()

    # main_010_aug_test()

    # for i in range(25000):
    #     main_010_aug_test(save_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/llj_0-9/AUG/aug_20230918")

    # main_010_aug_test_AbC(save_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/AbC/test_res_20230904")
    # for i in range(10000):
    #     main_010_aug_test_AbC(save_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/AbC/20230905/20230905_1")

    # for i in range(25000):
    #     main_010_aug_test_AbC_v2(save_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/gen_fake/gen_fake/fake_out/with_space/3")
    #
    # bg_img_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/qianyinibianyaqi/qianyinbianyaqi_bg/b1.jpg"
    # fg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/qianyinibianyaqi/qianyinbianyaqi66"
    # save_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/changchundianli_and_shiyougongye_data/changchundianlixunjian/train/v1_20230914/orig/3"
    # for i in range(100):
    #     main_test34(bg_img_path=bg_img_path, fg_path=fg_path, save_path=save_path, dstsz=(28, 45), aug_flag=True, m_ratio=0.025, max_num=9)

    # bg_img_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/qianyinibianyaqi/qianyinbianyaqi_bg/b2.jpg"
    # fg_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/qianyinibianyaqi/qianyinbianyaqi66"
    # save_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/changchundianli_and_shiyougongye_data/changchundianlixunjian/train/v1_20230914/orig/3"
    # for i in range(25000):
    #     main_test34(bg_img_path=bg_img_path, fg_path=fg_path, save_path=save_path, dstsz=(30, 77), aug_flag=True, m_ratio=0.025, max_num=8)

    # vstack_two_images(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/llj_0-9/llj_0-9_new")
    # cut_images(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/llj_0-9/llj_0-9_new_THR_INV")
    # stack_images(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/llj_0-9/0-9_output_ud")
    # crop_images(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/fake_number/old/fake_number9")
    # main_rename_test(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/fake_number/fake_number14_m3_cropped", add_str="m3")

    # main_test20()
    # main_test21()
    # main_test22()
    # main_test23()
    # main_test24()
    # for i in range(100):
    #     main_test24()
    # main_test25()
    # main_test26()
    # main_test27()
    # main_test28()
    # main_test29()
    # main_test30(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/train/crnn/v4+")

    # convert_ICDAR_to_custom_format(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/CRNN_OpenDataset/SVT")
    # main_test31(use_glob=True, n_subdir=2)
    # main_test33()
    # main_test35()
    # main_test36()
    # main_test37(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/changchundianli_and_shiyougongye_data/shiyougongyexunjian/train/v3/fake_number9_cropped")
    # main_test38()

    # do_aug(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/ccdl_sygy_data/shiyougongyexunjian/train/v4/llb_m3_fake1")
    # do_aug_multithreading(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/ccdl_sygy_data/shiyougongyexunjian/train/v4/llb_m3_fake1", split_n=8)

    # according_yolov8_pose_gen_head_bbx(data_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/4_cls_20230920/v3_4_cls_20230921/images", cls=2)

    # main_test41()
    # sliding_window_crop_test()
    # sliding_window_crop_test2()

    # crop_red_bbx_area(data_path="/home/zengyifan/wujiahu/data/004.Knife_Det/others/Robot_Test/Images/knife", expand_p=10)

    # main_test42()
    # main_test43()

    # file_path1 = "/home/zengyifan/wujiahu/GoroboAIReason/models/cigar_detect/cigar_det_yolov5s_cspnet_768_448_v2.onnx"
    # file_path2 = "/home/zengyifan/wujiahu/data/003.Cigar_Det/weights/v2_20230417/det/cigar_det_yolov5s_cspnet_768_448_v2.onnx"
    # file_path3 = "/home/zengyifan/wujiahu/data/003.Cigar_Det/weights/v2_20230419/det/cigar_det_yolov5s_cspnet_768_448_v2.onnx"
    # md5_value1 = calculate_md5(file_path1)
    # md5_value2 = calculate_md5(file_path2)
    # md5_value3 = calculate_md5(file_path3)
    # print("MD51: " + md5_value1)
    # print("MD52: " + md5_value2)
    # print("MD53: " + md5_value3)

    # main_test44()
    # main_test45()
    # main_test46()
    # main_test47()

    # rm_iou_larger_than_zero(data_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/20231206/AUG_20231208_cigar_cup")
    # makeBorderAndChangeYoloBbx(data_path="/home/zengyifan/wujiahu/data/004.Knife_Det/train/v2/NewFolder", n=3)
    # moveYoloLabelNumGreaterThanN(data_path="/home/zengyifan/wujiahu/data/004.Knife_Det/train/v2", N=2)
    # classify_color_and_gray_images(data_path="/home/zengyifan/wujiahu/data/000.Robot_BackFlow_Data/20231117/SmokeDetect/image/test")

    # img_path = "/home/zengyifan/wujiahu/data/000.Bg/007_bg_3885/images/007_bg_20230609_0000007.jpg"
    # img = cv2.imread(img_path)
    # dst = makeSunLightEffect(img, r=(50, 200), light_strength=300)
    # cv2.imwrite("/home/zengyifan/wujiahu/data/000.Bg/007_bg_3885/makeSunLightEffect_dst3.jpg", dst)

    # main_test48()
    # main_test49()
    # main_test50()
    # main_test51()
    # main_test52()
    # main_test53()
    # main_test54()
    # main_test55()
    # main_test56()
    # main_test57()
    # main_test58()
    # main_test59(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/dbnet_data/20240220")
    # main_test60(base_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/dbnet_data/20240220", dir1_name="gts", dir2_name="jsons", labelbee_json_label=True, move_or_delete="move", dir="dir1")

    # main_test61()

    # convertBaiduChineseOCRDatasetToCustomDatasetFormat(data_path="/home/disk/disk7/10021_bk/data/000.OCR/CRNN/data/train_Chinese/BaiduOCR_Chinese_Dataset")
    # main_test_10026_53()
    # main_test_10026_54()
    # main_test_10026_55()
    # main_test_10026_56()
    # main_test_10026_57()
    # main_test_10026_58()
    # main_test_10026_59()
    # main_test_10026_60()
    # main_test_10026_61()
    # main_test_10026_62()
    # main_test_10026_63()
    # main_test_10026_64()
    # main_test_10026_65()
    # main_test_10026_66()

    # main_test_10026_67_001_ICDAR2019_LSVT()
    # main_test_10026_68_002_ICDAR2017_RCTW_17()
    # main_test_10026_68_005_ICDAR2019_ArT()
    # main_test_10026_68_004_Chinese_Document_Text_Recognition()

    # main_test_10026_69()
    # main_test_10026_70()
    # main_test_10026_71()
    # main_test_10026_72()

    # main_test_10026_73()
    # main_test_10026_74()
    # main_test_10026_75()
    # main_test_10026_76()
    # main_test_10026_77()
    # main_test_10026_78()
    # main_test_10026_79()
    # main_test_10026_80()
    # main_test_10026_81()
    # main_test_10026_82()
    # main_test_10026_83()
    # main_test_10026_84()
    # main_test_10026_85()
    # main_test_10026_86()
    # main_test_10026_87()
    # main_test_10026_88()
    # main_test_10026_89()
    # main_test_10026_90()
    # main_test_10026_91()
    # main_test_10026_92()

    # split_dir_by_file_suffix(data_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/handpose/data/handpose_datasets_v2")

    # -------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------
    # img_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/test/dbnet/Alignment_test/010_0_dbnet_rename_20240220_0000000.jpg"
    # cv2img = cv2.imread(img_path)
    # h, w = cv2img.shape[:2]
    #
    # # cropped = cv2img[523:623, 882:1075]
    # # cv2.imwrite("{}".format(img_path.replace(".jpeg", "_cropped.jpg")), cropped)
    #
    # # -----------------------------------------
    #
    # cv2.imshow("cv2img", cv2img)
    # cv2.setMouseCallback("cv2img", click_event)
    # cv2.waitKey(0)
    # # -----------------------------------------

    # p1 = np.array([[887, 530], [1069, 540], [886, 607], [1066, 617]], dtype=np.float32)
    # p2 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    #
    # M = cv2.getPerspectiveTransform(p1, p2)
    # warped = cv2.warpPerspective(cv2img, M, (w, h))
    # cv2.imwrite("{}".format(img_path.replace(".jpeg", "_warpPerspective.jpg")), warped)
    # -------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------

    # # -------------------------------------------------------------------------------------------------------------------------------
    # # -------------------------------------------------------------------------------------------------------------------------------
    # MAPSIZE = 10 * 1024 * 1024 * 1024 * 1024
    # createDataset_v2(data_path="/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash_train.txt", checkValid=True, map_size=MAPSIZE)
    #
    # lmdb_data = LMDBImageDataset_v2(path="/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/lmdb")
    # img_x, target, input_len, target_len = lmdb_data.__getitem__(0)
    # print("++++++++++++")
    # # -------------------------------------------------------------------------------------------------------------------------------
    # # -------------------------------------------------------------------------------------------------------------------------------

    # =====================================================================================================================
    # ============================================= generate file list to txt =============================================
    # gen_data_txt_list(data_path="/home/disk/disk7/data/004.Knife_Det/test/test_20240426", one_dir_flag=True)

    # =====================================================================================================================
    # ================================================ extract video frames ===============================================
    # extract_one_video_frames(video_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/Robot_Test/Videos/20230718/20230718145019-c1.mp4", gap=1)
    # extract_videos_frames(base_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Robot_Test/Videos/20240304", gap=1, save_path="")
    # merge_dirs_to_one_dir(data_path="/home/disk/disk7/data/004.Knife_Det/Others/scissors", use_glob=False, n_subdir=2)
    # split_dir_multithread(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/test/dbnet/v1_add_20240203/output/20240202_merged_output_output_v5_/orig_img_no_label", split_n=4)
    # move_one_dir_content_to_other_dir(src_dir="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/CRNN_OpenDataset/Syn90k/New Folder_merged_split_16_dirs_merged", dst_dir="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/CRNN_OpenDataset/Syn90k/Syn90k_merged")

    # =====================================================================================================================
    # ================================================ random select files ================================================
    # random_select_files(data_path="/home/disk/disk7/data/013.Droplet_Det/test/v1_random_selected_1334/images", select_num=50, move_or_copy="copy", select_mode=0)
    # random_select_images_and_labels(data_path="/home/disk/disk7/data/013.Droplet_Det/train/v1", select_num=1334, move_or_copy="move", select_mode=0)

    # =====================================================================================================================
    # ======================================== labelbee json, VOC xml<--> yolo txt ========================================
    convert_labelbee_det_json_to_yolo_txt(data_path="/home/disk/disk7/data/004.Knife_Det/Others/knife_scissors", copy_image=True)
    # convert_yolo_txt_to_labelbee_det_json(data_path="/home/disk/disk7/data/004.Knife_Det/Others/knife_scissors")
    # convert_VOC_xml_to_yolo_txt(data_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/20230609/BdSLImset-master_merged", classes=['a', 'dh', 'o', 'ga', 'oo', 'e', 'i', 'kh', 'u', 'k'], val_percent=0.1)
    # convert_labelbee_kpt_json_to_yolo_txt(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/train/part/kpt", copy_image=False)
    # convert_labelbee_kpt_json_to_dbnet_gt(data_path="/home/disk/disk7/data/012.FastDeploy/train/v2/digital_pointer_meter/digital_pointer_meter_cropped/0", copy_image=True)
    # merge_det_bbx_and_kpt_points_to_yolov5_pose_labels(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/train/part", cls=0)
    # convert_labelbee_seg_json_to_png(base_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/dbnet_data/dbnet_seg")
    # coco2yolo(root="/home/zengyifan/wujiahu/data/000.Open_Dataset/coco")
    # convert_yolo_txt_to_VOC_xml(data_path="")  #TODO
    # labelbee_kpt_json_to_labelme_kpt_json(data_path="/home/zengyifan/wujiahu/code/heatmap_keypoint/data/20240123_Chinese_fname/NewFolder")
    # labelbee_kpt_json_to_labelme_kpt_json_multi_points(data_path="/home/zengyifan/wujiahu/code/heatmap_keypoint/data/20240123_Chinese_fname/NewFolder")
    # doPerspectiveTransformByLabelmeJson(data_path="/home/zengyifan/wujiahu/code/heatmap_keypoint/data/20240123_Chinese_fname/NewFolder_labelme_format", r=0)

    # dbnet_aug_data(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/train/dbnet/v1/train_20240222/dbnet_seg", bg_path="/home/zengyifan/wujiahu/data/000.Bg/bg_normal/bg_5000/images", maxnum=10000)
    # vis_dbnet_gt(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/train/dbnet/v1/train_20240222/dbnet_seg/output")
    # warpPerspective_img_via_labelbee_kpt_json(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/dbnet_data/20240220")

    # =====================================================================================================================
    # =============================================== visualize yolo label ================================================
    # vis_yolo_label(data_path="/home/disk/disk7/data/013.Droplet_Det/train/v1_random_selected_1334", print_flag=False, color_num=1000, rm_small_object=False, rm_size=32)  # TODO: 1.rm_small_object have bugs.

    # =====================================================================================================================
    # ================================================ change image suffix ================================================
    # convert_to_jpg_format(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/dbnet_data/20240220/images")
    # convert_to_png_format(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/test/dbnet/part_det_kpt_seg_labelbee/det/NewFolder")

    # =====================================================================================================================
    # ================================================== convert to gray ==================================================
    # convert_to_gray_image(data_path="")

    # =====================================================================================================================
    # ================================================ change txt content =================================================
    # change_txt_content(txt_base_path="/home/disk/disk7/data/006.Fire_Smoke_Det/train/smoke_fire/New_v5/test")
    # remove_yolo_txt_contain_specific_class(data_path="/home/disk/disk7/data/012.FastDeploy/train/v2", rm_cls=(2, ))
    # merge_txt(path1="/home/disk/disk7/data/006.Fire_Smoke_Det/train/smoke_fire/New_v5/others/train/labels-orig+80", path2="/home/disk/disk7/data/006.Fire_Smoke_Det/train/smoke_fire/New_v5/others/train/labels_pred_with_coco_weights")
    # list_yolo_labels(label_path="/home/disk/disk7/data/012.FastDeploy/train/v2/digital_pointer_meter/labels_new")
    # merge_txt_files(data_path="/home/disk/disk7/data/000.ChineseOCR/data/test")

    # =====================================================================================================================
    # ==================================================== crop image =====================================================
    # crop_image_according_labelbee_json(data_path="/home/zengyifan/wujiahu/data/004.Knife_Det/others/Others/20231018", crop_ratio=(1, 1.2, 1.5, ))
    # crop_ocr_rec_img_according_labelbee_det_json(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/test/crnn/others/20240122")
    # crop_image_according_yolo_txt(data_path="/home/disk/disk7/data/012.FastDeploy/train/v2/digital_pointer_meter", CLS=(0, ), crop_ratio=(1.0, ))  # 1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0,
    # random_crop_gen_cls_negative_samples(data_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/Others/HK_Prison/Images/wubao/images", random_size=(196, 224, 256, 288, 384), randint_low=1, randint_high=4, hw_dis=100, dst_num=1000)
    # seg_object_from_mask(base_path="/home/zengyifan/wujiahu/data/007.Lock_Det/seg/20230505")

    # =====================================================================================================================
    # ================================================= OCR data process ==================================================
    # crnn_data_makeBorder(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/CRNN_OpenDataset/ALL_20230907/train_renamed")
    # do_makeBorderv6(data_path="/home/disk/disk7/docker/Projects/000_ChineseOCR/ChineseOCR_images")

    # =====================================================================================================================
    # ======================================================= rename ======================================================
    # rnf = RenameFiles()
    # rnf.rename_files(data_path="/home/disk/disk7/data/004.Knife_Det/Others/scissors_merged", use_orig_name=False, new_name_prefix="004_scissors_20240428", zeros_num=7, start_num=0)
    # # rnf.rename_files(data_path="/home/disk/disk7/data/004.Knife_Det/Others/labeled/20240410/labels", use_orig_name=False, new_name_prefix="004_rename_20240410", zeros_num=7, start_num=0)
    # rnf.rename_labelbee_json_files(data_path="/home/disk/disk7/data/013.Droplet_Det/others/droplet_x2_collect/cut_out13/jsons", use_orig_name=False, new_name_prefix="013_x2_collect_cut_out13_rename_20240422", zeros_num=7, start_num=0)
    # rnf.rename_labelbee_json_files_test(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/dbnet_data/20240220/jsons")
    # rnf.rename_add_str_before_filename(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/Others/dbnet_data/20240220/output/output_warp_test_resize__output_v5/unexpected", add_str="orig")
    # rnf.rename_test_20240223(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/train/crnn/train/7")
    # rnf.check_label(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/train/crnn/train/6")

    # agd = AugData()
    # agd.change_brightness_add_noise_etc_multithread(img_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/20230609/BdSLImset-master_merged/BdSLImset-master_merged_cropped_merged", mtd_num=8)

    # Resz = ResizeImages()
    # Resz.resize_images(img_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_4_cls_20231225/yolov5_with_cls_cp_384_384/images", size=(384, 384), n=8)

    # # =====================================================================================================================
    # # ================================================= generate font char image ==================================================
    # font_dir = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/fonts/27_fonts"
    # save_path = "/home/zengyifan/wujiahu/data/010.Digital_Rec/others/gen_fake/output/font_char_img/bg1"
    # os.makedirs(save_path, exist_ok=True)
    #
    # FONT_CHARS_DICT = get_all_font_chars(font_dir=font_dir, word_set="0123456789.AbC")
    # print(FONT_CHARS_DICT)
    #
    # font_path_list = list(FONT_CHARS_DICT.keys())
    # for ft_path in tqdm(font_path_list):
    #     font_name = os.path.splitext(os.path.basename(ft_path))[0]
    #     ft = ImageFont.truetype(ft_path, size=48)
    #     for a in "0123456789.AbC":
    #         image = gen_img(imgsz=(64, 128), font=ft, alpha=a, target_len=1)
    #         cv2.imwrite("{}/{}_bg1_{}.jpg".format(save_path, font_name, a), image)

    # =====================================================================================================================
    # ========================================== select images according C++ output =======================================
    # select_images_according_C_Plus_Plus_det_output(txt_path="/home/zengyifan/wujiahu/data/000.Open_Dataset/000.Project_Test_Results/001_banner_det/open_imagev4_list_res_thr_0.60_20240122.txt",
    #                                                save_path_flag="current", save_no_det_res_img=False, crop_expand_ratio=1.1)
    #
    # select_images_according_C_Plus_Plus_det_cls_output(txt_path="/home/zengyifan/wujiahu/data/000.Open_Dataset/000.Project_Test_Results/004_knife_det/open_imagev4_list_res_thr_0.60_20240302.txt",
    #                                                    save_path_flag="current", save_path="", save_no_det_res_img=False, n_classes=2, crop_expand_ratio=1.1)
    #
    # select_images_according_C_Plus_Plus_det_cls_output_two_classes(txt_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/test/test_2_cls/images_list_res_thr_0.60_20231121_2_ratios.txt",
    #                                                    save_path_flag="current", save_path="", save_no_det_res_img=False, save_crop_img=True, save_src_img=False, save_vis_img=True, crop_expand_ratio=1.1)

    # select_images_according_C_Plus_Plus_det_cls_output_n_classes(txt_path="/home/zengyifan/wujiahu/data/000.Open_Dataset/000.Project_Test_Results/003_cigar_det/20231108/open_imagev4_list_res_thr_0.60_20231108.txt",
    #                                                              save_path_flag="current", save_path="", save_no_det_res_img=False, save_crop_img=True, save_src_img=False, save_vis_img=True, crop_expand_ratio=1.5, n_cls=4)

    # select_images_and_write_yolo_label_according_C_Plus_Plus_det_cls_output(img_path="/home/zengyifan/wujiahu/data/000.Open_Dataset/000.Project_Test_Results/003_cigar_det/20231204/C_Plus_Plus_det_output/open_imagev4/src_images/0",
    #                                                                         txt_path="/home/zengyifan/wujiahu/data/000.Open_Dataset/000.Project_Test_Results/003_cigar_det/20231204/open_imagev4_list_res_thr_0.50_20231204.txt")

    # select_images_and_write_yolo_label_according_C_Plus_Plus_det_output(img_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/VIDEOS/YouTube_20230516/C_Plus_Plus_det_output/20230516_video_frames_merged/vis_images",
    #                                                                     txt_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/VIDEOS/YouTube_20230516/20230516_video_frames_merged_list_res_thr_0.60_20230516.txt")

    # select_images_according_txt_list(data_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/train/20230201_2_cls/train", print_flag=True)
    # select_images_according_txt_file(txt_file="/home/zengyifan/wujiahu/yolo/yolov5-6.2/runs/detect/006_SSOD/not_labeled_006_768_20230630_yolov5s_2_cls_20230630_SSOD_base2/0.6.txt", img_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/train/smoke_fire/SSOD/train/not_labeled", save_path=None, cp_mv_del="copy")
    # select_images_according_yolo_label(data_path="/home/zengyifan/wujiahu/data/007.Lock_Det/train/20230320/train_3985")
    # select_same_files(dir1="/home/zengyifan/wujiahu/yolo/yolov5-6.2/runs/detect/006_SSOD/cmp_20230704/unexpected_base", dir2="/home/zengyifan/wujiahu/yolo/yolov5-6.2/runs/detect/006_SSOD/cmp_20230704/unexpected_SSOD", select_dir="dir2")
    # select_specific_file_by_name(data_path="/home/disk/disk7/data/010.Digital_Rec/crnn/train/15_cls/v6/v6_20231122/000", key_words=["horizontal", ], mode=0, cp_or_mv="move")
    # select_specific_file_by_name(data_path="/home/disk/disk7/data/000.OpenDatasets/coco/val/labels", key_words=["person0", "knife", ], mode=0, cp_or_mv="move")
    # select_horizontal_images(data_path="/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/train_v1_add_20240315/005_ICDAR2019-ArT/img_with_label_in_fname")

    # =====================================================================================================================
    # ==================================================== delete or move =================================================
    # remove_corrupt_images_pil(img_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/20230606_data/20230605_merged", move_or_delete="delete")
    # remove_corrupt_images_pil_v2(img_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/20230606_data/20230605_merged", move_or_delete="delete")
    # remove_corrupt_images_pil_v2_main(img_path="/home/disk/disk7/data/004.Knife_Det/Others/scissors_merged", move_or_delete="move")  # ATTENTION: Can work, GOOD!
    # remove_corrupt_images_cv2_v2_main(img_path="/home/disk/disk7/data/004.Knife_Det/Others/scissors_merged", move_or_delete="move")  # ATTENTION: Can work, GOOD!
    # remove_corrupt_images_opencv(img_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/001")
    # mv_or_remove_small_images(img_path="/home/zengyifan/wujiahu/data/000.Open_Dataset/coco/train2017/train2017_cropped/Random_Selected/1.1_random_selected_25000", rmsz=64, mode=0)

    # data_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/v4_5_cls_20231122/20231122_Pexels/mouth/mouth_cropped/2"
    # dirs = get_file_list(data_path)
    # for d in tqdm(dirs):
    #     d_path = data_path + "/{}".format(d)
    #     remove_small_images(img_path=d_path, rmsz=48, mode=0)

    # remove_specific_file_by_name_index(data_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/optimize_det_model/C_Plus_Plus_det_output/20230128_Pexels_videos_video_frames_merged/no_det_res", key_index=5)
    # remove_specific_file_by_name(data_path="/home/zengyifan/wujiahu/data/005.Skating_Det/train/v1_orig/labels", key_words=["paste", "aug", ], mode=0)
    # move_or_delete_file_exist_corresponding_file(base_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_5_cls_20231128_cp/train/train", dir1_name="images", dir2_name="labels_new", move_or_delete="copy", dir="dir1")
    # move_or_delete_file_exist_corresponding_file_under_diffirent_dir(dir_path1="/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_5_cls_20230717/v2_20230602/images",
    #                                                                  dir_path2="/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_5_cls_20230717/v2_20230602/milk_tea_cup/labels_new",
    #                                                                  unexpected_path="", flag="copy", dir="dir1")
    # move_or_delete_file_not_exist_corresponding_file(base_path="/home/disk/disk7/data/012.FastDeploy/train/v2/digits_det", dir1_name="images", dir2_name="labels", labelbee_json_label=False, move_or_delete="move", dir="dir2")
    # move_or_delete_file_not_exist_corresponding_file(base_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_5_cls_20231128/train/no_label/20231208_Pexels_video_frames_merged", dir1_name="images", dir2_name="jsons", labelbee_json_label=True, move_or_delete="move", dir="dir1")
    # move_or_delete_file_not_exist_corresponding_file_under_diffirent_dir(dir_path1="", dir_path2="", unexpected_path="", move_or_delete="delete", dir="dir2")
    # move_or_delete_file_not_exist_corresponding_file_under_diffirent_dir_v2(dir_path1="/home/zengyifan/wujiahu/data/Open_Dataset/005_skating_det/20230221/C_Plus_Plus_det_output/coco_train2017/crop_images/1.0",
    #                                                                         dir_path2="/home/zengyifan/wujiahu/data/Open_Dataset/005_skating_det/20230221/C_Plus_Plus_det_output/coco_train2017/vis_images",
    #                                                                         unexpected_path="", move_or_delete="delete")

    # for ii in [1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0]:
    #     move_or_delete_file_not_exist_corresponding_expand_ratio_cropped_images(dir_path1="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/cls/cls_train/3_cls/v3_20230710/not_label_34349/not_label_34349_cropped/0/1.0",
    #                                                                             dir_path2="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/cls/cls_train/3_cls/v3_20230710/not_label_34349/not_label_34349_cropped/0/{}".format(ii),
    #                                                                             unexpected_path="", move_or_delete="move")

    # mv_or_rm_black_images(img_path="/home/zengyifan/wujiahu/data/002.Exit_Light_Det/cls/v2_20230403_4_cls/0", flag="mv", pixel_sum=500000)
    # mv_or_rm_black_images(img_path="/home/zengyifan/wujiahu/data/002.Exit_Light_Det/cls/v2_20230403_4_cls/1", flag="mv", pixel_sum=500000)
    # mv_or_rm_black_images(img_path="/home/zengyifan/wujiahu/data/002.Exit_Light_Det/cls/v2_20230403_4_cls/2", flag="mv", pixel_sum=500000)
    # mv_or_rm_black_images(img_path="/home/zengyifan/wujiahu/data/002.Exit_Light_Det/cls/v2_20230403_4_cls/3", flag="mv", pixel_sum=500000)

    # ssim_move_or_remove_same_images(img_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/test/dbnet/v1_add_20240203/GOOD", imgsz=(64, 64), move_or_remove="move")
    # ssim_move_or_remove_same_images_multithread(img_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/test/dbnet/v1_add_20240203/GOOD", imgsz=(32, 32), move_or_remove="move")

    # =====================================================================================================================
    # ======================================================= copy ========================================================
    # copy_n_times(data_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/Robot_Test/Videos/v2/train/0", n=1, save_path="current", print_flag=True)
    # copy_file_according_txt(txt_path="/home/zengyifan/wujiahu/data/002.Exit_Light_Det/others/robot_test/20230321/pilotLamp/2_list.txt",
    #                         save_path="/home/zengyifan/wujiahu/data/002.Exit_Light_Det/others/robot_test/20230321/pilotLamp/2_cp")
    # copy_file_by_name(data_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/train/20230303_2_cls/train_127318/labels", key_words=["Pexels", "pexels", ], mode=0)

    # =========================================================================================================================================================================================================================
    # ===================================================================================================== AUG DATA ==========================================================================================================
    # =========================================================================================================================================================================================================================
    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Object detection data augmentation -----------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # det_data_aug_main(data_path="/home/zengyifan/wujiahu/data/007.Lock_Det/train/20230609/add_2", aug_num=100)
    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Object detection data augmentation -----------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ paste cropped image through (split & joint) ------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # selected_dir = ""
    # bg_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/wabishi_chishouzhi".format(selected_dir)  # bg_path = "/home/zengyifan/wujiahu/data/001.Banner_Det/bg/bg_5000".format(selected_dir)
    # bg_images_dir_name, bg_labels_dir_name = "images", "labels"
    # # /home/zengyifan/wujiahu/data/003.Cigar_Det/seg_crop/cropped/cigar_pure_bg/1.1_resize
    # # /home/zengyifan/wujiahu/data/003.Cigar_Det/seg_crop/cropped/cup
    # cropped_object_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/seg_crop/cropped/cigar_pure_bg/1.1_resize"
    # save_path = "/home/zengyifan/wujiahu/data/003.Cigar_Det/train/v3_5_cls_20231128/train/no_label/wabishi_chishouzhi_20231214_AUG".format(selected_dir)
    # # # # scale_type = 1, scale_ratio = 0.02; scale_type = 2, scale_ratio = 0.5
    # paste_cropped_object_for_det_aug_data_train_negative_samples_multi_thread_v5_main(bg_path, bg_images_dir_name, bg_labels_dir_name, cropped_object_path, save_path,
    #                                                                                   random_N=1, scatter_bbxs_num=5, dis_thresh=50, scale_flag=True, scale_type=1, scale_ratio=0.025, cls=0, add_rename_str="003_wabishi_chishouzhi_aug_20231214_cigar_001")

    # ---------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- PIL paste cropped image -----------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # pil_paste_cropped_object_for_det_aug_data_train_negative_samples_multi_thread_v6_main(bg_path, bg_images_dir_name, bg_labels_dir_name, cropped_object_path, save_path,
    #                                                                                       paste_largest_num=1, add_rename_str="pasted", scale_flag=True, scale_ratio=0.025, cls=0, dis_thresh=50, scatter_bbxs_num=5)

    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- images add -----------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # selected_dir = ""
    # bg_path = "/home/zengyifan/wujiahu/data/001.Banner_Det/bg/bg_1000/images".format(selected_dir)
    # seg_object_path = "/home/zengyifan/wujiahu/data/007.Lock_Det/seg/20230505/output"
    # data_path = "/home/zengyifan/wujiahu/data/007.Lock_Det/seg/20230505"
    # save_path = "/home/zengyifan/wujiahu/data/007.Lock_Det/train/20230505".format(selected_dir)
    # add_black_bg_object_for_det_aug_data_multi_thread_main(bg_path, seg_object_path=seg_object_path, data_path=data_path, save_data_path=save_path, random_N=2, cls=0, rename_add_str="lock_aug_20230505")

    # ---------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- seamless paste ---------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # selected_dir = ""
    # bg_path = "/home/zengyifan/wujiahu/data/001.Banner_Det/bg/bg_1000".format(selected_dir)
    # bg_img_dir_name, bg_lbl_dir_name = "images", "labels"
    # object_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/seg/seamless_clone/smoke"
    # save_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/seamless_paste_test_20230412_smoke".format(selected_dir)
    # seamless_paste_main_v6(bg_path, bg_img_dir_name, bg_lbl_dir_name, object_path, save_path, obj_num=1, affine_num=1, threshold_min_thr=10, medianblur_k=5, pixel_thr=10, iou_thr=0.05, bbx_thr=0.80, cls=0, rename_add_str="exit_light_20230411", random_scale_flag="big_images", adaptiveThreshold=False)

    # ---------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------- seamless clone ---------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # seamless_clone(bg_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/prison_data/filter_prison_img/20221111152028_8b46d1_61.avi_0025090.jpg", obj_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/seg/for_paste/smoke_for_paste_0000000.jpg")

    # bg_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/Others/HK_Prison/Videos_Frames/Gray_videos_frames_20230531/20230531005246-c1"
    # obj_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/seg_crop/for_seamless_clone/smoke"
    # save_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/Others/seamless_clone_test_20230531_Gray_Frames"
    # # aug_img_with_seamless_clone(bg_path=bg_path, obj_path=obj_path, save_path=save_path, aug_n=3, label=1, fname_add_content="fire_20230322")
    # aug_img_with_seamless_clone_multi_thread(bg_path, obj_path, save_path, aug_n=1, label=0, fname_add_content="smoke_gray_20230531")

    # =========================================================================================================================================================================================================================
    # ===================================================================================================== AUG DATA ==========================================================================================================
    # =========================================================================================================================================================================================================================

    # =========================================================================================================================================================================================================================
    # ================================================================================================= YOLOv5 inference ======================================================================================================
    # =========================================================================================================================================================================================================================
    # onnx_path = r"E:\GraceKafuu\Python\yolov5-6.2\yolov5s.onnx"
    # img_path = r"E:\GraceKafuu\Python\yolov5-6.2\data\images\bus.jpg"
    #
    # model = YOLOv5_ONNX(onnx_path)
    # model_input_size = (448, 768)
    # img0, img, src_size = model.pre_process(img_path, img_size=model_input_size)
    # print("src_size: ", src_size)
    #
    # t1 = time.time()
    # pred = model.inference(img)
    # t2 = time.time()
    # print("{:.12f}".format(t2 - t1))
    #
    # out_bbx = model.post_process(pred, src_size, img_size=model_input_size)
    # print("out_bbx: ", out_bbx)
    # for b in out_bbx:
    #     cv2.rectangle(img0, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
    # cv2.imshow("test", img0)
    # cv2.waitKey(0)
    # =========================================================================================================================================================================================================================
    # ================================================================================================= YOLOv5 inference ======================================================================================================
    # =========================================================================================================================================================================================================================

    # =========================================================================================================================================================================================================================
    # ================================================================================================= YOLOv8 inference ======================================================================================================
    # =========================================================================================================================================================================================================================
    # onnx_path = r"E:\GraceKafuu\Python\ultralytics-main\yolov8n.onnx"
    # img_path = r"E:\GraceKafuu\Python\yolov5-6.2\data\images\bus.jpg"
    #
    # model = YOLOv8_ONNX(onnx_path)
    # model_input_size = (640, 640)
    # img0, img, src_size = model.pre_process(img_path, img_size=model_input_size)
    # print("src_size: ", src_size)
    #
    # t1 = time.time()
    # pred = model.inference(img)
    # t2 = time.time()
    # print("{:.12f}".format(t2 - t1))
    #
    # out_bbx = model.post_process(pred, src_size, img_size=model_input_size)
    # print("out_bbx: ", out_bbx)
    # for b in out_bbx:
    #     cv2.rectangle(img0, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
    # cv2.imshow("test", img0)
    # cv2.waitKey(0)

    # =========================================================================================================================================================================================================================
    # ================================================================================================= YOLOv8 inference ======================================================================================================
    # =========================================================================================================================================================================================================================

    # =========================================================================================================================================================================================================================
    # =============================================================================================== cls model inference =====================================================================================================
    # =========================================================================================================================================================================================================================
    # TODO: 1.Check which onnxruntime-gpu version can work, i.e. use "CUDAExecutionProvider". 2.AUC-ROC
    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/code/image_classification/weights/smoking/train_003_cls_v8.8_20230710_ELU_mean_0.5_std_0.5/model/best_model_376_99.8736.onnx", n_classes=2, device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc(test_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/cls/cls_test/20230616/1", output_path=None, save_FP_FN_img=False, save_dir_name="cigar_cls_mbv2_128_128_v3_20230707", mv_or_cp="copy", NP="P", metrics=True)  # ATTENTION: arg NP!!!

    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/code/image_classification/weights/occlusion_det/000.Occlusion_Det_cls_train_20230713/model/best_model_314_99.9200.onnx", n_classes=2, input_size=(256, 256), device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc(test_path="/home/zengyifan/wujiahu/data/000.Occlusion_Det/cls/test/1", output_path=None, save_FP_FN_img=True, save_dir_name="20230714", mv_or_cp="copy", NP="P", metrics=True)  # ATTENTION: arg NP!!!

    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/code/image_classification/weights/smoke_fire/006_cls_20230804_finetune/model/best_model_047_99.9313.onnx", n_classes=3, input_size=(128, 128), device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc(test_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/cls/cls_test/3_cls/v1_test_20230712/0", output_path=None, save_FP_FN_img=False, save_dir_name="20230808", mv_or_cp="copy", NP="N", metrics=True)  # ATTENTION: arg NP!!!

    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/code/image_classification/weights/smoke_fire/006_cls_20231012/model/latest_model_393_99.9967.onnx", n_classes=3, input_size=(128, 128), device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc(test_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/cls/cls_test/3_cls/v1_test_20230712/0", output_path=None, save_FP_FN_img=True, save_dir_name="20231013", mv_or_cp="copy", NP="N", metrics=True)  # ATTENTION: arg NP!!!

    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/data/004.Knife_Det/weights/v4_20230721/cls/knife_cls_mbv2_128_128_v4.onnx", n_classes=2, input_size=(128, 128), device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc(test_path="/home/zengyifan/wujiahu/data/004.Knife_Det/cls/train/v5_add/1", output_path=None, save_FP_FN_img=True, save_dir_name="20231013", mv_or_cp="copy", NP="P", metrics=True)  # ATTENTION: arg NP!!!

    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/weights/cls/v3_20231013/smoke-fire_cls_mbv2_128_128_v3.onnx", n_classes=3, input_size=(128, 128), device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc_n_cls(test_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/cls/cls_train/3_cls/v7_add_20231016/0", output_path=None, save_pred_true=False, save_pred_false=True, save_dir_name="20231016", mv_or_cp="copy")  # ATTENTION: arg NP!!!

    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/data/004.Knife_Det/weights/v5_20231020/cls/knife_cls_mbv2_128_128_v5.onnx", n_classes=2, input_size=(128, 128), device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc_n_cls(test_path="/home/zengyifan/wujiahu/data/004.Knife_Det/cls/test/v4/0", output_path=None, save_pred_true=False, save_pred_false=True, save_dir_name="20231020", mv_or_cp="copy")  # ATTENTION: arg NP!!!

    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/weights/cls/v3_20231013/smoke-fire_cls_mbv2_128_128_v3.onnx", n_classes=3, input_size=(128, 128), device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc_n_cls(test_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/cls/cls_test/3_cls/v1_test_20230712/2", output_path=None, save_pred_true=False, save_pred_false=False, save_dir_name="20231030", mv_or_cp="copy")  # ATTENTION: arg NP!!!

    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/weights/cls/v4_20231117/smoke-fire_cls_mbv2_128_128_v4.onnx", n_classes=3, input_size=(128, 128), device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc_n_cls(test_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/cls/cls_test/3_cls/v1_test_20230712/2", output_path=None, save_pred_true=False, save_pred_false=False, save_dir_name="20231117", mv_or_cp="copy")  # ATTENTION: arg NP!!!

    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/code/image_classification/weights/smoking/003_cls_v8.8_keep_ratio_20231117/model/best_model_265_99.9858.onnx", n_classes=2, input_size=(128, 128), keep_ratio_flag=True, device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc_n_cls(test_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/cls/cls_test/20230616/1", output_path=None, save_pred_true=False, save_pred_false=False, save_dir_name="20231120", mv_or_cp="copy")  # ATTENTION: arg NP!!!

    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/GoroboAIReason/models/cigar_detect/cigar_cls_mbv2_128_128_v3.onnx", n_classes=2, input_size=(128, 128), keep_ratio_flag=False, device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc_n_cls(test_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/cls/cls_test/20230616/1", output_path=None, save_pred_true=False, save_pred_false=False, save_dir_name="20231121", mv_or_cp="copy")  # ATTENTION: arg NP!!!

    # cls_model = ClsModel(model_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/weights/cls/big_imgs/v1_big_img_20231117/smoke-fire-whole-img_cls_mbv2_256_256_v1.onnx", n_classes=3, input_size=(256, 256), keep_ratio_flag=False, device="cuda:0", print_infer_time=False)
    # acc = cls_model.cal_acc_n_cls(test_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/cls/cls_test/big_img/0", output_path=None, save_pred_true=False, save_pred_false=False, save_dir_name="20231227", mv_or_cp="copy")  # ATTENTION: arg NP!!!

    # knife_cls_mbv2_128_128_v4: {'0': 0.9979127760079095, '1': 0.0020872239920905196}2
    # knife_cls_mbv2_128_128_v5: {'0': 0.99802262990223, '1': 0.0019773700977699657}

    # {'0': 0.9343544857768052, '1': 0.029905178701677606, '2': 0.03574033552151714}
    # TP, FP, FN, TN: 0, 41, 0, 1281
    # Accuracy: 0.968986384266 Precision: 0.000000000000 Recall: 0.000000000000 F1: 0.000000000000 Specificity: 0.968986384266

    # best_model_211_99.9995: {'0': 0.9358426005132592, '1': 0.0641573994867408} {'0': 0.06682899031238668, '1': 0.9331710096876134}
    # best_model_195_99_9995: {'0': 0.9308525805531793, '1': 0.06914741944682064} {'0': 0.06600010361083769, '1': 0.9339998963891623}
    # best_model_184_99.9988: {'0': 0.935129740518962, '1': 0.06487025948103792}  {'0': 0.06320261099310988, '1': 0.9367973890068901}
    # best_model_241_99.9995: {'0': 0.9394069004847448, '1': 0.060593099515255204} {'0': 0.06750246075739523, '1': 0.9324975392426048}
    # best_model_251_99.9995: {'0': 0.9392643284858854, '1': 0.06073567151411463} {'0': 0.06776148785162928, '1': 0.9322385121483707}
    # best_model_260_99.9995: {'0': 0.9345594525235243, '1': 0.06544054747647562} {'0': 0.06532663316582915, '1': 0.9346733668341709}
    # best_model_283_99.9995: {'0': 0.9362703165098375, '1': 0.06372968349016253} {'0': 0.06568927109775682, '1': 0.9343107289022432}

    # 2023.06.21 Test Results:
    # v3 20230621 latest_model_254_99.9938.onnx:  {'0': 0.8873587008438146, '1': 0.11264129915618531}  {'0': 0.03060689964828257,  '1': 0.9693931003517174}
    # v3 20230621 best_model_154_99.9941.onnx:    {'0': 0.887438306002229,  '1': 0.11256169399777105}  {'0': 0.030008231684501983, '1': 0.969991768315498}
    # v3 20230612 cigar_cls_mbv2_128_128_v3.onnx: {'0': 0.9361566629517593, '1': 0.06384333704824073}  {'0': 0.031168150864326873, '1': 0.9688318491356731}
    # v3 20230704 best_model_192_99.9917.onnx:
    # v2 cigar_cls_mbv2_128_128_v2.onnx:          {'0': 0.9059067027543385, '1': 0.09409329724566153}  {'0': 0.0407618049839108,   '1': 0.9592381950160892}
    # v1 cigar_cls_mbv2_128_128_v1.onnx:          {'0': 0.9265244387836332, '1': 0.07347556121636682}  {'0': 0.061909750804460074, '1': 0.93809024919554}

    # smoke fire:
    # best_model_158_99.9898.onnx:     {'0': 0.9905178701677607, '1': 0.0029175784099197666, '2': 0.006564551422319475}  {'0': 0.0011, '1': 0.99308, '2': 0.00582}
    # v1 best_model_216_99.9924.onnx:  {'0': 0.9919766593727206, '1': 0.0036469730123997084, '2': 0.00437636761487965}   {'0': 0.0012, '1': 0.9964,  '2': 0.0024}
    # v2 best_model_041_99.9312.onnx:  {'0': 0.9978118161925602, '1': 0.0007293946024799417, '2': 0.0014587892049598833} {'0': 0.0007, '1': 0.99828, '2': 0.00102}
    # v2 best_model_047_99.9313.onnx:  {'0': 0.9978118161925602, '1': 0.0007293946024799417, '2': 0.0014587892049598833} {'0': 0.00064, '1': 0.99836, '2': 0.001}

    # smoke_fire v2:
    # {'0': 0.9978118161925602, '1': 0.0007293946024799417, '2': 0.0014587892049598833}
    # {'0': 0.00028, '1': 0.99816, '2': 0.00156}
    # {'0': 0.002526338421844765, '1': 0.0008062782197376908, '2': 0.9966673833584175}

    # smoke_fire v3:
    # {'0': 0.9343544857768052, '1': 0.030634573304157548, '2': 0.0350109409190372}
    # {'0': 0.00112, '1': 0.98944, '2': 0.00944}
    # {'0': 0.0030817745287751736, '1': 0.0038522181609689675, '2': 0.9930660073102558}

    # smoke_fire v4:
    # {'0': 0.9168490153172867, '1': 0.02261123267687819, '2': 0.060539752005835154}
    # {'0': 0.00158, '1': 0.98392, '2': 0.0145}
    # {'0': 0.00490933849351394, '1': 0.00490933849351394, '2': 0.9901813230129721}

    # 003_smoking:
    # ---- 003_cls_v8.8_keep_ratio_20231117/model/best_model_265_99.9858.onnx:
    # keep_ratio=True: {'0': 0.8458844133099825, '1': 0.15411558669001751}  {'0': 0.05427673426625758, '1': 0.9457232657337424}
    # keep_ratio=False: {'0': 0.9013692087247254, '1': 0.09863079127527463},  {'0': 0.15551148694155503, '1': 0.844488513058445}
    # ---- cigar_cls_mbv2_128_128_v3.onnx:
    # keep_ratio=True: {'0': 0.9539086132781405, '1': 0.04609138672185958}  {'0': 0.19627329192546583, '1': 0.8037267080745342}
    # keep_ratio=False: {'0': 0.9361566629517593, '1': 0.06384333704824073} {'0': 0.031168150864326873, '1': 0.9688318491356731}

    # --------------------------------------------------------------------------------------------------------------------------
    # v1:
    # v1_big_img_20231117/smoke-fire-whole-img_cls_mbv2_256_256_v1.onnx
    # {'0': 0.3155481450604419, '1': 0.11588161734055856, '2': 0.5685702375989996}
    # {'0': 0.016432986582240315, '1': 0.029549223579074326, '2': 0.9540177898386853}

    # v2:
    # 006_cls_big_imgs_20231221/model/best_model_114_99.9808.onnx:
    # {'0': 1.0, '1': 0.0, '2': 0.0}
    # {'0': 0.0006030453791647821, '1': 0.00015076134479119553, '2': 0.999246193276044}

    # 006_cls_big_imgs_20231221/model/latest_model_174_99.9802.onnx:
    # {'0': 1.0, '1': 0.0, '2': 0.0}
    # {'0': 0.0004522840343735866, '1': 0.00015076134479119553, '2': 0.9993969546208352}

    # =========================================================================================================================================================================================================================
    # =============================================================================================== cls model inference =====================================================================================================
    # =========================================================================================================================================================================================================================




























