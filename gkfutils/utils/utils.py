# -*- coding:utf-8 -*-

"""
# @Time       : 2022/5/13 13:56, 2024/3/29 14:30 Update
# @Author     : GraceKafuu
# @Email      : 
# @File       : utils.py
# @Software   : PyCharm

Description:
1.
2.
3.

"""

import os
import re
import cv2
import numpy as np
import json
import pandas as pd
import random
import shutil
import time
import threading
import struct
import pickle
import hashlib
from tqdm import tqdm
from glob import glob
import socket
import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler

try:
    from accessdbtools import AccessDatabase, AccessTableData
except ImportError as e:
    print(e)


def get_strftime():
    datetime = time.strftime("%Y%m%d", time.localtime(time.time()))
    return datetime


def timestamp_to_strftime(curr_timestamp):
    strftime_ = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(curr_timestamp))
    return strftime_


def strftime_to_timestamp(curr_strftime):
    pass


def get_date_time(mode=0):
    datetime1 = time.strftime("%Y %m %d %H:%M:%S", time.localtime(time.time()))
    datetime2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    datetime3 = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))

    if mode == 0:
        return datetime1
    elif mode == 1:
        return datetime2
    elif mode == 2:
        return datetime3
    else:
        print("mode should be 0, 1, 2")


def get_file_list(data_path, abspath=False):
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


def get_dir_list(data_path, abspath=False):
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


def get_dir_file_list(data_path, abspath=False):

    list_ = sorted(os.listdir(data_path))
    if abspath:
        list_new = []
        for f in list_:
            f_path = data_path + "/{}".format(f)
            list_new.append(f_path)
        return list_new
    else:
        return list_


def get_dir_name(data_path):
    dir_name = os.path.basename(data_path)
    return dir_name


def get_base_name(data_path):
    base_name = os.path.basename(data_path)
    return base_name


def get_baseName_fileName_suffix(file_path):
    base_name = os.path.basename(file_path)
    file_name = os.path.splitext(base_name)[0]
    suffix = os.path.splitext(base_name)[1]
    return base_name, file_name, suffix


def make_save_path(data_path, dir_name_add_str="results"):
    dir_name = get_base_name(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_{}".format(dir_name, dir_name_add_str)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def LogInit(prex):
    """
    日志按日输出 单进程适用
    """
    log_fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    formatter = logging.Formatter(log_fmt)
    # 创建TimedRotatingFileHandler对象
    # dirname, filename = os.path.split(os.path.abspath(__file__))
    dirname = os.getcwd()
    # logpath = os.path.dirname(os.getcwd()) + '/Logs/'
    logpath = dirname + '/Logs/'
    if not os.path.exists(logpath):
        os.mkdir(logpath)
    log_file_handler = TimedRotatingFileHandler(filename=logpath + prex+'-log.', when="D", interval=1)
    log_file_handler.suffix = prex + "-%Y-%m-%d_%H-%M-%S.log"
    log_file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(prex)
    log.addHandler(log_file_handler)

    return log


def gen_data_txt_list(data_path, one_dir_flag=True):
    if one_dir_flag:
        data_list = sorted(os.listdir(data_path))
        txt_save_path = os.path.abspath(os.path.join(data_path, "../")) + "/{}_list.txt".format(data_path.split("/")[-1])
        with open(txt_save_path, 'w', encoding='utf-8') as fw:
            for f in data_list:
                f_abs_path = data_path + "/{}".format(f)
                fw.write("{}\n".format(f_abs_path))

        print("Success! Generated files list txt --> {}".format(txt_save_path))
    else:
        dirs = sorted(os.listdir(data_path))
        unexpected_dirs = ["others"]

        txt_save_path = os.path.abspath(os.path.join(data_path, "../")) + "/{}_list.txt".format(data_path.split("/")[-1])
        with open(txt_save_path, 'w', encoding='utf-8') as fw:
            for d in dirs:
                d_path = data_path + "/{}".format(d)
                if d in unexpected_dirs: continue
                if os.path.isfile(d_path): continue

                data_list = sorted(os.listdir(d_path))
                for f in data_list:
                    if f.endswith(".jpg") or f.endswith(".png") or f.endswith("bmp") or f.endswith(".jpeg"):
                        f_abs_path = d_path + "/{}".format(f)
                        fw.write("{}\n".format(f_abs_path))


def change_txt_content(txt_base_path):
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


class RenameFiles(object):
    def rename_files(self, data_path, use_orig_name=False, new_name_prefix="", zeros_num=7, start_num=0):
        data_list = sorted(os.listdir(data_path))
        for i in range(len(data_list)):
            img_abs_path = data_path + "/" + data_list[i]
            file_ends = os.path.splitext(data_list[i])[1]
            orig_name = os.path.splitext(data_list[i])[0]
            if use_orig_name:
                if zeros_num == 2:
                    new_name = "{}_{:02d}{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 3:
                    new_name = "{}_{:03d}{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 4:
                    new_name = "{}_{:04d}{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 5:
                    new_name = "{}_{:05d}{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 6:
                    new_name = "{}_{:06d}{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 7:
                    new_name = "{}_{:07d}{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 8:
                    new_name = "{}_{:08d}{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
            else:
                if zeros_num == 2:
                    new_name = "{}_{:02d}{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 3:
                    new_name = "{}_{:03d}{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 4:
                    new_name = "{}_{:04d}{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 5:
                    new_name = "{}_{:05d}{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 6:
                    new_name = "{}_{:06d}{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 7:
                    new_name = "{}_{:07d}{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 8:
                    new_name = "{}_{:08d}{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)

    def rename_labelbee_json_files(self, data_path, use_orig_name=False, new_name_prefix="", zeros_num=7, start_num=0):
        data_list = sorted(os.listdir(data_path))
        for i in range(len(data_list)):
            img_abs_path = data_path + "/" + data_list[i]
            file_ends = os.path.splitext(data_list[i])[1]
            orig_name = os.path.splitext(data_list[i])[0]
            if use_orig_name:
                if zeros_num == 2:
                    new_name = "{}_{:02d}.jpg{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 3:
                    new_name = "{}_{:03d}.jpg{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 4:
                    new_name = "{}_{:04d}.jpg{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 5:
                    new_name = "{}_{:05d}.jpg{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 6:
                    new_name = "{}_{:06d}.jpg{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 7:
                    new_name = "{}_{:07d}.jpeg{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 8:
                    new_name = "{}_{:08d}.jpg{}".format(orig_name, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
            else:
                if zeros_num == 2:
                    new_name = "{}_{:02d}.jpg{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 3:
                    new_name = "{}_{:03d}.jpg{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 4:
                    new_name = "{}_{:04d}.jpg{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 5:
                    new_name = "{}_{:05d}.jpg{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 6:
                    new_name = "{}_{:06d}.jpg{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 7:
                    new_name = "{}_{:07d}.jpeg{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)
                elif zeros_num == 8:
                    new_name = "{}_{:08d}.jpg{}".format(new_name_prefix, i + start_num, file_ends)
                    os.rename(img_abs_path, data_path + "/" + new_name)

    def rename_labelbee_json_files_test(self, data_path):
        data_list = sorted(os.listdir(data_path))
        for i in range(len(data_list)):
            img_abs_path = data_path + "/" + data_list[i]
            # file_ends = os.path.splitext(data_list[i])[1]
            # orig_name = os.path.splitext(data_list[i])[0]
            # new_name = "{}_{:02d}.jpg{}".format(orig_name, i + start_num, file_ends)
            os.rename(img_abs_path, data_path + "/" + data_list[i].replace(".jpeg.json", ".jpg.json"))

    def rename_add_str_before_filename(self, data_path, add_str=""):
        data_list = sorted(os.listdir(data_path))
        for i in range(len(data_list)):
            img_abs_path = data_path + "/" + data_list[i]
            # file_ends = os.path.splitext(data_list[i])[1]
            # orig_name = os.path.splitext(data_list[i])[0]
            # new_name = "{}_{:02d}.jpg{}".format(orig_name, i + start_num, file_ends)
            os.rename(img_abs_path, data_path + "/{}_{}".format(add_str, data_list[i]))

    def rename_test_20240223(self, data_path):
        data_list = sorted(os.listdir(data_path))
        for i in range(len(data_list)):
            img_abs_path = data_path + "/" + data_list[i]
            file_ends = os.path.splitext(data_list[i])[1]
            orig_name = os.path.splitext(data_list[i])[0]
            label = orig_name.split("=")[-1]

            # orig_name_ = ""
            orig_name_ = "=".join([ni for ni in orig_name.split("=")[:-1]])

            if label[-2] != ".":
                label = label[:-1] + "." + label[-1]

            new_name = "{}={}{}".format(orig_name_, label, file_ends)

            # new_name = "{}_{:02d}.jpg{}".format(orig_name, i + start_num, file_ends)
            os.rename(img_abs_path, data_path + "/{}".format(new_name))

    def check_label(self, data_path):
        data_list = sorted(os.listdir(data_path))
        for i in range(len(data_list)):
            img_abs_path = data_path + "/" + data_list[i]
            file_ends = os.path.splitext(data_list[i])[1]
            orig_name = os.path.splitext(data_list[i])[0]
            label = orig_name.split("=")[-1]
            if label != "98.304":
                print(img_abs_path)

            # # orig_name_ = ""
            # orig_name_ = "=".join([ni for ni in orig_name.split("=")[:-1]])
            #
            # if label[-2] != ".":
            #     label = label[:-1] + "." + label[-1]
            #
            # new_name = "{}={}{}".format(orig_name_, label, file_ends)
            #
            # # new_name = "{}_{:02d}.jpg{}".format(orig_name, i + start_num, file_ends)
            # os.rename(img_abs_path, data_path + "/{}".format(new_name))


def unzip_lots_of_files(data_path):
    zip_list = sorted(os.listdir(data_path))
    for f in zip_list:
        f_abs_path = data_path + "/{}".format(f)
        file_name = os.path.splitext(f)[0]
        dir_name = os.path.abspath(os.path.join(f_abs_path, "../..")) + "/{}".format(file_name)
        cmd_line = "tar -xf %s -C %s" % (f_abs_path, dir_name)
        os.makedirs(dir_name, exist_ok=True)

        print(cmd_line)
        os.system(cmd_line)


def merge_dirs_to_one_dir(data_path, use_glob=True, n_subdir=2):
    dst_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_merged".format(data_path.split("/")[-1])
    os.makedirs(dst_path, exist_ok=True)

    if use_glob:
        dir_list = glob(data_path + "{}/*".format(n_subdir * "/*"), recursive=True)
        for f in dir_list:
            if os.path.isfile(f):
                fname = os.path.basename(f)
                # f_abs_path = d_path + "/{}".format(fname)
                f_dst_path = dst_path + "/{}".format(fname)
                shutil.move(f, f_dst_path)
                # print("{} --> {}".format(f, f_dst_path))
    else:
        dir_list = os.listdir(data_path)
        for d in dir_list:
            d_path = data_path + "/{}".format(d)
            d_list = os.listdir(d_path)
            for f in d_list:
                f_abs_path = d_path + "/{}".format(f)
                f_dst_path = dst_path + "/{}".format(f)
                shutil.move(f_abs_path, f_dst_path)
                # print("{} --> {}".format(f_abs_path, f_dst_path))

    shutil.rmtree(data_path)


def random_select_files(data_path, select_num=1000, move_or_copy="copy", select_mode=0):
    data_list = sorted(os.listdir(data_path))
    data_dir_name = os.path.basename(data_path)

    if select_mode == 0:
        selected = random.sample(data_list, select_num)
        save_path = os.path.abspath(os.path.join(data_path, "../")) + "/Random_Selected/{}_random_selected_{}".format(data_dir_name, select_num)
        os.makedirs(save_path, exist_ok=True)
    else:
        selected = random.sample(data_list, len(data_list) - select_num)
        save_path = os.path.abspath(os.path.join(data_path, "../")) + "/Random_Selected/{}_random_selected_{}".format(data_dir_name, len(data_list) - select_num)
        os.makedirs(save_path, exist_ok=True)

    for s in tqdm(selected):
        f_src_path = data_path + "/{}".format(s)
        f_dst_path = save_path + "/{}".format(s)

        if move_or_copy == "copy":
            shutil.copy(f_src_path, f_dst_path)
        elif move_or_copy == "move":
            shutil.move(f_src_path, f_dst_path)
        else:
            print("Error!")


def copy_file_exist_corresponding_file(dir1, dir2):
    """
    :param dir1:
    :param dir2:
    :return:
    """
    file1_list = [i.split(".")[0] for i in os.listdir(dir1)]
    file2_list = [i.split(".")[0] for i in os.listdir(dir2)]

    same_files = set(file1_list) & set(file2_list)

    same_dir1_path = os.path.abspath(os.path.join(dir1, "../..")) + "/New/{}".format(os.path.basename(dir1))
    same_dir2_path = os.path.abspath(os.path.join(dir1, "../..")) + "/New/{}".format(os.path.basename(dir2))
    os.makedirs(same_dir1_path, exist_ok=True)
    os.makedirs(same_dir2_path, exist_ok=True)

    for j in same_files:
        try:
            dir1_file_ends = os.path.splitext(os.listdir(dir1)[0])[1]
            dir2_file_ends = os.path.splitext(os.listdir(dir2)[0])[1]

            # dir1
            dir1_file_abs_path = dir1 + "/{}{}".format(j, dir1_file_ends)
            dir1_file_dst_path = same_dir1_path + "/{}{}".format(j, dir1_file_ends)
            shutil.copy(dir1_file_abs_path, dir1_file_dst_path)
            print("shutil.copy: {} --> {}".format(dir1_file_abs_path, dir1_file_dst_path))

            # dir2
            dir2_file_abs_path = dir2 + "/{}{}".format(j, dir2_file_ends)
            dir2_file_dst_path = same_dir2_path + "/{}{}".format(j, dir2_file_ends)
            shutil.copy(dir2_file_abs_path, dir2_file_dst_path)
            print("shutil.copy: {} --> {}".format(dir2_file_abs_path, dir2_file_dst_path))

        except Exception as Error:
            print(Error)


def select_file_exist_corresponding_file(base_path):
    dir1 = base_path + "/images"
    dir2 = base_path + "/labels"
    copy_file_exist_corresponding_file(dir1, dir2)


def select_same_files(dir1, dir2, select_dir="dir1"):
    file1_list = [i.split(".")[0] for i in os.listdir(dir1)]
    file2_list = [i.split(".")[0] for i in os.listdir(dir2)]

    dir1_name = os.path.basename(dir1)
    dir2_name = os.path.basename(dir2)

    same_files = set(file1_list) & set(file2_list)
    if select_dir == "dir1":
        same_dir_path = os.path.abspath(os.path.join(dir1, "../..")) + "/{}_same_files".format(dir1_name)
        os.makedirs(same_dir_path, exist_ok=True)
    elif select_dir == "dir2":
        same_dir_path = os.path.abspath(os.path.join(dir1, "../..")) + "/{}_same_files".format(dir2_name)
        os.makedirs(same_dir_path, exist_ok=True)
    else:
        print("'select_dir' should be dir1 or dir2.")

    dir1_file_ends = os.path.splitext(os.listdir(dir1)[0])[1]

    for j in same_files:
        try:
            # dir1
            if select_dir == "dir1":
                same_file_abs_path = dir1 + "/{}{}".format(j, dir1_file_ends)
                same_file_dst_path = same_dir_path + "/{}{}".format(j, dir1_file_ends)
                shutil.copy(same_file_abs_path, same_file_dst_path)
                print("shutil.copy: {} --> {}".format(same_file_abs_path, same_file_dst_path))
            elif select_dir == "dir2":
                same_file_abs_path = dir2 + "/{}{}".format(j, dir1_file_ends)
                same_file_dst_path = same_dir_path + "/{}{}".format(j, dir1_file_ends)
                shutil.copy(same_file_abs_path, same_file_dst_path)
                print("shutil.copy: {} --> {}".format(same_file_abs_path, same_file_dst_path))
            else:
                print("'select_dir' should be dir1 or dir2.")
        except Exception as Error:
            print(Error)


def move_or_delete_file_not_exist_corresponding_file(base_path, dir1_name="images", dir2_name="labels", labelbee_json_label=False, move_or_delete="delete", dir="dir2"):
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

    unexpected = list(set(file1_list) ^ set(file2_list))

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

                    # file_abs_path = dir1 + "/{}.jpeg".format(j, dir1_file_ends)
                    # file_dst_path = unexpected_path + "/{}.jpeg".format(j, dir1_file_ends)
                    # try:
                    #     shutil.move(file_abs_path, file_dst_path)
                    #     print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))
                    # except Exception as Error:
                    #     print(Error)

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
                    file_abs_path = dir2 + "/{}.jpg{}".format(j, dir2_file_ends)
                    file_dst_path = unexpected_path + "/{}.jpg{}".format(j, dir2_file_ends)
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


def move_or_delete_file_not_exist_corresponding_file_under_diffirent_dir(dir_path1="images", dir_path2="labels", unexpected_path="", move_or_delete="delete", dir="dir2"):
    """
    :param dir_path1:
    :param dir_path2:
    :param move_or_delete: "move" or "delete"
    :param dir: files in which dir will be move or delete
    :return:
    """
    # dir1 = base_path + "/{}".format(dir1_name)
    # dir2 = base_path + "/{}".format(dir2_name)
    file1_list = [os.path.splitext(i)[0] for i in os.listdir(dir_path1)]
    file2_list = [os.path.splitext(i)[0] for i in os.listdir(dir_path2)]

    unexpected = set(file1_list) ^ set(file2_list)

    if move_or_delete == "move":
        if unexpected_path is None:
            unexpected_path = os.path.abspath(os.path.join(dir_path1, "../..")) + "/unexpected"
            os.makedirs(unexpected_path, exist_ok=True)
        else:
            os.makedirs(unexpected_path, exist_ok=True)

    for j in unexpected:
        try:
            dir1_file_ends = os.path.splitext(os.listdir(dir_path1)[0])[1]
            dir2_file_ends = os.path.splitext(os.listdir(dir_path2)[0])[1]

            if move_or_delete == "delete":
                if dir == "dir1":
                    file_abs_path = dir_path1 + "/{}{}".format(j, dir1_file_ends)
                    os.remove(file_abs_path)
                    print("os.remove: --> {}".format(file_abs_path))
                else:
                    file_abs_path = dir_path2 + "/{}{}".format(j, dir2_file_ends)
                    os.remove(file_abs_path)
                    print("os.remove: --> {}".format(file_abs_path))
            # move
            else:
                if dir == "dir1":
                    file_abs_path = dir_path1 + "/{}{}".format(j, dir1_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir1_file_ends)
                    shutil.move(file_abs_path, file_dst_path)
                    print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))
                else:
                    file_abs_path = dir_path2 + "/{}{}".format(j, dir2_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir2_file_ends)
                    shutil.move(file_abs_path, file_dst_path)
                    print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))

        except Exception as Error:
            print(Error)


def move_or_delete_file_not_exist_corresponding_file_under_diffirent_dir_v2(dir_path1="images", dir_path2="labels", unexpected_path="", move_or_delete="delete"):
    """
    :param dir_path1:
    :param dir_path2:
    :param move_or_delete: "move" or "delete"
    :param dir: files in which dir will be move or delete
    :return:
    """
    # dir1 = base_path + "/{}".format(dir1_name)
    # dir2 = base_path + "/{}".format(dir2_name)
    # file1_list = [os.path.splitext(i)[0] for i in os.listdir(dir_path1)]
    # file2_list = [os.path.splitext(i)[0] for i in os.listdir(dir_path2)]

    file1_list = sorted(list(set([i.split("_")[0] for i in os.listdir(dir_path1)])))
    file2_list = sorted([i.split(".")[0] for i in os.listdir(dir_path2)])

    unexpected = set(file1_list) ^ set(file2_list)

    if move_or_delete == "move":
        if unexpected_path is None:
            unexpected_path = os.path.abspath(os.path.join(dir_path1, "../..")) + "/unexpected"
            os.makedirs(unexpected_path, exist_ok=True)
        else:
            os.makedirs(unexpected_path, exist_ok=True)

    for j in unexpected:
        try:
            dir1_file_ends = os.path.splitext(os.listdir(dir_path1)[0])[1]
            dir2_file_ends = os.path.splitext(os.listdir(dir_path2)[0])[1]

            if move_or_delete == "delete":
                file_abs_path = dir_path2 + "/{}{}".format(j, dir2_file_ends)
                os.remove(file_abs_path)
                print("os.remove: --> {}".format(file_abs_path))
            # move
            else:
                file_abs_path = dir_path2 + "/{}{}".format(j, dir2_file_ends)
                file_dst_path = unexpected_path + "/{}{}".format(j, dir2_file_ends)
                shutil.move(file_abs_path, file_dst_path)
                print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))

        except Exception as Error:
            print(Error)


def move_or_delete_file_not_exist_corresponding_expand_ratio_cropped_images(dir_path1="images", dir_path2="labels", unexpected_path="", move_or_delete="delete"):
    dir1_name = os.path.basename(dir_path1)
    dir2_name = os.path.basename(dir_path2)

    file1_list = sorted(list(set([i[:-7] for i in os.listdir(dir_path1)])))
    file2_list = sorted(list(set([i[:-7] for i in os.listdir(dir_path2)])))

    unexpected = set(file1_list) ^ set(file2_list)

    if move_or_delete == "move":
        if unexpected_path == "":
            unexpected_path = os.path.abspath(os.path.join(dir_path1, "../..")) + "/unexpected"
            os.makedirs(unexpected_path, exist_ok=True)
        else:
            os.makedirs(unexpected_path, exist_ok=True)

    for j in unexpected:
        try:
            dir1_file_ends = os.path.splitext(os.listdir(dir_path1)[0])[1]
            dir2_file_ends = os.path.splitext(os.listdir(dir_path2)[0])[1]

            if move_or_delete == "delete":
                file_abs_path = dir_path2 + "/{}{}{}".format(j, dir2_name, dir2_file_ends)
                os.remove(file_abs_path)
                print("os.remove: --> {}".format(file_abs_path))
            # move
            else:
                file_abs_path = dir_path2 + "/{}{}{}".format(j, dir2_name, dir2_file_ends)
                file_dst_path = unexpected_path + "/{}{}{}".format(j, dir2_name, dir2_file_ends)
                shutil.move(file_abs_path, file_dst_path)
                print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))

        except Exception as Error:
            print(Error)


def move_or_delete_file_exist_corresponding_file(base_path, dir1_name="images", dir2_name="labels", move_or_delete="delete", dir="dir2"):
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
    file2_list = [os.path.splitext(i)[0] for i in os.listdir(dir2)]

    unexpected = set(file1_list) & set(file2_list)

    if move_or_delete == "move" or move_or_delete == "copy":
        unexpected_path = os.path.abspath(os.path.join(dir1, "../..")) + "/unexpected"
        os.makedirs(unexpected_path, exist_ok=True)

    for j in unexpected:
        try:
            dir1_file_ends = os.path.splitext(os.listdir(dir1)[0])[1]
            dir2_file_ends = os.path.splitext(os.listdir(dir2)[0])[1]

            if move_or_delete == "delete":
                if dir == "dir1":
                    # file_abs_path = dir1 + "/{}{}".format(j, dir1_file_ends)
                    file_abs_path = dir1 + "/{}{}".format(j, dir1_file_ends)
                    os.remove(file_abs_path)
                    print("os.remove: --> {}".format(file_abs_path))
                else:
                    file_abs_path = dir2 + "/{}{}".format(j, dir2_file_ends)
                    os.remove(file_abs_path)
                    print("os.remove: --> {}".format(file_abs_path))
            elif move_or_delete == "move":
                if dir == "dir1":
                    file_abs_path = dir1 + "/{}{}".format(j, dir1_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir1_file_ends)
                    shutil.move(file_abs_path, file_dst_path)
                    print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))
                else:
                    file_abs_path = dir2 + "/{}{}".format(j, dir2_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir2_file_ends)
                    shutil.move(file_abs_path, file_dst_path)
                    print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))
            elif move_or_delete == "copy":
                if dir == "dir1":
                    file_abs_path = dir1 + "/{}{}".format(j, dir1_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir1_file_ends)
                    shutil.copy(file_abs_path, file_dst_path)
                    print("shutil.copy: {} --> {}".format(file_abs_path, file_dst_path))
                else:
                    file_abs_path = dir2 + "/{}{}".format(j, dir2_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir2_file_ends)
                    shutil.copy(file_abs_path, file_dst_path)
                    print("shutil.copy: {} --> {}".format(file_abs_path, file_dst_path))
            else:
                print("'move_or_delete' Error")

        except Exception as Error:
            print(Error)


def move_or_delete_file_exist_corresponding_file_under_diffirent_dir(dir_path1="images", dir_path2="labels", unexpected_path="", flag="cp", dir="dir2"):
    """
    :param dir1:
    :param dir2:
    :param move_or_delete: "move" or "delete"
    :param dir: files in which dir will be move or delete
    :return:
    """
    # dir1 = base_path + "/{}".format(dir1_name)
    # dir2 = base_path + "/{}".format(dir2_name)
    # file1_list = [os.path.splitext(i)[0] for i in os.listdir(dir1)]
    # file2_list = [os.path.splitext(i)[0] for i in os.listdir(dir2)]
    file1_list = [os.path.splitext(i)[0] for i in os.listdir(dir_path1)]
    file2_list = [os.path.splitext(i)[0] for i in os.listdir(dir_path2)]

    unexpected = set(file1_list) & set(file2_list)

    if flag == "move" or flag == "copy" or flag == "mv" or flag == "cp":
        if unexpected_path == "":
            unexpected_path = os.path.abspath(os.path.join(dir_path1, "../..")) + "/milk_tea_cup/same_files"
            os.makedirs(unexpected_path, exist_ok=True)
        else:
            os.makedirs(unexpected_path, exist_ok=True)

    dir1_file_ends = os.path.splitext(os.listdir(dir_path1)[0])[1]
    dir2_file_ends = os.path.splitext(os.listdir(dir_path2)[0])[1]

    for j in unexpected:
        try:
            if flag == "delete" or flag == "del" or flag == "rm" or flag == "remove":
                if dir == "dir1":
                    # file_abs_path = dir1 + "/{}{}".format(j, dir1_file_ends)
                    file_abs_path = dir_path1 + "/{}{}".format(j, dir1_file_ends)
                    os.remove(file_abs_path)
                    print("os.remove: --> {}".format(file_abs_path))
                else:
                    file_abs_path = dir_path2 + "/{}{}".format(j, dir2_file_ends)
                    os.remove(file_abs_path)
                    print("os.remove: --> {}".format(file_abs_path))
            # copy
            elif flag == "copy" or flag == "cp":
                if dir == "dir1":
                    file_abs_path = dir_path1 + "/{}{}".format(j, dir1_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir1_file_ends)
                    shutil.copy(file_abs_path, file_dst_path)
                    print("shutil.copy: {} --> {}".format(file_abs_path, file_dst_path))
                else:
                    file_abs_path = dir_path2 + "/{}{}".format(j, dir2_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir2_file_ends)
                    shutil.copy(file_abs_path, file_dst_path)
                    print("shutil.copy: {} --> {}".format(file_abs_path, file_dst_path))
            # move
            else:
                if dir == "dir1":
                    file_abs_path = dir_path1 + "/{}{}".format(j, dir1_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir1_file_ends)
                    shutil.move(file_abs_path, file_dst_path)
                    print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))
                else:
                    file_abs_path = dir_path2 + "/{}{}".format(j, dir2_file_ends)
                    file_dst_path = unexpected_path + "/{}{}".format(j, dir2_file_ends)
                    shutil.move(file_abs_path, file_dst_path)
                    print("shutil.move: {} --> {}".format(file_abs_path, file_dst_path))

        except Exception as Error:
            print(Error)


def remove_specific_file_by_name(data_path, key_words, mode=0):
    file_list = sorted(os.listdir(data_path))

    for f in file_list:
        f_abs_path = data_path + "/{}".format(f)
        try:
            if mode == 0:
                for i in range(len(key_words)):
                    if key_words[i] in f:
                        os.remove(f_abs_path)
                        print("Removed --> {}".format(f_abs_path))
            else:
                for i in range(len(key_words)):
                    if key_words[i] not in f:
                        os.remove(f_abs_path)
                        print("Removed --> {}".format(f_abs_path))

        except Exception as Error:
            print(Error)


def select_specific_file_by_name(data_path, key_words, mode=0, cp_or_mv="move"):
    file_list = sorted(os.listdir(data_path))
    data_dir_name = os.path.basename(data_path)

    if mode == 0:
        save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_selected_contain_key_words_{}".format(data_dir_name, key_words[0])
    else:
        save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_selected_not_contain_key_words_{}".format(data_dir_name, key_words[0])
    os.makedirs(save_path, exist_ok=True)

    for f in file_list:
        f_abs_path = data_path + "/{}".format(f)
        try:
            if mode == 0:
                for i in range(len(key_words)):
                    if key_words[i] in f:
                        f_dst_path = save_path + "/{}".format(f)
                        if cp_or_mv == "copy" or cp_or_mv == "cp":
                            shutil.copy(f_abs_path, f_dst_path)
                        elif cp_or_mv == "move" or cp_or_mv == "mv":
                            shutil.move(f_abs_path, f_dst_path)
                        break
            else:
                flag = False
                for i in range(len(key_words)):
                    if key_words[i] in f:
                        # f_dst_path = save_path + "/{}".format(f)
                        # shutil.copy(f_abs_path, f_dst_path)
                        # break
                        flag = True
                        break

                if not flag:
                    f_dst_path = save_path + "/{}".format(f)
                    if cp_or_mv == "copy" or cp_or_mv == "cp":
                        shutil.copy(f_abs_path, f_dst_path)
                    elif cp_or_mv == "move" or cp_or_mv == "mv":
                        shutil.move(f_abs_path, f_dst_path)

        except Exception as Error:
            print(Error)


def remove_specific_file_by_name_index(data_path, key_index=5):
    file_list = sorted(os.listdir(data_path))

    for f in file_list:
        file_index = int(os.path.splitext(f)[0].split("_")[-1])
        file_abs_path = data_path + "/{}".format(f)
        try:
            if file_index % key_index != 0:
                os.remove(file_abs_path)
                print("Removed --> {}".format(file_abs_path))

        except Exception as Error:
            print(Error)


def copy_n_times(data_path, n=10, save_path="current", print_flag=True):
    data_list = sorted(os.listdir(data_path))

    dir_name = os.path.basename(data_path)
    if save_path == "current":
        save_path = data_path
    else:
        save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_copyed_{}_times".format(dir_name, n)
        os.makedirs(save_path, exist_ok=True)

    for f in tqdm(data_list):
        f_name, f_suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
        if f_suffix != ".json":
            f_abs_path = data_path + "/{}".format(f)
            f_dst_names = []
            for i in range(n):
                f_dst_names.append("{}_cp{}{}".format(f_name, i + 1, f_suffix))

            for j in f_dst_names:
                f_dst_path = save_path + "/{}".format(j)
                shutil.copy(f_abs_path, f_dst_path)
                if print_flag:
                    print("{} --> {}".format(f_abs_path, f_dst_path))
        # labelbee json files
        else:
            f_name, f_suffix = os.path.splitext(f_name)[0], os.path.splitext(f_name)[1] + ".json"
            f_abs_path = data_path + "/{}".format(f)
            f_dst_names = []
            for i in range(n):
                f_dst_names.append("{}_cp{}{}".format(f_name, i + 1, f_suffix))

            for j in f_dst_names:
                f_dst_path = save_path + "/{}".format(j)
                shutil.copy(f_abs_path, f_dst_path)
                if print_flag:
                    print("{} --> {}".format(f_abs_path, f_dst_path))


def get_sub_dir_file_list(base_path):
    """
    :param base_path:
    :return: file abs path
    """
    all_files = []
    dir_list = sorted(os.listdir(base_path))
    for d in dir_list:
        d_abs_path = base_path + "/{}".format(d)
        file_list = sorted(os.listdir(d_abs_path))
        for f in file_list:
            f_abs_path = d_abs_path + "/{}".format(f)
            all_files.append(f_abs_path)

    return all_files


def get_sub_dir_list(base_path):
    all_dirs = []
    dir_list = sorted(os.listdir(base_path))
    for d in dir_list:
        d_abs_path = base_path + "/{}".format(d)
        all_dirs.append(d_abs_path)

    return all_dirs


def copy_file_according_txt(txt_path="", save_path=""):
    os.makedirs(save_path, exist_ok=True)

    data = open(txt_path, "r", encoding="utf-8")
    lines = data.readlines()
    data.close()

    for l in lines:
        f_abs_path = l.strip()
        fname = os.path.basename(f_abs_path)
        f_dst_path = save_path + "/{}".format(fname)

        shutil.copy(f_abs_path, f_dst_path)


def copy_file_by_name(data_path, key_words, mode):
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_copyed_by_name".format(dir_name)
    os.makedirs(save_path, exist_ok=True)

    file_list = sorted(os.listdir(data_path))

    for f in file_list:
        f_abs_path = data_path + "/{}".format(f)
        try:
            if mode == 0:
                for i in range(len(key_words)):
                    if key_words[i] in f:
                        f_dst_path = save_path + "/{}".format(f)
                        shutil.copy(f_abs_path, f_dst_path)
                        print("Copyed --> {}".format(f_dst_path))
            else:
                for i in range(len(key_words)):
                    if key_words[i] not in f:
                        f_dst_path = save_path + "/{}".format(f)
                        shutil.copy(f_abs_path, f_dst_path)
                        print("Copyed --> {}".format(f_dst_path))

        except Exception as Error:
            print(Error)


def return_json_data(data_path, dt, d):
    if not d:
        json_data = json.load(open(data_path + "/annotations/tiny_set_{}.json".format(dt), "r", encoding="utf-8"))
        return json_data
    else:
        json_data = json.load(open(data_path + "/annotations/tiny_set_{}_with_dense.json".format(dt), "r", encoding="utf-8"))
        return json_data


def split_dir(data_path, split_n=5):
    """
    If a directory contains large amount of files, then split to split_n dirs.
    :param data_path:
    :param split_n:
    :return:
    """
    dir_name = os.path.basename(data_path)
    for i in range(split_n):
        save_path_i = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_{:03d}".format(dir_name, i)
        os.makedirs(save_path_i, exist_ok=True)

    file_list = sorted(os.listdir(data_path))
    len_ = len(file_list)

    file_lists = []
    for j in range(split_n):
        file_lists.append(file_list[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

    for i, files in enumerate(file_lists):
        for f in files:
            f_abs_path = data_path + "/{}".format(f)
            f_name = os.path.splitext(f)[0]
            save_path_i = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_{:03d}".format(dir_name, i)
            f_dst_path = save_path_i + "/{}".format(f)
            shutil.move(f_abs_path, f_dst_path)


def split_dir_base(i, file_list, data_path, save_path):
    dir_name = os.path.basename(data_path)
    save_path_i = save_path + "/{}_{:03d}".format(dir_name, i)
    os.makedirs(save_path_i, exist_ok=True)
    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        f_name = os.path.splitext(f)[0]
        f_dst_path = save_path_i + "/{}".format(f)
        shutil.move(f_abs_path, f_dst_path)


def split_dir_multithread(data_path, split_n=8):
    dir_name = os.path.basename(data_path)
    img_list = os.listdir(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "../..")) + "/{}_split_{}_dirs".format(dir_name, split_n)
    os.makedirs(save_path, exist_ok=True)

    len_ = len(img_list)

    img_lists = []
    for j in range(split_n):
        img_lists.append(img_list[int(len_ * (j / split_n)):int(len_ * ((j + 1) / split_n))])

    t_list = []
    for i in range(split_n):
        list_i = img_lists[i]
        t = threading.Thread(target=split_dir_base, args=(i, list_i, data_path, save_path,))
        t_list.append(t)

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()


def get_file_type(file_name, max_len=16):
    """

    :param file_name:
    :param max_len:
    :return:
    """
    with open(file_name, "rb") as fo:
        byte = fo.read(max_len)

    byte_list = struct.unpack('B' * max_len, byte)
    code = ''.join([('%X' % each).zfill(2) for each in byte_list])

    return code


def change_Linux_conda_envs_bin_special_files_content(conda_envs_path):
    """
    Can work!
    :param conda_envs_path:
    :return:
    """
    replace_str = "#!{}".format(conda_envs_path)

    file_list = sorted(os.listdir(conda_envs_path))
    for f in file_list:
        f_abs_path = conda_envs_path + "/{}".format(f)
        byte = get_file_type(f_abs_path)
        # print("{}: {}".format(f, byte))

        if byte == "23212F686F6D652F7A656E6779696661":  # 23212F686F6D652F77756A696168752F, 23212F686F6D652F6C69757A68656E78, 23212F686F6D652F7A656E6779696661
            with open(f_abs_path, "r+", encoding="utf-8") as fo:
                lines = fo.readlines()
                lines0_cp = lines[0].strip()
                lines0_split_python = lines0_cp.split("/python")
                if lines0_split_python[0] != replace_str:
                    if len(lines0_split_python) < 2:
                        lines[0] = replace_str + "/python\n"
                        print("{}: {} --> {}/python".format(f, lines0_cp, replace_str))
                    else:
                        lines[0] = replace_str + "/python{}\n".format(lines0_split_python[1])
                        print("{}: {} --> {}/python{}".format(f, lines0_cp, replace_str, lines0_split_python[1]))

            with open(f_abs_path, "w", encoding="utf-8") as fw:
                fw.writelines(lines)


def dict_save_to_file(data_path, flag="pickle"):
    """

    :param data_path:
    :param flag:
    :return:
    """
    file_list = sorted(os.listdir(data_path))
    list_dict = {}
    for i, f in tqdm(enumerate(file_list)):
        if str(i) not in list_dict.keys():
            list_dict[str(i)] = f

    if flag == "pickle":
        with open("10010_list_dict.pickle", "wb") as fw:
            pickle.dump(list_dict, fw)
    elif flag == "numpy":
        np.save("10010_list_dict.npy", list_dict)
    elif flag == "json":
        with open("10010_list_dict.json", "w", encoding="utf-8") as fw:
            json.dump(list_dict, fw)
    else:
        print("Please enter one of pickle, numpy or json!")


def load_saved_dict_file(file_path):
    """

    :param file_path:
    :param flag:
    :return:
    """
    if file_path.endswith("pickle"):
        with open(file_path, "rb") as fr:
            dict_ = pickle.load(fr)
        return dict_.items()
    elif file_path.endswith("npy"):
        dict_ = np.load(file_path, allow_pickle=True).item()
        return dict_
    elif file_path.endswith("json"):
        with open(file_path, "r", encoding="utf-8") as fr:
            dict_ = json.load(fr)
        return dict_
    else:
        print("Please input one of pickle, numpy or json file!")


def compare_two_dicts(file_path1, file_path2):
    dict_data1 = load_saved_dict_file(file_path1)
    dict_data2 = load_saved_dict_file(file_path2)
    list1 = list(dict_data1.values())
    list2 = list(dict_data2.values())
    diff = set(list1) ^ set(list2)
    print(diff)


def merge_txt(path1, path2):
    txt_list1 = sorted(os.listdir(path1))
    txt_list2 = sorted(os.listdir(path2))

    same_files = list((set(txt_list1) & set(txt_list2)))

    for f in tqdm(same_files):
        f1_abs_path = path1 + "/{}".format(f)
        f2_abs_path = path2 + "/{}".format(f)

        with open(f1_abs_path, "r", encoding="utf-8") as fr1:
            f1_lines = fr1.readlines()

        with open(f2_abs_path, "r", encoding="utf-8") as fr2:
            f2_lines = fr2.readlines()

        with open(f1_abs_path, "a", encoding="utf-8") as fa1:
            for l2 in f2_lines:
                if l2 not in f1_lines:
                    fa1.write(l2)

                    print("{} --> {}".format(l2.strip(), f1_abs_path))

        print("----------------------------")


def calculate_md5(file_path):
    with open(file_path, "rb") as file:
        data = file.read()

    md5_hash = hashlib.md5()
    md5_hash.update(data)
    md5_value = md5_hash.hexdigest()

    return md5_value


def read_csv(file_path):
    csv_data = pd.read_csv(file_path)
    return csv_data.values


def split_dir_by_file_suffix(data_path):
    save_path = make_save_path(data_path, "splited_by_file_suffix")

    suffixes = []
    file_list = get_file_list(data_path)
    for f in file_list:
        file_name, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
        if suffix not in suffixes:
            suffixes.append(suffix)

    for s in suffixes:
        if s != "":
            s_save_path = save_path + "/{}".format(s.replace(".", ""))
            os.makedirs(s_save_path, exist_ok=True)

    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        file_name, suffix = os.path.splitext(f)[0], os.path.splitext(f)[1]
        if suffix != "":
            f_dst_path = save_path + "/{}/{}".format(suffix.replace(".", ""), f)
            shutil.move(f_abs_path, f_dst_path)


def isAllDigits(string):
    pattern = r'^\d+$'
    if re.match(pattern, string):
        return True
    else:
        return False


def isAllChinese(string):
    pattern = '[\u4e00-\u9fa5]+'
    if re.match(pattern, string) and len(string) == len(set(string)):
        return True
    else:
        return False


def findSubStrIndex(substr, str, time):
    """
    # 找字符串substr在str中第time次出现的位置
    """
    times = str.count(substr)
    if (times == 0) or (times < time):
        pass
    else:
        i = 0
        index = -1
        while i < time:
            index = str.find(substr, index+1)
            i += 1
        return index


def change_console_str_color():
    """
    @Time: 2021/1/22 21:16
    @Author: gracekafuu
    https://blog.csdn.net/qq_34857250/article/details/79673698

    """

    print('This is a \033[1;35m test \033[0m!')
    print('This is a \033[1;32;43m test \033[0m!')
    print('\033[1;33;44mThis is a test !\033[0m')


def remove_Thumbs(img_path):
    thumbs = img_path + "/Thumbs.db"
    if os.path.exists(thumbs):
        os.remove(thumbs)
        print("Removed --> {}".format(thumbs))


def remove_list_repeat_elements(list1):
    list2 = []
    [list2.append(i) for i in list1 if i not in list2]

    return list2


def rename_files(imgPath, start_idx):
    imgList = sorted(os.listdir(imgPath))
    for i in range(len(imgList)):
        imgAbsPath = imgPath + "\\" + imgList[i]
        ends = os.path.splitext(imgList[i])[1]
        newName = "{:08d}{}".format(i + start_idx, ends)
        os.rename(imgAbsPath, imgPath + "\\" + newName)

    print("Renamed!")


def udp_send_txt_content(txtfile, client="127.0.0.1", port=60015):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    with open(txtfile) as f:
        msgs = f.readlines()

    while True:
        for msg in msgs:
            msg = msg.strip().replace("\\", "/")
            if not msg: break
            sock.sendto(bytes(msg, "utf-8"), (client, port))
            print("UDP sent: {}".format(msg))
            time.sleep(.0001)
        sock.close()


def majority_element(arr):
    if arr == []:
        return None
    else:
        dict_ = {}
        for key in arr:
            dict_[key] = dict_.get(key, 0) + 1
        maybe_maj_element = max(dict_, key=lambda k: dict_[k])
        maybe_maj_key = [k for k, v in dict_.items() if v == dict_[maybe_maj_element]]

        if len(maybe_maj_key) == 1:
            maj_element = maybe_maj_element
            return maj_element
        else:
            return None


def second_majority_element(arr, remove_first_mj):
    for i in range(len(arr)):
        if remove_first_mj in arr:
            arr.remove(remove_first_mj)
    if arr != []:
        second_mj = majorityElement_v2(arr)
        return second_mj
    else:
        return None


def find_chinese(chars):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', chars)
    return chinese


def RANSAC_fit_2Dline(X_data, Y_data, iters=100000, sigma=0.25, pretotal=0, P=0.99):
    """

    Parameters
    ----------
    X
    Y
    # 使用RANSAC算法估算模型
    # 迭代最大次数，每次得到更好的估计会优化iters的数值
    iters = 100000
    # 数据和模型之间可接受的差值
    sigma = 0.25
    # 最好模型的参数估计和内点数目
    best_a = 0
    best_b = 0
    pretotal = 0
    # 希望的得到正确模型的概率
    P = 0.99
    Returns
    -------

    """

    SIZE = X_data.shape[0]

    best_a = 0
    best_b = 0

    for i in range(iters):
        # 随机在数据中红选出两个点去求解模型
        # sample_index = random.sample(range(SIZE), 2)
        sample_index = random.choices(range(SIZE), k=2)
        x_1 = X_data[sample_index[0]]
        x_2 = X_data[sample_index[1]]
        y_1 = Y_data[sample_index[0]]
        y_2 = Y_data[sample_index[1]]

        # y = ax + b 求解出a，b
        try:
            a = (y_2 - y_1) / ((x_2 - x_1) + 1e-2)
            b = y_1 - a * x_1
        except Exception as Error:
            print("RANSAC_fit_2Dline: a = (y_2 - y_1) / (x_2 - x_1) --> {}".format(Error))

        # 算出内点数目
        total_inlier = 0
        for index in range(SIZE):
            y_estimate = a * X_data[index] + b
            if abs(y_estimate - Y_data[index]) < sigma:
                total_inlier = total_inlier + 1

        # 判断当前的模型是否比之前估算的模型好
        if total_inlier > pretotal:
            # iters = math.log(1 - P) / math.log(1 - pow(total_inlier / (SIZE), 2))
            pretotal = total_inlier
            best_a = a
            best_b = b

        # 判断是否当前模型已经符合超过一半的点
        if total_inlier > SIZE // 2:
            break

    return best_a, best_b


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


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # 限制最小的值
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    # 一个圆对应内切正方形的高斯分布

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        # 将高斯分布覆盖到heatmap上，取最大，而不是叠加
    return heatmap


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def merge_mdb_files(source_files, target_file):
    # 打开目标数据库（用于存储合并后的数据）
    target_db = AccessDatabase(target_file, create_if_missing=True)

    # 遍历所有源文件
    for source_file in source_files:
        source_db = AccessDatabase(source_file)

        # 遍历所有表
        for table_name in source_db.table_names:
            # 读取表数据
            table_data = source_db.read_table(table_name)

            # 将表数据写入目标数据库
            target_db.write_table(table_name, table_data)


def merge_txt_files(data_path):
    file_list = get_file_list(data_path)
    merged_txt_path = data_path + "/merged.txt"
    fw = open(merged_txt_path, "w", encoding="utf-8")

    for f in file_list:
        f_path = data_path + "/{}".format(f)
        if os.path.isfile(f_path) and f_path.endswith(".txt"):
            with open(f_path, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
                fw.writelines(lines)
    fw.close()


def ocr_data_gen_train_txt(data_path, LABEL):
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
            if l not in LABEL:
                num_ += 1

        if os.path.exists(f) and num_ == 0:
            content = "{} {}\n".format(f, label)
            fw.write(content)

    fw.close()


def ocr_data_gen_train_txt_v2(data_path, LABEL):
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


def random_select_files_according_txt(data_path, select_percent):
    assert os.path.isfile(data_path) and data_path.endswith(".txt"), "{} should be *.txt"
    save_path = data_path.replace(".txt", "_random_selected_{}_percent.txt".format(select_percent))

    fr = open(data_path, "r", encoding="utf-8")
    lines = fr.readlines()
    fr.close()

    fw = open(save_path, "w", encoding="utf-8")

    num = int(len(lines) * select_percent)
    selected = random.sample(lines, num)
    for l in selected:
        fw.write(l)

    fw.close()


def read_ocr_lables(lbl_path):
    CH_SIM_CHARS = ' '
    ch_sim_chars = open(lbl_path, "r", encoding="utf-8")
    lines = ch_sim_chars.readlines()
    for l in lines:
        CH_SIM_CHARS += l.strip()
    alpha = CH_SIM_CHARS  # len = 1 + 6867 = 6868
    return alpha



if __name__ == '__main__':
    pass

































