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
import sys
import cv2
import time
import json
import glob
import random
import shutil
import struct
import pickle
import socket
import logging
import hashlib
import zipfile
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm


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
    # print("Create successful! save_path: {}".format(save_path))
    return save_path


def rename_files(data_path, use_orig_name=False, new_name_prefix="", zeros_num=7, start_num=0):
    data_list = sorted(os.listdir(data_path))
    length= len(data_list)
    for i in range(length):
        img_abs_path = data_path + "/" + data_list[i]
        orig_name = os.path.splitext(data_list[i])[0]
        file_ends = os.path.splitext(data_list[i])[1]
        if use_orig_name:
            new_name = "{}_{:0{}d}{}".format(orig_name, i + start_num, zeros_num, file_ends)
            os.rename(img_abs_path, data_path + "/" + new_name)
        else:
            if new_name_prefix is None or new_name_prefix == "":
                new_name = "{:0{}d}{}".format(i + start_num, zeros_num, file_ends)
            else:
                new_name = "{}_{:0{}d}{}".format(new_name_prefix, i + start_num, zeros_num, file_ends)
            os.rename(img_abs_path, data_path + "/" + new_name)


def save_file_path_to_txt(data_path: str, abspath=True):
    assert type(data_path) == str, "{} should be str!".format(data_path)
    dirname = os.path.basename(data_path)
    data_list = sorted(os.listdir(data_path))
    txt_save_path = os.path.abspath(os.path.join(data_path, "../{}_list.txt".format(dirname)))
    with open(txt_save_path, 'w', encoding='utf-8') as fw:
        for f in data_list:
            if abspath:
                f_abs_path = data_path + "/{}".format(f)
                f_abs_path = f_abs_path.replace("\\", "/")
                fw.write("{}\n".format(f_abs_path))
            else:
                fw.write("{}\n".format(f))

    print("Success! --> {}".format(txt_save_path))


def save_all_files_path_to_txt(data_path: str, abspath=True):
    assert type(data_path) == str, "{} should be str!".format(data_path)
    dirname = os.path.basename(data_path)
    dirs = sorted(os.listdir(data_path))
    txt_save_path = os.path.abspath(os.path.join(data_path, "../{}_list.txt".format(dirname)))
    with open(txt_save_path, 'w', encoding='utf-8') as fw:
        for d in dirs:
            d_path = data_path + "/{}".format(d)
            data_list = sorted(os.listdir(d_path))
            
            for f in data_list:
                if abspath:
                    f_abs_path = d_path + "/{}".format(f)
                    f_abs_path = f_abs_path.replace("\\", "/")
                    fw.write("{}\n".format(f_abs_path))
                else:
                    fw.write("{}\n".format(f))

    print("Success! --> {}".format(txt_save_path))




def untar_many_files(data_path):
    tar_list = sorted(os.listdir(data_path))
    for f in tar_list:
        f_abs_path = data_path + "/{}".format(f)
        if os.path.isfile(f_abs_path):
            file_name = os.path.splitext(f)[0]
            dir_name = os.path.abspath(os.path.join(f_abs_path, "../..")) + "/{}".format(file_name)
            cmd_line = "tar -xf %s -C %s" % (f_abs_path, dir_name)
            os.makedirs(dir_name, exist_ok=True)

            print(cmd_line)
            os.system(cmd_line)


def unzip_many_files(data_path):
    pass


def merge_dirs(data_path, use_glob=False, n_subdir=2):
    """
    TODO: 增加各文件夹的文件名是否相同, 防止覆盖
    """
    dir_name = get_dir_name(data_path)
    dst_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_merged".format(dir_name)
    os.makedirs(dst_path, exist_ok=True)

    if use_glob:
        dir_list = glob.glob(data_path + "{}/*".format(n_subdir * "/*"), recursive=True)
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


def random_select_files(data_path, mvcp="copy", select_num=1000, select_mode=0):
    data_list = sorted(os.listdir(data_path))
    dir_name = os.path.basename(data_path)

    assert select_num <= len(data_list), "{} > total num!".format(select_num)

    if select_mode == 0:
        selected = random.sample(data_list, select_num)
        save_path = os.path.abspath(os.path.join(data_path, "../")) + "/Random_Selected/{}_random_selected_{}".format(dir_name, select_num)
        os.makedirs(save_path, exist_ok=True)
    else:
        selected = random.sample(data_list, len(data_list) - select_num)
        save_path = os.path.abspath(os.path.join(data_path, "../")) + "/Random_Selected/{}_random_selected_{}".format(dir_name, len(data_list) - select_num)
        os.makedirs(save_path, exist_ok=True)

    for s in tqdm(selected):
        f_src_path = data_path + "/{}".format(s)
        f_dst_path = save_path + "/{}".format(s)

        if mvcp == "copy" or mvcp == "cp":
            shutil.copy(f_src_path, f_dst_path)
        elif mvcp == "move" or mvcp == "mv":
            shutil.move(f_src_path, f_dst_path)
        else:
            print("Error: mvcp should be one of [copy, cp, move, mv]!")
            raise ValueError


def get_json_data(data_path):
    fr = open(data_path, "r", encoding="utf-8")
    json_data = json.load(fr)
    fr.close()
    return json_data


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
        print("flag should be one of pickle, numpy or json!")
        raise ValueError


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
        raise ValueError


def compare_two_dict_files(file_path1, file_path2):
    dict_data1 = load_saved_dict_file(file_path1)
    dict_data2 = load_saved_dict_file(file_path2)
    list1 = list(dict_data1.values())
    list2 = list(dict_data2.values())
    diff = set(list1) ^ set(list2)
    return diff


def read_csv(file_path):
    csv_data = pd.read_csv(file_path)
    return csv_data.values


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


def find_chinese(chars):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', chars)
    return chinese


def find_sub_str_index(substr, str, time):
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


def examples_change_console_str_color():
    """
    @Time: 2021/1/22 21:16
    @Author: gracekafuu
    https://blog.csdn.net/qq_34857250/article/details/79673698

    """

    print('This is a \033[1;35m test \033[0m!')
    print('This is a \033[1;32;43m test \033[0m!')
    print('\033[1;33;44mThis is a test !\033[0m')


def remove_list_repeat_elements(list1):
    list2 = []
    [list2.append(i) for i in list1 if i not in list2]

    return list2


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
        second_mj = majority_element(arr)
        return second_mj
    else:
        return None


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


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # 限制最小的值
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
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


def calculate_md5(file_path):
    with open(file_path, "rb") as file:
        data = file.read()

    md5_hash = hashlib.md5()
    md5_hash.update(data)
    md5_value = md5_hash.hexdigest()

    return md5_value


def calculate_hash(file_path, hash_algorithm='sha256'):
    hash_obj = hashlib.new(hash_algorithm)
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(65536)
            if not data: break
            hash_obj.update(data)

    return hash_obj.hexdigest()


def move_same_file(data_path):
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "../{}_same_files".format(dir_name)))
    os.makedirs(save_path, exist_ok=True)

    file_list = get_file_list(data_path)
    duplicates = {}

    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        f_hash = calculate_hash(f_abs_path, hash_algorithm='sha256')
        if f_hash in duplicates:
            duplicates[f_hash].append(f)
        else:
            duplicates[f_hash] = [f]

    duplicates_new = {k: v for k, v in duplicates.items() if len(v) > 1}

    for k, v in duplicates_new.items():
        for fi in v[1:]:
            f_src_path = data_path + "/{}".format(fi)
            f_dst_path = save_path + "/{}".format(fi)
            shutil.move(f_src_path, f_dst_path)


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


# ---------------------------------------------------------------
def get_fname_ws(f, file_list):
    """
    f: file name with no suffix
    """
    fname_ws = ""
    for fn in file_list:
        if f in fn:
            fname_ws = fn
            break
    return fname_ws


def process_via_filename(path1, path2, save_path="", with_suffix=True, flag="diff", mvcp="mv"):
    """
    :param dir1:
    :param dir2:
    :param move_or_delete: "move" or "delete"
    :param dir: files in which dir will be move or delete
    :return:
    """
    assert flag == "same" or flag == "diff", "flag should be 'same' or 'diff'!"
    assert mvcp in ["move", "mv", "copy", "cp"], 'mvcp not in ["move", "mv", "copy", "cp"]!'
    dir1_name = get_dir_name(path1)
    dir2_name = get_dir_name(path2)
    file1_list = get_file_list(path1)
    file2_list = get_file_list(path2)

    par_path = os.path.join(path1, "..")


    if save_path is None or save_path == "":
        save_path = make_save_path(par_path, relative=".", add_str="Processed_{}_{}".format(dir1_name, dir2_name))
    else:
        os.makedirs(save_path, exist_ok=True)

    same_path = save_path + "/same"
    diff_path = save_path + "/diff"
    same_path1 = same_path + "/path1/{}".format(dir1_name)
    same_path2 = same_path + "/path2/{}".format(dir2_name)
    diff_path1 = diff_path + "/path1/{}".format(dir1_name)
    diff_path2 = diff_path + "/path2/{}".format(dir2_name)
    os.makedirs(same_path1, exist_ok=True)
    os.makedirs(same_path2, exist_ok=True)
    os.makedirs(diff_path1, exist_ok=True)
    os.makedirs(diff_path2, exist_ok=True)

    if with_suffix:
        same_list = list(set(file1_list) & set(file2_list))
        diff_list = list(set(file1_list) ^ set(file2_list))

        if flag == "same":
            for f in same_list:
                f_src_path1 = path1 + "/{}".format(f)
                f_src_path2 = path2 + "/{}".format(f)
                f_dst_path1 = same_path1 + "/{}".format(f)
                f_dst_path2 = same_path2 + "/{}".format(f)
                
                if mvcp == "move" or mvcp == "mv":
                    shutil.move(f_src_path1, f_dst_path1)
                    shutil.move(f_src_path2, f_dst_path2)
                else:
                    shutil.copy(f_src_path1, f_dst_path1)
                    shutil.copy(f_src_path2, f_dst_path2)
        else:
            for f in diff_list:
                f_src_path1 = path1 + "/{}".format(f)
                f_src_path2 = path2 + "/{}".format(f)
                f_dst_path1 = diff_path1 + "/{}".format(f)
                f_dst_path2 = diff_path2 + "/{}".format(f)
                
                if mvcp == "move" or mvcp == "mv":
                    if os.path.exists(f_src_path1):
                        shutil.move(f_src_path1, f_dst_path1)
                    if os.path.exists(f_src_path2):
                        shutil.move(f_src_path2, f_dst_path2)
                else:
                    if os.path.exists(f_src_path1):
                        shutil.copy(f_src_path1, f_dst_path1)
                    if os.path.exists(f_src_path2):
                        shutil.copy(f_src_path2, f_dst_path2)

    else:
        file1_list_ns = [os.path.splitext(fn)[0] for fn in sorted(os.listdir(path1))]  # ns: no suffix
        file2_list_ns = [os.path.splitext(fn)[0] for fn in sorted(os.listdir(path2))]  # ns: no suffix
        same_list = list(set(file1_list_ns) & set(file2_list_ns))
        diff_list = list(set(file1_list_ns) ^ set(file2_list_ns))
        
        if flag == "same":
            for f in same_list:
                fname_ws1 = get_fname_ws(f, file1_list)
                fname_ws2 = get_fname_ws(f, file2_list)
                f_src_path1 = path1 + "/{}".format(fname_ws1)
                f_src_path2 = path2 + "/{}".format(fname_ws2)
                f_dst_path1 = same_path1 + "/{}".format(fname_ws1)
                f_dst_path2 = same_path2 + "/{}".format(fname_ws2)
                
                if mvcp == "move" or mvcp == "mv":
                    shutil.move(f_src_path1, f_dst_path1)
                    shutil.move(f_src_path2, f_dst_path2)
                else:
                    shutil.copy(f_src_path1, f_dst_path1)
                    shutil.copy(f_src_path2, f_dst_path2)
        else:
            for f in diff_list:
                fname_ws1 = get_fname_ws(f, file1_list)
                fname_ws2 = get_fname_ws(f, file2_list)

                if fname_ws1 != "" and fname_ws2 == "":
                    fname_ws = fname_ws1
                elif fname_ws1 == "" and fname_ws2 != "":
                    fname_ws = fname_ws2
                else:
                    print("fname_ws1: {} fname_ws2: {}".format(fname_ws1, fname_ws2))
                    raise Exception("Error")

                f_src_path1 = path1 + "/{}".format(fname_ws)
                f_src_path2 = path2 + "/{}".format(fname_ws)
                f_dst_path1 = diff_path1 + "/{}".format(fname_ws)
                f_dst_path2 = diff_path2 + "/{}".format(fname_ws)
                
                if mvcp == "move" or mvcp == "mv":
                    # if os.path.exists(f_src_path1):
                    #     shutil.move(f_src_path1, f_dst_path1)
                    if os.path.exists(f_src_path2):
                        shutil.move(f_src_path2, f_dst_path2)
                else:
                    # if os.path.exists(f_src_path1):
                    #     shutil.copy(f_src_path1, f_dst_path1)
                    if os.path.exists(f_src_path2):
                        shutil.copy(f_src_path2, f_dst_path2)


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
        f_abs_path = data_path + "/{}".format(f)
        f_dst_names = []
        for i in range(n):
            f_dst_names.append("{}_cp{}{}".format(f_name, i + 1, f_suffix))

        for j in f_dst_names:
            f_dst_path = save_path + "/{}".format(j)
            shutil.copy(f_abs_path, f_dst_path)
            if print_flag:
                print("{} --> {}".format(f_abs_path, f_dst_path))


def copy_via_txt(txt_path="", save_path=""):
    os.makedirs(save_path, exist_ok=True)

    data = open(txt_path, "r", encoding="utf-8")
    lines = data.readlines()
    data.close()

    for l in lines:
        f_abs_path = l.strip()
        fname = os.path.basename(f_abs_path)
        f_dst_path = save_path + "/{}".format(fname)

        shutil.copy(f_abs_path, f_dst_path)


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


def get_file_type_code(file_name, max_len=16):
    """
    :param file_name:
    :param max_len:
    :return: type_code: 23212F686F6D652F7A656E6779696661
    """
    with open(file_name, "rb") as fo:
        byte = fo.read(max_len)

    byte_list = struct.unpack('B' * max_len, byte)
    code = ''.join([('%X' % each).zfill(2) for each in byte_list])

    return code


def change_conda_envs_files_content(conda_envs_path):
    """
    Can work!
    :param conda_envs_path:
    :return:
    """
    replace_str = "#!{}".format(conda_envs_path)

    file_list = sorted(os.listdir(conda_envs_path))
    for f in file_list:
        f_abs_path = conda_envs_path + "/{}".format(f)
        byte = get_file_type_code(f_abs_path)
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


def merge_txt_content(path1, path2):
    """ path1 <-- path2 """
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

        print("OK!")


# def merge_txt_files(data_path):
#     dirname = os.path.basename(data_path)
#     save_path = make_save_path(data_path, "{}_merged_txt".format(dirname))

#     dir_list = sorted(os.listdir(data_path))
#     file_list = []

#     for d in dir_list:
#         d_path = data_path + "/{}".format(d)
#         d_file_list = sorted(os.listdir(d_path))
#         for f in d_file_list:
#             if f not in file_list:
#                 file_list.append(f)

#     for f in file_list:


#         for d in dir_list:
#             df_path = data_path + "/{}/{}".format(d, f)
#             if not os.path.exists(df_path): continue
#             if os.path.isfile(df_path) and df_path.endswith(".txt"):
#                 with open(df_path, "r", encoding="utf-8") as fr:
#                     lines = fr.readlines()
                    





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


def random_select_files_via_txt(data_path, select_percent):
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


class Logger(object):
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }
    def __init__(self, filename, level='info', when='D', interval=1, backCount=3, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        from logging.handlers import TimedRotatingFileHandler

        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt) # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level)) # 设置日志级别
        sh = logging.StreamHandler() # 往屏幕上输出
        sh.setFormatter(format_str) # 设置屏幕上显示的格式

        dirname = os.getcwd()
        # logpath = os.path.dirname(os.getcwd()) + '/Logs/'
        logpath = dirname + '/Logs/'
        os.makedirs(logpath, exist_ok=True)

        # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒、M 分、H 小时、D 天、W 每星期（interval==0时代表星期一）、midnight 每天凌晨
        th = TimedRotatingFileHandler(filename=logpath + filename +'.', when=when, interval=1, backupCount=backCount, encoding='utf-8')
        th.setFormatter(format_str) # 设置文件里写入的格式
        th.suffix = "-%Y-%m-%d_%H-%M-%S.log"

        self.logger.addHandler(sh) # 把对象加到logger里
        self.logger.addHandler(th)
        

def unzip_file(path, pwd, save_path):
    with zipfile.ZipFile(path, 'r') as zip:
        try:
            zip.extractall(save_path, pwd=pwd.encode('utf-8'))
            print("解压成功，密码是：%s"%(pwd))
            return True
        except Exception as e:
            pass


def create_passward(words, repeat=6):
    import itertools as its
    words = its.product(words, repeat=repeat)
    for i in words:
        yield ''.join(i)
    

def crack_passward(file_path, words='0123456789', repeat=6):
    assert file_path.endswith(".zip"), "{} should be *.zip".format(file_path)
    save_path = make_save_path(file_path, ".", "Extracted")

    pwd = create_passward(words, repeat=repeat)
    for p in pwd:

        flag = unzip_file(file_path, p, save_path)
        if flag: break


def send_email(from_addr, to_addr, subject, password):
    import smtplib
    from email import encoders
    from email.header import Header
    from email.mime.text import MIMEText
    from email.utils import parseaddr, formataddr

    """ 
    # 这里的密码是开启smtp服务时输入的客户端登录授权码，并不是邮箱密码
    # 现在很多邮箱都需要先开启smtp才能这样发送邮件
    send_email(u"zmmbb100@163.com",u"zmmbb100@163.com",u"主题",u"SHMILLavender66")
    """

    msg = MIMEText("Test_2019_4_14",'html','utf-8')
    msg['From'] = u'<%s>' % from_addr
    msg['To'] = u'<%s>' % to_addr
    msg['Subject'] = subject

    smtp = smtplib.SMTP_SSL('smtp.163.com', 465)
    smtp.set_debuglevel(1)
    smtp.ehlo("smtp.163.com")
    smtp.login(from_addr, password)
    smtp.sendmail(from_addr, [to_addr], msg.as_string())


def create_word_cloud(txt_fpath="yxy.txt", font_path="jingboran.ttf"):
    from wordcloud import WordCloud
    import jieba
    import matplotlib.pyplot as plt

    text = open(txt_fpath,'r').read()

    cut_text = jieba.cut(text)
    result = '/'.join(cut_text)

    wc = WordCloud(font_path=font_path,background_color='white',width=1000,height=800,max_font_size=200,max_words=10000)
    wc.generate(result)
    wc.to_file(txt_fpath.replace(".txt",".png"))

    plt.figure()
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


def remove_corrupt_img(data_path):
    file_list = get_file_list(data_path)

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        if img is None:
            print("{} is corrupt".format(f_abs_path))
            os.remove(f_abs_path)


def process_db(db_path, m):
    import sqlite3
    from datetime import datetime

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    assert m in ["read", "r", "write", "w"], 'm should be one of ["read", "r", "write", "w"]!'

    if m == "read" or m == "r":
        table_list = [a for a in cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]

        print('table_list is :\n', table_list)

        # 获取表的列名（字段名），保存在col_names列表,每个表的字段名集为一个元组
        col_names = []
        for i in table_list:
            col_name = cursor.execute('PRAGMA table_info ({})'.format(i[0])).fetchall()
            col_name = [x[1] for x in col_name]
            col_names.append(col_name)

        print(col_names)

        for tab in table_list:
            # Can work
            # df = pd.read_sql_query('SELECT * FROM {}'.format(tab[0]), conn)
            # print('table ', tab[0], 'head is :\n', df.head())
            # print('table ', tab[0], 'shape is :\n', df.shape)
            # # df.to_excel('./'+tab[0]+'.xlsx')

            # Can work
            result = cursor.execute("SELECT * FROM {}".format(tab[0])).fetchall()
            print(result)
    else:
        # 还有问题，插入失败（sqlite3.OperationalError: near "%": syntax error），待解决，2025.02.13
        insert_sql = "INSERT INTO server_info (id, project_name, hostname, username, password, port, server_state, status, create_time, create_by, update_time, zip_file, install_type, install_state, remarks) VALUES (%d, %s, %s, %s, %s, %d, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        insert_data = (5, 'Test服务器', '10.10.11.205', 'gx', '123456', 22, '1', '0', datetime.now(), '', datetime.now(), 'D:/ubuntu-install/guoxun.tar.gz', '1', '0', '')
        cursor.execute(insert_sql, insert_data)
        conn.commit()
        print("数据插入成功！")

    cursor.close()
    conn.close()

    return


def get_system_platform():
    """
    https://blog.csdn.net/JiuYux/article/details/135343778
    """
    if sys.platform.startswith("win"):
        print("当前系统是Windows")
    elif sys.platform.startswith("linux"):
        print("当前系统是Linux")
    elif sys.platform.startswith("darwin"):
        print("当前系统是Mac OS")
    else:
        print("当前系统是其他操作系统")


def remove_file_by_filename_keywords(data_path, k=""):
    file_list = get_file_list(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        if k in fname:
            f_abs_path = data_path + "/{}".format(f)
            os.remove(f_abs_path)
            print("{} is removed".format(f_abs_path))


def move_file_by_filename_keywords(data_path, k=""):
    file_list = get_file_list(data_path)
    save_path = make_save_path(data_path, ".", "moved")
    for f in file_list:
        fname = os.path.splitext(f)[0]
        if k in fname:
            f_abs_path = data_path + "/{}".format(f)
            f_dst_path = save_path + "/{}".format(f)
            shutil.move(f_abs_path, f_dst_path)
            print("{} --> {}".format(f_abs_path, f_dst_path))


if __name__ == '__main__':
    pass

    # # words = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:,.<>?`~"
    # words='0123456789'
    # for n in range(1, 9):
    #     print(n)
    #     crack_passward(file_path="D:/GraceKafuu/Music/zcx/zcx.zip", words=words, repeat=n)

    # merge_txt_content(path1=r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v2\500\labels", path2=r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v2\500\labels_T-shirt")
    # rename_files(data_path=r"G:\Gosion\data\006.Belt_Torn_Det\data\videos\20250807_frames_merged", new_name_prefix="rename_20250807", start_num=0)
    # rename_files(data_path=r"G:\Gosion\data\007.PPE_Det\data\helmet\train\labels", new_name_prefix="person_helmet_1_20250402", start_num=0)

    # data_path = r"D:\Gosion\Projects\GuanWangLNG\leaking-20250223"
    # dir_list = os.listdir(data_path)
    # for i, d in enumerate(dir_list):
    #     dir_path = data_path + "/{}".format(d)
    #     rename_files(data_path=dir_path, new_name_prefix="leaking-20250223_{}_{}".format(d, i), start_num=0)


    # remove_corrupt_img(data_path=r"D:\Gosion\Projects\002.Smoking_Det\data\New_All\Add\Det\v1_add\v2\train\images")

    # move_same_file(data_path=r"G:\Gosion\data\007.PPE_Det\data\v1\all_yolo_format\images")

    # # TODO: 增加各文件夹的文件名是否相同, 防止覆盖
    # merge_dirs(data_path=r"G:\Gosion\data\006.Belt_Torn_Det\data\videos\20250822-大坝纸壳测试\caijishuju_20250823_frames")


    random_select_files(data_path=r"G:\Gosion\data\006.Belt_Torn_Det\data\videos\20250822-大坝纸壳测试\caijishuju_20250823_frames_merged", mvcp="move", select_num=500)

    # process_via_filename(path1=r"G:\Gosion\data\006.Belt_Torn_Det\data\videos\20250811\20250808_frames_labelbee\images0", path2=r"G:\Gosion\data\006.Belt_Torn_Det\data\videos\20250811\20250808_frames_labelbee\images", save_path="", with_suffix=False, flag="same", mvcp="cp")
    

    # process_db(db_path=r"D:\Gosion\Projects\Algorithm_Deploy_GUI\env_manage\AppData\env_manage.db", m="w")

    # save_file_path_to_txt(data_path=r"G:\Gosion\code\07-3d-line\data\1200w\1200w_frames\Video_2025_04_17_113309_2", abspath=True)
    # save_all_files_path_to_txt(data_path=r"D:\Gosion\code\others\Python\arcface-pytorch-master\data\Datasets\lfw\lfw-align-128", abspath=True)

    # logger = Logger(filename="Test-Logger", level='info', when='D', interval=1, backCount=30)
    # logger.logger.info("test")
    # logger.logger.info("===========================================")

    # get_system_platform()

    # remove_file_by_filename_keywords(data_path=r"D:\Gosion\data\006.Belt_Torn_Det\data\pose\v4\v4_yitiji\000\labels", k="masked_cleaned")
    # move_file_by_filename_keywords(data_path=r"D:\Gosion\data\006.Belt_Torn_Det\data\pose\v4\v3\train_labelbee_format\jsons", k="aug")

    

    

































