import os
import cv2
import time
import torch
import torchvision
import onnxruntime
import numpy as np
import random
import threading
from tqdm import tqdm
from PIL import Image
from .utils import (
    bbox_voc_to_yolo, bbox_yolo_to_voc,
    scale_uint16, 
)


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
                bb = bbox_voc_to_yolo(size, (x_min, y_min, x_max, y_max))
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
                bb = bbox_voc_to_yolo((h, w), (x_min, y_min, x_max, y_max))

                cv2.imwrite("{}/{}_added_{}_{}.jpg".format(save_image_path, bg_img_name, j, rename_add_str), added_res)
                txt_save_path_added = save_txt_path + "/{}_added_{}_{}.txt".format(bg_img_name, j, rename_add_str)
                with open(txt_save_path_added, "w", encoding="utf-8") as fw:
                    txt_content = "{}".format(cls) + " " + " ".join([str(a) for a in bb]) + "\n"
                    fw.write(txt_content)

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


if __name__ == '__main__':
    pass