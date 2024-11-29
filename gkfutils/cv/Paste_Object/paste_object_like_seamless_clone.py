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
    scale_uint16, thresh_img,
    cal_iou, 

)


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
                            gb_VOC_bbx = bbox_yolo_to_voc(bg_size, gb_yolo_bbx)
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
                    final_yolo_bbx = bbox_voc_to_yolo(bg_size, final_VOC_bbx)
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
                        bi_VOC_bbx = bbox_yolo_to_voc(bg_size, final_yolo_bbxes[bi])
                        bj_VOC_bbx = bbox_yolo_to_voc(bg_size, final_yolo_bbxes[bj])

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
                            gb_VOC_bbx = bbox_yolo_to_voc(bg_size, gb_yolo_bbx)
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
                    final_yolo_bbx = bbox_voc_to_yolo(bg_size, final_VOC_bbx)
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
                        bi_VOC_bbx = bbox_yolo_to_voc(bg_size, final_yolo_bbxes[bi])
                        bj_VOC_bbx = bbox_yolo_to_voc(bg_size, final_yolo_bbxes[bj])

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
                            gb_VOC_bbx = bbox_yolo_to_voc(bg_size, gb_yolo_bbx)
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
                    final_yolo_bbx = bbox_voc_to_yolo(bg_size, final_VOC_bbx)
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
                        bi_VOC_bbx = bbox_yolo_to_voc(bg_size, final_yolo_bbxes[bi])
                        bj_VOC_bbx = bbox_yolo_to_voc(bg_size, final_yolo_bbxes[bj])

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
                            gb_VOC_bbx = bbox_yolo_to_voc(bg_size, gb_yolo_bbx)
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
                    final_yolo_bbx = bbox_voc_to_yolo(bg_size, final_VOC_bbx)
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
                        bi_VOC_bbx = bbox_yolo_to_voc(bg_size, final_yolo_bbxes[bi])
                        bj_VOC_bbx = bbox_yolo_to_voc(bg_size, final_yolo_bbxes[bj])

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


if __name__ == '__main__':
    pass