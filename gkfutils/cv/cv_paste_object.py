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
    scale_uint16, cal_iou,
    
)


# ======================================================================================================================================
# ================================== Paste cropped object for det train negative samples multi thread ==================================
# ======================================================================================================================================

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
                        bbx_VOC_format = bbox_yolo_to_voc(bg_size, l_)

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
                bb = bbox_voc_to_yolo(bg_size, [bbx_new[0], bbx_new[1], bbx_new[0] + bbx_new[2], bbx_new[1] + bbx_new[3]])
                txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bb]) + "\n"
                fw.write(txt_content)
            else:
                # bb = convert_bbx_VOC_to_yolo(bg_size, [bb_[0], bb_[0] + bb_[2], bb_[1], bb_[1] + bb_[3]])
                bb = bbox_voc_to_yolo(bg_size, [bb_[0], bb_[1], bb_[0] + bb_[2], bb_[1] + bb_[3]])
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


if __name__ == '__main__':
    pass