import os
import cv2
import time
import torch
import torchvision
import onnxruntime
import numpy as np
import random
import threading
import copy
from tqdm import tqdm
from PIL import Image
from .utils import (
    bbox_voc_to_yolo, bbox_yolo_to_voc,
    scale_uint16, scale_down_bbx,
    cal_iou, 
)


# ======================================================================================================================================
# ================================ PIL paste cropped object for det train negative samples multi thread ================================
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
            bbx_VOC_format = bbox_yolo_to_voc(img_size, l_)
            bbxes.append(bbx_VOC_format)
    return bbxes


def write_yolo_label_pil_paste_cropped_object_aug_data(labels_save_path, img_name, pasted_poses, bg_size, add_rename_str="pasted", scale_flag=False, scale_ratio=0.04, cls=0):
    txt_save_path_added_res = "{}/{}_{}_v6.txt".format(labels_save_path, img_name, add_rename_str)
    with open(txt_save_path_added_res, "w", encoding="utf-8") as fw:
        for bb_ in pasted_poses:
            if scale_flag:
                bbx_new = scale_down_bbx(bb_, scale_ratio=scale_ratio)
                # bb = convert_bbx_VOC_to_yolo((bbx_new[0], bbx_new[0] + bbx_new[2], bbx_new[1], bbx_new[1] + bbx_new[3]), bg_size)
                bb = bbox_voc_to_yolo((bbx_new[0], bbx_new[1], bbx_new[0] + bbx_new[2], bbx_new[1] + bbx_new[3]), bg_size)
                txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bb]) + "\n"
                fw.write(txt_content)
            else:
                # bb = convert_bbx_VOC_to_yolo((bb_[0], bb_[0] + bb_[2], bb_[1], bb_[1] + bb_[3]), bg_size)
                bb = bbox_voc_to_yolo((bb_[0], bb_[1], bb_[0] + bb_[2], bb_[1] + bb_[3]), bg_size)
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
# ================================ PIL paste cropped object for det train negative samples multi thread ================================
# ======================================================================================================================================


if __name__ == '__main__':
    pass