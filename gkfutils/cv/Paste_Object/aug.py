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
import math
from .utils import (
    bbox_voc_to_yolo, bbox_yolo_to_voc,
    scale_uint16, 
)

from ..utils import (
    timestamp_to_strftime, get_file_list,
    get_base_name, get_dir_name,
    make_save_path,
)


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

    M = get_trans_mat((-width / 2, -height / 2), degrees_src, translate_src, scale_src, shear_src, perspective_src)

    border_color = (random.randint(220, 250), random.randint(220, 250), random.randint(220, 250))
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)
    return img


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

    datetime = timestamp_to_strftime()

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

    datetime = timestamp_to_strftime()

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
        base_name = get_base_name(f_abs_path)
        file_name = os.path.splitext(base_name)[0]
        suffix = os.path.splitext(base_name)[1]
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