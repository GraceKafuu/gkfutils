# -*- coding:utf-8 -*-

"""
# @Time       : 2022/05/13 13:56 Update
#               2024/03/29 14:30 Update
#               2024/10/14 16:15 Update
# @Author     : GraceKafuu
# @Email      : gracekafuu@gmail.com
# @File       : main_test.py
# @Software   : PyCharm

Description:
1.
2.
3.

Change Log:
1.
2.
3.

"""


from cv.utils import *

import os
import re
import cv2
import json
import random
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image


if __name__ == '__main__':
    # img_path = "./data/images/0.jpg"
    # dst_path = img_path.replace(".jpg", "_res.jpg")
    # img = cv2.imread(img_path)
    # # res = rotate(img, random=False, p=1, algorithm=algorithm, center=(100, 100), angle=angle, scale=1, expand=expand)
    # # res = flip(img, random=False, p=1, m=-1)
    # # res = scale(img, random=False, p=1, fx=0.0, fy=0.5)
    # # res = resize(img, random=False, p=1, dsz=(1920, 1080), interpolation=cv2.INTER_LINEAR)
    # # res = equalize_hist(img, random=False, p=1, m=1)
    # # res = change_brightness(img, random=False, p=1, value=100)
    # # # res = gamma_correction(img, random=False, p=1, value=1.3)
    # # res = gaussian_noise(img, random=False, p=1, mean=0, var=0.1)
    # # res = poisson_noise(img, random=False, p=1)
    # # res = sp_noise(img, random=False, p=1, salt_p=0.0, pepper_p=0.001)
    # # res = make_sunlight_effect(img, random=False, p=1, center=(200, 200), effect_r=70, light_strength=170)
    # # res = color_distortion(img, random=False, p=1, value=-50)
    # # res = change_contrast_and_brightness(img, random=False, p=1, alpha=0.5, beta=90)
    # # res = clahe(img, random=False, p=1, m=1, clipLimit=2.0, tileGridSize=(8, 8))
    # # res = change_hsv(img, random=False, p=1, hgain=0.5, sgain=0.5, vgain=0.5)
    # # res = gaussian_blur(img, random=False, p=1, k=5)
    # # res = motion_blur(img, random=False, p=1, k=15, angle=90)
    # # res = median_blur(img, random=False, p=1, k=3)
    # # res = transperent_overlay(img, random=False, p=1, rect=(50, 50, 80, 100))
    # # res = dilation_erosion(img, random=False, p=1, flag="erode", scale=(6, 8))
    # res = make_rain_effect(img, random=False, p=1, m=1, length=20, angle=75, noise=500)
    # cv2.imwrite(dst_path, res)

    img_path = "./data/images/0.jpg"
    dst_path = img_path.replace(".jpg", "_res.jpg")
    if os.path.exists(dst_path): os.remove(dst_path)
    shutil.rmtree("./data/images_results")
    data_path = "./data/images"
    save_path = make_save_path(data_path=data_path, relative=".", add_str="results")
    file_list = get_file_list(data_path)
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        for i in range(10):
            # res = rotate(img, random=True, p=0.5, algorithm=algorithm, angle=(0, 360), expand=True)
            # res = flip(img, random=True, p=0.5)
            # res = scale(img, random=True, p=1, fx=(0.01, 2.0), fy=(0.01, 2.0))
            # res = resize(img, random=True, p=1, r=(0.01, 2.0), interpolation=cv2.INTER_LINEAR)
            # res = equalize_hist(img, random=True, p=0.5)
            # res = change_brightness(img, random=True, p=1, value=(-100, 100))
            # res = gamma_correction(img, random=True, p=1, value=(0.2, 1.8))
            # res = gaussian_noise(img, random=True, p=1, mean=0, var=0.5)
            # res = poisson_noise(img, random=True, p=1)
            # res = sp_noise(img, random=True, p=1, salt_p=0.01, pepper_p=0.01)
            # res = make_sunlight_effect(img, random=True, p=1, effect_r=(10, 80), light_strength=170)
            # res = color_distortion(img, random=True, p=1, value=(-50, 50))
            # res = change_contrast_and_brightness(img, random=True, p=1, alpha=1.0, beta=90)
            # res = clahe(img, random=True, p=1, m=1)
            # res = change_hsv(img, random=True, p=1, p=1, hgain=0.5, sgain=0.5, vgain=0.5)
            # res = gaussian_blur(img, random=True, p=1)
            # res = motion_blur(img, random=True, p=1)
            # res = transperent_overlay(img, random=True, p=1, max_h_r=1.0, max_w_r=0.25)
            # res = dilation_erosion(img, random=True, p=1, flag="erode")
            res = make_rain_effect(img, random=True, p=1, m=1, length=(10, 90), angle=(0, 180), noise=(100, 500))
            f_dst_path = save_path + "/{}_{}.jpg".format(fname, i)
            cv2.imwrite(f_dst_path, res)

    









































    # # alpha = read_ocr_lables(lbl_path="/home/wujiahu/code/crnn.pytorch-2024.03.12/utils/gen_fake/words/chinese_simple_with_special_chars.txt")
    # alpha = ' ' + '0123456789' + '.:/\\-' + 'AbC' + '℃' + 'MPa' + '㎡m³'
    # # print(len(alpha))
    # # # ocr_data_gen_train_txt(data_path="/home/disk/disk7/data/000.ChineseOCR/data/train/no_slash/train_v1", LABEL=alpha)
    # ocr_data_gen_train_txt_v2(data_path="/home/disk/disk7/data/010.Digital_Rec/crnn/train/v2/v2_add_20240425", LABEL=alpha)
    # check_ocr_label(data_path="/home/disk/disk7/data/000.ChineseOCR/data/train/merged.txt", label=alpha)
    # random_select_files_according_txt(data_path="/home/disk/disk7/data/010.Digital_Rec/crnn/train/v2/v1_15_cls_20240117_add.txt", select_percent=0.25)

    # =====================================================================================================================
    # ==================================================== Simple Test ====================================================
    # b1 = [0, 0, 10, 10]
    # b2 = [2, 2, 12, 12]
    # iou = cal_iou(b1, b2)
    # print(iou)

    # convert_Stanford_Dogs_Dataset_annotations_to_yolo_format(data_path="/home/zengyifan/wujiahu/data/Open_Dataset/Stanford Dogs Dataset")
    # convert_WiderPerson_Dataset_annotations_to_yolo_format(data_path="/home/zengyifan/wujiahu/data/Open_Dataset/WiderPerson/yolo_format")
    # convert_TinyPerson_Dataset_annotations_to_yolo_format(data_path="/home/zengyifan/wujiahu/data/Open_Dataset/TinyPerson")
    # convert_AI_TOD_Dataset_to_yolo_format(data_path="/home/zengyifan/wujiahu/data/Open_Dataset/AI-TOD")

    # check_yolo_dataset_classes(data_path="/home/zengyifan/wujiahu/data/Open_Dataset/WiderPerson/yolo_format_random_selected_500")
    # crop_one_image(img_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/seg_crop/for_seamless_clone/20230530/smoke/006_20230530_0001501.jpg", crop_area=[0, 1080, 0, 1920 // 3 * 2 - 20])
    # create_pure_images(save_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/Others/create_image_test", max_pixel_value=10, save_num=50000 * 2, p=0.7)
    # classify_images_via_bgr_values(img_path="/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/cls/1.1")

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

    # vstack_two_images(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/llj_0-9/llj_0-9_new")
    # cut_images(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/llj_0-9/llj_0-9_new_THR_INV")
    # stack_images(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/llj_0-9/0-9_output_ud")
    # crop_images(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/fake_number/old/fake_number9")
    # main_rename_test(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/from_lzx/gen_number_code/fake_number/fake_number14_m3_cropped", add_str="m3")

    # convert_ICDAR_to_custom_format(data_path="/home/zengyifan/wujiahu/data/010.Digital_Rec/others/CRNN_OpenDataset/SVT")
    # according_yolov8_pose_gen_head_bbx(data_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/4_cls_20230920/v3_4_cls_20230921/images", cls=2)

    # sliding_window_crop_test()
    # sliding_window_crop_test2()

    # crop_red_bbx_area(data_path="/home/zengyifan/wujiahu/data/004.Knife_Det/others/Robot_Test/Images/knife", expand_p=10)


    # file_path1 = "/home/zengyifan/wujiahu/GoroboAIReason/models/cigar_detect/cigar_det_yolov5s_cspnet_768_448_v2.onnx"
    # md5_value1 = calculate_md5(file_path1)
    # print("MD51: " + md5_value1)

    # rm_iou_larger_than_zero(data_path="/home/zengyifan/wujiahu/data/003.Cigar_Det/others/Others/20231206/AUG_20231208_cigar_cup")
    # makeBorderAndChangeYoloBbx(data_path="/home/zengyifan/wujiahu/data/004.Knife_Det/train/v2/NewFolder", n=3)
    # moveYoloLabelNumGreaterThanN(data_path="/home/zengyifan/wujiahu/data/004.Knife_Det/train/v2", N=2)
    # classify_color_and_gray_images(data_path="/home/zengyifan/wujiahu/data/000.Robot_BackFlow_Data/20231117/SmokeDetect/image/test")

    # img_path = "/home/zengyifan/wujiahu/data/000.Bg/007_bg_3885/images/007_bg_20230609_0000007.jpg"
    # img = cv2.imread(img_path)
    # dst = makeSunLightEffect(img, r=(50, 200), light_strength=300)
    # cv2.imwrite("/home/zengyifan/wujiahu/data/000.Bg/007_bg_3885/makeSunLightEffect_dst3.jpg", dst)

    # convertBaiduChineseOCRDatasetToCustomDatasetFormat(data_path="/home/disk/disk7/10021_bk/data/000.OCR/CRNN/data/train_Chinese/BaiduOCR_Chinese_Dataset")
    
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
    # convert_labelbee_det_json_to_yolo_txt(data_path="/home/disk/disk7/data/004.Knife_Det/Others/knife_scissors", copy_image=True)
    # convert_unknow_det_json_to_yolo_txt(data_path="/home/disk/disk7/data/004.Knife_Det/det/train/v7/data_20240730/001/train", copy_image=True)
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
    # convert_to_binary_image(data_path="/home/disk/disk7/wujiahu/data/010.Digital_Rec/Others/data/yueyahu/half_cropped_20240909", thr_low=88)

    # =====================================================================================================================
    # ================================================ change txt content =================================================
    # change_txt_content(txt_base_path="/home/disk/disk7/data/013.Droplet_Det/test/v3")
    # remove_yolo_txt_contain_specific_class(data_path="/home/disk/disk7/wujiahu/data/014.Fishing_Det/data/det/v1/train/000/New", rm_cls=(0, ))
    # remove_yolo_txt_small_bbx(data_path="/home/disk/disk7/data/004.Knife_Det/det/train/v7/2_cls/train/20240729/New/NewFolder", rm_cls=(0, ), rmsz=(48, 48))
    # select_yolo_txt_contain_specific_class(data_path="/home/disk/disk7/data/013.Droplet_Det/train/v3/lables-3", select_cls=(3, ))
    # merge_txt(path1="/home/disk/disk7/wujiahu/data/014.Fishing_Det/data/det/v1/train/000/New/NewFolder/labels", path2="/home/disk/disk7/wujiahu/data/014.Fishing_Det/data/det/v1/train/000/New/NewFolder/labels_pred")
    # list_yolo_labels(label_path="/home/disk/disk7/data/013.Droplet_Det/train/v4/labels")
    # merge_txt_files(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/v2/horizontal/train/txt/20240925_New")

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

    # alpha = read_ocr_lables(lbl_path="/home/wujiahu/code/CRNN_PyTorch_2024.05.28/utils/gen_fake/words/chinese_simple_with_special_chars.txt")
    # alpha = read_ocr_lables(lbl_path="/home/wujiahu/GraceKafuu/GraceKafuu_v1.0.0/Python/CV_v1.0.0/OCR/PyTorchOCR/Rec/CRNN/CRNN_PyTorch_2024.08.02/words/chinese_chars_v1_21159.txt")
    # alpha = ' ' + '0123456789' + '.:/\\-' + 'AbC' + '℃' + 'MPa' + '㎡m³'
    # alpha = ' ' + '0123456789' + '.:/\\-' + 'AbC' + '℃' + 'm³'
    # alpha = ' ' + '0123456789' + '.:/\\-' + 'AbC'
    # alpha = ' ' + '0123456789' + '.:/\\-' + 'ABbC'
    # alpha = ' ' + '0123456789' + '.:/\\-' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    # print(len(alpha))
    # ocr_data_gen_train_txt(data_path="/home/disk/disk7/data/010.Digital_Rec/crnn/train/v5/20240607_img_warp", LABEL=alpha)
    # ocr_data_gen_train_txt_v2(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/v2/horizontal/train/016_CUTE80_ICDAR2003_ICDAR2013_ICDAR2015_IIIT5K_SVT", LABEL=alpha)
    # ocr_data_merge_train_txt_files_v2(data_path="/home/disk/disk7/data/010.Digital_Rec/crnn/train/v7/same_chars", LABEL=alpha)
    # check_ocr_label(data_p/ath="/home/disk/disk7/data/010.Digital_Rec/crnn/test/v5/All/merged.txt", label=alpha)
    # random_select_files_according_txt(data_path="/home/disk/disk7/wujiahu/data/010.Digital_Rec/data/crnn/train/v7/NewFolder_20240910/merged.txt", select_percent=0.25)
    # random_select_files_from_txt(data_path="/home/disk/disk7/data/010.Digital_Rec/crnn/train/v4_aug.txt", n=2500)
    # convert_text_renderer_json_to_my_dataset_format(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/v2/horizontal/train/014_text_renderer/20240927_001")
    # convert_Synthetic_Chinese_String_Dataset_labels(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/OpenDatasets/ChineseData_20240918/Synthetic_Chinese_String_Datasets_labels")
    # convert_to_ocr_rec_data_mtwi(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/OpenDatasets/ChineseData_20240918/mtwi/icpr_mtwi_train")
    # convert_to_ocr_rec_data_ShopSign1(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/OpenDatasets/ChineseData_20240918/ShopSign_1265/000")
    # convert_to_ocr_rec_data_ShopSign2(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/OpenDatasets/ChineseData_20240918/ShopSign_1265/001")
    # ocr_train_txt_change_to_abs_path()
    # get_ocr_train_txt_alpha(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/v2/horizontal/train/txt/20240924/merged_new.txt")
    # 003: not_in_alpha_21160:  ®íƧʌΛи—′∶┅▪☆　〇のサシジマ㸃凉＂＇＋－／＜＝＞＼＾＿｀
    # 006: not_in_alpha_21160:  ®—∶▪　〇の㸃＂＇＋－／＝＼＿｀
    # 008: not_in_alpha_21160:  àéüОПР–—―′※∶⑾⑿⒀⒂⒃⒄⒅⒆⒈⒉⒊─━│┌┐╱■□▲△◆◇○◎●★☆〇『』てな＋
    # 011: not_in_alpha_21160:  üи—―′″※ⅰ∕∮⒂└○●☆　』〓〖〗ぁ﹪＂＋－／＜＞＿￠
    # check_ocr_train_txt(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/v2/horizontal/train/txt/20240925_New/003_Chinese_Street_View_Text_Recognition_new.txt")
    # random_select_images_from_ocr_train_txt(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/v2/horizontal/train/003_Chinese_Street_View_Text_Recognition/train.txt", select_num= 5000)
    # ocr_train_txt_split_to_train_and_test(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/eng/OpenDatasets/Syn90k/Syn90k.txt", train_percent=0.8)


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
    # rename_files_under_dirs(data_path="/home/disk/disk7/wujiahu/data/013.Droplet_Det/others/data/images/20240828_frames")


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
    # by hash value: sha256, md5
    # move_same_file(data_path="/home/disk/disk7/wujiahu/data/014.Fishing_Det/data/cls/v1/train/0_/NewFolder_merged")


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




























