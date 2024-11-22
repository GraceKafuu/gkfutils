import gkfutils

import os
import cv2
import shutil
import numpy as np
from PIL import Image


def image_processing():
    img_path = "./data/images/0.jpg"
    dst_path = img_path.replace(".jpg", "_res.jpg")
    img = cv2.imread(img_path)
    res = gkfutils.cv.utils.rotate(img, random=False, p=1, algorithm="pil", center=(100, 100), angle=angle, scale=1, expand=expand)
    res = gkfutils.cv.utils.flip(img, random=False, p=1, m=-1)
    res = gkfutils.cv.utils.scale(img, random=False, p=1, fx=0.0, fy=0.5)
    res = gkfutils.cv.utils.resize(img, random=False, p=1, dsz=(1920, 1080), interpolation=cv2.INTER_LINEAR)
    res = gkfutils.cv.utils.equalize_hist(img, random=False, p=1, m=1)
    res = gkfutils.cv.utils.change_brightness(img, random=False, p=1, value=100)
    res = gkfutils.cv.utils.gamma_correction(img, random=False, p=1, value=1.3)
    res = gkfutils.cv.utils.gaussian_noise(img, random=False, p=1, mean=0, var=0.1)
    res = gkfutils.cv.utils.poisson_noise(img, random=False, p=1)
    res = gkfutils.cv.utils.sp_noise(img, random=False, p=1, salt_p=0.0, pepper_p=0.001)
    res = gkfutils.cv.utils.make_sunlight_effect(img, random=False, p=1, center=(200, 200), effect_r=70, light_strength=170)
    res = gkfutils.cv.utils.color_distortion(img, random=False, p=1, value=-50)
    res = gkfutils.cv.utils.change_contrast_and_brightness(img, random=False, p=1, alpha=0.5, beta=90)
    res = gkfutils.cv.utils.clahe(img, random=False, p=1, m=1, clipLimit=2.0, tileGridSize=(8, 8))
    res = gkfutils.cv.utils.change_hsv(img, random=False, p=1, hgain=0.5, sgain=0.5, vgain=0.5)
    res = gkfutils.cv.utils.gaussian_blur(img, random=False, p=1, k=5)
    res = gkfutils.cv.utils.motion_blur(img, random=False, p=1, k=15, angle=90)
    res = gkfutils.cv.utils.median_blur(img, random=False, p=1, k=3)
    res = gkfutils.cv.utils.transperent_overlay(img, random=False, p=1, rect=(50, 50, 80, 100))
    res = gkfutils.cv.utils.dilation_erosion(img, random=False, p=1, flag="erode", scale=(6, 8))
    res = gkfutils.cv.utils.make_rain_effect(img, random=False, p=1, m=1, length=20, angle=75, noise=500)
    res = gkfutils.cv.utils.compress(img, random=False, p=1, quality=80)
    res = gkfutils.cv.utils.exposure(img, random=False, p=1, rect=(100, 150, 200, 180))
    res = gkfutils.cv.utils.change_definition(img, random=False, p=1, r=0.5)
    res = gkfutils.cv.utils.stretch(img, random=False, p=1, r=0.5)
    res = gkfutils.cv.utils.crop(img, random=False, p=1, rect=(0, 0, 100, 200))
    res = gkfutils.cv.utils.make_mask(img, random=False, p=1, rect=(0, 0, 100, 200), color=(255, 0, 255))
    res = gkfutils.cv.utils.squeeze(img, random=False, p=1, degree=20)
    res = gkfutils.cv.utils.make_haha_mirror_effect(img, random=False, p=1, center=(150, 150), r=10, degree=20)
    res = gkfutils.cv.utils.warp_img(img, random=False, p=1, degree=10)
    res = gkfutils.cv.utils.enhance_gray_value(img, random=False, p=1, gray_range=(0, 255))
    res = gkfutils.cv.utils.homomorphic_filter(img, random=False, p=1)
    res = gkfutils.cv.utils.contrast_stretch(img, random=False, p=1, alpha=0.25, beta=0.75)
    res = gkfutils.cv.utils.log_transformation(img, random=False, p=1)
    res = gkfutils.cv.utils.translate(img, random=False, p=1, tx=-20, ty=30, border_color=(114, 0, 114), dstsz=None)
    cv2.imwrite(dst_path, res)


def image_processing_aug():
    img_path = "./data/images/0.jpg"
    dst_path = img_path.replace(".jpg", "_res.jpg")
    if os.path.exists(dst_path): os.remove(dst_path)
    shutil.rmtree("./data/images_results")
    data_path = "./data/images"
    save_path = gkfutils.make_save_path(data_path=data_path, relative=".", add_str="results")
    file_list = gkfutils.get_file_list(data_path)
    p = 1

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        img = cv2.imread(f_abs_path)
        
        img = gkfutils.cv.utils.dilate_erode(img, random=True, p=p, flag=np.random.choice(["dilate", "erode"]))

        rdm0 = np.random.choice(np.arange(2))
        if rdm0 == 0:
            img = gkfutils.cv.utils.scale(img, random=True, p=p, fx=(0.5, 1.5), fy=(0.5, 1.5))
        else:
            img = gkfutils.cv.utils.stretch(img, random=True, p=p, r=(0.25, 1.25))

        rdm1 = np.random.choice(np.arange(5))
        if rdm1 == 0:
            img = gkfutils.cv.utils.change_brightness(img, random=True, p=p, value=(-75, 75))
        elif rdm1 == 1:
            img = gkfutils.cv.utils.gamma_correction(img, random=True, p=p, value=(0.5, 1.5))
        elif rdm1 == 2:
            img = gkfutils.cv.utils.change_contrast_and_brightness(img, random=True, p=p, alpha=(0.25, 0.75), beta=(0, 75))
        elif rdm1 == 3:
            img = gkfutils.cv.utils.clahe(img, random=True, p=p, m=np.random.choice([0, 1]),  clipLimit=(2.0, 4.0), tileGridSize=(4, 16))
        else:
            img = gkfutils.cv.utils.log_transformation(img, random=True, p=p)

        rdm2 = gkfutils.cv.utils.np.random.choice(np.arange(6))
        if rdm2 == 0:
            img = gkfutils.cv.utils.gaussian_noise(img, random=True, p=p, mean=(0, 1), var=(0.1, 0.25))
        elif rdm2 == 1:
            img = gkfutils.cv.utils.poisson_noise(img, random=True, p=p, n=(2, 5))
        elif rdm2 == 2:
            img = gkfutils.cv.utils.sp_noise(img, random=True, p=p, salt_p=(0.01, 0.025), pepper_p=(0.01, 0.025))
        elif rdm2 == 3:
            img = gkfutils.cv.utils.gaussian_blur(img, random=True, p=p)
        elif rdm2 == 4:
            img = gkfutils.cv.utils.motion_blur(img, random=True, p=p, angle=(-180, 180))
        else:
            img = gkfutils.cv.utils.median_blur(img, random=True, p=p)
        
        rdm3 = np.random.choice(np.arange(2))
        if rdm3 == 0:
            img = gkfutils.cv.utils.color_distortion(img, random=True, p=p, value=(-360, 360))
        else:
            img = gkfutils.cv.utils.change_hsv(img, random=True, p=p, hgain=(0.25, 0.75), sgain=(0.25, 0.75), vgain=(0.25, 0.75))
        
        img = gkfutils.cv.utils.transperent_overlay(img, random=True, p=p, max_h_r=1.0, max_w_r=0.5, alpha=(0.1, 0.6))

        # rdm4 = np.random.choice(np.arange(3))
        # if rdm4 == 0:
        #     img = gkfutils.cv.utils.dilate_erode(img, random=True, p=p, flag=np.random.choice(["dilate", "erode"]))
        # elif rdm4 == 1:
        #     img = gkfutils.cv.utils.open_close_gradient(img, random=True, p=p, flag=np.random.choice(["open", "close", "gradient"]))
        # else:
        #     img = gkfutils.cv.utils.tophat_blackhat(img, random=True, p=p, flag=np.random.choice(["tophat", "blackhat"]))

        rdm5 = np.random.choice(np.arange(2))
        if rdm5 == 0:
            img = gkfutils.cv.utils.make_sunlight_effect(img, random=True, p=p, effect_r=(10, 80), light_strength=(50, 80))
        else:
            img = gkfutils.cv.utils.make_rain_effect(img, random=True, p=p, m=np.random.choice([0, 1]), length=(10, 90), angle=(0, 180), noise=(100, 500))
        
        img = gkfutils.cv.utils.compress(img, random=True, p=p, quality=(25, 95))
        img = gkfutils.cv.utils.rotate(img, random=True, p=p, algorithm="pil", angle=(-45, 45), expand=True)

        # 以下OCR数据增强时不建议使用:
        # img = gkfutils.cv.utils.flip(img, random=True, p=p, m=np.random.choice([-1, 0, 1]))  # m=np.random.choice([-1, 0, 1])
        # img = gkfutils.cv.utils.equalize_hist(img, random=True, p=p, m=1)  # m=np.random.choice([0, 1])
        # img = gkfutils.cv.utils.translate(img, random=True, p=p, tx=(-50, 50), ty=(-50, 50), dstsz=None)

        # 以下还存在问题, 需要优化:
        # img = gkfutils.cv.utils.warp_and_deform(img, random=True, p=p, a=(5, 15), b=(1, 5), gridspace=(10, 20))
        # img = gkfutils.cv.utils.normalize(img, random=True, p=p, alpha=0, beta=1, norm_type=np.random.choice([cv2.NORM_MINMAX, cv2.NORM_L2]))  # 容易变黑图

        f_dst_path = save_path + "/{}.jpg".format(fname)
        cv2.imwrite(f_dst_path, img)


def make_border():
    # img_path = "./data/images/3.jpg"
    # dst_path = img_path.replace(".jpg", "_res.jpg")
    img_path = "./data/images/long.png"
    dst_path = img_path.replace(".png", "_res.png")
    img = cv2.imread(img_path)
    # res = gkfutils.cv.utils.make_border_v7(img, (64, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # res = gkfutils.cv.utils.make_border_v7(img, (256, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # res = gkfutils.cv.utils.make_border_v7(img, (64, 256), random=False, base_side="H", ppocr_format=True, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # cv2.imwrite(dst_path, res)

    res = gkfutils.cv.utils.make_border_v7(img, (64, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=True, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    if isinstance(res, list):
        for i in range(len(res)):
            cv2.imwrite(dst_path.replace(".png", "_res_{}.png".format(i)), res[i])
    else:
        cv2.imwrite(dst_path, res)


def det_labels_convertion():
    """ yolo <-> labelbee """
    # labelbee_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_labelbee_format", copy_images=True, small_bbx_thresh=3, cls_plus=-1)  # OK
    # yolo_to_labelbee(data_path="E:/GraceKafuu/yolo/coco128/data", copy_images=True, small_bbx_thresh=3, cls_plus=1)  # OK

    """ yolo <-> voc """
    # coco_classes = get_coco_names()
    # voc_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_voc_format", classes=coco_classes, copy_images=True, small_bbx_thresh=3, cls_plus=0)  # OK
    # yolo_to_voc(data_path="E:/GraceKafuu/yolo/coco128/data", classes=coco_classes, copy_images=True, small_bbx_thresh=3, cls_plus=0)  # OK

    """ yolo <-> coco """
    # categories = get_coco_categories()
    # coco_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_coco_format", json_name="instances_val2017_20241121.json", copy_images=False, small_bbx_thresh=3, cls_plus=0)  # OK
    # yolo_to_coco(data_path="E:/GraceKafuu/yolo/coco128/data", json_name="instances_val2017_20241121.json", categories=categories, copy_images=False, small_bbx_thresh=3, cls_plus=0)  # OK

    """ yolo <-> labelme """
    # TODO
    # labelme_to_yolo()
    # yolo_to_labelme()


if __name__ == '__main__':
    print(gkfutils.__version__)


    # ======== Base utils ========
    # gkfutils.rename_files(data_path="E:\\Gosuncn\\Projects\\006.Fire_Smoke_Det\\SSOD_test\\unlabel_pred_same", use_orig_name=False, new_name_prefix="Test", zeros_num=20, start_num=0)
    # gkfutils.save_file_path_to_txt(data_path="E:\\Gosuncn\\Projects\\006.Fire_Smoke_Det\\SSOD_test\\unlabel_pred_same", abspath=True)
    # gkfutils.merge_dirs(data_path="data/test")
    # gkfutils.random_select_files(data_path="data/images", mvcp="copy", select_num=5, select_mode=0)
    
    # strftime = gkfutils.timestamp_to_strftime(timestamp=123456789.00)
    # timestamp = gkfutils.strftime_to_timestamp(strftime="2024-11-15 09:12:00")
    # curr_time = gkfutils.get_date_time()
    # file_list = gkfutils.get_file_list(data_path="data/images", abspath=True)
    # dir_list = gkfutils.get_dir_list(data_path="data")
    # dir_file_list = gkfutils.get_dir_file_list(data_path="data")
    # base_name = gkfutils.get_base_name("data/images/0.jpg")  # 0.jpg
    # base_name = gkfutils.get_base_name("data/images")  # images
    # dir_name = gkfutils.get_dir_name(data_path="data/images")  # images
    # file_name = gkfutils.get_file_name(data_path="data/images/0.jpg")  # 0
    # file_name = gkfutils.get_file_name_with_suffix(data_path="data/images/0.jpg")  # 0.jpg
    # suffix = gkfutils.get_suffix(data_path="data/images/0.jpg")  # .jpg
    # save_path = gkfutils.make_save_path(data_path="data/images", relative=".", add_str="test")
    # gkfutils.split_dir_multithread(data_path="", split_n=10)


    # ======== CV ========
    # image_processing()
    # image_processing_aug()
    # make_border()
    # det_labels_convertion()

    # iou = gkfutils.cv.utils.cal_iou(bbx1=[0, 0, 10, 10], bbx2=[2, 2, 12, 12])
    # gkfutils.cv.utils.extract_one_gif_frames(gif_path="")
    # gkfutils.cv.utils.extract_one_video_frames(video_path="", gap=5)
    # gkfutils.cv.utils.extract_videos_frames(base_path="", gap=5, save_path="")
    # gkfutils.cv.utils.convert_to_jpg_format(data_path="")
    # gkfutils.cv.utils.convert_to_png_format(data_path="")
    # gkfutils.cv.utils.convert_to_gray_image(data_path="")
    # gkfutils.cv.utils.convert_to_binary_image(data_path="", thr_low=88)
    # gkfutils.cv.utils.crop_image_according_labelbee_json(data_path="", crop_ratio=(1, 1.2, 1.5, ))
    # gkfutils.cv.utils.crop_ocr_rec_img_according_labelbee_det_json(data_path="")
    # gkfutils.cv.utils.crop_image_according_yolo_txt(data_path="", CLS=(0, ), crop_ratio=(1.0, ))  # 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0
    # gkfutils.cv.utils.random_crop_gen_cls_negative_samples(data_path="", random_size=(196, 224, 256, 288, 384), randint_low=1, randint_high=4, hw_dis=100, dst_num=1000)
    # gkfutils.cv.utils.seg_object_from_mask(base_path="")


    # ======== Object detection utils ========
    """ yolo <-> labelbee """
    # gkfutils.cv.utils.labelbee_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_labelbee_format", copy_images=True, small_bbx_thresh=3, cls_plus=-1)  # OK
    # gkfutils.cv.utils.yolo_to_labelbee(data_path="E:/GraceKafuu/yolo/coco128/data", copy_images=True, small_bbx_thresh=3, cls_plus=1)  # OK

    """ yolo <-> voc """
    # coco_classes = gkfutils.cv.utils.get_coco_names()
    # gkfutils.cv.utils.voc_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_voc_format", classes=coco_classes, copy_images=True, small_bbx_thresh=3, cls_plus=0)  # OK
    # gkfutils.cv.utils.yolo_to_voc(data_path="E:/GraceKafuu/yolo/coco128/data", classes=coco_classes, copy_images=True, small_bbx_thresh=3, cls_plus=0)  # OK

    """ yolo <-> coco """
    # categories = gkfutils.cv.utils.get_coco_categories()
    # gkfutils.cv.utils.coco_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_coco_format", json_name="instances_val2017_20241121.json", copy_images=False, small_bbx_thresh=3, cls_plus=0)  # OK
    # gkfutils.cv.utils.yolo_to_coco(data_path="E:/GraceKafuu/yolo/coco128/data", json_name="instances_val2017_20241121.json", categories=categories, copy_images=False, small_bbx_thresh=3, cls_plus=0)  # OK

    # gkfutils.cv.utils.labelbee_kpt_to_yolo(data_path="", copy_image=False)
    # gkfutils.cv.utils.labelbee_kpt_to_dbnet(data_path="", copy_image=True)
    # gkfutils.cv.utils.labelbee_kpt_to_labelme_kpt(data_path="")
    # gkfutils.cv.utils.labelbee_kpt_to_labelme_kpt_multi_points(data_path="")
    # gkfutils.cv.utils.labelbee_seg_to_png(data_path="")

    # gkfutils.cv.utils.convert_Stanford_Dogs_Dataset_annotations_to_yolo_format(data_path="")
    # gkfutils.cv.utils.convert_WiderPerson_Dataset_annotations_to_yolo_format(data_path="")
    # gkfutils.cv.utils.convert_TinyPerson_Dataset_annotations_to_yolo_format(data_path="")
    # gkfutils.cv.utils.convert_AI_TOD_Dataset_to_yolo_format(data_path="")

    # gkfutils.cv.utils.random_select_yolo_images_and_labels(data_path="", select_num=500, move_or_copy="copy", select_mode=0)
    # gkfutils.cv.utils.vis_yolo_label(data_path="", print_flag=False, color_num=1000, rm_small_object=False, rm_size=32)  # TODO: 1.rm_small_object have bugs.
    # gkfutils.cv.utils.list_yolo_labels(label_path="")
    # gkfutils.cv.utils.change_txt_content(txt_base_path="")
    # gkfutils.cv.utils.remove_yolo_txt_contain_specific_class(data_path="", rm_cls=(0, ))
    # gkfutils.cv.utils.remove_yolo_txt_small_bbx(data_path="", rm_cls=(0, ), rmsz=(48, 48))
    # gkfutils.cv.utils.select_yolo_txt_contain_specific_class(data_path="", select_cls=(3, ))
    # gkfutils.cv.utils.merge_txt(path1="", path2="")
    # gkfutils.cv.utils.merge_txt_files(data_path="")


    # ======== OCR ========
    # gkfutils.cv.utils.dbnet_aug_data(data_path="", bg_path="", maxnum=10000)
    # gkfutils.cv.utils.vis_dbnet_gt(data_path="")
    # gkfutils.cv.utils.warpPerspective_img_via_labelbee_kpt_json(data_path="")
    # alpha = gkfutils.cv.utils.read_ocr_lables(lbl_path="")  # alpha = ' ' + '0123456789' + '.:/\\-' + 'ABbC'
    # gkfutils.cv.utils.check_ocr_label(data_path="", label=alpha)
    # gkfutils.cv.utils.ocr_data_gen_train_txt_v2(data_path="", LABEL=alpha)
    # gkfutils.cv.utils.random_select_files_according_txt(data_path="", select_percent=0.25)
    # gkfutils.cv.utils.make_border_v7(img, (64, 256), random=True, base_side="H", ppocr_format=False, r1=0.75, r2=0.25, sliding_window=False, specific_color=True, gap_r=(0, 7 / 8), last_img_make_border=True)
    # gkfutils.cv.utils.ocr_data_gen_train_txt(data_path="", LABEL=alpha)
    # gkfutils.cv.utils.ocr_data_gen_train_txt_v2(data_path="", LABEL=alpha)
    # gkfutils.cv.utils.ocr_data_merge_train_txt_files_v2(data_path="", LABEL=alpha)
    # gkfutils.cv.utils.random_select_files_according_txt(data_path="", select_percent=0.25)
    # gkfutils.cv.utils.random_select_files_from_txt(data_path="", n=2500)
    # gkfutils.cv.utils.convert_text_renderer_json_to_my_dataset_format(data_path="")
    # gkfutils.cv.utils.convert_Synthetic_Chinese_String_Dataset_labels(data_path="")
    # gkfutils.cv.utils.convert_mtwi_to_ocr_rec_data(data_path="")
    # gkfutils.cv.utils.convert_ShopSign_to_ocr_rec_data(data_path="")
    # gkfutils.cv.utils.ocr_train_txt_change_to_abs_path()
    # gkfutils.cv.utils.get_ocr_train_txt_alpha(data_path="")
    # gkfutils.cv.utils.check_ocr_train_txt(data_path="")
    # gkfutils.cv.utils.random_select_images_from_ocr_train_txt(data_path="", select_num= 5000)
    # gkfutils.cv.utils.ocr_train_txt_split_to_train_and_test(data_path="", train_percent=0.8)






