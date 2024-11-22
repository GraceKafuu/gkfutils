gkf means "GraceKafuu", gkfutils is my personal python package and my study to make a .whl file.


<details open>
<summary>Install</summary>
Pip install the gkfutils package.
  
[![PyPI - Version](https://img.shields.io/pypi/v/gkfutils?logo=pypi&logoColor=white)](https://pypi.org/project/gkfutils/)

```bash
pip install gkfutils
```

<details open>
<summary>Examples</summary>

```bash
import gkfutils

print(gkfutils.__version__)

# ======== Base utils ========
gkfutils.rename_files(data_path="E:\\Gosuncn\\Projects\\006.Fire_Smoke_Det\\SSOD_test\\unlabel_pred_same", use_orig_name=False, new_name_prefix="Test", zeros_num=20, start_num=0)
gkfutils.save_file_path_to_txt(data_path="E:\\Gosuncn\\Projects\\006.Fire_Smoke_Det\\SSOD_test\\unlabel_pred_same", abspath=True)
gkfutils.merge_dirs(data_path="data/test")
gkfutils.random_select_files(data_path="data/images", mvcp="copy", select_num=5, select_mode=0)

strftime = gkfutils.timestamp_to_strftime(timestamp=123456789.00)
timestamp = gkfutils.strftime_to_timestamp(strftime="2024-11-15 09:12:00")
curr_time = gkfutils.get_date_time()
file_list = gkfutils.get_file_list(data_path="data/images", abspath=True)
dir_list = gkfutils.get_dir_list(data_path="data")
dir_file_list = gkfutils.get_dir_file_list(data_path="data")
base_name = gkfutils.get_base_name("data/images/0.jpg")  # 0.jpg
base_name = gkfutils.get_base_name("data/images")  # images
dir_name = gkfutils.get_dir_name(data_path="data/images")  # images
file_name = gkfutils.get_file_name(data_path="data/images/0.jpg")  # 0
file_name = gkfutils.get_file_name_with_suffix(data_path="data/images/0.jpg")  # 0.jpg
suffix = gkfutils.get_suffix(data_path="data/images/0.jpg")  # .jpg
save_path = gkfutils.make_save_path(data_path="data/images", relative=".", add_str="test")
gkfutils.split_dir_multithread(data_path="", split_n=10)


# ======== CV ========  yolo <--> voc <--> labelbee <--> coco
""" yolo <-> labelbee """
gkfutils.cv.utils.labelbee_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_labelbee_format", copy_images=True, small_bbx_thresh=3, cls_plus=-1)  # OK
gkfutils.cv.utils.yolo_to_labelbee(data_path="E:/GraceKafuu/yolo/coco128/data", copy_images=True, small_bbx_thresh=3, cls_plus=1)  # OK

""" yolo <-> voc """
coco_classes = gkfutils.cv.utils.get_coco_names()
gkfutils.cv.utils.voc_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_voc_format", classes=coco_classes, copy_images=True, small_bbx_thresh=3, cls_plus=0)  # OK
gkfutils.cv.utils.yolo_to_voc(data_path="E:/GraceKafuu/yolo/coco128/data", classes=coco_classes, copy_images=True, small_bbx_thresh=3, cls_plus=0)  # OK

""" yolo <-> coco """
categories = gkfutils.cv.utils.get_coco_categories()
gkfutils.cv.utils.coco_to_yolo(data_path="E:/GraceKafuu/yolo/coco128/data_coco_format", json_name="instances_val2017_20241121.json", copy_images=False, small_bbx_thresh=3,cls_plus=0)  # OK
gkfutils.cv.utils.yolo_to_coco(data_path="E:/GraceKafuu/yolo/coco128/data", json_name="instances_val2017_20241121.json", categories=categories, copy_images=False, small_bbx_thresh=3, cls_plus=0)  # OK

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

```
