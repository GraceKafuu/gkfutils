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

################################################################################
# 1.生成一个txt文件，内容是路径下文件的绝对路径/相对路径。（路径下不包含子目录）
gkfutils.utils.gen_file_list(data_path="", abspath=True)

################################################################################
# 2.提取视频帧
gkfutils.cv.utils.extract_one_gif_frames(gif_path="")
gkfutils.cv.utils.extract_one_video_frames(video_path="", gap=5)
gkfutils.cv.utils.extract_videos_frames(base_path="", save_path="", gap=5)

################################################################################
# 3.切分文件夹内的文件至多个文件夹中
gkfutils.utils.split_dir_multithread(data_path="", split_n=10)
gkfutils.utils.split_dir_by_file_suffix(data_path="")

################################################################################
# 4.随机选取文件
gkfutils.utils.random_select_files(data_path="", select_num=500, move_or_copy="copy", select_mode=0)
gkfutils.cv.utils.random_select_images_and_labels(data_path="", select_num=500, move_or_copy="copy", select_mode=0)  # 选取yolo格式的图片及对应的标签

################################################################################
# 5.IoU
b1 = [0, 0, 10, 10]
b2 = [2, 2, 12, 12]
iou = cal_iou(b1, b2)
print(iou)

################################################################################
# 6.yolo <--> voc <--> labelbee <--> coco
gkfutils.cv.utils.labelbee2yolo(data_path="", copy_image=True)
gkfutils.cv.utils.yolo2labelbee(data_path="")
gkfutils.cv.utils.voc2yolo(data_path="", classes=['dog', ], val_percent=0.1)
# gkfutils.cv.utils.yolo2voc(data_path="")  # TODO
gkfutils.cv.utils.labelbee_kpt_to_yolo(data_path="", copy_image=False)
gkfutils.cv.utils.labelbee_kpt_to_dbnet(data_path="", copy_image=True)
gkfutils.cv.utils.labelbee_seg_to_png(data_path="")
gkfutils.cv.utils.coco2yolo(root="")
# gkfutils.cv.utils.yolo2coco(root="")  # TODO
gkfutils.cv.utils.labelbee_kpt_to_labelme_kpt(data_path="")
gkfutils.cv.utils.labelbee_kpt_to_labelme_kpt_multi_points(data_path="")

gkfutils.cv.utils.convert_Stanford_Dogs_Dataset_annotations_to_yolo_format(data_path="")
gkfutils.cv.utils.convert_WiderPerson_Dataset_annotations_to_yolo_format(data_path="")
gkfutils.cv.utils.convert_TinyPerson_Dataset_annotations_to_yolo_format(data_path="")
gkfutils.cv.utils.convert_AI_TOD_Dataset_to_yolo_format(data_path="")

################################################################################
# 7.OCR
gkfutils.cv.utils.dbnet_aug_data(data_path="", bg_path="", maxnum=10000)
gkfutils.cv.utils.vis_dbnet_gt(data_path="")
gkfutils.cv.utils.warpPerspective_img_via_labelbee_kpt_json(data_path="")

alpha = ' ' + '0123456789' + '.:/\\-' + 'ABbC'
gkfutils.cv.utils.ocr_data_gen_train_txt_v2(data_path="", LABEL=alpha)
gkfutils.cv.utils.check_ocr_label(data_path="", label=alpha)
gkfutils.cv.utils.random_select_files_according_txt(data_path="", select_percent=0.25)

################################################################################
# 8.


```
