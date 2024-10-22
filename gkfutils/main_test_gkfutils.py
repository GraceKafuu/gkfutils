import gkfutils


if __name__ == '__main__':
    gkfutils.utils.gen_file_list(data_path="", abspath=True)

    gkfutils.cv.utils.extract_one_gif_frames(gif_path="")
    gkfutils.cv.utils.extract_one_video_frames(video_path="", gap=5)
    gkfutils.cv.utils.extract_videos_frames(base_path="", gap=5, save_path="")

    gkfutils.utils.split_dir_multithread(data_path="", split_n=10)
    gkfutils.utils.split_dir_by_file_suffix(data_path="")

    gkfutils.utils.random_select_files(data_path="", select_num=500, move_or_copy="copy", select_mode=0)
    gkfutils.cv.utils.random_select_images_and_labels(data_path="", select_num=500, move_or_copy="copy", select_mode=0)

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

    gkfutils.cv.utils.vis_yolo_label(data_path="", print_flag=False, color_num=1000, rm_small_object=False, rm_size=32)  # TODO: 1.rm_small_object have bugs.
    gkfutils.cv.utils.list_yolo_labels(label_path="")
    gkfutils.cv.utils.change_txt_content(txt_base_path="")
    gkfutils.cv.utils.remove_yolo_txt_contain_specific_class(data_path="", rm_cls=(0, ))
    gkfutils.cv.utils.remove_yolo_txt_small_bbx(data_path="", rm_cls=(0, ), rmsz=(48, 48))
    gkfutils.cv.utils.select_yolo_txt_contain_specific_class(data_path="", select_cls=(3, ))
    gkfutils.cv.utils.merge_txt(path1="", path2="")
    gkfutils.cv.utils.merge_txt_files(data_path="")


    gkfutils.cv.utils.dbnet_aug_data(data_path="", bg_path="", maxnum=10000)
    gkfutils.cv.utils.vis_dbnet_gt(data_path="")
    gkfutils.cv.utils.warpPerspective_img_via_labelbee_kpt_json(data_path="")

    alpha = ' ' + '0123456789' + '.:/\\-' + 'ABbC'
    gkfutils.cv.utils.ocr_data_gen_train_txt_v2(data_path="", LABEL=alpha)
    gkfutils.cv.utils.check_ocr_label(data_path="", label=alpha)
    gkfutils.cv.utils.random_select_files_according_txt(data_path="", select_percent=0.25)






