import os
import cv2
import numpy as np
from gkfutils.cv.utils import log_transformation


def main():
    data_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\Daba_Data\data_20251215\data_20251215_4yi\20251212"
    save_path = data_path + "_results"
    os.makedirs(save_path, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    file_list = sorted(os.listdir(data_path))
    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = os.path.join(data_path, f)
        img = cv2.imread(f_abs_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blured = cv2.GaussianBlur(gray, (5, 5), 0)
        blured = cv2.GaussianBlur(blured, (5, 5), 0)
        blured = cv2.GaussianBlur(blured, (5, 5), 0)
        blured = cv2.medianBlur(blured, 5)
        equalized = clahe.apply(blured)
        # log_out = log_transformation(equalized)


        f_dst_path1 = os.path.join(save_path, "{}_equalized1.jpg".format(fname))
        # f_dst_path2 = os.path.join(save_path, "{}_log_out.jpg".format(fname))


        cv2.imwrite(f_dst_path1, equalized)
        # cv2.imwrite(f_dst_path2, log_out)






if __name__ == '__main__':
    main()

