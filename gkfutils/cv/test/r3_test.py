import os
import cv2
import numpy as np
from scipy.stats import norm


def main():
    data_path = r"G:\Gosion\projects\010.BeltTornDetectionR3\data\test"
    save_path = r"G:\Gosion\projects\010.BeltTornDetectionR3\data\test_out"
    os.makedirs(save_path, exist_ok=True)

    file_list = sorted(os.listdir(data_path))

    k1 = np.array([
        [-1, -1, 4, -1, -1],
        [-1, -1, 4, -1, -1],
        [4, 4, 4, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]
    ], dtype=np.float32)

    k2 = np.array([
        [-1, -1, 4, -1, -1],
        [-1, -1, 4, -1, -1],
        [-1, -1, 4, 4, 4],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]
    ], dtype=np.float32)

    k3 = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [4, 4, 4, -1, -1],
        [-1, -1, 4, -1, -1],
        [-1, -1, 4, -1, -1]
    ], dtype=np.float32)

    k4 = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, 4, 4, 4],
        [-1, -1, 4, -1, -1],
        [-1, -1, 4, -1, -1]
    ], dtype=np.float32)

    k5 = np.array([
        [-1, -1, -1, -1, 4],
        [-1, -1, -1, 4, -1],
        [4, 4, 4, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]
    ], dtype=np.float32)

    k6 = np.array([
        [4, -1, -1, -1, -1],
        [-1, 4, -1, -1, -1],
        [-1, -1, 4, 4, 4],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]
    ], dtype=np.float32)

    k7 = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [4, 4, 4, -1, -1],
        [-1, -1, -1, 4, -1],
        [-1, -1, -1, -1, 4]
    ], dtype=np.float32)

    k8 = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, 4, 4, 4],
        [-1, 4, -1, -1, -1],
        [4, -1, -1, -1, -1]
    ], dtype=np.float32)

    # k1 = np.array([[-1,  1, -1],
    #                [-1,  1,  1],
    #                [-1, -1, -1]], dtype=np.float32)
    # # 核2：左端点 + 向下突起
    # k2 = np.array([[-1, -1, -1],
    #                [-1,  1,  1],
    #                [-1,  1, -1]], dtype=np.float32)
    # # 核3：右端点 + 向上突起
    # k3 = np.array([[-1,  1, -1],
    #                [ 1,  1, -1],
    #                [-1, -1, -1]], dtype=np.float32)
    # # 核4：右端点 + 向下突起
    # k4 = np.array([[-1, -1, -1],
    #                [ 1,  1, -1],
    #                [-1,  1, -1]], dtype=np.float32)


    for f in file_list:
        img_path = os.path.join(data_path, f)
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_out1 = cv2.filter2D(img_gray, -1, k1)
        img_out2 = cv2.filter2D(img_gray, -1, k2)
        img_out3 = cv2.filter2D(img_gray, -1, k3)
        img_out4 = cv2.filter2D(img_gray, -1, k4)
        img_out5 = cv2.filter2D(img_gray, -1, k5)
        img_out6 = cv2.filter2D(img_gray, -1, k6)
        img_out7 = cv2.filter2D(img_gray, -1, k7)
        img_out8 = cv2.filter2D(img_gray, -1, k8)

        cv2.imwrite(os.path.join(save_path, "k1_" + f), img_out1)
        cv2.imwrite(os.path.join(save_path, "k2_" + f), img_out2)
        cv2.imwrite(os.path.join(save_path, "k3_" + f), img_out3)
        cv2.imwrite(os.path.join(save_path, "k4_" + f), img_out4)
        cv2.imwrite(os.path.join(save_path, "k5_" + f), img_out5)
        cv2.imwrite(os.path.join(save_path, "k6_" + f), img_out6)
        cv2.imwrite(os.path.join(save_path, "k7_" + f), img_out7)
        cv2.imwrite(os.path.join(save_path, "k8_" + f), img_out8)

def z_score(x, eps=1e-9):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + eps)



def main2():
    img_path = r"G:\Gosion\projects\010.BeltTornDetectionR3\data\test\_home_sy_MVviewer_pictures_Pic_2026_03_06_141328_6_2.bmp"
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imwrite(r"G:\Gosion\projects\010.BeltTornDetectionR3\data\test\_home_sy_MVviewer_pictures_Pic_2026_03_06_141328_6_2.bmp_binary.png", binary)

    rows_sum = np.sum(gray, axis=1)
    print(rows_sum)

    median_value = np.median(rows_sum)
    print(median_value)

    z_thresh = norm.ppf((1 + 0.99) / 2)
    zscore_thr1 = -z_thresh
    zscore_thr2 = z_thresh
    print(zscore_thr1, zscore_thr2)

    z = z_score(rows_sum)
    print(z)

    print(abs(z) > zscore_thr2)

    y_coords = np.where(abs(z) > zscore_thr2)
    print(y_coords)

    y_coords_mean = np.mean(y_coords)
    print(y_coords_mean)

    cv2.line(img, (0, int(y_coords_mean)), (img.shape[1], int(y_coords_mean)), (255, 0, 255), 2)

    cv2.imwrite(r"G:\Gosion\projects\010.BeltTornDetectionR3\data\test\_home_sy_MVviewer_pictures_Pic_2026_03_06_141328_6_2.bmp_out.png", img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







if __name__ == '__main__':
    # main()
    main2()