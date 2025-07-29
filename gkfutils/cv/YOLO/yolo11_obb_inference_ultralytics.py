import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
import cv2
import numpy as np


def inference():
    model = YOLO(r"D:\Gosion\code\others\Python\ultralytics-8.3.72\weights\yolo11s-obb.pt", task='obb')  # load a custom model
    img_path = r"F:\ultralytics\datasets\VisDrone\VisDrone2019-DET-val\images\0000026_01000_d_0000026.jpg"
    img = cv2.imread(img_path)
    results = model.predict(img, imgsz=640)

    for result in results:
        obb = result.obb
        det_num = len(obb)

        xyxyxyxy = obb.xyxyxyxy.cpu().numpy()
        cls = obb.cls.cpu().numpy()
        conf = obb.conf.cpu().numpy()

        for i in range(det_num):
            p0 = list(map(round, xyxyxyxy[i][0].tolist()))
            p1 = list(map(round, xyxyxyxy[i][1].tolist()))
            p2 = list(map(round, xyxyxyxy[i][2].tolist()))
            p3 = list(map(round, xyxyxyxy[i][3].tolist()))
            cv2.line(img, p0, p1, (0, 0, 255), 2)
            cv2.line(img, p1, p2, (0, 0, 255), 2)
            cv2.line(img, p2, p3, (0, 0, 255), 2)
            cv2.line(img, p3, p0, (0, 0, 255), 2)
            cv2.putText(img, f'Class: {cls[i]}, Conf: {conf[i]:.2f}', (p0[0], p0[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    inference()