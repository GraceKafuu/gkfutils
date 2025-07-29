import os
import cv2
import sys
import yaml
import time
import threading
import numpy as np
import onnxruntime as ort
import torch
import torchvision
from PIL import Image
from time import perf_counter


# wrapper functions to execute time of functions
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        # count time (ms)
        exec_time = (perf_counter() - start_time) * 1000
        print(f"Execution time: {exec_time:.2f} ms")
        return result
    return wrapper


class YOLOv5_CV2(object):
    """
    opencv-contrib-python  4.9.0.80
    opencv-python          4.9.0.80
    opencv-python-headless 4.9.0.80
    存在的问题: 有的模型加载失败, 可能的原因是训练的时候opencv, torch, onnx等的版本也需是特定的...
    """
    def __init__(self, model_path, imgsz=(640, 640), conf=0.60, iou=0.45, device='cpu'):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.cuda = True if self.device == "gpu" else False

        self.net = self.build_model(self.cuda)
        # self.class_list = self.load_classes()

    def get_config(self):
        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
    
    def load_classes(self):
        class_list = []
        with open(self.class_path, "r", encoding="utf-8") as f:
            class_list = [cname.strip() for cname in f.readlines()]
        return class_list
    
    def build_model(self, cuda):
        net = cv2.dnn.readNet(self.model_path)
        if cuda:
            print("Attempting to use CUDA")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    
    def preprocess(self, img):
        self.origsz = img.shape[:2]
        img = img.astype(np.uint8)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, self.imgsz[::-1], swapRB=True, crop=False)
        return blob
    
    @timeit
    def inference(self, blob):
        self.net.setInput(blob)
        preds = self.net.forward()
        return preds

    def postprocess(self, pred):
        if not isinstance(pred, np.ndarray):
            raise TypeError("Output data should be a NumPy array")

        class_ids = []
        confidences = []
        boxes = []

        rows = pred.shape[0]  # Use the first dimension as rows
        x_factor = self.origsz[1] / self.imgsz[1]
        y_factor = self.origsz[0] / self.imgsz[0]

        for r in range(rows):
            row = pred[r]
            confidence = row[4]
            if confidence >= self.conf:
                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        boxes = np.array(boxes)

        unique_class_ids = np.unique(class_ids)
        result_class_ids = []
        result_confidences = []
        result_boxes = []

        results = []

        for class_id in unique_class_ids:
            class_mask = np.array([i == class_id for i in class_ids])
            class_boxes = boxes[class_mask]
            class_confidences = np.array(confidences)[class_mask]
            indexes = cv2.dnn.NMSBoxes(class_boxes.tolist(), class_confidences.tolist(), self.conf, self.iou)
            for i in indexes:
                # result_confidences.append(class_confidences[i])
                # result_class_ids.append(class_id)
                # result_boxes.append(class_boxes[i])
                resulti = [class_boxes[i], class_id, class_confidences[i]]
                results.append(resulti)

        return results
    
    def vis_results(self, img, results):
        for pts, label, prob in results:
            cv2.rectangle(img, (pts[0], pts[1]), (pts[2], pts[3]), (0,255,0), 2) 
            cv2.putText(img, str(label) + ' ' + str(prob), (pts[0], pts[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)

        return img

    def detect(self, img0, show=False):
        img = self.preprocess(img0)
        pred = self.inference(img)
        results = self.postprocess(pred[0])
        out = self.vis_results(img0, results)

        if show:
            cv2.imshow("out", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return out
    
    def detect_folder(self, imgs_path, out_path):
        file_list = sorted(os.listdir(imgs_path))
        if not out_path or out_path is None:
            out_path = "./output"
        os.makedirs(out_path, exist_ok=True)

        for f in file_list:
            f_abs_path = imgs_path + "/{}".format(f)
            f_dst_path = out_path + "/{}".format(f)
            img = cv2.imread(f_abs_path)
            out = self.detect(img)
            cv2.imwrite(f_dst_path, out)



if __name__=="__main__":
    # onnx_path = r"D:\Gosion\code\others\Python\yolov5-master\yolov5s.onnx"
    onnx_path = r"D:\Gosion\code\gitee\GuanWangLNG\src\192.168.1.5\weights\hand_detection\hand_det_yolov5s_640_640_v1.0.0.onnx"
    img_pth = r"G:\Gosion\data\004.Out_GuardArea_Det\resources\20250725\plan_light_1933425297818984450.jpg"
    
    yolov5 = YOLOv5_CV2(onnx_path, conf=0.01)
    img = cv2.imread(img_pth)

    out = yolov5.detect(img, show=True)

    # imgs_path = r"G:\Gosion\data\000.Test_Data\images\detect_test"
    # save_path = r"G:\Gosion\data\000.Test_Data\images\detect_test_results2"
    # yolov5.detect_folder(imgs_path, save_path)