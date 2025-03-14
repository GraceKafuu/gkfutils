import os
import cv2
import numpy as np
import sys
import threading
import yaml


lock = threading.Lock()


class YOLOv5_CV2(object):
    """
    opencv-contrib-python  4.9.0.80
    opencv-python          4.9.0.80
    opencv-python-headless 4.9.0.80
    """
    def __init__(self, config_path, detection_name, cuda=False):
        self.config_path = config_path
        self.detection_name = detection_name
        self.config = self.get_config()
        self.weight_path = self.config["{}".format(self.detection_name)]["path"]["det_model_path"].replace("\\", "/")
        self.class_path = self.config["{}".format(self.detection_name)]["path"]["det_class_path"].replace("\\", "/")
        self.net = self.build_model(cuda)
        self.class_list = self.load_classes()
        self.cuda = cuda
        self.INPUT_WIDTH = int(self.weight_path.split("_")[-3])
        self.INPUT_HEIGHT = int(self.weight_path.split("_")[-2])
        self.score_threshold = self.config["{}".format(self.detection_name)]["threshold"]["score_threshold"]
        self.nms_threshold = self.config["{}".format(self.detection_name)]["threshold"]["nms_threshold"]
        self.confidence_threshold = self.config["{}".format(self.detection_name)]["threshold"]["confidence_threshold"]

    def get_config(self):
        with open(self.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
    
    def build_model(self, cuda):
        net = cv2.dnn.readNet(self.weight_path)
        if cuda:
            print("Attempting to use CUDA")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def inference(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        return preds

    def load_classes(self):
        class_list = []
        with open(self.class_path, "r", encoding="utf-8") as f:
            class_list = [cname.strip() for cname in f.readlines()]
        return class_list

    def wrap_detection(self, input_image, output_data):
        if not isinstance(output_data, np.ndarray):
            raise TypeError("Output data should be a NumPy array")

        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]  # Use the first dimension as rows
        image_width, image_height, _ = input_image.shape
        x_factor = image_width / self.INPUT_WIDTH
        y_factor = image_height / self.INPUT_HEIGHT

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= self.confidence_threshold:
                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if classes_scores[class_id] > self.score_threshold:
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
        for class_id in unique_class_ids:
            class_mask = np.array([i == class_id for i in class_ids])
            class_boxes = boxes[class_mask]
            class_confidences = np.array(confidences)[class_mask]
            indexes = cv2.dnn.NMSBoxes(class_boxes.tolist(), class_confidences.tolist(), self.score_threshold, self.nms_threshold)
            for i in indexes:
                result_confidences.append(class_confidences[i])
                result_class_ids.append(class_id)
                result_boxes.append(class_boxes[i])

        return result_class_ids, result_confidences, result_boxes

    def format_yolov5(self, frame):
        row, col, _ = frame.shape
        max_dim = max(col, row)
        result = np.zeros((max_dim, max_dim, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def detect(self, src, SCORE_THRESHOLD, NMS_THRESHOLD, CONFIDENCE_THRESHOLD):
        self.score_threshold = SCORE_THRESHOLD
        self.nms_threshold = NMS_THRESHOLD
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        class_list = self.load_classes()
        inputImage = self.format_yolov5(src)
        lock.acquire()
        outs = self.inference(inputImage)
        lock.release()
        class_ids, confidences, boxes = self.wrap_detection(inputImage, outs[0])
        name = []
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = np.random.choice(256, size=3)
            cv2.rectangle(src, box, color, 2)
            cv2.rectangle(src, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(src, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
            name.append(class_list[classid])

        return src, class_ids, confidences, boxes



class YOLO(YOLOv5_CV2):
    def __init__(self, config_path, detection_name, cuda=False):
        super(YOLO, self).__init__(config_path, detection_name, cuda)

    def cal_iou(self, bbx1, bbx2):
        """
        b1 = [0, 0, 10, 10]
        b2 = [2, 2, 12, 12]
        iou = cal_iou(b1, b2)  # 0.47058823529411764

        p --> bbx1
        q --> bbx2
        :param bbx1:
        :param bbx2:
        :return:
        """

        px1, py1, px2, py2 = bbx1[0], bbx1[1], bbx1[2], bbx1[3]
        qx1, qy1, qx2, qy2 = bbx2[0], bbx2[1], bbx2[2], bbx2[3]
        area1 = abs(px2 - px1) * abs(py2 - py1)
        area2 = abs(qx2 - qx1) * abs(qy2 - qy1)

        # cross point --> c
        cx1 = max(px1, qx1)
        cy1 = max(py1, qy1)
        cx2 = min(px2, qx2)
        cy2 = min(py2, qy2)

        cw = cx2 - cx1
        ch = cy2 - cy1
        if cw <= 0 or ch <= 0:
            return 0

        carea = cw * ch
        iou = carea / (area1 + area2 - carea)
        return iou
    
    def xywh2xyxy(self, x):
        y = [x[0], x[1], x[0] + x[2], x[1] + x[3]]
        return y

    def smoking_predict(self, img, SCORE_THRESHOLD, NMS_THRESHOLD, CONFIDENCE_THRESHOLD):
        self.score_threshold = SCORE_THRESHOLD
        self.nms_threshold = NMS_THRESHOLD
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        class_list = self.load_classes()
        inputImage = self.format_yolov5(img)
        lock.acquire()
        outs = self.inference(inputImage)
        lock.release()
        class_ids, confidences, boxes = self.wrap_detection(inputImage, outs[0])

        assert len(class_ids) == len(confidences) and len(boxes) == len(confidences), "Assertion Error!"
        targets = []
        flag = False

        for i in range(len(class_ids)):
            class_id_i = class_ids[i]
            confidence_i = confidences[i]
            box_i = boxes[i]
            box_i_xyxy = self.xywh2xyxy(box_i)
            
            if class_id_i == 1:
                for j in range(len(class_ids)):
                    class_id_j = class_ids[j]
                    confidence_j = confidences[j]
                    box_j = boxes[j]
                    box_j_xyxy = self.xywh2xyxy(box_j)

                    if class_id_j == 0:
                        iou = self.cal_iou(box_i_xyxy, box_j_xyxy)
                        target = [class_id_i, confidence_i, box_i]
                        if iou > 0 and target not in targets:
                            targets.append(target)
                    
        if len(targets) > 0:
            flag = True
            for t in targets:
                color = colors[int(t[0]) % len(colors)]
                cv2.rectangle(img, t[2], color, 2)
                cv2.rectangle(img, (t[2][0], t[2][1] - 20), (t[2][0] + t[2][2], t[2][1]), color, -1)
                cv2.putText(img, class_list[t[0]], (t[2][0], t[2][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        return flag, img


yolo = YOLO(config_path=r"gkfutils/cv/gosion/config.yaml", detection_name="smoking_detection", cuda=False)


def detect(img, pts=None, score="0.6, 0.45, 0.6"):
    s = score.replace(" ", "").split(",")
    SCORE_THRESHOLD = float(s[0])
    NMS_THRESHOLD = float(s[1])
    CONFIDENCE_THRESHOLD = float(s[2])

    pts = np.array(pts.replace(" ", "").split(","), dtype="float32")
    pts = pts.reshape(-1, 2)
        
    # pts = pts.reshape((-1, 2))
    # if pts.shape[0] == 1:
    #     img = img
    # elif pts.shape[0] == 4:
    #     img = four_point_crop(img, pts)
    # else:
    #     img = mask_outside_polygon(img, pts)

    flag, img = yolo.smoking_predict(img, SCORE_THRESHOLD, NMS_THRESHOLD, CONFIDENCE_THRESHOLD)

    return flag, img


if __name__ == "__main__":
    # img_path = r"D:\Gosion\Projects\002.Smoking_Det\data\v4\val\images\0000044.jpg"
    # img_path = r"D:\Gosion\Projects\data\images\192.168.22.220_01_20240423105134225.jpg"
    img_path = r"D:\Gosion\Projects\data\images\192.168.22.220_01_20240423105140801.jpg"
    img = cv2.imread(img_path)
    flag, img = detect(img, pts="0, 0, 0, 0, 0, 0, 0, 0", score="0.6, 0.45, 0.6")

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


