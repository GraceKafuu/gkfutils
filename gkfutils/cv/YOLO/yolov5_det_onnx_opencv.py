import os
import cv2
import yaml
import numpy as np
import onnxruntime as ort
from time import perf_counter
from typing import List
import time
from pathlib import Path

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


coco_class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                    "toothbrush"]

# generate different colors for coco classes
colors = np.random.uniform(0, 255, size=(len(coco_class_names), 3))


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.45


class YOLOv5_CV2:
    def __init__(self, model_path, imgsz=(640, 640), conf=0.60, iou=0.45, device='cpu'):
        self.model_path = model_path
        self.imgsz = imgsz  # (height, width)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.cuda = (device == 'gpu')
        self.net = self.build_model(self.model_path)

    def build_model(self, model_path: str) -> cv2.dnn_Net:
        """
        build the model with opencv dnn module
        Args:
            model_path: the path of the model, the model should be in onnx format

        Returns:
            the model object
        """
        # check if the model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"model file {model_path} not found")
        model = cv2.dnn.readNetFromONNX(model_path)

        # check if the opencv-python in your environment supports cuda
        cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

        if cuda_available:  # if cuda is available, use cuda
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:  # if cuda is not available, use cpu
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return model

    def inference(self, image: np.ndarray, model: cv2.dnn_Net) -> np.ndarray:
        """
        inference the model with the input image
        Args:
            image: the input image in numpy array format, the shape should be (height, width, channel),
            the color channels should be in GBR order, like the original opencv image format
            model: the model object

        Returns:
            the output data of the model, the shape should be (1, 25200, nc+5), nc is the number of classes
        """
        # image preprocessing, include resize, normalization, channel swap like BGR to RGB, and convert to blob format
        # get a 4-dimensional Mat with NCHW dimensions order.
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)

        # the alternative way to get the blob
        # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # input_image = cv2.resize(src=rgb, dsize=(INPUT_WIDTH, INPUT_HEIGHT))
        # blob_img = np.float32(input_image) / 255.0
        # input_x = blob_img.transpose((2, 0, 1))
        # blob = np.expand_dims(input_x, 0)

        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # set the input data
        model.setInput(blob)

        start = time.perf_counter()
        # inference
        outs = model.forward()

        end = time.perf_counter()

        print("inference time: ", end - start)

        # the shape of the output data is (1, 25200, nc+5), nc is the number of classes
        return outs

    def xywh_to_xyxy(self, bbox_xywh, image_width, image_height):
        """
        Convert bounding box coordinates from (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max) format.

        Parameters:
            bbox_xywh (list or tuple): Bounding box coordinates in (center_x, center_y, width, height) format.
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            tuple: Bounding box coordinates in (x_min, y_min, x_max, y_max) format.
        """
        center_x, center_y, width, height = bbox_xywh
        x_min = max(0, int(center_x - width / 2))
        y_min = max(0, int(center_y - height / 2))
        x_max = min(image_width - 1, int(center_x + width / 2))
        y_max = min(image_height - 1, int(center_y + height / 2))
        return x_min, y_min, x_max, y_max

    def wrap_detection(
            self, 
            input_image: np.ndarray,
            output_data: np.ndarray,
            labels: List[str],
            confidence_threshold: float = 0.6
        ):
        # the shape of the output_data is (25200,5+nc),
        # the first 5 elements are [x, y, w, h, confidence], the rest are prediction scores of each class

        image_width, image_height, _ = input_image.shape
        x_factor = image_width / INPUT_WIDTH
        y_factor = image_height / INPUT_HEIGHT

        # transform the output_data[:, 0:4] from (x, y, w, h) to (x_min, y_min, x_max, y_max)

        indices = cv2.dnn.NMSBoxes(output_data[:, 0:4].tolist(), output_data[:, 4].tolist(), 0.6, 0.4)

        raw_boxes = output_data[:, 0:4][indices]
        raw_confidences = output_data[:, 4][indices]
        raw_class_prediction_probabilities = output_data[:, 5:][indices]

        criteria = raw_confidences > confidence_threshold
        raw_class_prediction_probabilities = raw_class_prediction_probabilities[criteria]
        raw_boxes = raw_boxes[criteria]
        raw_confidences = raw_confidences[criteria]

        bounding_boxes, confidences, class_ids = [], [], []
        for class_prediction_probability, box, confidence in zip(raw_class_prediction_probabilities, raw_boxes,
                                                                raw_confidences):
            #
            # find the least and most probable classes' indices and their probabilities
            # min_val, max_val, min_loc, mac_loc = cv2.minMaxLoc(class_prediction_probability)
            most_probable_class_index = np.argmax(class_prediction_probability)
            label = labels[most_probable_class_index]
            confidence = float(confidence)

            # bounding_boxes.append(box)
            # confidences.append(confidence)
            # class_ids.append(most_probable_class_index)

            x, y, w, h = box
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            bounding_box = [left, top, width, height]
            bounding_boxes.append(bounding_box)
            confidences.append(confidence)
            class_ids.append(most_probable_class_index)

        return class_ids, confidences, bounding_boxes

    def detect(self, frame, show=False):
        start = time.perf_counter()
        # image preprocessing, the trick is to make the frame to be a square but not twist the image
        row, col, _ = frame.shape  # get the row and column of the origin frame array
        _max = max(row, col)  # get the max value of row and column
        input_image = np.zeros((_max, _max, 3), dtype=np.uint8)  # create a new array with the max value
        input_image[:row, :col, :] = frame  # paste the original frame  to make the input_image to be a square
        # inference
        output_data = self.inference(input_image, self.net)  # the shape of output_data is (1, 25200, 85)

        # 4. wrap the detection result
        class_ids, confidences, boxes = self.wrap_detection(input_image, output_data[0], coco_class_names)

        # 5. draw the detection result on the frame
        for (class_id, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(class_id) % len(colors)]
            label = coco_class_names[int(class_id)]

            xmin, ymin, width, height = box
            cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), color, 2)
            # cv2.rectangle(frame, box, color, 2)
            # cv2.rectangle(frame, [box[0], box[1], box[2], box[3]], color, thickness=2)

            # cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + 100, box[1]), color, -1)
            cv2.putText(frame, str(label), (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, "{:.2f}".format(confidence), (box[0] + 120, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        finish = time.perf_counter()
        FPS = round(1.0 / (finish - start), 2)
        cv2.putText(frame, str(FPS), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 6. show the frame
        if show:
            cv2.imshow("Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return frame

    def detect_folder(self, imgs_path, out_path):
        os.makedirs(out_path, exist_ok=True)
        for fname in os.listdir(imgs_path):
            if fname.lower().endswith(('.jpg','.png','.jpeg')):
                img = cv2.imread(os.path.join(imgs_path, fname))
                out = self.detect(img)
                cv2.imwrite(os.path.join(out_path, fname), out)



if __name__=="__main__":
    """
    问题记录：
    1.使用直接导出的yolov5s.onnx, opset为10-16都跑不起来, 报错信息如下:
    [ERROR:0@0.570] global onnx_importer.cpp:1034 cv::dnn::dnn4_v20231225::ONNXImporter::handleNode DNN/ONNX: ERROR during processing node with 1 inputs and 1 outputs: [Floor]:(onnx_node!/model.11/Floor) from domain='ai.onnx'
    Traceback (most recent call last):
    File "d:\GraceKafuu\Python\github\gkfutils\gkfutils\cv\YOLO\yolov5_det_onnx_opencv.py", line 167, in <module>
        yolov5 = YOLOv5_CV2(model_path, conf=0.60)
    File "d:\GraceKafuu\Python\github\gkfutils\gkfutils\cv\YOLO\yolov5_det_onnx_opencv.py", line 36, in __init__
        self.net = self.build_model(self.cuda)
    File "d:\GraceKafuu\Python\github\gkfutils\gkfutils\cv\YOLO\yolov5_det_onnx_opencv.py", line 52, in build_model
        net = cv2.dnn.readNetFromONNX(model_path)
    cv2.error: OpenCV(4.9.0) D:\a\opencv-python\opencv-python\opencv\modules\dnn\src\onnx\onnx_importer.cpp:1053: error: (-2:Unspecified error) in function 'cv::dnn::dnn4_v20231225::ONNXImporter::handleNode'
    > Node [Floor@ai.onnx]:(onnx_node!/model.11/Floor) parse error: OpenCV(4.9.0) D:\a\opencv-python\opencv-python\opencv\modules\dnn\src\layers\elementwise_layers.cpp:261: error: (-215:Assertion 
    failed) src.size == dst.size && src.type() == dst.type() && src.isContinuous() && dst.isContinuous() && src.type() == CV_32F in function 'cv::dnn::ElementWiseLayer<struct cv::dnn::FloorFunctor>::forward'
    >
    
    2. 需要执行 python -m onnxsim yolov5s.onnx yolov5s_sim.onnx, 使用 yolov5s_sim.onnx
    pip install onnx-simplifier
    python -m onnxsim yolov5s.onnx yolov5s_sim.onnx
    """
    model_path = r"G:\Gosion\code\Python\yolov5-master\yolov5s_sim.onnx"
    img_path = r"G:\Gosion\projects\000.Test_Data\images\192.168.22.220_01_20240423104805321.jpg"
    
    yolov5 = YOLOv5_CV2(model_path, conf=0.60)
    img = cv2.imread(img_path)

    out = yolov5.detect(img, show=True)

    # imgs_path = r"G:\Gosion\data\000.Test_Data\images\detect_test"
    # save_path = r"G:\Gosion\data\000.Test_Data\images\detect_test_results2"
    # yolov5.detect_folder(imgs_path, save_path)
