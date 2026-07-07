import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.ndimage import label

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import onnxruntime as ort
import math


class YOLO11:
    """YOLOv8 YOLO11 object detection model class for handling ONNX inference and visualization.

    This class provides functionality to load a YOLOv8 YOLO11 ONNX model, perform inference on images, and visualize the
    detection results with bounding boxes and labels.

    Attributes:
        model_path (str): Path to the ONNX model file.
        input_image (str): Path to the input image file.
        conf_threshold (float): Confidence threshold for filtering detections.
        iou_threshold (float): IoU threshold for non-maximum suppression.
        classes (list[str]): List of class names from the COCO dataset.
        color_palette (np.ndarray): Random color palette for visualizing different classes.
        input_width (int): Width dimension of the model input.
        input_height (int): Height dimension of the model input.
        img (np.ndarray): The loaded input image.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.

    Methods:
        letterbox: Resize and reshape images while maintaining aspect ratio by adding padding.
        draw_detections: Draw bounding boxes and labels on the input image based on detected objects.
        preprocess: Preprocess the input image before performing inference.
        postprocess: Perform post-processing on the model's output to extract and visualize detections.
        main: Perform inference using an ONNX model and return the output image with drawn detections.
    """

    def __init__(self, model_path: str, conf_threshold: float, iou_threshold: float):
        """Initialize an instance of the YOLOv8 YOLO11 class.

        Args:
            model_path (str): Path to the ONNX model.
            conf_threshold (float): Confidence threshold for filtering detections.
            iou_threshold (float): IoU threshold for non-maximum suppression.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load the class names from the COCO dataset
        # self.classes = YAML.load(check_yaml("coco8.yaml"))["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(80, 3))

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(self.model_path, providers=providers or available)

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

    def letterbox(self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            img (np.ndarray): Resized and padded image.
            pad (tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: list[float], score: float, class_id: int, color: tuple[int]) -> None:
        """Draw bounding boxes and labels on the input image based on the detected objects."""
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        if color is None:
            color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        # label = f"{self.classes[class_id]}: {score:.2f}"
        label = f"{class_id}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img) -> tuple[np.ndarray, tuple[int, int]]:
        """Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            image_data (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            pad (tuple[int, int]): Padding values (top, left) applied during letterboxing.
        """
        assert isinstance(img, np.ndarray), f'input is not np.ndarray and is {type(input)}!'
        # Read the input image using OpenCV
        # self.img = cv2.imread(self.input_image)
        self.img = img

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img, pad = self.letterbox(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = image_data[None].astype(np.float32)

        # Return the preprocessed image data
        return image_data, pad

    def postprocess(self, output: list[np.ndarray], pad: tuple[int, int]) -> np.ndarray:
        """Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs. It applies
        non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (list[np.ndarray]): The output arrays from the model.
            pad (tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.conf_threshold:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        out = []
        # Iterate over the selected indices after non-maximum suppression
        for i in np.array(indices).flatten():
            # Get the box, score, and class ID corresponding to the index
            box = boxes[int(i)]
            score = scores[int(i)]
            class_id = class_ids[int(i)]
            box.extend([score, class_id])
            out.append(box)

        return out
    
    def visualize(self, img, output, color=None):
        for b in output:
            self.draw_detections(img, b[:4], b[4], b[5], color)
        
        return img
    
    def detect(self, img, show=False, color=None):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        
        # Preprocess the image data
        img_data, pad = self.preprocess(img)
        # Run inference using the preprocessed image data
        t1 = time.time()
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        t2 = time.time()
        print("inference time: ", t2 - t1)
        # Perform post-processing on the outputs to obtain output image
        output = self.postprocess(outputs, pad)

        if show:
            for b in output:
                self.draw_detections(img, b[:4], b[4], b[5], color)
            cv2.imshow("detect", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return output


def yolo11_inference_example():
    # model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\yolo11s.onnx"
    # img_path = r"G:\Gosion\projects\000.Test_Data\images\192.168.22.220_01_20240423104805321.jpg"

    model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\detect\helmet_head_v4\weights\best.onnx"
    img_path = r"G:\Gosion\projects\011.ZhongyuanGuanxi\data\helmet_test.jpg"
    # img_path = r"G:\Gosion\projects\011.ZhongyuanGuanxi\data\helmet_workCloth\plan_light_2054754412798361601.jpg"

    # model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\detect\train3\weights\best.onnx"
    # img_path = r"G:\Gosion\projects\009.TuoGun_Det\videos\frames\20251221_merged\20251221_101111_output_000000001.jpg"

    img = cv2.imread(img_path)

    yolo11 = YOLO11(model_path, 0.60, 0.45)
    output = yolo11.detect(img, show=True, color=None)
    # print(output)

    # for i in range(30):
    #     output = yolo11.detect(img, show=False, color=None)
    #     # print(output)


def yolo11_inference_dir_example():
    model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\detect\train5\weights\best.onnx"

    data_path = r"G:\Gosion\projects\009.TuoGun_Det\resources\images\XianChang\data_20260216"
    dir_list = os.listdir(data_path)

    save_path = r"G:\Gosion\projects\009.TuoGun_Det\resources\images\XianChang\output_data_20260216"
    no_result_path = os.path.join(save_path, "no_result")
    result_path = os.path.join(save_path, "result")
    os.makedirs(no_result_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    for d in dir_list:
        d_path = os.path.join(data_path, d)

        dd_list = os.listdir(d_path)
        for dd in dd_list:
            dd_path = os.path.join(d_path, dd)
            file_list = sorted(os.listdir(dd_path))

            for f in file_list:
                fname = os.path.splitext(f)[0]
                img_path = dd_path + "/{}".format(f)

                # img = cv2.imread(img_path)
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

                yolo11 = YOLO11(model_path, 0.60, 0.45)
                output = yolo11.detect(img, show=False, color=None)
                img_vis = yolo11.visualize(img, output)
                
                if len(output) > 0:
                    img_save_path = result_path + "/{}_{}_{}_{}.jpg".format(d, fname, len(output), time.time())
                else:
                    img_save_path = no_result_path + "/{}_{}_{}_{}.jpg".format(d, fname, len(output), time.time())
                
                cv2.imencode('.jpg', img_vis)[1].tofile(img_save_path)




class LaserDetectV1:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = self.session.get_inputs()[0].shape
        self.ouput_size = self.session.get_outputs()[0].shape

        self.imgsz = self.input_size[2:]

    def pil2cv(self, image):
        assert isinstance(image, Image.Image), f'Input image type is not PIL.image and is {type(image)}!'
        if len(image.split()) == 1:
            return np.asarray(image)
        elif len(image.split()) == 3:
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        elif len(image.split()) == 4:
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGR)
        else:
            return None
    
    def preprocess(self, input):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        if isinstance(input, str):
            img = cv2.imread(input)
        elif isinstance(input, Image.Image):
            img = self.pil2cv(input)
        else:
            assert isinstance(input, np.ndarray), f'input is not np.ndarray and is {type(input)}!'
            img = input

        # Get the height and width of the input image
        self.origsize = img.shape[:2]

        # Convert the image color space from BGR to RGB
        # img = self.img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, self.imgsz[::-1])
        # img = pre_transform(img)

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data


    def inference(self, input):
        ort_outs = self.session.run(None, {self.input_name: input})
        return ort_outs
    
    def postprocess(self, ort_outs):
        rh = 64 / self.origsize[0]
        rw = 256 / self.origsize[1]

        w_gird = np.linspace(0, self.imgsz[1], 256)

        output_rows = ort_outs[0][0, :]
        # print(output_rows)

        output_rows_orig = output_rows / rh
        output_cols_orig = w_gird / rw
        # print(output_rows_orig)

        out = []

        for i in range(256):
            pi = (round(output_cols_orig[i]), round(output_rows_orig[i]))
            # cv2.circle(img_orig, pi, 5, (255, 0, 255), -1)
            out.append(pi)

        return out
    
    def detect(self, img, vis=False):
        img_np = self.preprocess(img)
        ort_outs = self.inference(img_np)
        out = self.postprocess(ort_outs)

        if vis:
            img_vis = self.visualize(img, out)
            cv2.imshow("img_vis", img_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return np.asarray(out)
    
    def visualize(self, img, out):
        for i in range(len(out)):
            cv2.circle(img, out[i], 5, (255, 0, 255), -1)

        return img
    

class LaserDetectV2:
    def __init__(self, model_path, num_cols=256):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = self.session.get_inputs()[0].shape
        self.ouput_size = self.session.get_outputs()[0].shape

        self.imgsz = self.input_size[2:]
        self.num_cols = num_cols

    def preprocess(self, img):
        self.ori_img = img.copy()
        height, width, _ = img.shape
        self.origsize = (height, width)
        
        # 与训练相同的 Transform (不含增强)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(input_tensor).unsqueeze(0)

        return input_tensor.numpy()

    def inference(self, input):
        ort_outs = self.session.run(None, {self.input_name: input})
        return ort_outs
    
    def postprocess(self, ort_outs):
        # 4. 后处理：结合固定的 X 坐标，并反归一化回原图尺寸
        points = []
        for col_idx in range(self.num_cols):
            # 还原 X 归一化坐标
            col_mid_x = (col_idx + 0.5) / self.num_cols
            
            # 预测的 Y 坐标
            y_val = ort_outs[0][0][col_idx]
            
            # 转换为原图绝对像素坐标
            pixel_x = int(col_mid_x * self.origsize[1])
            pixel_y = int(y_val * self.origsize[0])
            
            points.append((pixel_x, pixel_y))

        return points
    
    def detect(self, img, vis=False):
        img_np = self.preprocess(img)
        ort_outs = self.inference(img_np)
        out = self.postprocess(ort_outs)

        if vis:
            img_vis = self.visualize(img, out)
            cv2.imshow("img_vis", img_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return np.asarray(out)
    
    def visualize(self, img, out):
        for pt in out:
            # 防止预测值轻微越界导致画图崩溃，做一层边界保护
            cx = max(0, min(self.origsize[1] - 1, pt[0]))
            cy = max(0, min(self.origsize[0] - 1, pt[1]))
            cv2.circle(img, (cx, cy), 2, (255, 0, 255), -1)

        return img


# ---------------------------------------------------------
# 3. 网络模型定义 (基于强力的 ResNet 骨干)
# ---------------------------------------------------------
class CurveRegressor(nn.Module):
    def __init__(self, num_outputs=256):
        super().__init__()
        # 使用经典的 ResNet34 作为特征提取器（对坐标回归任务性价比极高）
        # self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 替换最后的分类全连接层为坐标回归层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1), # 防止轻微过拟合
            nn.Linear(512, num_outputs) # 输出 256 个 Y 坐标
        )

    def forward(self, x):
        return self.backbone(x)
    

def test_and_visualize(model_path, test_img_path, num_cols=256, imgsz=(96, 256), show=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = CurveRegressor(num_outputs=num_cols)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. 读取并预处理单张单图
    ori_img = cv2.imread(test_img_path)
    height, width, _ = ori_img.shape
    
    # 与训练相同的 Transform (不含增强)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(input_tensor).unsqueeze(0).to(device) # 增加 Batch 维度
    
    # 3. 模型预测
    with torch.no_grad():
        pred_y = model(input_tensor).squeeze(0).cpu().numpy() # 形状: (256,)
        
    # 4. 后处理：结合固定的 X 坐标，并反归一化回原图尺寸
    points = []
    for col_idx in range(num_cols):
        # 还原 X 归一化坐标
        col_mid_x = (col_idx + 0.5) / num_cols
        
        # 预测的 Y 坐标
        y_val = pred_y[col_idx]
        
        # 转换为原图绝对像素坐标
        pixel_x = int(col_mid_x * width)
        pixel_y = int(y_val * height)
        
        points.append((pixel_x, pixel_y))
        
    # 5. 在原图上绘制预测的曲线点 (绿色)
    img_vis = ori_img.copy()
    for pt in points:
        # 防止预测值轻微越界导致画图崩溃，做一层边界保护
        cx = max(0, min(width - 1, pt[0]))
        cy = max(0, min(height - 1, pt[1]))
        cv2.circle(img_vis, (cx, cy), 2, (255, 0, 255), -1)
        
    # 展示结果
    if show:
        cv2.imshow("Prediction Result", img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_vis


def detect_laser_by_model(img):
    pass


def get_green_color_mask(
        img,
        lower1=(35, 43, 3),
        upper1=(77, 255, 255),
        lower2=(78, 43, 3),
        upper2=(95, 255, 255)
    ):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义绿色在HSV中的范围
    # 注意：HSV中H通道范围是0-179（OpenCV中）
    if lower1 is None:
        lower1 = np.array([35, 43, 46])   # 绿色范围下限
    if upper1 is None:
        upper1 = np.array([77, 255, 255]) # 绿色范围上限

    mask1 = cv2.inRange(hsv, lower1, upper1)
    
    # 可选：添加第二个绿色范围（针对亮绿色）
    if lower2 is None:
        lower2 = np.array([78, 43, 46])   # 另一种绿色范围
    if upper2 is None:
        upper2 = np.array([95, 255, 255])

    mask2 = cv2.inRange(hsv, lower2, upper2)
    
    # 合并两个掩膜
    mask = cv2.bitwise_or(mask1, mask2)

    return mask


def detect_laser_by_color(img, vis=False):
    mask = get_green_color_mask(img)
    out = cv2.bitwise_and(img, img, mask=mask)

    if vis:
        cv2.imshow("mask", mask)
        cv2.imshow("out", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out
    


def laserDetectByThreshStatistics(img: np.ndarray) -> list:
    dstPts = []
    rows, cols = img.shape[:2]

    # 只处理 y: 4 ~ rows-4
    y_valid = np.arange(4, rows - 4, dtype=np.int32)
    x_valid = np.arange(cols, dtype=np.int32)

    # 取出 4 行切片（向量化！）
    row_m2 = img[y_valid - 2][:, x_valid]
    row_m1 = img[y_valid - 1][:, x_valid]
    row_0  = img[y_valid][:, x_valid]
    row_p1 = img[y_valid + 1][:, x_valid]

    # 
    cond1 = (row_m2 == 255) & (row_m1 == 255) & (row_0 == 0) & (row_p1 == 0)
    cond2 = (row_m2 == 0) & (row_m1 == 0) & (row_0 == 255) & (row_p1 == 255)
    mask = cond1 | cond2

    # 广播 y 坐标
    y_grid = np.broadcast_to(y_valid[:, None], mask.shape)

    # 按列求和 & 计数
    sumY = np.sum(np.where(mask, y_grid, 0), axis=0)
    count = np.sum(mask, axis=0)

    # keep = count >= 4
    keep = count >= 2
    x_keep = x_valid[keep]
    y_keep = sumY[keep] // count[keep]

    # 输出点
    dstPts.extend([(x, y) for x, y in zip(x_keep, y_keep)])
    return dstPts


def fit_plot(img, model, r=4, color=(255, 0, 255)):
    """
    a, b, c = model
    print(f"拟合的抛物线方程: y = {a:.3f}x² + {b:.3f}x + {c:.3f}")
    """
    if model is None: return img

    vis = img.copy()
    imgsz = img.shape[:2]
    x = np.linspace(0, imgsz[1], imgsz[1] - 1)
    y = np.polyval(model, x)

    for i in range(len(x)):
        cv2.circle(vis, (int(x[i]), int(y[i])), r, color, -1)

    return vis


def compute_curvature(curve):
    """
    计算有序离散点集每个点的曲率
    :param curve: list, 有序点集 [(x0,y0), (x1,y1), ...]
    :return: list, 每个点对应的曲率值，长度与curve一致
    """
    n = len(curve)
    curvature = [0.0] * n
    if n < 3:
        return curvature

    for i in range(1, n - 1):
        # 当前点、前一个点、后一个点
        x_prev, y_prev = curve[i-1]
        x_curr, y_curr = curve[i]
        x_next, y_next = curve[i+1]

        # 一阶差分 (dx, dy)
        dx1 = x_curr - x_prev
        dy1 = y_curr - y_prev
        dx2 = x_next - x_curr
        dy2 = y_next - y_curr

        # 一阶导、二阶导近似
        dx = (dx1 + dx2) / 2.0
        dy = (dy1 + dy2) / 2.0
        ddx = dx2 - dx1
        ddy = dy2 - dy1

        # 曲率计算公式
        numerator = abs(dx * ddy - ddx * dy)
        denominator = math.pow(dx**2 + dy**2, 1.5)

        # 避免除0
        if denominator < 1e-6:
            curv = 0.0
        else:
            curv = numerator / denominator

        curvature[i] = curv

    return curvature


def filter_adjacent_points(point_list):
    """
    优化版：先按x坐标连续分组 → 剔除长度≤3的组 → 再筛选相邻点x差>5或y差>5
    :param point_list: 点集列表，格式为 [(x1,y1), (x2,y2), ...]
    :return: 符合条件的相邻点对列表
    """
    if not point_list:
        return []

    # ===================== 第一步：按 x 坐标连续 分组（连通分组）=====================
    sorted_points = sorted(point_list, key=lambda p: p[0])  # 先按x排序，保证连续判断正确
    groups = []
    current_group = [sorted_points[0]]

    for p in sorted_points[1:]:
        last_x = current_group[-1][0]
        current_x = p[0]
        # x坐标连续（差=1）则归为同一组
        if current_x == last_x + 1:
            current_group.append(p)
        else:
            groups.append(current_group)
            current_group = [p]
    # 加入最后一组
    groups.append(current_group)

    # ===================== 第二步：剔除组内元素数量 ≤3 的组 =====================
    valid_groups = [g for g in groups if len(g) > 5]

    # ===================== 第三步：在有效组内执行原有筛选逻辑 =====================
    result = []
    for group in valid_groups:
        # 遍历组内相邻点
        for i in range(len(group) - 1):
            p1 = group[i]
            p2 = group[i + 1]
            x_diff = abs(p2[0] - p1[0])
            y_diff = abs(p2[1] - p1[1])
            # 你代码里写的是 y差>5，我保持一致
            if x_diff > 5 or y_diff > 5:
                result.append((p1, p2))

    return result


def detect_breaks_via_curvature(curve):
    """
    C++ 函数的精准 Python 翻译
    :param curve: 点列表，每个元素是 (x, y) 元组 / list
    :param breaks: 输出断裂段，每个元素是 (x_start, x_end)，直接在原列表上追加
    """
    breaks = []

    n = len(curve)
    if n < 10:
        return

    WIN = 10  # 方向计算窗口

    # 曲率计算函数（你需要自己实现，接口保持一致）
    curvature = compute_curvature(curve)

    filter_results = filter_adjacent_points(curve)

    # 遍历曲率（对应 C++: i < curvature.size() - WIN）
    for i in range(len(curvature) - WIN):
        miny = 10000
        maxy = -1

        # 窗口内找最小/最大 y
        for ii in range(i, i + WIN):
            y = curve[ii][1]
            if y < miny:
                miny = y
            if y > maxy:
                maxy = y

        # y 方向跨度
        ydis = abs(miny - maxy)

        # 原判断条件：曲率>0.7 且 y 跨度在 5~200 之间
        if curvature[i] > 1.0 and 5 < ydis < 200:
            x_start = curve[i][0]
            x_end = curve[i][0] + WIN
            breaks.append((x_start, x_end))

    return breaks


def group_by_x_center(points, threshold):
    """
    按x坐标中心点距离分组
    :param points: 原始点列表 [(x,y), ...]
    :param threshold: 中心点距离阈值
    :return: 分组结果 [[min_x, max_x], ...]
    """
    # 提取所有x坐标并保持原有顺序
    x_list = [p[0] for p in points]
    if not x_list:
        return []
    
    groups = []
    # 初始化第一组
    current_group = [x_list[0]]
    
    for i in range(1, len(x_list)):
        prev_x = x_list[i-1]
        curr_x = x_list[i]
        # 计算前后两个点各自的中心点
        center_prev = (prev_x + prev_x) / 2  # 单个点中心点就是自身x
        center_curr = (curr_x + curr_x) / 2
        # 中心点距离
        center_dist = abs(center_curr - center_prev)
        
        if center_dist < threshold:
            # 距离小于阈值，加入当前组
            current_group.append(curr_x)
        else:
            # 超出阈值，保存当前组，新建分组
            groups.append([min(current_group), max(current_group)])
            current_group = [curr_x]
    
    # 处理最后一组
    groups.append([min(current_group), max(current_group)])
    return groups


def find_curve_endpoints(img, valid_points):
    # 1. 创建空白画布，绘制有效点（黄色小点）
    vp_img = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(len(valid_points)):
        cv2.circle(vp_img, valid_points[i], 2, (0, 255, 255), -1)

    vp_img_gray = cv2.cvtColor(vp_img, cv2.COLOR_BGR2GRAY)

    # 2. 连通域分析 + 过滤小区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vp_img_gray, connectivity=8)
    areas = stats[:, -1]
    filtered_img = np.zeros_like(vp_img_gray)

    # 存储每个有效连通域的左右端点
    region_endpoints = []  

    for i in range(1, num_labels):
        area = areas[i]
        if area <= 25:
            continue
        
        # 保留当前连通域
        filtered_img[labels == i] = 255
        
        # 从stats提取外接矩形，得到左右端点x坐标
        x, y, w, h = stats[i][0], stats[i][1], stats[i][2], stats[i][3]
        left_x = x                # 最左x
        right_x = x + w           # 最右x
        # 取矩形中心y作为端点y（适配线状点集）
        mid_y = y + h / 2
        
        left_pt = (int(left_x), int(mid_y))
        right_pt = (int(right_x), int(mid_y))
        
        region_endpoints.append({
            "label_id": i,
            "area": area,
            "left_point": left_pt,
            "right_point": right_pt
        })

    # 5. 绘制端点（红色大圆点）
    result_img = vp_img.copy()
    for endp in region_endpoints:
        cv2.circle(result_img, endp["left_point"], 4, (255, 0, 255), -1)  # 红色端点
        cv2.circle(result_img, endp["right_point"], 4, (255, 0, 255), -1)  # 红色端点

    return region_endpoints, result_img, filtered_img


# 新增过滤函数：端点与断裂区间横向距离 < 25 则保留该断裂区间
def filter_breaks_by_endpoints(break_intervals, endpoints, dist_thresh=25):
    filtered_result = []
    for x1, x2 in break_intervals:
        valid = False
        # 遍历所有连通域端点
        for ep in endpoints:
            lx = ep["left_point"][0]
            rx = ep["right_point"][0]
            # 计算端点x坐标 与 断裂区间[x1,x2] 的最短横向距离
            def get_min_dist(pt_x):
                if x1 <= pt_x <= x2:
                    return 0
                elif pt_x < x1:
                    return x1 - pt_x
                else:
                    return pt_x - x2
            # 左端点、右端点任一满足距离阈值即生效
            dist_l = get_min_dist(lx)
            dist_r = get_min_dist(rx)
            if dist_l < dist_thresh or dist_r < dist_thresh:
                valid = True
                break
        if valid:
            filtered_result.append([x1, x2])
    return filtered_result



def get_thresh(img, alg=0, timeit=False):
    t1  = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if alg == 0:
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif alg == 1:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    elif alg == 2:
        _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = cv2.bitwise_or(thresh1, thresh2)
    elif alg == 3:
        _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh3 = cv2.bitwise_or(thresh1, thresh2)

        green_mask = get_green_color_mask(
            img,
            lower1=(35, 43, 46),
            upper1=(77, 255, 255),
            lower2=(78, 43, 46),
            upper2=(95, 255, 255)
        )
        thresh = cv2.bitwise_or(thresh3, green_mask)
    elif alg == 4:
        thresh = get_green_color_mask(
            img,
            lower1=(35, 43, 46),
            upper1=(77, 255, 255),
            lower2=(78, 43, 46),
            upper2=(95, 255, 255)
        )
    else:
        print("Invalid alg index!( 0-4 )")
        thresh = gray

    t2 = time.time()
    if timeit: print(t2 - t1)
    
    return thresh



def detect_laser_by_statistics(img):

    thresh = get_thresh(img, alg=0, timeit=False)
    dstPts = laserDetectByThreshStatistics(thresh)

    for i in range(len(dstPts)):
        cv2.circle(img, dstPts[i], 2, (0, 0, 255), -1)

    # dstPts = np.array(dstPts)

    # model = LaserDetectV1(model_path)
    model = LaserDetectV2(model_path, num_cols=256)
    out = model.detect(img, vis=False)

    fit_model = np.polyfit(out[:, 0], out[:, 1], deg=4)
    fitted_img = fit_plot(img, fit_model, r=4)

    valid_points, outliers = filter_outliers_by_curve(dstPts, fit_model)
    for i in range(len(valid_points)):
        cv2.circle(img, valid_points[i], 2, (0, 255, 255), -1)
    for i in range(len(outliers)):
        cv2.circle(img, outliers[i], 2, (0, 128, 0), -1)

    

    # # 不同的连通域赋予不同的颜色
    # output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # for i in range(1, num_labels):
    #     mask = labels == i
    #     output[:, :, 0][mask] = np.random.randint(0, 256)
    #     output[:, :, 1][mask] = np.random.randint(0, 256)
    #     output[:, :, 2][mask] = np.random.randint(0, 256)

    breaks = detect_breaks_via_curvature(valid_points)
    endpoints, result_img, filtered_img = find_curve_endpoints(img, valid_points)
    result = group_by_x_center(breaks, 50)

    result = filter_breaks_by_endpoints(result, endpoints)

    for i in range(len(result)):
        cv2.rectangle(img, (result[i][0], 0), (result[i][1], img.shape[0]), (255, 20, 128), 2)

    
    cv2.imshow("img", img)
    cv2.imshow("fitted_img", fitted_img)
    cv2.imshow("result_img", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


def detect_breaks_by_model(img):
    pass

def detect_breaks_by_contours(img):
    pass

def detect_breaks_by_features(img):
    pass


# 使用numpy的更高效版本
def group_continuous_array(arr):
    """
    使用numpy函数将连续的索引值分组（更高效）
    """
    if len(arr) == 0:
        return []
    
    # 如果输入是np.where的结果，提取索引数组
    if isinstance(arr, tuple) and len(arr) > 0:
        arr = arr[0]
    
    # 计算不连续点的位置
    diff = np.diff(arr)
    break_points = np.where(diff != 1)[0] + 1
    
    # 使用split分割数组
    groups = np.split(arr, break_points)
    
    return [group.tolist() for group in groups]


def get_group_results_no_y_coords(
        img, groups, min_dis, max_dis, vis_box_expand_pixels, filterEdgeNoLaserResultsFlag=True, filterEdgeNoLaserRatio=0.25
    ):
    imgsz = img.shape[:2]
    left_edge = int(imgsz[1] * filterEdgeNoLaserRatio)
    right_edge = int(imgsz[1] * (1 - filterEdgeNoLaserRatio))
    boxes = []
    wides = []

    for i, gi in enumerate(groups):
        start = gi[0]
        end = gi[-1]

        if filterEdgeNoLaserResultsFlag:
            if end < left_edge or start > right_edge:
                continue
        
        # assert end >= start, "end < start!"
        # print(f"组 {i+1}: 起始={start}, 结束={end}, 值={column_means[start:end+1]}")

        fakeDis = end - start + 1

        if fakeDis >= imgsz[1]: continue
        if fakeDis < min_dis: continue
        if fakeDis > max_dis: continue

        # 去除两端激光线有一点点断裂的误报
        if fakeDis < min_dis * 5 and (start < 15 or end > imgsz[1] - 15): continue

        x1_vis = start - vis_box_expand_pixels
        x2_vis = end + vis_box_expand_pixels
        # y1_vis = round(np.mean(min_y_coords[start:end])) - vis_box_expand_pixels * 2
        # y2_vis = round(np.mean(max_y_coords[start:end])) + vis_box_expand_pixels * 2

        y1_vis = 0
        y2_vis = imgsz[0]

        if x1_vis < 0: x1_vis = 0
        if y1_vis < 0: y1_vis = 0
        if x2_vis > imgsz[1]: x2_vis = imgsz[1]
        if y2_vis > imgsz[0]: y2_vis = imgsz[0]

        visBox = [x1_vis, y1_vis, x2_vis - x1_vis, y2_vis - y1_vis]
        # boxes.append([start, 0, fakeDis, imgsz[0]])

        # 去除光线影响

        boxes.append(visBox)
        wides.append(fakeDis)

    return boxes, wides


def detectNoLaser(
        img, vis_box_expand_pixels=25, zscore_thr1=-5, zscore_thr2=5, min_dis=10, max_dis=1000,
        col_sum_r1=0.5, zScoreJudgmentMethod=3, detectNoLaserAreaMethod=0, filterEdgeNoLaserResultsFlag=True,
        filterEdgeNoLaserRatio=0.25, lower_green1=None, upper_green1=None, lower_green2=None, upper_green2=None
    ):
    imgsz = img.shape[:2]
    newsz = (64, 128)
    hwr = (newsz[0] / imgsz[0], newsz[1] / imgsz[1])

    # resize主要是考虑速度问题
    # img_new = cv2.resize(img, newsz[::-1])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # # #### 绿色 HSV V最小值 范围在[0, 46]区间内
    # # #### 将原本的46范围改小有助于检测亮度值比较低的绿色成分
    # # #### 默认绿色范围
    # # #### p_hsvLower1: "35, 43, 46"
    # p_hsvLower1: "35, 43, 3"
    # p_hsvUpper1: "77, 255, 255"
    # # #### 第二个绿色范围（针对亮绿色）
    # # #### p_hsvLower2: "78, 43, 46"
    # p_hsvLower2: "78, 43, 3"
    # p_hsvUpper2: "95, 255, 255"

    # 定义绿色在HSV中的范围
    # 注意：HSV中H通道范围是0-179（OpenCV中）
    if lower_green1 is None or upper_green1 is None:
        lower_green1 = np.array([35, 43, 46])   # 绿色范围下限
        upper_green1 = np.array([77, 255, 255]) # 绿色范围上限
    
    # 创建绿色掩膜
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    
    # 可选：添加第二个绿色范围（针对亮绿色）
    if lower_green2 is None or upper_green2 is None:
        lower_green2 = np.array([78, 43, 46])   # 另一种绿色范围
        upper_green2 = np.array([95, 255, 255])
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    
    # 合并两个掩膜
    mask = cv2.bitwise_or(mask1, mask2)

    column_sums = np.sum(mask, axis=0)
    column_sums_median = np.median(column_sums)
    column_sums_mean = np.mean(column_sums)
    column_sums_std = np.std(column_sums)

    # zscore = z_score(column_sums, column_sums_mean, column_sums_std)

    # x_coords2 = np.where((zscore < zscore_thr1) | (zscore > zscore_thr2) | (column_sums < (min(column_sums_median, column_sums_mean) * col_sum_r1)) | (column_sums > (max(column_sums_median, column_sums_mean) * col_sum_r2)))
    # x_coords2 = np.where(column_sums <= 255 * col_sum_r1)
    if detectNoLaserAreaMethod == 0:
        x_coords2 = np.where(column_sums <= 0)
    elif detectNoLaserAreaMethod == 1:
        x_coords2 = np.where(column_sums < (min(column_sums_median, column_sums_mean) * col_sum_r1))
    
    groups2 = group_continuous_array(x_coords2[0])
    boxes, wides = get_group_results_no_y_coords(img, groups2, min_dis, imgsz[1], vis_box_expand_pixels, filterEdgeNoLaserResultsFlag, filterEdgeNoLaserRatio)

    return boxes, wides



def filter_outliers_by_curve(points, coeffs, max_distance=25):
    """
    根据4次拟合曲线剔除距离过远的异常点
    
    参数:
        points: 列表/数组，格式为 [(x1,y1), (x2,y2), ...]
        coeffs: 4次曲线系数，格式 [a4, a3, a2, a1, a0]
                对应方程 y = a4*x⁴ + a3*x³ + a2*x² + a1*x + a0
        max_distance: 最大允许距离，默认25
    
    返回:
        valid_points: 距离 ≤ max_distance 的有效点
        outliers: 被剔除的异常点
    """
    # 转换为numpy数组方便计算
    points = np.array(points)
    x = points[:, 0]
    y_actual = points[:, 1]
    
    # 计算曲线上的理论y值
    y_curve = np.polyval(coeffs, x)
    
    # 计算点到曲线的垂直距离（对于函数曲线，垂直距离最常用）
    distances = np.abs(y_actual - y_curve)
    
    # 筛选有效点 & 异常点
    valid_mask = distances <= max_distance
    valid_points = points[valid_mask].tolist()
    outliers = points[~valid_mask].tolist()
    
    return valid_points, outliers



if __name__ == "__main__":
    # img_path = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\kpt\v12\train\images\rename_20250807_0012682.jpg"
    # img_path = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\laser_detect\v1\images\Video_2025_03_08_111948_1_output_000004160.jpg"
    img_path = r"G:\Gosion\projects\006.Belt_Torn_Det\data\Train\kpt\v12\train\images\Video_2025_08_27_173442_1_output_000001197.jpg"
    img = cv2.imread(img_path)

    # model_path = r"G:\Gosion\projects\006.Belt_Torn_Det\code\find_curve\weights\best.pth"
    model_path = r"G:\Gosion\projects\006.Belt_Torn_Det\code\find_curve\weights\best.onnx"
    # test_and_visualize(model_path, img_path, num_cols=256, imgsz=(96, 256), show=True)
    # test_and_visualize(model_path, img_path, num_cols=256, imgsz=(224, 224), show=True)

    
    # out = detect_laser_by_color(img)
    # cv2.imshow("out", out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    detect_laser_by_statistics(img)
