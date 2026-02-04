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
# from src.utils import cv2pil, pil2cv


class YOLOv5_ORT(object):
    """
    yolov5-6.2 yolov5-7.0
    onnx-opset 10
    """
    def __init__(self, model_path, conf_threshold=0.6, iou_threshold=0.45, class_path=None, cuda=False):
        self.cuda = cuda and torch.cuda.is_available()
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.cuda else ['CPUExecutionProvider']
        self.model_path = model_path
        self.class_path = class_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.input_names = self.session.get_inputs()[0].name
        self.output_names = self.session.get_outputs()[0].name
        self.input_size = [int(self.session.get_inputs()[0].shape[2]), int(self.session.get_inputs()[0].shape[3])]

    def load_classes(self):
        class_list = []
        with open(self.class_path, "r", encoding="utf-8") as f:
            class_list = [cname.strip() for cname in f.readlines()]
        return class_list

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
    
    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_area(self, box):
        # box = xyxy(4,n)
        return (box[2] - box[0]) * (box[3] - box[1])
    
    def box_iou(self, box1, box2, eps=1e-7):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / (self.box_area(box1.T)[:, None] + self.box_area(box2.T) - inter + eps)
    
    def non_max_suppression(self, prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.3 + 0.03 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        # output = [torch.zeros((0, 6), device=prediction.device)] * bs
        output = [torch.zeros((0, 6))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                # v = torch.zeros((len(lb), nc + 5), device=x.device)
                v = torch.zeros((len(lb), nc + 5))
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                # conf, j = x[:, 5:].max(1, keepdim=True)
                conf, j = torch.tensor(x[:, 5:]).float().max(1, keepdim=True)
                # x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                x = torch.cat((torch.tensor(box), conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                # x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
                x = x[(x[:, 5:6] == torch.tensor(classes)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                # x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True

            output[xi] = x[i]
            # if (time.time() - t) > time_limit:
            #     LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            #     break  # time limit exceeded

        return output
    
    def clip_coords(self, boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords
    
    def pre_process(self, img, stride=32):
        img1 = img.copy()
        self.srcsz = img1.shape[:2]
        img1 = self.letterbox(img1, self.input_size, stride=stride, auto=False)[0]
        img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img1 = np.ascontiguousarray(img1)
        img1 = img1.astype(dtype=np.float32)
        img1 /= 255.0
        img1 = np.expand_dims(img1, axis=0)
        return img1
    
    def inference(self, img):
        pred = self.session.run([self.output_names], {self.input_names: img})[0]
        return pred

    def post_process(self, pred):
        output = self.non_max_suppression(pred, self.conf_threshold, self.iou_threshold, agnostic=False)
        out = []
        for i, det in enumerate(output):  # detections per image
            if len(det):
                det[:, :4] = self.scale_coords(self.input_size, det[:, :4], self.srcsz).round()
                for *xyxy, conf, cls in reversed(det):
                    outi = [int(round(ci)) for ci in torch.tensor(xyxy).view(1, 4).view(-1).tolist()]
                    outi.append(float(conf.numpy()))
                    outi.append(int(cls.numpy()))
                    out.append(outi)

        return out
    
    def detect(self, img, show=False):
        img1 = self.pre_process(img)
        pred = self.inference(img1)
        out = self.post_process(pred)

        if show:
            for b in out:
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
                cv2.putText(img, "{}: {:.2f}".format(b[5], b[4]), (b[0], b[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            cv2.imshow("detect", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return out



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
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        # Perform post-processing on the outputs to obtain output image
        output = self.postprocess(outputs, pad)

        if show:
            for b in output:
                self.draw_detections(img, b[:4], b[4], b[5], color)
            cv2.imshow("detect", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return output


def yolov5_inference_example():
    img_path = r"G:\Gosion\projects\000.Test_Data\images\zidane.jpg"
    model_path = r"G:\Gosion\code\Python\yolov5-6.2\yolov5s.onnx"

    yolo = YOLOv5_ORT(model_path, 0.60, 0.45, None, cuda=False)
    img = cv2.imread(img_path)
    res = yolo.detect(img, show=True)


def yolo11_inference_example():
    # model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\yolo11s.onnx"
    # img_path = r"G:\Gosion\projects\000.Test_Data\images\zidane.jpg"

    model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\detect\train3\weights\best.onnx"
    img_path = r"G:\Gosion\projects\009.TuoGun_Det\videos\frames\20251221_merged\20251221_101111_output_000000001.jpg"

    img = cv2.imread(img_path)

    yolo11 = YOLO11(model_path, 0.60, 0.45)
    output = yolo11.detect(img, show=True, color=None)
    print(output)


def yolo11_inference_example_dir():
    model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\detect\train4\weights\best.onnx"

    data_path = r"G:\Gosion\projects\009.TuoGun_Det\images\XianChang\data_v1.0.13_20260202\data\000"
    dir_list = os.listdir(data_path)

    save_path = r"G:\Gosion\projects\009.TuoGun_Det\images\XianChang\output"
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




if __name__ == '__main__':
    # yolov5_inference_example()
    # yolo11_inference_example()
    yolo11_inference_example_dir()


