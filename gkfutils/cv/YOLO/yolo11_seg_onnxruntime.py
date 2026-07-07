

# from ultralytics import YOLO


# # model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\yolo11s-seg.onnx"
# model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\segment\train\weights\best.pt"
# img_path = r"G:\Gosion\projects\012.JurongDianchang\data\belt_deviate\v2\val\images\17769370682862047245390297952257.jpg"

# # Load a pretrained YOLO model (recommended for training)
# model = YOLO(model_path)

# results = model.predict(source=img_path,
#                         imgsz=640,
#                         project='segment/predict',
#                         name='exp',
#                         save=True,
#                         conf=0.4,
#                         iou=0.7,
#                         device='cpu')



# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


from __future__ import annotations

import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, YAML, nms, ops
from ultralytics.utils.checks import check_yaml


class YOLOv8Seg:
    """YOLOv8 segmentation model for performing instance segmentation using ONNX Runtime.

    This class implements a YOLOv8 instance segmentation model using ONNX Runtime for inference. It handles
    preprocessing of input images, running inference with the ONNX model, and postprocessing the results to generate
    bounding boxes and segmentation masks.

    Attributes:
        session (ort.InferenceSession): ONNX Runtime inference session for model execution.
        imgsz (tuple[int, int]): Input image size as (height, width) for the model.
        classes (dict): Dictionary mapping class indices to class names from the dataset.
        conf (float): Confidence threshold for filtering detections.
        iou (float): IoU threshold used by non-maximum suppression.

    Methods:
        letterbox: Resize and pad image while maintaining aspect ratio.
        preprocess: Preprocess the input image before feeding it into the model.
        postprocess: Post-process model predictions to extract meaningful results.
        process_mask: Process prototype masks with predicted mask coefficients to generate instance segmentation masks.

    Examples:
        >>> model = YOLOv8Seg("yolov8n-seg.onnx", conf=0.25, iou=0.7)
        >>> img = cv2.imread("image.jpg")
        >>> results = model(img)
        >>> cv2.imshow("Segmentation", results[0].plot())
    """

    def __init__(self, onnx_model: str, conf: float = 0.25, iou: float = 0.7, imgsz: int | tuple[int, int] = 640):
        """Initialize the instance segmentation model using an ONNX model.

        Args:
            onnx_model (str): Path to the ONNX model file.
            conf (float, optional): Confidence threshold for filtering detections.
            iou (float, optional): IoU threshold for non-maximum suppression.
            imgsz (int | tuple[int, int], optional): Input image size of the model. Can be an integer for square input
                or a tuple for rectangular input.
        """
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(onnx_model, providers=providers or available)

        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        self.classes = 1
        self.input_w, self.input_h = self.input_shape[3], self.input_shape[2]

        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        # self.classes = YAML.load(check_yaml(r"G:\Gosion\code\Python\ultralytics-8.3.243\ultralytics\cfg\datasets\coco128-seg.yaml"))["names"]
        self.classes = {"0": "belt"} 
        self.conf = conf
        self.iou = iou

    def __call__(self, img: np.ndarray) -> list[Results]:
        """Run inference on the input image using the ONNX model.

        Args:
            img (np.ndarray): The original input image in BGR format.

        Returns:
            (list[Results]): Processed detection results after post-processing, containing bounding boxes and
                segmentation masks.
        """
        prep_img = self.preprocess(img, self.imgsz)
        outs = self.session.run(None, {self.session.get_inputs()[0].name: prep_img})
        return self.postprocess(img, prep_img, outs)

    def letterbox(self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> np.ndarray:
        """Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image in BGR format.
            new_shape (tuple[int, int], optional): Target shape as (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
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

        return img

    def preprocess(self, img: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
        """Preprocess the input image before feeding it into the model.

        Args:
            img (np.ndarray): The input image in BGR format.
            new_shape (tuple[int, int]): The target shape for resizing as (height, width).

        Returns:
            (np.ndarray): Preprocessed image ready for model inference, with shape (1, 3, height, width) and normalized
                to [0, 1].
        """
        img = self.letterbox(img, new_shape)
        img = img[..., ::-1].transpose([2, 0, 1])[None]  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255  # Normalize to [0, 1]
        return img
    
    def preprocess2(self, image: np.ndarray):
        """YOLOv5预处理（与训练一致） - 与你的原始代码完全相同"""
        origin_h, origin_w = image.shape[:2]

        # 缩放+填充（保持长宽比）
        ratio = min(self.input_w / origin_w, self.input_h / origin_h)
        new_w, new_h = int(origin_w * ratio), int(origin_h * ratio)
        img_resized = cv2.resize(image, (new_w, new_h))

        # 计算填充
        pad_w = (self.input_w - new_w) // 2
        pad_h = (self.input_h - new_h) // 2
        img_padded = cv2.copyMakeBorder(
            img_resized, pad_h, self.input_h - new_h - pad_h,
            pad_w, self.input_w - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=[114, 114, 114]
        )

        # BGR转RGB + 归一化（YOLOv5默认方式）
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb / 255.0  # 缩放至0-1

        # 转换为NCHW格式（1,3,640,640）
        img_nchw = np.transpose(img_normalized, (2, 0, 1)).astype(np.float32)
        input_data = np.expand_dims(img_nchw, axis=0)

        # 保存原始信息（用于后处理）
        origin_info = (origin_h, origin_w, pad_h, pad_w, ratio)
        return input_data, origin_info

    def postprocess(self, img: np.ndarray, prep_img: np.ndarray, outs: list) -> list[Results]:
        """Post-process model predictions to extract meaningful results.

        Args:
            img (np.ndarray): The original input image.
            prep_img (np.ndarray): The preprocessed image used for inference.
            outs (list): Model outputs containing predictions and prototype masks.

        Returns:
            (list[Results]): Processed detection results containing bounding boxes and segmentation masks.
        """
        preds, protos = (torch.from_numpy(p) for p in outs)
        preds = nms.non_max_suppression(preds, self.conf, self.iou, nc=len(self.classes))

        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = ops.scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            masks = self.process_mask(protos[i], pred[:, 6:], pred[:, :4], img.shape[:2])
            results.append(Results(img, path="", names=self.classes, boxes=pred[:, :6], masks=masks))

        return results

    def process_mask(
        self, protos: torch.Tensor, masks_in: torch.Tensor, bboxes: torch.Tensor, shape: tuple[int, int]
    ) -> torch.Tensor:
        """Process prototype masks with predicted mask coefficients to generate instance segmentation masks.

        Args:
            protos (torch.Tensor): Prototype masks with shape (mask_dim, mask_h, mask_w).
            masks_in (torch.Tensor): Predicted mask coefficients with shape (N, mask_dim), where N is number of
                detections.
            bboxes (torch.Tensor): Bounding boxes with shape (N, 4), where N is number of detections.
            shape (tuple[int, int]): The size of the input image as (height, width).

        Returns:
            (torch.Tensor): Binary segmentation masks with shape (N, height, width).
        """
        c, mh, mw = protos.shape  # CHW
        masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # Matrix multiplication
        masks = ops.scale_masks(masks[None], shape)[0]  # Scale masks to original image size
        masks = ops.crop_mask(masks, bboxes)  # Crop masks to bounding boxes
        return masks.gt_(0.0)  # Convert to binary masks
    
    # YOLOv8/9/11通用后处理，包括：阈值过滤与NMS+masks处理 - 与你的原始代码完全相同
    def postprocess_v8(self, preds, im0, ratio, pad_w, pad_h, conf_threshold=0.3, iou_threshold=0.45, nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # 与bbox区别：Two outputs: 检测头的输出(1, 38, 8400), 分割头的输出(1, 32, 160, 160)

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum('bcn->bnc', x)  # (1, 8400, 38)

        # Predictions filtering by conf-threshold，不包括后32维的向量（32维的向量可以看作是与每个检测框关联的分割 mask 的系数或权重）
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        # 经过NMS后的值, np.array([[x, y, w, h, conf, cls, nm], ...]), shape=(-1, 4 + 1 + 1 + 32)
        if len(x) > 0:
            nms_indices = cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)
            if len(nms_indices) > 0:
                x = x[nms_indices.flatten()]
            else:
                x = np.array([])

        # 重新缩放边界框，为画图做准备
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= ratio

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # 与bbox区别：增加masks处理
            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)

            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []

    def masks2segments(self, masks):
        """
        It takes a list of masks(n,h,w) and returns a list of segments(n,xy)
        """
        segments = []
        for x in masks.astype('uint8'):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype('float32'))
        return segments

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum('HWN -> NHW', masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    def crop_mask(self, masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def scale_mask(self, masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')

        # 确保裁剪区域有效
        top = max(0, top)
        left = max(0, left)
        bottom = min(masks.shape[0], bottom)
        right = min(masks.shape[1], right)

        if bottom > top and right > left:
            masks = masks[top:bottom, left:right]
        else:
            # 如果裁剪区域无效，使用全尺寸
            pass

        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks

    def infer_masks(self, image: np.ndarray):
        """
        完整推理流程
        Args:
            image: BGR图像 [H, W, 3]
        Returns:
            belt_mask, coal_mask: 二值mask [H, W]
        """
        # 预处理
        input_data, origin_info = self.preprocess(image)

        # 推理
        output = self.inference(input_data)

        # 后处理
        origin_h, origin_w, pad_h, pad_w, ratio = origin_info
        boxes, segments, masks = self.postprocess_v8(output, image, ratio, pad_w, pad_h, 0.3, 0.45, nm=32)

        # 分离皮带和煤的mask
        belt_mask = np.zeros((origin_h, origin_w), dtype=np.uint8)
        coal_mask = np.zeros((origin_h, origin_w), dtype=np.uint8)

        if len(boxes) > 0:
            for i, (*box, conf, cls_id) in enumerate(boxes):
                if int(cls_id) == 0:  # 皮带
                    if i < len(masks):
                        belt_mask = np.maximum(belt_mask, masks[i].astype(np.uint8))
                elif int(cls_id) == 1:  # 煤
                    if i < len(masks):
                        coal_mask = np.maximum(coal_mask, masks[i].astype(np.uint8))

        return belt_mask, coal_mask
    
    def infer_masks2(self, image: np.ndarray):
        """
        完整推理流程
        Args:
            image: BGR图像 [H, W, 3]
        Returns:
            belt_mask, coal_mask: 二值mask [H, W]
        """
        # 预处理
        input_data, origin_info = self.preprocess2(image)

        # 推理
        # output = self.inference(input_data)
        output = self.session.run(None, {self.session.get_inputs()[0].name: input_data})

        # 后处理
        origin_h, origin_w, pad_h, pad_w, ratio = origin_info
        boxes, segments, masks = self.postprocess_v8(output, image, ratio, pad_w, pad_h, 0.3, 0.45, nm=32)

        # 分离皮带和煤的mask
        belt_mask = np.zeros((origin_h, origin_w), dtype=np.uint8)
        coal_mask = np.zeros((origin_h, origin_w), dtype=np.uint8)

        if len(boxes) > 0:
            for i, (*box, conf, cls_id) in enumerate(boxes):
                if int(cls_id) == 0:  # 皮带
                    if i < len(masks):
                        belt_mask = np.maximum(belt_mask, masks[i].astype(np.uint8))
                elif int(cls_id) == 1:  # 煤
                    if i < len(masks):
                        coal_mask = np.maximum(coal_mask, masks[i].astype(np.uint8))

        return belt_mask, coal_mask


if __name__ == "__main__":
    # model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\yolo11s-seg.onnx"
    # img_path = r"G:\Gosion\projects\000.Test_Data\images\zidane.jpg"

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default=model_path, help="Path to ONNX model")
    # parser.add_argument("--source", type=str, default=img_path, help="Path to input image")
    # parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    # parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    # args = parser.parse_args()

    # model = YOLOv8Seg(args.model, args.conf, args.iou)
    # img = cv2.imread(args.source)
    # # results = model(img)
    # belt_mask, coal_mask = model.infer_masks2(img)

    # # cv2.imshow("Segmented Image", results[0].plot())
    # cv2.imshow("belt_mask", belt_mask*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\segment\train2\weights\best.onnx"
    data_path = r"G:\Gosion\projects\012.JurongDianchang\data\paopian_orig"
    save_path = r"G:\Gosion\projects\012.JurongDianchang\data\paopian_orig_output2"
    os.makedirs(save_path, exist_ok=True)
    file_list = sorted(os.listdir(data_path))


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default=model_path, help="Path to ONNX model")
    # # parser.add_argument("--source", type=str, default=img_path, help="Path to input image")
    # parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    # parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    # args = parser.parse_args()

    model = YOLOv8Seg(model_path, 0.6, 0.7)

    for f in file_list:
        img_path = os.path.join(data_path, f)
        img = cv2.imread(img_path)
        # results = model(img)
        belt_mask, coal_mask = model.infer_masks2(img)

        # cv2.imshow("Segmented Image", results[0].plot())
        # cv2.imshow("belt_mask", belt_mask*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        output_path = os.path.join(save_path, f)
        cv2.imwrite(output_path, belt_mask*255)
