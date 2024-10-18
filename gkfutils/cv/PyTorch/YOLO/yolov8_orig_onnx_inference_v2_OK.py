import os
import cv2
import onnxruntime
import torch
import torchvision
import numpy as np
import time


class YOLOv8_ONNX(object):
    def __init__(self, onnx_path):
        cuda = torch.cuda.is_available()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.input_names = self.session.get_inputs()[0].name
        self.output_names = self.session.get_outputs()[0].name

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
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
        Returns:
            y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
    
    def clip_boxes(self, boxes, shape):
        """
        It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
        shape

        Args:
          boxes (torch.Tensor): the bounding boxes to clip
          shape (tuple): the shape of the image
        """
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        """
        Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
        (img1_shape) to the shape of a different image (img0_shape).

        Args:
          img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
          boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
          img0_shape (tuple): the shape of the target image, in the format of (height, width).
          ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                             calculated based on the size difference between the two images.

        Returns:
          boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes
    
    def box_iou(self, box1, box2, eps=1e-7):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
            eps

        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
        """
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300,
                            nc=0,  # number of classes (optional)
                            max_time_img=0.05,
                            max_nms=30000,
                            max_wh=7680,
        ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Arguments:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        prediction = torch.Tensor(prediction)

        device = prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + max_time_img * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x.transpose(0, -1)[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)
            box = self.xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            if multi_label:
                i, j = (cls > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            # if mps:
            #     output[xi] = output[xi].to(device)
            # if (time.time() - t) > time_limit:
            #     LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            #     break  # time limit exceeded

        return output
    
    def pre_process(self, img_path, img_size=(640, 640), stride=32):
        # img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        # img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        # img /= 255  # 0 - 255 to 0.0 - 1.0
        # return img

        img0 = cv2.imread(img_path)
        src_size = img0.shape[:2]
        img = self.letterbox(img0, img_size, stride=stride, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img0, img, src_size
    
    def inference(self, img):
        # im = im.cpu().numpy()  # torch to numpy
        pred = self.session.run([self.output_names], {self.input_names: img})[0]
        return pred

    def post_process(self, preds, src_size, img_size, conf_thres=0.25, iou_thres=0.45):
        preds = self.non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        agnostic=False,
                                        max_det=300,
                                        classes=None)
        # results = []
        # for i, pred in enumerate(preds):
        #     # orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        #     if not isinstance(orig_imgs, torch.Tensor):
        #         pred[:, :4] = self.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        #     path, _, _, _, _ = self.batch
        #     img_path = path[i] if isinstance(path, list) else path
        #     results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        # return results
        out_bbx = []
        for i, det in enumerate(preds):  # detections per image
            if len(det):
                det[:, :4] = self.scale_boxes(img_size, det[:, :4], src_size).round()
                for *xyxy, conf, cls in reversed(det):
                    x1y1x2y2_VOC = [int(round(ci)) for ci in torch.tensor(xyxy).view(1, 4).view(-1).tolist()]
                    x1y1x2y2_VOC.append(float(conf.numpy()))
                    x1y1x2y2_VOC.append(int(cls.numpy()))
                    out_bbx.append(x1y1x2y2_VOC)

        return out_bbx


if __name__ == '__main__':
    # onnx_path = r"E:\GraceKafuu\Python\yolov5-6.2\yolov5s.onnx"
    onnx_path = r"E:\GraceKafuu\Python\ultralytics-main\yolov8n.onnx"
    img_path = r"E:\GraceKafuu\Python\yolov5-6.2\data\images\bus.jpg"

    model = YOLOv8_ONNX(onnx_path)
    model_input_size = (640, 640)
    img0, img, src_size = model.pre_process(img_path, img_size=model_input_size)
    print("src_size: ", src_size)

    t1 = time.time()
    pred = model.inference(img)
    t2 = time.time()
    print("{:.12f}".format(t2 - t1))

    # tt = []
    # for i in range(100):
    #     t1 = time.time()
    #     pred = model.inference(img)
    #     t2 = time.time()
    #     print(t2 - t2)
    #     tt.append(t2 - t1)
    #
    # print(np.mean(tt))

    out_bbx = model.post_process(pred, src_size, img_size=model_input_size)
    print("out_bbx: ", out_bbx)
    for b in out_bbx:
        cv2.rectangle(img0, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
    cv2.imshow("test", img0)
    cv2.waitKey(0)