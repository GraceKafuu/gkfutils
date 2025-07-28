import time
import cv2
import logging
import numpy as np
import onnxruntime as ort


class YOLOv5_ORT(object):
    def __init__(self, model_path, imgsz=(640, 640), conf=0.60, iou=0.45, device='cpu'):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device != 'cpu' else ['CPUExecutionProvider']

        # Create an inference session using the ONNX model and specify execution providers
        self.session = ort.InferenceSession(self.model_path, providers=providers)

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

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

    def clip_coords(self, boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        # np.array (faster grouped)
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

    def nms(self, bboxes, scores, iou_thresh):
        """

        :param bboxes: 检测框列表
        :param scores: 置信度列表
        :param iou_thresh: IOU阈值
        :return:
        """
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (y2 - y1) * (x2 - x1)

        # 结果列表
        result = []
        index = scores.argsort()[::-1]  # 对检测框按照置信度进行从高到低的排序，并获取索引
        # 下面的操作为了安全，都是对索引处理
        while index.size > 0:
            # 当检测框不为空一直循环
            i = index[0]
            result.append(i)  # 将置信度最高的加入结果列表

            # 计算其他边界框与该边界框的IOU
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])
            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)
            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            # 只保留满足IOU阈值的索引
            idx = np.where(ious <= iou_thresh)[0]
            index = index[idx + 1]  # 处理剩余的边框
        # bboxes, scores = bboxes[result], scores[result]
        # return bboxes, scores
        return result

    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

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

        t = time.time()
        output = [np.zeros((0,6))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = np.zeros((len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = np.concatenate((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])
            # Detections matrix nx6 (xyxy, conf, cls)
            conf, j = x[:, 5:].max(1, keepdims=True), x[:, 5:].argmax(1)[:,None]
            x = np.concatenate((box, conf, j), 1)[conf.reshape(-1) > conf_thres]
            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = self.nms(boxes, scores, iou_thres)  # NMS
            if len(i) > max_det:  # limit detections
                i = i[:max_det]
            '''
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            '''

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                logging.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output
    
    def preprocess(self, img0):
        self.orig_shape = img0.shape
        # Set Dataprocess & Run inference
        img = self.letterbox(img0, new_shape=self.imgsz, auto=False)[0]
        # print("===", img.shape)
        # cv2.imwrite('result/0414_auto.jpg', img)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = img.astype(dtype=np.float32)
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        self.img_shape = img.shape[2:]
        return img
    
    def inference(self, img):
        pred = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: img})[0]
        return pred
    
    def postprocess(self, pred):
        pred = self.non_max_suppression(pred, self.conf, self.iou, None, False, max_det=1000)
        det = pred[0] # detections single image
        # Process detections

        results = []

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = self.scale_coords(self.img_shape, det[:, :4], self.orig_shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = str(cls)
                prob = round(float(conf), 2)  # round 2
                # c_x = (int(xyxy[0]) + int(xyxy[2])) / 2
                # c_y = (int(xyxy[1]) + int(xyxy[3])) / 2
                # Img vis
                xmin, ymin, xmax, ymax = xyxy
                newpoints = [(int(xmin), int(ymin)), (int(xmax), int(ymax))]

                result = [newpoints, label, prob]
                results.append(result)
                
        return results
    
    def vis_results(self, img, results):
        for pts, label, prob in results:
            newpoints = np.array(pts)    
            cv2.rectangle(img, newpoints[0], newpoints[1], (0,255,0), 2) 
            cv2.putText(img, label + '_' + str(prob), newpoints[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)

        return img
    
    def detect(self, img0):

        """
        对输入的图片进行目标检测，返回对应的类别的检测框,并可视化输出
        #输出后自己做对应的逻辑处理
        """

        img = self.preprocess(img0)
        # Inference
        pred = self.inference(img)   
        results = self.postprocess(pred)
        self.vis_results(img0, results)
        
        return img0

    

 
 
if __name__=="__main__":
    onnx_path = r"D:\Gosion\code\others\Python\yolov5-master\yolov5s.onnx"
    img_pth = r"G:\Gosion\data\004.Out_GuardArea_Det\resources\20250725\plan_light_1933425297818984450.jpg"
    
    yolov5 = YOLOv5_ORT(onnx_path, conf=0.25)
    img = cv2.imread(img_pth)

    out = yolov5.detect(img)

    cv2.imshow("out", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()