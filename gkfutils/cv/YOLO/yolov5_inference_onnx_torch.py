import os
import cv2
import onnxruntime
import torch
import torchvision
import numpy as np
import time
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


class YOLOv5_ONNX(object):
    def __init__(self, onnx_path, num_classes=80, conf_thres=0.60, iou_thres=0.45):
        cuda = torch.cuda.is_available()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        # providers = ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.input_names = self.session.get_inputs()[0].name
        self.output_names = self.session.get_outputs()[0].name
        self.input_size = self.session.get_inputs()[0].shape
        self.output_size = self.session.get_outputs()[0].shape

        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

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

        t = time.time()
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
    
    def pre_process(self, img_path, img_size=(640, 640), stride=32):
        img0 = cv2.imread(img_path)
        self.img0 = img0.copy()
        src_size = img0.shape[:2]
        img = self.letterbox(img0, img_size, stride=stride, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img0, img, src_size
    
    @timeit
    def inference(self, img):
        pred = self.session.run([self.output_names], {self.input_names: img})[0]
        return pred

    def post_process(self, pred, src_size, img_size):
        output = self.non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=False)
        out_bbx = []
        for i, det in enumerate(output):  # detections per image
            if len(det):
                det[:, :4] = self.scale_coords(img_size, det[:, :4], src_size).round()
                for *xyxy, conf, cls in reversed(det):
                    x1y1x2y2_VOC = [int(round(ci)) for ci in torch.tensor(xyxy).view(1, 4).view(-1).tolist()]
                    x1y1x2y2_VOC.append(float(conf.numpy()))
                    x1y1x2y2_VOC.append(int(cls.numpy()))
                    out_bbx.append(x1y1x2y2_VOC)

        return out_bbx
    
    def random_color(self):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)

        return (b, g, r)

    
    def vis_result(self, out_bbx):
        colors = []
        for i in range(self.num_classes):
            colors.append(self.random_color())

        for b in out_bbx:
            cls = int(b[5])
            cv2.rectangle(self.img0, (b[0], b[1]), (b[2], b[3]), colors[cls], 2)
            cv2.putText(self.img0, "{}: {:.2f}".format(cls, b[4]), (b[0], b[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, colors[cls], 2)

        cv2.imshow("result", self.img0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

def bbox_voc_to_yolo(imgsz, box):
    """
    VOC --> YOLO
    :param imgsz: [H, W]
    :param box:
    orig: [xmin, xmax, ymin, ymax], deprecated;
    new:  [xmin, ymin, xmax, ymax], 2024.03.29, WJH.
    :return: [x, y, w, h]
    """
    dh = 1. / (imgsz[0])
    dw = 1. / (imgsz[1])
    # x = (box[0] + box[1]) / 2.0
    # y = (box[2] + box[3]) / 2.0
    # w = box[1] - box[0]
    # h = box[3] - box[2]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = int(round(x)) * dw
    w = int(round(w)) * dw
    y = int(round(y)) * dh
    h = int(round(h)) * dh

    if x < 0: x = 0
    if y < 0: y = 0
    if w > 1: w = 1
    if h > 1: h = 1
    assert x <= 1, "x: {}".format(x)
    assert y <= 1, "y: {}".format(y)
    assert w >= 0, "w: {}".format(w)
    assert h >= 0, "h: {}".format(h)

    return [x, y, w, h]


def yolo_inference_save_labels(data_path, model_path, model_label=0, dst_label=0, cls_name="person"):
    base_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "../labels_{}".format(cls_name)))
    os.makedirs(save_path, exist_ok=True)
    file_list = os.listdir(data_path)
    
    model = YOLOv5_ONNX(model_path, conf_thres=0.60, iou_thres=0.45)
    model_input_size = (640, 640)

    for f in file_list:
        fname = os.path.splitext(f)[0]
        f_abs_path = data_path + "/{}".format(f)
        # img = cv2.imread(f_abs_path)
        # imgsz = img.shape[:2]

        txt_save_path = save_path + "/{}.txt".format(fname)

        try:
            img0, img, src_size = model.pre_process(f_abs_path, img_size=model_input_size)
            pred = model.inference(img)
            out_bbx = model.post_process(pred, src_size, img_size=model_input_size)

            with open(txt_save_path, "w", encoding="utf-8") as fw:
                for b in out_bbx:
                    x1, y1, x2, y2, conf, cls = b
                    bbox_yolo = bbox_voc_to_yolo(src_size, [x1, y1, x2, y2])

                    # txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bbox_yolo]) + "\n"
                    # fw.write(txt_content)

                    if int(cls) == model_label:
                        txt_content = "{}".format(dst_label) + " " + " ".join([str(b) for b in bbox_yolo]) + "\n"
                        fw.write(txt_content)

                    # txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bbox_yolo]) + "\n"
                    # fw.write(txt_content)

                    # if int(cls) == 1 or int(cls) == 2:
                    #     txt_content = "{}".format(cls) + " " + " ".join([str(b) for b in bbox_yolo]) + "\n"
                    #     fw.write(txt_content)

        except Exception as e:
            print(e)
                

def inference_one(onnx_path, img_path):
    model = YOLOv5_ONNX(onnx_path)
    # model_input_size = (448, 768)
    model_input_size = (640, 640)
    img0, img, src_size = model.pre_process(img_path, img_size=model_input_size)

    pred = model.inference(img)

    out_bbx = model.post_process(pred, src_size, img_size=model_input_size)
    model.vis_result(out_bbx)




    

if __name__ == '__main__':
    # # onnx_path = r"/home/zengyifan/wujiahu/yolo/yolov5-6.2/runs/train/006_768_20230313_2_cls/weights/best.onnx"
    # # onnx_path = r"D:\Gosion\Projects\coal_conveying_corridor\weights\helmet_detection\helmet_det_yolov5s_640_640_v1.0.0.onnx"
    # # onnx_path = r"D:\Gosion\Projects\管网LNG\gitee\pipechina_beihaihaikou\weights\violated_sitting_detection\violated_sitting_det_yolov5s_640_640_v1.0.0.onnx"
    # # onnx_path = r"D:\Gosion\Projects\管网LNG\gitee\pipechina_beihaihaikou\weights\out_guardarea_detection\out_guardarea_det_yolov5s_640_640_v1.0.0.onnx"
    # # onnx_path = r"D:\Gosion\Python\yolov5-master\runs\train\003.violated_sitting_det_v2\weights\best.onnx"
    # onnx_path = r"D:\Gosion\Python\yolov5-master\runs\train\005.calling_det_v1\weights\best.onnx"
    # # onnx_path = r"D:\Gosion\Python\yolov5-master\runs\train\004.out_guardarea_det_v2\weights\best.onnx"
    # # onnx_path = r"E:\GraceKafuu\Python\ultralytics-main\yolov8s.onnx"
    # # img_path = "/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/paper/Image9685.jpg"
    # # img_path = r"D:\Gosion\Projects\data\images\southeast.jpg"
    # # img_path = r"D:\Gosion\Projects\data\images\20250217151944.jpg"
    # # img_path = r"D:\Gosion\Projects\GuanWangLNG\loubaowubao-0218\sitting\test_0000006.png"
    # img_path = r"D:\Gosion\Projects\data\images\20250220165421.png"
    # # img_path = r"D:\Gosion\Projects\data\images\20250220165611.jpg"

    # model = YOLOv5_ONNX(onnx_path)
    # # model_input_size = (448, 768)
    # model_input_size = (640, 640)
    # img0, img, src_size = model.pre_process(img_path, img_size=model_input_size)
    # print("src_size: ", src_size)

    # t1 = time.time()
    # pred = model.inference(img)
    # t2 = time.time()
    # print("{:.12f}s".format(t2 - t1))

    # # tt = []
    # # for i in range(100):
    # #     t1 = time.time()
    # #     pred = model.inference(img)
    # #     t2 = time.time()
    # #     print(t2 - t2)
    # #     tt.append(t2 - t1)
    
    # # print(np.mean(tt))

    # out_bbx = model.post_process(pred, src_size, img_size=model_input_size)
    # print("out_bbx: ", out_bbx)
    # for b in out_bbx:
    #     cv2.rectangle(img0, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
    #     # cv2.putText(img0, "smoking: {:.2f} concentration: {}".format(b[4], 2), (b[0], b[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    #     cv2.putText(img0, "{}: {:.2f}".format(b[5], b[4]), (b[0], b[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # # cv2.imwrite("/home/zengyifan/wujiahu/data/006.Fire_Smoke_Det/others/paper/Image9685_pred.jpg", img0)
    # # cv2.imwrite(r"D:\Gosion\Projects\data\images\southeast_pred.jpg", img0)
    # cv2.imshow("test", img0)
    # cv2.waitKey(0)

    # data_path=r"D:\Gosion\Projects\003.Sitting_Det\data\v2\train\images"
    # model_path=r"D:\Gosion\Projects\管网LNG\gitee\pipechina_beihaihaikou\weights\violated_sitting_detection\violated_sitting_det_yolov5s_640_640_v1.0.0.onnx"
    # # data_path=r"D:\Gosion\Projects\004.OutGuardArea_Det\data\v2\train\images"
    # # model_path=r"D:\Gosion\Projects\管网LNG\gitee\pipechina_beihaihaikou\weights\out_guardarea_detection\out_guardarea_det_yolov5s_640_640_v1.0.0.onnx"
    # yolo_inference_save_labels(data_path=data_path, model_path=model_path)
    # data_path=r"D:\Gosion\Projects\GuanWangLNG\images"
    # # model_path=r"D:\Gosion\Python\yolov5-master\runs\train\003.violated_sitting_det_v2\weights\best.onnx"
    # model_path=r"D:\Gosion\Python\yolov5-master\runs\train\004.out_guardarea_det_v2\weights\best.onnx"
    # yolo_inference_save_labels(data_path=data_path, model_path=model_path)

    # data_path=r"D:\Gosion\Projects\GuanWangLNG\loubaowubao-0218\sitting\images"
    # model_path=r"D:\Gosion\Python\yolov5-master\runs\train\003.violated_sitting_det_v2\weights\best.onnx"
    # yolo_inference_save_labels(data_path=data_path, model_path=model_path)

    # data_path=r"D:\Gosion\Projects\001.Leaking_Liquid_Det\data\20250219\v2\Random_Selected\images"
    # model_path=r"D:\Gosion\Projects\管网LNG\gitee\pipechina_beihaihaikou\weights\leaking_liquid_detection\leaking_liquid_det_yolov5s_640_640_v1.0.0.onnx"
    # yolo_inference_save_labels(data_path=data_path, model_path=model_path)

    # data_path=r"D:\Gosion\Projects\001.Leaking_Liquid_Det\data\20250219\v2\train\images"
    # model_path=r"D:\Gosion\Python\yolov5-master\runs\train\001.leaking_liquid_det_v2_780\weights\best.onnx"
    # yolo_inference_save_labels(data_path=data_path, model_path=model_path)

    # data_path=r"G:\Gosion\data\007.PPE_Det\data\v1\train\images"
    # model_path=r"D:\Gosion\code\gitee\GuanWangLNG\src\pipechina_beihaihaikou\weights\helmet_detection\helmet_det_yolov5s_640_640_v1.0.1.onnx"
    # yolo_inference_save_labels(data_path=data_path, model_path=model_path, addStr="helmet")

    # data_path=r"G:\Gosion\data\007.PPE_Det\data\v1\no_person\images"
    # model_path=r"D:\Gosion\code\others\Python\yolov5-master\yolov5s.onnx"
    # yolo_inference_save_labels(data_path=data_path, model_path=model_path, addStr="person")
    
    # onnx_path=r"D:\Gosion\code\others\Python\yolov5-master\runs\train\007.PPE_Det_v1\weights\best.onnx"
    # img_path=r"G:\Gosion\data\000.OpenDatasets\VOC2028\VOC2028\SafetyHelmet\images\train2028\000002.jpg"
    # inference_one(onnx_path, img_path)

    # onnx_path=r"D:\Gosion\code\others\Python\yolov5-master\runs\train\007.PPE_Det_v1\weights\best.onnx"
    # data_path = r"G:\Gosion\data\000.OpenDatasets\VOC2028\VOC2028\JPEGImages"
    # file_list = sorted(os.listdir(data_path))
    # for f in file_list:
    #     fname = os.path.splitext(f)[0]
    #     f_abs_path = data_path + "/{}".format(f)
    #     inference_one(onnx_path, f_abs_path)


    data_path=r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\v2\500\images"
    # model_path=r"D:\Gosion\code\gitee\GuanWangLNG\src\pipechina_beihaihaikou\weights\smoking_detection\smoking_det_yolov5s_640_640_v1.0.0.onnx"
    # model_path=r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\helmet_head_model\helmet_det_yolov5s_640_640_v1.0.1.onnx"
    # model_path=r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\cotta_model\best.onnx"
    model_path=r"G:\Gosion\data\000.ShowRoom_Algrithom\Person_Helmet_T-shirt\model\best.onnx"
    yolo_inference_save_labels(data_path=data_path, model_path=model_path, model_label=3, dst_label=3, cls_name="T-shirt")

            

