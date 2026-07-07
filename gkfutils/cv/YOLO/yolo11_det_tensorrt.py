import cv2
import os
import sys
import time
import numpy as np
import tensorrt as trt
import torch
import torchvision
from ultralytics import YOLO

# # ==================== 【第一道防线：Windows DLL 冲突防御】 ====================
# if sys.platform == "win32":
#     for path in sys.path:
#         if "site-packages" in path and os.path.isdir(path):
#             trt_libs_dir = os.path.join(path, "tensorrt_libs")
#             if os.path.exists(trt_libs_dir):
#                 os.add_dll_directory(trt_libs_dir)
#                 print(f"[INFO] 已成功锁定并注入 TensorRT 底层库路径: {trt_libs_dir}")
#                 break

# COCO 数据集 80 个标准类别名称
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# class YOLO11TensorRT:
#     def __init__(self, engine_path, conf_thres=0.25, iou_thres=0.45):
#         self.logger = trt.Logger(trt.Logger.WARNING)
#         self.conf_thres = conf_thres
#         self.iou_thres = iou_thres
        
#         # 1. 读取引擎文件
#         print(f"[INFO] 正在读取引擎文件: {engine_path}")
#         with open(engine_path, "rb") as f:
#             file_bytes = f.read()

#         # 2. 自动剥离 Ultralytics 元数据头部
#         engine_bytes = file_bytes
#         try:
#             if len(file_bytes) > 4:
#                 meta_len = int.from_bytes(file_bytes[:4], byteorder="little", signed=True)
#                 if 0 < meta_len < len(file_bytes) - 4:
#                     meta_str = file_bytes[4 : 4 + meta_len].decode("utf-8", errors="ignore")
#                     if any(k in meta_str for k in ["names", "stride", "task", "version"]):
#                         print(f"[INFO] 检测到内嵌的 Ultralytics 元数据（长度: {meta_len} 字节），正在自动剥离...")
#                         engine_bytes = file_bytes[4 + meta_len : ]
#         except Exception as e:
#             print(f"[WARNING] 尝试解析元数据时出现异常: {e}")

#         # 3. 反序列化
#         print(f"[INFO] 正在反序列化加载纯净的 TensorRT 引擎...")
#         self.runtime = trt.Runtime(self.logger)
#         self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
            
#         if self.engine is None:
#             raise RuntimeError("错误：无法反序列化引擎文件！")
            
#         self.context = self.engine.create_execution_context()
        
#         # TensorRT 10.x 命名绑定机制获取输入输出
#         self.input_name = self.engine.get_tensor_name(0)
#         self.output_name = self.engine.get_tensor_name(1)
#         self.input_shape = self.engine.get_tensor_shape(self.input_name)
#         self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
#         print(f"[INFO] 引擎加载成功！输入张量: {self.input_name} {tuple(self.input_shape)}")
#         print(f"[INFO] 引擎加载成功！输出张量: {self.output_name} {tuple(self.output_shape)}")
        
#         # 4. 显存托管预分配
#         self.output_tensor = torch.empty(tuple(self.output_shape), dtype=torch.float32, device="cuda")
        
#         # 5. 【核心修复】创建非默认 CUDA 流，消除 TRT Stream 性能警告
#         self.torch_stream = torch.cuda.Stream()

#     def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
#         """高度还原 Ultralytics 标准的预处理：保持比例缩放，并进行灰色填充"""
#         shape = img.shape[:2]  # 当前图片的 [高, 宽]
        
#         # 计算缩放比例
#         r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
#         # 计算缩放后的未填充尺寸
#         new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
#         dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽高总填充量
        
#         dw /= 2  # 平分到两边
#         dh /= 2
        
#         if shape[::-1] != new_unpad:  # 缩放图片
#             img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
#         top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#         left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
#         # 填充四周形成 640x640 矩阵
#         img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
#         return img, r, (left, top)

#     def preprocess(self, frame):
#         """完整的预处理流水线"""
#         self.orig_shape = frame.shape

#         input_size = (self.input_shape[2], self.input_shape[3]) # (640, 640)
        
#         # 1. Letterbox 变换
#         img_padded, self.ratio, self.pad = self.letterbox(frame, input_size)
        
#         # 2. BGR 转 RGB -> HWC 转 CHW
#         img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
#         img_chw = img_rgb.transpose((2, 0, 1))
        
#         # 3. 归一化并确保内存连续
#         img_norm = np.ascontiguousarray(img_chw, dtype=np.float32) / 255.0
        
#         # 4. 推送到显存
#         input_tensor = torch.from_numpy(img_norm).unsqueeze(0).to("cuda")
#         return input_tensor

#     def infer(self, input_tensor):
#         """核心推理流程"""
#         # 1. 前处理
#         # input_tensor, ratio, (pad_w, pad_h) = self.preprocess(frame)
        
#         # 2. 绑定执行上下文的张量地址
#         self.context.set_input_shape(self.input_name, input_tensor.shape)
#         self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
#         self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())
        
#         # 3. 使用独立的独立 CUDA 流执行异步推理
#         with torch.cuda.stream(self.torch_stream):
#             self.context.execute_async_v3(stream_handle=self.torch_stream.cuda_stream)
#             self.torch_stream.synchronize() # 精确同步本流
        
#         # 4. 获取原始输出数据
#         raw_output = self.output_tensor.cpu().numpy()

#         return raw_output

#     def postprocess(self, raw_output):
#         """后处理：解析 (1, 84, 8400) 矩阵并进行 OpenCV NMS 过滤"""
#         # 剥离 batch 维度，由 (1, 84, 8400) 变为 (84, 8400)
#         output = raw_output[0]
#         # 转置矩阵，变为 (8400, 84) 方便每行读取
#         output = output.T
        
#         boxes = []
#         scores = []
#         class_ids = []
        
#         (pad_w, pad_h) = self.pad
#         orig_h, orig_w = self.orig_shape[:2]
        
#         # 1. 提取矩形框与分类得分
#         box_cls_matrix = output
#         # 前 4 列为 cx, cy, w, h
#         pred_boxes = box_cls_matrix[:, :4]
#         # 后 80 列为类别概率
#         pred_scores = box_cls_matrix[:, 4:]
        
#         # 找出每行中概率最大的类别及其得分
#         max_scores = np.max(pred_scores, axis=1)
#         max_class_ids = np.argmax(pred_scores, axis=1)
        
#         # 置信度初筛过滤，极大加速后续计算
#         keep_indices = np.where(max_scores > self.conf_thres)[0]
        
#         for idx in keep_indices:
#             cx, cy, w, h = pred_boxes[idx]
#             score = max_scores[idx]
#             class_id = max_class_ids[idx]
            
#             # 从 [中心点, 宽高] 转换为 [左上角x, 左上角y, 宽, 高]（OpenCV NMS 要求的标准格式）
#             x = cx - w / 2
#             y = cy - h / 2
            
#             # 核心算法：反向映射坐标回去（扣除 Letterbox 的填充灰边，并除以缩放比例）
#             x_raw = (x - pad_w) / self.ratio
#             y_raw = (y - pad_h) / self.ratio
#             w_raw = w / self.ratio
#             h_raw = h / self.ratio
            
#             # 防止框越界超出原始图像范围
#             x_raw = max(0, min(x_raw, orig_w - 1))
#             y_raw = max(0, min(y_raw, orig_h - 1))
#             w_raw = min(w_raw, orig_w - x_raw)
#             h_raw = min(h_raw, orig_h - y_raw)
            
#             boxes.append([int(x_raw), int(y_raw), int(w_raw), int(h_raw)])
#             scores.append(float(score))
#             class_ids.append(int(class_id))
            
#         # 2. 执行非极大值抑制 (NMS)，消除重叠的多余目标框
#         indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        
#         final_boxes = []
#         final_scores = []
#         final_class_ids = []
        
#         if len(indices) > 0:
#             for i in indices.flatten():
#                 final_boxes.append(boxes[i])
#                 final_scores.append(scores[i])
#                 final_class_ids.append(class_ids[i])
                
#         return final_boxes, final_scores, final_class_ids

#     def draw_predictions(self, frame, boxes, scores, class_ids):
#         """在图上绘制矩形框、标签和置信度"""
#         result_img = frame.copy()
#         for box, score, class_id in zip(boxes, scores, class_ids):
#             x, y, w, h = box
#             label = f"{COCO_CLASSES[class_id]}: {score:.2f}"
            
#             # 随机生成或指定颜色
#             color = (0, 255, 0) # 绿色
            
#             # 画目标边界框
#             cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
            
#             # 绘制文字背景小标签
#             (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#             cv2.rectangle(result_img, (x, y - th - 5), (x + tw, y), color, -1)
            
#             # 写上标签文字
#             cv2.putText(result_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
#         return result_img


class YOLO11TensorRT:
    def __init__(self, engine_path, conf_thres=0.25, iou_thres=0.45):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        print(f"[INFO] 正在读取引擎文件: {engine_path}")
        with open(engine_path, "rb") as f:
            file_bytes = f.read()

        engine_bytes = file_bytes
        try:
            if len(file_bytes) > 4:
                meta_len = int.from_bytes(file_bytes[:4], byteorder="little", signed=True)
                if 0 < meta_len < len(file_bytes) - 4:
                    meta_str = file_bytes[4 : 4 + meta_len].decode("utf-8", errors="ignore")
                    if any(k in meta_str for k in ["names", "stride", "task", "version"]):
                        print(f"[INFO] 检测到内嵌的 Ultralytics 元数据（长度: {meta_len} 字节），正在自动剥离...")
                        engine_bytes = file_bytes[4 + meta_len : ]
        except Exception as e:
            print(f"[WARNING] 尝试解析元数据时出现异常: {e}")

        print(f"[INFO] 正在反序列化加载纯净的 TensorRT 引擎...")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
            
        if self.engine is None:
            raise RuntimeError("错误：无法反序列化引擎文件！")
            
        self.context = self.engine.create_execution_context()
        
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        print(f"[INFO] 引擎加载成功！输入张量: {self.input_name} {tuple(self.input_shape)}")
        print(f"[INFO] 引擎加载成功！输出张量: {self.output_name} {tuple(self.output_shape)}")
        
        # 显存托管预分配
        self.output_tensor = torch.empty(tuple(self.output_shape), dtype=torch.float32, device="cuda")
        self.torch_stream = torch.cuda.Stream()

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (left, top)

    def preprocess(self, frame):
        self.orig_shape = frame.shape
        input_size = (self.input_shape[2], self.input_shape[3])
        
        img_padded, self.ratio, self.pad = self.letterbox(frame, input_size)
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose((2, 0, 1))
        img_norm = np.ascontiguousarray(img_chw, dtype=np.float32) / 255.0
        
        input_tensor = torch.from_numpy(img_norm).unsqueeze(0).to("cuda")
        return input_tensor

    # def infer(self, input_tensor):
    #     self.context.set_input_shape(self.input_name, input_tensor.shape)
    #     self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
    #     self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())
        
    #     with torch.cuda.stream(self.torch_stream):
    #         self.context.execute_async_v3(stream_handle=self.torch_stream.cuda_stream)
    #         self.torch_stream.synchronize()
        
    #     # ⚡【关键改动】不再在此处执行 .cpu().numpy()，保持在显存中直接交给后处理
    #     return self.output_tensor

    def infer(self, input_tensor):
        self.context.set_input_shape(self.input_name, input_tensor.shape)
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())
        
        # 创建 CUDA 事件用于精准测量 GPU 时间
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record(self.torch_stream)
        with torch.cuda.stream(self.torch_stream):
            self.context.execute_async_v3(stream_handle=self.torch_stream.cuda_stream)
        end_event.record(self.torch_stream)
        
        self.torch_stream.synchronize()
        
        # 计算纯显卡推理耗时（毫秒）
        infer_time_ms = start_event.elapsed_time(end_event)
        print(f"[GPU Pure Infer] {infer_time_ms:.2f} ms")
        
        return self.output_tensor

    def postprocess(self, output_tensor):
        """🚀 全流程 GPU 加速后处理"""
        # 1. 矩阵维度变换 (1, 84, 8400) -> (8400, 84)
        output = output_tensor[0].T
        
        pred_boxes = output[:, :4]   # (8400, 4) -> cx, cy, w, h
        pred_scores = output[:, 4:]  # (8400, 80) -> classes scores
        
        # 2. GPU 端计算最大得分和分类 ID
        max_scores, class_ids = torch.max(pred_scores, dim=1)
        
        # 3. 过滤低置信度候选框
        keep_mask = max_scores > self.conf_thres
        if not keep_mask.any():
            return [], [], []
            
        boxes = pred_boxes[keep_mask]
        scores = max_scores[keep_mask]
        class_ids = class_ids[keep_mask]
        
        # 4. 坐标转换 [cx, cy, w, h] -> [x1, y1, x2, y2]
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        
        # 5. ⚡ 调用高效的 GPU Batched NMS
        keep_indices = torchvision.ops.batched_nms(boxes_xyxy, scores, class_ids, self.iou_thres)
        if len(keep_indices) == 0:
            return [], [], []
            
        # 6. 核心优化：此时只将经过 NMS 过滤后留下的极少数（几个到几十个）目标框传回 CPU
        final_boxes = boxes_xyxy[keep_indices].cpu().numpy()
        final_scores = scores[keep_indices].cpu().numpy()
        final_class_ids = class_ids[keep_indices].cpu().numpy()
        
        # 7. 在 CPU 上还原原始分辨率坐标
        pad_w, pad_h = self.pad
        orig_h, orig_w = self.orig_shape[:2]
        
        restored_boxes = []
        for box in final_boxes:
            x1_raw = (box[0] - pad_w) / self.ratio
            y1_raw = (box[1] - pad_h) / self.ratio
            x2_raw = (box[2] - pad_w) / self.ratio
            y2_raw = (box[3] - pad_h) / self.ratio
            
            # 转为 OpenCV 标准输入格式 [x_left, y_top, width, height]
            w_raw = x2_raw - x1_raw
            h_raw = y2_raw - y1_raw
            
            x_clip = max(0, min(x1_raw, orig_w - 1))
            y_clip = max(0, min(y1_raw, orig_h - 1))
            w_clip = min(w_raw, orig_w - x_clip)
            h_clip = min(h_raw, orig_h - y_clip)
            
            restored_boxes.append([int(x_clip), int(y_clip), int(w_clip), int(h_clip)])
            
        return restored_boxes, final_scores.tolist(), final_class_ids.tolist()

    def draw_predictions(self, frame, boxes, scores, class_ids):
        result_img = frame.copy()
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box
            label = f"{COCO_CLASSES[class_id]}: {score:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_img, (x, y - th - 5), (x + tw, y), color, -1)
            cv2.putText(result_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return result_img


def main_image():
    engine_file = "../weights/yolo11n_fp32.engine"
    # engine_file = "../weights/yolo11n_fp16.engine"
    # engine_file = "../weights/yolo11n_int8.engine"
    
    test_image_path = r"../data/image/bus.jpg"
    frame = cv2.imread(test_image_path)

    # 初始化模型
    trt_model = YOLO11TensorRT(engine_file, conf_thres=0.25, iou_thres=0.45)

    t0 = time.time()

    input_tensor = trt_model.preprocess(frame)
    
    # # 跑一次空推理预热 GPU 算力通道
    # _ = trt_model.infer(input_tensor)
    
    raw_output = trt_model.infer(input_tensor)
    boxes, scores, class_ids = trt_model.postprocess(raw_output)
    
    total_cost = (time.time() - t0) * 1000
    print(f"总计算耗时（含前后处理）: {total_cost:.2f} ms")
    print(f"检测到的目标数量: {len(boxes)}")
    
    result_frame = trt_model.draw_predictions(frame, boxes, scores, class_ids)
    
    cv2.imshow("YOLO11 Real-time TensorRT", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main_video():
    # engine_file = "../weights/yolo11n_fp32.engine"
    # engine_file = "../weights/yolo11n_fp16.engine"
    engine_file = "../weights/yolo11n_int8.engine"
    
    trt_model = YOLO11TensorRT(engine_file, conf_thres=0.25, iou_thres=0.45)


    # 打开摄像头或视频流（此处以本地默认摄像头为例）
    cap = cv2.VideoCapture("../data/video/3209298-hd_1920_1080_25fps.mp4")
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("开始实时推理，按 'q' 键退出...")

    # 关键点：使用 stream=True。这会使 model 返回一个 Python 生成器 (Generator)
    # 每次循环只在内存中保留当前帧的推理结果，极大优化高频连续采集下的吞吐量

    while True:
        ret, frame = cap.read()

        t0 = time.time()

        input_tensor = trt_model.preprocess(frame)
        raw_output = trt_model.infer(input_tensor)
        boxes, scores, class_ids = trt_model.postprocess(raw_output)

        total_cost = (time.time() - t0) * 1000
        print(f"总计算耗时（含前后处理）: {total_cost:.2f} ms")
        print(f"检测到的目标数量: {len(boxes)}")
        
        result_frame = trt_model.draw_predictions(frame, boxes, scores, class_ids)

        cv2.imshow("YOLO11 Real-time TensorRT", result_frame)
            
        # 响应键盘事件退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def ultralytics_main():
    # 加载 TensorRT 模型
    # model = YOLO("../weights/yolo11n_fp32.engine", task="detect")
    # model = YOLO("../weights/yolo11n_fp16.engine", task="detect")
    model = YOLO("../weights/yolo11n_int8.engine", task="detect")

    # 打开摄像头或视频流（此处以本地默认摄像头为例）
    cap = cv2.VideoCapture("../data/video/3209298-hd_1920_1080_25fps.mp4")
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("开始实时推理，按 'q' 键退出...")

    # 关键点：使用 stream=True。这会使 model 返回一个 Python 生成器 (Generator)
    # 每次循环只在内存中保留当前帧的推理结果，极大优化高频连续采集下的吞吐量

    while True:
        ret, frame = cap.read()
    
        results = model(frame, stream=True, conf=0.25, device=0, imgsz=640)

        for result in results:
            # 渲染当前的推理帧
            annotated_frame = result.plot()
            
            # 显示画面
            cv2.imshow("YOLO11 Real-time TensorRT", annotated_frame)
            
        # 响应键盘事件退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main_image()
    main_video()
    # ultralytics_main()