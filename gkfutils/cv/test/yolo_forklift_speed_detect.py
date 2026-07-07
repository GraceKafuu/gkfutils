import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import defaultdict



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


def calculate_speed(track_history, track_id, dt, pixel_per_meter):
    """
    稳定速度计算算法：
    1. 取最近 5 帧中心点
    2. 计算总位移（像素）
    3. 转为真实距离（米）
    4. 速度 = 距离 / 时间 → km/h
    5. 滑动平滑，防止跳变
    """
    history = track_history[track_id]
    if len(history) < 5:
        return 0.0

    # 最近5帧
    x1, y1, t1 = history[-5]
    x2, y2, t2 = history[-1]

    # 像素距离
    dist_pixel = np.hypot(x2 - x1, y2 - y1)

    # 时间差
    delta_t = t2 - t1
    if delta_t <= 0:
        return 0.0

    # 真实距离
    dist_m = dist_pixel / pixel_per_meter

    # 速度 m/s → km/h
    speed_mps = dist_m / delta_t
    speed_kmh = speed_mps * 3.6

    return round(speed_kmh, 1)

    
def detect_forklift_speed():
    # 初始化你自己的 YOLO11
    # ====================== 配置 ======================
    # MODEL_PATH = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\detect\forklift_v2\weights\best.onnx"          # 你的ONNX模型
    # VIDEO_PATH = r"G:\Gosion\projects\011.ZhongyuanGuanxi\data\forklift\forklift_videos\17751080339262039575089774800897.mp4"   # 输入视频
    # VIDEO_PATH = r"G:\Gosion\projects\011.ZhongyuanGuanxi\data\1号测点视频\17761605493372043990498942857218.mp4"   # 输入视频
    VIDEO_PATH = r"G:\Gosion\projects\011.ZhongyuanGuanxi\data\forklift\forklift_videos\17751080350572039574937311850498.mp4"   # 输入视频
    # VIDEO_PATH = r"G:\Gosion\projects\006.Belt_Torn_Det\data\resources\videos\Video_2025_08_27_171729_1_2.avi"   # 输入视频
    CLASS_NAMES = ["forklift", "person"] # 0:叉车 1:人
    # PIXEL_PER_METER = 65                # 像素对应实际1米（根据场景调整）
    CONF_THRESH = 0.35                  # 置信度
    IOU_THRESH = 0.5
    # ==================================================

    # model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\detect\Forklift_Det_2\weights\best.onnx"
    model_path = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\detect\forklift_v2\weights\best.onnx"
    yolo11 = YOLO11(model_path, 0.60, 0.45)

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1.0 / fps
    print(f"视频 FPS: {fps}")

    # 追踪历史：ID → [(x, y, time), ...]
    track_history = defaultdict(list)
    forklift_exist = False
    current_speed = 0.0
    last_time = 0
    speeds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo11.detect(frame, show=False)

        has_forklift_current = False
        forklift_box = None

        # 遍历检测结果
        for box in results:
            x, y, w, h, score, cls_id = box
            if cls_id == 0:  # 只处理叉车
                has_forklift_current = True
                forklift_exist = True
                forklift_box = box

                # 计算中心点
                cx = int(x + w // 2)
                cy = int(y + h // 2)

                # 保存轨迹
                track_history[0].append((cx, cy, last_time))

                # 限制历史长度
                if len(track_history[0]) > 20:
                    track_history[0].pop(0)

                # 计算速度
                PIXEL_PER_METER = w / 1.5
                current_speed = calculate_speed(track_history, 0, frame_interval, PIXEL_PER_METER)

        # ====================== 绘制 ======================
        if has_forklift_current and forklift_box is not None:
            x, y, w, h, score, _ = forklift_box
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, f"Forklift {score:.2f}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示状态
        status = "Forklift: YES" if forklift_exist else "Forklift: NO"
        speed_str = f"Speed: {current_speed:.1f} km/h" if forklift_exist else "Speed: -"
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, speed_str, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示画面
        cv2.imshow("Forklift Speed Detection", frame)
        if cv2.waitKey(1) == 27:
            break

        # 视频帧率为25FPS, 每帧0.04秒
        last_time += 0.04
        speeds.append(current_speed)

    cap.release()
    cv2.destroyAllWindows()

    # ====================== 最终输出 ======================
    print("\n===== 检测完成 =====")
    print(f"视频中是否存在叉车: {'是' if forklift_exist else '否'}")
    if forklift_exist:
        speed_mean = np.mean(speeds)
        print(f"叉车的平均速度约: {speed_mean:.1f} km/h")


if __name__ == "__main__":
    detect_forklift_speed()

