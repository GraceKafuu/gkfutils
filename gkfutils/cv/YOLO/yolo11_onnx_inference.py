# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm


class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.array): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


class YOLO11_ORT:
    """YOLO model for handling inference and visualization."""
    def __init__(self, onnx_model, num_kpt=17, confidence_thres=0.60, iou_thres=0.45, num_classes=80):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.num_kpt = num_kpt
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Create an inference session using the ONNX model and specify execution providers
        self.session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Load the class names from the COCO dataset
        # self.classes = yaml_load(check_yaml("coco8.yaml"))["names"].

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(num_classes, 3))

        self.colors = Colors()
        self.limb_color = self.colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = self.colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.skeleton = [
                    [16, 14],
                    [14, 12],
                    [17, 15],
                    [15, 13],
                    [12, 13],
                    [6, 12],
                    [7, 13],
                    [6, 7],
                    [6, 8],
                    [7, 9],
                    [8, 10],
                    [9, 11],
                    [2, 3],
                    [1, 2],
                    [1, 3],
                    [2, 4],
                    [3, 5],
                    [4, 6],
                    [5, 7],
                ]

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
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

    def preprocess(self, input):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        if isinstance(input, str):
            self.img = cv2.imread(input)
        elif isinstance(input, Image.Image):
            self.img = self.pil2cv(input)
        else:
            assert isinstance(input, np.ndarray), f'input is not np.ndarray and is {type(input)}!'
            self.img = input

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def det_postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
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
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return input_image
    
    def det_detect(self, img_path):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        
        # Preprocess the image data
        img_data = self.preprocess(img_path)

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        output = self.det_postprocess(self.img, outputs)

        return output
    
    def cv2pil(self, image):
        assert isinstance(image, np.ndarray), f'Input image type is not cv2 and is {type(image)}!'
        if len(image.shape) == 2:
            return Image.fromarray(image)
        elif len(image.shape) == 3:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            return None

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
        
    def xywh2xyxy(self, xywh):
        xyxy = np.copy(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        return xyxy

    def nms(self, boxes, scores, keypoints, iou_threshold=0.45, candidate_size=200):
        """
        Args:
            boxex (N, 4): boxes in corner-form.
            scores (N, 1): scores.
            iou_threshold: intersection over union threshold.
            candidate_size: only consider the candidates with the highest scores.
        Returns:
            picked: a list of indexes of the kept boxes
        """
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[::-1]
        indexes = indexes[:candidate_size]

        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current)
            if len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[1:]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]

        return boxes[picked, :], scores[picked], keypoints[picked, :]


    def iou_of(self, boxes0, boxes1, eps=1e-5):
        """Return intersection-over-union (Jaccard index) of boxes.
        Args:
            boxes0 (N, 4): ground truth boxes.
            boxes1 (N or 1, 4): predicted boxes.
            eps: a small number to avoid 0 as denominator.
        Returns:
            iou (N): IoU values.
        """
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)


    def area_of(self, left_top, right_bottom):
        """Compute the areas of rectangles given two corners.
        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.

        Returns:
            area (N): return the area.
        """
        hw = np.clip((right_bottom - left_top), a_min=0.0, a_max=1.0)
        return hw[..., 0] * hw[..., 1]
    
    def pose_postprocess(self, preds, width_radio, height_radio, filter_threshold=0.25, iou_threshold=0.45):
        preds = preds.transpose([1, 0])

        preds = preds[preds[:, 4] > filter_threshold]
        if len(preds) > 0:
            boxes = preds[:, :4]
            boxes = self.xywh2xyxy(boxes)
            scores = preds[:, 4]
            keypoints = preds[:, 5:]

            boxes, scores, keypoints = self.nms(boxes, scores, keypoints, iou_threshold=iou_threshold)

            boxes[:, 0] *= width_radio
            boxes[:, 1] *= height_radio
            boxes[:, 2] *= width_radio
            boxes[:, 3] *= height_radio

            keypoints = keypoints.reshape([-1, self.num_kpt, 3])
            keypoints[:, :, 0] *= width_radio
            keypoints[:, :, 1] *= height_radio

        else:
            boxes = np.array([])
            scores = np.array([])
            keypoints = np.array([])

        return boxes, scores, keypoints
    
    def draw_kpts(self, img, kpts, shape=(640, 640), radius=5, kpt_line=True):
        """
        Plot keypoints on the image.

        Args:
            img (ndarray): Original image
            kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
            shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
            radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                    for human pose. Default is True.

        Note:
            `kpt_line=True` currently only supports human pose plotting.
        """
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in self.kpt_color[i]] if is_pose else self.colors(i)
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                cv2.circle(img, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(img, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

        return img
    
    def cal_angle_via_vector_cross(self, p1, p2, p3):
        """
        通过向量叉乘计算角度
        """

        v12 = p2 - p1
        v13 = p3 - p1

        v = v12[0] * v13[0] + v12[1] * v13[1]
        len_v12 = np.sqrt(v12[0] ** 2 + v12[1] ** 2)
        len_v13 = np.sqrt(v13[0] ** 2 + v13[1] ** 2)
        angle = np.arccos(v / (len_v12 * len_v13))
        angle = angle * 180 / np.pi

        return angle
    
    def sitting_or_standing(self, k, angle_thr=165):
        """
        -1: 判断不了 0: 坐着 1: 站着
        假如身体与双腿的夹角大于45°则认为是坐着
        """
        if k.shape[0] != 17: return -1
        k = k[:, :-1]
        angle_11_5_13 = self.cal_angle_via_vector_cross(k[11], k[5], k[13])
        angle_13_11_15 = self.cal_angle_via_vector_cross(k[13], k[11], k[15])
        angle_12_6_14 = self.cal_angle_via_vector_cross(k[12], k[6], k[14])
        angle_14_12_16 = self.cal_angle_via_vector_cross(k[14], k[12], k[16])

        sum = 0
        if angle_11_5_13 > angle_thr:
            sum += 1
        if angle_13_11_15 > angle_thr:
            sum += 1
        if angle_12_6_14 > angle_thr:
            sum += 1
        if angle_14_12_16 > angle_thr:
            sum += 1

        if sum >= 3:
            return "Standing"
        else:
            return "Sitting"

    def sitting_or_standing_detect(self, img_path):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        
        # Preprocess the image data
        img_data = self.preprocess(img_path)

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

        boxes, scores, keypoints = self.pose_postprocess(
            outputs[0][0],
            width_radio=self.img_width / self.input_width,
            height_radio=self.img_height / self.input_height,
        )

        for b, s, k in zip(boxes, scores, keypoints):
            b = list(map(round, b))
            self.img = self.draw_kpts(self.img, k, self.img.shape, radius=5, kpt_line=True)

            print("img_path: {}".format(img_path))
            res = self.sitting_or_standing(k, angle_thr=145)
            print("Res: {}\n".format(res))
            cv2.rectangle(self.img, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
            cv2.putText(self.img, "Person: {:.2f} {}".format(s, res), (b[0], b[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    def pose_detect(self, img_path):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        
        # Preprocess the image data
        img_data = self.preprocess(img_path)

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

        boxes, scores, keypoints = self.pose_postprocess(
            outputs[0][0],
            width_radio=self.img_width / self.input_width,
            height_radio=self.img_height / self.input_height,
        )

        for b, s, k in zip(boxes, scores, keypoints):
            b = list(map(round, b))
            self.img = self.draw_kpts(self.img, k, self.img.shape, radius=5, kpt_line=True)

            cv2.rectangle(self.img, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
            cv2.putText(self.img, "0: {:.2f}".format(s), (b[0], b[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return self.img


if __name__ == "__main__":
    # model_path = r"D:\Gosion\Python\ultralytics-8.3.72\weights\yolo11s.onnx"
    # model_path = r"D:\Gosion\Python\ultralytics-8.3.72\weights\yolo11s-pose.onnx"
    model_path = r"D:\Gosion\Python\ultralytics-8.3.72\runs\train-pose\006.belt_torn_pose_v2\weights\best.onnx"
    model = YOLO11_ORT(model_path, num_kpt=2)

    # # data_path = r"D:\Gosion\Projects\003.Sitting_Det\v1\val\images"
    # # data_path = r"D:\Gosion\Projects\003.Sitting_Det\v1\val\test_images"
    # data_path = r"D:\Gosion\Projects\003.Sitting_Det\v1\val\test_2"
    # save_path = os.path.abspath(os.path.join(data_path, "..")) + "/sitting_or_standing_results"
    # os.makedirs(save_path, exist_ok=True)

    # file_list = sorted(os.listdir(data_path))
    # for f in tqdm(file_list):
    #     f_abs_path = data_path + "/{}".format(f)
    #     f_dst_path = save_path + "/{}".format(f)
    #     output = model.pose_detect(f_abs_path)
    #     cv2.imwrite(f_dst_path, output)


    # # f_abs_path = r"D:\Gosion\Projects\003.Sitting_Det\data\v1\val\images\192.168.45.192_01_20250117112730288.jpg"
    # f_abs_path = r"D:\Gosion\Projects\006.Belt_Torn_Det\data\det_pose\v1\v1_102_20250304\val\images\1_output_000000169.jpg"
    # output = model.pose_detect(f_abs_path)
    # # output = model.det_detect(f_abs_path)

    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.imshow("output", output)
    # cv2.waitKey(0)


    data_path = r"D:\Gosion\Projects\006.Belt_Torn_Det\data\pose\v2\val_not_labeled\images"
    save_path = data_path + "_vis_yolo11_pose"
    os.makedirs(save_path, exist_ok=True)

    file_list = sorted(os.listdir(data_path))
    for f in tqdm(file_list):
        f_abs_path = data_path + "/{}".format(f)
        f_dst_path = save_path + "/{}".format(f)
        output = model.pose_detect(f_abs_path)
        cv2.imwrite(f_dst_path, output)