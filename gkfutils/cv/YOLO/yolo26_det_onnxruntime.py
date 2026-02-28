# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
 
import argparse
from typing import List, Tuple
 
import cv2
import numpy as np
import onnxruntime as ort
import torch
import time
from ultralytics.utils.checks import check_requirements


class YOLO26:
    def __init__(self, onnx_model: str, input_image: str, confidence_thres: float):
        """
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres

        # Create an inference session using the ONNX model and specify execution providers
        self.session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
 
        # 类别索引
        self.classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
 
        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
 
    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        图片缩放，等比例缩放加114补黑边
        """
        shape = img.shape[:2]  # current shape [height, width]
 
        # 缩放尺寸
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
 
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
 
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
 
        return img, (top, left),r
 
    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int) -> None:
        """Draw bounding boxes and labels on the input image based on the detected objects."""
        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = box
 
        # Retrieve the color for the class ID
        color = self.color_palette[class_id]
 
        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
 
        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"
 
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
 
    def preprocess(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        前处理
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(self.input_image)
 
        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]
 
        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
 
        img, pad,r = self.letterbox(img, (self.input_width, self.input_height))
 
        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0
 
        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
 
        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
 
        # Return the preprocessed image data
        return image_data, pad, r
    
    def postprocess_yolo26(self, input_image: np.ndarray, output: List[np.ndarray], pad: Tuple[int, int],r) -> np.ndarray:
        # Transpose and squeeze the output to match the expected shape
        outputs = np.squeeze(output[0])
        boxes, scores, class_ids = [], [], []
        for det in outputs:
            x1, y1, x2, y2,  conf,class_id= det
            if conf >= self.confidence_thres:
                boxes.append([
                    int((x1 - pad[1]) / r),
                    int((y1 - pad[0]) / r),
                    int((x2 - pad[1]) / r),
                    int((y2 - pad[0]) / r)
                ])
                scores.append(conf)
                class_ids.append(int(class_id))
        # Iterate over the selected indices after non-maximum suppression
        for i in range(len(class_ids)):
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id=class_ids[i]
            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)
 
        # Return the modified input image
        return input_image
 
    def main(self) -> np.ndarray:
        """
        Perform inference using an ONNX model and return the output image with drawn detections.
        Returns:
            (np.ndarray): The output image with drawn detections.
        """
        
        # Get the model inputs
        model_inputs = self.session.get_inputs()
 
        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
 
        # Preprocess the image data
        img_data, pad, r = self.preprocess()
 
        # Run inference using the preprocessed image data
        t1=time.time()
        outputs = self.session.run(None, {model_inputs[0].name: img_data})
        t2=time.time()
        print("inference time: ", t2-t1)
 
        # Perform post-processing on the outputs to obtain output image
        return self.postprocess_yolo26(self.img, outputs, pad,r)
 
 
if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=r"G:\Gosion\code\Python\ultralytics-8.4.18\yolo26s.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", type=str, default=r"G:\Gosion\projects\000.Test_Data\images\zidane.jpg", help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
 
    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")
 
    # Create an instance of the YOLOv8 class with the specified arguments
    detection = YOLO26(args.model, args.img, args.conf_thres)
 
    # Perform object detection and obtain the output image
    output_image = detection.main()
    
    # for _ in range(30):
    #     output_image = detection.main()

    # Display the output image in a window
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)
    # cv2.imwrite("output.jpg", output_image)
 
    # Wait for a key press to exit
    cv2.waitKey(0)