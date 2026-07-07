import cv2
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import onnxruntime as ort

# ====================== YOLO11 类（来自 main1.py） ======================
class YOLO11:
    def __init__(self, model_path: str, conf_threshold: float, iou_threshold: float):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.color_palette = np.random.uniform(0, 255, size=(80, 3))
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(self.model_path, providers=providers or available)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

    def letterbox(self, img: np.ndarray, new_shape=(640, 640)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, (top, left)

    def preprocess(self, img):
        self.img = img
        self.img_height, self.img_width = self.img.shape[:2]
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img_rgb, pad = self.letterbox(img_rgb, (self.input_width, self.input_height))
        image_data = np.array(img_rgb) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = image_data[None].astype(np.float32)
        return image_data, pad

    def postprocess(self, output: list, pad: tuple):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.conf_threshold:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        out = []
        for i in np.array(indices).flatten():
            box = boxes[int(i)]
            score = scores[int(i)]
            class_id = class_ids[int(i)]
            box.extend([score, class_id])
            out.append(box)
        return out

    def detect(self, img):
        img_data, pad = self.preprocess(img)
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        return self.postprocess(outputs, pad)

# ====================== 多目标跟踪器（SORT & ByteTrack） ======================
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i+4] = 1.0
        self.kf.measurementMatrix = np.zeros((4, 7), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-4
        self.kf.processNoiseCov[4:, 4:] *= 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        x1, y1, x2, y2 = bbox[:4]
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        cx = x1 + w/2.0
        cy = y1 + h/2.0
        s = w * h
        r = w / h
        self.kf.statePre = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float32)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

    def _clamp_state(self):
        self.kf.statePost[2] = max(self.kf.statePost[2], 1e-6)
        self.kf.statePost[3] = max(self.kf.statePost[3], 1e-6)
        self.kf.statePre[2] = max(self.kf.statePre[2], 1e-6)
        self.kf.statePre[3] = max(self.kf.statePre[3], 1e-6)

    def predict(self):
        if abs(self.kf.statePost[4]) + abs(self.kf.statePost[5]) + abs(self.kf.statePost[6]) < 1e-6:
            self.kf.statePost[4] = 0.01
            self.kf.statePost[5] = 0.01
        prediction = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self._clamp_state()
        return self.get_state()

    def update(self, bbox):
        x1, y1, x2, y2 = bbox[:4]
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        cx = x1 + w/2.0
        cy = y1 + h/2.0
        s = w * h
        r = w / h
        measurement = np.array([cx, cy, s, r], dtype=np.float32)
        self.kf.correct(measurement)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self._clamp_state()

    def get_state(self):
        cx, cy, s, r = self.kf.statePost[:4]
        s = max(s, 1e-6)
        r = max(r, 1e-6)
        w = np.sqrt(s * r)
        h = s / w
        x1 = cx - w/2.0
        y1 = cy - h/2.0
        x2 = cx + w/2.0
        y2 = cy + h/2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

class BaseTracker:
    def __init__(self, max_lost=30):
        self.trackers = []
        self.max_lost = max_lost
        self.frame_count = 0

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections), dtype=int), np.empty(0, dtype=int)
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=int), np.arange(len(trackers), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)

        iou_matrix = np.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        except ValueError:
            return np.empty((0, 2), dtype=int), np.arange(len(detections), dtype=int), np.arange(len(trackers), dtype=int)

        matched = np.array(list(zip(row_ind, col_ind)), dtype=int).reshape(-1, 2)
        valid_matches = [m for m in matched if iou_matrix[m[0], m[1]] >= iou_threshold]
        if valid_matches:
            matches = np.array(valid_matches).reshape(-1, 2)
        else:
            matches = np.empty((0, 2), dtype=int)

        matched_det = set(matches[:, 0]) if matches.shape[0] > 0 else set()
        matched_trk = set(matches[:, 1]) if matches.shape[0] > 0 else set()
        unmatched_dets = np.array([d for d in range(len(detections)) if d not in matched_det], dtype=int)
        unmatched_trks = np.array([t for t in range(len(trackers)) if t not in matched_trk], dtype=int)
        return matches, unmatched_dets, unmatched_trks

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(float(boxA[0]), float(boxB[0]))
        yA = max(float(boxA[1]), float(boxB[1]))
        xB = min(float(boxA[2]), float(boxB[2]))
        yB = min(float(boxA[3]), float(boxB[3]))
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return inter / (areaA + areaB - inter + 1e-6)

    def _remove_dead(self):
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_lost]

class SortTracker(BaseTracker):
    def update(self, detections):
        self.frame_count += 1
        for trk in self.trackers:
            trk.predict()
        preds = [trk.get_state() for trk in self.trackers]
        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            detections, preds, iou_threshold=0.3)
        for m in matches:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(detections[det_idx])
        for det_idx in unmatched_dets:
            if detections[det_idx][4] >= 0.5:
                trk = KalmanBoxTracker(detections[det_idx])
                self.trackers.append(trk)
        self._remove_dead()
        ret = []
        for trk in self.trackers:
            if trk.time_since_update < 1:
                box = trk.get_state()
                ret.append([*box, trk.id])
        return ret

class ByteTrackTracker(BaseTracker):
    def __init__(self, high_thresh=0.5, low_thresh=0.1, max_lost=30):
        super().__init__(max_lost)
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

    def update(self, detections):
        self.frame_count += 1
        high_dets = [d for d in detections if d[4] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d[4] < self.high_thresh]

        for trk in self.trackers:
            trk.predict()
        preds = [trk.get_state() for trk in self.trackers]

        matches1, unmatched_high, unmatched_trks = self.associate_detections_to_trackers(
            high_dets, preds, iou_threshold=0.3)
        for m in matches1:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(high_dets[det_idx])

        if len(unmatched_trks) > 0 and len(low_dets) > 0:
            unmatched_trk_preds = [preds[t] for t in unmatched_trks]
            matches2, unmatched_low, _ = self.associate_detections_to_trackers(
                low_dets, unmatched_trk_preds, iou_threshold=0.3)
            for m in matches2:
                det_idx, trk_idx = m
                original_trk_idx = unmatched_trks[trk_idx]
                self.trackers[original_trk_idx].update(low_dets[det_idx])
        else:
            unmatched_low = list(range(len(low_dets)))

        for det_idx in unmatched_high:
            trk = KalmanBoxTracker(high_dets[det_idx])
            self.trackers.append(trk)

        self._remove_dead()
        ret = []
        for trk in self.trackers:
            if trk.time_since_update < 1:
                box = trk.get_state()
                ret.append([*box, trk.id])
        return ret

# ====================== 叉车速度估计器（重构输出） ======================
class ForkliftSpeedEstimator:
    def __init__(self, model_path, real_width_m=1.5, tracker_type='sort'):
        self.detector = YOLO11(model_path, 0.60, 0.45)
        self.real_width_m = real_width_m
        if tracker_type == 'bytetrack':
            self.tracker = ByteTrackTracker(high_thresh=0.5, low_thresh=0.1)
        else:
            self.tracker = SortTracker()
        self.track_history = {}
        self.speed_history = defaultdict(list)
        self.detection_frames = defaultdict(int)
        self.valid_id_map = {}
        self.next_valid_id = 1
        self.min_frames = 50

        # 用于生成最终输出图像
        self.first_frame = None
        self.best_frame = None
        self.best_display_count = 0

    def pixel_to_meter_scale(self, box_width_pixels):
        if box_width_pixels <= 0:
            return None
        return self.real_width_m / box_width_pixels

    def estimate_speed(self, track_id, current_center, current_width, frame_idx, fps):
        if track_id not in self.track_history:
            return None
        prev = self.track_history[track_id]
        prev_center = prev["center"]
        prev_width = prev["width"]
        prev_frame = prev["frame_idx"]
        dt = (frame_idx - prev_frame) / fps
        if dt <= 0:
            return None
        dx = current_center[0] - prev_center[0]
        dy = current_center[1] - prev_center[1]
        pixel_dist = np.sqrt(dx**2 + dy**2)
        avg_width = (current_width + prev_width) / 2.0
        scale = self.pixel_to_meter_scale(avg_width)
        if scale is None:
            return None
        real_dist = pixel_dist * scale
        speed = real_dist / dt
        return speed

    def update_history(self, track_id, center, width, frame_idx):
        self.track_history[track_id] = {"center": center, "width": width, "frame_idx": frame_idx}

    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频文件")
            return None, None, False, []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps,
                                         (int(cap.get(3)), int(cap.get(4))))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 记录第一帧
            if self.first_frame is None:
                self.first_frame = frame.copy()

            # 1. YOLO11 检测
            dets = self.detector.detect(frame)
            # 2. 筛选叉车 (class_id == 0)
            forklift_dets = []
            for d in dets:
                if d[5] == 0:
                    x, y, w, h = d[:4]
                    score = d[4]
                    x1, y1, x2, y2 = x, y, x+w, y+h
                    forklift_dets.append([x1, y1, x2, y2, score])
            # 3. 更新跟踪器
            tracked_objects = self.tracker.update(forklift_dets)

            annotated = frame.copy()
            current_display_ids = set()

            for obj in tracked_objects:
                x1, y1, x2, y2, tid = obj
                w = x2 - x1
                cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                self.detection_frames[tid] += 1

                speed = self.estimate_speed(tid, (cx, cy), w, frame_idx, fps)
                self.update_history(tid, (cx, cy), w, frame_idx)
                if speed is not None:
                    self.speed_history[tid].append(speed)

                # 达到最小帧数后分配连续 ID
                if tid not in self.valid_id_map and self.detection_frames[tid] >= self.min_frames:
                    self.valid_id_map[tid] = self.next_valid_id
                    self.next_valid_id += 1

                # 绘制框（所有跟踪目标都绘制）
                if tid in self.valid_id_map:
                    display_id = self.valid_id_map[tid]
                    current_display_ids.add(display_id)
                    avg_speed = np.mean(self.speed_history[tid]) if self.speed_history[tid] else 0.0
                    speed_kmh = speed * 3.6 if speed is not None else None
                    avg_kmh = avg_speed * 3.6
                    label = f"ID:{display_id} {speed_kmh:.1f}km/h" if speed_kmh else f"ID:{display_id} --"
                    label += f" Avg:{avg_kmh:.1f}km/h"
                    color = (0, 255, 0)  # 有效目标绿色
                else:
                    label = "tracking..."
                    color = (0, 255, 255)  # 未达到阈值的用黄色

                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 更新最佳帧
            if len(current_display_ids) > self.best_display_count:
                self.best_display_count = len(current_display_ids)
                self.best_frame = annotated.copy()

            if output_path:
                out_writer.write(annotated)
            else:
                cv2.imshow("Forklift Speed Estimation", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

        cap.release()
        if output_path:
            out_writer.release()
        cv2.destroyAllWindows()

        # ====================== 生成最终返回结果 ======================
        # 确定输出图像
        if self.best_frame is not None:
            forklift_img = self.best_frame
        else:
            forklift_img = self.first_frame if self.first_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        # 提取满足 min_frames 的平均速度
        valid_speeds = []
        forklift_exist = False
        for tid, mapped_id in sorted(self.valid_id_map.items(), key=lambda x: x[1]):
            speeds = self.speed_history[tid]
            if len(speeds) >= self.min_frames:
                avg = np.mean(speeds) * 3.6
                avg = round(avg, 2)
                valid_speeds.append(avg)

        if valid_speeds:
            forklift_exist = True

        # 在输出图像上绘制状态信息
        status = "Forklift: YES" if forklift_exist else "Forklift: NO"
        cv2.putText(forklift_img, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if forklift_exist:
            for i, avg_speed in enumerate(valid_speeds):
                speed_str = f"ID {i+1}: Speed: {avg_speed:.2f} km/h"
                cv2.putText(forklift_img, speed_str, (20, 80 + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(forklift_img, "Speed: -", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        print("\n===== 检测完成 =====")
        print(f"视频中是否存在叉车: {'是' if forklift_exist else '否'}")
        print("===== 各叉车平均速度（km/h）=====")
        for i, avg in enumerate(valid_speeds):
            print(f"ID {i+1}: {avg:.2f} km/h")

        return forklift_img, None, forklift_exist, valid_speeds

# ====================== 主程序 ======================
if __name__ == "__main__":
    MODEL_PATH = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\detect\forklift_v2\weights\best.onnx"
    VIDEO_PATH = r"G:\Gosion\projects\011.ZhongyuanGuanxi\data\forklift2\plan_video_2062090081220636673.mp4"
    FORKLIFT_REAL_WIDTH_M = 1.5

    estimator = ForkliftSpeedEstimator(MODEL_PATH, FORKLIFT_REAL_WIDTH_M, tracker_type='bytetrack')
    result_img, add_diff, exist, speeds = estimator.process_video(VIDEO_PATH, output_path="output_speed.mp4")
    
    # 如需保存最终图像
    if result_img is not None:
        cv2.imwrite("final_result.jpg", result_img)