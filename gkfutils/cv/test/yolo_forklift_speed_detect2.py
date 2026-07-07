import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# -------------------- 配置参数 --------------------
VIDEO_PATH = r"G:\Gosion\projects\011.ZhongyuanGuanxi\data\forklift\forklift_videos\17751081010232038849811150745602.mp4"
# VIDEO_PATH = r"G:\Gosion\projects\011.ZhongyuanGuanxi\data\forklift2\plan_video_2061357246864109570.mp4"
MODEL_PATH = r"G:\Gosion\code\Python\ultralytics-8.3.243\runs\detect\forklift_v2\weights\best.onnx"
FORKLIFT_REAL_WIDTH_M = 1.5                 # 叉车实际宽度（米），根据实际车型调整
TRACKER_CONFIG = "bytetrack.yaml"           # 跟踪器配置文件
MIN_FRAMES = 50  # 最少检测帧数阈值，可配置为类属性或常量
# -------------------------------------------------

class ForkliftSpeedEstimator:
    """叉车检测与多目标速度估计器（显示各叉车平均速度）"""
    def __init__(self, model_path, real_width_m=1.5):
        self.model = YOLO(model_path)
        self.real_width_m = real_width_m
        # 记录每个跟踪ID上一帧的状态：{"center": (cx,cy), "frame_idx": int, "width": float}
        self.track_history = {}
        # 记录每个跟踪ID的瞬时速度历史，用于计算平均速度
        self.speed_history = defaultdict(list)

        self.detection_frames = defaultdict(int)   # 原始 ID -> 累计检测帧数
        self.valid_id_map = {}                     # 原始 ID -> 连续映射 ID
        self.next_valid_id = 1

    def pixel_to_meter_scale(self, box_width_pixels):
        """利用叉车实际宽度和检测框宽度计算像素到米的缩放因子"""
        if box_width_pixels <= 0:
            return None
        return self.real_width_m / box_width_pixels

    def estimate_speed(self, track_id, current_center, current_width, frame_idx, fps):
        """估算当前帧该叉车的瞬时速度（米/秒）"""
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

        # 使用当前帧与上一帧宽度的均值作为尺度参考，更稳定
        avg_width = (current_width + prev_width) / 2.0
        scale = self.pixel_to_meter_scale(avg_width)
        if scale is None:
            return None

        real_dist = pixel_dist * scale        # 米
        speed = real_dist / dt                # 米/秒
        return speed

    def update_history(self, track_id, center, width, frame_idx):
        """更新跟踪历史（用于下一帧速度计算）"""
        self.track_history[track_id] = {
            "center": center,
            "width": width,
            "frame_idx": frame_idx
        }

    def process_video(self, video_path, output_path=None):
        """主处理流程：跟踪叉车并显示各自的平均速度"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频文件")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25  # 默认帧率

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps,
                                         (int(cap.get(3)), int(cap.get(4))))

        # 使用YOLO的跟踪模式，stream=True返回生成器，persist=True保持ID一致
        results = self.model.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False
        )

        frame_idx = 0
        for result in results:
            frame = result.orig_img
            annotated = frame.copy()

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()

                for box, tid, cls in zip(boxes, track_ids, classes):
                    if cls != 0:
                        continue

                    # 记录该 ID 出现的帧数（不论速度是否有效）
                    self.detection_frames[tid] += 1

                    x1, y1, x2, y2 = box[:4]
                    w = x2 - x1
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    speed = self.estimate_speed(tid, (cx, cy), w, frame_idx, fps)
                    self.update_history(tid, (cx, cy), w, frame_idx)
                    if speed is not None:
                        self.speed_history[tid].append(speed)

                    # 达到检测帧数阈值后，分配连续映射 ID
                    if tid not in self.valid_id_map and self.detection_frames[tid] >= MIN_FRAMES:
                        self.valid_id_map[tid] = self.next_valid_id
                        self.next_valid_id += 1

                    # 只绘制已映射的有效目标
                    if tid not in self.valid_id_map:
                        continue

                    display_id = self.valid_id_map[tid]

                    avg_speed = np.mean(self.speed_history[tid]) if self.speed_history[tid] else 0.0

                    # 绘制检测框
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    speed_kmh = speed * 3.6 if speed is not None else None
                    avg_kmh = avg_speed * 3.6
                    label_parts = [f"ID:{display_id}"]
                    if speed_kmh is not None:
                        label_parts.append(f"{speed_kmh:.1f}km/h")
                    else:
                        label_parts.append("--")
                    label_parts.append(f"Avg:{avg_kmh:.1f}km/h")
                    label = " ".join(label_parts)

                    cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示或写入
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

        # 最终统计只输出有效映射 ID
        print("\n===== 各叉车平均速度（km/h）=====")
        valid_speeds_all = []
        for tid, mapped_id in sorted(self.valid_id_map.items(), key=lambda x: x[1]):
            speeds = self.speed_history[tid]
            if len(speeds) >= MIN_FRAMES:
                avg = np.mean(speeds) * 3.6
                valid_speeds_all.extend(speeds)
                print(f"ID:{mapped_id}  平均速度: {avg:.2f} km/h （检测帧数: {self.detection_frames[tid]}）")
        if valid_speeds_all:
            global_avg = np.mean(valid_speeds_all) * 3.6
            print(f"\n有效叉车整体平均速度: {global_avg:.2f} km/h")
        else:
            print("无满足帧数阈值的叉车")


# -------------------- 运行示例 --------------------
if __name__ == "__main__":
    estimator = ForkliftSpeedEstimator(MODEL_PATH, FORKLIFT_REAL_WIDTH_M)
    estimator.process_video(VIDEO_PATH, output_path="output_speed.mp4")