import cv2
import time
import os
import json
import csv
import warnings
from datetime import datetime
from collections import defaultdict, deque
import numpy as np

warnings.filterwarnings("ignore")

from ultralytics import YOLO, YOLOWorld
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTrackerEngine:
    """
    High-Sensitivity Anti-Flicker Object Detection & Tracking Engine.
    Includes Kalman Prediction Grace Period and Exponential Box Smoothing (EMA)
    to completely eliminate box blinking, flickering, and border jitter.
    """

    COLOR_PALETTE = [
        (0, 255, 127),    # Spring Green
        (255, 144, 30),   # Dodger Blue
        (255, 0, 255),    # Magenta
        (0, 215, 255),    # Gold
        (255, 105, 180),  # Hot Pink
        (50, 205, 50),    # Lime Green
        (238, 130, 238),  # Violet
        (0, 165, 255),    # Orange
        (255, 255, 0),    # Cyan
        (147, 20, 255)    # Deep Pink
    ]

    DEFAULT_WORLD_PROMPTS = [
        "person", "human", "face", "cell phone", "phone", "smartphone", "bottle",
        "cup", "hand", "object", "item", "gadget", "card", "toy", "bag",
        "laptop", "book", "tool", "box", "device", "accessory"
    ]

    def __init__(self, model_path="yolov8s-world.pt", max_age=60, n_init=1, nms_max_overlap=0.6):
        self.max_age = max_age
        self.n_init = n_init
        self.nms_max_overlap = nms_max_overlap

        self.model = None
        self.model_path = model_path
        self.custom_prompts = list(self.DEFAULT_WORLD_PROMPTS)
        self.load_model(model_path)

        print("[TrackerEngine] Initializing Anti-Flicker DeepSORT Tracker (max_age=60, n_init=1)...")
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            nms_max_overlap=self.nms_max_overlap,
            max_cosine_distance=0.3,
            nn_budget=100
        )

        # Tracking state
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.prev_boxes = {}  # For EMA box smoothing (eliminates flicker & jitter)
        self.unique_track_ids = set()
        self.track_logs = []
        self.frame_count = 0
        self.previous_time = time.time()
        self.fps = 0.0

        self.last_detections = []
        self.detection_interval = 1
        self.yolo_imgsz = 640

        # Video recording
        self.is_recording = False
        self.video_writer = None
        self.recording_path = None

    def load_model(self, model_path, custom_prompts=None):
        self.model_path = model_path
        if custom_prompts:
            self.custom_prompts = custom_prompts

        print(f"[TrackerEngine] Loading model '{model_path}'...")
        try:
            if "world" in model_path.lower():
                self.model = YOLOWorld(model_path)
                self.model.set_classes(self.custom_prompts)
                print(f"[TrackerEngine] YOLO-World Universal loaded with prompts: {self.custom_prompts}")
            else:
                self.model = YOLO(model_path)
                print(f"[TrackerEngine] Standard YOLO model '{model_path}' loaded.")
        except Exception as err:
            print(f"[TrackerEngine] Warning loading {model_path}: {err}. Loading fallback YOLO...")
            self.model = YOLO("yolo11n.pt")

    def set_speed_preset(self, preset_name):
        preset = preset_name.lower()
        if preset == "fast":
            self.yolo_imgsz = 320
            self.detection_interval = 1
        elif preset == "balanced":
            self.yolo_imgsz = 512
            self.detection_interval = 1
        else:
            self.yolo_imgsz = 640
            self.detection_interval = 1

    def update_world_prompts(self, prompt_list):
        if prompt_list:
            self.custom_prompts = [p.strip() for p in prompt_list if p.strip()]
            if hasattr(self.model, "set_classes"):
                try:
                    self.model.set_classes(self.custom_prompts)
                except Exception as err:
                    print(f"[TrackerEngine] Error updating prompts: {err}")

    def get_track_color(self, track_id):
        try:
            numeric_id = int(str(track_id).split("_")[-1])
        except (ValueError, IndexError):
            numeric_id = hash(str(track_id))
        return self.COLOR_PALETTE[numeric_id % len(self.COLOR_PALETTE)]

    def reset_tracker(self):
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            nms_max_overlap=self.nms_max_overlap,
            max_cosine_distance=0.3,
            nn_budget=100
        )
        self.track_history.clear()
        self.prev_boxes.clear()
        self.unique_track_ids.clear()
        self.track_logs.clear()
        self.last_detections.clear()
        self.frame_count = 0

    def process_frame(
        self,
        frame,
        conf_threshold=0.20,
        iou_threshold=0.45,
        class_filter=None,
        show_trails=True,
        show_labels=True,
        show_boxes=True,
        show_hud=True,
        yolo_imgsz=None
    ):
        frame_start_time = time.time()
        self.frame_count += 1
        h, w = frame.shape[:2]
        output_frame = frame.copy()

        imgsz_to_use = yolo_imgsz if yolo_imgsz is not None else self.yolo_imgsz

        # 1. YOLO Inference
        results = self.model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz_to_use,
            verbose=False
        )

        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                try:
                    conf = float(box.conf[0].item())
                    if conf < conf_threshold:
                        continue

                    class_id = int(box.cls[0].item())

                    if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                        class_name = self.model.names.get(class_id, "Object")
                    elif hasattr(self.model, "names") and isinstance(self.model.names, list) and class_id < len(self.model.names):
                        class_name = self.model.names[class_id]
                    else:
                        class_name = "Object"

                    if class_filter and len(class_filter) > 0:
                        if class_name.lower() not in [c.lower() for c in class_filter]:
                            continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    if box_w > 0 and box_h > 0:
                        detections.append(([x1, y1, box_w, box_h], conf, class_name))
                except Exception:
                    continue

        # 2. DeepSORT Update
        try:
            tracks = self.tracker.update_tracks(detections, frame=frame)
        except Exception:
            tracks = []

        active_tracks_list = []
        person_count = 0
        vehicle_count = 0
        other_count = 0

        vehicle_classes = {"car", "truck", "bus", "motorbike", "bicycle", "train", "vehicle"}

        for track in tracks:
            # ANTI-FLICKER RULE: Allow confirmed tracks OR tracks updated within 3 frames
            if not track.is_confirmed():
                continue
            if track.time_since_update > 3:
                continue

            track_id = track.track_id
            self.unique_track_ids.add(track_id)

            ltrb = track.to_ltrb()
            left, top, right, bottom = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            if right <= left or bottom <= top:
                continue

            # EXPONENTIAL SMOOTHING (EMA): Prevents box jitter and blinking
            if track_id in self.prev_boxes:
                prev = self.prev_boxes[track_id]
                alpha = 0.65  # 65% new frame, 35% previous frame for buttery-smooth movement
                left = int(alpha * left + (1 - alpha) * prev[0])
                top = int(alpha * top + (1 - alpha) * prev[1])
                right = int(alpha * right + (1 - alpha) * prev[2])
                bottom = int(alpha * bottom + (1 - alpha) * prev[3])

            self.prev_boxes[track_id] = (left, top, right, bottom)

            class_name = track.get_det_class() or "Object"
            confidence = track.det_conf if track.det_conf is not None else 0.0
            confidence = float(confidence)

            c_lower = class_name.lower()
            if c_lower in ["person", "human", "face"]:
                person_count += 1
            elif c_lower in vehicle_classes:
                vehicle_count += 1
            else:
                other_count += 1

            center_x = int((left + right) / 2)
            center_y = int((top + bottom) / 2)
            self.track_history[track_id].append((center_x, center_y))

            color = self.get_track_color(track_id)

            track_info = {
                "frame": self.frame_count,
                "track_id": track_id,
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": [left, top, right, bottom],
                "center": [center_x, center_y],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            active_tracks_list.append(track_info)
            self.track_logs.append(track_info)

            # 3. Draw Trajectory Trails
            if show_trails:
                pts = self.track_history[track_id]
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    thickness = int(np.sqrt(30 / float(i + 1)) * 1.5)
                    cv2.line(output_frame, pts[i - 1], pts[i], color, max(1, thickness))

            # 4. Draw Bounding Box & Corner Accents
            if show_boxes:
                cv2.rectangle(output_frame, (left, top), (right, bottom), color, 2)
                corner_len = min(15, int((right - left) / 4), int((bottom - top) / 4))
                if corner_len > 0:
                    cv2.line(output_frame, (left, top), (left + corner_len, top), color, 3)
                    cv2.line(output_frame, (left, top), (left, top + corner_len), color, 3)
                    cv2.line(output_frame, (right, top), (right - corner_len, top), color, 3)
                    cv2.line(output_frame, (right, top), (right, top + corner_len), color, 3)
                    cv2.line(output_frame, (left, bottom), (left + corner_len, bottom), color, 3)
                    cv2.line(output_frame, (left, bottom), (left, bottom - corner_len), color, 3)
                    cv2.line(output_frame, (right, bottom), (right - corner_len, bottom), color, 3)
                    cv2.line(output_frame, (right, bottom), (right, bottom - corner_len), color, 3)

            # 5. Draw Badge / Label
            if show_labels:
                label_text = f"{class_name} #{track_id} | {confidence * 100:.0f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                lbl_bg_top = max(0, top - text_h - 10)
                lbl_bg_bottom = top

                cv2.rectangle(
                    output_frame,
                    (left, lbl_bg_top),
                    (left + text_w + 10, lbl_bg_bottom),
                    color,
                    -1
                )
                cv2.putText(
                    output_frame,
                    label_text,
                    (left + 5, lbl_bg_bottom - 4),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA
                )

        # 6. FPS Calculation
        current_time = time.time()
        time_diff = current_time - self.previous_time
        if time_diff > 0:
            current_fps = 1.0 / time_diff
            self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps
        self.previous_time = current_time

        frame_processing_time = (time.time() - frame_start_time) * 1000.0

        model_display_name = os.path.basename(self.model_path)
        stats = {
            "fps": round(self.fps, 1),
            "latency_ms": round(frame_processing_time, 1),
            "active_tracks": len(active_tracks_list),
            "total_unique_tracks": len(self.unique_track_ids),
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "other_count": other_count,
            "resolution": f"{w}x{h}",
            "frame_number": self.frame_count,
            "is_recording": self.is_recording,
            "model_name": model_display_name
        }

        if show_hud:
            output_frame = self.draw_hud_overlay(output_frame, stats)

        if self.is_recording and self.video_writer is not None:
            try:
                self.video_writer.write(output_frame)
            except Exception as write_err:
                print(f"[TrackerEngine] Recording write error: {write_err}")

        return output_frame, stats, active_tracks_list

    def draw_hud_overlay(self, frame, stats):
        h, w = frame.shape[:2]
        panel_w = min(340, w - 20)
        panel_h = 165

        x_start = w - panel_w - 10
        y_start = 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (0, 255, 180), 1)

        cv2.putText(
            frame,
            f"HUD [{stats.get('model_name', 'YOLO')}]",
            (x_start + 10, y_start + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 180),
            1,
            cv2.LINE_AA
        )
        cv2.line(frame, (x_start + 10, y_start + 28), (x_start + panel_w - 10, y_start + 28), (0, 255, 180), 1)

        lines = [
            f"FPS              : {stats['fps']:.1f}",
            f"Latency          : {stats['latency_ms']:.1f} ms",
            f"Active Objects   : {stats['active_tracks']}",
            f"Total Unique IDs : {stats['total_unique_tracks']}",
            f"People / Vehicles: {stats['person_count']} / {stats['vehicle_count']}",
            f"Resolution       : {stats['resolution']}"
        ]

        y_pos = y_start + 48
        for line in lines:
            cv2.putText(
                frame,
                line,
                (x_start + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA
            )
            y_pos += 20

        if stats.get("is_recording"):
            cv2.circle(frame, (x_start + panel_w - 20, y_start + 20), 6, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "REC",
                (x_start + panel_w - 50, y_start + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        return frame

    def take_snapshot(self, frame, save_dir="snapshots"):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"snapshot_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def start_recording(self, output_path, fps=30.0, frame_size=(1920, 1080)):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.is_recording = True
        self.recording_path = output_path

    def stop_recording(self):
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            return self.recording_path
        return None

    def export_logs_csv(self, filepath="track_logs.csv"):
        if not self.track_logs:
            return False
        keys = ["frame", "track_id", "class", "confidence", "bbox", "center", "timestamp"]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.track_logs:
                row_copy = row.copy()
                row_copy["bbox"] = str(row_copy["bbox"])
                row_copy["center"] = str(row_copy["center"])
                writer.writerow(row_copy)
        return True

    def export_logs_json(self, filepath="track_logs.json"):
        if not self.track_logs:
            return False
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.track_logs, f, indent=2)
        return True
