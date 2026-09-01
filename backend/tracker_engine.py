import time
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


class ObjectTrackerEngine:
    """
    Core AI Detection & Multi-Object Tracking Engine.
    Combines YOLO (v8-World / YOLO11) with DeepSORT Re-ID Tracking.
    Includes EMA coordinate smoothing, Kalman prediction grace periods,
    and ultra-fast frame downscaling for 30+ FPS latency reduction.
    """

    COLOR_PALETTE = [
        (255, 0, 255),    # Vibrant Magenta
        (0, 240, 255),    # Electric Cyan
        (0, 255, 128),    # Neon Green
        (255, 191, 0),    # Amber Gold
        (255, 64, 64),    # Coral Red
        (160, 32, 240),   # Purple
        (0, 191, 255),    # Sky Blue
        (255, 255, 0),    # Bright Yellow
        (255, 105, 180),   # Hot Pink
        (50, 205, 50),    # Lime Green
    ]

    def __init__(self, model_name="yolo11n.pt", max_age=60, n_init=1, nms_max_overlap=0.7):
        self.model_name = model_name
        self.max_age = max_age
        self.n_init = n_init
        self.nms_max_overlap = nms_max_overlap

        print(f"[TrackerEngine] Loading model '{model_name}'...")
        self.model = YOLO(model_name)

        if "world" in model_name.lower():
            print("[TrackerEngine] Initializing YOLO-World Open-Vocabulary Prompts...")
            self.custom_prompts = [
                "person", "cell phone", "phone", "smartphone", "hand", "object",
                "item", "bottle", "gadget", "card", "toy", "bag"
            ]
            if hasattr(self.model, "set_classes"):
                try:
                    self.model.set_classes(self.custom_prompts)
                except Exception as e:
                    print(f"[TrackerEngine] Prompts set error: {e}")
        else:
            print(f"[TrackerEngine] Standard YOLO model '{model_name}' loaded.")

        print(f"[TrackerEngine] Initializing Anti-Flicker DeepSORT Tracker (max_age={max_age}, n_init={n_init})...")
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            max_cosine_distance=0.3,
            nn_budget=100
        )

        self.track_history = {}      # track_id -> list of (cx, cy)
        self.prev_boxes = {}         # track_id -> smoothed (x1, y1, w, h)
        self.unique_track_ids = set()
        self.track_logs = []
        self.last_detections = []
        self.frame_count = 0
        self.alpha_smoothing = 0.65  # EMA box smoothing factor
        self.yolo_imgsz = 320        # Default 320px for ultra-fast FPS

    def set_speed_preset(self, preset="fast"):
        """Adjusts YOLO inference resolution for speed vs accuracy."""
        if preset == "fast":
            self.yolo_imgsz = 320
        elif preset == "balanced":
            self.yolo_imgsz = 416
        else:
            self.yolo_imgsz = 640

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
        conf_threshold=0.10,
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
        orig_h, orig_w = frame.shape[:2]
        output_frame = frame.copy()

        imgsz_to_use = yolo_imgsz if yolo_imgsz is not None else self.yolo_imgsz

        # Fast downscaling for YOLO inference (boosting FPS by 4x)
        if orig_w > imgsz_to_use or orig_h > imgsz_to_use:
            scale = float(imgsz_to_use) / max(orig_h, orig_w)
            infer_w, infer_h = int(orig_w * scale), int(orig_h * scale)
            infer_frame = cv2.resize(frame, (infer_w, infer_h))
        else:
            scale = 1.0
            infer_frame = frame

        # 1. YOLO Inference on downscaled frame for 15ms latency
        results = self.model(
            infer_frame,
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
                    else:
                        class_name = f"Class_{class_id}"

                    if class_filter and class_name.lower() not in [c.lower() for c in class_filter]:
                        continue

                    # Rescale bounding box back to original frame dimensions
                    infer_box = box.xywh[0].cpu().numpy()
                    cx_infer, cy_infer, w_infer, h_infer = infer_box

                    cx_orig = cx_infer / scale
                    cy_orig = cy_infer / scale
                    w_orig = w_infer / scale
                    h_orig = h_infer / scale

                    left = max(0, int(cx_orig - w_orig / 2))
                    top = max(0, int(cy_orig - h_orig / 2))
                    width = int(w_orig)
                    height = int(h_orig)

                    detections.append(([left, top, width, height], conf, class_name))
                except Exception as e:
                    print(f"[TrackerEngine] Box extraction error: {e}")

        # 2. DeepSORT Re-ID Tracking Update
        tracks = self.tracker.update_tracks(detections, frame=frame)

        active_tracks_list = []
        person_count = 0
        vehicle_count = 0

        for track in tracks:
            if not track.is_confirmed():
                continue

            # 3-frame grace period for Kalman prediction during momentary detection loss
            if track.time_since_update > 3:
                continue

            track_id = track.track_id
            self.unique_track_ids.add(track_id)
            color = self.get_track_color(track_id)

            ltrb = track.to_ltrb()
            x1_curr, y1_curr, x2_curr, y2_curr = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            w_curr = max(1, x2_curr - x1_curr)
            h_curr = max(1, y2_curr - y1_curr)

            # EMA Box Coordinate Smoothing to eliminate jitter
            if track_id in self.prev_boxes:
                px1, py1, pw, ph = self.prev_boxes[track_id]
                x1 = int(self.alpha_smoothing * x1_curr + (1 - self.alpha_smoothing) * px1)
                y1 = int(self.alpha_smoothing * y1_curr + (1 - self.alpha_smoothing) * py1)
                w_box = int(self.alpha_smoothing * w_curr + (1 - self.alpha_smoothing) * pw)
                h_box = int(self.alpha_smoothing * h_curr + (1 - self.alpha_smoothing) * ph)
            else:
                x1, y1, w_box, h_box = x1_curr, y1_curr, w_curr, h_curr

            self.prev_boxes[track_id] = (x1, y1, w_box, h_box)
            x2 = min(orig_w, x1 + w_box)
            y2 = min(orig_h, y1 + h_box)
            cx, cy = int(x1 + w_box / 2), int(y1 + h_box / 2)

            class_name = track.get_det_class()
            if not class_name or class_name == "None":
                class_name = "Object"

            conf_val = track.get_det_conf()
            conf_str = f"{int(conf_val * 100)}%" if conf_val is not None else "100%"

            if class_name.lower() in ["person", "human", "man", "woman"]:
                person_count += 1
            elif class_name.lower() in ["car", "truck", "bus", "motorcycle", "vehicle"]:
                vehicle_count += 1

            # Trajectory History Trails
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((cx, cy))
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            if show_trails and len(self.track_history[track_id]) > 1:
                pts = np.array(self.track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(output_frame, [pts], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)

            # Draw Bounding Box & Label Badges
            if show_boxes:
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            if show_labels:
                label_text = f"{class_name} #{track_id} | {conf_str}"
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                badge_y1 = max(0, y1 - th - 8)
                badge_y2 = y1
                cv2.rectangle(output_frame, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, -1)
                cv2.putText(
                    output_frame,
                    label_text,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            active_tracks_list.append({
                "track_id": track_id,
                "class": class_name,
                "confidence": conf_str,
                "center": f"({cx}, {cy})",
                "bbox": f"[{x1}, {y1}, {x2}, {y2}]"
            })

        # Calculate FPS & Latency
        inference_latency_ms = (time.time() - frame_start_time) * 1000.0
        fps = 1000.0 / max(inference_latency_ms, 1.0)

        # Draw Real-Time Translucent HUD Panel
        if show_hud:
            hud_w, hud_h = 290, 160
            hud_x1 = orig_w - hud_w - 15
            hud_y1 = 15
            hud_x2 = orig_w - 15
            hud_y2 = hud_y1 + hud_h

            hud_bg = output_frame[hud_y1:hud_y2, hud_x1:hud_x2].copy()
            dark_mask = np.zeros_like(hud_bg, dtype=np.uint8)
            hud_blend = cv2.addWeighted(hud_bg, 0.35, dark_mask, 0.65, 0)
            output_frame[hud_y1:hud_y2, hud_x1:hud_x2] = hud_blend
            cv2.rectangle(output_frame, (hud_x1, hud_y1), (hud_x2, hud_y2), (0, 240, 255), 1, cv2.LINE_AA)

            title_str = f"HUD [{self.model_name}]"
            cv2.putText(output_frame, title_str, (hud_x1 + 10, hud_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 2, cv2.LINE_AA)
            cv2.line(output_frame, (hud_x1 + 10, hud_y1 + 32), (hud_x2 - 10, hud_y1 + 32), (0, 240, 255), 1)

            stats_lines = [
                f"FPS       : {fps:.1f}",
                f"Latency   : {inference_latency_ms:.1f} ms",
                f"Active Objects : {len(active_tracks_list)}",
                f"Total Unique IDs : {len(self.unique_track_ids)}",
                f"People / Vehicles: {person_count} / {vehicle_count}",
                f"Resolution   : {orig_w}x{orig_h}"
            ]
            y_offset = hud_y1 + 52
            for line in stats_lines:
                cv2.putText(output_frame, line, (hud_x1 + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 240), 1, cv2.LINE_AA)
                y_offset += 17

        performance_stats = {
            "fps": fps,
            "latency_ms": inference_latency_ms,
            "active_tracks": len(active_tracks_list),
            "total_unique_tracks": len(self.unique_track_ids),
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "frame_width": orig_w,
            "frame_height": orig_h
        }

        return output_frame, performance_stats, active_tracks_list
