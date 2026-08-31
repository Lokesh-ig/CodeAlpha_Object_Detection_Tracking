import sys
import os
import cv2
import time
import json
from datetime import datetime

# Adjust Python path to find backend module if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QComboBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QFrame, QHeaderView,
    QGroupBox, QSizePolicy, QToolBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont

try:
    from backend.tracker_engine import ObjectTrackerEngine
except ImportError:
    from tracker_engine import ObjectTrackerEngine


# ==============================================================================
# VIDEO THREAD FOR ASYNCHRONOUS FRAME PROCESSING
# ==============================================================================

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object, dict, list)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.running = False
        self.paused = False
        self.source_type = "webcam"  # webcam, video_file, image
        self.source_path = 0
        self.cap = None

        # Settings
        self.conf_threshold = 0.50
        self.iou_threshold = 0.45
        self.class_filter = []
        self.show_trails = True
        self.show_labels = True
        self.show_boxes = True
        self.show_hud = True

    def set_source(self, source_type, source_path):
        self.source_type = source_type
        self.source_path = source_path
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_settings(self, conf, iou, class_filter, trails, labels, boxes, hud):
        self.conf_threshold = conf
        self.iou_threshold = iou
        self.class_filter = class_filter
        self.show_trails = trails
        self.show_labels = labels
        self.show_boxes = boxes
        self.show_hud = hud

    def run(self):
        self.running = True

        if self.source_type == "image":
            frame = cv2.imread(str(self.source_path))
            if frame is not None:
                annotated, stats, active_tracks = self.engine.process_frame(
                    frame,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                    class_filter=self.class_filter,
                    show_trails=self.show_trails,
                    show_labels=self.show_labels,
                    show_boxes=self.show_boxes,
                    show_hud=self.show_hud
                )
                self.change_pixmap_signal.emit(annotated, stats, active_tracks)
            self.running = False
            return

        # Webcam or Video File
        self.cap = cv2.VideoCapture(self.source_path)
        if self.source_type == "webcam":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        while self.running:
            if self.paused:
                self.msleep(50)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.source_type == "video_file":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.msleep(30)
                    continue

            if self.source_type == "webcam":
                frame = cv2.flip(frame, 1)

            annotated_frame, stats, active_tracks = self.engine.process_frame(
                frame,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                class_filter=self.class_filter,
                show_trails=self.show_trails,
                show_labels=self.show_labels,
                show_boxes=self.show_boxes,
                show_hud=self.show_hud
            )

            self.change_pixmap_signal.emit(annotated_frame, stats, active_tracks)
            self.msleep(10)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.running = False
        self.wait()


# ==============================================================================
# MAIN GUI WINDOW (EXPANDED DETECTION VIEWPORT MODE)
# ==============================================================================

class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CodeAlpha - Real-Time Object Detection & Tracking Dashboard")
        self.resize(1600, 950)

        # Model weights path resolution
        model_p = "yolo11n.pt"
        if not os.path.exists(model_p) and os.path.exists(os.path.join(parent_dir, "yolo11n.pt")):
            model_p = os.path.join(parent_dir, "yolo11n.pt")
        elif not os.path.exists(model_p) and os.path.exists(os.path.join("backend", "yolo11n.pt")):
            model_p = os.path.join("backend", "yolo11n.pt")

        self.engine = ObjectTrackerEngine(model_p)
        self.thread = VideoThread(self.engine)
        self.thread.change_pixmap_signal.connect(self.update_image_and_stats)

        # UI state
        self.current_frame = None
        self.sidebars_visible = True
        self.stretch_mode = False  # False: Keep Aspect, True: Fill Canvas

        self.apply_dark_theme()
        self.init_ui()
        self.start_stream()

    def apply_dark_theme(self):
        """Applies high-tech dark mode theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0c0e17;
            }
            QWidget {
                background-color: #0c0e17;
                color: #e0e6ed;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #1f263d;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
                color: #00f0ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }
            QPushButton {
                background-color: #161b2e;
                color: #ffffff;
                border: 1px solid #28314e;
                border-radius: 5px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #212946;
                border-color: #00f0ff;
            }
            QPushButton:pressed {
                background-color: #00f0ff;
                color: #000000;
            }
            QPushButton#accentButton {
                background-color: #00d2ff;
                color: #050b14;
                border: none;
                font-weight: bold;
            }
            QPushButton#accentButton:hover {
                background-color: #00f0ff;
            }
            QPushButton#recordButton {
                background-color: #e63946;
                color: #ffffff;
                border: none;
            }
            QPushButton#recordButton:hover {
                background-color: #ff4d5a;
            }
            QPushButton#maxViewportButton {
                background-color: #102a45;
                color: #00f0ff;
                border: 1px solid #00f0ff;
                font-size: 13px;
                font-weight: bold;
                padding: 6px 14px;
                border-radius: 6px;
            }
            QPushButton#maxViewportButton:hover {
                background-color: #00f0ff;
                color: #0c0e17;
            }
            QComboBox {
                background-color: #161b2e;
                border: 1px solid #28314e;
                border-radius: 5px;
                padding: 4px 8px;
                color: #ffffff;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #161b2e;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #00f0ff;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #00f0ff;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
            QCheckBox {
                spacing: 6px;
                color: #c0ccda;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #28314e;
                background-color: #161b2e;
            }
            QCheckBox::indicator:checked {
                background-color: #00f0ff;
                border-color: #00f0ff;
            }
            QTableWidget {
                background-color: #101422;
                border: 1px solid #1f263d;
                border-radius: 5px;
                gridline-color: #1a2136;
            }
            QHeaderView::section {
                background-color: #161b2e;
                color: #00f0ff;
                padding: 4px;
                border: none;
                font-weight: bold;
            }
            QLabel#kpiValue {
                font-size: 18px;
                font-weight: bold;
                color: #00f0ff;
            }
            QLabel#kpiTitle {
                font-size: 10px;
                color: #8898aa;
                text-transform: uppercase;
            }
            QFrame#kpiCard {
                background-color: #121626;
                border: 1px solid #1c2338;
                border-radius: 6px;
            }
        """)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # ----------------------------------------------------------------------
        # LEFT CONTROL SIDEBAR (COMPACT 260px)
        # ----------------------------------------------------------------------
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)
        self.left_panel.setFixedWidth(260)

        header = QLabel("⚡ CODEALPHA TRACKER")
        header.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        header.setStyleSheet("color: #00f0ff;")
        left_layout.addWidget(header)

        # 1. Source Controls
        source_group = QGroupBox("INPUT SOURCE")
        sg_layout = QVBoxLayout(source_group)
        sg_layout.setContentsMargins(6, 6, 6, 6)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam (Camera 0)", "Webcam (Camera 1)", "Video File...", "Static Image..."])
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        sg_layout.addWidget(self.source_combo)
        left_layout.addWidget(source_group)

        # 2. Detection Tuning
        tune_group = QGroupBox("DETECTION TUNING")
        tg_layout = QVBoxLayout(tune_group)
        tg_layout.setContentsMargins(6, 6, 6, 6)

        self.conf_label = QLabel("Confidence: 0.50")
        tg_layout.addWidget(self.conf_label)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.on_settings_changed)
        tg_layout.addWidget(self.conf_slider)

        self.iou_label = QLabel("IoU Threshold: 0.45")
        tg_layout.addWidget(self.iou_label)
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(5, 95)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self.on_settings_changed)
        tg_layout.addWidget(self.iou_slider)

        tg_layout.addWidget(QLabel("Class Filter:"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(["All Classes", "person", "car", "cell phone", "bottle", "cup", "chair", "laptop"])
        self.class_combo.currentIndexChanged.connect(self.on_settings_changed)
        tg_layout.addWidget(self.class_combo)

        left_layout.addWidget(tune_group)

        # 3. Visual Toggles
        vis_group = QGroupBox("VISUAL OVERLAYS")
        vg_layout = QVBoxLayout(vis_group)
        vg_layout.setContentsMargins(6, 6, 6, 6)

        self.chk_trails = QCheckBox("Trajectory Trails 🌀")
        self.chk_trails.setChecked(True)
        self.chk_trails.stateChanged.connect(self.on_settings_changed)
        vg_layout.addWidget(self.chk_trails)

        self.chk_boxes = QCheckBox("Bounding Boxes 📦")
        self.chk_boxes.setChecked(True)
        self.chk_boxes.stateChanged.connect(self.on_settings_changed)
        vg_layout.addWidget(self.chk_boxes)

        self.chk_labels = QCheckBox("Class Badges 🏷️")
        self.chk_labels.setChecked(True)
        self.chk_labels.stateChanged.connect(self.on_settings_changed)
        vg_layout.addWidget(self.chk_labels)

        self.chk_hud = QCheckBox("HUD Stats Panel 📊")
        self.chk_hud.setChecked(True)
        self.chk_hud.stateChanged.connect(self.on_settings_changed)
        vg_layout.addWidget(self.chk_hud)

        left_layout.addWidget(vis_group)

        # 4. Action Buttons
        act_group = QGroupBox("ACTIONS")
        ag_layout = QVBoxLayout(act_group)
        ag_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_toggle_play = QPushButton("⏸️ Pause Stream")
        self.btn_toggle_play.setObjectName("accentButton")
        self.btn_toggle_play.clicked.connect(self.toggle_pause)
        ag_layout.addWidget(self.btn_toggle_play)

        self.btn_snapshot = QPushButton("📸 Take Snapshot")
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        ag_layout.addWidget(self.btn_snapshot)

        self.btn_record = QPushButton("🎥 Start Video Record")
        self.btn_record.setObjectName("recordButton")
        self.btn_record.clicked.connect(self.toggle_recording)
        ag_layout.addWidget(self.btn_record)

        self.btn_reset = QPushButton("🔄 Reset Tracker")
        self.btn_reset.clicked.connect(self.reset_tracker)
        ag_layout.addWidget(self.btn_reset)

        self.btn_export = QPushButton("💾 Export Logs")
        self.btn_export.clicked.connect(self.export_logs)
        ag_layout.addWidget(self.btn_export)

        left_layout.addWidget(act_group)
        left_layout.addStretch()

        # ----------------------------------------------------------------------
        # CENTER LARGE DETECTION VIEWPORT (EXPANDING)
        # ----------------------------------------------------------------------
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)

        # Viewport Header Toolbar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(4, 2, 4, 2)

        self.status_bar_label = QLabel("Stream Active: Live Webcam (Camera 0)")
        self.status_bar_label.setStyleSheet("color: #00f0ff; font-weight: bold;")
        top_bar.addWidget(self.status_bar_label)
        top_bar.addStretch()

        # Maximize Viewport Button
        self.btn_max_viewport = QPushButton("📐 MAXIMIZE DETECTION AREA")
        self.btn_max_viewport.setObjectName("maxViewportButton")
        self.btn_max_viewport.clicked.connect(self.toggle_max_viewport)
        top_bar.addWidget(self.btn_max_viewport)

        # Canvas Scaling Mode Button
        self.btn_fit_mode = QPushButton("🔍 Fit: Aspect Ratio")
        self.btn_fit_mode.clicked.connect(self.toggle_fit_mode)
        top_bar.addWidget(self.btn_fit_mode)

        center_layout.addLayout(top_bar)

        # Video Canvas - Expands to maximum window size
        self.video_canvas = QLabel("Initializing Large Detection Viewport...")
        self.video_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_canvas.setStyleSheet("""
            QLabel {
                background-color: #04060d;
                border: 2px solid #00f0ff;
                border-radius: 8px;
            }
        """)
        self.video_canvas.setMinimumSize(640, 480)
        center_layout.addWidget(self.video_canvas, 1)

        # ----------------------------------------------------------------------
        # RIGHT ANALYTICS SIDEBAR (COMPACT 270px)
        # ----------------------------------------------------------------------
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)
        self.right_panel.setFixedWidth(270)

        right_header = QLabel("📊 ANALYTICS DASHBOARD")
        right_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        right_header.setStyleSheet("color: #00f0ff;")
        right_layout.addWidget(right_header)

        # KPI Grid Layout
        kpi_widget = QWidget()
        kpi_grid = QVBoxLayout(kpi_widget)
        kpi_grid.setContentsMargins(0, 0, 0, 0)
        kpi_grid.setSpacing(4)

        r1_layout = QHBoxLayout()
        self.card_fps, self.lbl_fps_val = self.create_kpi_card("FPS", "0.0")
        self.card_lat, self.lbl_lat_val = self.create_kpi_card("LATENCY", "0.0 ms")
        r1_layout.addWidget(self.card_fps)
        r1_layout.addWidget(self.card_lat)
        kpi_grid.addLayout(r1_layout)

        r2_layout = QHBoxLayout()
        self.card_active, self.lbl_active_val = self.create_kpi_card("ACTIVE OBJECTS", "0")
        self.card_unique, self.lbl_unique_val = self.create_kpi_card("TOTAL UNIQUE IDs", "0")
        r2_layout.addWidget(self.card_active)
        r2_layout.addWidget(self.card_unique)
        kpi_grid.addLayout(r2_layout)

        r3_layout = QHBoxLayout()
        self.card_people, self.lbl_people_val = self.create_kpi_card("PERSON COUNT", "0")
        self.card_vehicles, self.lbl_vehicles_val = self.create_kpi_card("VEHICLES / OTHER", "0")
        r3_layout.addWidget(self.card_people)
        r3_layout.addWidget(self.card_vehicles)
        kpi_grid.addLayout(r3_layout)

        right_layout.addWidget(kpi_widget)

        right_layout.addWidget(QLabel("LIVE TRACK LOG:"))
        self.track_table = QTableWidget()
        self.track_table.setColumnCount(4)
        self.track_table.setHorizontalHeaderLabels(["ID", "Class", "Conf", "Center"])
        self.track_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.track_table.verticalHeader().setVisible(False)
        right_layout.addWidget(self.track_table, 1)

        # Assemble Main Layout
        main_layout.addWidget(self.left_panel)
        main_layout.addWidget(center_panel, 1)
        main_layout.addWidget(self.right_panel)

    def create_kpi_card(self, title, default_val):
        card = QFrame()
        card.setObjectName("kpiCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)

        lbl_title = QLabel(title)
        lbl_title.setObjectName("kpiTitle")
        layout.addWidget(lbl_title)

        lbl_val = QLabel(default_val)
        lbl_val.setObjectName("kpiValue")
        layout.addWidget(lbl_val)

        return card, lbl_val

    # --------------------------------------------------------------------------
    # CONTROLS & TOGGLES
    # --------------------------------------------------------------------------

    def toggle_max_viewport(self):
        """Toggles sidebars on/off so video detection canvas takes 100% viewport space."""
        self.sidebars_visible = not self.sidebars_visible
        if self.sidebars_visible:
            self.left_panel.show()
            self.right_panel.show()
            self.btn_max_viewport.setText("📐 MAXIMIZE DETECTION AREA")
            self.btn_max_viewport.setStyleSheet("""
                background-color: #102a45; color: #00f0ff; border: 1px solid #00f0ff;
            """)
        else:
            self.left_panel.hide()
            self.right_panel.hide()
            self.btn_max_viewport.setText("🔙 SHOW CONTROLS & ANALYTICS")
            self.btn_max_viewport.setStyleSheet("""
                background-color: #00f0ff; color: #000000; font-weight: bold;
            """)

    def toggle_fit_mode(self):
        """Toggles between Keep Aspect Ratio and Fill / Stretch Canvas."""
        self.stretch_mode = not self.stretch_mode
        if self.stretch_mode:
            self.btn_fit_mode.setText("🔍 Fill: Entire Viewport")
        else:
            self.btn_fit_mode.setText("🔍 Fit: Aspect Ratio")

    def start_stream(self):
        self.on_settings_changed()
        if not self.thread.isRunning():
            self.thread.start()

    def on_source_changed(self):
        idx = self.source_combo.currentIndex()
        if idx == 0:
            self.thread.set_source("webcam", 0)
            self.status_bar_label.setText("Stream Active: Live Webcam (Camera 0)")
            self.start_stream()
        elif idx == 1:
            self.thread.set_source("webcam", 1)
            self.status_bar_label.setText("Stream Active: Live Webcam (Camera 1)")
            self.start_stream()
        elif idx == 2:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
            if file_path:
                self.thread.set_source("video_file", file_path)
                self.status_bar_label.setText(f"File Active: {os.path.basename(file_path)}")
                self.start_stream()
            else:
                self.source_combo.setCurrentIndex(0)
        elif idx == 3:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.jpg *.png *.jpeg *.bmp)")
            if file_path:
                self.thread.set_source("image", file_path)
                self.status_bar_label.setText(f"Image Active: {os.path.basename(file_path)}")
                self.start_stream()
            else:
                self.source_combo.setCurrentIndex(0)

    def on_settings_changed(self):
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0
        self.conf_label.setText(f"Confidence: {conf:.2f}")
        self.iou_label.setText(f"IoU Threshold: {iou:.2f}")

        class_sel = self.class_combo.currentText()
        class_filter = [] if class_sel == "All Classes" else [class_sel]

        self.thread.update_settings(
            conf=conf,
            iou=iou,
            class_filter=class_filter,
            trails=self.chk_trails.isChecked(),
            labels=self.chk_labels.isChecked(),
            boxes=self.chk_boxes.isChecked(),
            hud=self.chk_hud.isChecked()
        )

    def toggle_pause(self):
        if self.thread.paused:
            self.thread.paused = False
            self.btn_toggle_play.setText("⏸️ Pause Stream")
        else:
            self.thread.paused = True
            self.btn_toggle_play.setText("▶️ Resume Stream")

    def reset_tracker(self):
        self.engine.reset_tracker()
        self.track_table.setRowCount(0)
        QMessageBox.information(self, "Reset Tracker", "Tracker state & object history cleared.")

    def take_snapshot(self):
        if self.current_frame is not None:
            saved_path = self.engine.take_snapshot(self.current_frame)
            QMessageBox.information(self, "Snapshot Saved", f"Snapshot saved to:\n{saved_path}")

    def toggle_recording(self):
        if not self.engine.is_recording:
            os.makedirs("recordings", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = os.path.join("recordings", f"recording_{timestamp}.mp4")
            h, w = self.current_frame.shape[:2] if self.current_frame is not None else (1080, 1920)
            self.engine.start_recording(out_file, fps=25.0, frame_size=(w, h))
            self.btn_record.setText("⏹️ Stop Recording")
            self.btn_record.setStyleSheet("background-color: #2d3555; color: #ff4d5a; font-weight: bold;")
        else:
            rec_path = self.engine.stop_recording()
            self.btn_record.setText("🎥 Start Video Record")
            self.btn_record.setStyleSheet("")
            QMessageBox.information(self, "Recording Saved", f"Video stream saved to:\n{rec_path}")

    def export_logs(self):
        msg = QMessageBox()
        msg.setWindowTitle("Export Track Logs")
        msg.setText("Select log export format:")
        btn_csv = msg.addButton("Export CSV", QMessageBox.ButtonRole.AcceptRole)
        btn_json = msg.addButton("Export JSON", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)

        msg.exec()

        if msg.clickedButton() == btn_csv:
            os.makedirs("exports", exist_ok=True)
            path = os.path.join("exports", f"track_log_{int(time.time())}.csv")
            if self.engine.export_logs_csv(path):
                QMessageBox.information(self, "Export Success", f"Logs exported to CSV:\n{path}")
        elif msg.clickedButton() == btn_json:
            os.makedirs("exports", exist_ok=True)
            path = os.path.join("exports", f"track_log_{int(time.time())}.json")
            if self.engine.export_logs_json(path):
                QMessageBox.information(self, "Export Success", f"Logs exported to JSON:\n{path}")

    def update_image_and_stats(self, annotated_frame, stats, active_tracks):
        self.current_frame = annotated_frame.copy()

        # 1. Canvas Image Conversion
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale pixmap based on selected mode
        aspect_flag = Qt.AspectRatioMode.IgnoreAspectRatio if self.stretch_mode else Qt.AspectRatioMode.KeepAspectRatio
        scaled_pixmap = pixmap.scaled(
            self.video_canvas.size(),
            aspect_flag,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_canvas.setPixmap(scaled_pixmap)

        # 2. Update KPI Stats
        self.lbl_fps_val.setText(f"{stats['fps']:.1f}")
        self.lbl_lat_val.setText(f"{stats['latency_ms']:.1f} ms")
        self.lbl_active_val.setText(str(stats['active_tracks']))
        self.lbl_unique_val.setText(str(stats['total_unique_tracks']))
        self.lbl_people_val.setText(str(stats['person_count']))
        self.lbl_vehicles_val.setText(str(stats['vehicle_count'] + stats['other_count']))

        # 3. Update Active Track Table
        self.track_table.setRowCount(len(active_tracks))
        for row_idx, track_data in enumerate(active_tracks):
            self.track_table.setItem(row_idx, 0, QTableWidgetItem(f"#{track_data['track_id']}"))
            self.track_table.setItem(row_idx, 1, QTableWidgetItem(str(track_data['class'])))
            self.track_table.setItem(row_idx, 2, QTableWidgetItem(f"{track_data['confidence']*100:.0f}%"))
            c_str = f"({track_data['center'][0]},{track_data['center'][1]})"
            self.track_table.setItem(row_idx, 3, QTableWidgetItem(c_str))

    def closeEvent(self, event):
        self.thread.stop()
        if self.engine.is_recording:
            self.engine.stop_recording()
        event.accept()


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DashboardWindow()
    window.showMaximized()  # Open maximized for largest viewport area
    sys.exit(app.exec())
