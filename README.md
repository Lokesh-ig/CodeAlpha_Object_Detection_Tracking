# 🛡️ CodeAlpha Universal Object Detection & Tracking Dashboard

A high-performance real-time Object Detection and Tracking system built using **Python**, **PyQt6**, **YOLO11**, **YOLO-World Universal**, **DeepSORT**, and **OpenCV**.

This project provides both a **Modern Desktop GUI Dashboard** (`app_gui.py`) with a **YOLO-World Universal Open-Vocabulary Engine** and a **High-Tech OpenCV HUD CLI Mode** (`main.py`) for live webcam streams, video files, and static image analysis.

---

## 🌟 Key Features

### 🌍 YOLO-World Universal Open-Vocabulary Tracker (`yolov8s-world.pt`)
- **Detect & Track Anything**: Detects and tracks **ANY object** in real-time without custom training!
- **Dynamic Text Prompts**: Easily specify custom text prompts in the GUI (e.g. `person, object, item, gadget, card, hand, toy, bottle, phone, bag, tool`).
- **AI Model Engine Switcher**: Toggle seamlessly between:
  - 🌍 `YOLO-World Universal (yolov8s-world.pt)` – Open-vocabulary universal object detection.
  - ⚡ `YOLO11 Nano (yolo11n.pt)` – Fast 80-class everyday tracker.
  - 🎯 `YOLO11 Small (yolo11s.pt)` – High-accuracy 80-class tracker.

---

### 🖥️ Modern Desktop GUI Dashboard (`app_gui.py`)
- **Maximized Detection Viewport**: Click `📐 MAXIMIZE DETECTION AREA` to hide sidebars and expand the video feed to fill **100% of the screen**.
- **Multi-Source Input Support**: Switch between **Live Webcams**, **Video Files** (`.mp4`, `.avi`, `.mkv`), or **Static Images** (`.jpg`, `.png`).
- **Real-Time Detection Tuning**: Live sliders for **Confidence Threshold** ($0.05 - 0.95$) and **IoU NMS Threshold**.
- **Visual Overlays**:
  - 🌀 **Motion Trajectory Trails**: Color-coded breadcrumb lines showing object motion paths over time.
  - 📦 **Bounding Boxes & Corner Accents**: High-contrast bounding boxes with futuristic corner targeting marks.
  - 🏷️ **Class Badges & ID Tags**: Solid badges displaying Object Class, Tracking ID, and Confidence %.
  - 📊 **Translucent HUD Overlay**: Live overlay stats panel directly on the video feed.
- **Analytics Dashboard & Active Track Log**:
  - Real-time KPI metrics cards for **FPS**, **Latency (ms)**, **Active Tracks**, **Total Unique IDs**, **Person Count**, and **Vehicle / Object Counts**.
  - Live updating table displaying active object IDs, classes, confidence scores, and center $(X,Y)$ coordinates.
- **Tools & Actions**:
  - 📸 **Snapshot Generator**: Save high-resolution snapshots to `snapshots/`.
  - 🎥 **Stream Recorder**: One-click MP4 video stream recording to `recordings/`.
  - 🔄 **Tracker Reset**: Clear tracking history and reset DeepSORT IDs.
  - 💾 **Data Export**: Export complete session tracking logs to **CSV** or **JSON** files in `exports/`.

---

### 🎮 High-Tech HUD CLI Mode (`main.py`)
A fast OpenCV window HUD interface for lightweight monitoring with full keyboard shortcuts:

| Shortcut Key | Action |
|:------------:|:-------|
| `[SPACE]` | Pause / Resume live video stream |
| `[S]` | Take high-resolution snapshot |
| `[R]` | Start / Stop video stream recording (MP4) |
| `[T]` | Toggle trajectory motion trails |
| `[H]` | Toggle translucent HUD stats panel |
| `[C]` | Reset tracker state & clear ID history |
| `[E]` | Export tracking log to CSV and JSON |
| `[Q]` / `[ESC]` | Quit application |

---

## 🛠️ System Architecture

```text
Input (Webcam / Video / Image)
           │
           ▼
 YOLO11 / YOLO-World Universal Engine
 (Text Prompts: person, object, item, gadget...)
           │
           ▼
 Confidence & Class Filtering
           │
           ▼
   DeepSORT Tracker ──► Trajectory History & ID Assignment
           │
           ▼
    Render Overlay ──► Trajectories + Boxes + Badges + HUD Panel
           │
           ├─────────────────────────┐
           ▼                         ▼
   PyQt6 GUI Window           OpenCV HUD Window
  (app_gui.py)                   (main.py)
```

---

## 🚀 Getting Started

### 1. Environment Setup & Installation
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launching the GUI Dashboard (with YOLO-World)
```bash
python app_gui.py
```

### 3. Launching the OpenCV HUD Mode
```bash
python main.py
```

---

## 📜 License & Credits

Developed for **CodeAlpha** Object Detection & Tracking.  
Powered by [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/), [DeepSORT Realtime](https://github.com/levan92/deep_sort_realtime), OpenCV, and PyQt6.