# 🛡️ CodeAlpha Universal Object Detection & Tracking System

A high-performance real-time Object Detection and Tracking system built using **Python**, **Streamlit**, **PyQt6**, **YOLO11**, **YOLO-World Universal**, **DeepSORT**, and **OpenCV**.

This project provides 3 execution modes:
1. **🌐 Web App Dashboard (`app_web.py`)**: Browser-accessible UI built with Streamlit.
2. **🖥️ Modern Desktop GUI Dashboard (`app_gui.py`)**: High-performance desktop application built with PyQt6.
3. **🎮 High-Tech OpenCV HUD CLI Mode (`main.py`)**: Fast OpenCV window with keyboard shortcuts.

---

## 🌟 Key Features

### 🌐 Streamlit Web App Dashboard (`app_web.py`)
- **Browser-Accessible**: Runs in any modern web browser (Chrome, Edge, Safari, Firefox).
- **Interactive Multi-Source Inputs**: Stream live webcam, upload video files (`.mp4`, `.avi`), or process static images (`.jpg`, `.png`).
- **Live Analytics Metrics**: Real-time KPI cards for FPS, Latency (ms), Active Tracks, Total Unique IDs, and Person Count.
- **YOLO-World Universal Prompt Tuning**: Type custom open-vocabulary prompts on the fly.
- **Live Active Track Data Table**: Real-time pandas DataFrame showing track IDs, classes, confidence scores, and center coordinates.

---

### 🌍 YOLO-World Universal Open-Vocabulary Engine (`yolov8s-world.pt`)
- **Detect & Track Anything**: Detects and tracks **ANY object** in real-time without custom training!
- **Dynamic Text Prompts**: Easily specify custom text prompts in the UI (e.g. `person, object, item, gadget, card, hand, toy, bottle, phone, bag, tool`).
- **AI Model Engine Switcher**: Toggle seamlessly between:
  - 🌍 `YOLO-World Universal (yolov8s-world.pt)` – Open-vocabulary universal object detection.
  - ⚡ `YOLO11 Nano (yolo11n.pt)` – Fast 80-class everyday tracker.
  - 🎯 `YOLO11 Small (yolo11s.pt)` – High-accuracy 80-class tracker.

---

## 🛠️ System Architecture

```text
Input (Webcam / Video File / Image)
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
           ├─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   Streamlit Web App         PyQt6 Desktop App         OpenCV HUD Window
     (app_web.py)               (app_gui.py)               (main.py)
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

### 2. Launching the Web App Dashboard 🌐
```bash
streamlit run app_web.py
```

### 3. Launching the Desktop GUI Application 🖥️
```bash
python app_gui.py
```

### 4. Launching the OpenCV HUD CLI Mode 🎮
```bash
python main.py
```

---

## 📜 License & Credits

Developed for **CodeAlpha** Object Detection & Tracking.  
Powered by [Streamlit](https://streamlit.io/), [Ultralytics YOLO-World](https://docs.ultralytics.com/models/yolo-world/), [DeepSORT Realtime](https://github.com/levan92/deep_sort_realtime), OpenCV, and PyQt6.