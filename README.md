# 🛡️ CodeAlpha Object Detection & Tracking Dashboard

A high-performance real-time Object Detection and Tracking system built using **Python**, **PyQt6**, **YOLO11**, **DeepSORT**, and **OpenCV**.

This project provides both a **Modern Desktop GUI Dashboard** (`app_gui.py`) and a **High-Tech OpenCV HUD CLI Mode** (`main.py`) for live webcam streams, video files, and static image analysis.

---

## 🌟 Key Features

### 🖥️ Modern Desktop GUI Dashboard (`app_gui.py`)
- **Dark Mode Modern Aesthetic**: Clean, responsive layout designed with high-contrast accent highlights.
- **Multi-Source Input Support**: Switch between **Live Webcams**, **Video Files** (`.mp4`, `.avi`, `.mkv`), or **Static Images** (`.jpg`, `.png`).
- **Real-Time Detection & Tracking Tuning**: Live sliders for **Confidence Threshold** ($0.05 - 0.95$) and **IoU NMS Threshold**.
- **Class Filtering**: Select specific object classes (e.g., *Person*, *Car*, *Cell Phone*, *Bottle*, *Cup*, *Chair*, *Laptop*) or process *All Classes*.
- **Visual Customization**:
  - 🌀 **Motion Trajectory Trails**: Color-coded breadcrumb lines showing object motion paths over time.
  - 📦 **Bounding Boxes & Corner Accents**: High-contrast bounding boxes with futuristic corner targeting marks.
  - 🏷️ **Class Badges & ID Tags**: Solid badges displaying Object Class, Tracking ID, and Confidence %.
  - 📊 **Translucent HUD Overlay**: Toggle live overlay stats directly on the video feed.
- **Analytics Dashboard**: Real-time KPI metrics cards for **FPS**, **Latency (ms)**, **Active Tracks**, **Total Unique IDs Assigned**, **Person Count**, and **Vehicle / Object Counts**.
- **Live Track Table**: Auto-updating table displaying active object IDs, classes, confidence scores, and center $(X,Y)$ coordinates.
- **Tools & Actions**:
  - 📸 **Snapshot Generator**: Save instant high-resolution snapshots to `snapshots/`.
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
 YOLO11 Object Detection
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

### 1. Prerequisites
Python 3.10+ installed.

### 2. Environment Setup & Installation
Activate the virtual environment and install the required dependencies:

```bash
# Windows
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launching the GUI Dashboard
To open the modern Desktop GUI application:

```bash
python app_gui.py
```

### 4. Launching the OpenCV HUD CLI Mode
To run the lightweight OpenCV HUD window:

```bash
python main.py
```

---

## 📁 Project Structure

```
CodeAlpha_Object_Detection_Tracking/
├── app_gui.py              # Main Desktop GUI Application (PyQt6)
├── main.py                 # OpenCV HUD CLI Entrypoint
├── tracker_engine.py      # Core Detection & Tracking Engine (YOLO11 + DeepSORT)
├── yolo11n.pt             # Pretrained YOLO11 Model Weights
├── requirements.txt        # Python Dependencies
├── snapshots/             # Generated high-resolution snapshots
├── recordings/            # Recorded MP4 video streams
└── exports/               # Exported CSV / JSON track logs
```

---

## 📜 License & Credits

Developed for **CodeAlpha** Object Detection & Tracking.  
Powered by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [DeepSORT Realtime](https://github.com/levan92/deep_sort_realtime), OpenCV, and PyQt6.