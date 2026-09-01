import sys
import os
import cv2
import time
import tempfile
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Adjust Python path for backend imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.tracker_engine import ObjectTrackerEngine
except ImportError:
    from tracker_engine import ObjectTrackerEngine


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & RESPONSIVE CSS
# ==============================================================================

st.set_page_config(
    page_title="CodeAlpha - Universal Object Tracker Web App",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme Styling & Universal Device Responsiveness
st.markdown("""
    <style>
    /* Main Background & Text */
    .stApp {
        background-color: #0c0e17;
        color: #e0e6ed;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] {
        font-size: 20px;
        font-weight: bold;
        color: #00f0ff;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 11px;
        color: #8898aa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="metric-container"] {
        background-color: #121626;
        border: 1px solid #1c2338;
        border-radius: 8px;
        padding: 8px 12px;
    }

    /* Sidebar Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: bold;
        padding: 6px 12px;
        border: 1px solid #28314e;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        border-color: #00f0ff;
        box-shadow: 0 0 8px rgba(0, 240, 255, 0.4);
    }

    /* Video Player Canvas Styling */
    [data-testid="stImage"] img {
        border-radius: 8px;
        border: 2px solid #00f0ff;
        object-fit: contain;
    }

    /* Universal Responsive Layout for Mobile, Tablet, & Desktop */
    @media (max-width: 768px) {
        div[data-testid="column"] {
            width: 50% !important;
            flex: 1 1 45% !important;
            min-width: 130px !important;
            margin-bottom: 8px;
        }
        .stApp {
            padding: 4px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 16px;
        }
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# ENGINE CACHING & INITIALIZATION
# ==============================================================================

@st.cache_resource
def get_tracker_engine(model_name="yolov8s-world.pt"):
    return ObjectTrackerEngine(model_name)


# Main Streamlit App
def main():
    st.title("🛡️ CodeAlpha - Universal Object Detection & Tracking Web App")
    st.caption("Powered by YOLO-World Universal Engine, DeepSORT, and Streamlit")

    # Initialize Session State for Stream Control
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = True

    # --------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------------------------------------------------
    st.sidebar.header("⚡ CONTROL CENTER")

    # 0. Stream Control Buttons
    st.sidebar.subheader("🎥 Stream Controls")
    btn_col1, btn_col2 = st.sidebar.columns(2)

    if btn_col1.button("▶️ Start Stream"):
        st.session_state.webcam_running = True
        st.rerun()

    if btn_col2.button("⏹️ Stop Stream"):
        st.session_state.webcam_running = False
        st.rerun()

    # 1. Model Selector
    model_choice = st.sidebar.selectbox(
        "YOLO AI Model Engine",
        [
            "🌍 YOLO-World Universal (yolov8s-world.pt)",
            "⚡ YOLO11 Nano (yolo11n.pt)",
            "🎯 YOLO11 Small (yolo11s.pt)"
        ]
    )

    model_key = "yolov8s-world.pt" if "world" in model_choice else ("yolo11n.pt" if "11n" in model_choice else "yolo11s.pt")
    engine = get_tracker_engine(model_key)

    # 2. Text Prompts (For YOLO-World)
    if "world" in model_choice:
        world_prompts_str = st.sidebar.text_input(
            "World Detection Prompts",
            value="person, cell phone, phone, smartphone, hand, object, item, bottle, gadget, card, toy, bag"
        )
        if world_prompts_str:
            prompts = [p.strip() for p in world_prompts_str.split(",") if p.strip()]
            engine.update_world_prompts(prompts)

    # 3. Speed Mode
    speed_preset = st.sidebar.selectbox(
        "Speed & Performance Mode",
        [
            "⚡ High FPS Mode (320px Fast)",
            "⚖️ Balanced Mode (416px Recommended)",
            "🎯 Max Accuracy (640px Precision)"
        ],
        index=0  # Default High FPS for Web Responsiveness
    )
    if "High FPS" in speed_preset:
        engine.set_speed_preset("fast")
    elif "Max Accuracy" in speed_preset:
        engine.set_speed_preset("accurate")
    else:
        engine.set_speed_preset("balanced")

    # 4. Input Source Selection
    input_source = st.sidebar.selectbox(
        "Input Source",
        ["Live Webcam Stream", "Upload Video File (.mp4, .avi)", "Upload Image File (.jpg, .png)"]
    )

    # 5. Detection Tuning Sliders
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.30, 0.05)
    iou_thresh = st.sidebar.slider("IoU Threshold", 0.05, 0.95, 0.45, 0.05)

    # 6. Visual Toggles
    show_trails = st.sidebar.checkbox("Trajectory Trails 🌀", value=True)
    show_boxes = st.sidebar.checkbox("Bounding Boxes 📦", value=True)
    show_labels = st.sidebar.checkbox("Class Badges 🏷️", value=True)
    show_hud = st.sidebar.checkbox("HUD Stats Panel 📊", value=True)

    if st.sidebar.button("🔄 Reset Tracker"):
        engine.reset_tracker()
        st.sidebar.success("Tracker reset!")

    # --------------------------------------------------------------------------
    # MAIN DISPLAY LAYOUT (AUTO-RESPONSIVE)
    # --------------------------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)
    kpi_fps = col1.empty()
    kpi_lat = col2.empty()
    kpi_active = col3.empty()
    kpi_unique = col4.empty()
    kpi_people = col5.empty()

    video_placeholder = st.empty()

    st.subheader("📋 Active Track Log Table")
    table_placeholder = st.empty()

    # Initialize KPI Default Displays
    kpi_fps.metric("FPS", "0.0")
    kpi_lat.metric("LATENCY", "0.0 ms")
    kpi_active.metric("ACTIVE TRACKS", "0")
    kpi_unique.metric("TOTAL UNIQUE IDs", str(len(engine.unique_track_ids)))
    kpi_people.metric("PEOPLE COUNT", "0")

    # --------------------------------------------------------------------------
    # STREAM PROCESSING LOOPS
    # --------------------------------------------------------------------------

    if input_source == "Live Webcam Stream":
        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not cap.isOpened():
                st.error("✗ Unable to access camera device. Ensure no other application (like main.py or app_gui.py) is using the webcam.")
                st.session_state.webcam_running = False
            else:
                try:
                    frame_idx = 0
                    while st.session_state.webcam_running:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("Webcam frame capture paused or unavailable.")
                            time.sleep(0.05)
                            break

                        frame_idx += 1
                        frame = cv2.flip(frame, 1)

                        annotated_frame, stats, active_tracks = engine.process_frame(
                            frame,
                            conf_threshold=conf_thresh,
                            iou_threshold=iou_thresh,
                            show_trails=show_trails,
                            show_labels=show_labels,
                            show_boxes=show_boxes,
                            show_hud=show_hud
                        )

                        # Resize frame for responsive web rendering
                        web_frame = cv2.resize(annotated_frame, (854, 480))
                        rgb_frame = cv2.cvtColor(web_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        # Update KPI Metrics Cards
                        kpi_fps.metric("FPS", f"{stats['fps']:.1f}")
                        kpi_lat.metric("LATENCY", f"{stats['latency_ms']:.1f} ms")
                        kpi_active.metric("ACTIVE TRACKS", stats['active_tracks'])
                        kpi_unique.metric("TOTAL UNIQUE IDs", stats['total_unique_tracks'])
                        kpi_people.metric("PEOPLE COUNT", stats['person_count'])

                        # Throttle DataFrame UI rendering to every 3 frames for zero lag
                        if frame_idx % 3 == 0:
                            if active_tracks:
                                df = pd.DataFrame(active_tracks)[["track_id", "class", "confidence", "center", "bbox"]]
                                table_placeholder.dataframe(df, use_container_width=True)
                            else:
                                table_placeholder.text("No active objects currently tracked.")

                        time.sleep(0.002)
                finally:
                    cap.release()
        else:
            video_placeholder.info("⏸️ Stream is currently STOPPED. Click '▶️ Start Stream' in the sidebar to start live tracking.")

    elif input_source == "Upload Video File (.mp4, .avi)":
        uploaded_video = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            try:
                frame_idx = 0
                while cap.isOpened() and st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    frame_idx += 1
                    annotated_frame, stats, active_tracks = engine.process_frame(
                        frame,
                        conf_threshold=conf_thresh,
                        iou_threshold=iou_thresh,
                        show_trails=show_trails,
                        show_labels=show_labels,
                        show_boxes=show_boxes,
                        show_hud=show_hud
                    )

                    web_frame = cv2.resize(annotated_frame, (854, 480))
                    rgb_frame = cv2.cvtColor(web_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    kpi_fps.metric("FPS", f"{stats['fps']:.1f}")
                    kpi_lat.metric("LATENCY", f"{stats['latency_ms']:.1f} ms")
                    kpi_active.metric("ACTIVE TRACKS", stats['active_tracks'])
                    kpi_unique.metric("TOTAL UNIQUE IDs", stats['total_unique_tracks'])
                    kpi_people.metric("PEOPLE COUNT", stats['person_count'])

                    if frame_idx % 3 == 0:
                        if active_tracks:
                            df = pd.DataFrame(active_tracks)[["track_id", "class", "confidence", "center", "bbox"]]
                            table_placeholder.dataframe(df, use_container_width=True)

                    time.sleep(0.002)
            finally:
                cap.release()

    elif input_source == "Upload Image File (.jpg, .png)":
        uploaded_image = st.sidebar.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            annotated_frame, stats, active_tracks = engine.process_frame(
                frame,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                show_trails=show_trails,
                show_labels=show_labels,
                show_boxes=show_boxes,
                show_hud=show_hud
            )

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            kpi_fps.metric("FPS", f"{stats['fps']:.1f}")
            kpi_lat.metric("LATENCY", f"{stats['latency_ms']:.1f} ms")
            kpi_active.metric("ACTIVE TRACKS", stats['active_tracks'])
            kpi_unique.metric("TOTAL UNIQUE IDs", stats['total_unique_tracks'])
            kpi_people.metric("PEOPLE COUNT", stats['person_count'])

            if active_tracks:
                df = pd.DataFrame(active_tracks)[["track_id", "class", "confidence", "center", "bbox"]]
                table_placeholder.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
