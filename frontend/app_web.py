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
# STREAMLIT PAGE CONFIGURATION & RESPONSIVE STYLING
# ==============================================================================

st.set_page_config(
    page_title="CodeAlpha - Universal Object Tracker Web Service",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0c0e17;
        color: #e0e6ed;
    }
    div[data-testid="stMetricValue"] {
        font-size: 20px;
        font-weight: bold;
        color: #00f0ff;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 11px;
        color: #8898aa;
        text-transform: uppercase;
    }
    div[data-testid="metric-container"] {
        background-color: #121626;
        border: 1px solid #1c2338;
        border-radius: 8px;
        padding: 8px 12px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: bold;
        padding: 6px 12px;
        border: 1px solid #28314e;
    }
    [data-testid="stImage"] img {
        border-radius: 8px;
        border: 2px solid #00f0ff;
        object-fit: contain;
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# ENGINE CACHING & INITIALIZATION
# ==============================================================================

@st.cache_resource
def get_tracker_engine(model_name="yolo11n.pt"):
    return ObjectTrackerEngine(model_name)


def main():
    st.title("🛡️ CodeAlpha - Universal Object Detection & Tracking Web Service")
    st.caption("Enterprise AI Vision Dashboard Powered by YOLO & DeepSORT")

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = True

    if "cap" not in st.session_state:
        st.session_state.cap = None

    if "frame_counter" not in st.session_state:
        st.session_state.frame_counter = 0

    # --------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------------------------------------------------
    st.sidebar.header("⚡ CONTROL CENTER")

    btn_col1, btn_col2 = st.sidebar.columns(2)
    if btn_col1.button("▶️ Start Stream"):
        st.session_state.webcam_running = True
        st.rerun()

    if btn_col2.button("⏹️ Pause Stream"):
        st.session_state.webcam_running = False
        if st.session_state.cap is not None:
            try:
                st.session_state.cap.release()
            except Exception:
                pass
            st.session_state.cap = None
        st.rerun()

    model_choice = st.sidebar.selectbox(
        "YOLO AI Model Engine",
        [
            "⚡ YOLO11 Nano (yolo11n.pt) [Ultra Fast & Lightweight]",
            "🎯 YOLO11 Small (yolo11s.pt)",
            "🌍 YOLO-World Universal (yolov8s-world.pt)"
        ],
        index=0
    )

    model_key = "yolov8s-world.pt" if "world" in model_choice else ("yolo11s.pt" if "11s" in model_choice else "yolo11n.pt")
    engine = get_tracker_engine(model_key)

    if "world" in model_choice:
        world_prompts_str = st.sidebar.text_input(
            "World Detection Prompts",
            value="person, cell phone, phone, smartphone, hand, object, item, bottle, gadget, card, toy, bag"
        )
        if world_prompts_str:
            prompts = [p.strip() for p in world_prompts_str.split(",") if p.strip()]
            engine.update_world_prompts(prompts)

    speed_preset = st.sidebar.selectbox(
        "Speed & Performance Mode",
        [
            "⚡ High FPS Mode (320px Fast)",
            "⚖️ Balanced Mode (416px Recommended)",
            "🎯 Max Accuracy (640px Precision)"
        ],
        index=0
    )
    if "High FPS" in speed_preset:
        engine.set_speed_preset("fast")
    elif "Max Accuracy" in speed_preset:
        engine.set_speed_preset("accurate")
    else:
        engine.set_speed_preset("balanced")

    input_source = st.sidebar.selectbox(
        "Input Source",
        [
            "Live Camera Stream (Webcam)",
            "📹 Demo Sample Video Stream (Auto-Loop Instant AI)",
            "Upload Video File (.mp4, .avi)",
            "Upload Image File (.jpg, .png)"
        ],
        index=0
    )

    # Low light sensitive confidence slider default (0.10 default for dark/backlit rooms)
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.10, 0.05)
    iou_thresh = st.sidebar.slider("IoU Threshold", 0.05, 0.95, 0.45, 0.05)

    show_trails = st.sidebar.checkbox("Trajectory Trails 🌀", value=True)
    show_boxes = st.sidebar.checkbox("Bounding Boxes 📦", value=True)
    show_labels = st.sidebar.checkbox("Class Badges 🏷️", value=True)
    show_hud = st.sidebar.checkbox("HUD Stats Panel 📊", value=True)

    if st.sidebar.button("🔄 Reset Tracker"):
        engine.reset_tracker()
        st.sidebar.success("Tracker reset!")

    # --------------------------------------------------------------------------
    # MAIN DISPLAY LAYOUT
    # --------------------------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)
    kpi_fps = col1.empty()
    kpi_lat = col2.empty()
    kpi_active = col3.empty()
    kpi_unique = col4.empty()
    kpi_people = col5.empty()

    kpi_fps.metric("FPS", "0.0")
    kpi_lat.metric("LATENCY", "0.0 ms")
    kpi_active.metric("ACTIVE TRACKS", "0")
    kpi_unique.metric("TOTAL UNIQUE IDs", str(len(engine.unique_track_ids)))
    kpi_people.metric("PEOPLE COUNT", "0")

    video_container = st.container()

    st.subheader("📋 Active Track Log Table")
    table_placeholder = st.empty()

    # --------------------------------------------------------------------------
    # STREAM PROCESSING LOOPS
    # --------------------------------------------------------------------------

    if input_source == "Live Camera Stream (Webcam)":
        if st.session_state.webcam_running:
            if st.session_state.cap is None or not st.session_state.cap.isOpened():
                try:
                    st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not st.session_state.cap.isOpened():
                        st.session_state.cap = cv2.VideoCapture(0)
                    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                except Exception:
                    st.session_state.cap = None

            cap = st.session_state.cap
            camera_active = False

            if cap is not None and cap.isOpened():
                video_placeholder = video_container.empty()
                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    camera_active = True
                    st.session_state.frame_counter += 1
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

                    web_frame = cv2.resize(annotated_frame, (854, 480))
                    rgb_frame = cv2.cvtColor(web_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    kpi_fps.metric("FPS", f"{stats['fps']:.1f}")
                    kpi_lat.metric("LATENCY", f"{stats['latency_ms']:.1f} ms")
                    kpi_active.metric("ACTIVE TRACKS", stats['active_tracks'])
                    kpi_unique.metric("TOTAL UNIQUE IDs", stats['total_unique_tracks'])
                    kpi_people.metric("PEOPLE COUNT", stats['person_count'])

                    if st.session_state.frame_counter % 3 == 0:
                        if active_tracks:
                            df = pd.DataFrame(active_tracks)[["track_id", "class", "confidence", "center", "bbox"]]
                            table_placeholder.dataframe(df, use_container_width=True)
                        else:
                            table_placeholder.text("No active objects currently tracked.")

                    time.sleep(0.005)

            if not camera_active:
                with video_container:
                    img_file_buffer = st.camera_input("📷 Live Camera Stream (Click 'Take Photo' below to activate live AI tracking)")
                    video_output = st.empty()
                    if img_file_buffer is not None:
                        bytes_data = img_file_buffer.getvalue()
                        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

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
                        video_output.image(rgb_frame, channels="RGB", use_container_width=True)

                        kpi_fps.metric("FPS", f"{stats['fps']:.1f}")
                        kpi_lat.metric("LATENCY", f"{stats['latency_ms']:.1f} ms")
                        kpi_active.metric("ACTIVE TRACKS", stats['active_tracks'])
                        kpi_unique.metric("TOTAL UNIQUE IDs", stats['total_unique_tracks'])
                        kpi_people.metric("PEOPLE COUNT", stats['person_count'])

                        if active_tracks:
                            df = pd.DataFrame(active_tracks)[["track_id", "class", "confidence", "center", "bbox"]]
                            table_placeholder.dataframe(df, use_container_width=True)
                        else:
                            table_placeholder.text("No active objects detected in this snapshot. Try lowering Confidence Threshold in sidebar.")

    elif input_source == "📹 Demo Sample Video Stream (Auto-Loop Instant AI)":
        with video_container:
            video_placeholder = st.empty()

        sample_path = os.path.join(parent_dir, "recordings", "recording_20260901_165140.mp4")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(current_dir, "sample.mp4")

        cap = cv2.VideoCapture(sample_path)
        if not cap.isOpened():
            frame_idx = 0
            while st.session_state.webcam_running:
                frame_idx += 1
                synthetic = np.zeros((480, 854, 3), dtype=np.uint8)
                synthetic[:] = (20, 24, 38)
                
                cx = int(427 + 250 * np.sin(frame_idx * 0.05))
                cy = int(240 + 100 * np.cos(frame_idx * 0.05))
                cv2.circle(synthetic, (cx, cy), 35, (0, 240, 255), -1)
                cv2.putText(synthetic, "Demo Tracking Object", (cx - 70, cy - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                annotated_frame, stats, active_tracks = engine.process_frame(
                    synthetic,
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

                time.sleep(0.01)
        else:
            try:
                frame_idx = 0
                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

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

                    time.sleep(0.005)
            finally:
                cap.release()

    elif input_source == "Upload Video File (.mp4, .avi)":
        with video_container:
            uploaded_video = st.file_uploader("📁 Choose or Drag & Drop any video file here:", type=["mp4", "avi", "mov", "mkv"])
            video_placeholder = st.empty()

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

                    time.sleep(0.005)
            finally:
                cap.release()

    elif input_source == "Upload Image File (.jpg, .png)":
        with video_container:
            uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
            image_placeholder = st.empty()

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
            image_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

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
