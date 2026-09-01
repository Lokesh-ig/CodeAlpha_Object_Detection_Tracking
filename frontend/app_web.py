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

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & RESPONSIVE CSS
# ==============================================================================

st.set_page_config(
    page_title="CodeAlpha - Universal Object Tracker Web App",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme Styling
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
    .stButton>button:hover {
        border-color: #00f0ff;
        box-shadow: 0 0 8px rgba(0, 240, 255, 0.4);
    }
    [data-testid="stImage"] img {
        border-radius: 8px;
        border: 2px solid #00f0ff;
        object-fit: contain;
    }
    @media (max-width: 768px) {
        div[data-testid="column"] {
            width: 50% !important;
            flex: 1 1 45% !important;
            min-width: 130px !important;
            margin-bottom: 8px;
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


# WebRTC Video Processor Class for Real-Time Browser Stream Tracking
if HAS_WEBRTC:
    class ObjectTrackingVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.engine = get_tracker_engine()
            self.conf_thresh = 0.30
            self.iou_thresh = 0.45
            self.show_trails = True
            self.show_boxes = True
            self.show_labels = True
            self.show_hud = True

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            annotated_frame, stats, active_tracks = self.engine.process_frame(
                img,
                conf_threshold=self.conf_thresh,
                iou_threshold=self.iou_thresh,
                show_trails=self.show_trails,
                show_labels=show_labels,
                show_boxes=show_boxes,
                show_hud=self.show_hud
            )

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# Main Streamlit App
def main():
    st.title("🛡️ CodeAlpha - Universal Object Detection & Tracking Web App")
    st.caption("Powered by YOLO-World Universal Engine, DeepSORT, and Streamlit")

    # Session State Initialization for Camera & Stream Controls
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
        index=0
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
        ["Live WebRTC Browser Stream (Real-Time Tracking)", "Upload Video File (.mp4, .avi)", "Camera Snapshot / Static Image"]
    )

    # 5. Detection Tuning Sliders
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.20, 0.05)
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
    # MAIN DISPLAY LAYOUT
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

    # Default KPI displays
    kpi_fps.metric("FPS", "0.0")
    kpi_lat.metric("LATENCY", "0.0 ms")
    kpi_active.metric("ACTIVE TRACKS", "0")
    kpi_unique.metric("TOTAL UNIQUE IDs", str(len(engine.unique_track_ids)))
    kpi_people.metric("PEOPLE COUNT", "0")

    # --------------------------------------------------------------------------
    # STREAM PROCESSING LOOPS
    # --------------------------------------------------------------------------

    if input_source == "Live WebRTC Browser Stream (Real-Time Tracking)":
        if HAS_WEBRTC:
            st.info("🎥 **Real-Time WebRTC Live Tracking**: Click **START** on the video window below to stream your camera live with real-time detection & tracking!")
            ctx = webrtc_streamer(
                key="object-tracker-stream",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                video_processor_factory=ObjectTrackingVideoProcessor,
                async_processing=True,
            )
            if ctx.video_processor:
                ctx.video_processor.conf_thresh = conf_thresh
                ctx.video_processor.iou_thresh = iou_thresh
                ctx.video_processor.show_trails = show_trails
                ctx.video_processor.show_boxes = show_boxes
                ctx.video_processor.show_labels = show_labels
                ctx.video_processor.show_hud = show_hud
        else:
            st.error("streamlit-webrtc package not found.")

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

                    time.sleep(0.005)
            finally:
                cap.release()

    elif input_source == "Camera Snapshot / Static Image":
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
        img_file_buffer = st.camera_input("📷 Take Live Camera Snapshot")

        target_frame = None
        if uploaded_image:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            target_frame = cv2.imdecode(file_bytes, 1)
        elif img_file_buffer:
            bytes_data = img_file_buffer.getvalue()
            target_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if target_frame is not None:
            annotated_frame, stats, active_tracks = engine.process_frame(
                target_frame,
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
