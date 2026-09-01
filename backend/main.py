import cv2
import time
import os
import sys
from datetime import datetime

from tracker_engine import ObjectTrackerEngine


# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_PATH = "yolov8s-world.pt"
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
WINDOW_NAME = "CodeAlpha - High-Tech Universal Object Detection & Tracking HUD"


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    print("=" * 70)
    print(" 🚀 CODEALPHA UNIVERSAL OBJECT DETECTION & TRACKING - HIGH-TECH HUD MODE")
    print("=" * 70)

    # Initialize Tracker Engine with YOLO-World Universal
    engine = ObjectTrackerEngine(model_path=MODEL_PATH)

    # Open Camera
    print(f"\nOpening Camera index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("✗ Error: Could not open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera connected: {actual_w}x{actual_h}")

    # Create Window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, actual_w, actual_h)

    # Print Controls Banner
    print("\n" + "-" * 70)
    print(" 🎮 KEYBOARD SHORTCUT CONTROLS:")
    print("   [SPACE] : Pause / Resume video stream")
    print("   [M]     : Switch Model (YOLO-World Universal <-> YOLO11 Nano)")
    print("   [S]     : Take high-resolution snapshot")
    print("   [R]     : Toggle video stream recording (MP4)")
    print("   [T]     : Toggle motion trajectory trails")
    print("   [H]     : Toggle translucent HUD overlay")
    print("   [C]     : Reset tracker state & clear object history")
    print("   [E]     : Export session track logs to CSV & JSON")
    print("   [Q]     : Quit application")
    print("-" * 70 + "\n")

    # State variables
    paused = False
    show_trails = True
    show_hud = True
    conf_threshold = 0.30  # Optimized default confidence for small/handheld items
    iou_threshold = 0.45
    paused_frame = None
    current_model = MODEL_PATH

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Warning: Failed to capture camera frame.")
                    time.sleep(0.01)
                    continue

                # Mirror frame horizontally for natural view
                frame = cv2.flip(frame, 1)

                annotated_frame, stats, active_tracks = engine.process_frame(
                    frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    show_trails=show_trails,
                    show_hud=show_hud
                )
                paused_frame = annotated_frame.copy()
                display_frame = annotated_frame
            else:
                display_frame = paused_frame if paused_frame is not None else frame
                cv2.putText(
                    display_frame,
                    "|| STREAM PAUSED",
                    (display_frame.shape[1] // 2 - 120, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            # Display frame
            cv2.imshow(WINDOW_NAME, display_frame)

            # Keyboard Input Handling
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q or ESC to quit
                print("\n[User Request] Quitting application...")
                break
            elif key == ord(" "):  # Space to pause/resume
                paused = not paused
                print(f"[State] Stream {'PAUSED' if paused else 'RESUMED'}")
            elif key == ord("m"):  # M to switch models
                current_model = "yolo11n.pt" if "world" in current_model else "yolov8s-world.pt"
                engine.load_model(current_model)
                print(f"🔄 Switched AI Model Engine -> {current_model}")
            elif key == ord("s"):  # S for Snapshot
                snap_frame = paused_frame if paused and paused_frame is not None else frame
                saved_path = engine.take_snapshot(snap_frame)
                print(f"📸 Snapshot saved: {saved_path}")
            elif key == ord("r"):  # R for Record
                if not engine.is_recording:
                    os.makedirs("recordings", exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join("recordings", f"recording_{ts}.mp4")
                    engine.start_recording(out_path, fps=25.0, frame_size=(actual_w, actual_h))
                    print(f"🎥 Recording STARTED -> {out_path}")
                else:
                    rec_path = engine.stop_recording()
                    print(f"⏹️ Recording STOPPED -> {rec_path}")
            elif key == ord("t"):  # T for Trails
                show_trails = not show_trails
                print(f"[Visual] Trajectory trails {'ENABLED' if show_trails else 'DISABLED'}")
            elif key == ord("h"):  # H for HUD
                show_hud = not show_hud
                print(f"[Visual] HUD Overlay {'ENABLED' if show_hud else 'DISABLED'}")
            elif key == ord("c"):  # C for Reset
                engine.reset_tracker()
                print("🔄 Tracker reset.")
            elif key == ord("e"):  # E for Export
                os.makedirs("exports", exist_ok=True)
                ts = int(time.time())
                engine.export_logs_csv(os.path.join("exports", f"logs_{ts}.csv"))
                engine.export_logs_json(os.path.join("exports", f"logs_{ts}.json"))
                print("💾 Logs exported to exports/ directory.")

    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    finally:
        print("\nCleaning up resources...")
        if engine.is_recording:
            engine.stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera released and window destroyed.")
        print("Application closed.")


if __name__ == "__main__":
    main()