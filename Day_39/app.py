import os
import tempfile

import cv2
import imageio
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from traffic_violation import TrafficMonitor, render_wrong_way_variant, render_dashboard_variant

st.set_page_config(page_title="Traffic Violation Detection", layout="wide")

MAX_PREVIEW_WIDTH = 900
MAX_PREVIEW_HEIGHT = 600
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
defaults = {
    "zone_points": [],
    "media_path": None,
    "media_type": None,       # "image" or "video"
    "first_frame": None,
    "frame_width": None,
    "frame_height": None,
    "processing": False,
    "uploaded_name": None,
    "output_path": None,
    "output_ready": False,
    "final_stats": None,
    "error_message": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("Traffic Violation Detection System")


def _reset_all():
    for key, value in defaults.items():
        st.session_state[key] = value


def _reset_output_state():
    st.session_state.output_path = None
    st.session_state.output_ready = False
    st.session_state.final_stats = None
    st.session_state.processing = False
    st.session_state.error_message = None


def _compute_preview_size(width, height):
    """Scale to fit within the preview box while preserving aspect ratio.
    Dimensions forced to even numbers since libx264 needs even width/height."""
    scale = min(MAX_PREVIEW_WIDTH / width, MAX_PREVIEW_HEIGHT / height, 1.0)
    new_w = max(2, int(width * scale))
    new_h = max(2, int(height * scale))
    if new_w % 2 != 0:
        new_w -= 1
    if new_h % 2 != 0:
        new_h -= 1
    return new_w, new_h


# ----------------------------------------------------------------------------
# Sidebar settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05,
                            help="Lower = detects more objects but more false positives.")
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.5, 0.05,
                               help="Lower = stricter removal of duplicate/overlapping boxes.")
    allowed_direction = st.selectbox("Allowed Traffic Direction", ["UP", "DOWN", "LEFT", "RIGHT"], index=3)
    variant = st.radio("Output Variant", ["Wrong Way Detection", "Traffic Violation Dashboard"])
    zone_type = st.selectbox("Restricted Zone Shape", ["Polygon", "Line"])
    st.markdown("---")
    if st.button("Reset App"):
        _reset_all()
        st.rerun()

# ----------------------------------------------------------------------------
# 1) Upload media (image or video)
# ----------------------------------------------------------------------------
st.subheader("1. Upload Image or Video")
uploaded_file = st.file_uploader(
    "Choose a video or image file",
    type=list(VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.uploaded_name:
        ext = uploaded_file.name.split(".")[-1].lower()
        media_type = "image" if ext in IMAGE_EXTENSIONS else "video"

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        tfile.write(uploaded_file.read())
        tfile.close()

        st.session_state.media_path = tfile.name
        st.session_state.media_type = media_type
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.zone_points = []
        _reset_output_state()

        try:
            if media_type == "video":
                cap = cv2.VideoCapture(tfile.name)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise ValueError("Could not read any frame from this video.")
            else:
                frame = cv2.imread(tfile.name)
                if frame is None:
                    raise ValueError("Could not read this image file.")

            h, w = frame.shape[:2]
            new_w, new_h = _compute_preview_size(w, h)
            frame = cv2.resize(frame, (new_w, new_h))
            st.session_state.first_frame = frame
            st.session_state.frame_width = new_w
            st.session_state.frame_height = new_h
        except Exception as e:
            st.session_state.first_frame = None
            st.error(f"Failed to load file: {e}. Please try a different file.")

# ----------------------------------------------------------------------------
# 2) Mark restricted zone
# ----------------------------------------------------------------------------
st.subheader("2. Mark Restricted Zone")

if st.session_state.first_frame is not None:
    st.caption("Click on the frame below to add points. Polygon needs 3+ points, Line needs 2+ points.")

    preview = st.session_state.first_frame.copy()
    pts = st.session_state.zone_points
    for pt in pts:
        cv2.circle(preview, pt, 5, (0, 255, 255), -1)
    if len(pts) >= 2:
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [pts_arr], False, (0, 0, 255), 2)
    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    coords = streamlit_image_coordinates(preview_rgb, key="zone_canvas")
    if coords is not None:
        point = (int(coords["x"]), int(coords["y"]))
        if not st.session_state.zone_points or st.session_state.zone_points[-1] != point:
            st.session_state.zone_points.append(point)
            st.rerun()

    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        if st.button("Undo Last Point"):
            if st.session_state.zone_points:
                st.session_state.zone_points.pop()
                st.rerun()
    with col_b:
        if st.button("Clear Zone"):
            st.session_state.zone_points = []
            st.rerun()
    with col_c:
        st.write(f"Points marked: {len(st.session_state.zone_points)}")
else:
    st.info("Upload an image or video above to mark a restricted zone.")

# ----------------------------------------------------------------------------
# 3) Start processing
# ----------------------------------------------------------------------------
st.subheader("3. Start Processing")
start_button = st.button("Start Processing", type="primary", disabled=st.session_state.media_path is None)

if start_button:
    if st.session_state.media_path is None:
        st.error("Please upload a file first.")
    elif zone_type == "Polygon" and len(st.session_state.zone_points) < 3:
        st.error("Please mark at least 3 points for a polygon zone.")
    elif zone_type == "Line" and len(st.session_state.zone_points) < 2:
        st.error("Please mark at least 2 points for a line zone.")
    else:
        _reset_output_state()
        st.session_state.processing = True

# ----------------------------------------------------------------------------
# 4) Processing (branches for image vs video)
# ----------------------------------------------------------------------------
if st.session_state.processing:
    status_line = st.empty()
    progress_bar = st.progress(0)
    status_line.info("Processing... please wait.")

    try:
        monitor = TrafficMonitor(
            confidence=confidence,
            iou=iou_threshold,
            allowed_direction=allowed_direction,
            restricted_zone=list(st.session_state.zone_points),
            zone_type=zone_type,
            frame_width=st.session_state.frame_width,
            frame_height=st.session_state.frame_height,
        )

        if st.session_state.media_type == "image":
            # -------- Single image: one pass through the model --------
            frame = cv2.imread(st.session_state.media_path)
            if frame is None:
                raise ValueError("Could not re-read the uploaded image.")

            frame, detections = monitor.process_frame(frame)
            if variant == "Wrong Way Detection":
                output_frame = render_wrong_way_variant(frame, detections, monitor)
            else:
                output_frame = render_dashboard_variant(frame, detections, monitor)

            out_dir = tempfile.mkdtemp()
            output_path = os.path.join(out_dir, "output.png")
            cv2.imwrite(output_path, output_frame)

            progress_bar.progress(1.0)
            st.session_state.output_path = output_path
            st.session_state.final_stats = monitor.get_stats()
            st.session_state.output_ready = True

        else:
            # -------- Video: frame-by-frame with progress bar --------
            cap = cv2.VideoCapture(st.session_state.media_path)
            if not cap.isOpened():
                raise ValueError("Could not open the uploaded video.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

            out_dir = tempfile.mkdtemp()
            output_path = os.path.join(out_dir, "output.mp4")

            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )

            frame_count = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    frame, detections = monitor.process_frame(frame)

                    if variant == "Wrong Way Detection":
                        output_frame = render_wrong_way_variant(frame, detections, monitor)
                    else:
                        output_frame = render_dashboard_variant(frame, detections, monitor)

                    writer.append_data(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))

                    if total_frames > 0:
                        pct = min(1.0, frame_count / total_frames)
                        progress_bar.progress(pct)
                        status_line.info(f"Processing... frame {frame_count} of {total_frames} ({int(pct * 100)}%)")
                    else:
                        status_line.info(f"Processing... frame {frame_count}")
            finally:
                cap.release()
                writer.close()

            st.session_state.output_path = output_path
            st.session_state.final_stats = monitor.get_stats()
            st.session_state.output_ready = True
            progress_bar.progress(1.0)

        status_line.success("Processing complete.")

    except Exception as e:
        st.session_state.error_message = str(e)
        status_line.error(f"Processing failed: {e}")

    st.session_state.processing = False
    st.rerun()

if st.session_state.error_message:
    st.error(f"Last run failed: {st.session_state.error_message}")

# ----------------------------------------------------------------------------
# 5) Output + downloads + stats
# ----------------------------------------------------------------------------
if st.session_state.output_ready and st.session_state.output_path:
    st.subheader("4. Output")

    if st.session_state.media_type == "image":
        st.image(st.session_state.output_path, channels="BGR", use_container_width=False)
        with open(st.session_state.output_path, "rb") as f:
            st.download_button("Download Output Image", data=f.read(),
                                file_name="traffic_violation_output.png", mime="image/png")
    else:
        video_col, _ = st.columns([2, 1])
        with video_col:
            st.video(st.session_state.output_path)
        with open(st.session_state.output_path, "rb") as f:
            st.download_button("Download Output Video", data=f.read(),
                                file_name="traffic_violation_output.mp4", mime="video/mp4")

    stats = st.session_state.final_stats
    if stats:
        st.markdown("---")
        st.subheader("Summary")

        row1 = st.columns(4)
        row1[0].metric("Total Vehicles", stats["total_vehicles"])
        row1[1].metric("Total Violations", stats["total_violations"])
        row1[2].metric("Wrong Way Violations", stats["wrong_way_violations"])
        row1[3].metric("Restricted Zone Violations", stats["restricted_zone_violations"])

        row2 = st.columns(4)
        row2[0].metric("Cars", stats["class_counts"].get("Car", 0))
        row2[1].metric("Trucks", stats["class_counts"].get("Truck", 0))
        row2[2].metric("Buses", stats["class_counts"].get("Bus", 0))
        row2[3].metric("Motorcycles", stats["class_counts"].get("Motorcycle", 0))

        if stats["violations"]:
            df = pd.DataFrame(stats["violations"])
            st.dataframe(df[["id", "type", "class", "timestamp", "frame"]], use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Violations CSV Report",
                data=csv_bytes,
                file_name="violations_report.csv",
                mime="text/csv",
            )
        else:
            st.info("No violations recorded.")

        if stats["total_violations"] > 0:
            st.warning(f"ALERT: {stats['total_violations']} VIOLATIONS DETECTED")
        else:
            st.success("TRAFFIC STATUS: NORMAL")