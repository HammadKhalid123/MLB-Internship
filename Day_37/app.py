import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from people_detection import PeopleDetector, convert_to_h264

st.set_page_config(page_title="Smart People Counting System", layout="wide")

st.title("Smart People Counting System")
st.caption("YOLO based people detection with entry/exit line counting and region based counting")

if "detector" not in st.session_state:
    st.session_state.detector = None
if "detector_model" not in st.session_state:
    st.session_state.detector_model = None

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox(
    "YOLO Model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    index=0
)
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
iou = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.45, 0.05)

st.sidebar.header("Tracking Settings")
tracker_choice = st.sidebar.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"], index=0)
show_ids = st.sidebar.checkbox("Show Track IDs", value=True)
show_trail = st.sidebar.checkbox("Show Movement Trail", value=True)
trail_length = st.sidebar.slider("Trail Length", 5, 100, 30, 5)

st.sidebar.header("Display Settings")
box_thickness = st.sidebar.slider("Bounding Box Thickness", 1, 5, 2, 1)
box_color_hex = st.sidebar.color_picker("Bounding Box Color", "#00FF00")
box_color_rgb = tuple(int(box_color_hex[i:i + 2], 16) for i in (1, 3, 5))
box_color_bgr = (box_color_rgb[2], box_color_rgb[1], box_color_rgb[0])

st.sidebar.header("Counting Feature")
counting_mode = st.sidebar.radio(
    "Select Counting Method",
    ["None", "Entry/Exit Line Counting", "Region Based Counting"]
)


@st.cache_resource
def load_detector(model_path):
    return PeopleDetector(model_path)


if st.session_state.detector_model != model_choice:
    st.session_state.detector = load_detector(model_choice)
    st.session_state.detector_model = model_choice

detector = st.session_state.detector

uploaded_file = st.file_uploader(
    "Upload a Video",
    type=["mp4", "avi", "mov", "mkv"]
)


def extract_click_points(objects, scale_x, scale_y):
    points = []
    for obj in objects:
        if obj["type"] == "circle":
            radius = obj.get("radius", 0) * obj.get("scaleX", 1)
            cx = obj["left"] + radius
            cy = obj["top"] + radius
            points.append((int(cx * scale_x), int(cy * scale_y)))
    return points


def extract_polygon_points(objects, scale_x, scale_y):
    for obj in objects:
        if obj["type"] == "path":
            points = []
            for cmd in obj["path"]:
                if len(cmd) >= 3 and cmd[0] in ("M", "L"):
                    points.append((int(cmd[1] * scale_x), int(cmd[2] * scale_y)))
            if len(points) >= 3:
                return points
        elif obj["type"] == "polygon":
            raw_points = obj.get("points", [])
            left = obj.get("left", 0)
            top = obj.get("top", 0)
            points = [(int((p["x"] + left) * scale_x), int((p["y"] + top) * scale_y)) for p in raw_points]
            if len(points) >= 3:
                return points
    return None


if uploaded_file is None:
    st.info("Upload a video to get started.")
    st.stop()

file_id = f"{uploaded_file.name}_{uploaded_file.size}"

if st.session_state.get("current_file_id") != file_id:
    temp_input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_input_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        st.error("Could not read the uploaded video.")
        st.stop()

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = first_frame.shape[:2]

    display_width = 700
    scale = display_width / orig_width
    display_height = int(orig_height * scale)
    resized_pil = Image.fromarray(first_frame_rgb).resize((display_width, display_height))

    st.session_state.current_file_id = file_id
    st.session_state.temp_input_path = temp_input_path
    st.session_state.orig_width = orig_width
    st.session_state.orig_height = orig_height
    st.session_state.display_width = display_width
    st.session_state.display_height = display_height
    st.session_state.resized_pil = resized_pil
    st.session_state.first_frame = first_frame
    st.session_state.line_canvas_key = 0
    st.session_state.roi_canvas_key = 0

temp_input_path = st.session_state.temp_input_path
orig_width = st.session_state.orig_width
orig_height = st.session_state.orig_height
display_width = st.session_state.display_width
display_height = st.session_state.display_height
resized_pil = st.session_state.resized_pil
first_frame = st.session_state.first_frame

st.subheader("Uploaded Video Preview")
st.video(temp_input_path)

line_points = None
roi_points = None

if counting_mode == "Entry/Exit Line Counting":
    st.subheader("Mark the Counting Line")
    st.info("Click once for the START point and once for the END point of the line.")

    reset_col, _ = st.columns([1, 4])
    if reset_col.button("Reset Points"):
        st.session_state.line_canvas_key += 1
        st.rerun()

    canvas_result = st_canvas(
        fill_color="#FF0000",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=resized_pil,
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode="point",
        point_display_radius=6,
        key=f"line_canvas_video_{st.session_state.line_canvas_key}"
    )

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        scale_x = orig_width / display_width
        scale_y = orig_height / display_height
        clicked_points = extract_click_points(objects, scale_x, scale_y)

        if len(clicked_points) >= 2:
            line_points = clicked_points[:2]
            st.success(f"Start point: {line_points[0]}   End point: {line_points[1]}")

            preview = first_frame.copy()
            cv2.line(preview, line_points[0], line_points[1], (0, 0, 255), 3)
            cv2.circle(preview, line_points[0], 8, (0, 255, 0), -1)
            cv2.circle(preview, line_points[1], 8, (255, 0, 0), -1)
            cv2.putText(preview, "START", (line_points[0][0] + 10, line_points[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(preview, "END", (line_points[1][0] + 10, line_points[1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Line Preview", use_container_width=True)
        elif len(clicked_points) == 1:
            st.warning(f"Start point: {clicked_points[0]}. Now click the END point.")
        else:
            st.warning("Click two points on the frame above to set the line.")

elif counting_mode == "Region Based Counting":
    st.subheader("Draw a Region of Interest on the First Frame")
    st.info("Click to place polygon points, double click to close the shape.")

    reset_col, _ = st.columns([1, 4])
    if reset_col.button("Reset Region"):
        st.session_state.roi_canvas_key += 1
        st.rerun()

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 0, 0.2)",
        stroke_width=3,
        stroke_color="#FFFF00",
        background_image=resized_pil,
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode="polygon",
        key=f"roi_canvas_video_{st.session_state.roi_canvas_key}"
    )

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        scale_x = orig_width / display_width
        scale_y = orig_height / display_height
        roi_points = extract_polygon_points(objects, scale_x, scale_y)
        if roi_points is not None:
            st.success(f"Region set with {len(roi_points)} points")
        else:
            st.warning("No region detected yet. Draw a closed shape on the frame above.")

run_clicked = st.button("Run Processing", type="primary")

if run_clicked:
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(value):
        progress_bar.progress(value)
        status_text.text(f"Processing: {int(value * 100)}%")

    raw_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    final_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    with st.spinner("Running YOLO detection and tracking..."):
        stats = detector.process_video(
            input_path=temp_input_path,
            output_path=raw_output_path,
            confidence=confidence,
            iou=iou,
            line_points=line_points,
            roi_points=roi_points,
            tracker=tracker_choice,
            box_color=box_color_bgr,
            box_thickness=box_thickness,
            show_ids=show_ids,
            show_trail=show_trail,
            trail_length=trail_length,
            progress_callback=update_progress
        )

    status_text.text("Converting video for browser playback...")
    convert_to_h264(raw_output_path, final_output_path)
    status_text.text("Done")

    st.subheader("Processed Video Output")
    video_bytes = open(final_output_path, "rb").read()
    st.video(video_bytes)

    st.subheader("Detection Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Max People Detected", stats["max_count"])
    m2.metric("Total Unique People", stats["unique_people_seen"])
    m3.metric("Frames Processed", stats["total_frames"])

    if line_points is not None:
        m4, m5 = st.columns(2)
        m4.metric("Entry Count", stats["entry_count"])
        m5.metric("Exit Count", stats["exit_count"])

    if roi_points is not None:
        st.metric("Unique People In ROI", stats["unique_in_roi"])

    st.download_button(
        "Download Processed Video",
        data=video_bytes,
        file_name="processed_output.mp4",
        mime="video/mp4"
    )

    os.remove(raw_output_path)