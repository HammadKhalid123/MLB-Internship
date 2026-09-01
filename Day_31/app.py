import os
import time
import tempfile
import streamlit as st
import cv2

from vehicle_counting import load_model, process_video, SCRIPT_DIR

st.set_page_config(page_title="Smart Vehicle Counting System", layout="wide")

st.title("Smart Vehicle Counting System")

with st.sidebar:
    st.header("Settings")

    tracker_choice = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])
    conf_value = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
    iou_value = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

    model_path = os.path.join(SCRIPT_DIR, "yolov8n.pt")

uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    input_folder = os.path.join(SCRIPT_DIR, "input_videos")
    output_folder = os.path.join(SCRIPT_DIR, "processed_videos")
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    input_path = os.path.join(input_folder, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    output_filename = os.path.splitext(uploaded_file.name)[0] + ".mp4"
    output_path = os.path.join(output_folder, output_filename)

    st.video(input_path)

    start_button = st.button("Process Video")

    if start_button:
        model = load_model(model_path)

        progress_bar = st.progress(0)
        status_text = st.empty()

        def frame_callback(frame, vehicle_counts, frame_num, total_frames):
            progress = frame_num / total_frames if total_frames > 0 else 0
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing frame {frame_num}/{total_frames}")

        start_time = time.time()

        final_counts = process_video(
            model,
            input_path,
            output_path,
            tracker=tracker_choice,
            conf=conf_value,
            iou=iou_value,
            frame_callback=frame_callback
        )

        elapsed_time = time.time() - start_time

        status_text.text(f"Processing complete in {elapsed_time:.2f} seconds")

        st.success("Video processed successfully")

        st.subheader("Final Vehicle Counts")
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        result_col1.metric("Cars", final_counts["car"])
        result_col2.metric("Motorcycles", final_counts["motorcycle"])
        result_col3.metric("Buses", final_counts["bus"])
        result_col4.metric("Trucks", final_counts["truck"])

        st.subheader("Processed Video")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "Download Processed Video",
                f,
                file_name=f"processed_{output_filename}",
                mime="video/mp4"
            )