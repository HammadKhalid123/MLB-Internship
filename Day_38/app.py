import streamlit as st
import numpy as np
import cv2
import tempfile
import io
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from code import load_image, apply_segmentation, convert_to_pil
from security import load_model, get_first_frame, parse_polygon_objects, scale_polygons, process_video, save_events_csv

st.set_page_config(page_title="Intelligent Vision Suite", layout="wide")

st.title("Intelligent Vision Suite")

tab1, tab2 = st.tabs(["🎥 Security Monitoring", "🖼️ Segmentation"])

with tab1:
    st.header("Intelligent Security Monitoring System")
    st.write("Upload a video, mark one or more polygon ROIs, then track people and log entry/exit events.")

    video_file = st.file_uploader("Upload Security Video", type=["mp4", "avi", "mov"], key="video_upload")

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_path = tfile.name

        frame, width, height, fps = get_first_frame(video_path)

        if frame is not None:
            st.subheader("Mark ROI on First Frame")
            st.write(f"Original video size: {width} x {height}. Click to add polygon points, double-click to close a shape.")

            canvas_width = 700
            canvas_height = int(height * (canvas_width / width))
            display_frame = cv2.resize(frame, (canvas_width, canvas_height))

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=Image.fromarray(display_frame),
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="polygon",
                key="roi_canvas"
            )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                conf = st.slider("YOLO Confidence", 0.10, 0.90, 0.30)
            with col_b:
                frame_skip = st.slider("Process Every Nth Frame", 1, 5, 1)
            with col_c:
                stable_frames = st.slider("Stable Frames Before Event", 1, 10, 2)

            col_d, col_e = st.columns(2)
            with col_d:
                model_choice = st.selectbox("YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=1)
            with col_e:
                imgsz = st.selectbox("Inference Resolution", [640, 960, 1280], index=1)

            if st.button("Run Security Monitoring"):
                if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                    polygons = parse_polygon_objects(canvas_result.json_data["objects"])
                    if len(polygons) == 0:
                        st.warning("Please draw at least one closed ROI polygon before running.")
                    else:
                        scale_x = width / canvas_width
                        scale_y = height / canvas_height
                        rois = scale_polygons(polygons, scale_x, scale_y)

                        model = load_model(model_choice)
                        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

                        progress_bar = st.progress(0)

                        def update_progress(p):
                            progress_bar.progress(p)

                        with st.spinner("Running YOLO person tracking and event analytics..."):
                            events, summary = process_video(video_path, rois, model, conf, frame_skip, stable_frames, imgsz, output_path, update_progress)

                        st.success("Processing complete")
                        st.video(output_path)

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Unique People", summary["unique_people"])
                        m2.metric("Entries", summary["entries"])
                        m3.metric("Exits", summary["exits"])
                        m4.metric("Max Active", summary["max_active"])

                        st.write(f"Total Events Logged: {len(events)}")

                        csv_path = save_events_csv(events)
                        with open(csv_path, "rb") as f:
                            st.download_button("Download Event Log CSV", f, file_name="events.csv", mime="text/csv")
                else:
                    st.warning("Please draw at least one ROI polygon before running.")

with tab2:
    st.header("Image Segmentation")
    st.write("Upload an image, choose a segmentation method, and download the processed output.")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="seg_upload")

    method = st.selectbox("Select Segmentation Method", ["Binary", "Otsu", "Adaptive"])

    thresh_val = 127
    block_size = 11
    c = 2
    adaptive_method = "mean"

    if method == "Binary":
        thresh_val = st.slider("Threshold Value", 0, 255, 127)

    if method == "Adaptive":
        adaptive_method = st.selectbox("Adaptive Method", ["mean", "gaussian"])
        block_size = st.slider("Block Size", 3, 51, 11, step=2)
        c = st.slider("C Value", 0, 20, 2)

    if uploaded_file is not None:
        image = load_image(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        result = apply_segmentation(image, method, thresh_val, block_size, c, adaptive_method)

        with col2:
            st.subheader("Segmented Output")
            st.image(result, use_column_width=True, clamp=True, channels="GRAY")

        result_pil = convert_to_pil(result)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Processed Image",
            data=byte_im,
            file_name="segmented_output.png",
            mime="image/png"
        )
    else:
        st.info("Please upload an image to get started.")