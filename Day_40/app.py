import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from analytics_code import process_video


st.set_page_config(
    page_title="Smart Video Analytics System",
    layout="wide"
)

st.title("Smart Video Analytics System")

st.write(
    "Upload a video to detect, track and analyze objects using YOLO and BoT-SORT."
)

st.sidebar.header("Settings")

image_size_choice = st.sidebar.radio(
    "Image Size",
    ["640px image size", "480px image size"]
)

if image_size_choice == "640px image size":
    imgsz = 640
else:
    imgsz = 480

frame_skip_enabled = st.sidebar.checkbox(
    "Frame skipping enabled"
)

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(uploaded_file.read())
        input_path = temp_file.name

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    ret, first_frame = cap.read()

    cap.release()

    st.subheader("Video Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Width", f"{width}px")

    with col2:
        st.metric("Height", f"{height}px")

    with col3:
        st.metric("Video FPS", f"{video_fps:.1f}")

    st.subheader("Mark Region of Interest (ROI)")

    roi_type = st.radio(
        "Choose ROI",
        ["Full Video", "Custom ROI"],
        horizontal=True
    )

    if roi_type == "Full Video":

        roi_points = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1)
        ]

    else:

        if not ret:

            st.error("Could not read the first frame for ROI marking.")
            st.stop()

        canvas_width = 700

        canvas_height = int(height * (canvas_width / width))

        background_image = Image.fromarray(
            cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        ).resize((canvas_width, canvas_height))

        st.write(
            "Click points on the frame below to mark the ROI. "
            "Lines will connect each point as you click. Double-click to close the shape."
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.15)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=background_image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="polygon",
            key="roi_canvas"
        )

        roi_points = None

        if (
            canvas_result.json_data is not None
            and len(canvas_result.json_data["objects"]) > 0
        ):

            polygon_object = canvas_result.json_data["objects"][-1]

            raw_points = []

            if "path" in polygon_object:

                for command in polygon_object["path"]:

                    if command[0] in ("M", "L"):

                        raw_points.append((command[1], command[2]))

            elif "points" in polygon_object:

                left = polygon_object.get("left", 0)
                top = polygon_object.get("top", 0)

                for point in polygon_object["points"]:

                    raw_points.append(
                        (
                            left + point["x"],
                            top + point["y"]
                        )
                    )

            scale_x = width / canvas_width
            scale_y = height / canvas_height

            scaled_points = []

            for point_x, point_y in raw_points:

                scaled_x = int(point_x * scale_x)
                scaled_y = int(point_y * scale_y)

                scaled_x = max(0, min(scaled_x, width - 1))
                scaled_y = max(0, min(scaled_y, height - 1))

                scaled_points.append((scaled_x, scaled_y))

            if len(scaled_points) >= 3:

                roi_points = scaled_points

        if roi_points is None:

            st.info("Mark at least 3 points to define a valid ROI shape.")

            st.stop()

    st.write(f"ROI points: {roi_points}")

    if st.button(
        "Start Processing",
        type="primary"
    ):

        output_path = os.path.join(
            tempfile.gettempdir(),
            "processed_video.mp4"
        )

        csv_path = os.path.join(
            tempfile.gettempdir(),
            "events.csv"
        )

        progress_bar = st.progress(0)

        status_text = st.empty()

        def update_progress(progress, message):

            progress_bar.progress(progress)

            status_text.write(message)

        try:

            result = process_video(
                input_path=input_path,
                output_path=output_path,
                csv_path=csv_path,
                roi_points=roi_points,
                imgsz=imgsz,
                frame_skip_enabled=frame_skip_enabled,
                progress_callback=update_progress
            )

            st.success("Video analysis completed successfully.")

            st.subheader("Analytics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Current Objects",
                    result["current_objects"]
                )

            with col2:
                st.metric(
                    "Unique Objects",
                    result["unique_objects"]
                )

            with col3:
                st.metric(
                    "Entered",
                    result["entered"]
                )

            with col4:
                st.metric(
                    "Exited",
                    result["exited"]
                )

            st.metric(
                "Average Processing FPS",
                f'{result["average_fps"]:.1f}'
            )

            st.subheader("Processed Video")

            with open(output_path, "rb") as video_file:

                video_bytes = video_file.read()

            st.video(video_bytes)

            st.download_button(
                "Download Processed Video",
                data=video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

            st.subheader("Events CSV")

            if os.path.exists(csv_path):

                with open(csv_path, "rb") as csv_file:

                    csv_bytes = csv_file.read()

                st.download_button(
                    "Download events.csv",
                    data=csv_bytes,
                    file_name="events.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(f"Error: {e}")

        finally:

            if os.path.exists(input_path):

                os.remove(input_path)