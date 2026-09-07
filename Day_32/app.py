import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import imageio
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO
from tracking_code import (
    sahi_detection,
    draw_detections,
    check_parking_occupancy,
    draw_parking_spots,
    draw_statistics_panel
)

st.set_page_config(
    page_title="Smart Parking Monitoring System",
    page_icon="🅿️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("yolov8n-visdrone.pt")

def build_marking_preview(image_rgb, saved_spots, current_points):
    preview = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    for idx, spot in enumerate(saved_spots):
        pts = np.array(spot, dtype=np.int32)
        cv2.polylines(preview, [pts], True, (0, 255, 0), 2)
        cv2.putText(preview, str(idx + 1), tuple(pts[0]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if len(current_points) > 0:
        pts = np.array(current_points, dtype=np.int32)
        for point in current_points:
            cv2.circle(preview, point, 5, (0, 0, 255), -1)
        if len(current_points) > 1:
            cv2.polylines(preview, [pts], False, (0, 0, 255), 2)
        if len(current_points) > 2:
            cv2.line(preview, current_points[-1], current_points[0], (0, 165, 255), 1)

    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

def process_video(input_path, output_path, spots, model, mode, progress_bar, stats_placeholder, frame_skip, slice_size, overlap_ratio):
    video = cv2.VideoCapture(input_path)
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None
    )

    occupancy_history = []
    frame_index = 0

    last_detections = []
    last_occupied_flags = [False] * len(spots)

    while True:
        success, frame = video.read()
        if not success:
            break

        if frame_index % frame_skip == 0:
            detections = sahi_detection(
                image=frame,
                model=model,
                slice_width=slice_size,
                slice_height=slice_size,
                overlap_width_ratio=overlap_ratio,
                overlap_height_ratio=overlap_ratio,
                confidence_threshold=0.10,
                iou_threshold=0.3
            )
            occupied_flags = check_parking_occupancy(detections, spots)
            last_detections = detections
            last_occupied_flags = occupied_flags
        else:
            detections = last_detections
            occupied_flags = last_occupied_flags

        total_spaces = len(spots)
        occupied_spaces = sum(occupied_flags)
        available_spaces = total_spaces - occupied_spaces
        occupancy_percentage = (occupied_spaces / total_spaces * 100) if total_spaces > 0 else 0

        occupancy_history.append(occupancy_percentage)

        output_frame = draw_parking_spots(frame, spots, occupied_flags)

        output_frame = draw_detections(output_frame, detections, model)

        output_frame = draw_statistics_panel(
            output_frame,
            total_spaces,
            occupied_spaces,
            available_spaces,
            occupancy_percentage,
            mode
        )

        writer.append_data(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))

        frame_index += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_index / total_frames, 1.0))

        stats_placeholder.markdown(
            f"""
            <div style='display: flex; gap: 20px; padding: 10px;'>
                <span style='font-size: 14px;'><b>Total Spaces:</b> {total_spaces}</span>
                <span style='font-size: 14px;'><b>Occupied:</b> {occupied_spaces}</span>
                <span style='font-size: 14px;'><b>Available:</b> {available_spaces}</span>
                <span style='font-size: 14px;'><b>Occupancy:</b> {occupancy_percentage:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    video.release()
    writer.close()
    return occupancy_history

def main():
    st.title("Smart Parking Monitoring System")
    st.markdown("---")

    if "spots" not in st.session_state:
        st.session_state.spots = []
    if "first_frame" not in st.session_state:
        st.session_state.first_frame = None
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
    if "uploaded_file_key" not in st.session_state:
        st.session_state.uploaded_file_key = None

    with st.sidebar:
        st.header("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a parking lot video",
            type=["mp4", "avi", "mov", "mkv"]
        )

        if uploaded_file is not None:
            current_file_key = (uploaded_file.name, uploaded_file.size)

            if st.session_state.uploaded_file_key != current_file_key:
                st.session_state.uploaded_file_key = current_file_key
                st.session_state.spots = []
                st.session_state.canvas_key += 1
                st.session_state.first_frame = None
                st.session_state.video_path = None

                temp_dir = tempfile.mkdtemp()
                video_path = os.path.join(temp_dir, uploaded_file.name)
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.session_state.video_path = video_path

                video = cv2.VideoCapture(video_path)
                success, frame = video.read()
                video.release()
                if success:
                    st.session_state.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.success("✅ Video uploaded successfully!")

    if st.session_state.first_frame is not None:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Mark Parking Spaces")
            st.info("Mark the corner points for one parking spot, then click 'Save Spot' before marking the next spot.")
            image = Image.fromarray(st.session_state.first_frame)
            canvas_width = 800
            scale = canvas_width / image.width
            canvas_height = int(image.height * scale)
            resized_image = image.resize((canvas_width, canvas_height))

            canvas_result = st_canvas(
                fill_color="rgba(255,165,0,0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=resized_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="point",
                point_display_radius=5,
                key=f"canvas_{st.session_state.canvas_key}"
            )

            current_points = []
            if canvas_result.json_data is not None:
                for obj in canvas_result.json_data["objects"]:
                    x = obj["left"] / scale
                    y = obj["top"] / scale
                    current_points.append((int(x), int(y)))

        with col2:
            st.subheader("Parking Spots")
            st.write(f"**Total spots marked:** {len(st.session_state.spots)}")

            preview = build_marking_preview(
                st.session_state.first_frame,
                st.session_state.spots,
                current_points
            )
            st.image(preview, caption="Preview", use_container_width=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("Save Spot", use_container_width=True):
                    if len(current_points) >= 3:
                        st.session_state.spots.append(current_points)
                        st.session_state.canvas_key += 1
                        st.rerun()
                    else:
                        st.warning("Need at least 3 points!")

            with col_b:
                if st.button("Undo", use_container_width=True):
                    if st.session_state.spots:
                        st.session_state.spots.pop()
                        st.rerun()

            with col_c:
                if st.button("Clear All", use_container_width=True):
                    st.session_state.spots = []
                    st.session_state.canvas_key += 1
                    st.rerun()

        st.markdown("---")
        st.subheader("Run Analysis")

        mode = st.radio(
            "Select Mode",
            ["Parking Monitor", "Parking Analytics"],
            horizontal=True,
            help="Parking Monitor: Shows vehicle detections | Parking Analytics: Shows occupancy graph"
        )

        with st.expander("⚙️ Performance Settings"):
            frame_skip = st.slider(
                "Frame Skip",
                min_value=1,
                max_value=10,
                value=3,
                help="Har N-wan frame pe hi detection chalega, beech ke frames pe pichla result reuse hoga. Zyada value = fast lekin kam smooth."
            )
            slice_size = st.select_slider(
                "Slice Size",
                options=[320, 416, 512, 640, 800],
                value=640,
                help="Chota slice size = zyada accuracy lekin slow. Bara slice size = fast lekin thori kam accuracy small objects ke liye."
            )
            overlap_ratio = st.slider(
                "Slice Overlap Ratio",
                min_value=0.0,
                max_value=0.6,
                value=0.2,
                step=0.05,
                help="Zyada overlap = zyada accuracy lekin slow. Kam overlap = fast."
            )

        if st.button("Start Analysis", type="primary"):
            if len(st.session_state.spots) == 0:
                st.warning("Please mark at least one parking spot!")
            else:
                model = load_model()
                output_dir = "output_videos"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "parking_output.mp4")

                progress_bar = st.progress(0)
                stats_placeholder = st.empty()

                with st.spinner("Processing video..."):
                    occupancy_history = process_video(
                        input_path=st.session_state.video_path,
                        output_path=output_path,
                        spots=st.session_state.spots,
                        model=model,
                        mode=mode,
                        progress_bar=progress_bar,
                        stats_placeholder=stats_placeholder,
                        frame_skip=frame_skip,
                        slice_size=slice_size,
                        overlap_ratio=overlap_ratio
                    )

                st.success("✅ Processing complete!")

                total_spaces = len(st.session_state.spots)
                final_occupancy = occupancy_history[-1] if occupancy_history else 0
                occupied_spaces = int(round(final_occupancy / 100 * total_spaces)) if total_spaces > 0 else 0
                available_spaces = total_spaces - occupied_spaces

                st.markdown("---")
                st.subheader("📊 Final Parking Statistics")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Spaces", total_spaces, delta=None)
                col2.metric("Occupied", occupied_spaces, delta_color="inverse")
                col3.metric("Available", available_spaces)
                col4.metric("Occupancy", f"{final_occupancy:.1f}%")

                if mode == "Parking Analytics" and occupancy_history:
                    st.subheader("📈 Occupancy Over Time")
                    st.line_chart(occupancy_history)

                st.subheader("🎥 Processed Video")
                with open(output_path, "rb") as video_file:
                    video_bytes = video_file.read()

                st.video(video_bytes)

                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name="parking_output.mp4",
                    mime="video/mp4",
                    type="primary"
                )

    else:
        st.info("Upload a parking lot video from the sidebar to get started!")

if __name__ == "__main__":
    main()