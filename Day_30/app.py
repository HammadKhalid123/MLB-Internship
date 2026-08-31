import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from tracking_code import (
    load_model,
    list_sample_videos,
    get_video_metadata,
    process_video
)


st.set_page_config(
    page_title="Smart Object Tracking System",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Smart Object Tracking System")
st.write("Upload a video or pick a sample, choose a tracker, and track objects with unique IDs.")


@st.cache_resource
def get_model():
    return load_model()


st.sidebar.header("Tracking Settings")

tracker = st.sidebar.selectbox(
    "Select Tracker",
    ["bytetrack.yaml", "botsort.yaml"]
)

conf = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("Performance")

resolution_choice = st.sidebar.selectbox(
    "Processing Resolution",
    ["480p (fastest)", "720p (balanced)", "960p", "Original (slowest, best quality)"],
    index=1
)

resolution_map = {
    "480p (fastest)": 480,
    "720p (balanced)": 720,
    "960p": 960,
    "Original (slowest, best quality)": None
}
max_width = resolution_map[resolution_choice]

frame_skip = st.sidebar.slider(
    "Frame Skip (1 = process every frame)",
    min_value=1,
    max_value=5,
    value=1,
    step=1,
    help="Higher values skip frames to speed things up further, at the cost of a "
         "choppier, shorter output video. Leave at 1 for smooth output."
)

st.sidebar.caption(
    "Tip: this app runs on CPU. Lower resolution and higher frame skip = much "
    "faster processing, especially for 4K source videos."
)

st.sidebar.markdown("---")
st.sidebar.caption("Model: YOLOv8n")

source_choice = st.radio(
    "Choose Video Source",
    ["Upload Your Own Video", "Use Sample Video"],
    horizontal=True
)

video_path = None
video_display_source = None

if source_choice == "Upload Your Own Video":
    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        suffix = Path(uploaded_video.name).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_video.read())
        temp_file.close()

        video_path = temp_file.name
        video_display_source = uploaded_video

else:
    sample_videos = list_sample_videos("input_videos")

    if not sample_videos:
        st.warning("No sample videos found in the input_videos folder.")
    else:
        selected_sample = st.selectbox(
            "Select a Sample Video",
            sample_videos,
            format_func=lambda p: p.name
        )
        video_path = str(selected_sample)
        video_display_source = str(selected_sample)


if video_path is not None:
    st.subheader("Input Video")
    st.video(video_display_source)

    metadata = get_video_metadata(video_path)

    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    meta_col1.metric("Total Frames", metadata["total_frames"])
    meta_col2.metric("FPS", f"{metadata['fps']:.1f}")
    meta_col3.metric("Resolution", f"{metadata['width']}x{metadata['height']}")
    meta_col4.metric("Duration (s)", f"{metadata['duration_seconds']:.1f}")

    if metadata["width"] > 1920 and max_width is None:
        st.warning(
            "This is a high-resolution (4K+) video and 'Original' resolution is "
            "selected. Processing on CPU at full resolution can take a very long "
            "time — consider choosing a lower Processing Resolution above."
        )

    if st.button("Start Tracking", type="primary"):
        model = get_model()

        progress_bar = st.progress(0, text="Starting...")

        def update_progress(current, total):
            fraction = current / total if total else 0
            progress_bar.progress(
                min(fraction, 1.0),
                text=f"Processing frame {current} / {total}"
            )

        try:
            with st.spinner("Processing video... Please wait."):
                result = process_video(
                    video_path=video_path,
                    model=model,
                    output_dir="saved_videos",
                    tracker=tracker,
                    conf=conf,
                    progress_callback=update_progress,
                    max_width=max_width,
                    frame_skip=frame_skip
                )
        except Exception as error:
            progress_bar.empty()
            st.error(f"Processing failed: {error}")
            st.stop()

        progress_bar.empty()
        st.success("Video processing completed!")

        st.subheader("Tracking Summary")

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Unique Objects", result["number_of_unique_objects"])
        stat_col2.metric("Frames Processed", result["processed_frames"])
        stat_col3.metric("Resolution", f"{result['frame_width']}x{result['frame_height']}")
        stat_col4.metric("Processing Time (s)", f"{result['processing_time_seconds']:.1f}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Unique Object IDs")
            st.write(result["unique_ids"])

        with col2:
            st.subheader("Objects Per Class")
            if result["class_wise_counts"]:
                class_df = pd.DataFrame(
                    list(result["class_wise_counts"].items()),
                    columns=["Class", "Unique Count"]
                )
                st.dataframe(class_df, hide_index=True, use_container_width=True)
            else:
                st.write("No objects detected.")

        output_path = result["output_path"]

        if Path(output_path).exists():
            st.subheader("Processed Video")
            st.video(output_path)

            with open(output_path, "rb") as video_file:
                st.download_button(
                    label="Download Processed Video",
                    data=video_file,
                    file_name=Path(output_path).name,
                    mime="video/mp4"
                )
else:
    st.info("Upload a video or select a sample video to get started.")