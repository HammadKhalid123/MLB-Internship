import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from image_detection import load_model, process_image
from video_detection import process_video

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* ================================
       MAIN APP
    ================================= */
    .stApp {
    background-color: #0a0a0a;
    color: #f5f5f5;
}

/* ================================
   SIDEBAR
================================ */
section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #262626;
}

/* ================================
   HEADINGS
================================ */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* ================================
   GENERAL TEXT
================================ */
p, span, label {
    color: #e5e5e5;
}

/* ================================
   BUTTON
================================ */
.stButton > button {
    background-color: #ff4b4b;
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #ff6b6b;
    color: #ffffff !important;
}

/* ================================
   DOWNLOAD BUTTON
================================ */
.stDownloadButton > button {
    background-color: #22c55e;
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stDownloadButton > button:hover {
    background-color: #16a34a;
}

/* ================================
   FILE UPLOADER
================================ */
div[data-testid="stFileUploader"] {
    background-color: #141414;
    border: 1px dashed #3a3a3a;
    border-radius: 12px;
    padding: 1rem;
}

/* ================================
   VIDEO
================================ */
div[data-testid="stVideo"] {
    display: flex;
    justify-content: center;
}

div[data-testid="stVideo"] video {
    max-height: 420px;
    width: auto;
    border-radius: 12px;
    border: 1px solid #262626;
}

/* ================================
   IMAGE
================================ */
div[data-testid="stImage"] {
    display: flex;
    justify-content: center;
}

div[data-testid="stImage"] img {
    max-height: 420px;
    width: auto;
    border-radius: 12px;
    border: 1px solid #262626;
}

/* ================================
   TABS
================================ */
div[data-baseweb="tab-list"] {
    background-color: #111111;
    border-radius: 10px;
    padding: 4px;
}

button[data-baseweb="tab"] {
    color: #cccccc !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #ff4b4b;
    color: #ffffff !important;
    border-radius: 8px;
}

/* ================================
   METRIC
================================ */
div[data-testid="stMetric"] {
    background-color: #141414;
    border: 1px solid #262626;
    border-radius: 12px;
    padding: 1rem;
}

/* ================================
   DATAFRAME
================================ */
.stDataFrame {
    background-color: #141414;
}

/* ================================
   HORIZONTAL LINE
================================ */
hr {
    border-color: #262626;
}

/* ================================
   SIDEBAR SELECTBOX LABEL
================================ */
section[data-testid="stSidebar"] .stSelectbox label {
    font-weight: 600;
    color: #e5e5e5 !important;
}

/* ==================================================
   SELECTBOX - SELECTED VALUE
================================================== */

/* Main selectbox container */
div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border-radius: 8px !important;
}

/* Inner selectbox */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Selected text */
div[data-baseweb="select"] span {
    color: #000000 !important;
}

/* Selected value and all text inside select */
div[data-baseweb="select"] * {
    color: #000000 !important;
}

/* ==================================================
   SELECTBOX - DROPDOWN MENU
================================================== */

/* Dropdown popup */
div[data-baseweb="popover"] {
    background-color: #ffffff !important;
}

/* Everything inside dropdown */
div[data-baseweb="popover"] * {
    color: #000000 !important;
}

/* Listbox */
div[role="listbox"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Individual options */
div[role="option"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Option text */
div[role="option"] span {
    color: #000000 !important;
}

/* Option hover */
div[role="option"]:hover {
    background-color: #e5e5e5 !important;
    color: #000000 !important;
}

div[role="option"]:hover span {
    color: #000000 !important;
}

/* Selected option */
div[role="option"][aria-selected="true"] {
    background-color: #d4d4d4 !important;
    color: #000000 !important;
}

div[role="option"][aria-selected="true"] span {
    color: #000000 !important;
}

/* BaseWeb menu */
div[data-baseweb="menu"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* List items */
li[role="option"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

li[role="option"] * {
    color: #000000 !important;
}

/* Hover */
li[role="option"]:hover {
    background-color: #e5e5e5 !important;
    color: #000000 !important;
}

/* Selected */
li[role="option"][aria-selected="true"] {
    background-color: #d4d4d4 !important;
    color: #000000 !important;
}

/* ==================================================
   SELECTBOX INPUT / PLACEHOLDER
================================================== */

div[data-baseweb="select"] input {
    color: #000000 !important;
    background-color: #ffffff !important;
}

div[data-baseweb="select"] input::placeholder {
    color: #000000 !important;
}

/* Dropdown arrow */
div[data-baseweb="select"] svg {
    fill: #000000 !important;
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

BASE_DIR = Path(__file__).resolve().parent

SAMPLE_IMAGES_DIR = BASE_DIR / "input_images"
SAMPLE_VIDEOS_DIR = BASE_DIR / "input_videos"

NO_SAMPLE_LABEL = "— None —"


@st.cache_resource
def get_model():
    return load_model()


def list_sample_files(directory, extensions):
    if not directory.exists():
        return []

    return sorted(
        [item for item in directory.iterdir() if item.suffix.lower() in extensions],
        key=lambda item: item.name.lower(),
    )


def detections_to_dataframe(detections):
    if not detections:
        return pd.DataFrame(
            columns=["Class", "Confidence (%)", "X1", "Y1", "X2", "Y2"]
        )

    rows = []
    for item in detections:
        rows.append(
            {
                "Class": item["class_name"],
                "Confidence (%)": round(item["confidence_percentage"], 2),
                "X1": round(item["x1"], 1),
                "Y1": round(item["y1"], 1),
                "X2": round(item["x2"], 1),
                "Y2": round(item["y2"], 1),
            }
        )

    return pd.DataFrame(rows)


sample_images = list_sample_files(SAMPLE_IMAGES_DIR, IMAGE_EXTENSIONS)
sample_videos = list_sample_files(SAMPLE_VIDEOS_DIR, VIDEO_EXTENSIONS)

with st.sidebar:
    st.header("📂 Sample Files")
    st.caption("Test the app instantly using files already in the project folders.")

    st.subheader("Sample Images")
    if sample_images:
        selected_sample_image_name = st.selectbox(
            "Choose a sample image",
            options=[NO_SAMPLE_LABEL] + [item.name for item in sample_images],
            key="sample_image_select",
        )
    else:
        selected_sample_image_name = NO_SAMPLE_LABEL
        st.info(f"No images found in '{SAMPLE_IMAGES_DIR}/'.")

    st.divider()

    st.subheader("Sample Videos")
    if sample_videos:
        selected_sample_video_name = st.selectbox(
            "Choose a sample video",
            options=[NO_SAMPLE_LABEL] + [item.name for item in sample_videos],
            key="sample_video_select",
        )
    else:
        selected_sample_video_name = NO_SAMPLE_LABEL
        st.info(f"No videos found in '{SAMPLE_VIDEOS_DIR}/'.")


st.title("🎯 YOLO Object Detection")
st.write("Upload an image or a video, or pick a sample from the sidebar, and detect objects with bounding boxes, class names, and confidence scores.")

st.divider()

model = get_model()

tab_image, tab_video = st.tabs(["🖼️ Image", "🎬 Video"])

with tab_image:
    conf_image = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="conf_image",
    )

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader",
    )

    image_source_path = None

    if uploaded_image is not None:
        suffix = Path(uploaded_image.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_image.getbuffer())
            image_source_path = temp_file.name
    elif selected_sample_image_name != NO_SAMPLE_LABEL:
        image_source_path = str(SAMPLE_IMAGES_DIR / selected_sample_image_name)
        st.caption(f"Using sample: {selected_sample_image_name}")

    if image_source_path is not None:
        run_detection = st.button("Run Detection", key="run_image")

        if run_detection:
            with st.spinner("Detecting objects..."):
                output = process_image(image_source_path, model=model, conf=conf_image)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original")
                st.image(Image.open(image_source_path))

            with col2:
                st.subheader("Detected")
                st.image(output["annotated_image"][:, :, ::-1])

            st.divider()

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Objects Detected", output["number_of_detections"])
            metric_col2.metric("Image Width", output["frame_width"])
            metric_col3.metric("Image Height", output["frame_height"])

            st.subheader("Detections")
            st.dataframe(
                detections_to_dataframe(output["detections"]),
                use_container_width=True,
            )

            with open(output["output_path"], "rb") as file:
                st.download_button(
                    "Download Processed Image",
                    data=file.read(),
                    file_name=Path(output["output_path"]).name,
                    mime="image/jpeg",
                )
    else:
        st.info("Upload an image or pick a sample image from the sidebar to get started.")

with tab_video:
    conf_video = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="conf_video",
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    video_source_path = None

    if uploaded_video is not None:
        suffix = Path(uploaded_video.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_video.getbuffer())
            video_source_path = temp_file.name
    elif selected_sample_video_name != NO_SAMPLE_LABEL:
        video_source_path = str(SAMPLE_VIDEOS_DIR / selected_sample_video_name)
        st.caption(f"Using sample: {selected_sample_video_name}")

    if video_source_path is not None:
        preview_col1, preview_col2, preview_col3 = st.columns([1, 2, 1])
        with preview_col2:
            st.video(video_source_path)

        run_detection_video = st.button("Run Detection", key="run_video")

        if run_detection_video:
            progress_bar = st.progress(0, text="Processing video...")

            def update_progress(current_frame, total_frames):
                fraction = min(current_frame / total_frames, 1.0)
                progress_bar.progress(
                    fraction, text=f"Processing frame {current_frame}/{total_frames}"
                )

            with st.spinner("Detecting objects in video..."):
                output = process_video(
                    video_source_path,
                    model=model,
                    conf=conf_video,
                    progress_callback=update_progress,
                )

            progress_bar.empty()

            st.subheader("Processed Video")

            with open(output["output_path"], "rb") as video_file:
                video_bytes = video_file.read()

            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            with result_col2:
                st.video(video_bytes, format="video/mp4")

            st.divider()

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Total Detections", output["total_detections"])
            metric_col2.metric("Processed Frames", output["processed_frames"])
            metric_col3.metric("FPS", round(output["fps"], 2))

            st.subheader("Detections")
            st.dataframe(
                detections_to_dataframe(output["detections"]),
                use_container_width=True,
            )

            st.download_button(
                "Download Processed Video",
                data=video_bytes,
                file_name=Path(output["output_path"]).name,
                mime="video/mp4",
            )
    else:
        st.info("Upload a video or pick a sample video from the sidebar to get started.")