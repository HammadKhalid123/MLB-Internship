import streamlit as st
import cv2
import tempfile
import os
from video_processing import process_video_file
from webcam_processing import process_webcam

st.set_page_config(page_title="Video Edge Detection Studio", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    section[data-testid="stSidebar"] {
        background-color: #f7f8fa;
        border-right: 1px solid #e2e5e9;
    }
    h1, h2, h3, h4 {
        color: #1a1a1a;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: #ffffff;
    }
    .stDownloadButton>button {
        background-color: #16a34a;
        color: #ffffff;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #2563eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "stop_processing" not in st.session_state:
    st.session_state.stop_processing = False

if "output_ready_path" not in st.session_state:
    st.session_state.output_ready_path = None

if "output_ready_name" not in st.session_state:
    st.session_state.output_ready_name = None

if "output_ready_bytes" not in st.session_state:
    st.session_state.output_ready_bytes = None

st.sidebar.title("⚙️ Controls")

source = st.sidebar.radio("Select Source", ["Upload Video", "Webcam"])

st.sidebar.subheader("Processing Settings")

blur_kernel = st.sidebar.slider("Gaussian Blur Kernel Size", 1, 31, 5, step=2)
canny_low = st.sidebar.slider("Canny Lower Threshold", 0, 255, 100)
canny_high = st.sidebar.slider("Canny Upper Threshold", 0, 255, 200)

st.sidebar.subheader("Output Settings")

output_filename = st.sidebar.text_input("Output File Name", "output.mp4")

# Ensure the output filename always has a valid video extension.
# Without this, VideoWriter (inside video_processing.py / webcam_processing.py)
# can silently fail to write a playable file, which is a common reason the
# download button either doesn't appear or produces a broken video.
valid_extensions = (".mp4", ".avi", ".mov", ".mkv")
if not output_filename.lower().endswith(valid_extensions):
    output_filename = output_filename.strip() or "output"
    output_filename += ".mp4"

uploaded_file = None
camera_index = 0

if source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
else:
    camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)

st.sidebar.subheader("Actions")

col_start, col_stop = st.sidebar.columns(2)
start_button = col_start.button("▶ Start")
stop_button = col_stop.button("■ Stop")

if stop_button:
    st.session_state.stop_processing = True

st.title("🎥 Video Edge Detection Studio")
st.write("Convert frames to grayscale, apply Gaussian Blur, and run Canny Edge Detection in real time.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Frame")
    original_placeholder = st.empty()

with col2:
    st.subheader("Processed Frame")
    processed_placeholder = st.empty()

status_col1, status_col2 = st.columns([3, 1])

with status_col1:
    progress_placeholder = st.empty()

with status_col2:
    info_placeholder = st.empty()


def stop_flag():
    return st.session_state.stop_processing


def update_frames(original, processed, frame_count, total_frames):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    original_placeholder.image(original_rgb, use_container_width=True)
    processed_placeholder.image(processed_rgb, use_container_width=True)

    info_placeholder.metric("Frame", frame_count)

    if total_frames > 0:
        progress_placeholder.progress(min(frame_count / total_frames, 1.0))
    else:
        progress_placeholder.progress(0)


def finalize_output(output_path, display_name):
    """
    Validate that the processed file actually exists and is non-empty,
    then load it into memory as bytes and store it in session_state.

    Loading the bytes into session_state (instead of re-opening the file
    handle on every rerun) avoids file-lock issues and guarantees the
    download button has stable data to serve, which is the main reason
    the download button was failing to show/work correctly before.
    """
    if not output_path or not os.path.exists(output_path):
        st.error("Processing finished, but the output file was not found on disk.")
        st.session_state.output_ready_path = None
        st.session_state.output_ready_bytes = None
        st.session_state.output_ready_name = None
        return

    if os.path.getsize(output_path) == 0:
        st.error(
            "Processing finished, but the output file is empty (0 bytes). "
            "Check that the video writer inside video_processing.py / "
            "webcam_processing.py is being released/closed properly."
        )
        st.session_state.output_ready_path = None
        st.session_state.output_ready_bytes = None
        st.session_state.output_ready_name = None
        return

    with open(output_path, "rb") as f:
        video_bytes = f.read()

    st.session_state.output_ready_path = output_path
    st.session_state.output_ready_bytes = video_bytes
    st.session_state.output_ready_name = display_name

    st.success(f"Processing complete. Saved as {display_name}")


if start_button:
    st.session_state.stop_processing = False
    st.session_state.output_ready_path = None
    st.session_state.output_ready_bytes = None
    st.session_state.output_ready_name = None

    if source == "Upload Video":
        if uploaded_file is None:
            st.warning("Please upload a video file first.")
        else:
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(uploaded_file.read())
            temp_input.close()

            output_path = os.path.join(tempfile.gettempdir(), output_filename)

            with st.spinner("Processing video..."):
                process_video_file(
                    temp_input.name,
                    output_path,
                    blur_kernel,
                    canny_low,
                    canny_high,
                    frame_callback=update_frames,
                    stop_flag=stop_flag,
                )

            os.unlink(temp_input.name)

            finalize_output(output_path, output_filename)

    else:
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        with st.spinner("Processing webcam feed..."):
            process_webcam(
                output_path,
                blur_kernel,
                canny_low,
                canny_high,
                frame_callback=update_frames,
                stop_flag=stop_flag,
                camera_index=int(camera_index),
            )

        finalize_output(output_path, output_filename)

st.divider()

if st.session_state.output_ready_bytes:
    st.download_button(
        "⬇ Download Processed Video",
        data=st.session_state.output_ready_bytes,
        file_name=st.session_state.output_ready_name,
        mime="video/mp4",
    )