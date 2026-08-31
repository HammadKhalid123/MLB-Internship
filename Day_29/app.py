import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import subprocess
import imageio_ffmpeg
from pathlib import Path

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Football Player & Ball Detection",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Player & Ball Detection App")
st.write("Upload an image or video. The trained YOLO model will detect **players** and the **ball** with confidence scores.")

# -----------------------------
# Load Model (cached so it loads only once)
# -----------------------------
# Path resolve app.py ki apni location ke relative hoti hai (chahe Streamlit
# Cloud repo root se run kare ya tum apne PC pe Day_29 folder se run karo)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "football_player_v2" / "weights" / "best.pt")

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}. Please check the path.")
        st.stop()
    return YOLO(model_path)

model = load_model(MODEL_PATH)

# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.write("**Classes:**")
st.sidebar.write("🟢 Player")
st.sidebar.write("🔴 Ball")

# -----------------------------
# File Upload
# -----------------------------
file_type = st.radio("Select File Type", ["Image", "Video"], horizontal=True)

if file_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with st.spinner("Running detection..."):
            results = model.predict(
                source=np.array(image),
                conf=conf_threshold,
                iou=iou_threshold
            )

        result = results[0]
        result_img = result.plot()  # numpy array (BGR)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Detected Image")
            st.image(result_img_rgb, use_container_width=True)

        # Show detection details
        st.subheader("📊 Detection Results")
        if len(result.boxes) == 0:
            st.warning("No objects detected. Try lowering the confidence threshold.")
        else:
            names = model.names
            detection_data = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                detection_data.append({
                    "Class": names[cls_id],
                    "Confidence": f"{conf:.2%}"
                })
            st.table(detection_data)

        # Save & download result
        save_path = "prediction_result.jpg"
        Image.fromarray(result_img_rgb).save(save_path)

        with open(save_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Result Image",
                data=f,
                file_name="prediction_result.jpg",
                mime="image/jpeg"
            )

elif file_type == "Video":
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        input_video_path = tfile.name

        st.video(input_video_path)

        if st.button("🚀 Run Detection on Video"):
            cap = cv2.VideoCapture(input_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            raw_output_path = "prediction_result_raw.mp4"
            output_path = "prediction_result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            status_text = st.empty()

            frame_count = 0
            with st.spinner("Processing video, please wait..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(
                        source=frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)

                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")

            cap.release()
            out.release()

            # Step 1: get the ffmpeg binary (downloads on first run only - show separate status)
            with st.spinner("Preparing ffmpeg (first run may download ~25MB, please wait)..."):
                try:
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception as e:
                    st.error(f"Could not get ffmpeg binary: {e}")
                    ffmpeg_exe = None

            # Step 2: re-encode with ffmpeg to H.264 so it plays in the browser
            if ffmpeg_exe:
                with st.spinner("Finalizing video (encoding for browser playback)..."):
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    try:
                        result_proc = subprocess.run(
                            [
                                ffmpeg_exe, "-y",
                                "-i", raw_output_path,
                                "-vcodec", "libx264",
                                "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart",
                                output_path
                            ],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        if result_proc.returncode != 0:
                            st.error("⚠️ ffmpeg encoding failed:")
                            st.code(result_proc.stderr[-1500:])
                            output_path = raw_output_path
                        else:
                            os.remove(raw_output_path)
                    except subprocess.TimeoutExpired:
                        st.error("⚠️ ffmpeg took too long and was stopped. Showing raw video instead.")
                        output_path = raw_output_path
                    except Exception as e:
                        st.error(f"⚠️ Video re-encoding failed: {e}")
                        output_path = raw_output_path
            else:
                output_path = raw_output_path

            st.success("✅ Video processing completed!")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Result Video",
                    data=f,
                    file_name="prediction_result.mp4",
                    mime="video/mp4"
                )

st.markdown("---")
st.caption("Built with Streamlit + Ultralytics YOLO | Day 29 - Custom Object Detection Project")