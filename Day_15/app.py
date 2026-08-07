import streamlit as st
import tempfile
import os
import cv2
from PIL import Image

from mini_project import (
    load_model,
    detect_image,
    detect_video,
    analyze_result,
    plot_class_distribution,
    plot_confidence_distribution,
    plot_class_percentage
)

st.set_page_config(page_title="Object Detection App", page_icon="🎯", layout="wide")

st.title("🎯 YOLO Object Detection")
st.write("Upload an image or video to run detection and download the results.")


@st.cache_resource
def get_model():
    return load_model("best.pt")


model = get_model()

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.write("Upload an image or video below to get started.")

file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if file is not None:
    file_type = file.type.split("/")[0]

    if file_type == "image":
        image = Image.open(file).convert("RGB")
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        image.save(temp_path)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with st.spinner("Running detection..."):
            result = detect_image(model, temp_path, conf=conf)
            annotated = result.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Detected Image")
            st.image(annotated, use_container_width=True)

        stats = analyze_result(result, model)

        st.markdown("---")
        st.subheader("Detection Summary")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Objects", stats["total_objects"])
        m2.metric("Classes Detected", len(stats["class_count"]))
        avg_conf = (
            sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
            if stats["confidence_scores"] else 0
        )
        m3.metric("Avg Confidence", f"{avg_conf:.2f}")

        if stats["detections"]:
            st.subheader("Detections")
            st.dataframe(stats["detections"], use_container_width=True)

            plot_class_distribution(stats["class_count"], "Graphs/class_distribution.png")
            plot_confidence_distribution(stats["confidence_scores"], "Graphs/confidence_distribution.png")
            plot_class_percentage(stats["class_count"], "Graphs/class_percentage.png")

            g1, g2, g3 = st.columns(3)
            with g1:
                st.image("Graphs/class_distribution.png")
            with g2:
                st.image("Graphs/confidence_distribution.png")
            with g3:
                st.image("Graphs/class_percentage.png")
        else:
            st.warning("No objects detected in this image.")

        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(tempfile.gettempdir(), "detected_" + file.name)
        cv2.imwrite(out_path, annotated_bgr)

        with open(out_path, "rb") as f:
            st.download_button("Download Detected Image", f, file_name="detected_" + file.name)

    elif file_type == "video":
        temp_input = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_input, "wb") as f:
            f.write(file.read())

        temp_output = os.path.join(tempfile.gettempdir(), "detected_" + file.name)

        st.subheader("Original Video")
        st.video(temp_input)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for frame_count, total_frames in detect_video(model, temp_input, temp_output, conf=conf):
            progress = frame_count / total_frames if total_frames else 0
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing frame {frame_count}/{total_frames}")

        status_text.text("Detection complete!")

        st.subheader("Detected Video")
        st.video(temp_output)

        with open(temp_output, "rb") as f:
            st.download_button("Download Detected Video", f, file_name="detected_" + file.name)

else:
    st.info("Please upload an image or video to begin.")