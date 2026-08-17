import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

from source_code import (
    to_grayscale,
    gaussian_blur,
    canny_edges,
    rotate_image,
    enhance_image,
    binary_threshold,
    detect_shapes,
    sharpen_image,
    flip_image,
)

st.set_page_config(page_title="Image Processing Studio", page_icon="🖼️", layout="wide")

st.title("🖼️ Image Processing Studio")
st.write("Upload an image, choose a processing filter, and download the result.")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

    filter_option = st.selectbox(
        "Choose a filter",
        [
            "Original",
            "Grayscale",
            "Gaussian Blur",
            "Canny Edge Detection",
            "Rotate",
            "Enhance (Brightness/Contrast)",
            "Binary Threshold",
            "Shape Detection",
            "Sharpen",
            "Flip",
        ],
    )

    params = {}

    if filter_option == "Gaussian Blur":
        k = st.slider("Kernel size (odd number)", 1, 25, 5, step=2)
        params["ksize"] = (k, k)
        params["sigma"] = st.slider("Sigma", 0, 10, 0)

    elif filter_option == "Canny Edge Detection":
        params["threshold1"] = st.slider("Threshold 1", 0, 500, 100)
        params["threshold2"] = st.slider("Threshold 2", 0, 500, 200)

    elif filter_option == "Rotate":
        params["angle"] = st.slider("Angle", -180, 180, 45)
        params["scale"] = st.slider("Scale", 0.1, 3.0, 1.0)

    elif filter_option == "Enhance (Brightness/Contrast)":
        params["alpha"] = st.slider("Contrast (alpha)", 0.1, 3.0, 1.5)
        params["beta"] = st.slider("Brightness (beta)", -100, 100, 20)

    elif filter_option == "Binary Threshold":
        params["thresh"] = st.slider("Threshold value", 0, 255, 127)
        params["maxval"] = st.slider("Max value", 0, 255, 255)

    elif filter_option == "Shape Detection":
        params["thresh"] = st.slider("Threshold value", 0, 255, 127)
        params["epsilon_factor"] = st.slider("Approximation factor", 0.01, 0.10, 0.04)

    elif filter_option == "Flip":
        flip_choice = st.radio("Flip direction", ["Horizontal", "Vertical", "Both"])
        flip_map = {"Horizontal": 1, "Vertical": 0, "Both": -1}
        params["flip_code"] = flip_map[flip_choice]


def apply_filter(img, option, params):
    if option == "Original":
        return img
    elif option == "Grayscale":
        return to_grayscale(img)
    elif option == "Gaussian Blur":
        return gaussian_blur(img, params["ksize"], params["sigma"])
    elif option == "Canny Edge Detection":
        return canny_edges(img, params["threshold1"], params["threshold2"])
    elif option == "Rotate":
        return rotate_image(img, params["angle"], params["scale"])
    elif option == "Enhance (Brightness/Contrast)":
        return enhance_image(img, params["alpha"], params["beta"])
    elif option == "Binary Threshold":
        return binary_threshold(img, params["thresh"], params["maxval"])
    elif option == "Shape Detection":
        shapes, contours, hierarchy = detect_shapes(
            img, params["thresh"], 255, params["epsilon_factor"]
        )
        output = img.copy()
        for shape_name, contour, approx in shapes:
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(
                output, shape_name, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        return output
    elif option == "Sharpen":
        return sharpen_image(img)
    elif option == "Flip":
        return flip_image(img, params["flip_code"])
    return img


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    processed_img = apply_filter(original_img, filter_option, params)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        if len(processed_img.shape) == 2:
            st.image(processed_img, use_container_width=True, clamp=True)
        else:
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    if len(processed_img.shape) == 2:
        result_pil = Image.fromarray(processed_img)
    else:
        result_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))

    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    byte_data = buf.getvalue()

    st.download_button(
        label="⬇️ Download Processed Image",
        data=byte_data,
        file_name="processed_image.png",
        mime="image/png",
    )

else:
    st.info("Please upload an image from the sidebar to get started.")