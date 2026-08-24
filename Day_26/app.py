import streamlit as st
import numpy as np
import cv2 as cv
from segmentation_code import (
    Processing,
    Binary,
    Adaptive_Threshold,
    Otsu_Threshold,
    compare_results,
    foreground,
    background
)

st.set_page_config(page_title="Image Segmentation App", layout="wide")

st.title("🖼️ Image Segmentation & Thresholding")
st.write("Upload an image, apply thresholding techniques, and segment the foreground from the background.")

st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp"])

methods = st.sidebar.multiselect(
    "Select Thresholding Methods",
    ["Binary", "Adaptive", "Otsu"],
    default=["Binary", "Adaptive", "Otsu"]
)

show_comparison = st.sidebar.checkbox("Show Comparison Strip", value=True)
show_segmentation = st.sidebar.checkbox("Show Foreground/Background", value=True)

best_method = st.sidebar.selectbox(
    "Select Best Result to Save",
    methods if methods else ["Binary", "Adaptive", "Otsu"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

    gray = Processing(image)

    results = {}
    if "Binary" in methods:
        results["Binary"] = Binary(gray)
    if "Adaptive" in methods:
        results["Adaptive"] = Adaptive_Threshold(gray)
    if "Otsu" in methods:
        results["Otsu"] = Otsu_Threshold(gray)

    st.subheader("Original Image")
    left, mid, right = st.columns([1, 4, 1])
    with mid:
        h, w = image.shape[:2]
        target_h = 600
        target_w = int(w * (target_h / h))
        resized_original = cv.resize(image, (target_w, target_h))
        st.image(cv.cvtColor(resized_original, cv.COLOR_BGR2RGB))

    if results:
        st.subheader("Thresholding Results")
        cols = st.columns(len(results))
        for col, (name, res) in zip(cols, results.items()):
            with col:
                st.image(res, caption=name, use_container_width=True)

    if show_comparison and "Binary" in results and "Adaptive" in results and "Otsu" in results:
        st.subheader("Comparison")
        comparison = compare_results(image, gray, results["Binary"], results["Adaptive"], results["Otsu"])
        st.image(cv.cvtColor(comparison, cv.COLOR_BGR2RGB), use_container_width=True)

    if show_segmentation and results:
        st.subheader("Foreground / Background Segmentation")
        seg_w = 650
        h, w = image.shape[:2]
        seg_h = int(h * (seg_w / w))
        for name, mask in results.items():
            fg = cv.resize(foreground(image, mask), (seg_w, seg_h))
            bg = cv.resize(background(image, mask), (seg_w, seg_h))
            st.markdown(f"**{name} Mask**")
            c1, c2 = st.columns(2, gap="small")
            with c1:
                st.image(cv.cvtColor(fg, cv.COLOR_BGR2RGB), caption=f"{name} - Foreground", use_container_width=True)
            with c2:
                st.image(cv.cvtColor(bg, cv.COLOR_BGR2RGB), caption=f"{name} - Background", use_container_width=True)

    if results and best_method in results:
        st.subheader("Save Best Result")
        best_result = results[best_method]
        is_success, buffer = cv.imencode(".png", best_result)
        if is_success:
            st.download_button(
                label=f"Download {best_method} Result",
                data=buffer.tobytes(),
                file_name=f"best_result_{best_method.lower()}.png",
                mime="image/png"
            )
else:
    st.info("Please upload an image to get started.")