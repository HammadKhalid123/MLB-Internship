import streamlit as st
import cv2 as cv
import numpy as np
import tempfile
import os
from feature_detection import harris_corner_detection, orb_keypoint_detection
from feature_matching import orb_knn_matching, orb_bruteforce_matching

st.set_page_config(page_title="Image Feature Matching System", layout="wide")
st.markdown("""
<style>
h1 {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    padding-bottom: 10px;
}
h2, h3 {
    color: #2575fc;
    border-bottom: 2px solid #2575fc33;
    padding-bottom: 6px;
}
div[data-testid="stMetric"] {
    background: #f0f4ff;
    border: 1px solid #2575fc55;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(37, 117, 252, 0.15);
}
div[data-testid="stMetric"] label {
    color: #2575fc !important;
}
div[data-testid="stFileUploader"] {
    border: 2px dashed #2575fc88;
    border-radius: 12px;
    padding: 10px;
}
.stButton>button {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px 25px;
    border: none;
}
.stDownloadButton>button {
    background-color: white;
    color: #2575fc;
    border: 1px solid #2575fc;
    border-radius: 8px;
}
div[data-testid="stRadio"] {
    background: #f0f4ff;
    padding: 10px 15px;
    border-radius: 10px;
    border: 1px solid #2575fc33;
}
</style>
""", unsafe_allow_html=True)

st.title("Image Feature Matching System")
st.caption("Compare two images using ORB and Harris Corner detection with KNN / Brute-Force matching")

OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_WIDTH = 550

col1, col2 = st.columns(2, gap="small")
with col1:
    uploaded_file1 = st.file_uploader("Upload First Image", type=["png", "jpg", "jpeg"], key="img1")
with col2:
    uploaded_file2 = st.file_uploader("Upload Second Image", type=["png", "jpg", "jpeg"], key="img2")

matching_method = st.radio("Select Matching Method", ["KNN + Ratio Test", "Brute Force (crossCheck)"], horizontal=True)

def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.close()
    return temp_file.name

def bgr_to_rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def show_image_with_download(img_bgr, caption, filename, width=IMG_WIDTH):
    st.image(bgr_to_rgb(img_bgr), caption=caption, width=width)
    success, buffer = cv.imencode(".png", img_bgr)
    if success:
        st.download_button(
            label="Download",
            data=buffer.tobytes(),
            file_name=filename,
            mime="image/png",
            key=filename
        )

if uploaded_file1 is not None and uploaded_file2 is not None:
    path1 = save_uploaded_file(uploaded_file1)
    path2 = save_uploaded_file(uploaded_file2)
    st.subheader("Uploaded Images")
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.image(uploaded_file1, caption="Image 1", width=IMG_WIDTH)
    with col2:
        st.image(uploaded_file2, caption="Image 2", width=IMG_WIDTH)
    if st.button("Run Feature Matching"):
        with st.spinner("Detecting keypoints and matching features..."):
            if matching_method == "KNN + Ratio Test":
                img_matches, keypoints1, keypoints2, descriptors1, descriptors2, good_matches = orb_knn_matching(path1, path2)
                matches_filename = "matches_knn.png"
            else:
                img_matches, keypoints1, keypoints2, descriptors1, descriptors2, good_matches = orb_bruteforce_matching(path1, path2)
                matches_filename = "matches_bruteforce.png"
            img_kp1, _, _ = orb_keypoint_detection(path1)
            img_kp2, _, _ = orb_keypoint_detection(path2)
            img_harris1, num_corners1 = harris_corner_detection(path1)
            img_harris2, num_corners2 = harris_corner_detection(path2)
            cv.imwrite(os.path.join(OUTPUT_DIR, matches_filename), img_matches)
            cv.imwrite(os.path.join(OUTPUT_DIR, "keypoints_image1.png"), img_kp1)
            cv.imwrite(os.path.join(OUTPUT_DIR, "keypoints_image2.png"), img_kp2)
            cv.imwrite(os.path.join(OUTPUT_DIR, "harris_image1.png"), img_harris1)
            cv.imwrite(os.path.join(OUTPUT_DIR, "harris_image2.png"), img_harris2)
        st.success(f"Results saved to '{OUTPUT_DIR}' folder.")
        st.subheader("Keypoint Statistics")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Keypoints in Image 1", len(keypoints1))
        with stat_col2:
            st.metric("Keypoints in Image 2", len(keypoints2))
        with stat_col3:
            st.metric("Good Matches Found", len(good_matches))
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            show_image_with_download(img_matches, "Matched Features", matches_filename, width=900)
        st.subheader("Harris vs ORB Comparison")
        st.markdown(
            "Harris Corner Detection finds raw corner points based on intensity gradients "
            "(no descriptors, not matchable across images). ORB detects keypoints along with "
            "binary descriptors, making it suitable for matching features between two images."
        )
        comp_col1, comp_col2 = st.columns(2, gap="small")
        with comp_col1:
            st.markdown("**Harris Corners**")
            hstat1, hstat2 = st.columns(2)
            with hstat1:
                st.metric("Image 1 Corners", int(num_corners1))
            with hstat2:
                st.metric("Image 2 Corners", int(num_corners2))
        with comp_col2:
            st.markdown("**ORB Keypoints**")
            ostat1, ostat2 = st.columns(2)
            with ostat1:
                st.metric("Image 1 Keypoints", len(keypoints1))
            with ostat2:
                st.metric("Image 2 Keypoints", len(keypoints2))
        comparison_table = {
            "Aspect": ["Detects", "Descriptors", "Cross-Image Matching", "Rotation Invariant", "Scale Invariant"],
            "Harris Corner": ["Corners", "No", "Not Supported", "No", "No"],
            "ORB": ["Keypoints", "Yes (Binary)", "Supported", "Yes", "Yes"]
        }
        st.table(comparison_table)
        img_col1, img_col2 = st.columns(2, gap="small")
        with img_col1:
            show_image_with_download(img_harris1, f"Image 1 - Harris Corners ({int(num_corners1)})", "harris_image1.png")
            show_image_with_download(img_kp1, "Image 1 - ORB Keypoints", "keypoints_image1.png")
        with img_col2:
            show_image_with_download(img_harris2, f"Image 2 - Harris Corners ({int(num_corners2)})", "harris_image2.png")
            show_image_with_download(img_kp2, "Image 2 - ORB Keypoints", "keypoints_image2.png")
    os.unlink(path1)
    os.unlink(path2)
else:
    st.info("Please upload both images to proceed.")