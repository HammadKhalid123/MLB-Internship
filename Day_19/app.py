import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import os

from mini_project import save_output


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Shape Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Custom CSS — clean, minimal, white theme, no icons
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
    }

    section[data-testid="stSidebar"] {
        background-color: #fafafa;
        border-right: 1px solid #e6e6e6;
    }

    h1, h2, h3, h4 {
        color: #111111;
        font-weight: 600;
    }

    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: -8px;
        margin-bottom: 25px;
    }

    .card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .card h4 {
        margin-top: 0;
        color: #111111;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }

    .section-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 18px;
        margin-bottom: 6px;
    }

    div.stButton > button {
        background-color: #111827;
        color: white;
        border-radius: 8px;
        padding: 0.55em 1.4em;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #374151;
        color: white;
    }

    div.stDownloadButton > button {
        background-color: #ffffff;
        color: #111827;
        border-radius: 8px;
        padding: 0.55em 1.4em;
        font-weight: 600;
        border: 1px solid #111827;
        width: 100%;
    }
    div.stDownloadButton > button:hover {
        background-color: #111827;
        color: white;
    }

    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    hr {
        border: none;
        border-top: 1px solid #e6e6e6;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Core detection logic (parametrized version of mini_project.detect_shapes)
# ---------------------------------------------------------
THRESHOLD_TYPES = {
    "Binary": cv2.THRESH_BINARY,
    "Binary Inverted": cv2.THRESH_BINARY_INV,
    "Otsu (Automatic)": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    "Adaptive Mean": "adaptive_mean",
    "Adaptive Gaussian": "adaptive_gaussian",
}


def detect_shapes_custom(
    image_path,
    threshold_type,
    threshold_value,
    blur_ksize,
    min_area,
    epsilon_factor,
    square_tolerance,
    contour_color,
    box_color,
    text_color,
    show_area,
    show_perimeter,
):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur_ksize > 0:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if threshold_type in ("adaptive_mean", "adaptive_gaussian"):
        method = (
            cv2.ADAPTIVE_THRESH_MEAN_C
            if threshold_type == "adaptive_mean"
            else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
        binary = cv2.adaptiveThreshold(
            gray, 255, method, cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        binary = cv2.threshold(gray, threshold_value, 255, threshold_type)[1]

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
        corners = len(approx)
        x, y, w, h = cv2.boundingRect(contour)

        if corners == 3:
            shape = "Triangle"
        elif corners == 4:
            shape = "Square" if abs(w - h) < square_tolerance else "Rectangle"
        elif corners >= 7:
            shape = "Circle"
        else:
            shape = "Polygon"

        cv2.drawContours(img, [contour], -1, contour_color, 3)
        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)

        cv2.putText(
            img, shape, (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2,
        )

        line_y = y - 12
        if show_area:
            cv2.putText(
                img, f"Area: {area:.0f}", (x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2,
            )
            line_y = y + h + 20

        if show_perimeter:
            cv2.putText(
                img, f"Perimeter: {perimeter:.0f}", (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2,
            )

        results.append({"shape": shape, "area": area, "perimeter": perimeter})

    return img, results


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# ---------------------------------------------------------
# Sidebar — Parameters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Parameters")

    st.markdown('<div class="section-label">Thresholding</div>', unsafe_allow_html=True)

    threshold_label = st.selectbox("Threshold type", list(THRESHOLD_TYPES.keys()), index=0)
    threshold_type = THRESHOLD_TYPES[threshold_label]

    is_adaptive = threshold_type in ("adaptive_mean", "adaptive_gaussian")
    is_otsu = threshold_label == "Otsu (Automatic)"

    threshold_value = st.slider(
        "Threshold value",
        min_value=0,
        max_value=255,
        value=127,
        disabled=is_adaptive or is_otsu,
        help="Ignored for Adaptive and Otsu modes.",
    )

    blur_ksize = st.slider(
        "Blur strength (0 = off)",
        min_value=0,
        max_value=15,
        value=0,
        help="Applies Gaussian blur before thresholding to reduce noise.",
    )

    st.markdown('<div class="section-label">Contour Filtering</div>', unsafe_allow_html=True)

    min_area = st.slider("Minimum area (px²)", min_value=0, max_value=5000, value=500, step=50)

    epsilon_factor = st.slider(
        "Approximation accuracy",
        min_value=0.01,
        max_value=0.10,
        value=0.04,
        step=0.01,
        help="Lower values keep more corners, higher values simplify shapes more.",
    )

    square_tolerance = st.slider(
        "Square vs rectangle tolerance (px)",
        min_value=0,
        max_value=30,
        value=10,
        help="Max width/height difference to classify a 4-sided shape as a square.",
    )

    st.markdown('<div class="section-label">Display Options</div>', unsafe_allow_html=True)

    show_area = st.checkbox("Show area label", value=True)
    show_perimeter = st.checkbox("Show perimeter label", value=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        contour_color_hex = st.color_picker("Contour", "#0000FF")
    with col_b:
        box_color_hex = st.color_picker("Box", "#00FF00")
    with col_c:
        text_color_hex = st.color_picker("Text", "#FF0000")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("# Shape Detection")
st.markdown(
    '<p class="subtitle">Adjust the parameters in the sidebar, then detect shapes in your image.</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp"],
)

st.markdown("---")


# ---------------------------------------------------------
# Main Content
# ---------------------------------------------------------
if uploaded_file is None:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding:60px 20px;">
            <h4 style="border:none;">No image uploaded</h4>
            <p style="color:#6b7280;">Upload an image from the sidebar to begin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    temp_input_path = "temp_input.png"
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    detect_clicked = st.button("Detect Shapes")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card"><h4>Original Image</h4>', unsafe_allow_html=True)
        st.image(Image.open(uploaded_file), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if detect_clicked:
        with st.spinner("Processing..."):
            output_img, results = detect_shapes_custom(
                temp_input_path,
                threshold_type=threshold_type,
                threshold_value=threshold_value,
                blur_ksize=blur_ksize,
                min_area=min_area,
                epsilon_factor=epsilon_factor,
                square_tolerance=square_tolerance,
                contour_color=hex_to_bgr(contour_color_hex),
                box_color=hex_to_bgr(box_color_hex),
                text_color=hex_to_bgr(text_color_hex),
                show_area=show_area,
                show_perimeter=show_perimeter,
            )

            output_path = os.path.join("output", "result.png")
            save_output(output_img, output_path)

            output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

        with col2:
            st.markdown('<div class="card"><h4>Detected Shapes</h4>', unsafe_allow_html=True)
            st.image(output_img_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        if len(results) == 0:
            st.warning("No shapes were detected. Try adjusting the threshold or minimum area.")
        else:
            st.write(f"**{len(results)} shape(s) detected**")
            st.write("")

            st.markdown('<div class="card"><h4>Shape Details</h4>', unsafe_allow_html=True)
            table_data = [
                {
                    "#": i + 1,
                    "Shape": r["shape"],
                    "Area (px²)": f"{r['area']:.0f}",
                    "Perimeter (px)": f"{r['perimeter']:.0f}",
                }
                for i, r in enumerate(results)
            ]
            st.dataframe(table_data, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

            is_success, buffer = cv2.imencode(".png", output_img)
            if is_success:
                st.download_button(
                    label="Download Result Image",
                    data=io.BytesIO(buffer).getvalue(),
                    file_name="detected_shapes.png",
                    mime="image/png",
                )