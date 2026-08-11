import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from mini_project import (
    load_image,
    convert_to_grayscale,
    apply_morphology,
    detect_document_boundary,
    draw_document_boundary,
    save_output,
)

st.set_page_config(
    page_title="Document Boundary Detector",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #ffffff;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
            max-width: 900px;
        }

        .app-header {
            padding: 24px 26px;
            border-radius: 14px;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
            border: 1px solid #e5e9f0;
            margin-bottom: 22px;
        }
        .app-header h1 {
            margin: 0;
            font-size: 26px;
            font-weight: 700;
            color: #111827;
        }
        .app-header p {
            margin: 6px 0 0 0;
            font-size: 14px;
            color: #6b7280;
        }

        .section-card {
            background-color: #ffffff;
            border: 1px solid #e5e9f0;
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .section-title {
            font-size: 15px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 10px;
        }

        .badge-success {
            display: inline-block;
            padding: 5px 14px;
            border-radius: 999px;
            background-color: #ecfdf5;
            color: #047857;
            font-weight: 600;
            font-size: 13px;
            border: 1px solid #a7f3d0;
        }
        .badge-error {
            display: inline-block;
            padding: 5px 14px;
            border-radius: 999px;
            background-color: #fef2f2;
            color: #b91c1c;
            font-weight: 600;
            font-size: 13px;
            border: 1px solid #fecaca;
        }

        div[data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }
        div[data-testid="stImage"] img {
            max-width: 100%;
            max-height: 60vh;
            width: auto;
            height: auto;
            object-fit: contain;
            border-radius: 10px;
            border: 1px solid #e5e9f0;
        }

        section[data-testid="stSidebar"] {
            background-color: #fafafa;
            border-right: 1px solid #e5e9f0;
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}

        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.7rem;
                padding-right: 0.7rem;
                padding-top: 0.8rem;
            }
            .app-header {
                padding: 16px 16px;
                border-radius: 10px;
            }
            .app-header h1 {
                font-size: 20px;
            }
            .app-header p {
                font-size: 12.5px;
            }
            div[data-testid="stImage"] img {
                max-height: 40vh;
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 12px;
                padding: 6px 8px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>📄 Document Boundary Detector</h1>
        <p>Upload a scanned document photo to automatically detect its edges and boundary
        using classical computer vision (Grayscale → Blur → Canny → Morphology → Contour Detection).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

MORPH_OPERATIONS = {
    "Closing (fill gaps)": "closing",
    "Opening (remove noise)": "opening",
    "Erosion": "erosion",
    "Dilation": "dilation",
    "Gradient (outline)": "gradient",
    "Top Hat": "tophat",
    "Black Hat": "blackhat",
}

KERNEL_SHAPES = {
    "Rectangle": "rect",
    "Ellipse": "ellipse",
    "Cross": "cross",
}

with st.sidebar:
    st.markdown("### ⚙️ Edge Detection Settings")

    blur_kernel = st.slider("Gaussian Blur Kernel Size", 1, 15, 5, step=2)
    canny_low = st.slider("Canny Lower Threshold", 0, 255, 50)
    canny_high = st.slider("Canny Upper Threshold", 0, 255, 150)

    st.markdown("---")
    st.markdown("### 🧩 Morphological Operation")

    morph_label = st.selectbox("Operation", list(MORPH_OPERATIONS.keys()), index=0)
    morph_operation = MORPH_OPERATIONS[morph_label]

    kernel_shape_label = st.selectbox("Kernel Shape", list(KERNEL_SHAPES.keys()), index=0)
    kernel_shape = KERNEL_SHAPES[kernel_shape_label]

    morph_kernel_size = st.slider("Kernel Size", 3, 21, 5, step=2)
    morph_iterations = st.slider("Iterations", 1, 5, 2)

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.info(
        "1. Grayscale conversion\n"
        "2. Gaussian blur (noise reduction)\n"
        "3. Canny edge detection\n"
        "4. Morphological operation (your choice)\n"
        "5. Largest contour → document boundary"
    )

uploaded_file = st.file_uploader(
    "Upload a document image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        img = load_image(tmp_path)
        gray = convert_to_grayscale(img)

        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        blur = cv2.GaussianBlur(gray, (k, k), 0)

        edges = cv2.Canny(blur, canny_low, canny_high)

        morph = apply_morphology(
            edges,
            operation=morph_operation,
            kernel_shape=kernel_shape,
            kernel_size=morph_kernel_size,
            iterations=morph_iterations,
        )

        boundary = detect_document_boundary(morph)
        final_image = draw_document_boundary(img, boundary)

        if boundary is not None:
            st.markdown(
                f'<span class="badge-success">✅ Boundary detected — {len(boundary)} points</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge-error">❌ No boundary detected — try adjusting the thresholds or operation</span>',
                unsafe_allow_html=True,
            )

        st.write("")

        st.markdown('<div class="section-title">Pipeline Steps</div>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["Original", "Gray", "Blur", "Edges", "Morph", "Final"]
        )

        with tab1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with tab2:
            st.image(gray, use_container_width=True, channels="GRAY")
        with tab3:
            st.image(blur, use_container_width=True, channels="GRAY")
        with tab4:
            st.image(edges, use_container_width=True, channels="GRAY")
        with tab5:
            st.caption(f"Operation: {morph_label} · Shape: {kernel_shape_label} · Kernel: {morph_kernel_size}x{morph_kernel_size} · Iterations: {morph_iterations}")
            st.image(morph, use_container_width=True, channels="GRAY")
        with tab6:
            st.image(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown('<div class="section-title">Save Output</div>', unsafe_allow_html=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("outputs", f"document_boundary_{timestamp}.jpg")
        save_output(final_image, output_path)

        success, buffer = cv2.imencode(".jpg", final_image)
        if success:
            st.download_button(
                label="⬇️ Download Final Image",
                data=buffer.tobytes(),
                file_name=f"document_boundary_{timestamp}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

        st.caption(f"Also saved locally to: `{output_path}`")

    except ValueError as e:
        st.error(f"Error: {e}")

    finally:
        os.remove(tmp_path)

else:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Get started</div>
            Upload a document image above to see the full boundary-detection pipeline in action.
        </div>
        """,
        unsafe_allow_html=True,
    )