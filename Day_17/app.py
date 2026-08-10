import io
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from document_enhancement import (
    correct_perspective,
    convert_to_grayscale,
    reduce_noise,
    enhance_brightness_contrast,
    sharpen_image,
)

st.set_page_config(
    page_title="Document Enhancement Studio",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---------- Base app background & text ---------- */
    .stApp {
        background: #f7f8fa;
    }

    .stApp, .stApp p, .stApp li,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    h1, h2, h3, h4, h5, h6 {
        color: #1f2430;
    }

    /* ---------- Header banner ---------- */
    .app-header {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
        margin-bottom: 1.4rem;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.9rem;
        color: #111827;
        font-weight: 700;
    }
    .app-header p {
        margin: 0.35rem 0 0 0;
        color: #4b5563;
        font-size: 0.95rem;
    }

    .step-pill {
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3b5bdb;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.4rem;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] * {
        color: #1f2430 !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #6b7280 !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #111827 !important;
    }

    /* ---------- General text colors ---------- */
    .stApp label, .stApp span,
    div[data-testid="stMetricLabel"],
    .stRadio label, .stCheckbox label, .stSlider label,
    .stTabs [data-baseweb="tab"] p {
        color: #1f2430 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #111827 !important;
    }

    /* ---------- Inputs ---------- */
    div[data-testid="stNumberInput"] input {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    /* ---------- Metric cards ---------- */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 0.7rem 0.9rem;
        border-radius: 12px;
    }

    /* ---------- Buttons ---------- */
    .stButton>button, .stDownloadButton>button {
        border-radius: 10px;
        border: 1px solid #3b5bdb;
        background-color: #3b5bdb;
        color: #ffffff !important;
        font-weight: 600;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #2f4bc4;
        border-color: #2f4bc4;
        color: #ffffff !important;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-bottom: none;
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background: #eef2ff !important;
    }

    /* ---------- Alerts (info / warning / success) ---------- */
    div[data-testid="stAlert"] p {
        color: #1f2430 !important;
    }

    .footer-note {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def to_cv2(pil_image):
    arr = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def to_pil(cv2_image):
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image)
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def auto_detect_document_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = image.shape[:2]
    image_area = h * w

    for contour in contours[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * image_area:
            pts = approx.reshape(4, 2).astype("float32")
            return order_points(pts)
    return None


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def manual_corners_from_sliders(image, tl, tr, br, bl):
    h, w = image.shape[:2]

    def pct_to_xy(p):
        return [w * p[0] / 100.0, h * p[1] / 100.0]

    return np.float32([pct_to_xy(tl), pct_to_xy(tr), pct_to_xy(br), pct_to_xy(bl)])


def process_pipeline(image, settings):
    result = image.copy()

    if settings["do_perspective"] and settings["corners"] is not None:
        result = correct_perspective(
            result,
            settings["corners"],
            width=settings["out_width"],
            height=settings["out_height"],
        )

    if settings["do_grayscale"]:
        result = convert_to_grayscale(result)

    if settings["do_denoise"]:
        result = reduce_noise(result)

    if settings["do_brightness_contrast"]:
        result = enhance_brightness_contrast(
            result,
            brightness=settings["brightness"],
            contrast=settings["contrast"],
        )

    if settings["do_sharpen"]:
        result = sharpen_image(result)

    return result


def image_to_bytes(pil_image, fmt="PNG"):
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return buf.getvalue()


st.markdown(
    """
    <div class="app-header">
        <span class="step-pill">MINI PROJECT · IMAGE PROCESSING</span>
        <h1>🧾 Document Enhancement Studio</h1>
        <p>Perspective correction · Grayscale · Denoising · Brightness &amp; contrast · Sharpening — all in one interactive pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")

    do_perspective = st.checkbox("1. Perspective correction", value=False)
    perspective_mode = "Auto-detect"
    if do_perspective:
        perspective_mode = st.radio(
            "Corner selection",
            ["Auto-detect", "Manual (sliders)"],
            horizontal=True,
        )

    do_grayscale = st.checkbox("2. Convert to grayscale", value=True)
    do_denoise = st.checkbox("3. Reduce noise", value=True)
    do_brightness_contrast = st.checkbox("4. Brightness / contrast", value=True)

    brightness, contrast = 20, 1.2
    if do_brightness_contrast:
        brightness = st.slider("Brightness", -100, 100, 20)
        contrast = st.slider("Contrast", 0.5, 3.0, 1.2, 0.1)

    do_sharpen = st.checkbox("5. Sharpen", value=True)

    st.markdown("---")
    out_width = st.number_input("Output width", 200, 2000, 600, step=50)
    out_height = st.number_input("Output height", 200, 2000, 800, step=50)

    st.markdown("---")
    st.caption("Built with OpenCV + Streamlit")

tab_single, tab_batch = st.tabs(["📄 Single Image", "🗂️ Batch (Dataset of 10+)"])

with tab_single:
    uploaded_file = st.file_uploader(
        "Upload a document image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="single"
    )

    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        cv_img = to_cv2(pil_img)

        corners = None
        if do_perspective:
            if perspective_mode == "Auto-detect":
                corners = auto_detect_document_corners(cv_img)
                if corners is None:
                    st.warning(
                        "Could not auto-detect document edges. Switch to Manual mode to set corners."
                    )
            else:
                st.markdown("**Set corner positions (% of image width/height)**")
                c1, c2 = st.columns(2)
                with c1:
                    tl_x = st.slider("Top-left X %", 0, 100, 5)
                    tl_y = st.slider("Top-left Y %", 0, 100, 5)
                    bl_x = st.slider("Bottom-left X %", 0, 100, 5)
                    bl_y = st.slider("Bottom-left Y %", 0, 100, 95)
                with c2:
                    tr_x = st.slider("Top-right X %", 0, 100, 95)
                    tr_y = st.slider("Top-right Y %", 0, 100, 5)
                    br_x = st.slider("Bottom-right X %", 0, 100, 95)
                    br_y = st.slider("Bottom-right Y %", 0, 100, 95)
                corners = manual_corners_from_sliders(
                    cv_img, (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)
                )

        settings = {
            "do_perspective": do_perspective,
            "corners": corners,
            "out_width": int(out_width),
            "out_height": int(out_height),
            "do_grayscale": do_grayscale,
            "do_denoise": do_denoise,
            "do_brightness_contrast": do_brightness_contrast,
            "brightness": brightness,
            "contrast": contrast,
            "do_sharpen": do_sharpen,
        }

        with st.spinner("Processing image..."):
            processed = process_pipeline(cv_img, settings)
        processed_pil = to_pil(processed)

        m1, m2, m3 = st.columns(3)
        m1.metric("Original size", f"{pil_img.width}×{pil_img.height}")
        m2.metric("Output size", f"{processed_pil.width}×{processed_pil.height}")
        m3.metric("Mode", "Grayscale" if do_grayscale else "Color")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(pil_img, use_container_width=True)
        with col2:
            st.markdown("**Enhanced**")
            st.image(processed_pil, use_container_width=True)

        st.download_button(
            "⬇️ Download enhanced image",
            data=image_to_bytes(processed_pil),
            file_name=f"enhanced_{uploaded_file.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.info("Upload a document photo to start. Tip: enable auto perspective detection for tilted scans.")

with tab_batch:
    st.markdown(
        "Upload **10 or more** document images to run the same enhancement pipeline "
        "across the whole dataset and download everything as a single ZIP."
    )
    batch_files = st.file_uploader(
        "Upload dataset images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="batch",
    )

    if batch_files:
        st.write(f"**{len(batch_files)} image(s) uploaded**")
        if len(batch_files) < 10:
            st.warning("Project requirement asks for at least 10 images. Add a few more for the final submission.")

        run_batch = st.button("🚀 Run batch processing", use_container_width=True)

        if run_batch:
            zip_buffer = io.BytesIO()
            progress = st.progress(0.0, text="Starting...")
            preview_cols = st.columns(4)

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, f in enumerate(batch_files):
                    pil_img = Image.open(f)
                    cv_img = to_cv2(pil_img)

                    corners = None
                    if do_perspective:
                        corners = auto_detect_document_corners(cv_img)

                    settings = {
                        "do_perspective": do_perspective,
                        "corners": corners,
                        "out_width": int(out_width),
                        "out_height": int(out_height),
                        "do_grayscale": do_grayscale,
                        "do_denoise": do_denoise,
                        "do_brightness_contrast": do_brightness_contrast,
                        "brightness": brightness,
                        "contrast": contrast,
                        "do_sharpen": do_sharpen,
                    }

                    processed = process_pipeline(cv_img, settings)
                    processed_pil = to_pil(processed)

                    out_name = f"enhanced_{f.name.rsplit('.', 1)[0]}.png"
                    zf.writestr(out_name, image_to_bytes(processed_pil))

                    if i < 4:
                        with preview_cols[i]:
                            st.image(processed_pil, caption=out_name, use_container_width=True)

                    progress.progress((i + 1) / len(batch_files), text=f"Processed {i + 1}/{len(batch_files)}")

            st.success("Batch processing complete!")
            st.download_button(
                "⬇️ Download all enhanced images (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="enhanced_documents.zip",
                mime="application/zip",
                use_container_width=True,
            )
    else:
        st.info("Upload your dataset (own photos or images from Kaggle / Google Images / DocLayNet).")

st.markdown(
    '<div class="footer-note">Document Enhancement Studio · OpenCV pipeline wrapped in a Streamlit UI</div>',
    unsafe_allow_html=True,
)