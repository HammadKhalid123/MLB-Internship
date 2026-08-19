import streamlit as st
import cv2

import processing
import tesseract_ocr
import easy_ocr
import doctr_ocr
import paddle_ocr


st.set_page_config(
    page_title="OCR Studio",
    page_icon="📄",
    layout="wide"
)

OCR_ENGINES = {
    "Tesseract": tesseract_ocr,
    "EasyOCR": easy_ocr,
    "docTR": doctr_ocr,
    "PaddleOCR": paddle_ocr,
}


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #fafafa;
        }
        h1, h2, h3 {
            color: #111827;
            font-family: 'Segoe UI', sans-serif;
        }
        p, span, label {
            color: #374151;
        }

        /* --- Primary buttons --- */
        .stButton>button {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: #ffffff;
            border-radius: 10px;
            border: none;
            padding: 0.7rem 1.6rem;
            font-weight: 600;
            font-size: 1rem;
            width: 100%;
            box-shadow: 0 3px 10px rgba(220, 38, 38, 0.3);
            transition: all 0.15s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #dc2626, #b91c1c);
            color: #ffffff;
            box-shadow: 0 5px 14px rgba(185, 28, 28, 0.4);
            transform: translateY(-1px);
        }
        .stButton>button:active {
            transform: translateY(0px);
            box-shadow: 0 2px 6px rgba(185, 28, 28, 0.3);
        }

        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        .stTextArea textarea {
            background-color: #ffffff;
            border-radius: 8px;
            color: #111827;
        }

        /* --- Download button --- */
        .stDownloadButton>button {
            background: linear-gradient(135deg, #059669, #047857);
            color: #ffffff;
            border-radius: 10px;
            border: none;
            font-weight: 600;
            box-shadow: 0 2px 6px rgba(5, 150, 105, 0.25);
            transition: all 0.15s ease-in-out;
        }
        .stDownloadButton>button:hover {
            background: linear-gradient(135deg, #047857, #065f46);
            color: #ffffff;
            box-shadow: 0 4px 10px rgba(4, 120, 87, 0.35);
            transform: translateY(-1px);
        }

        div[data-testid="stMetricValue"] {
            color: #2563eb;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource(show_spinner=False)
def get_model(engine_name):
    module = OCR_ENGINES[engine_name]
    return module.load_model()


def build_extracted_text(detections):
    return "\n".join(d["text"] for d in detections if d["text"].strip())


def main():
    inject_css()

    st.title("📄 OCR Studio")
    st.caption("Upload a document image, choose an OCR engine, and extract clean text in seconds.")

    with st.sidebar:
        st.header("Settings")

        engine_name = st.selectbox(
            "OCR Engine",
            list(OCR_ENGINES.keys()),
            help="Choose which OCR engine to run on your image."
        )

        apply_preprocessing = st.checkbox(
            "Apply preprocessing",
            value=True,
            help="Grayscale, blur and threshold the image before running OCR."
        )

        st.markdown("---")
        st.markdown("**About**")
        st.write(
            "This app extracts text from document images using four OCR "
            "engines: Tesseract, EasyOCR, docTR and PaddleOCR. Upload an "
            "image, pick an engine, and download the extracted text."
        )

    uploaded_file = st.file_uploader(
        "Upload a document image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"]
    )

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    file_bytes = uploaded_file.read()
    image = processing.load_image_from_bytes(file_bytes)

    if image is None:
        st.error("Could not read the uploaded image. Please try a different file.")
        return

    processed_image = processing.preprocess_image(image)
    ocr_ready_image = (
        processing.to_three_channel(processed_image)
        if apply_preprocessing
        else image
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width=450)

    run_clicked = st.button(f"Extract Text with {engine_name}")

    if not run_clicked:
        with col2:
            st.subheader("Detected Text Regions")
            st.info("Results will appear here after extraction.")
        return

    with st.spinner(f"Running {engine_name}..."):
        model = get_model(engine_name)
        detections = OCR_ENGINES[engine_name].run_ocr(model, ocr_ready_image)
        annotated_image = processing.draw_boxes(image, detections)

    with col2:
        st.subheader("Detected Text Regions")
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), width=450)

    extracted_text = build_extracted_text(detections)

    st.markdown("---")

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Regions Detected", len(detections))
    metric_col2.metric("Engine Used", engine_name)

    st.subheader("Extracted Text")

    # NOTE: previously this used st.markdown('<div class="result-card">...')
    # to wrap the text_area in a styled card. Streamlit widgets don't
    # actually render *inside* markdown divs like that - they render as
    # separate elements right after. That's what caused the empty
    # "banner" box you saw. st.container(border=True) is the correct,
    # native way to get a card look around a widget.
    with st.container(border=True):
        st.text_area(
            "Result",
            extracted_text if extracted_text else "No text detected.",
            height=300,
            label_visibility="collapsed"
        )

    st.download_button(
        label="⬇ Download Extracted Text",
        data=extracted_text,
        file_name=f"extracted_text_{engine_name.lower()}.txt",
        mime="text/plain"
    )

    with st.expander("View detection details (confidence scores)"):
        for d in detections:
            if not d["text"].strip():
                continue
            if isinstance(d["confidence"], (int, float)):
                conf_display = f"{d['confidence']:.2f}"
            else:
                conf_display = "N/A"
            st.write(f"**{d['text']}** — confidence: {conf_display}")


if __name__ == "__main__":
    main()