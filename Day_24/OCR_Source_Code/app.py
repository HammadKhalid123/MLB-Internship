import streamlit as st
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

import rapid_ocr
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
    "RapidOCR": rapid_ocr,
}

MAX_WORKERS_PER_ENGINE = {
    "Tesseract": 4,
    "EasyOCR": 2,
    "docTR": 1,
    "PaddleOCR": 1,
    "RapidOCR": 3,
}


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0a0a0a;
        }
        h1, h2, h3 {
            color: #f5f5f5;
            font-family: 'Segoe UI', sans-serif;
        }
        p, span, label {
            color: #d4d4d4;
        }

        .stButton>button {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: #ffffff;
            border-radius: 10px;
            border: none;
            padding: 0.7rem 1.6rem;
            font-weight: 600;
            font-size: 1rem;
            width: 100%;
            box-shadow: 0 3px 10px rgba(79, 70, 229, 0.3);
            transition: all 0.15s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #4f46e5, #4338ca);
            color: #ffffff;
            box-shadow: 0 5px 14px rgba(67, 56, 202, 0.4);
            transform: translateY(-1px);
        }
        .stButton>button:active {
            transform: translateY(0px);
            box-shadow: 0 2px 6px rgba(67, 56, 202, 0.3);
        }

        section[data-testid="stSidebar"] {
            background-color: #111111;
            border-right: 1px solid #262626;
        }

        .stTextArea textarea {
            background-color: #171717;
            border: 1px solid #262626;
            border-radius: 8px;
            color: #f5f5f5;
        }

        .stDownloadButton>button {
            background: linear-gradient(135deg, #10b981, #059669);
            color: #ffffff;
            border-radius: 10px;
            border: none;
            font-weight: 600;
            box-shadow: 0 2px 6px rgba(16, 185, 129, 0.25);
            transition: all 0.15s ease-in-out;
        }
        .stDownloadButton>button:hover {
            background: linear-gradient(135deg, #059669, #047857);
            color: #ffffff;
            box-shadow: 0 4px 10px rgba(5, 150, 105, 0.35);
            transform: translateY(-1px);
        }

        div[data-testid="stMetricValue"] {
            color: #818cf8;
        }
        div[data-testid="stMetricLabel"] {
            color: #a3a3a3;
        }

        [data-testid="stFileUploader"] section {
            background-color: #171717;
            border: 1px dashed #404040;
            border-radius: 10px;
        }

        .stContainer, div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #111111;
            border: 1px solid #262626;
        }

        .streamlit-expanderHeader {
            background-color: #171717;
            color: #f5f5f5;
        }

        .stCheckbox label {
            color: #d4d4d4;
        }

        .stCaption {
            color: #a3a3a3;
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


def process_single_document(uploaded_file, model, engine_module, apply_preprocessing):
    file_bytes = uploaded_file.read()
    image = processing.load_image_from_bytes(file_bytes)

    if image is None:
        return {
            "filename": uploaded_file.name,
            "error": "Could not read the uploaded image."
        }

    processed_image = processing.preprocess_image(image)
    ocr_ready_image = (
        processing.to_three_channel(processed_image)
        if apply_preprocessing
        else image
    )

    start_time = time.time()
    detections = engine_module.run_ocr(model, ocr_ready_image)
    elapsed_time = time.time() - start_time

    annotated_image = processing.draw_boxes(image, detections)
    extracted_text = build_extracted_text(detections)

    return {
        "filename": uploaded_file.name,
        "error": None,
        "original_image": image,
        "annotated_image": annotated_image,
        "detections": detections,
        "extracted_text": extracted_text,
        "elapsed_time": elapsed_time,
    }


def process_all_documents(uploaded_files, model, engine_module, apply_preprocessing, max_workers):
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_document, f, model, engine_module, apply_preprocessing)
            for f in uploaded_files
        ]
        for future in futures:
            results.append(future.result())

    return results


def render_result(result, engine_name):
    if result["error"]:
        st.error(f"{result['filename']}: {result['error']}")
        return

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(result["original_image"], cv2.COLOR_BGR2RGB), width=450)

    with col2:
        st.subheader("Detected Text Regions")
        st.image(cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB), width=450)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Regions Detected", len(result["detections"]))
    metric_col2.metric("Engine Used", engine_name)
    metric_col3.metric("Extraction Time", f"{result['elapsed_time']:.2f} sec")

    st.subheader("Extracted Text")

    with st.container(border=True):
        st.text_area(
            "Result",
            result["extracted_text"] if result["extracted_text"] else "No text detected.",
            height=300,
            label_visibility="collapsed",
            key=f"text_area_{result['filename']}"
        )

    st.download_button(
        label="⬇ Download Extracted Text",
        data=result["extracted_text"],
        file_name=f"extracted_{result['filename']}.txt",
        mime="text/plain",
        key=f"download_{result['filename']}"
    )

    with st.expander("View detection details (confidence scores)"):
        for d in result["detections"]:
            if not d["text"].strip():
                continue
            if isinstance(d["confidence"], (int, float)):
                conf_display = f"{d['confidence']:.2f}"
            else:
                conf_display = "N/A"
            st.write(f"**{d['text']}** — confidence: {conf_display}")


def main():
    inject_css()

    st.title("📄 OCR Studio")
    st.caption("Upload document images, choose an OCR engine, and extract clean text in seconds.")

    with st.sidebar:
        st.header("Settings")

        engine_name = st.selectbox(
            "OCR Engine",
            list(OCR_ENGINES.keys()),
            help="Choose which OCR engine to run on your images."
        )

        apply_preprocessing = st.checkbox(
            "Apply preprocessing",
            value=True,
            help="Grayscale, blur and threshold the image before running OCR."
        )

        st.markdown("---")
        st.markdown("**About**")
        st.write(
            "This app extracts text from document images using five OCR "
            "engines: Tesseract, EasyOCR, docTR, PaddleOCR and RapidOCR. "
            "Upload one or more images, pick an engine, and download the "
            "extracted text."
        )

    uploaded_files = st.file_uploader(
        "Upload document images",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload one or more images to get started.")
        return

    run_clicked = st.button(f"Extract Text with {engine_name} ({len(uploaded_files)} file(s))")

    if not run_clicked:
        st.info("Results will appear here after extraction.")
        return

    workers = MAX_WORKERS_PER_ENGINE.get(engine_name, 1)

    with st.spinner(f"Running {engine_name} on {len(uploaded_files)} document(s)..."):
        model = get_model(engine_name)
        overall_start = time.time()
        results = process_all_documents(
            uploaded_files,
            model,
            OCR_ENGINES[engine_name],
            apply_preprocessing,
            max_workers=workers
        )
        overall_elapsed = time.time() - overall_start

    st.markdown("---")
    st.metric("Total Batch Time", f"{overall_elapsed:.2f} seconds")

    for result in results:
        st.markdown("---")
        st.markdown(f"### {result['filename']}")
        render_result(result, engine_name)


if __name__ == "__main__":
    main()