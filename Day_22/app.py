import streamlit as st
import cv2
import tempfile
import os
from source_code import get_reader, process_image


@st.cache_resource
def load_reader():
    return get_reader()


reader = load_reader()

st.title("OCR Text Extractor")
st.write("Upload an image to extract text using EasyOCR")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    original_image = cv2.cvtColor(cv2.imread(temp_path), cv2.COLOR_BGR2RGB)

    if st.button("Extract Text"):
        with st.spinner("Extracting text..."):
            data = process_image(temp_path, reader, output_dir="output_images", output_txt="output.txt")

        if data is None:
            st.error("No text detected")
        else:
            annotated_image = cv2.cvtColor(data["annotated_image"], cv2.COLOR_BGR2RGB)

            extracted_text = "\n".join([detection[1] for detection in data["result"]])

            col1, col2 = st.columns(2)

            with col1:
                st.image(original_image, caption="Original Image")

            with col2:
                st.image(annotated_image, caption="Detected Text Regions")

            st.text_area("Extracted Text", extracted_text, height=300)

            with open("output.txt", "rb") as f:
                st.download_button("Download Text File", f, file_name="output.txt")

    os.remove(temp_path)