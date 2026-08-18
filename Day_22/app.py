import gradio as gr
import cv2
from source_code import get_reader, process_image

reader = get_reader()


def run_ocr(image_path):
    if image_path is None:
        return None, "", None

    data = process_image(image_path, reader, output_dir="output_images", output_txt="output.txt")

    if data is None:
        return None, "No text detected", None

    annotated_image = cv2.cvtColor(data["annotated_image"], cv2.COLOR_BGR2RGB)

    extracted_text = "\n".join([detection[1] for detection in data["result"]])

    return annotated_image, extracted_text, "output.txt"


demo = gr.Interface(
    fn=run_ocr,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=[
        gr.Image(label="Detected Text Regions"),
        gr.Textbox(label="Extracted Text", lines=15),
        gr.File(label="Download Text File")
    ],
    title="OCR Text Extractor",
    description="Upload an image to extract text using EasyOCR"
)

if __name__ == "__main__":
    demo.launch()