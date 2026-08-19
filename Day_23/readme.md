# OCR Application

## Overview

Today, I worked with multiple OCR libraries and explored how they can be used to extract text from images and documents.

## OCR Libraries Used

I worked with:

- **DocTR** – Deep learning based OCR for text detection and recognition.
- **PaddleOCR** – Used for text detection and recognition.
- **EasyOCR** – Used for simple text extraction from images.
- **Tesseract OCR** – Used through the `pytesseract` Python library.

## Preprocessing Techniques

I applied different image preprocessing techniques to improve OCR results, including:

- Image resizing
- Grayscale conversion
- Noise reduction
- Thresholding
- Contrast enhancement
- Morphological operations

The preprocessing technique depends on the quality and type of the input image.

## Challenges Faced

Some challenges I faced during the implementation were:

- OCR accuracy can decrease with low-quality or blurry images.
- Different OCR libraries perform differently on different types of images.
- Text with unusual fonts, rotations, or complex backgrounds can be difficult to recognize.
- Tables and complex document layouts require more than basic text extraction.
- Installing and managing multiple OCR libraries can sometimes cause dependency issues.

## Possible Improvements

The application can be improved by:

- Adding automatic image preprocessing.
- Adding text orientation detection and correction.
- Improving support for tables and structured documents.
- Adding better handling for low-resolution images.
- Adding confidence-based filtering of OCR results.
- Supporting multiple languages.
- Improving the Streamlit UI and adding downloadable OCR results.
- Adding PDF and multi-page document processing.