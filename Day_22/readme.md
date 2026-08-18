# OCR Text Detection

## What is OCR?

OCR (Optical Character Recognition) is a technology used to recognize text from images and convert it into editable and searchable digital text. It is useful for extracting text from documents, screenshots, scanned pages, and other images.

## OCR Library Used

I used **EasyOCR** for text extraction. EasyOCR is simple to implement, supports multiple languages, and provides good accuracy without requiring us to train an OCR model from scratch.

## Preprocessing Techniques

To improve OCR results, I applied different image preprocessing techniques, including:

- Image resizing to make text clearer
- Grayscale conversion
- Noise reduction
- Thresholding to improve text-background separation
- Contrast enhancement

These techniques helped EasyOCR detect text more clearly, especially when the input image was blurry or had a complex background.

## Challenges Faced

During the OCR process, I faced several challenges:

- Blurry and low-quality images
- Different font sizes and styles
- Complex or noisy backgrounds
- Small and unclear text
- Incorrect recognition of some characters
- Difficulty extracting text from tables and complex layouts

Overall, preprocessing the images before applying OCR significantly improved the quality of the extracted text.