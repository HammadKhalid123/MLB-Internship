import easyocr


def load_model():
    return easyocr.Reader(['en'], gpu=False)


def run_ocr(model, image):
    result = model.readtext(image, paragraph=False)
    detections = []

    for bbox, text, confidence in result:
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]

        detections.append({
            "text": text,
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence
        })

    return detections