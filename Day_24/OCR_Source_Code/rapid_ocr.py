# rapid_ocr.py
from rapidocr_onnxruntime import RapidOCR


def load_model():
    return RapidOCR()


def run_ocr(model, image):
    result, elapse = model(image)
    detections = []

    if result:
        for item in result:
            bbox = item[0]
            text = item[1]
            confidence = item[2]

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            detections.append({
                "text": text,
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence
            })

    detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return detections