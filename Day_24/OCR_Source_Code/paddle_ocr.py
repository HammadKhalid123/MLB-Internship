import os
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "huggingface"

from paddleocr import PaddleOCR


def load_model():
    return PaddleOCR(
        lang='en',
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    )


def run_ocr(model, image):
    result = model.predict(image)
    detections = []

    if result and isinstance(result[0], dict):
        for res in result:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("rec_polys", res.get("dt_polys", []))

            for i, poly in enumerate(polys):
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                detections.append({
                    "text": texts[i] if i < len(texts) else "",
                    "bbox": (x1, y1, x2, y2),
                    "confidence": scores[i] if i < len(scores) else None
                })
    else:
        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]

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