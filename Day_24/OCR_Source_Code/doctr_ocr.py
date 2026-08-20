from doctr.models import ocr_predictor


def load_model():
    return ocr_predictor(
        det_arch='db_mobilenet_v3_large',
        reco_arch='crnn_mobilenet_v3_small',
        pretrained=True,
        assume_straight_pages=True,
    )


def run_ocr(model, image):
    h, w = image.shape[:2]
    result = model([image])
    detections = []

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                if not line.words:
                    continue

                text = " ".join(word.value for word in line.words)

                xs1, ys1, xs2, ys2 = [], [], [], []
                for word in line.words:
                    (wx1, wy1), (wx2, wy2) = word.geometry
                    xs1.append(wx1 * w)
                    ys1.append(wy1 * h)
                    xs2.append(wx2 * w)
                    ys2.append(wy2 * h)

                x1, y1 = min(xs1), min(ys1)
                x2, y2 = max(xs2), max(ys2)

                confidence = sum(word.confidence for word in line.words) / len(line.words)

                detections.append({
                    "text": text,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })

    return detections