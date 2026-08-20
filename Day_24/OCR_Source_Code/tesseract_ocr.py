import cv2
import pytesseract

TESSERACT_CONFIG = "--oem 3 --psm 6"
MIN_DIMENSION = 1000


def load_model():
    return None


def _upscale_if_needed(image):
    h, w = image.shape[:2]
    scale = 1.0

    if max(h, w) < MIN_DIMENSION:
        scale = MIN_DIMENSION / max(h, w)

    if scale > 1.0:
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )

    return image, scale


def run_ocr(model, image):
    resized_image, scale = _upscale_if_needed(image)

    data = pytesseract.image_to_data(
        resized_image,
        config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT
    )

    # Group words by (block_num, par_num, line_num)
    lines = {}

    for i in range(len(data["text"])):
        text = data["text"][i]

        if not text.strip():
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])

        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = None

        if confidence is not None and confidence < 0:
            confidence = None

        if key not in lines:
            lines[key] = {
                "words": [],
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "confidences": [],
            }

        line = lines[key]
        line["words"].append(text)
        line["x1"] = min(line["x1"], x)
        line["y1"] = min(line["y1"], y)
        line["x2"] = max(line["x2"], x + w)
        line["y2"] = max(line["y2"], y + h)

        if confidence is not None:
            line["confidences"].append(confidence)

    detections = []

    # Preserve reading order (block, paragraph, line)
    for key in sorted(lines.keys()):
        line = lines[key]

        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]

        if scale != 1.0:
            x1, y1, x2, y2 = (
                int(x1 / scale),
                int(y1 / scale),
                int(x2 / scale),
                int(y2 / scale),
            )

        confidence = (
            sum(line["confidences"]) / len(line["confidences"])
            if line["confidences"] else None
        )

        detections.append({
            "text": " ".join(line["words"]),
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence,
            "block_num": key[0],
            "par_num": key[1],
            "line_num": key[2],
        })

    return detections