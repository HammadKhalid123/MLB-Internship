import cv2
import numpy as np


def load_image_from_bytes(file_bytes):
    file_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    return image


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    return thresh


def to_three_channel(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def draw_boxes(image, detections):
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        cv2.rectangle(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )
    return output