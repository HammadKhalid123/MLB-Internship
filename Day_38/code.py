import cv2
import numpy as np
from PIL import Image

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def binary_threshold(image, thresh_val=127):
    gray = to_gray(image)
    _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return result

def otsu_threshold(image):
    gray = to_gray(image)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result

def adaptive_threshold(image, method="mean", block_size=11, c=2):
    gray = to_gray(image)
    if block_size % 2 == 0:
        block_size += 1
    if method == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    result = cv2.adaptiveThreshold(gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c)
    return result

def apply_segmentation(image, method, thresh_val=127, block_size=11, c=2, adaptive_method="mean"):
    if method == "Binary":
        return binary_threshold(image, thresh_val)
    elif method == "Otsu":
        return otsu_threshold(image)
    elif method == "Adaptive":
        return adaptive_threshold(image, adaptive_method, block_size, c)
    else:
        raise ValueError("Invalid segmentation method")

def convert_to_pil(result_image):
    return Image.fromarray(result_image)