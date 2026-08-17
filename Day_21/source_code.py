import cv2
import numpy as np


def load_image(path):
    return cv2.imread(path)


def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, ksize=(5, 5), sigma=0):
    return cv2.GaussianBlur(img, ksize, sigma)


def canny_edges(img, threshold1=100, threshold2=200):
    gray = to_grayscale(img)
    blur = gaussian_blur(gray)
    return cv2.Canny(blur, threshold1, threshold2)


def rotate_image(img, angle=45, scale=1.0):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (width, height))


def enhance_image(img, alpha=1.5, beta=20):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def binary_threshold(img, thresh=127, maxval=255):
    gray = to_grayscale(img)
    _, thresholded = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
    return thresholded


def detect_shapes(img, thresh=127, maxval=255, epsilon_factor=0.04):
    thresholded = binary_threshold(img, thresh, maxval)
    contours, hierarchy = cv2.findContours(
        thresholded,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    shapes = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
        vertices = len(approx)

        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            shape = "Rectangle/Square"
        elif vertices > 4:
            shape = "Circle"
        else:
            shape = "Unknown"

        shapes.append((shape, contour, approx))

    return shapes, contours, hierarchy


def sharpen_image(img, kernel=None):
    if kernel is None:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
    return cv2.filter2D(img, -1, kernel)


def flip_image(img, flip_code=1):
    return cv2.flip(img, flip_code)