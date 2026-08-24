import cv2 as cv

def Processing(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray

def Binary(gray):
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    return binary

def Adaptive_Threshold(gray):
    adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return adaptive_thresh

def Otsu_Threshold(gray):
    _, otsu_thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return otsu_thresh

def compare_results(image, gray, binary, adaptive_thresh, otsu_thresh):
    gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    binary_bgr = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    adaptive_bgr = cv.cvtColor(adaptive_thresh, cv.COLOR_GRAY2BGR)
    otsu_bgr = cv.cvtColor(otsu_thresh, cv.COLOR_GRAY2BGR)

    size = (300, 250)

    images = [
        cv.resize(image, size),
        cv.resize(gray_bgr, size),
        cv.resize(binary_bgr, size),
        cv.resize(adaptive_bgr, size),
        cv.resize(otsu_bgr, size)
    ]

    labels = ["Original", "Grayscale", "Binary", "Adaptive", "Otsu"]

    for img, label in zip(images, labels):
        cv.putText(img, label, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    comparison = cv.hconcat(images)
    return comparison

def foreground(image, mask):
    return cv.bitwise_and(image, image, mask=mask)

def background(image, mask):
    return cv.bitwise_and(image, image, mask=cv.bitwise_not(mask))