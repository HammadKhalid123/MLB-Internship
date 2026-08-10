import cv2
import numpy as np


# Load image for enhancement
img1 = cv2.imread("sample_images/test1.jpg")

if img1 is None:
    print("Error: Could not read the image.")
    exit()


# Increase contrast
img1_enhanced = cv2.convertScaleAbs(
    img1,
    alpha=1.5,
    beta=0
)


# Increase brightness
img1_enhanced_bright = cv2.convertScaleAbs(
    img1,
    alpha=1.0,
    beta=50
)


cv2.imwrite(
    "output_images/enhanced_img1.jpg",
    img1_enhanced
)

cv2.imwrite(
    "output_images/enhanced_bright_img1.jpg",
    img1_enhanced_bright
)


# Gaussian Blur
gaussian_image = cv2.GaussianBlur(
    img1,
    (15, 15),
    0
)

cv2.imwrite(
    "output_images/gaussian_blur_img1.jpg",
    gaussian_image
)

print(
    "Saved enhanced and blurred images at: "
    "output_images/enhanced_img1.jpg and "
    "output_images/gaussian_blur_img1.jpg"
)


# Load image for blur and sharpening
img3 = cv2.imread("sample_images/test3.jpg")

if img3 is None:
    print("Error: Could not read the image.")
    exit()


# Median Blur
median_blur_image = cv2.medianBlur(
    img3,
    5
)

cv2.imwrite(
    "output_images/median_blur_img3.jpg",
    median_blur_image
)

print(
    "Saved median blurred image at: "
    "output_images/median_blur_img3.jpg"
)


# Bilateral Filter
bilateral_filtered_image = cv2.bilateralFilter(
    img3,
    9,
    75,
    75
)

cv2.imwrite(
    "output_images/bilateral_filter_img3.jpg",
    bilateral_filtered_image
)

print(
    "Saved bilateral filtered image at: "
    "output_images/bilateral_filter_img3.jpg"
)


# Image Sharpening
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

sharpened_image = cv2.filter2D(
    img3,
    -1,
    kernel
)

cv2.imwrite(
    "output_images/sharpened_img3.jpg",
    sharpened_image
)

print(
    "Saved sharpened image at: "
    "output_images/sharpened_img3.jpg"
)