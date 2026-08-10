import cv2
import numpy as np
import os


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def correct_perspective(image, points, width=600, height=800):
    pts1 = np.float32(points)
    pts2 = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    corrected_image = cv2.warpPerspective(image, matrix, (width, height))
    return corrected_image


def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def reduce_noise(image):
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised_image


def enhance_brightness_contrast(image, brightness=20, contrast=1.2):
    enhanced_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return enhanced_image


def sharpen_image(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def save_image(image, output_path):
    output_folder = os.path.dirname(output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    success = cv2.imwrite(output_path, image)
    if not success:
        raise ValueError(f"Could not save image: {output_path}")
    print(f"Image saved successfully at: {output_path}")