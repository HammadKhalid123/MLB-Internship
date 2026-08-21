import cv2 as cv
import numpy as np

def harris_corner_detection(image_path):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv.dilate(dst, None)

    num_corners = np.sum(dst > 0.01 * dst.max())

    img_result = img.copy()
    img_result[dst > 0.01 * dst.max()] = [0, 0, 255]

    return img_result, num_corners


def orb_keypoint_detection(image_path):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    img_keypoints = cv.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

    return img_keypoints, keypoints, descriptors