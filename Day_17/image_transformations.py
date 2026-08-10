import cv2
import numpy as np

# Translation

img1 = cv2.imread("sample_images/test1.jpg")

if img1 is None:
    print("Error: Could not read the image.")
    exit()

cv2.imshow("Original Image - Translation", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

height, width = img1.shape[:2]

tx = 100
ty = 50

M = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

translated_img = cv2.warpAffine(
    img1,
    M,
    (width, height)
)

cv2.imwrite(
    "output_images/translated_img1.jpg",
    translated_img
)

print(
    "Saved translated image at: "
    "output_images/translated_img1.jpg"
)

# Rotation

img2 = cv2.imread("sample_images/test2.jpg")

if img2 is None:
    print("Error: Could not read the image.")
    exit()

cv2.imshow("Original Image - Rotation", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

height, width = img2.shape[:2]

center = (
    width // 2,
    height // 2
)

angle = 45
scale = 1.0

M = cv2.getRotationMatrix2D(
    center,
    angle,
    scale
)

rotated_img = cv2.warpAffine(
    img2,
    M,
    (width, height)
)

cv2.imwrite(
    "output_images/rotated_img2.jpg",
    rotated_img
)

print(
    "Saved rotated image at: "
    "output_images/rotated_img2.jpg"
)

# Scaling

img3 = cv2.imread("sample_images/test3.jpg")

if img3 is None:
    print("Error: Could not read the image.")
    exit()

cv2.imshow("Original Image - Scaling", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

img3_scaled_down = cv2.resize(
    img3,
    None,
    fx=0.5,
    fy=0.5
)

cv2.imwrite(
    "output_images/scale_down_img3.jpg",
    img3_scaled_down
)

print(
    "Saved scaled down image at: "
    "output_images/scale_down_img3.jpg"
)

img3_scaled_up = cv2.resize(
    img3,
    None,
    fx=2,
    fy=2
)

cv2.imwrite(
    "output_images/scale_up_img3.jpg",
    img3_scaled_up
)

print(
    "Saved scaled up image at: "
    "output_images/scale_up_img3.jpg"
)

# Affine Transformation

img4 = cv2.imread("sample_images/test4.jpg")

if img4 is None:
    print("Error: Could not read the image.")
    exit()

cv2.imshow("Original Image - Affine", img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

height, width = img4.shape[:2]

pts1 = np.float32([
    [50, 50],
    [200, 50],
    [50, 200]
])

pts2 = np.float32([
    [10, 100],
    [200, 50],
    [100, 250]
])

M = cv2.getAffineTransform(
    pts1,
    pts2
)

affine_transformed_img4 = cv2.warpAffine(
    img4,
    M,
    (width, height)
)

cv2.imwrite(
    "output_images/affine_transformed_img4.jpg",
    affine_transformed_img4
)

print(
    "Saved affine transformed image at: "
    "output_images/affine_transformed_img4.jpg"
)


# Perspective Transformation

img5 = cv2.imread("sample_images/test5.jpg")

if img5 is None:
    print("Error: Could not read the image.")
    exit()

cv2.imshow("Original Image - Perspective", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()


pts1 = np.float32([
    [10, 80],
    [200, 100],
    [350, 450],
    [80, 420]
])

pts2 = np.float32([
    [0, 0],
    [600, 0],
    [600, 400],
    [0, 400]
])

M = cv2.getPerspectiveTransform(
    pts1,
    pts2
)

perspective_transformed_img5 = cv2.warpPerspective(
    img5,
    M,
    (600, 400)
)

cv2.imwrite(
    "output_images/perspective_transformed_img5.jpg",
    perspective_transformed_img5
)

print(
    "Saved perspective transformed image at: "
    "output_images/perspective_transformed_img5.jpg"
)