import cv2
import numpy as np

img = cv2.imread("input_images/img1.jpg")

if img is None:
    print("Error: Could not read the image.")
    exit()

# Show Image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Dimensions
image_height, image_width, _ = img.shape
print(f"Image Dimensions: Width={image_width}, Height={image_height}")

# Channels
if len(img.shape) == 3:
    channels = img.shape[2]
else:
    channels = 1

print(f"Channels: {channels}")

# RGB to Grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("output_images/gray_img1.jpg", gray_img)

# Resize Image
img2 = cv2.imread("input_images/img2.jpg")

if img2 is None:
    print("Error: Could not read the image.")
    exit()

resize_img = cv2.resize(img2, (200, 200))
cv2.imwrite("output_images/resize_img2.jpg", resize_img)

# Resize Image into Different Dimensions
img3 = cv2.imread("input_images/img3.jpg")

if img3 is None:
    print("Error: Could not read the image.")
    exit()

resolutions = {
    "SD_480p": (640, 480),
    "HD_720p": (1280, 720),
    "Full_HD_1080p": (1920, 1080),
    "2K": (2048, 1080),
    "4K": (3840, 2160),
}

for name, res in resolutions.items():
    resized_img = cv2.resize(
        img3,
        res,
        interpolation=cv2.INTER_CUBIC
    )

    output_path = f"output_images/resize_res_{name}.png"
    cv2.imwrite(output_path, resized_img)

    print(f"Saved {name} image at: {output_path}")

# Crop Image
crop_img = img3[100:400, 150:450]

cv2.imwrite("output_images/crop_img3.jpg", crop_img)
print("Saved cropped image at: output_images/crop_img3.jpg")

# Rotate Image
img4 = cv2.imread("input_images/img4.jpg")

if img4 is None:
    print("Error: Could not read the image.")
    exit()

img_90 = cv2.rotate(img4, cv2.ROTATE_90_CLOCKWISE)
img_180 = cv2.rotate(img4, cv2.ROTATE_180)
img_270 = cv2.rotate(img4, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imwrite("output_images/rotate_90_img4.jpg", img_90)
cv2.imwrite("output_images/rotate_180_img4.jpg", img_180)
cv2.imwrite("output_images/rotate_270_img4.jpg", img_270)

# Flip Image
img5 = cv2.imread("input_images/img5.jpg")

if img5 is None:
    print("Error: Could not read the image.")
    exit()

img_flip_horizontal = cv2.flip(img5, 1)
img_flip_vertical = cv2.flip(img5, 0)

cv2.imwrite(
    "output_images/flip_horizontal_img5.jpg",
    img_flip_horizontal
)

cv2.imwrite(
    "output_images/flip_vertical_img5.jpg",
    img_flip_vertical
)

# Draw Shapes on Image
cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), 3)

cv2.circle(img2, (350, 150), 60, (255, 0, 0), -1)

cv2.line(img, (50, 300), (450, 300), (0, 255, 0), 5)

points = np.array([
    [100, 50],
    [200, 80],
    [250, 180],
    [120, 220]
], np.int32)

cv2.polylines(
    img,
    [points],
    True,
    (0, 255, 255),
    3
)

# Text on Image
text_img3 = cv2.putText(
    img3,
    "Hammad",
    (40, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

cv2.imwrite("output_images/text_img3.jpg", text_img3)

print("Saved image with text at: output_images/text_img3.jpg")