# Day 16 - OpenCV Image Processing Fundamentals

Today I learned and implemented basic image processing operations using **Python, OpenCV, and NumPy**.

## Topics Covered

### 1. Reading and Displaying Images

Used `cv2.imread()` to read images and `cv2.imshow()` to display them.

### 2. Image Dimensions

Used the image shape to find the **width, height, and number of channels**.

### 3. Image Channels

Checked the number of channels in an image using the image shape.

### 4. RGB to Grayscale

Converted a color image into a grayscale image using:

`cv2.cvtColor()` with `cv2.COLOR_BGR2GRAY`.

### 5. Image Resizing

Resized images using `cv2.resize()`.

Different resolutions were also implemented:

* SD (480p)
* HD (720p)
* Full HD (1080p)
* 2K
* 4K

`cv2.INTER_CUBIC` interpolation was used for resizing.

### 6. Image Cropping

Cropped a specific region of an image using NumPy array slicing.

### 7. Image Rotation

Implemented image rotation at:

* 90° clockwise
* 180°
* 270° clockwise / 90° counterclockwise

Using `cv2.rotate()`.

### 8. Image Flipping

Implemented:

* Horizontal flipping
* Vertical flipping

Using `cv2.flip()`.

### 9. Drawing Shapes

Learned how to draw different shapes on images using OpenCV:

* Rectangle using `cv2.rectangle()`
* Circle using `cv2.circle()`
* Line using `cv2.line()`
* Polygon using `cv2.polylines()`

### 10. Adding Text

Added text to an image using `cv2.putText()`.

## Technologies Used

* **Python**
* **OpenCV**
* **NumPy**

---

## Key Learning

This task helped me understand the fundamentals of **image processing using OpenCV**.

I learned how to read, display, resize, crop, rotate, flip, and convert images to grayscale. I also learned how to draw shapes and add text to images.

These basic operations are important building blocks for more advanced **Computer Vision and Image Processing applications**.
