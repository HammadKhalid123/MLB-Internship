# Day 17 - Image Transformations & Image Enhancement

Today I learned and implemented **image transformation and image enhancement techniques** using **Python, OpenCV, and NumPy**.

## Image Transformations

### 1. Translation

Translation moves an image horizontally or vertically using `cv2.warpAffine()`.

### 2. Rotation

Rotation rotates an image by a specific angle using `cv2.getRotationMatrix2D()` and `cv2.warpAffine()`.

### 3. Scaling

Scaling changes the size of an image using `cv2.resize()`.

Both **scaling up** and **scaling down** were implemented.

### 4. Affine Transformation

Affine transformation transforms an image using **three corresponding source and destination points**.

It can change the position, rotation, scale, and shape of an image while preserving parallel lines.

### 5. Perspective Transformation

Perspective transformation transforms an image using **four corresponding points**.

It is useful for correcting tilted or distorted documents and was implemented using:

* `cv2.getPerspectiveTransform()`
* `cv2.warpPerspective()`

---

## Image Enhancement

### Brightness

Brightness adjusts how light or dark an image appears using `cv2.convertScaleAbs()`.

### Contrast

Contrast increases or decreases the difference between bright and dark areas of an image.

### Gaussian Blur

Gaussian Blur smooths the image and reduces noise using `cv2.GaussianBlur()`.

### Median Blur

Median Blur reduces noise while preserving edges. It is especially useful for **salt-and-pepper noise**.

### Bilateral Filter

Bilateral filtering reduces noise while preserving important edges using `cv2.bilateralFilter()`.

### Image Sharpening

Image sharpening enhances edges and makes details such as document text clearer using a sharpening kernel and `cv2.filter2D()`.

---

## Biggest Impact

**Perspective Transformation** had the biggest impact on document quality because it corrects the distortion caused when a document is photographed from an angle.

It makes the document appear straight and improves its readability, making it especially useful for **document processing and OCR applications**.

---

## Challenges

During this task, I faced and learned to handle the following challenges:

* Selecting the correct four points for perspective transformation.
* Understanding transformation matrices and their parameters.
* Choosing suitable blur kernel sizes.
* Balancing brightness, contrast, and sharpening without losing image details.
* Understanding the differences between various image enhancement filters.
* Understanding how different preprocessing techniques affect the final image quality.

---

## Conclusion

Day 17 provided practical experience with fundamental image transformation and enhancement techniques using **OpenCV and NumPy**.

The **Document Image Enhancement Tool** combined multiple techniques into a single pipeline to transform a photographed document into a cleaner, straighter, and more readable image.
