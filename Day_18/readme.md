# Day 18 - Edge Detection & Morphological Operations

## Overview

Day 18 focused on **Edge Detection and Morphological Operations** using OpenCV. The main goal was to detect edges, improve image structures, and detect document boundaries.

## Edge Detection

Three edge detection techniques were implemented:

* **Sobel:** Detects edges using first-order image gradients. Sobel X detects vertical edges, while Sobel Y detects horizontal edges.
* **Laplacian:** Uses a second-order derivative to detect rapid intensity changes. It can detect edges in multiple directions but is more sensitive to noise.
* **Canny:** A multi-stage edge detection algorithm that produces thin and cleaner edges. It gave the best results for document boundary detection.

## Morphological Operations

The following operations were implemented:

* **Erosion:** Shrinks white/foreground regions and removes small noise.
* **Dilation:** Expands white/foreground regions and connects broken parts.
* **Opening:** Erosion followed by dilation; mainly used to remove small noise.
* **Closing:** Dilation followed by erosion; fills small gaps and connects broken edges.
* **Morphological Gradient:** Difference between dilation and erosion; highlights object boundaries.
* **Top Hat:** Difference between the original image and opening; highlights small bright regions.
* **Black Hat:** Difference between closing and the original image; highlights small dark regions.

## Document Boundary Detection

The document boundary detection pipeline was:

```text
Grayscale
    ↓
Gaussian Blur
    ↓
Canny Edge Detection
    ↓
Morphological Closing
    ↓
Contour Detection
    ↓
Largest Contour
```

### Best Combination

The best results were achieved using **Grayscale + Gaussian Blur + Canny + Morphological Closing + Contour Detection**.

Canny provided cleaner edges, while Morphological Closing helped connect broken document boundaries.

## Challenges

The main challenges faced were:

* Uneven lighting and shadows created unwanted edges.
* Canny sometimes produced broken document edges.
* Background objects generated additional contours.
* Different document angles made boundary detection more difficult.
* Choosing suitable Canny threshold values affected the detection results.

## Technologies Used

* Python
* OpenCV
* NumPy
