# Day 19 - Contours & Shape Detection

## Overview

Today I learned how to detect and analyze objects using OpenCV contours and built a basic shape detection system.

## What are Contours?

Contours are curves that represent the boundary or outline of an object in an image. They are useful for detecting, measuring, and analyzing objects.

## How Contour Detection Works

1. Read the image.
2. Convert it to grayscale.
3. Apply thresholding to create a binary image.
4. Use `cv2.findContours()` to detect boundaries.
5. Calculate contour area and perimeter.
6. Draw contours and bounding rectangles.
7. Use `approxPolyDP()` to identify shapes.

## Shapes Detected

* Triangle
* Square
* Rectangle
* Circle
* Polygon

The program also displays the **area and perimeter** of detected shapes and saves the final output.

## Challenges

* Choosing the correct threshold.
* Removing small unwanted contours caused by noise.
* Selecting a suitable epsilon value for `approxPolyDP()`.
* Distinguishing circles from polygons.

## Tools Used

* Python
* OpenCV
* NumPy
