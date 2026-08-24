# Day 26 – Introduction to Image Segmentation

## What is Image Segmentation?

Image segmentation is a Computer Vision technique that divides an image into meaningful regions or groups of pixels. Unlike object detection, segmentation identifies the exact pixels that belong to an object. It is commonly used in medical imaging, autonomous vehicles, agriculture, and image editing.

## Binary, Adaptive, and Otsu Thresholding

* **Binary Thresholding:** Uses one fixed threshold value for the entire image. Pixels above the threshold are converted to white, while pixels below it become black.
* **Adaptive Thresholding:** Calculates different threshold values for different regions of the image. It works better when the image has uneven lighting or shadows.
* **Otsu Thresholding:** Automatically calculates an optimal global threshold value based on the image histogram. It is useful when the foreground and background have clear intensity differences.

## Best Method for My Dataset

**Otsu Thresholding** worked best for my dataset because it automatically selected a suitable threshold value and produced better foreground/background separation without requiring manual threshold tuning. However, Adaptive Thresholding can perform better on images with uneven illumination.

## Challenges Faced

* Selecting the correct threshold value for different images.
* Handling noise and shadows during segmentation.
* Uneven lighting sometimes caused incorrect foreground/background separation.
* Finding the best preprocessing techniques to improve the segmentation results.
* Different images required different thresholding approaches.
