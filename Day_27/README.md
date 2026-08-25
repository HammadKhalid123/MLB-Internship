# Day 27 - Smart Object Detection with YOLOv8

## Overview

This project demonstrates Object Detection using a pre-trained YOLOv8 model. The application detects multiple objects in images and videos, draws bounding boxes around them, and displays their class names and confidence scores.

The project processes:

- Images
- Multiple objects in a single image
- Short videos
- Frame-by-frame object detection

The processed images and videos are saved with the detected objects.

---

## What is Object Detection?

Object Detection is a Computer Vision task that identifies objects in an image or video and determines their locations.

For every detected object, the model provides:

- Class Name
- Class ID
- Confidence Score
- Bounding Box Coordinates

For example, an image may contain:

- Person
- Car
- Bus
- Dog

The model detects all objects and draws a separate bounding box around each object.

---

## Object Detection vs Image Classification

Image Classification predicts what an image contains and usually assigns a label to the complete image.

For example:

```text
Image → Car