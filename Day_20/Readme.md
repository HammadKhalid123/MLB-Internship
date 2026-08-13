# Video Processing with OpenCV

## Overview

This project demonstrates basic video processing using OpenCV. It reads a video frame by frame, applies image processing techniques, displays the processed frames, and saves the output video.

## How OpenCV Reads Videos

OpenCV uses `cv2.VideoCapture()` to open a video. The `cap.read()` function reads one frame at a time and returns:

* `ret` – indicates whether the frame was successfully read.
* `frame` – the current video frame.

The process continues until the video ends or the user presses `q`.

## FPS

FPS means **Frames Per Second**. It represents how many frames are displayed or processed in one second.

For example, a 30 FPS video contains approximately 30 frames per second. FPS is important for smooth video playback and real-time processing.

## Processing Techniques

The following techniques were applied to video frames:

* **Grayscale Conversion** – converted color frames into grayscale.
* **Canny Edge Detection** – detected edges using lower and upper thresholds of 100 and 200.
* **Frame-by-Frame Processing** – processed each video frame individually.
* **Video Writing** – saved the processed frames as a new MP4 video.

## Challenges / Blockers

* Understanding how video frames are read and processed one at a time.
* Understanding the difference between video FPS and processing FPS.
* `CAP_PROP_FRAME_COUNT` returned `-1` for one test video because its total frame count was not available from the video metadata.
* Ensuring that the output video uses the correct FPS, width, height, and frame format.
