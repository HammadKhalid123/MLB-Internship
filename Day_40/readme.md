# Day 40 - Smart Video Analytics System

## Overview

This project is a Smart Video Analytics System built using YOLOv8, OpenCV, Streamlit, and BoT-SORT. It processes a recorded video frame by frame, detects and tracks objects, displays real-time statistics, and records entry and exit events.

## How the System Works

The application allows the user to upload a recorded video through the Streamlit interface. Each video frame is processed using YOLOv8 for object detection and BoT-SORT for object tracking.

The system displays:

* Bounding boxes around detected objects
* Tracking IDs
* Current object count
* Unique object count
* Processing FPS
* ROI (Region of Interest)
* Entry and exit events

The processed video is saved as an output video, while detected events are stored in `events.csv`.

## Tracking IDs

BoT-SORT assigns a unique tracking ID to each detected object. The ID remains associated with the same object while it is being tracked across different frames.

For example, if a person is detected with ID `5`, the system continues to track that person using ID `5` in subsequent frames.

Tracking IDs are also used to calculate the number of unique objects in the complete video.

## Entry and Exit Detection

The user can define a Region of Interest (ROI) in the video.

The center point of each tracked object's bounding box is calculated and checked against the ROI.

* **Entry:** Object moves from outside the ROI to inside the ROI.
* **Exit:** Object moves from inside the ROI to outside the ROI.

Every detected event is saved in `events.csv` with:

* Track ID
* Event type
* Frame number
* Time in seconds

## FPS Results

FPS represents how many video frames the system can process per second.

The system calculates FPS based on the time required to process each frame, including object detection and tracking.

The observed FPS depends on the computer hardware, video resolution, number of objects, and YOLO model being used.

With YOLOv8n and BoT-SORT, the system provides a lightweight solution suitable for real-time or near-real-time video analytics on supported hardware.

## Best Configuration

The best configuration for this project was:

* Model: YOLOv8n
* Tracker: BoT-SORT
* Input: Recorded video
* Tracking: Persistent tracking IDs
* ROI: User-defined or full video
* Processing: Frame-by-frame

YOLOv8n was selected because it provides a good balance between detection speed and accuracy, while BoT-SORT provides reliable object tracking with persistent IDs.

## Problems Faced and Solutions

### 1. Maintaining Object IDs

Objects can move between frames, so simple object detection cannot determine whether an object is new or previously detected.

**Solution:** BoT-SORT was used to maintain persistent tracking IDs across frames.

### 2. Detecting Entry and Exit

It was necessary to determine when an object entered or left the selected ROI.

**Solution:** The center point of each bounding box was tracked. The previous ROI state of each tracking ID was compared with its current state to detect outside-to-inside and inside-to-outside transitions.

### 3. Counting Unique Objects

Counting detections in every frame would count the same object multiple times.

**Solution:** Tracking IDs were stored in a set. This allowed the system to count each tracked object only once.

### 4. Processing Speed

Running object detection and tracking on every frame can be computationally expensive.

**Solution:** YOLOv8n was used because it is a lightweight model and provides faster inference compared to larger YOLO models.

### 5. Saving Events

Entry and exit information needed to be stored for later analysis.

**Solution:** All detected events are automatically written to `events.csv` with the tracking ID, event type, frame number, and timestamp.
