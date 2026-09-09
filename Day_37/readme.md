# Day 37 – ROI, Line Crossing & Region Counting

## Overview

Today I worked on Region of Interest (ROI), Line Crossing, and Region Counting using YOLO object detection and tracking.

The main focus was to understand how selected points can be used in a video to define specific regions and how these regions can be used for person counting.

## Topics Covered

* Region of Interest (ROI)
* Selecting points on a video frame
* Drawing points and polygons on video frames
* Line Crossing
* Region Counting
* Person center-point calculation
* Point-in-polygon detection
* YOLO person tracking with ByteTrack
* Streamlit-based point selection

## Streamlit Point Selection

A Streamlit interface was created where the user can upload a video and select points directly on the video frame.

The selected points are stored as `(x, y)` coordinates and can be used to create a custom ROI.

## ROI

A Region of Interest defines a specific area of the video that should be monitored.

The person's center point is calculated from the YOLO bounding box:

```python
center_x = (x1 + x2) // 2
center_y = (y1 + y2) // 2
```

The center point is then checked to determine whether the person is inside the selected ROI.

## Line Crossing

Line Crossing is used to detect when a tracked person crosses a specific line.

It is useful for:

* Entry counting
* Exit counting
* People flow analysis
* Traffic monitoring

## Region Counting

Region Counting determines how many people are currently inside a defined region.

The general workflow is:

```text
Video
↓
YOLO Detection
↓
Person Detection
↓
ByteTrack Tracking
↓
Tracking ID
↓
Person Center Point
↓
Check ROI
↓
Region Count
```

## Difference Between Line Crossing and Region Counting

| Line Crossing                   | Region Counting                              |
| ------------------------------- | -------------------------------------------- |
| Uses a line                     | Uses an area                                 |
| Detects crossing events         | Detects people inside an area                |
| Useful for IN/OUT counting      | Useful for occupancy                         |
| Requires movement across a line | Checks whether a person is inside the region |

## Technologies Used

* Python
* OpenCV
* Ultralytics YOLO
* ByteTrack
* NumPy
* Streamlit
* Streamlit Drawable Canvas

## Outcome

By the end of Day 37, I understood how to allow users to select points on a video using Streamlit and how those points can be used to create ROIs, counting lines, and region-based person counting systems.
