# Day 31 - Vehicle Counting System

This project implements a vehicle counting system using OpenCV and YOLOv8 with ByteTrack.

## How Vehicle Counting Works

A counting line is drawn on the road by clicking two points on the video. YOLOv8 detects vehicles in each frame, while the tracking system follows their movement. When a vehicle crosses the counting line, it is counted.

## How Tracking IDs Prevent Duplicate Counting

Each detected vehicle is assigned a unique tracking ID. The ID remains associated with the same vehicle across multiple frames. A vehicle is counted only once when its tracking ID crosses the counting line, preventing duplicate counting.

## Vehicle Types Counted

The system can detect and count different vehicle classes supported by the YOLOv8 model, including:

* Car
* Motorcycle
* Bus
* Truck

## Challenges Faced

Some challenges included selecting an appropriate counting line position, maintaining accurate tracking when vehicles overlap, and handling changes in vehicle size and position across video frames. Choosing a suitable confidence threshold and tracker settings was also important for reliable detection and tracking.
