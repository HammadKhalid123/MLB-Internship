# Day 36 — Traffic Violation Detection System

## Overview

This project implements a rule-based traffic violation detection system using YOLO object detection, ByteTrack object tracking, OpenCV, and vehicle movement analysis.

The system detects vehicles, assigns unique tracking IDs, calculates their movement direction, detects wrong-way movement, monitors restricted zones, detects virtual line crossings, and generates violation statistics.

## Features

* Vehicle detection using YOLO
* Multi-object tracking using ByteTrack
* Unique tracking IDs for vehicles
* Vehicle movement direction calculation
* Wrong-way movement detection
* Virtual line detection
* Entry and exit zone monitoring
* Restricted zone detection
* Rule-based violation detection
* Duplicate violation prevention
* Violation statistics and counting
* Processed video output

## How Vehicle Tracking Works

The system first uses YOLO to detect vehicles in every video frame.

Vehicle classes used by the system are:

* Car
* Motorcycle
* Bus
* Truck

ByteTrack is then used to track detected vehicles across consecutive frames.

Each tracked vehicle receives a unique tracking ID.

For example:

```text
Frame 1 → Car ID 1
Frame 2 → Car ID 1
Frame 3 → Car ID 1
Frame 4 → Car ID 1
```

The same ID allows the system to follow a vehicle throughout the video instead of treating it as a new vehicle in every frame.

The center point of each vehicle's bounding box is stored to analyze its movement.

## How Direction Is Calculated

The center point of each tracked vehicle is calculated using its bounding box:

```python
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

The current center position is compared with the previous position.

```python
dx = current_x - previous_x
dy = current_y - previous_y
```

Based on these changes, the system determines whether the vehicle is moving:

* RIGHT
* LEFT
* UP
* DOWN
* UNKNOWN

For example:

```text
Previous position → (100, 200)
Current position  → (150, 200)

dx = 150 - 100 = 50
```

Since `dx` is positive, the vehicle is moving to the RIGHT.

A movement threshold is also used to ignore very small movements caused by detection or tracking noise.

## How Wrong-Way Vehicles Are Identified

The allowed traffic direction is defined according to the road.

For example:

```python
ALLOWED_DIRECTION = "RIGHT"
```

If a vehicle is moving RIGHT, it follows the expected direction.

```text
Vehicle → → → → →
Allowed direction → RIGHT
Result → Normal
```

If the vehicle moves LEFT:

```text
Vehicle ← ← ← ← ←
Allowed direction → RIGHT
Result → WRONG WAY
```

The rule-based system compares the detected vehicle direction with the predefined allowed direction.

When the vehicle moves in the opposite direction, a wrong-way violation is generated.

## Virtual Lines

A virtual line is an imaginary line drawn over the video frame.

For example:

```text
          Vehicle
             ↓
             ↓
-------------------------
       Virtual Line
-------------------------
             ↓
```

The line can be defined using its coordinates.

Example:

```python
LINE_Y = 400
```

The system compares the vehicle's previous and current positions.

If the vehicle moves from one side of the line to the other, the system detects a line-crossing event.

Virtual lines can be used for:

* Entry detection
* Exit detection
* Road crossing detection
* Traffic monitoring
* Rule-based events

## How Restricted Zones Are Defined

A restricted zone is defined as a rectangular region inside the video frame.

Example:

```python
restricted_zone = (400, 200, 700, 400)
```

The values represent:

```text
x1 = 400
y1 = 200
x2 = 700
y2 = 400
```

The system checks whether the center point of a tracked vehicle is inside this rectangle.

If the vehicle enters the restricted area:

```text
Vehicle → Restricted Zone
              ↓
          Violation
```

A restricted-zone violation event is generated.

Restricted zones can represent:

* No-entry areas
* Emergency lanes
* Pedestrian areas
* Parking-restricted areas
* Other prohibited regions

## How Duplicate Violations Are Avoided

A vehicle can remain in a violation state for many consecutive frames.

For example, if a vehicle is moving in the wrong direction for 100 frames, the system should not count 100 separate violations.

To prevent duplicate violations, the system keeps track of vehicles that have already generated a specific violation.

A set can be used:

```python
violations = set()
```

Before registering a violation, the system checks whether the vehicle ID has already been recorded.

Example:

```python
if wrong_way and track_id not in violations:

    violations.add(track_id)

    print(f"Vehicle {track_id}: WRONG WAY")
```

This ensures that the same vehicle is not repeatedly counted for the same violation.

## How Violation Statistics Are Generated

The system maintains counters for detected violations.

For example:

```text
Total Vehicles: 25
Wrong-Way Violations: 4
Line Crossings: 8
Restricted-Zone Violations: 3
```

The statistics can be generated by incrementing counters whenever a new violation is detected.

Example:

```python
wrong_way_count = 0
line_crossing_count = 0
restricted_zone_count = 0
```

When a new violation occurs:

```python
wrong_way_count += 1
```

At the end of processing, these values provide a summary of the traffic violations detected in the video.

## Rule-Based Event Detection

The system uses predefined rules instead of trying to learn traffic laws automatically.

Examples:

```text
IF vehicle moves opposite to allowed direction
→ WRONG-WAY VIOLATION

IF vehicle crosses virtual line
→ LINE-CROSSING EVENT

IF vehicle enters restricted zone
→ RESTRICTED-ZONE VIOLATION
```

This makes the system configurable for different roads and traffic scenarios.

## Difference Between the Two Demo Variants

The project contains two demo variants demonstrating different approaches to traffic violation detection.

### Variant 1 — Direction and Line-Based Detection

This variant focuses mainly on:

* Vehicle tracking
* Movement direction
* Allowed traffic direction
* Wrong-way detection
* Virtual line crossing

It is useful for demonstrating how vehicle movement can be analyzed using tracking history and simple geometric rules.

### Variant 2 — Zone-Based Traffic Monitoring

This variant extends the concept by using defined regions such as:

* Entry zones
* Exit zones
* Restricted zones

It can identify when a vehicle enters a prohibited area and can combine zone information with vehicle direction and tracking data.

The first variant is simpler and focuses on movement and line-based rules, while the second variant provides more flexible region-based traffic monitoring.

## Challenges and Limitations

### 1. Camera Angle

The direction calculation depends heavily on the camera perspective.

A road viewed from an unusual angle can make vehicle movement difficult to interpret correctly.

### 2. Tracking Errors

Vehicles can temporarily lose their tracking IDs because of:

* Occlusion
* Heavy traffic
* Similar-looking vehicles
* Poor video quality
* Sudden movement

### 3. Detection Accuracy

YOLO detection accuracy can be affected by:

* Low-resolution videos
* Poor lighting
* Night-time conditions
* Weather
* Vehicle overlap

### 4. Fixed Rules

The system uses predefined rules such as:

```python
ALLOWED_DIRECTION = "RIGHT"
```

Therefore, the rules need to be configured according to the specific camera and road.

### 5. Perspective Problems

A vehicle moving toward or away from the camera may not produce a simple LEFT/RIGHT movement.

More advanced systems can use road perspective, homography, trajectories, or calibrated coordinates.

### 6. Virtual Line Limitations

A simple horizontal or vertical line may not accurately represent complex road boundaries.

For complicated roads, polygon-based regions and more advanced geometric calculations may be required.

### 7. False Violations

Temporary tracking errors or noisy movement can sometimes result in incorrect violation detection.

Using multiple frames, trajectory smoothing, and additional validation can reduce false positives.

## Technologies Used

* Python
* OpenCV
* Ultralytics YOLO
* ByteTrack
* NumPy

## Workflow

```text
Input Traffic Video
        ↓
YOLO Vehicle Detection
        ↓
ByteTrack Tracking
        ↓
Vehicle Tracking IDs
        ↓
Center Point Calculation
        ↓
Position History
        ↓
Direction Calculation
        ↓
Line / Zone Analysis
        ↓
Rule-Based Violation Detection
        ↓
Duplicate Violation Filtering
        ↓
Violation Statistics
        ↓
Processed Output Video
```

## Conclusion

Day 36 demonstrates how object detection and tracking can be combined with geometric rules to build a traffic violation detection system.

Instead of only detecting vehicles, the system analyzes their movement and interaction with predefined lines and zones to identify events such as wrong-way movement, line crossing, and restricted-zone entry.
