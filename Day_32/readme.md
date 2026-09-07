# Smart Parking Monitoring System

A Streamlit app that detects vehicles in drone/CCTV footage and reports parking spot occupancy in real time.

## How Parking Spaces Are Defined

- The first frame of the uploaded video is extracted and shown on an interactive canvas.
- The user clicks 3+ corner points per parking spot (polygon, not just a rectangle) and clicks **Save Spot**.
- Each saved spot is stored as a list of `(x, y)` coordinates in `st.session_state.spots`, scaled back from canvas size to original video resolution.

## How Vehicle Detection Works

- Detection uses a YOLOv8 model (`yolov8n-visdrone.pt`) trained on the VisDrone dataset.
- Since vehicles can be very small in aerial footage, each frame is processed with **SAHI-style slicing**: the frame is cut into overlapping tiles (`slice_width`/`slice_height`, `overlap_ratio` — configurable in the UI), and YOLO runs on each tile separately.
- Detected boxes from all tiles are converted back to original frame coordinates and merged.
- Only vehicle classes are kept, filtered by `VEHICLE_CLASSES = [3, 4, 5, 8, 9]` which map to `car, van, truck, bus, motor` in the VisDrone class list (not COCO IDs).

## How Occupied/Free Status Is Calculated

- For each parking spot polygon, the center point of every detected vehicle box is checked against the polygon using `cv2.pointPolygonTest`.
- If any vehicle's center falls inside a spot's polygon, that spot is marked **Occupied**; otherwise **Free**.

## How Duplicate Vehicle Detection Is Handled

- Because slices overlap, the same vehicle can be detected multiple times across adjacent tiles.
- After merging all tile detections, **Non-Maximum Suppression (NMS)** is applied per class: detections are sorted by confidence, and any detection with IoU above the threshold (0.3) against a higher-confidence detection of the same class is discarded.
- Additionally, YOLO's built-in tracker (ByteTrack, `persist=True`) assigns consistent IDs across frames, which helps avoid re-counting the same vehicle as new across time.

## How Occupancy Percentage Is Calculated

```
occupancy % = (occupied_spaces / total_spaces) * 100
```

Calculated per processed frame and plotted over time in **Parking Analytics** mode via `st.line_chart`.

## Difference Between the Two Demo Modes

| | Parking Monitor | Parking Analytics |
|---|---|---|
| Vehicle detection boxes | ✅ Shown | ✅ Shown |
| Parking spot polygons + status | ✅ Shown | ✅ Shown |
| Occupancy status label ("Low/Medium/High") | ❌ | ✅ |
| Occupancy progress bar | ❌ | ✅ |
| Occupancy-over-time line chart (after processing) | ❌ | ✅ |

Both modes run the same detection pipeline; Analytics mode adds extra statistics overlays and a post-run trend chart.

## Challenges and Limitations

- **Small object detection**: Vehicles in aerial footage are tiny, making detection unreliable without slicing — but slicing adds computation cost and can split a single vehicle across tile boundaries, causing missed or duplicate detections at edges.
- **Class ID mismatch risk**: The model is VisDrone-trained, not COCO-trained. Using COCO class IDs (as in generic YOLO tutorials) silently detects the wrong object classes.
- **Speed vs. accuracy tradeoff**: Frame skipping and larger slice sizes speed up processing but can miss short-duration events and reduce detection of small/edge vehicles.
- **Static camera assumption**: Parking spots are marked once from the first frame; the system does not handle camera movement or angle changes mid-video.
- **Occupancy by center-point only**: A vehicle partially overlapping a spot (but centered outside it) won't mark that spot occupied, which can misclassify edge-parked vehicles.
- **Browser video playback**: Default OpenCV `mp4v` output isn't browser-playable; the app re-encodes with `imageio` + `libx264` to fix this, at some added processing time.