import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = [3, 4, 5, 8]

def create_slices(image, slice_width=640, slice_height=640, overlap_width_ratio=0.4, overlap_height_ratio=0.4):
    image_height, image_width = image.shape[:2]
    step_x = int(slice_width * (1 - overlap_width_ratio))
    step_y = int(slice_height * (1 - overlap_height_ratio))
    slices = []
    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            x1 = x
            y1 = y
            x2 = min(x1 + slice_width, image_width)
            y2 = min(y1 + slice_height, image_height)
            slice_image = image[y1:y2, x1:x2]
            slices.append({
                "image": slice_image,
                "x_offset": x1,
                "y_offset": y1
            })
            if x2 == image_width:
                break
            x += step_x
        if y2 == image_height:
            break
        y += step_y
    return slices

def convert_to_original_coordinates(box, x_offset, y_offset):
    x1, y1, x2, y2 = box
    return [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset]

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area if union_area > 0 else 0

def non_maximum_suppression(detections, iou_threshold=0.3):
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    selected = []
    while detections:
        best = detections.pop(0)
        selected.append(best)
        remaining = []
        for detection in detections:
            if detection["class_id"] != best["class_id"]:
                remaining.append(detection)
                continue
            if calculate_iou(best["box"], detection["box"]) < iou_threshold:
                remaining.append(detection)
        detections = remaining
    return selected

def sahi_detection(image, model, slice_width=416, slice_height=416, overlap_width_ratio=0.4, overlap_height_ratio=0.4, confidence_threshold=0.10, iou_threshold=0.3):
    slices = create_slices(image, slice_width, slice_height, overlap_width_ratio, overlap_height_ratio)
    all_detections = []

    for slice_data in slices:
        slice_image = slice_data["image"]
        x_offset = slice_data["x_offset"]
        y_offset = slice_data["y_offset"]

        results = model.track(
            source=slice_image,
            conf=confidence_threshold,
            classes=VEHICLE_CLASSES,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]
        if result.boxes is None:
            continue

        for box in result.boxes:
            if box.id is None:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            original_box = convert_to_original_coordinates(xyxy, x_offset, y_offset)

            all_detections.append({
                "box": original_box,
                "confidence": float(box.conf[0].cpu().numpy()),
                "class_id": int(box.cls[0].cpu().numpy()),
                "track_id": int(box.id[0].cpu().numpy())
            })

    return non_maximum_suppression(all_detections, iou_threshold)

def draw_detections(image, detections, model):
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["box"])
        class_name = model.names[detection["class_id"]]
        label = f"{class_name} ID:{detection['track_id']}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(output, label, (x1, max(20, y1 - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    return output

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def get_box_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def check_parking_occupancy(detections, parking_spots):
    occupied_flags = []

    for spot in parking_spots:
        occupied = False
        for detection in detections:
            center = get_box_center(detection["box"])
            if point_in_polygon(center, spot):
                occupied = True
                break
        occupied_flags.append(occupied)

    return occupied_flags

def draw_parking_spots(frame, parking_spots, occupied_flags):
    output = frame.copy()

    for idx, spot in enumerate(parking_spots):
        pts = np.array(spot, dtype=np.int32)
        color = (0, 255, 0)
        cv2.polylines(output, [pts], True, color, 2)

        status = "Occupied" if occupied_flags[idx] else "Free"
        center = pts.mean(axis=0).astype(int)
        cv2.putText(output, f"{idx + 1}:{status}",
                   (int(center[0]) - 40, int(center[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output

def draw_statistics_panel(frame, total_spaces, occupied_spaces, available_spaces, occupancy_percentage, mode="Parking Monitor"):
    output = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    line_height = 18
    padding = 8

    lines = [
        (f"Total Spaces: {total_spaces}", (255, 255, 255)),
        (f"Occupied: {occupied_spaces}", (0, 0, 255)),
        (f"Available: {available_spaces}", (0, 255, 0)),
        (f"Occupancy: {occupancy_percentage:.1f}%", (255, 255, 0)),
    ]

    if mode == "Parking Analytics":
        utilization_status = "Low" if occupancy_percentage < 40 else "Medium" if occupancy_percentage < 70 else "High"
        status_color = (0, 255, 0) if occupancy_percentage < 40 else (0, 255, 255) if occupancy_percentage < 70 else (0, 0, 255)
        lines.append((f"Status: {utilization_status}", status_color))

    max_text_width = 0
    for text, _ in lines:
        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        max_text_width = max(max_text_width, text_width)

    panel_width = max_text_width + padding * 2
    panel_height = len(lines) * line_height + padding * 2

    x_start, y_start = 10, 10
    overlay = output.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_width, y_start + panel_height), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.6, output, 0.4, 0)

    y_offset = y_start + padding + 12
    for text, color in lines:
        cv2.putText(output, text, (x_start + padding, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += line_height

    if mode == "Parking Analytics":
        bar_x, bar_y = x_start, y_start + panel_height + 8
        bar_width, bar_height = panel_width, 12
        status_color = lines[-1][1]
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        fill_width = int((occupancy_percentage / 100) * bar_width)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), status_color, -1)

    return output