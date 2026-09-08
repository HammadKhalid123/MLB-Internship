import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


class TrafficMonitor:
    def __init__(self, model_path="yolov8n.pt", confidence=0.4, allowed_direction="RIGHT",
                 restricted_zone=None, zone_type="Polygon", frame_width=640, frame_height=480,
                 movement_threshold=4, history_length=20):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.allowed_direction = allowed_direction
        self.restricted_zone = restricted_zone if restricted_zone else []
        self.zone_type = zone_type
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.movement_threshold = movement_threshold
        self.history_length = history_length

        self.track_history = defaultdict(list)

        # Real-time state per track id (recomputed every frame) -> used for drawing
        self.currently_wrong_way = set()
        self.currently_in_zone = set()

        # "Ever violated" sets -> used for logging / stats (edge-triggered, one entry per episode)
        self.wrong_way_ids = set()
        self.restricted_ids = set()

        self.violations = []
        self.class_counts = defaultdict(int)
        self.counted_ids = set()
        self.total_vehicles_ids = set()
        self.frame_index = 0

    def _get_direction(self, track_id):
        history = self.track_history[track_id]
        if len(history) < 2:
            return "Unknown", (0, 0)
        window = history[-self.history_length:]
        start_x, start_y = window[0]
        end_x, end_y = window[-1]
        dx = end_x - start_x
        dy = end_y - start_y
        if abs(dx) < self.movement_threshold and abs(dy) < self.movement_threshold:
            return "STOPPED", (0, 0)
        if abs(dx) >= abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        return direction, (dx, dy)

    def _point_side_of_line(self, p1, p2, point):
        return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])

    def _check_restricted_zone(self, track_id, center):
        if len(self.restricted_zone) < 2:
            return False
        if self.zone_type == "Polygon" and len(self.restricted_zone) >= 3:
            polygon = np.array(self.restricted_zone, dtype=np.int32)
            result = cv2.pointPolygonTest(polygon, (float(center[0]), float(center[1])), False)
            return result >= 0
        elif self.zone_type == "Line" and len(self.restricted_zone) >= 2:
            history = self.track_history[track_id]
            if len(history) < 2:
                return False
            prev_point = history[-2]
            for i in range(len(self.restricted_zone) - 1):
                p1 = self.restricted_zone[i]
                p2 = self.restricted_zone[i + 1]
                prev_side = self._point_side_of_line(p1, p2, prev_point)
                curr_side = self._point_side_of_line(p1, p2, center)
                if prev_side == 0 or curr_side == 0:
                    continue
                crossed = (prev_side > 0) != (curr_side > 0)
                if crossed:
                    x_min, x_max = min(p1[0], p2[0]) - 25, max(p1[0], p2[0]) + 25
                    y_min, y_max = min(p1[1], p2[1]) - 25, max(p1[1], p2[1]) + 25
                    if x_min <= center[0] <= x_max and y_min <= center[1] <= y_max:
                        return True
            return False
        return False

    def draw_zone(self, frame):
        if len(self.restricted_zone) >= 2:
            pts = np.array(self.restricted_zone, dtype=np.int32).reshape((-1, 1, 2))
            if self.zone_type == "Polygon" and len(self.restricted_zone) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
            else:
                cv2.polylines(frame, [pts], False, (0, 0, 255), 3)
            label_pos = tuple(self.restricted_zone[0])
            cv2.putText(frame, "RESTRICTED ZONE", (label_pos[0], max(20, label_pos[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    def process_frame(self, frame):
        self.frame_index += 1
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASSES.keys()),
            conf=self.confidence,
            verbose=False
        )
        detections = []
        active_ids = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            for box, cls, tid in zip(boxes, classes, ids):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(tid)
                class_id = int(cls)
                class_name = VEHICLE_CLASSES.get(class_id, "Vehicle")
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                active_ids.add(track_id)

                self.total_vehicles_ids.add(track_id)
                if track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.class_counts[class_name] += 1

                self.track_history[track_id].append(center)
                if len(self.track_history[track_id]) > self.history_length:
                    self.track_history[track_id].pop(0)

                # --- Direction check (real-time, recalculated every frame against the
                #     CURRENTLY selected allowed_direction) ---
                direction, vector = self._get_direction(track_id)
                is_wrong_way_now = direction not in ("STOPPED", "Unknown") and direction != self.allowed_direction

                if is_wrong_way_now:
                    self.currently_wrong_way.add(track_id)
                    # Log a violation only on the transition into the wrong-way state,
                    # so a vehicle already flagged doesn't spam the violation log every frame.
                    if track_id not in self.wrong_way_ids:
                        self.wrong_way_ids.add(track_id)
                        self.violations.append({
                            "id": track_id,
                            "type": "Wrong Way",
                            "class": class_name,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "frame": self.frame_index
                        })
                else:
                    self.currently_wrong_way.discard(track_id)

                # --- Restricted zone check (real-time) ---
                in_restricted_now = self._check_restricted_zone(track_id, center)
                if in_restricted_now:
                    self.currently_in_zone.add(track_id)
                    if track_id not in self.restricted_ids:
                        self.restricted_ids.add(track_id)
                        self.violations.append({
                            "id": track_id,
                            "type": "Restricted Zone",
                            "class": class_name,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "frame": self.frame_index
                        })
                else:
                    self.currently_in_zone.discard(track_id)

                detections.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "class": class_name,
                    "center": center,
                    "direction": direction,
                    "vector": vector,
                    "is_wrong_way": track_id in self.currently_wrong_way,
                    "in_restricted": track_id in self.currently_in_zone
                })

        # Clean up real-time state for tracks that vanished this frame so stale
        # flags don't linger if the id ever gets reused.
        for stale_id in list(self.currently_wrong_way - active_ids):
            self.currently_wrong_way.discard(stale_id)
        for stale_id in list(self.currently_in_zone - active_ids):
            self.currently_in_zone.discard(stale_id)

        return frame, detections

    def get_stats(self):
        wrong_way_count = len(self.wrong_way_ids)
        restricted_count = len(self.restricted_ids)
        return {
            "total_vehicles": len(self.total_vehicles_ids),
            "total_violations": wrong_way_count + restricted_count,
            "wrong_way_violations": wrong_way_count,
            "restricted_zone_violations": restricted_count,
            "class_counts": dict(self.class_counts),
            "violations": list(self.violations)
        }


def draw_arrow(frame, center, vector, color=(0, 0, 255), length=40):
    dx, dy = vector
    norm = (dx ** 2 + dy ** 2) ** 0.5
    if norm == 0:
        return frame
    dx, dy = dx / norm, dy / norm
    end_point = (int(center[0] + dx * length), int(center[1] + dy * length))
    cv2.arrowedLine(frame, center, end_point, color, 3, tipLength=0.4)
    return frame


def render_wrong_way_variant(frame, detections, monitor):
    frame = monitor.draw_zone(frame)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = (0, 0, 255) if det["is_wrong_way"] else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{det['id']} {det['class']}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, det["direction"], (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if det["is_wrong_way"]:
            draw_arrow(frame, det["center"], det["vector"], (0, 0, 255))
            cv2.putText(frame, "WRONG WAY", (x1, y2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        else:
            draw_arrow(frame, det["center"], det["vector"], (0, 200, 0))
        if det["in_restricted"]:
            cv2.putText(frame, "RESTRICTED VIOLATION", (x1, y2 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (30, 30, 30), -1)
    cv2.putText(frame, f"Allowed Direction: {monitor.allowed_direction}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Wrong Way Violations: {len(monitor.wrong_way_ids)}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Restricted Zone Violations: {len(monitor.restricted_ids)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
    return frame


def render_dashboard_variant(frame, detections, monitor):
    frame = monitor.draw_zone(frame)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = (0, 0, 255) if (det["is_wrong_way"] or det["in_restricted"]) else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{det['id']}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame