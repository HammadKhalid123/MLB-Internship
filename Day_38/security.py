import cv2
import numpy as np
import pandas as pd
import imageio
from ultralytics import YOLO

def load_model(model_name="yolov8s.pt"):
    return YOLO(model_name)

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not ret:
        return None, width, height, fps
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, width, height, fps

def point_in_polygon(point, polygon):
    polygon_np = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon_np, point, False)
    return result >= 0

class ROIEventTracker:
    def __init__(self, rois, stable_frames=2):
        self.rois = rois
        self.stable_frames = stable_frames
        self.states = {}
        self.events = []
        self.unique_ids = set()
        self.max_active = 0

    def register_id(self, track_id):
        self.unique_ids.add(track_id)

    def update(self, track_id, center, frame_idx, timestamp):
        for roi_idx, polygon in enumerate(self.rois):
            key = (track_id, roi_idx)
            inside_now = point_in_polygon(center, polygon)
            if key not in self.states:
                self.states[key] = {
                    "inside": False,
                    "pending": None,
                    "pending_count": 0,
                    "entry_time": None
                }
            state = self.states[key]
            if inside_now != state["inside"]:
                if state["pending"] == inside_now:
                    state["pending_count"] += 1
                else:
                    state["pending"] = inside_now
                    state["pending_count"] = 1
                if state["pending_count"] >= self.stable_frames:
                    state["inside"] = inside_now
                    state["pending"] = None
                    state["pending_count"] = 0
                    if inside_now:
                        state["entry_time"] = timestamp
                        self.events.append({
                            "track_id": track_id,
                            "roi": roi_idx,
                            "event": "enter",
                            "frame": frame_idx,
                            "time": round(timestamp, 2)
                        })
                    else:
                        entry_time = state["entry_time"]
                        duration = round(timestamp - entry_time, 2) if entry_time is not None else None
                        self.events.append({
                            "track_id": track_id,
                            "roi": roi_idx,
                            "event": "exit",
                            "frame": frame_idx,
                            "time": round(timestamp, 2),
                            "entry_time": round(entry_time, 2) if entry_time is not None else None,
                            "duration": duration
                        })
                        state["entry_time"] = None
            else:
                state["pending"] = None
                state["pending_count"] = 0

    def active_count(self, roi_idx=None):
        count = 0
        for (track_id, r_idx), state in self.states.items():
            if state["inside"]:
                if roi_idx is None or r_idx == roi_idx:
                    count += 1
        return count

    def refresh_max_active(self):
        current = self.active_count()
        if current > self.max_active:
            self.max_active = current

def parse_polygon_objects(objects):
    polygons = []
    for obj in objects:
        if "path" in obj:
            points = []
            for command in obj["path"]:
                if len(command) >= 3:
                    points.append([command[1], command[2]])
            if len(points) >= 3:
                polygons.append(points)
    return polygons

def scale_polygons(polygons, scale_x, scale_y):
    scaled = []
    for polygon in polygons:
        scaled.append([[p[0] * scale_x, p[1] * scale_y] for p in polygon])
    return scaled

def process_video(video_path, rois, model, conf=0.35, frame_skip=1, stable_frames=2, imgsz=960, output_path="output.mp4", progress_callback=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = ROIEventTracker(rois, stable_frames)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            timestamp = frame_idx / fps
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=conf,
                imgsz=imgsz,
                iou=0.5,
                tracker="bytetrack.yaml",
                verbose=False
            )
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    tracker.register_id(int(track_id))
                    tracker.update(int(track_id), (cx, cy), frame_idx, timestamp)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            tracker.refresh_max_active()
            for polygon in rois:
                pts = np.array(polygon, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
            active = tracker.active_count()
            cv2.putText(frame, f"Active in ROI: {active}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
        if progress_callback and total_frames > 0:
            progress_callback(min(frame_idx / total_frames, 1.0))
    cap.release()
    writer.close()
    entries = len([e for e in tracker.events if e["event"] == "enter"])
    exits = len([e for e in tracker.events if e["event"] == "exit"])
    summary = {
        "unique_people": len(tracker.unique_ids),
        "entries": entries,
        "exits": exits,
        "max_active": tracker.max_active
    }
    return tracker.events, summary

def save_events_csv(events, csv_path="events.csv"):
    df = pd.DataFrame(events)
    df.to_csv(csv_path, index=False)
    return csv_path