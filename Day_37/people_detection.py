import cv2
import numpy as np
import subprocess
import imageio_ffmpeg
from ultralytics import YOLO


class PeopleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.person_class_id = 0

    def get_line_side(self, point, line_start, line_end):
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        value = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if value > 0:
            return 1
        elif value < 0:
            return -1
        return 0

    def point_in_polygon(self, point, polygon):
        if polygon is None or len(polygon) < 3:
            return False
        polygon_np = np.array(polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(polygon_np, point, False)
        return result >= 0

    def draw_label(self, frame, text, x, y, bg_color, text_color=(255, 255, 255)):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), bg_color, -1)
        cv2.putText(frame, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    def process_image(self, image, confidence=0.5, iou=0.45, box_color=(0, 255, 0),
                       box_thickness=2, roi_points=None):
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            classes=[self.person_class_id],
            verbose=False
        )
        result = results[0]
        annotated = image.copy()
        count = 0
        roi_count = 0

        if roi_points is not None and len(roi_points) >= 3:
            roi_np = np.array(roi_points, dtype=np.int32)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [roi_np], (0, 255, 255))
            annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)
            cv2.polylines(annotated, [roi_np], True, (0, 255, 255), 2)

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            count += 1
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            in_roi = False
            if roi_points is not None and len(roi_points) >= 3:
                in_roi = self.point_in_polygon((cx, cy), roi_points)
                if in_roi:
                    roi_count += 1

            color = (0, 165, 255) if in_roi else box_color
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
            self.draw_label(annotated, f"Person {conf:.2f}", x1, y1, color, (0, 0, 0))

        cv2.putText(annotated, f"People Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if roi_points is not None and len(roi_points) >= 3:
            cv2.putText(annotated, f"In ROI: {roi_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2)

        stats = {"count": count, "roi_count": roi_count}
        return annotated, stats

    def process_video(self, input_path, output_path, confidence=0.5, iou=0.45,
                       line_points=None, roi_points=None, tracker="bytetrack.yaml",
                       box_color=(0, 255, 0), box_thickness=2, show_ids=True,
                       show_trail=True, trail_length=30, progress_callback=None):

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        track_history = {}
        track_side = {}
        entry_count = 0
        exit_count = 0
        max_count = 0
        unique_ids_in_roi = set()
        unique_ids_seen = set()
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                source=frame,
                conf=confidence,
                iou=iou,
                classes=[self.person_class_id],
                persist=True,
                tracker=tracker,
                verbose=False
            )
            result = results[0]
            annotated = frame.copy()

            if line_points is not None and len(line_points) == 2:
                cv2.line(annotated, line_points[0], line_points[1], (0, 0, 255), 3)
                cv2.circle(annotated, line_points[0], 6, (0, 0, 255), -1)
                cv2.circle(annotated, line_points[1], 6, (0, 0, 255), -1)

            if roi_points is not None and len(roi_points) >= 3:
                roi_np = np.array(roi_points, dtype=np.int32)
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [roi_np], (0, 255, 255))
                annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)
                cv2.polylines(annotated, [roi_np], True, (0, 255, 255), 2)

            current_count = 0
            roi_live_count = 0

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                current_count = len(boxes)

                for box, track_id, conf in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    unique_ids_seen.add(track_id)

                    in_roi = False
                    if roi_points is not None and len(roi_points) >= 3:
                        in_roi = self.point_in_polygon((cx, cy), roi_points)
                        if in_roi:
                            roi_live_count += 1
                            unique_ids_in_roi.add(track_id)

                    color = (0, 165, 255) if in_roi else box_color
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

                    label = f"Person {track_id} {conf:.2f}" if show_ids else f"Person {conf:.2f}"
                    self.draw_label(annotated, label, x1, y1, color, (0, 0, 0))
                    cv2.circle(annotated, (cx, cy), 4, (255, 0, 255), -1)

                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append((cx, cy))
                    if len(track_history[track_id]) > trail_length:
                        track_history[track_id].pop(0)

                    if show_trail:
                        pts = track_history[track_id]
                        for i in range(1, len(pts)):
                            cv2.line(annotated, pts[i - 1], pts[i], (255, 0, 255), 1)

                    if line_points is not None and len(line_points) == 2:
                        side = self.get_line_side((cx, cy), line_points[0], line_points[1])
                        prev_side = track_side.get(track_id)
                        if prev_side is not None and side != 0 and prev_side != side:
                            if side > 0:
                                entry_count += 1
                            else:
                                exit_count += 1
                        if side != 0:
                            track_side[track_id] = side

            max_count = max(max_count, current_count)

            info_y = 40
            cv2.putText(annotated, f"People Count: {current_count}", (20, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            info_y += 35
            cv2.putText(annotated, f"Max Count: {max_count}", (20, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 140, 0), 2)

            if line_points is not None and len(line_points) == 2:
                info_y += 35
                cv2.putText(annotated, f"Entry: {entry_count}  Exit: {exit_count}",
                            (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if roi_points is not None and len(roi_points) >= 3:
                info_y += 35
                cv2.putText(annotated, f"In ROI: {roi_live_count}", (20, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)

            writer.write(annotated)
            frame_index += 1

            if progress_callback is not None and total_frames > 0:
                progress_callback(min(frame_index / total_frames, 1.0))

        cap.release()
        writer.release()

        stats = {
            "max_count": max_count,
            "entry_count": entry_count,
            "exit_count": exit_count,
            "unique_people_seen": len(unique_ids_seen),
            "unique_in_roi": len(unique_ids_in_roi),
            "total_frames": frame_index,
            "fps": fps
        }
        return stats


def convert_to_h264(input_path, output_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, "-y", "-i", input_path,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(command, check=True, capture_output=True)