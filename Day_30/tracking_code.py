import time
from pathlib import Path
from collections import defaultdict

import cv2
import imageio.v2 as imageio
from ultralytics import YOLO


def load_model(model_path="yolov8n.pt"):
    return YOLO(model_path)


def list_sample_videos(input_dir="input_videos"):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return []
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in extensions],
        key=lambda p: p.name
    )


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if fps <= 0:
        fps = 30

    duration = total_frames / fps if fps else 0

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_seconds": duration
    }


def process_video(
    video_path,
    model=None,
    output_dir="saved_videos",
    tracker="bytetrack.yaml",
    conf=0.5,
    progress_callback=None
):
    if model is None:
        model = load_model()

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{video_path.stem}_tracked.mp4"

    metadata = get_video_metadata(video_path)
    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    total_frames = metadata["total_frames"]

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        pixelformat="yuv420p",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )

    unique_ids = set()
    class_unique_ids = defaultdict(set)
    detection_counts = defaultdict(int)

    start_time = time.time()
    processed_frames = 0

    font_scale = max(0.6, min(width, height) / 700)
    box_thickness = max(2, int(min(width, height) / 350))
    text_thickness = max(2, int(font_scale * 2))

    try:
        results = model.track(
            source=str(video_path),
            tracker=tracker,
            conf=conf,
            stream=True,
            persist=True,
            save=False,
            verbose=False
        )

        for result in results:
            processed_frames += 1
            frame = result.orig_img.copy()
            boxes = result.boxes

            if boxes is not None:
                ids = boxes.id.cpu().numpy() if boxes.id is not None else None

                for index, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].item())
                    class_id = int(box.cls[0].item())
                    class_name = result.names[class_id]

                    track_id = int(ids[index]) if ids is not None else -1

                    if track_id != -1:
                        unique_ids.add(track_id)
                        class_unique_ids[class_name].add(track_id)

                    detection_counts[class_name] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

                    label = f"ID:{track_id} {class_name} {confidence * 100:.0f}%"

                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
                    )

                    label_y1 = max(y1 - text_height - baseline - 12, 0)

                    cv2.rectangle(
                        frame,
                        (x1, label_y1),
                        (x1 + text_width + 10, label_y1 + text_height + baseline + 10),
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, label_y1 + text_height + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        text_thickness,
                        cv2.LINE_AA
                    )

            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if progress_callback is not None and total_frames:
                progress_callback(min(processed_frames, total_frames), total_frames)
    finally:
        writer.close()

    processing_time = time.time() - start_time

    class_wise_counts = {
        name: len(ids_set) for name, ids_set in class_unique_ids.items()
    }

    return {
        "video_name": video_path.name,
        "output_path": str(output_path),
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "duration_seconds": metadata["duration_seconds"],
        "processing_time_seconds": processing_time,
        "number_of_unique_objects": len(unique_ids),
        "unique_ids": sorted(unique_ids),
        "class_wise_counts": class_wise_counts,
        "detection_counts": dict(detection_counts)
    }