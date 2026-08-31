import time
from pathlib import Path
from collections import defaultdict

import cv2
import imageio.v2 as imageio
from ultralytics import YOLO


# Base directory = the folder this file (tracking_code.py) lives in.
# Makes relative paths (input_videos, saved_videos) work no matter what the
# current working directory is when Streamlit runs the app.
BASE_DIR = Path(__file__).resolve().parent


def load_model(model_path="yolov8n.pt"):
    return YOLO(model_path)


def list_sample_videos(input_dir="input_videos"):
    input_dir = Path(input_dir)
    if not input_dir.is_absolute():
        input_dir = BASE_DIR / input_dir

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


def _compute_resized_dims(width, height, max_width):
    if not max_width or width <= max_width:
        new_width, new_height = width, height
    else:
        scale = max_width / width
        new_width = max_width
        new_height = int(round(height * scale))

    new_width -= new_width % 2
    new_height -= new_height % 2
    return max(new_width, 2), max(new_height, 2)


def process_video(
    video_path,
    model=None,
    output_dir="saved_videos",
    tracker="bytetrack.yaml",
    conf=0.5,
    progress_callback=None,
    max_width=960,
    imgsz=640,
    frame_skip=1
):
    if model is None:
        model = load_model()

    video_path = Path(video_path)

    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{video_path.stem}_tracked.mp4"

    metadata = get_video_metadata(video_path)
    fps = metadata["fps"]
    orig_width = metadata["width"]
    orig_height = metadata["height"]
    total_frames = metadata["total_frames"]

    out_width, out_height = _compute_resized_dims(orig_width, orig_height, max_width)
    output_fps = fps / frame_skip if frame_skip > 1 else fps

    writer = imageio.get_writer(
        str(output_path),
        fps=output_fps,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        pixelformat="yuv420p",
        ffmpeg_params=[
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "ultrafast",   # much faster encoding than the default preset
            "-crf", "26"              # reasonable quality/size tradeoff at speed
        ]
    )

    unique_ids = set()
    class_unique_ids = defaultdict(set)
    detection_counts = defaultdict(int)

    start_time = time.time()
    processed_frames = 0
    written_frames = 0

    font_scale = max(0.6, min(out_width, out_height) / 700)
    box_thickness = max(2, int(min(out_width, out_height) / 350))
    text_thickness = max(2, int(font_scale * 2))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        writer.close()
        raise ValueError(f"Could not open video: {video_path}")

    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1

            # Skip frames if requested (speed lever)
            if frame_skip > 1 and (frame_index - 1) % frame_skip != 0:
                if progress_callback is not None and total_frames:
                    progress_callback(min(frame_index, total_frames), total_frames)
                continue

            if (out_width, out_height) != (orig_width, orig_height):
                frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

            processed_frames += 1

            result = model.track(
                frame,
                tracker=tracker,
                conf=conf,
                imgsz=imgsz,
                persist=True,
                verbose=False
            )[0]

            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
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
            written_frames += 1

            if progress_callback is not None and total_frames:
                progress_callback(min(frame_index, total_frames), total_frames)
    finally:
        cap.release()
        writer.close()

    processing_time = time.time() - start_time

    class_wise_counts = {
        name: len(ids_set) for name, ids_set in class_unique_ids.items()
    }

    return {
        "video_name": video_path.name,
        "output_path": str(output_path),
        "fps": output_fps,
        "frame_width": out_width,
        "frame_height": out_height,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "written_frames": written_frames,
        "duration_seconds": metadata["duration_seconds"],
        "processing_time_seconds": processing_time,
        "number_of_unique_objects": len(unique_ids),
        "unique_ids": sorted(unique_ids),
        "class_wise_counts": class_wise_counts,
        "detection_counts": dict(detection_counts)
    }