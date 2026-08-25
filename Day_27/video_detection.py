from pathlib import Path

import cv2
import imageio.v2 as imageio
from ultralytics import YOLO


def load_model(model_path="yolov8n.pt"):
    return YOLO(model_path)


def process_video(
    video_path,
    model=None,
    output_dir="output_videos",
    conf=0.5,
    progress_callback=None,
):
    if model is None:
        model = load_model()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions.")

    output_path = output_dir / f"{video_path.stem}_detected.mp4"

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        macro_block_size=None,
    )

    all_detections = []
    frame_number = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            results = model.predict(source=frame, conf=conf, verbose=False)
            result = results[0]

            annotated_frame = result.plot()
            writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

            for index, box in enumerate(result.boxes, start=1):
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                box_width = x2 - x1
                box_height = y2 - y1

                all_detections.append(
                    {
                        "frame_number": frame_number,
                        "detection_id": index,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "confidence_percentage": confidence * 100,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": box_width,
                        "height": box_height,
                    }
                )

            if progress_callback is not None and total_frames > 0:
                progress_callback(frame_number, total_frames)

    finally:
        cap.release()
        writer.close()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Output video was not created.")

    return {
        "detections": all_detections,
        "total_detections": len(all_detections),
        "total_frames": total_frames,
        "processed_frames": frame_number,
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "output_path": str(output_path),
    }


def main():
    video_path = "input_videos/car_video.mp4"
    output = process_video(video_path)

    print(f"\nProcessed Frames: {output['processed_frames']}")
    print(f"Total Detections: {output['total_detections']}")
    print(f"Saved Video: {output['output_path']}")

    for detection in output["detections"]:
        print("\n------------------------------")
        print(f"Frame: {detection['frame_number']}")
        print(f"Detection ID: {detection['detection_id']}")
        print(f"Class ID: {detection['class_id']}")
        print(f"Class Name: {detection['class_name']}")
        print(f"Confidence: {detection['confidence_percentage']:.2f}%")
        print(f"BBox XYXY: {detection['bbox_xyxy']}")
        print(f"Width: {detection['width']:.2f}")
        print(f"Height: {detection['height']:.2f}")


if __name__ == "__main__":
    main()