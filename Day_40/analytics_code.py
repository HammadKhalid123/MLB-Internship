from ultralytics import YOLO
import cv2
import numpy as np
import time
import csv
import imageio_ffmpeg
import imageio.v2 as imageio


model = YOLO("yolov8n.pt")


def process_video(
    input_path,
    output_path,
    csv_path,
    roi_points,
    imgsz=640,
    frame_skip_enabled=False,
    progress_callback=None
):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():

        raise ValueError(
            "Could not open the uploaded video."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if video_fps <= 0:

        video_fps = 30

    out = imageio.get_writer(
        output_path,
        fps=video_fps,
        codec="libx264",
        quality=None,
        bitrate=None,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=[
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart"
        ]
    )

    roi_polygon = np.array(roi_points, dtype=np.int32)

    previous_state = {}

    unique_ids = set()

    entered = 0
    exited = 0

    current_object_count = 0

    total_processing_time = 0

    processed_frames = 0

    events = []

    skip_interval = 2 if frame_skip_enabled else 1

    last_boxes = None

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        frame_start = time.time()

        if processed_frames % skip_interval == 0:

            results = model.track(
                frame,
                imgsz=imgsz,
                persist=True,
                tracker="botsort.yaml",
                verbose=False
            )

            boxes = results[0].boxes

            last_boxes = boxes

        else:

            boxes = last_boxes

        if boxes is not None:

            current_object_count = len(boxes)

        else:

            current_object_count = 0

        if boxes is not None and boxes.id is not None:

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

            coordinates = (
                boxes.xyxy
                .int()
                .cpu()
                .tolist()
            )

            unique_ids.update(track_ids)

            for track_id, box in zip(
                track_ids,
                coordinates
            ):

                x1, y1, x2, y2 = box

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                inside = (
                    cv2.pointPolygonTest(
                        roi_polygon,
                        (float(cx), float(cy)),
                        False
                    )
                    >= 0
                )

                if track_id not in previous_state:

                    previous_state[
                        track_id
                    ] = inside

                else:

                    previous_inside = (
                        previous_state[track_id]
                    )

                    if (
                        not previous_inside
                        and inside
                    ):

                        entered += 1

                        events.append({
                            "track_id": track_id,
                            "event": "entry",
                            "frame": processed_frames,
                            "time_seconds": round(
                                processed_frames / video_fps,
                                2
                            )
                        })

                    elif (
                        previous_inside
                        and not inside
                    ):

                        exited += 1

                        events.append({
                            "track_id": track_id,
                            "event": "exit",
                            "frame": processed_frames,
                            "time_seconds": round(
                                processed_frames / video_fps,
                                2
                            )
                        })

                    previous_state[
                        track_id
                    ] = inside

                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (0, 255, 0),
                    -1
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (
                        x1,
                        max(y1 - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        frame_time = (
            time.time() - frame_start
        )

        total_processing_time += frame_time

        processed_frames += 1

        if frame_time > 0:

            fps = 1 / frame_time

        else:

            fps = 0

        cv2.polylines(
            frame,
            [roi_polygon],
            True,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Current Objects: {current_object_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Unique Objects: {len(unique_ids)}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Entered: {entered}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Exited: {exited}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        out.append_data(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        if (
            progress_callback
            and total_frames > 0
        ):

            progress = (
                processed_frames
                / total_frames
            )

            progress_callback(
                min(progress, 1.0),
                f"Processing frame {processed_frames}/{total_frames}"
            )

    cap.release()

    out.close()

    if total_processing_time > 0:

        average_fps = (
            processed_frames
            / total_processing_time
        )

    else:

        average_fps = 0

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "track_id",
                "event",
                "frame",
                "time_seconds"
            ]
        )

        writer.writeheader()

        writer.writerows(events)

    if progress_callback:

        progress_callback(
            1.0,
            "Analysis completed."
        )

    return {
        "current_objects": current_object_count,
        "unique_objects": len(unique_ids),
        "entered": entered,
        "exited": exited,
        "average_fps": average_fps
    }