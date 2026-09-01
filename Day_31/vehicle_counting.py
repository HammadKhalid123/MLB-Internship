import os
import cv2
import imageio.v2 as imageio
from ultralytics import YOLO


CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(model_path="yolov8n.pt"):
    return YOLO(model_path)


def setup_video_io(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    writer = imageio.get_writer(
        output_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1
    )
    return cap, writer


def track_frame(model, frame, tracker="bytetrack.yaml", conf=0.4, iou=0.5):
    results = model.track(
        frame,
        persist=True,
        tracker=tracker,
        classes=[2, 3, 5, 7],
        conf=conf,
        iou=iou,
        verbose=False
    )
    return results[0]


def update_counts(result, counted_ids, vehicle_counts):
    if result.boxes.id is None:
        return

    ids = result.boxes.id.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for obj_id, cls_id in zip(ids, classes):
        obj_id = int(obj_id)
        cls_id = int(cls_id)

        if obj_id not in counted_ids:
            counted_ids.add(obj_id)
            class_name = CLASS_NAMES.get(cls_id)
            if class_name:
                vehicle_counts[class_name] += 1


def draw_boxes(frame, result):
    if result.boxes.id is None:
        return frame

    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for box, obj_id, cls_id, conf_score in zip(boxes, ids, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES.get(int(cls_id), "unknown")
        label = f"{class_name} ID:{int(obj_id)} {conf_score:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        label_y1 = max(y1 - text_height - baseline - 10, 0)
        cv2.rectangle(frame, (x1, label_y1), (x1 + text_width + 10, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                    font, font_scale, (0, 0, 0), thickness)

    return frame


def draw_counts(frame, vehicle_counts):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness = 3
    line_spacing = 50

    lines = [
        f"Cars: {vehicle_counts['car']}",
        f"Motorcycles: {vehicle_counts['motorcycle']}",
        f"Buses: {vehicle_counts['bus']}",
        f"Trucks: {vehicle_counts['truck']}"
    ]

    max_width = 0
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, text_width)

    box_height = line_spacing * len(lines) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (max_width + 40, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    y = 40
    for line in lines:
        cv2.putText(frame, line, (20, y), font, font_scale, (0, 255, 0), thickness)
        y += line_spacing

    return frame


def process_video(model, input_path, output_path, tracker="bytetrack.yaml",
                   conf=0.4, iou=0.5, frame_callback=None):
    cap, writer = setup_video_io(input_path, output_path)

    counted_ids = set()
    vehicle_counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        result = track_frame(model, frame, tracker=tracker, conf=conf, iou=iou)
        update_counts(result, counted_ids, vehicle_counts)
        frame = draw_boxes(frame, result)
        frame = draw_counts(frame, vehicle_counts)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

        if frame_callback is not None:
            frame_callback(frame, vehicle_counts, frame_num, total_frames)

    cap.release()
    writer.close()

    return vehicle_counts


def main():
    input_path = os.path.join(SCRIPT_DIR, "input_videos", "traffic1.mp4")
    output_folder = os.path.join(SCRIPT_DIR, "processed_videos")
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(input_path))[0] + ".mp4")

    model = load_model(os.path.join(SCRIPT_DIR, "yolov8n.pt"))
    process_video(model, input_path, output_path)


if __name__ == "__main__":
    main()