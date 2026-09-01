from ultralytics import YOLO
import cv2


DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


def load_model(model_path):
    return YOLO(model_path)


def track_frame(model, frame):
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )
    return results[0]


def draw_boxes(frame, result, model):
    boxes = result.boxes

    if boxes.id is None:
        return frame

    bboxes = boxes.xyxy.cpu().tolist()
    ids = boxes.id.int().cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()

    for bbox, obj_id, cls in zip(bboxes, ids, classes):
        x1, y1, x2, y2 = map(int, bbox)
        class_name = model.names[cls]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {obj_id} {class_name}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return frame


def resize_for_display(frame, width, height):
    return cv2.resize(frame, (width, height))


def run_tracking(model, video_path, window_name="Tracked Video"):
    cap = cv2.VideoCapture(video_path)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = track_frame(model, frame)
        frame = draw_boxes(frame, result, model)

        display_frame = resize_for_display(frame, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    model = load_model("../yolov8n.pt")
    run_tracking(model, "../input_videos/traffic3.mp4")


if __name__ == "__main__":
    main()