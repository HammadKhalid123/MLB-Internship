import cv2
from ultralytics import YOLO

VIDEO_PATH = "../input_videos/traffic1.mp4"
MODEL_PATH = "../yolov8n.pt"

WIDTH = 1000
HEIGHT = 700

CLASS_NAMES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

points = []


def draw_line(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))


def get_line_side(p1, p2, point):
    x1, y1 = p1
    x2, y2 = p2
    x, y = point
    value = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    if value > 0:
        return 1
    elif value < 0:
        return -1
    return 0


def draw_boxes(frame, boxes):
    bboxes = boxes.xyxy.cpu().tolist()
    ids = boxes.id.int().cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()

    for bbox, obj_id, cls in zip(bboxes, ids, classes):
        x1, y1, x2, y2 = map(int, bbox)
        class_name = CLASS_NAMES.get(cls, "unknown")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{class_name} ID:{obj_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return frame


def draw_counts(frame, vehicle_counts):
    y = 30
    for class_name, count in vehicle_counts.items():
        cv2.putText(
            frame,
            f"{class_name.capitalize()}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        y += 35
    return frame


model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

cv2.namedWindow("Video")
cv2.resizeWindow("Video", WIDTH, HEIGHT)
cv2.setMouseCallback("Video", draw_line)

ret, first_frame = cap.read()

if not ret:
    print("Could not read video")
    cap.release()
    cv2.destroyAllWindows()
    exit()

first_frame = cv2.resize(first_frame, (WIDTH, HEIGHT))

last_drawn_count = -1

while len(points) < 2:
    if len(points) != last_drawn_count:
        display_frame = first_frame.copy()

        for point in points:
            cv2.circle(display_frame, point, 5, (0, 0, 255), -1)

        cv2.imshow("Video", display_frame)
        last_drawn_count = len(points)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

track_sides = {}
counted_ids = set()

vehicle_counts = {name: 0 for name in CLASS_NAMES.values()}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        verbose=False,
        classes=list(CLASS_NAMES.keys()),
    )

    boxes = results[0].boxes

    if boxes.id is not None:
        bboxes = boxes.xyxy.cpu().tolist()
        ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()

        for bbox, obj_id, cls in zip(bboxes, ids, classes):
            x1, y1, x2, y2 = bbox
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            current_side = get_line_side(points[0], points[1], center)

            if obj_id in track_sides:
                previous_side = track_sides[obj_id]

                if previous_side != 0 and current_side != 0 and previous_side != current_side:
                    if obj_id not in counted_ids:
                        counted_ids.add(obj_id)
                        class_name = CLASS_NAMES.get(cls)
                        if class_name:
                            vehicle_counts[class_name] += 1

            track_sides[obj_id] = current_side

        frame = draw_boxes(frame, boxes)

    cv2.line(
        frame,
        points[0],
        points[1],
        (0, 0, 255),
        3
    )

    frame = draw_counts(frame, vehicle_counts)

    cv2.imshow("Video", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()