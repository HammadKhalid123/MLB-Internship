from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import cv2


def load_model(model_path="best.pt"):
    return YOLO(model_path)


def detect_image(model, image_path, conf=0.4):
    results = model.predict(source=image_path, conf=conf)
    return results[0]


def detect_video(model, video_path, output_path, conf=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        frame_count += 1
        yield frame_count, total_frames

    cap.release()
    writer.release()


def analyze_result(result, model):
    class_count = {}
    confidence_scores = []
    detections = []

    for box in result.boxes:
        class_name = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()

        confidence_scores.append(confidence)
        class_count[class_name] = class_count.get(class_name, 0) + 1

        detections.append({
            "class": class_name,
            "confidence": round(confidence, 2),
            "bbox": [round(v, 1) for v in bbox]
        })

    return {
        "total_objects": len(result.boxes),
        "class_count": class_count,
        "confidence_scores": confidence_scores,
        "detections": detections
    }


def plot_class_distribution(class_count, save_path="Graphs/class_distribution.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(class_count.keys(), class_count.values())
    plt.title("Detected Objects")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confidence_distribution(confidence_scores, save_path="Graphs/confidence_distribution.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(confidence_scores, bins=10)
    plt.title("Confidence Score Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_class_percentage(class_count, save_path="Graphs/class_percentage.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.pie(class_count.values(), labels=class_count.keys(), autopct="%1.1f%%")
    plt.title("Detected Classes")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def detect_objects_folder(model, source="Data/test/images", conf=0.5):
    results = model.predict(source=source, conf=conf)
    return results


def analyze_predictions(results, model):
    total_images = len(results)
    total_objects = 0
    class_count = {}
    confidence_scores = []

    for i, result in enumerate(results, start=1):
        print(f"\n{'='*50}")
        print(f"Image {i}: {result.path}")

        if len(result.boxes) == 0:
            print("No Objects Detected")
            continue

        print(f"Total Objects: {len(result.boxes)}")
        total_objects += len(result.boxes)

        for box in result.boxes:
            class_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            confidence_scores.append(confidence)
            class_count[class_name] = class_count.get(class_name, 0) + 1

            print(f"Class Name : {class_name}")
            print(f"Confidence : {confidence:.2f}")
            print(f"Bounding Box : {bbox}\n")

    print(f"\nTotal Images : {total_images}")
    print(f"Total Objects: {total_objects}")

    plot_class_distribution(class_count)
    plot_confidence_distribution(confidence_scores)
    plot_class_percentage(class_count)

    return {
        "total_images": total_images,
        "total_objects": total_objects,
        "class_count": class_count,
        "confidence_scores": confidence_scores
    }


def main():
    model = load_model("best.pt")
    results = detect_objects_folder(model)
    analyze_predictions(results, model)


if __name__ == "__main__":
    main()