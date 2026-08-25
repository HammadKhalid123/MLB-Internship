from pathlib import Path

from ultralytics import YOLO


def load_model(model_path="yolov8n.pt"):
    return YOLO(model_path)


def process_image(image_path, model=None, output_dir="output_images", conf=0.5):
    if model is None:
        model = load_model()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)

    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    result = results[0]

    annotated_image = result.plot()

    output_path = output_dir / f"{image_path.stem}_detected.jpg"
    result.save(filename=str(output_path))

    detections = []

    for index, box in enumerate(result.boxes, start=1):
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]
        confidence = float(box.conf[0].item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1

        detections.append(
            {
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
                "width": width,
                "height": height,
            }
        )

    return {
        "detections": detections,
        "number_of_detections": len(detections),
        "output_path": str(output_path),
        "annotated_image": annotated_image,
        "frame_width": int(result.orig_shape[1]),
        "frame_height": int(result.orig_shape[0]),
    }


def main():
    image_path = "input_images/img1.jpg"
    output = process_image(image_path)

    print(f"\nTotal detections: {output['number_of_detections']}")
    print(f"Saved image: {output['output_path']}")

    for detection in output["detections"]:
        print("\n------------------------------")
        print(f"Detection ID: {detection['detection_id']}")
        print(f"Class ID: {detection['class_id']}")
        print(f"Class Name: {detection['class_name']}")
        print(f"Confidence: {detection['confidence_percentage']:.2f}%")
        print(f"BBox XYXY: {detection['bbox_xyxy']}")
        print(f"Width: {detection['width']:.2f}")
        print(f"Height: {detection['height']:.2f}")


if __name__ == "__main__":
    main()