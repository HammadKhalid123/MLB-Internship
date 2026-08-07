from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8 model

results = model.predict(
    source="images/",
    show=True,
    save=True,
    project=".",
    name="detected_images",
    conf=0.5
)# Predict on webcam feed and display results

for i, result in enumerate(results):
    print(f"\n{'='*50}")
    print(f"Image {i+1}: {result.path}")  # Image ka naam
    
    if result.boxes is not None:
        print(f"Confidence Scores: {result.boxes.conf}")
        print(f"Class IDs: {result.boxes.cls}")
        print(f"Class Names: {[model.names[int(cls)] for cls in result.boxes.cls]}")
        print(f"Bounding Boxes: {result.boxes.xyxy}")
        print(f"Total Objects Detected: {len(result.boxes)}")
    else:
        print("No objects detected in this image!")