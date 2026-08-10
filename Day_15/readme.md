# Automatic Number Plate Recognition (ANPR) using YOLOv8

This project demonstrates the use of a **pre-trained YOLOv8 License Plate Detection model** to detect vehicle license plates in images. The model performs object detection by identifying the location of license plates and drawing bounding boxes around them along with confidence scores.

---

## Project Structure

```text
Day_15/
│
├── Data/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── Graphs/
│   ├── class_distribution.png
│   ├── confidence_distribution.png
│   └── class_percentage.png
│
├── mini_project.py
├── requirements.txt
└── README.md
```

---

## What is Object Detection?

Object Detection is a computer vision task that identifies **what objects are present in an image** and **where they are located**. It predicts the object class along with its bounding box coordinates and confidence score.

---

## Image Classification vs Object Detection

| Image Classification | Object Detection |
|----------------------|------------------|
| Predicts a single label for the entire image. | Detects one or more objects and their locations. |
| Does not provide object location. | Provides bounding boxes around detected objects. |
| Example: "This is a car." | Example: Detects the license plate and shows its location. |

---

## What is YOLO?

YOLO (You Only Look Once) is a real-time object detection algorithm that detects objects in a single forward pass of the neural network. It is widely used because of its high speed, accuracy, and ability to perform real-time object detection.

---

## Dataset Used

**Dataset:** Automatic Number Plate Recognition (ANPR)

The dataset contains vehicle images with annotated license plates in YOLO format. It is divided into:

- Training Set
- Validation Set
- Testing Set

---

## Pre-trained Model Used

Instead of the default COCO model, a **pre-trained YOLOv8 License Plate Detection (`best.pt`) model** was used to perform inference on the ANPR dataset.

---

## Objects Detected

The model detects the following object:

- **license_plate**

For every detected license plate, the model provides:

- Bounding Box
- Confidence Score
- Class Label

---

## Results

The project performs the following tasks:

- Loads a pre-trained YOLOv8 License Plate Detection model.
- Runs inference on the ANPR test dataset.
- Detects license plates in vehicle images.
- Displays bounding boxes around detected license plates.
- Prints confidence scores, class labels, and bounding box coordinates.
- Generates analysis graphs.

---

## Graphs Generated

The following graphs were created to analyze the detection results:

- **Class Distribution**
- **Confidence Distribution**
- **Class Percentage**

---

## Observations

- The pre-trained license plate detection model successfully detected license plates in many test images.
- Some images produced **no detections**, mainly due to challenging viewing angles, small license plates, image quality, or low confidence predictions.
- Compared to the default COCO YOLO model, the specialized license plate model produced significantly more accurate license plate detections.
- The project demonstrates how using a domain-specific pre-trained model can greatly improve detection performance for a specific object class.

---

## Technologies Used

- Python
- Ultralytics YOLOv8
- OpenCV
- Matplotlib
- NumPy

---

## Conclusion

This project demonstrates the complete object detection workflow using a pre-trained YOLOv8 License Plate Detection model. The model successfully detected vehicle license plates by predicting their locations, confidence scores, and class labels. It also highlights the importance of using a task-specific pre-trained model to achieve better performance than a general-purpose object detection model.