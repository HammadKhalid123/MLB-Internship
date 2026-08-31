# Day 29 — Custom YOLO Object Detection

## Dataset

For this project, I selected the **Football Players & Ball Detection** dataset from Roboflow Universe. I selected this dataset because it represents a practical real-world object detection problem where the model needs to detect multiple objects in the same image.

The dataset contains two main classes:

* **Player**
* **Ball**

It also provides experience with bounding-box annotations, multiple objects, different image conditions, and detecting small objects such as a football.

## Training Configuration

A pre-trained **YOLOv8n** model was used and fine-tuned on the custom dataset.

| Parameter  | Value            |
| ---------- | ---------------- |
| Model      | YOLOv8n          |
| Epochs     | 50               |
| Batch Size | 16               |
| Image Size | 640 × 640        |
| Framework  | Ultralytics YOLO |

The model was trained for 50 epochs with an image size of 640 × 640 and a batch size of 16. The pre-trained YOLO weights helped the model learn the custom classes more effectively.

## Final Evaluation Metrics

The trained model was evaluated using standard object detection metrics:

* **Precision:** [ADD SCORE]
* **Recall:** [ADD SCORE]
* **mAP@50:** [ADD SCORE]
* **mAP@50-95:** [ADD SCORE]

These metrics were used to measure how accurately the model detected players and the football.

## Challenges & Improvements

One of the main challenges was detecting the **football**, because it is usually much smaller than the players and can be difficult to identify in different images.

Another challenge was maintaining good detection performance across images with different backgrounds, object sizes, and positions.

To improve the model, I focused on:

* Training the model for more epochs.
* Using an appropriate image size of 640 × 640.
* Checking the quality and correctness of annotations.
* Improving the dataset quality and diversity.
* Monitoring precision, recall, and mAP during training.

The final trained model was then used for inference on test images to verify its real-world detection performance.
