from ultralytics import YOLO
from pathlib import Path
import shutil

# Load pretrained YOLO model
model = YOLO("football_player_v2/weights/best.pt")


# Train the model
model.train(
    data="data/data.yaml",
    epochs=40,
    imgsz=640,
    batch=8,
    project="runs",
    name="football_player_v2",
    exist_ok=True,
    resume=True
)

best_model = Path("runs/football_player/weights/best.pt")
models_dir = Path("models")

models_dir.mkdir(exist_ok=True)

if best_model.exists():
    shutil.copy(
        best_model,
        models_dir / "best.pt"
    )
    print("\nTraining completed successfully!")
    print("Best model saved at: models/best.pt")