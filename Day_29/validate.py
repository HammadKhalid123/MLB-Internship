from ultralytics import YOLO

# Load your best model
model = YOLO("football_player_v2/weights/best.pt")

# Validate
results = model.val(data="data/data.yaml")

# Sahi attribute names
print("\n📊 Final Validation Results:")
print(f"✅ mAP@50: {results.box.map50:.3f}")        # 0.788
print(f"✅ mAP@50-95: {results.box.map:.3f}")       # 0.518
print(f"✅ Precision: {results.box.mp:.3f}")        # 0.937
print(f"✅ Recall: {results.box.mr:.3f}")           # 0.750
print(f"✅ mAP@75: {results.box.map75:.3f}")