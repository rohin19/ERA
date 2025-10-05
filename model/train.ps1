# Single-shot YOLO training script for PowerShell
# Usage: powershell -File train.ps1
# Requires: ultralytics installed (pip install ultralytics)

$pythonCode = @'
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # base model; adjust if you have a custom init
model.train(
    data='../data/dataset.yolov11/data.yaml',  # updated to Roboflow dataset path
    project='model',
    name='runs',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    pretrained=True,
    workers=4,
)
'@

python -c $pythonCode
