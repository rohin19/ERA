# Single-shot YOLO training script for PowerShell
# Usage: powershell -File train.ps1
# Requires: ultralytics installed (pip install ultralytics)


$pythonCode = @'
import torch
torch.serialization.add_safe_globals([__import__('ultralytics.nn.tasks').nn.tasks.DetectionModel])
torch.serialization.add_safe_globals([__import__('torch.nn.modules.container').nn.modules.container.Sequential])
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
