#!/usr/bin/env bash
# Single-shot YOLO training script.
# Usage: bash model/train.sh
# Requires: ultralytics installed (pip install ultralytics)

set -euo pipefail

python - <<'PY'
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # base model; adjust if you have a custom init
model.train(
    data='../data/dataset_yolov11/data.yaml',  # updated to Roboflow dataset path
    project='model',
    name='runs',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    pretrained=True,
    workers=4,
)
PY
