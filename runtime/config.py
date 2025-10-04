"""Runtime configuration constants.
Edit values here (or later load from a YAML) to adjust capture & model settings.
Keep this minimal for now.
"""
from pathlib import Path

# Screen capture region of interest (x1, y1, width, height) in screen pixels.
ROI = (100, 120, 800, 1280)  # TODO: Calibrate

# Model & inference
ONNX_MODEL_PATH = Path('model/weights/best.onnx')
INPUT_SIZE = 640  # model square input
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 50

# Classes (must match data/classes.txt order)
CLASSES = [
    'card1','card2','card3','card4','card5','card6','card7','card8'
]

# Overlay
OVERLAY_FPS = 30

# Performance
WARMUP_FRAMES = 3

