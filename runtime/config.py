"""Runtime configuration constants.
Edit values here (or later load from a YAML) to adjust capture & model settings.
Keep this minimal for now.
"""
from pathlib import Path

# Screen capture region of interest (x1, y1, width, height) in screen pixels.
ROI = (100, 120, 800, 1280)  # TODO: Calibrate

# Model & inference
ONNX_MODEL_PATH = Path('model/weights/best.pt')
INPUT_SIZE = 640  # model square input
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 50

# Classes (must match data/classes.txt order)
CLASSES = [
    'Baby Dragon','Bomber','Dart Goblin','Giant','Hog Rider','Knight','Mini Pekka','Valkyrie'
]

# Overlay
OVERLAY_FPS = 30

# Performance
WARMUP_FRAMES = 3

