#!/usr/bin/env bash
# Quick validation / predict sanity check.
# Usage: bash model/validate.sh
set -euo pipefail
BEST_PT="model/runs/weights/best.pt"
if [ ! -f "$BEST_PT" ]; then
  echo "Missing $BEST_PT (train first)" >&2
  exit 1
fi
python - <<'PY'
from ultralytics import YOLO
m = YOLO('model/runs/weights/best.pt')
# Basic val (assumes val set exists)
try:
    m.val(data='model/cr_data.yaml', imgsz=640)
except Exception as e:
    print('Validation issue:', e)
# Run a single prediction on first val image if present
import glob
val_imgs = glob.glob('data/frames/val/*.jpg')
if val_imgs:
    r = m.predict(source=val_imgs[0], save=False, imgsz=640, conf=0.25)
    print('Sample prediction boxes:', r[0].boxes.shape if r else None)
else:
    print('No val images found for sample prediction.')
PY
