#!/usr/bin/env bash
# Export the best model weights to ONNX.
# Usage: bash model/export_onnx.sh
set -euo pipefail
BEST_PT="model/runs/weights/best.pt"
OUT="model/weights/best.onnx"

if [ ! -f "$BEST_PT" ]; then
  echo "Missing $BEST_PT (train first)" >&2
  exit 1
fi

python - <<'PY'
from ultralytics import YOLO
import os
pt = os.environ.get('BEST_PT','model/runs/weights/best.pt')
onnx_out = os.environ.get('OUT','model/weights/best.onnx')
model = YOLO(pt)
model.export(format='onnx', opset=17, dynamic=False, simplify=True)
# Move produced file if needed
def find_onnx(start):
    for root, _, files in os.walk(start):
        for f in files:
            if f.endswith('.onnx'):
                return os.path.join(root,f)
onnx_path = find_onnx('.')
if onnx_path and onnx_path != onnx_out:
    import shutil, pathlib
    pathlib.Path(os.path.dirname(onnx_out)).mkdir(parents=True, exist_ok=True)
    shutil.move(onnx_path, onnx_out)
    print(f"Exported ONNX -> {onnx_out}")
else:
    print("Could not locate ONNX output automatically.")
PY
