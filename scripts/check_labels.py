"""Validate dataset integrity.
Checks:
1. Every frame has a label file (may be empty) OR warn.
2. Class ids within bounds.
"""
import pathlib, sys

CLASSES = [l.strip() for l in open('data/classes.txt', 'r', encoding='utf-8') if l.strip()]
FRAME_DIRS = [pathlib.Path('data/frames/train'), pathlib.Path('data/frames/val')]
LABEL_DIRS = [pathlib.Path('data/labels/train'), pathlib.Path('data/labels/val')]

ok = True

for frame_dir, label_dir in zip(FRAME_DIRS, LABEL_DIRS):
    if not frame_dir.exists():
        continue
    images = list(frame_dir.glob('*.jpg'))
    for img in images:
        label = label_dir / (img.stem + '.txt')
        if not label.exists():
            print(f"WARN: Missing label for {img}")
            ok = False
        else:
            with open(label, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.split()
                    try:
                        cls_id = int(parts[0])
                    except ValueError:
                        print(f"BAD LINE (non-int class) {label}: {line.strip()}")
                        ok = False
                        continue
                    if cls_id < 0 or cls_id >= len(CLASSES):
                        print(f"BAD CLASS ID {cls_id} in {label}")
                        ok = False

if not ok:
    print("Dataset check: FAIL")
    sys.exit(1)
else:
    print("Dataset check: OK")
