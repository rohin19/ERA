"""Deterministic 80/20 split of frames & labels into train/ and val/.
Assumes all current frames are in data/frames/train initially.
"""
import pathlib, random, shutil

FRAMES = pathlib.Path('data/frames/train')
VAL_DIR = pathlib.Path('data/frames/val')
TRAIN_DIR = pathlib.Path('data/frames/train')
LABELS_TRAIN = pathlib.Path('data/labels/train')
LABELS_VAL = pathlib.Path('data/labels/val')
RATIO = 0.2
SEED = 42

def split():
    random.seed(SEED)
    imgs = sorted([p for p in FRAMES.glob('*.jpg')])
    random.shuffle(imgs)
    n_val = int(len(imgs) * RATIO)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_VAL.mkdir(parents=True, exist_ok=True)
    LABELS_TRAIN.mkdir(parents=True, exist_ok=True)
    moved = 0
    for i, img in enumerate(imgs):
        label = pathlib.Path('data/labels/train') / (img.stem + '.txt')
        if i < n_val:
            shutil.move(str(img), VAL_DIR / img.name)
            if label.exists():
                shutil.move(str(label), LABELS_VAL / label.name)
            moved += 1
    print(f"Moved {moved} images to val (ratio {RATIO}).")

if __name__ == '__main__':
    split()
