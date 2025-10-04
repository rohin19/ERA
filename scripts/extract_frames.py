"""Extract frames from videos in data/videos -> data/frames/train (default).
Adjust splitting later with split_train_val.py.
"""
import cv2
import pathlib

VIDEOS = pathlib.Path('data/videos')
OUT = pathlib.Path('data/frames/train')
FPS_STEP = 5  # take every 5th frame (adjust)


def extract():
    OUT.mkdir(parents=True, exist_ok=True)
    for vid in VIDEOS.glob('*.*'):
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f'Skip {vid} (cannot open)')
            continue
        idx = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FPS_STEP == 0:
                out_path = OUT / f"{vid.stem}_{idx:05d}.jpg"
                cv2.imwrite(str(out_path), frame)
                idx += 1
            frame_idx += 1
        cap.release()
        print(f"Extracted {idx} frames from {vid.name}")

if __name__ == '__main__':
    extract()
