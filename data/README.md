# Data Directory
Purpose: Hold raw capture videos, extracted frames, labels, and class list for YOLO training.

Structure:
- videos/        Raw screen capture source videos (.mp4, .mov). Do NOT edit in place.
- frames/        Extracted frame images (.jpg). Split into train/ and val/.
- labels/        YOLO txt annotation files (same basename as frames) in train/ and val/.
- classes.txt    EXACT class ordering used by training + inference.

Rules:
1. Keep exactly one label .txt per frame image when annotated.
2. If a frame has no objects, include an empty .txt file.
3. classes.txt lines are zero-index order; changing order invalidates labels.
4. Use only lowercase alphanumeric + underscores for raw file basenames.
5. Do not commit large raw videos if size > 50MB unless using LFS.

Annotation Format (YOLO): class x_center y_center width height (all normalized 0-1).
