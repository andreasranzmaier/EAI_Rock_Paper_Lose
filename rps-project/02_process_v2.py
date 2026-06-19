"""V2 preprocessing: grayscale + CLAHE only (no Otsu threshold).

Writes to images/processed_v2/ so the v1 binary dataset is preserved. Pair with
03_train_v2.py and the matching C++ preprocessing change (drop the Otsu step
in tflite_classifier.cpp's 1-channel path).
"""
import cv2
from pathlib import Path

RAW_DIR = Path("images/raw")
PROCESSED_DIR = Path("images/processed_v2")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

files = list(RAW_DIR.glob("*.jpeg")) + list(RAW_DIR.glob("*.jpg"))
print(f"Processing {len(files)} images...")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for src in files:
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (96, 72), interpolation=cv2.INTER_AREA)
    equalized = clahe.apply(small)
    cv2.imwrite(str(PROCESSED_DIR / src.name), equalized)

print("Done.")
