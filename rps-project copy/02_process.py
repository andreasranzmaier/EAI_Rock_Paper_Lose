import cv2
from pathlib import Path

RAW_DIR = Path("images/raw")
PROCESSED_DIR = Path("images/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

files = list(RAW_DIR.glob("*.jpeg")) + list(RAW_DIR.glob("*.jpg"))
print(f"Processing {len(files)} images...")

for src in files:
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (96, 72), interpolation=cv2.INTER_AREA)
    small = clahe.apply(small)
    _, out = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(PROCESSED_DIR / src.name), out)

print("Done.")
