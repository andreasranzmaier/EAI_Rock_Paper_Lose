"""V2 training: feeds grayscale (post-CLAHE) images from images/processed_v2/.

Writes model_v2.tflite + labels_v2.txt next to the v1 artifacts so both can
coexist while you compare. The C++ classifier preprocessing must drop the Otsu
threshold step to match.
"""
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

PROCESSED_DIR = Path("images/processed_v2")

files = sorted(list(PROCESSED_DIR.glob("*.jpeg")) + list(PROCESSED_DIR.glob("*.jpg")))
classes = sorted(set(p.stem.split("-")[-1] for p in files))
label_map = {c: i for i, c in enumerate(classes)}
print(f"Classes: {classes}  |  Images: {len(files)}")

# Raw 0-255 grayscale values; the model's Rescaling(1/255) layer normalises
# them internally so the C++ classifier can feed raw bytes at inference time.
X = np.array([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in files], dtype=np.float32)
X = X[..., np.newaxis]  # (N, 72, 96, 1)
y = np.array([label_map[p.stem.split("-")[-1]] for p in files])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1.0 / 255, input_shape=(72, 96, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(classes), activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=16,
          validation_data=(X_val, y_val), callbacks=[early_stop])

Path("model").mkdir(exist_ok=True)
model.save("model/model_v2.keras")
Path("model/labels_v2.txt").write_text("\n".join(classes))

# Round-trip via SavedModel to avoid the TF 2.16 Keras 3 MLIR converter bug.
saved_dir = Path("model/saved_model_v2")
model.export(str(saved_dir))
Path("model/model_v2.tflite").write_bytes(
    tf.lite.TFLiteConverter.from_saved_model(str(saved_dir)).convert()
)
print("Saved model/model_v2.tflite, model/labels_v2.txt, model/model_v2.keras")
