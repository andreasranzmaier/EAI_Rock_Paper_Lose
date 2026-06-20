import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

PROCESSED_DIR = Path("images/processed")

files = sorted(list(PROCESSED_DIR.glob("*.jpeg")) + list(PROCESSED_DIR.glob("*.jpg")))
classes = sorted(set(p.stem.split("-")[-1] for p in files))
label_map = {c: i for i, c in enumerate(classes)}
print(f"Classes: {classes}  |  Images: {len(files)}")

# Raw 0-255 values — the Rescaling layer in the model normalises them,
# matching what tflite_classifier.cpp feeds at inference time.
X = np.array([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in files], dtype=np.float32)
X = X[..., np.newaxis]  # (N, 72, 96, 1)
y = np.array([label_map[p.stem.split("-")[-1]] for p in files])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Rescaling(1/255) is baked into the model so the C++ classifier can feed raw
# 0-255 pixel values and the model normalises them internally.
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
model.save("model/model.keras")
Path("model/labels.txt").write_text("\n".join(classes))

# TF 2.16's MLIR-based TFLite converter crashes on Keras 3 Sequential models
# that include a Rescaling layer (LLVM "missing attribute 'value'" abort).
# Round-trip through SavedModel to avoid the buggy direct path, then clean
# up the intermediate so the model/ folder stays small.
import tempfile
with tempfile.TemporaryDirectory() as saved_dir:
    model.export(saved_dir)
    Path("model/model.tflite").write_bytes(
        tf.lite.TFLiteConverter.from_saved_model(saved_dir).convert()
    )
print("Saved model/model.tflite, model/labels.txt, model/model.keras")
