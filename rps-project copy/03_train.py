import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

PROCESSED_DIR = Path("images/processed")

# Load images and extract labels from filenames
files = sorted(PROCESSED_DIR.glob("*.jpeg"))
classes = sorted(set(p.stem.split("-")[-1] for p in files))
label_map = {c: i for i, c in enumerate(classes)}
print(f"Classes: {classes}  |  Images: {len(files)}")

X = np.array([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in files], dtype=np.float32) / 255.0
X = X[..., np.newaxis]  # (N, 72, 96, 1)
y = np.array([label_map[p.stem.split("-")[-1]] for p in files])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation="relu", input_shape=(72, 96, 1)),
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

# Export to TFLite + label map
Path("model").mkdir(exist_ok=True)
Path("model/model.tflite").write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())
Path("model/labels.txt").write_text("\n".join(classes))
print("Saved model/model.tflite and model/labels.txt")
