import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import shutil

np.random.seed(42)
tf.random.set_seed(42)

RAW_DIR = Path("images/raw")
PROCESSED_DIR = Path("images/processed")
IMG_SIZE = 160

if PROCESSED_DIR.exists():
    shutil.rmtree(PROCESSED_DIR)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

raw_files = list(RAW_DIR.glob("*.jpeg")) + list(RAW_DIR.glob("*.jpg"))
# remove any files that have garbage in filename
raw_files = [f for f in raw_files if f.stem.split("-")[-1] in ["rock", "paper", "scissors"]]

classes = sorted(set(p.stem.split("-")[-1] for p in raw_files))
label_map = {c: i for i, c in enumerate(classes)}
print(f"Classes: {classes}  |  Raw images: {len(raw_files)}")
print(f"Image count per class: { {c: sum(1 for p in raw_files if p.stem.split('-')[-1] == c) for c in classes} }")
X_all, y_all = [], []

for src in raw_files:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    X_all.append(small)
    y_all.append(label_map[src.stem.split("-")[-1]])
    dst = PROCESSED_DIR / src.name
    cv2.imwrite(str(dst), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

max_count = np.bincount(y_train).max()
X_bal, y_bal = [], []
for c in range(len(classes)):
    idx = np.where(y_train == c)[0]
    oversampled = np.random.choice(idx, size=max_count, replace=True)
    X_bal.append(X_train[oversampled])
    y_bal.append(y_train[oversampled])
X_train = np.concatenate(X_bal)
y_train = np.concatenate(y_bal)
print(f"Train after oversampling: {len(X_train)}  |  Per class: {max_count}")

aug = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.10, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.10, fill_mode="nearest"),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

base = tf.keras.applications.MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    include_preprocessing=True,
    weights="imagenet",
)
base.trainable = True

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    aug,
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(classes), activation="softmax"),
])

total_steps = len(X_train) // 32 * 200
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=total_steps,
    alpha=1e-4,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

class_counts = np.bincount(y_train)
class_weight = {i: len(y_train) / (len(classes) * count)
                for i, count in enumerate(class_counts)}
print(f"Class weights: { {classes[i]: f'{class_weight[i]:.2f}' for i in range(len(classes))} }")

model.fit(
    X_train, y_train,
    epochs=10, batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    class_weight=class_weight,
)

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n{'='*50}")
print(f"Train Loss: {train_loss:.4f}  |  Train Accuracy: {train_acc:.4f}")
print(f"Val   Loss: {val_loss:.4f}  |  Val   Accuracy: {val_acc:.4f}")
print(f"{'='*50}")

# Export — int8 quantized TFLite
Path("model").mkdir(exist_ok=True)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for i in range(0, len(X_train), 10):
        yield [X_train[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

Path("model/model.tflite").write_bytes(converter.convert())
Path("model/labels.txt").write_text("\n".join(classes))
print("Saved model/model.tflite and model/labels.txt")
