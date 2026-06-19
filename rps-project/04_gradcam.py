import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path("images/processed")
MODEL_PATH = Path("model/model.keras")

model = tf.keras.models.load_model(MODEL_PATH)
classes = sorted(set(p.stem.split("-")[-1] for p in PROCESSED_DIR.glob("*.jpeg")))
print(f"Classes: {classes}")

# Split Sequential layers at the last Conv2D so we can watch its output
last_conv_idx = next(i for i, l in reversed(list(enumerate(model.layers)))
                     if isinstance(l, tf.keras.layers.Conv2D))
last_conv = model.layers[last_conv_idx]
pre_layers = model.layers[:last_conv_idx + 1]
post_layers = model.layers[last_conv_idx + 1:]


def gradcam(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    x = tf.constant(img[np.newaxis, ..., np.newaxis])  # (1, 72, 96, 1)

    with tf.GradientTape() as tape:
        # Run up to last conv, then watch its output
        act = x
        for layer in pre_layers:
            act = layer(act)
        conv_out = act
        tape.watch(conv_out)

        # Continue forward from conv_out
        act = conv_out
        for layer in post_layers:
            act = layer(act)

        pred_idx = tf.argmax(act[0])
        score = act[:, pred_idx]

    grads = tape.gradient(score, conv_out)[0]       # (H, W, C)
    weights = tf.reduce_mean(grads, axis=(0, 1))    # (C,)
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (96, 72))

    return img, cam, classes[pred_idx.numpy()], act[0].numpy()


# Pick one image per class
samples = {}
for p in sorted(PROCESSED_DIR.glob("*.jpeg")):
    cls = p.stem.split("-")[-1]
    if cls not in samples:
        samples[cls] = p

n = len(samples)
fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

for col, (cls, path) in enumerate(sorted(samples.items())):
    img, cam, pred, probs = gradcam(path)

    axes[0, col].imshow(img, cmap="gray")
    axes[0, col].set_title(f"true: {cls}", fontsize=11)
    axes[0, col].axis("off")

    axes[1, col].imshow(img, cmap="gray")
    axes[1, col].imshow(cam, cmap="jet", alpha=0.5)
    axes[1, col].set_title(f"pred: {pred} ({probs.max():.0%})", fontsize=11)
    axes[1, col].axis("off")

plt.suptitle(f"Grad-CAM — {last_conv.name} ({last_conv.filters} filters)", fontsize=13)
plt.tight_layout()
plt.savefig("gradcam_output.png", dpi=150)
print("Saved gradcam_output.png")
