"""
TensorFlow CNN on MNIST, trained with an explicit tf.GradientTape loop
rather than model.fit(), to make the mechanics of backpropagation and
gradient descent visible rather than hidden behind a single call.

Data note: tf.keras.datasets.mnist.load_data()'s usual host
(storage.googleapis.com) was unreachable in the environment this was
verified in, so the original LeCun IDX files are downloaded here from a
standard GitHub mirror instead. If the usual host is reachable in your
environment, tf.keras.datasets.mnist.load_data() works as a drop-in
replacement for the loading block below.

Dependencies: tensorflow, numpy, matplotlib, requests
Run:  python tensorflow_mnist_p6.py
"""
import os
import gzip
import struct
import shutil
import numpy as np
import requests
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------------------------------------------------------------------
# Publication-style plot settings.
# ---------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Serif"],
    "axes.formatter.use_mathtext": True, "font.size": 11,
    "axes.labelsize": 12, "legend.fontsize": 9,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.6, "grid.linestyle": "--",
})


class CMTickFormatter(ScalarFormatter):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.set_useMathText(False)

    def __call__(self, x, pos=None):
        return f"${super().__call__(x, pos)}$"


def cm_x(ax):
    ax.xaxis.set_major_formatter(CMTickFormatter())


def cm_y(ax):
    ax.yaxis.set_major_formatter(CMTickFormatter())


def savefig_all(fig, basename, **kwargs):
    """Save a figure as both a PNG and a vector SVG copy (dpi/bbox come
    from the rcParams block above)."""
    fig.savefig(f"{basename}.png", **kwargs)
    fig.savefig(f"{basename}.svg", **kwargs)


# ---------------------------------------------------------------------
# Data: download and parse the original IDX-format MNIST files
# ---------------------------------------------------------------------
MNIST_BASE_URL = "https://github.com/fgnt/mnist/raw/master/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}
DATA_DIR = "mnist_data"
os.makedirs(DATA_DIR, exist_ok=True)


def download_mnist_files():
    for name, fname in MNIST_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            continue
        gz_path = path + ".gz"
        resp = requests.get(MNIST_BASE_URL + fname + ".gz", timeout=60)
        resp.raise_for_status()
        with open(gz_path, "wb") as f:
            f.write(resp.content)
        with gzip.open(gz_path, "rb") as f_in, open(path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)


def load_idx_images(path):
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def load_idx_labels(path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


download_mnist_files()
X_train = load_idx_images(os.path.join(DATA_DIR, MNIST_FILES["train_images"]))
y_train = load_idx_labels(os.path.join(DATA_DIR, MNIST_FILES["train_labels"]))
X_test = load_idx_images(os.path.join(DATA_DIR, MNIST_FILES["test_images"]))
y_test = load_idx_labels(os.path.join(DATA_DIR, MNIST_FILES["test_labels"]))
print(f"Train: {X_train.shape}, Test: {X_test.shape}")   # (60000, 28, 28), (10000, 28, 28)

# Normalize to [0, 1] and add a channel dimension for the convolutional layers
X_train = X_train.astype("float32")[..., None] / 255.0
X_test = X_test.astype("float32")[..., None] / 255.0

# Build a shuffled, batched tf.data pipeline -- TensorFlow's native, efficient
# way to feed data through training, handling batching/shuffling/prefetching
# without materializing the whole shuffled dataset in memory at once.
BATCH_SIZE = 128
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10),   # logits, no softmax -- see loss function below
    ])


model = build_model()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_loss_metric = tf.keras.metrics.Mean()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


@tf.function   # compiles this function into a TensorFlow graph for speed
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y_batch, logits)
    return loss


@tf.function
def val_step(x_batch, y_batch):
    logits = model(x_batch, training=False)
    val_acc_metric.update_state(y_batch, logits)


history = {"train_loss": [], "train_acc": [], "val_acc": []}
EPOCHS = 10
for epoch in range(EPOCHS):
    train_loss_metric.reset_state()
    train_acc_metric.reset_state()
    for x_batch, y_batch in train_ds:
        train_step(x_batch, y_batch)

    val_acc_metric.reset_state()
    for x_batch, y_batch in test_ds:
        val_step(x_batch, y_batch)

    history["train_loss"].append(float(train_loss_metric.result()))
    history["train_acc"].append(float(train_acc_metric.result()))
    history["val_acc"].append(float(val_acc_metric.result()))
    print(f"Epoch {epoch+1}/{EPOCHS}  "
          f"loss={history['train_loss'][-1]:.4f}  "
          f"train_acc={history['train_acc'][-1]:.4f}  "
          f"val_acc={history['val_acc'][-1]:.4f}")

# Both axes below are linear/numeric, so the CM tick formatter applies to both.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(history["train_loss"])
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss")
axes[0].set_title("Training loss (sparse categorical cross-entropy)")
cm_x(axes[0]); cm_y(axes[0])

axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["val_acc"], label="validation")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
axes[1].set_title("Accuracy per epoch")
axes[1].legend()
cm_x(axes[1]); cm_y(axes[1])
fig.tight_layout()
savefig_all(fig, "tf_training_curves_p6")
plt.close(fig)

with open("tf_results.txt", "w") as f:
    f.write(f"final_train_acc={history['train_acc'][-1]:.4f}\n")
    f.write(f"final_val_acc={history['val_acc'][-1]:.4f}\n")
    for i, (l, ta, va) in enumerate(zip(history["train_loss"], history["train_acc"], history["val_acc"])):
        f.write(f"epoch={i+1} loss={l:.4f} train_acc={ta:.4f} val_acc={va:.4f}\n")

print("done")
