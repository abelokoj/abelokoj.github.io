"""
PyTorch CNN for Fashion-MNIST clothing image classification, trained with
an explicit training loop (PyTorch's default style, as opposed to a
high-level fit() call).

Data note: torchvision.datasets.FashionMNIST's usual download host was
unreachable in the environment this was verified in, so the raw IDX files
are downloaded here directly from Zalando Research's own GitHub
repository instead. If the usual host is reachable in your environment,
torchvision.datasets.FashionMNIST(download=True) works as a drop-in
replacement for the loading block below.

Dependencies: torch, numpy, matplotlib, requests
Run:  python pytorch_fashion_mnist_p6.py
"""
import os
import gzip
import struct
import shutil
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
# Data: download and parse the original IDX-format Fashion-MNIST files
# ---------------------------------------------------------------------
FMNIST_BASE_URL = ("https://github.com/zalandoresearch/fashion-mnist/raw/"
                   "master/data/fashion/")
FMNIST_FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}
DATA_DIR = "fmnist_data"
os.makedirs(DATA_DIR, exist_ok=True)


def download_fmnist_files():
    for name, fname in FMNIST_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            continue
        gz_path = path + ".gz"
        resp = requests.get(FMNIST_BASE_URL + fname + ".gz", timeout=60)
        resp.raise_for_status()
        with open(gz_path, "wb") as f:
            f.write(resp.content)
        with gzip.open(gz_path, "rb") as f_in, open(path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)


def load_idx_images(path):
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols).copy()


def load_idx_labels(path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).copy()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

download_fmnist_files()
X_train_np = load_idx_images(os.path.join(DATA_DIR, FMNIST_FILES["train_images"]))
y_train_np = load_idx_labels(os.path.join(DATA_DIR, FMNIST_FILES["train_labels"]))
X_test_np = load_idx_images(os.path.join(DATA_DIR, FMNIST_FILES["test_images"]))
y_test_np = load_idx_labels(os.path.join(DATA_DIR, FMNIST_FILES["test_labels"]))

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Match transforms.ToTensor() [0,1] + Normalize((0.5,), (0.5,)) -> roughly [-1, 1]
X_train = (X_train_np.astype("float32") / 255.0 - 0.5) / 0.5
X_test = (X_test_np.astype("float32") / 255.0 - 0.5) / 0.5

train_set = TensorDataset(torch.from_numpy(X_train).unsqueeze(1),
                           torch.from_numpy(y_train_np).long())
test_set = TensorDataset(torch.from_numpy(X_test).unsqueeze(1),
                          torch.from_numpy(y_test_np).long())

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

print(f"Train: {len(train_set)}, Test: {len(test_set)}")


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),                   # logits over 10 classes
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier(x)


model = FashionCNN().to(device)
print(model)

criterion = nn.CrossEntropyLoss()   # combines log-softmax + NLL loss internally
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def train_one_epoch():
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


EPOCHS = 10
history = {"train_loss": [], "train_acc": [], "test_acc": []}
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    test_acc = evaluate(test_loader)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)
    print(f"Epoch {epoch+1}/{EPOCHS}  loss={train_loss:.4f}  "
          f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
          f"lr={scheduler.get_last_lr()[0]:.5f}")

# Both axes below are linear/numeric, so the CM tick formatter applies to both.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(history["train_loss"])
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss")
axes[0].set_title("Training loss")
cm_x(axes[0]); cm_y(axes[0])

axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["test_acc"], label="test")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
axes[1].set_title("Accuracy per epoch")
axes[1].legend()
cm_x(axes[1]); cm_y(axes[1])
fig.tight_layout()
savefig_all(fig, "pytorch_training_curves_p6")
plt.close(fig)

# Visualize a batch of predictions (images, no numeric axes, so no CM formatter here)
model.eval()
images, labels = next(iter(test_loader))
with torch.no_grad():
    preds = model(images.to(device)).argmax(dim=1).cpu()

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for ax, img, true, pred in zip(axes.ravel(), images[:16], labels[:16], preds[:16]):
    ax.imshow(img.squeeze(), cmap="gray")
    color = "green" if true == pred else "red"
    ax.set_title(f"{class_names[pred]}", fontsize=8, color=color)
    ax.axis("off")
fig.suptitle("Predictions (green=correct, red=incorrect)")
fig.tight_layout()
savefig_all(fig, "pytorch_predictions_p6")
plt.close(fig)

with open("torch_results.txt", "w") as f:
    f.write(f"final_train_acc={history['train_acc'][-1]:.4f}\n")
    f.write(f"final_test_acc={history['test_acc'][-1]:.4f}\n")
    for i, (l, ta, va) in enumerate(zip(history["train_loss"], history["train_acc"], history["test_acc"])):
        f.write(f"epoch={i+1} loss={l:.4f} train_acc={ta:.4f} test_acc={va:.4f}\n")

print("done")
