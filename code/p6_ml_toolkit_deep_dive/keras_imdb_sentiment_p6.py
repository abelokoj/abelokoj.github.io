"""
Keras Bidirectional LSTM for IMDB movie review sentiment classification,
trained with model.fit() and the EarlyStopping / ReduceLROnPlateau
callbacks.

Data note: keras.datasets.imdb.load_data()'s usual host
(storage.googleapis.com) was unreachable in the environment this was
verified in, so the same 50,000-review IMDB dataset is retrieved here as
raw text from a standard GitHub-hosted CSV mirror, and tokenized into a
10,000-word vocabulary here, replicating Keras's own preprocessing scheme
(reserved indices for pad/start/unknown, vocabulary capped at the 10,000
most frequent words) rather than relying on its pre-built encoding. If the
usual host is reachable in your environment, keras.datasets.imdb.load_data()
works as a drop-in replacement for the loading block below.

Dependencies: tensorflow, pandas, numpy, matplotlib, requests
Run:  python keras_imdb_sentiment_p6.py
"""
import os
import re
import requests
import numpy as np
import pandas as pd
from tensorflow import keras
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


VOCAB_SIZE = 10000     # keep only the 10,000 most frequent words
MAX_LEN = 200           # truncate/pad every review to 200 tokens

# ---------------------------------------------------------------------
# Data: retrieve raw IMDB review text and tokenize it ourselves
# ---------------------------------------------------------------------
IMDB_CSV_URL = ("https://raw.githubusercontent.com/Ankit152/"
                "IMDB-sentiment-analysis/master/IMDB-Dataset.csv")
IMDB_CSV_PATH = "imdb_raw.csv"

if not os.path.exists(IMDB_CSV_PATH):
    resp = requests.get(IMDB_CSV_URL, timeout=60)
    resp.raise_for_status()
    with open(IMDB_CSV_PATH, "w", encoding="utf-8") as f:
        f.write(resp.text)

df = pd.read_csv(IMDB_CSV_PATH)


def clean(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9' ]", " ", text)
    return text.lower()


texts = df["review"].apply(clean).tolist()
labels = (df["sentiment"] == "positive").astype(int).values

# Build the vocabulary the same way Keras's own imdb loader does: the top
# VOCAB_SIZE most frequent words, reserving indices 0-3 for
# pad/start/unknown/unused.
tokenizer = keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE - 3)
tokenizer.fit_on_texts(texts)


def encode(text):
    seq = [1]  # <start>
    for w in text.split():
        idx = tokenizer.word_index.get(w)
        seq.append(idx + 3 if (idx is not None and idx < VOCAB_SIZE - 3) else 2)  # 2 = <unk>
    return seq


sequences = [encode(t) for t in texts]

# Standard IMDB-style split: shuffle, then take 25,000/25,000.
rng = np.random.default_rng(42)
perm = rng.permutation(len(sequences))
train_idx, test_idx = perm[:25000], perm[25000:]

X_train_seq = [sequences[i] for i in train_idx]
X_test_seq = [sequences[i] for i in test_idx]
y_train = labels[train_idx]
y_test = labels[test_idx]

X_train = keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test = keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=MAX_LEN)

print(f"Train: {len(X_train)} reviews, Test: {len(X_test)} reviews")
print(f"Train label balance: {y_train.mean():.3f} positive")
print(f"Example review (as word indices): {X_train[0][:10]}...")

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(MAX_LEN,)),
    keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=32),
    keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)

# Both axes below are linear/numeric, so the CM tick formatter applies to both.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
axes[0].set_title("Binary cross-entropy loss")
axes[0].legend()
cm_x(axes[0]); cm_y(axes[0])

axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="validation")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
axes[1].set_title("Classification accuracy")
axes[1].legend()
cm_x(axes[1]); cm_y(axes[1])
fig.tight_layout()
savefig_all(fig, "keras_training_curves_p6")
plt.close(fig)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}, test loss: {test_loss:.4f}")

# ---------------------------------------------------------------------
# Evaluate on genuinely new text
# ---------------------------------------------------------------------
word_index = tokenizer.word_index


def predict_sentiment(text, tokenizer_word_index=word_index):
    tokens = text.lower().split()
    encoded = [1] + [
        (tokenizer_word_index.get(w) + 3)
        if (tokenizer_word_index.get(w) is not None and tokenizer_word_index.get(w) < VOCAB_SIZE - 3)
        else 2
        for w in tokens
    ]
    padded = keras.preprocessing.sequence.pad_sequences([encoded], maxlen=MAX_LEN)
    prob = model.predict(padded, verbose=0)[0, 0]
    return ("positive" if prob > 0.5 else "negative"), prob


sentiment, prob = predict_sentiment("this movie was a genuine waste of time")
print(f"Predicted: {sentiment} (p={prob:.3f})")

sentiment2, prob2 = predict_sentiment("an absolutely brilliant and moving film")
print(f"Predicted: {sentiment2} (p={prob2:.3f})")

with open("keras_results.txt", "w") as f:
    f.write(f"test_acc={test_acc:.4f} test_loss={test_loss:.4f}\n")
    f.write(f"sample1: 'waste of time' -> {sentiment} ({prob:.3f})\n")
    f.write(f"sample2: 'brilliant film' -> {sentiment2} ({prob2:.3f})\n")

print("done")
