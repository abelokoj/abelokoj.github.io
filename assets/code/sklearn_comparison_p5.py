"""
sklearn_comparison_p5.py
========================
Run EVERY section of the ML-toolkit post in one process:

    Part 1  scikit-learn   logistic regression, SVM, random forest,
                           gradient boosting, plus the three figures
    Part 2  XGBoost
    Part 3  TensorFlow / Keras
    Part 4  PyTorch

and finish with a side-by-side summary of all seven models.

Why one script
--------------
Every number the post reports should come from the same machine, the
same data split and the same seed. Mixing figures measured in different
environments invites small inconsistencies that are awkward to explain
later. Everything below shares one train/test split, built once and
handed to each section.

The code in each part is the code printed in the post, with three
additions that make the results reportable rather than merely printed:

  * seeds fixed everywhere, so a rerun reproduces the numbers;
  * the epochs actually run, since both deep-learning parts stop early
    and the count is part of the result;
  * trainable parameter counts, which is what makes the comparison with
    the 31-parameter logistic regression of Part 1 meaningful.

Any library that is missing or broken is reported with its reason and
skipped, so a partial environment still produces everything it can.

Usage
-----
    python sklearn_comparison_p5.py

Figures are written to the working directory as PNG and SVG. Runtime is
well under a minute on any modern machine; the dataset has 569 rows.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # write files, never open a window
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score,
                             confusion_matrix, classification_report)

SEED = 42

# ---------------------------------------------------------------------
# Publication-style plot settings (LaTeX-like serif rendering without
# requiring a system LaTeX install).
# ---------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Serif"],
    "axes.formatter.use_mathtext": True, "font.size": 11,
    "axes.labelsize": 12, "legend.fontsize": 9,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.8, "lines.linewidth": 1.3,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": "--",
    # Ticks point into the axes on all four sides, and the data runs to
    # the frame rather than sitting inside five percent of padding.
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.xmargin": 0.0, "axes.ymargin": 0.0,
})


class CMTickFormatter(ScalarFormatter):
    """Wrap each tick label in math mode so it renders in Computer Modern.

    Without this the tick numbers are typeset in the default sans font
    while every other label uses the serif family, which is visible at
    publication resolution.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.set_useMathText(False)

    def __call__(self, x, pos=None):
        return f"${super().__call__(x, pos)}$"


def cm_x(ax):
    """Apply the Computer-Modern tick formatter to the x-axis."""
    ax.xaxis.set_major_formatter(CMTickFormatter())


def cm_y(ax):
    """Apply the Computer-Modern tick formatter to the y-axis."""
    ax.yaxis.set_major_formatter(CMTickFormatter())


def cm_both(ax):
    """Apply it to both axes, for panels whose ticks are all numeric."""
    cm_x(ax)
    cm_y(ax)

#: Collected results from every part, printed as one table at the end.
SUMMARY = []


def savefig_all(fig, basename, **kwargs):
    """Save a figure as both a high-resolution raster (PNG) and a vector
    (SVG) copy, matching the naming convention used across the post."""
    # dpi and bbox now come from the rcParams block above, so they are
    # set in one place rather than repeated at every call site.
    fig.savefig(f"{basename}.png", **kwargs)
    fig.savefig(f"{basename}.svg", **kwargs)


def banner(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68, flush=True)


# ---------------------------------------------------------------------
# The shared split. Built once so that all four parts are genuinely
# solving the same problem on the same data.
# ---------------------------------------------------------------------
DATA = load_breast_cancer()
X, y = DATA.data, DATA.target
class_names = DATA.target_names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)


# =====================================================================
# PART 1: scikit-learn
# =====================================================================
def part1_sklearn():
    banner("PART 1  |  scikit-learn")
    import sklearn
    print(f"  scikit-learn {sklearn.__version__}")
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"classes={dict(zip(*np.unique(y, return_counts=True)))}")

    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=SEED)),
        ]),
        "SVM (RBF kernel)": Pipeline([
            ("scale", StandardScaler()),
            # A support vector machine produces decision-function values,
            # not probabilities. Passing probability=True asks SVC to fit
            # an internal Platt-scaling model to convert them, which
            # scikit-learn deprecated in version 1.9 and removes in 1.11.
            # CalibratedClassifierCV performs the same calibration
            # explicitly. On this dataset the two agree to four decimal
            # places on cross-validated AUC, test AUC and test accuracy,
            # so the replacement changes nothing but the deprecation.
            ("clf", CalibratedClassifierCV(
                SVC(kernel="rbf", random_state=SEED), ensemble=False)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED)),
        ]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(n_estimators=200,
                                               learning_rate=0.05,
                                               max_depth=3,
                                               random_state=SEED)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "roc_auc", "f1"]

    print("\n  5-fold cross-validation on training set:")
    for name, pipe in models.items():
        sc = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        print(f"    {name:22s} acc={sc['test_accuracy'].mean():.4f}"
              f"+/-{sc['test_accuracy'].std():.4f}  "
              f"auc={sc['test_roc_auc'].mean():.4f}  "
              f"f1={sc['test_f1'].mean():.4f}")

    print("\n  Held-out test set performance:")
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    test_results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, lw=1.6, label=rf"{name} (AUC$={auc:.3f}$)")
        test_results[name] = dict(auc=auc, y_pred=y_pred)
        print(f"    {name:22s} test_acc={acc:.4f}  test_auc={auc:.4f}")

        # Parameter counts, for the summary table. Only the linear model
        # has a count meaningfully comparable with a network: 30
        # coefficients plus an intercept.
        n_par = 31 if name == "Logistic Regression" else None
        SUMMARY.append((name, acc, auc, n_par))

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves: classical ML models on breast cancer diagnosis")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    cm_both(ax)          # both axes are numeric
    fig.tight_layout()
    savefig_all(fig, "roc_curves_p5")
    plt.close(fig)

    # ---- Feature importance from Random Forest ----
    rf = models["Random Forest"].named_steps["clf"]
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1][:12]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(range(len(order)), importances[order][::-1], color="tab:blue",
            alpha=0.85)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DATA.feature_names[i] for i in order][::-1],
                       fontsize=8.5)
    ax.set_xlabel("feature importance (mean decrease in impurity)")
    ax.set_title("Random Forest: top 12 most important features")
    ax.grid(alpha=0.4, ls="--", axis="x")
    # The y-axis carries feature names, so only the numeric x-axis gets
    # the Computer-Modern tick formatter.
    cm_x(ax)
    fig.tight_layout()
    savefig_all(fig, "feature_importance_p5")
    plt.close(fig)

    print("\n  Random Forest top features (compare with XGBoost gain below):")
    for i in order[:6]:
        print(f"    {DATA.feature_names[i]:24s} {importances[i]:.4f}")

    # ---- Confusion matrix for the best model on the test set ----
    best_name = max(test_results, key=lambda k: test_results[k]["auc"])
    best_pred = test_results[best_name]["y_pred"]
    cm = confusion_matrix(y_test, best_pred)
    print(f"\n  Best model by test AUC: {best_name}")
    print(classification_report(y_test, best_pred, target_names=class_names))

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix: {best_name}")
    # Both axes carry class names, and grid lines over a heatmap only
    # obscure it, so neither the formatter nor the grid is applied here.
    ax.grid(False)
    fig.tight_layout()
    savefig_all(fig, "confusion_matrix_p5")
    plt.close(fig)
    print("  figures written: roc_curves_p5, feature_importance_p5, "
          "confusion_matrix_p5")


# =====================================================================
# PART 2: XGBoost
# =====================================================================
def part2_xgboost():
    banner("PART 2  |  XGBoost")
    try:
        import xgboost as xgb
    except Exception as exc:
        # A partial or broken install can raise something other than
        # ImportError, so report the reason rather than crash.
        print("  SKIPPED: xgboost unavailable  (pip install xgboost)")
        print(f"           reason: {type(exc).__name__}: {exc}")
        return
    print(f"  xgboost {xgb.__version__}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,          # L2 regularization on leaf weights
        eval_metric="logloss",
        random_state=SEED,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  XGBoost CV AUC: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

    # No eval_set: the test set is used once, for the final score only.
    model.fit(X_train, y_train, verbose=False)

    # The model was fitted on a plain NumPy array, which carries no column
    # names, so the Booster falls back to its internal indices f0, f1, ...
    # and plot_importance labels the bars with those. Attaching the real
    # names here makes the figure readable and, more to the point, makes
    # it directly comparable with the Random Forest chart above. Fitting
    # on a pandas DataFrame would achieve the same thing automatically.
    model.get_booster().feature_names = list(DATA.feature_names)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Test ROC-AUC:  {auc:.4f}")
    SUMMARY.append(("XGBoost", acc, auc, None))

    # Feature importance by gain, comparable with the Random Forest
    # ranking above. Booster keys are "f<index>" into the feature list.
    # With feature_names attached above, get_score is keyed by name.
    gains = model.get_booster().get_score(importance_type="gain")
    top = sorted(gains.items(), key=lambda kv: -kv[1])[:6]
    print("  top features by gain:")
    for name, val in top:
        print(f"    {name:24s} gain={val:.2f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xgb.plot_importance(model, max_num_features=12, importance_type="gain",
                        ax=ax, values_format="{v:.1f}", show_values=True)
    ax.set_title("XGBoost: top 12 features by gain")
    ax.set_xlabel("importance (average gain per split)")
    ax.set_ylabel("")            # the names label themselves
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(alpha=0.4, ls="--", axis="x")
    cm_x(ax)             # y-axis carries feature names
    # Headroom so the printed bar values are not clipped at the frame.
    ax.set_xlim(0, ax.get_xlim()[1] * 1.18)
    fig.tight_layout()
    savefig_all(fig, "xgb_importance_p5")
    plt.close(fig)
    print("  figure written: xgb_importance_p5")


# =====================================================================
# PART 3: TensorFlow / Keras
# =====================================================================
def part3_keras():
    banner("PART 3  |  TensorFlow / Keras")
    try:
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # quieten startup
        import tensorflow as tf
        from tensorflow import keras
    except Exception as exc:
        print("  SKIPPED: tensorflow unavailable  (pip install tensorflow)")
        print(f"           reason: {type(exc).__name__}: {exc}")
        return
    print(f"  tensorflow {tf.__version__}")

    keras.utils.set_random_seed(SEED)
    # Deterministic kernels, so a rerun on the same machine and library
    # versions reproduces the same epochs and scores exactly.
    tf.config.experimental.enable_op_determinism()

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model = keras.Sequential([
        keras.layers.Input(shape=(30,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    history = model.fit(
        Xtr, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True)],
        verbose=0,
    )

    test_loss, test_acc, _ = model.evaluate(Xte, y_test, verbose=0)
    # Keras's metrics.AUC is a thresholded (200-bin) approximation. Use the
    # same exact estimator as every other section, sklearn's roc_auc_score,
    # so that the AUC column of the summary table is computed one way.
    test_auc = roc_auc_score(y_test, model.predict(Xte, verbose=0).ravel())
    n_epochs = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    n_par = model.count_params()

    print(f"  epochs actually run: {n_epochs} of 100")
    print(f"  best val_loss epoch: {best_epoch}  (weights restored to this)")
    print(f"  Test accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    print(f"  trainable params: {n_par}")
    SUMMARY.append(("TensorFlow/Keras", test_acc, test_auc, n_par))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].axvline(best_epoch - 1, color="gray", ls="--", lw=1)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("binary cross-entropy")
    axes[0].set_title("Loss"); axes[0].legend(frameon=False)
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].axvline(best_epoch - 1, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(frameon=False)
    for a in axes:
        cm_both(a)
    fig.tight_layout()
    savefig_all(fig, "keras_training_p5")
    plt.close(fig)
    print("  figure written: keras_training_p5")


# =====================================================================
# PART 4: PyTorch
# =====================================================================
def part4_pytorch():
    banner("PART 4  |  PyTorch")
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        print("  SKIPPED: torch unavailable  (pip install torch)")
        print(f"           reason: {type(exc).__name__}: {exc}")
        return
    print(f"  torch {torch.__version__}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    X_train_t = torch.tensor(Xtr, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(Xte, dtype=torch.float32)

    class BreastCancerNet(nn.Module):
        """The same 30 -> 32 -> 16 -> 1 network as the Keras model above."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(30, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),   # logits, no sigmoid; see the loss below
            )

        def forward(self, x):
            return self.net(x)

    model = BreastCancerNet()
    # BCEWithLogitsLoss folds the sigmoid into the loss, which is more
    # numerically stable than applying sigmoid and then taking BCE.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    n_train = int(0.8 * len(X_train_t))
    X_tr, X_val = X_train_t[:n_train], X_train_t[n_train:]
    y_tr, y_val = y_train_t[:n_train], y_train_t[n_train:]

    def train(batch_size):
        """Train a fresh network with early stopping (patience 10).

        batch_size=None reproduces the full-batch loop (one optimizer step
        per epoch); batch_size=32 matches the Keras model of Part 3.
        """
        torch.manual_seed(SEED)
        model = BreastCancerNet()
        criterion = nn.BCEWithLogitsLoss()   # sigmoid folded into the loss
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loader = None
        if batch_size is not None:
            g = torch.Generator().manual_seed(SEED)
            loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                                shuffle=True, generator=g)
        best_val, ctr, best_epoch, epochs_run, steps = float("inf"), 0, 0, 0, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        tr_hist, va_hist = [], []
        for epoch in range(100):
            model.train()
            batches = [(X_tr, y_tr)] if loader is None else loader
            total, count = 0.0, 0
            for xb, yb in batches:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                steps += 1
                total += loss.item() * len(xb); count += len(xb)
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val).item()
            epochs_run += 1
            tr_hist.append(total / count); va_hist.append(val_loss)
            if val_loss < best_val:
                best_val, ctr, best_epoch = val_loss, 0, epochs_run
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                ctr += 1
                if ctr >= 10:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(X_test_t)).numpy().ravel()
        return dict(model=model, prob=prob, epochs=epochs_run, best=best_epoch,
                    steps=steps, tr=tr_hist, va=va_hist)

    runs = {}
    for label, bs in (("full batch", None), ("minibatch (32)", 32)):
        r = train(bs)
        r["auc"] = roc_auc_score(y_test, r["prob"])
        r["acc"] = accuracy_score(y_test, (r["prob"] > 0.5).astype(int))
        runs[label] = r
        print(f"  [{label}] epochs run: {r['epochs']} of 100, best val_loss epoch: "
              f"{r['best']}, optimizer steps: {r['steps']}")
        print(f"  [{label}] Test accuracy: {r['acc']:.4f}, Test AUC: {r['auc']:.4f}")

    final = runs["minibatch (32)"]
    n_par = sum(p.numel() for p in final["model"].parameters())
    print(f"  trainable params: {n_par}")
    SUMMARY.append(("PyTorch", final["acc"], final["auc"], n_par))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (label, r) in zip(axes, runs.items()):
        ax.plot(r["tr"], label="train")
        ax.plot(r["va"], label="validation")
        ax.axvline(r["best"] - 1, color="gray", ls="--", lw=1)
        ax.set_xlabel("epoch"); ax.set_title(f"PyTorch, {label}")
        ax.legend(frameon=False)
        cm_both(ax)
    axes[0].set_ylabel("binary cross-entropy")
    fig.tight_layout()
    savefig_all(fig, "pytorch_training_p5")
    plt.close(fig)
    print("  figure written: pytorch_training_p5")


# =====================================================================
# Summary
# =====================================================================
def summary():
    banner("SUMMARY  |  all models, same split, same seed")
    print(f"  {'Model':22s}{'test acc':>10s}{'test AUC':>10s}{'params':>10s}")
    print("  " + "-" * 52)
    for name, acc, auc, par in SUMMARY:
        p = "" if par is None else f"{par:,}"
        print(f"  {name:22s}{acc:10.4f}{auc:10.4f}{p:>10s}")

    if SUMMARY:
        best = max(SUMMARY, key=lambda r: r[2])
        print(f"\n  Best test AUC: {best[0]} ({best[2]:.4f})")

    with open("results_p5.txt", "w") as f:
        for name, acc, auc, par in SUMMARY:
            f.write(f"{name}\ttest_acc={acc:.4f}\ttest_auc={auc:.4f}"
                    f"\tparams={par}\n")
    print("  results also written to results_p5.txt")


def main():
    print("Verification run: all four parts of the ML-toolkit post")
    print(f"python {sys.version.split()[0]}")
    print(f"train/test split: {len(X_train)} / {len(X_test)} samples, "
          f"seed = {SEED}")

    part1_sklearn()
    part2_xgboost()
    part3_keras()
    part4_pytorch()
    summary()

    print("\n" + "=" * 68)
    print("done -- paste the whole output above")
    print("=" * 68)


if __name__ == "__main__":
    main()