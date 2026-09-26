"""
Full scikit-learn tutorial: handwritten digit classification on the UCI
Optical Recognition of Handwritten Digits dataset (1797 real scanned
8x8 digit images, 10 classes). Written to be read top to bottom as a
teaching example: EDA -> preprocessing -> baseline models ->
hyperparameter tuning -> dimensionality-reduction visualization ->
learning curves -> final evaluation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

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
    """Save a figure as both a high-resolution raster (PNG) and a vector
    (SVG) copy, matching the naming convention used across the post."""
    fig.savefig(f"{basename}.png", **kwargs)
    fig.savefig(f"{basename}.svg", **kwargs)


# =====================================================================
# 1) Load and inspect
# =====================================================================
digits = load_digits()
X, y = digits.data, digits.target
print(f"Dataset shape: {X.shape}  (n_samples, n_features)")
print(f"Classes: {sorted(set(y))}")
print(f"Class balance: {np.bincount(y)}")

# ---- Visualize a handful of raw examples (images, no numeric axes) ----
fig, axes = plt.subplots(2, 10, figsize=(11, 2.6))
rng = np.random.default_rng(0)
sample_idx = rng.choice(len(X), 20, replace=False)
for ax, idx in zip(axes.ravel(), sample_idx):
    ax.imshow(digits.images[idx], cmap="gray_r")
    ax.set_title(str(y[idx]), fontsize=9)
    ax.axis("off")
fig.suptitle("Sample handwritten digits from the dataset")
fig.tight_layout()
savefig_all(fig, "sample_digits_p6")
plt.close(fig)

# =====================================================================
# 2) Train/test split
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# =====================================================================
# 3) Baseline model comparison via cross-validation
# =====================================================================
models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ]),
    "KNN (k=5)": Pipeline([
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "SVM (RBF)": Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n5-fold CV accuracy on training set:")
cv_scores = {}
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_scores[name] = scores
    print(f"{name:22s} {scores.mean():.4f} ± {scores.std():.4f}")

# Note: x-axis carries categorical model names, so only the numeric
# y-axis gets the CM tick formatter.
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.boxplot(cv_scores.values(), tick_labels=cv_scores.keys())
ax.set_ylabel("cross-validated accuracy")
ax.set_title("Model comparison: 5-fold CV accuracy distribution")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
ax.grid(axis="y")
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "model_comparison_boxplot_p6")
plt.close(fig)

# =====================================================================
# 4) Hyperparameter tuning: grid search on the best-looking model (SVM)
# =====================================================================
print("\nGrid search over SVM hyperparameters...")
svm_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", SVC(random_state=42)),
])
param_grid = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.001, 0.01, 0.1],
    "clf__kernel": ["rbf"],
}
grid = GridSearchCV(svm_pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")

best_model = grid.best_estimator_

# =====================================================================
# 5) PCA visualization of the feature space
# =====================================================================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
explained = pca.explained_variance_ratio_
print(f"\nPCA: first 2 components explain {explained.sum()*100:.1f}% of variance")

fig, ax = plt.subplots(figsize=(6.2, 5.4))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", s=11, alpha=0.75)
legend = ax.legend(*scatter.legend_elements(), title="digit", loc="center left",
                    bbox_to_anchor=(1.0, 0.5), fontsize=8.5, frameon=False)
ax.add_artist(legend)
ax.set_xlabel(rf"PC1 ({explained[0]*100:.1f}\% var)")
ax.set_ylabel(rf"PC2 ({explained[1]*100:.1f}\% var)")
ax.set_title("PCA projection of the 64-dimensional digit feature space")
cm_x(ax)
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "pca_projection_p6")
plt.close(fig)

# =====================================================================
# 6) Learning curve for the tuned model
# =====================================================================
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, cv=cv,
    train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy", n_jobs=1
)
train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(train_sizes, train_mean, "o-", ms=5, label="training score")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
ax.plot(train_sizes, val_mean, "s-", ms=5, label="cross-validation score")
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
ax.set_xlabel("training set size")
ax.set_ylabel("accuracy")
ax.set_title("Learning curve: tuned SVM")
ax.legend(loc="lower right", frameon=False)
cm_x(ax)
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "learning_curve_p6")
plt.close(fig)

# =====================================================================
# 7) Final held-out test evaluation
# =====================================================================
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nFinal held-out test accuracy: {test_acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Confusion matrix: axis ticks are the digit classes 0-9 themselves
# (genuinely numeric), so the CM tick formatter is applied here, unlike
# the string-labeled confusion matrix in the breast-cancer post.
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6.2, 5.4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
ax.set_xlabel("predicted digit")
ax.set_ylabel("true digit")
ax.set_title(f"Confusion matrix: tuned SVM (test accuracy $= {test_acc:.4f}$)")
ax.grid(False)
cm_x(ax)
cm_y(ax)
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
savefig_all(fig, "confusion_matrix_p6")
plt.close(fig)

# ---- Show a few misclassified examples (images, no numeric axes) ----
misclassified = np.where(y_pred != y_test)[0]
print(f"\nMisclassified: {len(misclassified)} / {len(y_test)}")
if len(misclassified) > 0:
    n_show = min(10, len(misclassified))
    fig, axes = plt.subplots(1, n_show, figsize=(1.3 * n_show, 1.5))
    if n_show == 1:
        axes = [axes]
    test_images = X_test.reshape(-1, 8, 8)
    for ax, idx in zip(axes, misclassified[:n_show]):
        ax.imshow(test_images[idx], cmap="gray_r")
        ax.set_title(rf"T:{y_test[idx]} P:{y_pred[idx]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Sample misclassified digits (True vs. Predicted)")
    fig.tight_layout()
    savefig_all(fig, "misclassified_p6")
    plt.close(fig)

with open("results.txt", "w") as f:
    f.write(f"best_params={grid.best_params_}\n")
    f.write(f"best_cv_accuracy={grid.best_score_:.4f}\n")
    f.write(f"test_accuracy={test_acc:.4f}\n")
    f.write(f"n_misclassified={len(misclassified)}\n")

print("done")