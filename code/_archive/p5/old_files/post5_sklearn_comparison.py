"""
Classical ML model comparison on the Breast Cancer Wisconsin (Diagnostic)
dataset -- the running example used throughout this post so that the
scikit-learn results, the XGBoost reference code, and the Keras/PyTorch
reference code are all solving the *same* problem.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.target_names
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
      f"classes={dict(zip(*np.unique(y, return_counts=True)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ]),
    "SVM (RBF kernel)": Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                            max_depth=3, random_state=42)),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "roc_auc", "f1"]

print("\n5-fold cross-validation on training set:")
cv_results = {}
for name, pipe in models.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    cv_results[name] = scores
    print(f"{name:22s} acc={scores['test_accuracy'].mean():.4f}"
          f"±{scores['test_accuracy'].std():.4f}  "
          f"auc={scores['test_roc_auc'].mean():.4f}  "
          f"f1={scores['test_f1'].mean():.4f}")

print("\nHeld-out test set performance:")
fig, ax = plt.subplots(figsize=(6.5, 6))
test_results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    test_results[name] = dict(auc=auc, y_pred=y_pred, y_prob=y_prob)
    acc = (y_pred == y_test).mean()
    print(f"{name:22s} test_acc={acc:.4f}  test_auc={auc:.4f}")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves: classical ML models on breast cancer diagnosis")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("roc_curves.png", dpi=150)
plt.close(fig)

# ---- Feature importance from Random Forest ----
rf = models["Random Forest"].named_steps["clf"]
importances = rf.feature_importances_
order = np.argsort(importances)[::-1][:12]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.barh(range(len(order)), importances[order][::-1])
ax.set_yticks(range(len(order)))
ax.set_yticklabels([data.feature_names[i] for i in order][::-1], fontsize=9)
ax.set_xlabel("feature importance (mean decrease in impurity)")
ax.set_title("Random Forest: top 12 most important features")
fig.tight_layout()
fig.savefig("feature_importance.png", dpi=150)
plt.close(fig)

# ---- Confusion matrix for the best model on the test set ----
best_name = max(test_results, key=lambda k: test_results[k]["auc"])
best_pred = test_results[best_name]["y_pred"]
cm = confusion_matrix(y_test, best_pred)
print(f"\nBest model by test AUC: {best_name}")
print(classification_report(y_test, best_pred, target_names=feature_names))

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(feature_names)
ax.set_yticks([0, 1]); ax.set_yticklabels(feature_names)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion matrix: {best_name}")
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=150)
plt.close(fig)

with open("results.txt", "w") as f:
    for name in models:
        f.write(f"{name}: test_auc={test_results[name]['auc']:.4f}\n")
    f.write(f"best_model={best_name}\n")

print("done")
