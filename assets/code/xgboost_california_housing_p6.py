"""
XGBoost on California Housing price prediction: linear baseline, early
stopping, randomized hyperparameter search, and feature importance.

Data note: sklearn's own fetch_california_housing() host was unreachable
in the environment this was verified in, so the same 8-feature dataset is
reconstructed here from the raw census extract distributed with the
companion repository for Geron's "Hands-On Machine Learning" (a standard,
widely used mirror of the same underlying 1990 census data). 207 of
20,640 rows with missing total_bedrooms are dropped. If sklearn's loader
is reachable in your environment, feel free to swap back to
`fetch_california_housing(as_frame=True)` directly.

Dependencies: numpy, pandas, xgboost, scikit-learn, scipy, matplotlib, requests
Run:  python xgboost_california_housing_p6.py
"""
import os
import io
import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
# Note: newer scikit-learn (>=1.4) removed mean_squared_error's `squared=`
# argument; root_mean_squared_error is the current equivalent of
# mean_squared_error(..., squared=False) used in the original post text.
from scipy.stats import uniform, randint
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
# Data: reconstruct the 8-feature California Housing dataset
# ---------------------------------------------------------------------
RAW_CSV_URL = ("https://raw.githubusercontent.com/ageron/handson-ml2/"
               "master/datasets/housing/housing.csv")
RAW_CSV_PATH = "housing_raw.csv"

if not os.path.exists(RAW_CSV_PATH):
    resp = requests.get(RAW_CSV_URL, timeout=30)
    resp.raise_for_status()
    with open(RAW_CSV_PATH, "w") as f:
        f.write(resp.text)

df = pd.read_csv(RAW_CSV_PATH)
df = df.dropna(subset=["total_bedrooms"])  # 207/20640 rows dropped (~1%)

X = pd.DataFrame({
    "MedInc": df["median_income"],
    "HouseAge": df["housing_median_age"],
    "AveRooms": df["total_rooms"] / df["households"],
    "AveBedrms": df["total_bedrooms"] / df["households"],
    "Population": df["population"],
    "AveOccup": df["population"] / df["households"],
    "Latitude": df["latitude"],
    "Longitude": df["longitude"],
})
y = df["median_house_value"] / 100_000.0  # match sklearn's $100k units

print(f"Reconstructed dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(X.describe())
print(f"\nTarget (median house value, $100k units): "
      f"mean={y.mean():.2f}, std={y.std():.2f}")

# ---------------------------------------------------------------------
# Step 1: linear baseline
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"Linear Regression  RMSE={root_mean_squared_error(y_test, y_pred_lr):.4f}  "
      f"MAE={mean_absolute_error(y_test, y_pred_lr):.4f}  "
      f"R2={r2_score(y_test, y_pred_lr):.4f}")

# ---------------------------------------------------------------------
# Step 2: XGBoost with a proper validation split and early stopping
# ---------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=1000,          # upper bound; early stopping will cut this short
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=30,
    random_state=42,
)
model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
print(f"Best iteration: {model.best_iteration}")
print(f"Best validation RMSE: {model.best_score:.4f}")

# Plot train vs validation RMSE across boosting rounds (both axes are
# linear/numeric, so the CM tick formatter applies to both).
results = model.evals_result()
epochs = len(results["validation_0"]["rmse"])
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(range(epochs), results["validation_0"]["rmse"], label="train")
ax.plot(range(epochs), results["validation_1"]["rmse"], label="validation")
ax.axvline(model.best_iteration, color="gray", linestyle="--",
           label=f"best iteration ({model.best_iteration})")
ax.set_xlabel("boosting round")
ax.set_ylabel("RMSE")
ax.set_title("XGBoost training curve: train vs. validation RMSE")
ax.legend()
cm_x(ax)
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "xgb_learning_curve_p6")
plt.close(fig)

# ---------------------------------------------------------------------
# Step 3: hyperparameter tuning via randomized search
# ---------------------------------------------------------------------
param_dist = {
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.29),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_lambda": uniform(0.0, 5.0),
    "reg_alpha": uniform(0.0, 2.0),
}
search = RandomizedSearchCV(
    xgb.XGBRegressor(n_estimators=300, objective="reg:squarederror", random_state=42),
    param_distributions=param_dist,
    n_iter=40,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best CV RMSE: {-search.best_score_:.4f}")

# ---------------------------------------------------------------------
# Step 4: feature importance and interpretation
# ---------------------------------------------------------------------
# Note: y-axis carries categorical feature names, so only the numeric
# x-axis gets the CM tick formatter.
fig, ax = plt.subplots(figsize=(7, 5))
xgb.plot_importance(model, importance_type="gain", max_num_features=8,
                     title="XGBoost feature importance (gain)", ax=ax,
                     values_format="{v:.2f}")
ax.set_xlim(0, 8.6)   # headroom so the bar-value labels are not clipped
ax.grid(axis="x")
cm_x(ax)
fig.tight_layout()
savefig_all(fig, "xgb_feature_importance_p6")
plt.close(fig)

# ---------------------------------------------------------------------
# Step 5: final test-set evaluation
# ---------------------------------------------------------------------
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"XGBoost (tuned)  RMSE={root_mean_squared_error(y_test, y_pred):.4f}  "
      f"MAE={mean_absolute_error(y_test, y_pred):.4f}  "
      f"R2={r2_score(y_test, y_pred):.4f}")

with open("xgboost_results.txt", "w") as f:
    f.write(f"n_rows={X.shape[0]}\n")
    f.write(f"linear_rmse={root_mean_squared_error(y_test, y_pred_lr):.4f} "
            f"linear_r2={r2_score(y_test, y_pred_lr):.4f}\n")
    f.write(f"xgb_best_iteration={model.best_iteration} xgb_best_val_rmse={model.best_score:.4f}\n")
    f.write(f"search_best_params={search.best_params_}\n")
    f.write(f"final_rmse={root_mean_squared_error(y_test, y_pred):.4f} "
            f"final_r2={r2_score(y_test, y_pred):.4f}\n")

print("done")
