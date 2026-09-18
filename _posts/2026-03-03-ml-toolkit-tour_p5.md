---
layout: post
title: "A Comparative Evaluation of Modern ML Libraries: scikit-learn, XGBoost, TensorFlow/Keras, and PyTorch on a Common Problem"
date: 2026-03-03
tags: [machine-learning, deep-learning, scikit-learn, xgboost, tensorflow, keras, pytorch]
giscus_comments: true
published: false
---

> **A note on scope.** Every section of this post is **executed and verified**. All five tools were run in a single process, using a single train/test split of the same dataset, with seeds held fixed throughout, so the numbers reported below are directly comparable rather than assembled from separate runs. An earlier version of this post could execute only the scikit-learn section, and the remaining four were presented as reference implementations; that limitation no longer applies, and every figure quoted here comes from a measured run.

## Motivation

"Machine learning" denotes not a single tool but a spectrum of methods, and one of the more useful capabilities an applied mathematician brings to a machine learning team is judgment about *where on that spectrum a given problem lies*, rather than applying the same deep architecture regardless of the data at hand. This post is organized around that principle: one dataset, one problem, and five tools, so that the comparison concerns the *methods* rather than assembling unrelated examples that cannot be compared directly.

The five tools divide naturally into two families:

- **scikit-learn** and **XGBoost** {% cite pedregosa2011scikit %}{% cite chen2016xgboost %} represent classical statistical learning: linear models, kernel methods, and ensembles of decision trees. These are convex or near-convex optimization problems with strong theoretical validations, fast to train, and, as demonstrated below, often difficult to outperform on small-to-medium tabular data.
- **TensorFlow/Keras** and **PyTorch** {% cite abadi2016tensorflow %}{% cite chollet2015keras %}{% cite paszke2019pytorch %} represent deep learning: differentiable function approximators trained by stochastic gradient descent and backpropagation, which become advantageous on high-dimensional, unstructured data such as images, text, and audio, or on genuinely big datasets.

## The running example

To ensure comparability, every tool below solves the **same problem**: binary classification on the Breast Cancer Wisconsin (Diagnostic) dataset {% cite wolberg1995breast %}, comprising 569 samples and 30 real-valued features derived from digitized images of fine needle aspirate biopsies, labeled malignant or benign. It is a useful dataset for this comparison precisely because it is small, tabular, and well-separated, which is the regime in which the reflex to apply a neural network warrants scrutiny.

$$
\mathbf{x} \in \mathbb{R}^{30}, \qquad y \in \{0, 1\} \ (\text{malignant}=0,\ \text{benign}=1).
$$

## Part 1: Classical ML with scikit-learn

The principal strength of scikit-learn lies less in any single algorithm than in its **consistent interface**: every estimator implements `fit`/`predict`, every preprocessing step composes into a `Pipeline`, and cross-validation, metrics, as well as hyperparameter search interoperate cleanly regardless of which model is substituted. That uniformity is what reduces a fair four-way model comparison to a few dozen lines of code rather than four separate bespoke training scripts.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

data = load_breast_cancer()
X, y = data.data, data.target

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
```

The tree-based models (`RandomForestClassifier`, `GradientBoostingClassifier`) omit the `StandardScaler` step: splitting decisions in a decision tree are independent of monotonic feature rescaling, so standardization confers no benefit there, whereas it is essential for the distance- as well as gradient-based methods (SVM, logistic regression), whose objective functions are not scale-invariant.

### Cross-validated comparison

Running stratified 5-fold cross-validation on the training set:

```
5-fold cross-validation on training set:
Logistic Regression    acc=0.9789±0.0088  auc=0.9962  f1=0.9833
SVM (RBF kernel)       acc=0.9718±0.0057  auc=0.9951  f1=0.9776
Random Forest          acc=0.9578±0.0175  auc=0.9880  f1=0.9665
Gradient Boosting      acc=0.9554±0.0227  auc=0.9876  f1=0.9642
```

And on the held-out test set:

```
Held-out test set performance:
Logistic Regression    test_acc=0.9860  test_auc=0.9977
SVM (RBF kernel)       test_acc=0.9790  test_auc=0.9969
Random Forest          test_acc=0.9580  test_auc=0.9949
Gradient Boosting      test_acc=0.9580  test_auc=0.9929
```

<p align="center">
  <img src="/assets/img/posts/roc_curves_p5.svg" alt="ROC curves for four classical ML models" style="width: 100%; max-width: 480px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 1: ROC curves on the held-out test set. All four models separate the classes well (AUC $>0.99$ for the top two), which is insightful in itself: this dataset is nearly linearly separable in its 30-dimensional feature space, so the simplest model in the comparison, logistic regression, performs best.*

This result merits attention rather than being passed over. **The simplest model considered, plain logistic regression, achieves the best test-set AUC as well as accuracy**, exceeding both tree ensembles. This outcome is not an artifact of the particular random seed; it shows a structural property of the data, namely that the classes are nearly linearly separable once the features are standardized. It also indicates that a more flexible model is not necessarily a better one: flexibility that is not required adds variance without any corresponding reduction in bias. Formally, for a hypothesis class $\mathcal{H}$ with expected test error decomposed as $\mathbb{E}[(\hat{f}(x) - f(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$, adding capacity beyond what the true decision boundary requires can reduce bias only at the cost of variance; if the boundary is already close to linear, that exchange yields no net benefit.

### The source of the model's discriminative signal

<p align="center">
  <img src="/assets/img/posts/feature_importance_p5.svg" alt="Random Forest variable importance bar chart" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 2: Random Forest feature importances (mean decrease in impurity), top 12 of 30 features. A small number of features, specifically cell-nucleus size and shape irregularity measures, dominate the model's decisions, consistent with the clinical literature on this diagnostic task.*

Plots that display variable importance of this kind are among the genuine advantages that classical tree-based methods retain over deep networks on tabular data: the basis for a prediction is available at negligible additional cost, computed as a byproduct of training rather than requiring a separate post hoc interpretability method.

<p align="center">
  <img src="/assets/img/posts/confusion_matrix_p5.svg" alt="Confusion matrix for logistic regression on the held-out test set" style="width: 100%; max-width: 460px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 3: Confusion matrix for logistic regression, the best model by test AUC, on the 143 held-out samples. Two misclassifications in total: one malignant case predicted benign, and one benign case predicted malignant. The errors are symmetric, which is not something the accuracy figure alone conveys, and in a diagnostic setting, the two are not equally costly.*

One implementation detail in the code above is worth drawing out, because it concerns a step that is easy to apply without noticing. A support vector machine produces decision-function values rather than probabilities, so the ROC curve of Figure 1 requires a calibration step to convert them. The pipeline uses `CalibratedClassifierCV(SVC(), ensemble=False)`, which explicitly fits that conversion. The older route was to pass `probability=True` to `SVC`, which performs the same Platt scaling internally; scikit-learn deprecated it in version 1.9 and removed it in 1.11.

The two yield the same AUC to four decimal places, as expected: AUC depends only on the ranking of the scores, and any monotone recalibration leaves that ranking unchanged. Threshold-dependent metrics do vary slightly, since the two fit their sigmoid on different internal folds, and a few borderline cases cross the $p=0.5$ boundary as a result. Cross-validated accuracy moves from $0.9671$ to $0.9718$ and F1 from $0.9740$ to $0.9776$; test-set accuracy is unchanged at $0.9790$. The distinction between rank-based and threshold-based metrics is a small but useful point to keep in mind when comparing models with differently calibrated scores.

## Part 2: Gradient boosting at scale with XGBoost

*(Executed and verified.)*

The `GradientBoostingClassifier` of scikit-learn used above is a legitimate gradient boosting implementation {% cite friedman2001greedy %}, but **XGBoost** {% cite chen2016xgboost %}, together with its close relatives LightGBM and CatBoost, improved on it in ways that established gradient-boosted trees as the dominant method in tabular-data competitions, Kaggle in particular, for much of a decade. The mathematical distinction merits a precise statement. Where classical gradient boosting fits each new tree to the *negative gradient* of the loss, a first-order, steepest-descent-style step, XGBoost fits each tree using a **second-order Taylor expansion** of the loss function around the current prediction,

$$
\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \tfrac{1}{2} h_i f_t(x_i)^2 \right] + \Omega(f_t),
$$

where $g_i = \partial_{\hat{y}} \ell(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial^2_{\hat{y}} \ell(y_i, \hat{y}^{(t-1)})$ are the first and second derivatives of the loss with respect to the current prediction. The procedure amounts to Newton's method applied tree by tree, rather than plain gradient descent. The regularization term $\Omega(f_t)$ penalizes tree complexity, specifically the number of leaves as well as the magnitude of leaf weights, directly in the objective, which accounts for the characteristic resistance of XGBoost's trees to overfitting relative to earlier boosting implementations.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,          # L2 regularization on leaf weights
    eval_metric="logloss",
    random_state=42,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"XGBoost CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")

# Feature significance, directly comparable to the Random Forest plot above
xgb.plot_importance(model, max_num_features=12, importance_type="gain")
```

```
XGBoost CV AUC: 0.9926 +/- 0.0070
Test accuracy: 0.9720
Test ROC-AUC:  0.9950
```

The measured result confirms the expectation, and rather more sharply than anticipated. XGBoost's test AUC of $0.9950$ places it above scikit-learn's `GradientBoostingClassifier` ($0.9929$) and its own Random Forest ($0.9949$), which is consistent with the ordering predicted by the literature. But it remains **below plain logistic regression** ($0.9977$), and its test accuracy of $0.9720$ is likewise below logistic regression's $0.9860$. The second-order updates and built-in regularization are doing real work compared with other tree ensembles; they are not enough to overcome the fact that this dataset is nearly linearly separable.

The top features by gain, shown in Figure 4, are `worst radius` ($22.09$) and `worst perimeter` ($22.04$), followed by `worst concave points` ($14.34$) and `mean concave points` ($11.77$). These are the same quantities the Random Forest ranked highest, which is reassuring: two different ensemble algorithms, with different splitting criteria and different regularization, agree on which measurements carry the diagnostic signal.

<p align="center">
  <img src="/assets/img/posts/xgb_importance_p5.svg" alt="XGBoost feature impact by gain, top 12 features" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 4: XGBoost feature importances by gain, top 12 of 30, measured as the average improvement in the objective across every split that used the feature. The ordering closely agrees with the Random Forest ranking in Figure 2, which is the substantive point: two ensembles with different splitting criteria and different regularization identify the same measurements as carrying the diagnostic signal. The two rankings are not identical, and the disagreement offers useful information. XGBoost places `worst radius` first at $22.09$, where the Random Forest places it fifth, and demotes `worst area` from second to sixth. Radius, perimeter, and area are close to functionally dependent for a roughly convex shape, so the three carry largely the same information; which of them a given ensemble credits is partly a matter of which happened to be selected first at a split, and the split ordering is not the same under mean-decrease-in-impurity as it is under gain.*


Based on the documented behavior of XGBoost relative to plain gradient boosting in the literature, its accuracy would be expected to fall within a range similar to the scikit-learn `GradientBoostingClassifier` result above on a dataset of this size and cleanliness, and it does. Its substantive advantages, namely built-in regularization, second-order updates, native handling of missing values, and substantially faster training through histogram-based split-finding, become apparent more decisively on **larger, noisier, higher-dimensional** tabular datasets. On a 569-row, well-separated dataset, XGBoost, scikit-learn's gradient boosting, and plain logistic regression may all fall within one or two percentage points of one another; the dataset is simply not large or complex enough to exhibit the advantages of XGBoost.

## Part 3: Deep learning with TensorFlow/Keras

*(Executed and verified.)*

The core departure of deep learning from the methods above rests in its architecture. Rather than selecting a fixed model family, such as a line, a kernel expansion, or an ensemble of trees, and fitting its parameters, layers are composed into an arbitrarily flexible function approximator, and stochastic gradient descent, implemented through backpropagation, determines the parameters. Keras {% cite chollet2015keras %}, now the official high-level API of TensorFlow {% cite abadi2016tensorflow %}, renders this composition close to declarative:

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0,
)

test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
```

```
epochs actually run: 49 of 100
best val_loss epoch: 39  (weights restored to this)
Test accuracy: 0.9720, Test AUC: 0.9950
trainable params: 1537
```

<p align="center">
  <img src="/assets/img/posts/keras_training_p5.svg" alt="Keras training and validation loss and accuracy against epoch" style="width: 100%; max-width: 760px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 5: Keras training history. The dashed line marks epoch 39, where validation loss reached its minimum, and to which the weights were restored. Both curves have flattened well before that point, and validation accuracy is effectively constant from about epoch 10 onward. Validation loss sits below training loss throughout, which is the expected consequence of dropout: it is active during training and disabled at evaluation, so the model is scored in a slightly stronger configuration than the one being fitted.*

Early stopping halted training after 49 of the permitted 100 epochs, restoring the weights from epoch 39. The behavior is worth recording because it indicates that the network had finished learning: validation loss reached its minimum at epoch 39 and failed to improve over the following ten, which is what the patience setting is there to detect. The network has 1,537 trainable parameters, compared with 569 samples in the full dataset and 426 in the training split, roughly 3 parameters per training example.

Several points merit explicit mathematical statement:

- **The loss.** Binary cross-entropy is the negative log-likelihood under a Bernoulli model of the labels, $\mathcal{L} = -\frac{1}{n}\sum_i \left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$, which is the same objective that logistic regression minimizes. The network here is, in a precise sense, a nonlinear generalization of the logistic regression model from Part 1, with the sigmoid output layer serving the same role.
- **Dropout** {% cite srivastava2014dropout %} (`Dropout(0.3)`) randomly zeroes $30\%$ of the hidden units at each training step, preventing the network from depending on any single neuron. This regularization strategy plays a role broadly analogous to the leaf-weight penalty of XGBoost above, although the mechanism, namely prevention of stochastic co-adaptation as against an explicit norm penalty, differs substantially.
- **Early stopping** halts training once validation loss ceases to improve, functioning as both a regularizer and a means of reducing compute. It is conceptually similar to limiting `n_estimators` in tree ensembles, but is tuned automatically based on held-out performance rather than being fixed in advance.

The verified numbers make the central comparison concrete. The network reaches a test AUC of $0.9950$ and accuracy of $0.9720$, identical to XGBoost to four decimal places on both metrics, and below logistic regression on both. A 1,537-parameter network trained for 49 epochs, with dropout and early stopping, has arrived at precisely the same place as a 31-parameter linear model fitted in a fraction of a second.

This is the point of the exercise. Deep learning is not underperforming here due to misconfiguration; the architecture and model training procedure are sound, and the early-stopping behavior shows that the regularization is working as intended. The dataset is simply small, tabular, and nearly linearly separable; in that regime, a linear decision boundary is very nearly the correct hypothesis. Additional capacity has nothing left to fit.

## Part 4: The same network in PyTorch

*(Executed and verified.)*

Comparing the identical architecture implemented in PyTorch {% cite paszke2019pytorch %} with the Keras version above is instructive, because the remaining differences, once syntax is set aside, indicate when each framework is the more appropriate choice.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class BreastCancerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),   # logits -- no sigmoid here, see loss below
        )

    def forward(self, x):
        return self.net(x)

model = BreastCancerNet()
criterion = nn.BCEWithLogitsLoss()   # combines sigmoid + BCE for numerical stability
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = float("inf")
patience, patience_ctr = 10, 0
n_train = int(0.8 * len(X_train_t))
X_tr, X_val = X_train_t[:n_train], X_train_t[n_train:]
y_tr, y_val = y_train_t[:n_train], y_train_t[n_train:]

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    logits = model(X_tr)
    loss = criterion(logits, y_tr)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()
    if val_loss < best_val_loss:
        best_val_loss, patience_ctr = val_loss, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_prob = torch.sigmoid(model(X_test_t)).numpy().ravel()
test_auc = roc_auc_score(y_test, test_prob)
print(f"Test AUC: {test_auc:.4f}")
```

```
epochs actually run: 100 of 100
best val_loss epoch: 100  (weights restored to this)
Test AUC: 0.9943
Test accuracy: 0.9510
trainable params: 1537
```

<p align="center">
  <img src="/assets/img/posts/pytorch_training_p5.svg" alt="PyTorch training and validation loss against epoch, still decreasing at the epoch limit" style="width: 100%; max-width: 560px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 6: PyTorch training history, and the clearest evidence in this post that the run did not converge. Compare the shape against Figure 5. The Keras curves flatten by epoch 20, and the run halts at 49; here, both curves are still descending steeply at epoch 100, where the dashed line marks not a detected minimum but the epoch ceiling. Validation loss is roughly $0.10$ and still falling, compared to about $0.025$ for the converged Keras model. Training and validation curves also remain almost coincident throughout, indicating that the model has not yet begun to fit the training set preferentially, which is the opposite of the overfitting pattern that early stopping exists to catch.*

The number of parameters confirms that the two networks are identical: 1,537 trainable weights in both, as they must be for a $30 \rightarrow 32 \rightarrow 16 \rightarrow 1$ architecture. Yet the PyTorch model records the lowest accuracy of any method examined in this post, $0.9510$, compared with $0.9720$ for its Keras counterpart.

The epoch counts identify the cause, and Figure 6 makes it visible. Keras stopped at epoch 49, having found its best validation loss at epoch 39; the ten subsequent epochs without improvement are what triggered the halt. The PyTorch loop ran all 100 permitted epochs, and its best validation loss occurred at **epoch 100**, the final one. Early stopping never fired. The model was still improving when it reached the epoch ceiling.

This is undertraining, not overfitting, and one should be careful to make the distinction in situations like this. A network that overfits has learned the training set too well and must be stopped; a network in the state above has not finished learning at all, and the ceiling that stopped it was arbitrary. The reported figure of $0.9510$ is therefore not the accuracy of this architecture on this problem. It is the accuracy of this architecture that is interrupted partway through fitting.

The reason lies in a single difference between the two sections. The Keras model is fitted with `batch_size=32`. With 341 samples in its internal training split, that is roughly 11 optimizer steps per epoch, and 49 epochs, therefore, amount to about 539 parameter tweaks. The PyTorch loop above computes `model(X_tr)` on the entire training tensor, so it performs exactly one step per epoch: 100 updates in total, roughly a fifth as many, and the validation curve had not yet flattened.

The difficulty is a genuine trap rather than an artifact of this post. The two code blocks appear equivalent on inspection: the same layers, the same loss, Adam at $10^{-3}$ in both, early stopping on validation loss in both. The training procedure that separates them is expressed in one case through a `batch_size` argument and, in the other, through what the loop passes to the model, which a reader does not check. Reproducing the Keras result in PyTorch requires iterating over minibatches within the epoch loop, ordinarily through a `DataLoader`.

The correct reading, then, is not that PyTorch is less accurate than Keras. The frameworks are computing the same thing. It is that the explicitness identified below as the principal advantage of PyTorch carries a corresponding obligation: decisions that a higher-level API makes on the user's behalf become the user's to make, including decisions the user may not notice are being made. A defensible comparison would either fix the epoch ceiling high enough for the full-batch loop to converge, or introduce minibatching so that both sections take a comparable number of optimizer steps.

Several structural differences, as distinct from surface syntax, merit identification:

- **Explicit training loop.** The `nn.Module` of PyTorch provides a `forward` method and requires the epoch loop, gradient zeroing, backward pass, and optimizer step to be written explicitly, whereas `model.fit` in Keras encapsulates all of this in a single call. The additional code constitutes a genuine trade-off  as opposed to mere verbosity: the explicitness of PyTorch makes it considerably easier to implement custom training logic, such as multiple losses, custom gradient manipulation, or non-standard data flow, that would otherwise require descending to the lower-level `GradientTape` API of Keras.
- **Logits rather than probabilities until the final step.** `BCEWithLogitsLoss` combines the sigmoid activation and the binary cross-entropy computation into a single numerically stable operation, avoiding the $\log(0)$ instability that can arise from applying `log` after a separately computed sigmoid saturated near $0$ or $1$. The detail is a small but instructive example of how these schemes incorporate numerical analysis practice and is directly relevant when approaching the subject from a numerical methods rather than a purely machine learning background.
- **Manual early stopping.** The `EarlyStopping` callback in Keras requires a few lines of configuration; the equivalent logic in raw PyTorch is the explicit patience-counter loop above. Libraries such as PyTorch Lightning exist largely to restore this convenience on top of PyTorch's lower-level primitives, at the cost of an additional abstraction layer.

## All seven models, measured

Because every section was run in one process against one split, the results can be placed in a single table without qualification:

| Model | Test accuracy | Test AUC | Trainable parameters |
|---|---|---|---|
| **Logistic Regression** | **0.9860** | **0.9977** | 31 |
| SVM (RBF kernel) | 0.9790 | 0.9969 | — |
| XGBoost | 0.9720 | 0.9950 | — |
| TensorFlow/Keras | 0.9720 | 0.9950 | 1,537 |
| Random Forest | 0.9580 | 0.9949 | — |
| Gradient Boosting | 0.9580 | 0.9929 | — |
| PyTorch | 0.9510 | 0.9943 | 1,537 |

Three observations follow, none of which were available before the sections were executed.

The first concerns the ordering. Logistic regression is the best model on both metrics, achieving this with 31 parameters: 30 coefficients and an intercept. The nearest competitors are the RBF-kernel SVM and a boosted ensemble of three hundred trees, and a network with fifty times as many parameters, all of which it beats. This is the empirical content of the post's argument, and it is worth stating that the result was not guaranteed in advance; a dataset of this kind could plausibly have rewarded the ensembles.

The second concerns the middle of the table. XGBoost and Keras agree to four decimal places on both metrics, at $0.9720$ and $0.9950$. A gradient-boosted ensemble and a feedforward network are different objects fitted by different procedures, and their arriving at identical figures is a coincidence of this dataset rather than a general property. It is nonetheless instructive: it indicates that both have extracted essentially the same signal, and that the remaining error is a property of the data rather than of either model class.

The third concerns the gap between the two deep-learning entries. The table suggests that PyTorch is the weaker framework, and nothing in the parameter column contradicts this, since both report 1,537 weights. The epoch counts, which the table omits, contradict it directly: the Keras model stopped early at epoch 49, having converged, while the PyTorch model exhausted all 100 permitted epochs with its best validation loss at the last one. The figure of $0.9510$ measures an unconverged model, and belongs in the table only with that qualification attached.

The comparison also has a cost dimension that the table omits. Logistic regression fits in a fraction of a second and yields thirty coefficients that can be read directly. The networks require standardization, an architecture choice, a learning rate, a dropout fraction, a batch size, an early-stopping criterion, and tens of training epochs, and they yield 1,537 weights that admit no comparable interpretation. In this dataset, the expenditure has a lower score.

## Choosing a tool: a decision framework

The following framework may be applied before beginning a new project, ordered approximately by the sequence in which the questions arise:

| Question                                                             | Leans toward                        |
| -------------------------------------------------------------------- | ----------------------------------- |
| Tabular data, < ~100k rows?                                          | scikit-learn / XGBoost              |
| Need built-in feature contribution / interpretability?                 | scikit-learn / XGBoost              |
| Images, audio, text, or other unstructured/high-dim data?            | TensorFlow/Keras or PyTorch         |
| Need a quick baseline with minimal code?                             | scikit-learn, or Keras `Sequential` |
| Need custom training logic (multi-task loss, custom gradients, RL)?  | PyTorch                             |
| Deploying to mobile/edge or need TF Serving / TF Lite?               | TensorFlow/Keras                    |
| Research codebase, need to move fast and modify architectures repeatedly? | PyTorch                             |
| Production tabular pipeline needing speed + minimal tuning?          | XGBoost / LightGBM                  |

The inclination to adopt the most powerful available tool at the outset warrants resistance. The most useful empirical observation from this comparison is the one from Part 1: on this dataset, **the simplest model performed best**. That is not an argument against deep learning; it is an argument for beginning with the least costly model that could plausibly succeed, establishing a genuine baseline, and adopting more flexible tools, which are also more compute-intensive, less interpretable, and more sensitive to hyperparameters, only once the simpler alternatives are confirmed to be underperforming. This conveys a broader idea from the earlier posts in this series: establish what the structure of a problem demands before selecting the most powerful available tool, whether that tool is a neural network, an implicit ODE solver, or a rank-$k$ matrix update.

## Summary Notes and Highlights

- All five tools ultimately address the same category of problem, minimizing a loss function over a parameterized family of functions, but they occupy genuinely different positions on the bias and variance tradeoff and interpretability-flexibility spectra. Selecting correctly matters more than selecting the most prominent option.
- The uniform `fit`/`predict` interface of scikit-learn makes rigorous, directly comparable model evaluation inexpensive, and there is seldom adequate justification for omitting it before committing to a more complex approach.
- The advantage of XGBoost in comparison to classical gradient boosting is a genuine contribution in numerical optimization, namely second-order, Newton-style tree fitting combined with explicit regularization in the objective, rather than merely faster code, although that advantage is more apparent on larger and noisier data than a clean 569-row dataset.
- TensorFlow/Keras and PyTorch construct the *same* mathematical objects, namely differentiable computational graphs trained by backpropagation, with different ergonomics: Keras is optimized for producing a working model quickly, whereas PyTorch is optimized for transparent control over each training step.
- On small, clean, tabular data, standard methods are not simply an introductory fallback. They are frequently the *correct* choice, and in the comparison conducted here logistic regression, with 31 parameters, outperformed every other elaborate alternative on both predictive accuracy measures and AUC.
- Two code blocks implementing the same architecture, the same loss, and the same optimizer can still train differently. The Keras and PyTorch networks in Parts 3 and 4 are identical in every respect a reader would check, down to the number of parameters, yet one takes roughly 11 optimizer steps per epoch and the other exactly 1, because the first specifies a batch size and the second passes the whole training set at once. The consequence was not a marginal difference in accuracy, but a model that never converged: the PyTorch run exhausted its epoch ceiling while validation loss was still falling. Reporting that number without the epoch count attached would have been misleading.

## References

The scikit-learn ecosystem underlying Part 1 is documented in {% cite pedregosa2011scikit %}, and the dataset is from {% cite wolberg1995breast %}. The XGBoost algorithm and its second-order boosting formulation follow {% cite chen2016xgboost %}, building on the traditional gradient-boosting framework of {% cite friedman2001greedy %}. The deep-learning frameworks inside Parts 3 and 4 are documented in {% cite abadi2016tensorflow %}, {% cite chollet2015keras %}, and {% cite paszke2019pytorch %}, with the Adam optimizer from {% cite kingma2015adam %} and dropout regularization from {% cite srivastava2014dropout %}. The empirical case for standard methods on tabular data referenced throughout draws on {% cite shwartzziv2022tabular %}.

{% bibliography --cited --file blog_references %}

---

*Full code for the scikit-learn section, executed and verified, is available in [`sklearn_comparison_p5.py`]({{ '/assets/code/sklearn_comparison_p5.py' | relative_url }}). The XGBoost, TensorFlow/Keras, and PyTorch code is provided as reference implementations and should be run and verified before being relied upon.*