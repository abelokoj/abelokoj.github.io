"""
Part A: A from-scratch Metropolis-Hastings sampler solving a small,
physically-motivated inverse problem: given noisy "sensor" arrival-time and
amplitude measurements, infer the unknown source parameters (amplitude and
location) of a wave pulse -- a toy stand-in for exactly the kind of inverse
problem underlying real-time tsunami early-warning systems.

Part B: Bayesian linear regression (via MCMC) on the real-world scikit-learn
diabetes dataset, comparing posterior credible intervals against classical
OLS point estimates and confidence intervals.
"""
import numpy as np
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
    """Save a figure as both a high-resolution raster (PNG) and a vector
    (SVG) copy, matching the naming convention used across the post."""
    fig.savefig(f"{basename}.png", **kwargs)
    fig.savefig(f"{basename}.svg", **kwargs)


rng = np.random.default_rng(7)

# =====================================================================
# PART A: Inferring a wave source from noisy sensor data
# =====================================================================
# Forward model: a 1D damped travelling pulse observed at a set of sensor
# positions. Given true source amplitude A and location x0, sensor k at
# position x_k records a peak amplitude that decays with distance and
# arrives at a time proportional to distance / wave speed.
SENSOR_POSITIONS = np.array([1.0, 2.5, 4.0, 6.0, 8.5])
WAVE_SPEED = 2.0
TRUE_AMPLITUDE = 3.0
TRUE_X0 = 0.2
NOISE_STD_AMP = 0.15
NOISE_STD_TIME = 0.08


def forward_model(amplitude, x0):
    dist = np.abs(SENSOR_POSITIONS - x0)
    peak_amp = amplitude / (1.0 + 0.3 * dist)     # geometric spreading/damping
    arrival_time = dist / WAVE_SPEED
    return peak_amp, arrival_time


# ---- Generate synthetic "observed" sensor data from the true source ----
true_peak_amp, true_arrival = forward_model(TRUE_AMPLITUDE, TRUE_X0)
obs_amp = true_peak_amp + rng.normal(0, NOISE_STD_AMP, size=len(SENSOR_POSITIONS))
obs_time = true_arrival + rng.normal(0, NOISE_STD_TIME, size=len(SENSOR_POSITIONS))

print("PART A: Wave-source inverse problem")
print(f"True source: amplitude={TRUE_AMPLITUDE}, x0={TRUE_X0}")
print(f"Observed peak amplitudes: {np.round(obs_amp, 3)}")
print(f"Observed arrival times:   {np.round(obs_time, 3)}")

# ---- Priors: broad, physically reasonable ranges ----
def log_prior(amplitude, x0):
    if not (0.0 < amplitude < 10.0):
        return -np.inf
    if not (-2.0 < x0 < 2.0):
        return -np.inf
    return 0.0   # flat (uniform) prior within bounds


# ---- Likelihood: Gaussian observation noise on both amplitude and time ----
def log_likelihood(amplitude, x0):
    pred_amp, pred_time = forward_model(amplitude, x0)
    ll_amp = -0.5 * np.sum(((obs_amp - pred_amp) / NOISE_STD_AMP) ** 2)
    ll_time = -0.5 * np.sum(((obs_time - pred_time) / NOISE_STD_TIME) ** 2)
    return ll_amp + ll_time


def log_posterior(amplitude, x0):
    lp = log_prior(amplitude, x0)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(amplitude, x0)


# ---- Metropolis-Hastings sampler, written from first principles ----
def metropolis_hastings(log_post_fn, theta0, n_samples, step_sizes, rng):
    theta = np.array(theta0, dtype=float)
    current_lp = log_post_fn(*theta)
    samples = np.zeros((n_samples, len(theta)))
    n_accept = 0
    for i in range(n_samples):
        proposal = theta + rng.normal(0, step_sizes)
        proposal_lp = log_post_fn(*proposal)
        log_accept_ratio = proposal_lp - current_lp
        if np.log(rng.uniform()) < log_accept_ratio:
            theta = proposal
            current_lp = proposal_lp
            n_accept += 1
        samples[i] = theta
    return samples, n_accept / n_samples


N_SAMPLES = 60000
BURN_IN = 10000
samples, accept_rate = metropolis_hastings(
    log_posterior, theta0=[1.0, 0.0], n_samples=N_SAMPLES,
    step_sizes=[0.15, 0.1], rng=rng
)
print(f"\nMCMC acceptance rate: {accept_rate:.3f}")

posterior = samples[BURN_IN:]
amp_samples, x0_samples = posterior[:, 0], posterior[:, 1]

amp_mean, amp_ci = amp_samples.mean(), np.percentile(amp_samples, [2.5, 97.5])
x0_mean, x0_ci = x0_samples.mean(), np.percentile(x0_samples, [2.5, 97.5])
print(f"Posterior amplitude: mean={amp_mean:.3f}, 95% CI=[{amp_ci[0]:.3f}, {amp_ci[1]:.3f}] "
      f"(true={TRUE_AMPLITUDE})")
print(f"Posterior x0:        mean={x0_mean:.3f}, 95% CI=[{x0_ci[0]:.3f}, {x0_ci[1]:.3f}] "
      f"(true={TRUE_X0})")

# ---- Figure: trace plots + posterior histograms + joint posterior ----
# All six panels here have purely numeric axes, so the CM tick formatter
# is applied uniformly across the whole grid.
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
axes[0, 0].plot(samples[:, 0], lw=0.4, color="tab:blue")
axes[0, 0].axvline(BURN_IN, color="red", linestyle="--", lw=1.2, label="end of burn-in")
axes[0, 0].axhline(TRUE_AMPLITUDE, color="green", linestyle=":", lw=1.2, label="true value")
axes[0, 0].set_title("Trace: amplitude"); axes[0, 0].legend(fontsize=7.5, frameon=False)

axes[0, 1].plot(samples[:, 1], lw=0.4, color="tab:orange")
axes[0, 1].axvline(BURN_IN, color="red", linestyle="--", lw=1.2)
axes[0, 1].axhline(TRUE_X0, color="green", linestyle=":", lw=1.2)
axes[0, 1].set_title("Trace: source location $x_0$")

axes[0, 2].scatter(amp_samples[::20], x0_samples[::20], s=2, alpha=0.3, color="tab:blue")
axes[0, 2].plot(TRUE_AMPLITUDE, TRUE_X0, "r*", markersize=14, label="true source")
axes[0, 2].set_xlabel("amplitude"); axes[0, 2].set_ylabel(r"$x_0$")
axes[0, 2].set_title("Joint posterior samples")
axes[0, 2].legend(fontsize=7.5, frameon=False)

axes[1, 0].hist(amp_samples, bins=60, density=True, alpha=0.75, color="tab:blue")
axes[1, 0].axvline(TRUE_AMPLITUDE, color="green", linestyle=":", lw=1.2, label="true")
axes[1, 0].axvline(amp_mean, color="black", linestyle="-", lw=1.2, label="posterior mean")
axes[1, 0].set_title("Posterior: amplitude"); axes[1, 0].legend(fontsize=7.5, frameon=False)

axes[1, 1].hist(x0_samples, bins=60, density=True, alpha=0.75, color="tab:orange")
axes[1, 1].axvline(TRUE_X0, color="green", linestyle=":", lw=1.2, label="true")
axes[1, 1].axvline(x0_mean, color="black", linestyle="-", lw=1.2, label="posterior mean")
axes[1, 1].set_title(r"Posterior: source location $x_0$"); axes[1, 1].legend(fontsize=7.5, frameon=False)

# Posterior predictive check: forward-simulate sensor amplitudes for
# posterior draws, compare envelope to actual observations
idx = rng.choice(len(posterior), 200, replace=False)
for i in idx:
    pa, _ = forward_model(posterior[i, 0], posterior[i, 1])
    axes[1, 2].plot(SENSOR_POSITIONS, pa, color="gray", alpha=0.05)
axes[1, 2].plot(SENSOR_POSITIONS, obs_amp, "ro", ms=5, label="observed")
axes[1, 2].plot(SENSOR_POSITIONS, true_peak_amp, "g^", ms=5, label="true (noiseless)")
axes[1, 2].set_xlabel("sensor position"); axes[1, 2].set_ylabel("peak amplitude")
axes[1, 2].set_title("Posterior predictive check")
axes[1, 2].legend(fontsize=7.5, frameon=False)

for ax in axes.ravel():
    cm_x(ax)
    cm_y(ax)

fig.tight_layout()
savefig_all(fig, "mcmc_wave_source_p8")
plt.close(fig)

# =====================================================================
# PART B: Bayesian linear regression on the real diabetes dataset
# =====================================================================
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

print("\n\nPART B: Bayesian linear regression on the diabetes dataset")
data = load_diabetes()
X_raw, y = data.data, data.target
feature_names = data.feature_names
print(f"Dataset: {X_raw.shape[0]} patients, {X_raw.shape[1]} features")
print(f"Target: quantitative measure of disease progression one year after baseline")

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
y_c = (y - y.mean()) / y.std()   # standardize target too, for a clean prior scale

# ---- Classical OLS baseline ----
ols = LinearRegression(fit_intercept=False).fit(X, y_c)
ols_coefs = ols.coef_
n, d = X.shape
resid = y_c - X @ ols_coefs
sigma2_hat = np.sum(resid ** 2) / (n - d)
XtX_inv = np.linalg.inv(X.T @ X)
ols_se = np.sqrt(np.diag(sigma2_hat * XtX_inv))
ols_ci = np.vstack([ols_coefs - 1.96 * ols_se, ols_coefs + 1.96 * ols_se]).T

# ---- Bayesian linear regression via MCMC over the regression coefficients ----
# Prior: independent Normal(0, tau^2) on each standardized coefficient (a
# standard weakly-informative "ridge-like" prior), plus a noise variance.
TAU2 = 4.0
NOISE_STD = 1.0   # fixed for simplicity -- a full model would put a prior on this too


def log_posterior_reg(beta):
    if np.any(np.abs(beta) > 20):
        return -np.inf
    resid = y_c - X @ beta
    log_lik = -0.5 * np.sum(resid ** 2) / NOISE_STD ** 2
    log_prior_beta = -0.5 * np.sum(beta ** 2) / TAU2
    return log_lik + log_prior_beta


def mh_vector(log_post_fn, theta0, n_samples, step_size, rng):
    theta = np.array(theta0, dtype=float)
    current_lp = log_post_fn(theta)
    samples = np.zeros((n_samples, len(theta)))
    n_accept = 0
    for i in range(n_samples):
        proposal = theta + rng.normal(0, step_size, size=len(theta))
        proposal_lp = log_post_fn(proposal)
        if np.log(rng.uniform()) < proposal_lp - current_lp:
            theta, current_lp = proposal, proposal_lp
            n_accept += 1
        samples[i] = theta
    return samples, n_accept / n_samples


N_SAMPLES_REG = 40000
BURN_IN_REG = 8000
beta0 = np.zeros(d)
reg_samples, reg_accept = mh_vector(
    log_posterior_reg, beta0, N_SAMPLES_REG, step_size=0.03, rng=rng
)
print(f"Regression MCMC acceptance rate: {reg_accept:.3f}")

reg_posterior = reg_samples[BURN_IN_REG:]
bayes_mean = reg_posterior.mean(axis=0)
bayes_ci = np.percentile(reg_posterior, [2.5, 97.5], axis=0).T

print("\nFeature            OLS coef [95% CI]              Bayes posterior mean [95% credible interval]")
for j, name in enumerate(feature_names):
    print(f"{name:8s}  {ols_coefs[j]:+.3f} [{ols_ci[j,0]:+.3f}, {ols_ci[j,1]:+.3f}]      "
          f"{bayes_mean[j]:+.3f} [{bayes_ci[j,0]:+.3f}, {bayes_ci[j,1]:+.3f}]")

# ---- Figure: OLS vs Bayesian coefficient comparison ----
# Note: y-axis carries categorical feature names, so only the numeric
# x-axis gets the CM tick formatter.
fig, ax = plt.subplots(figsize=(7.6, 5.2))
y_pos = np.arange(d)
ax.errorbar(ols_coefs, y_pos - 0.15,
            xerr=[ols_coefs - ols_ci[:, 0], ols_ci[:, 1] - ols_coefs],
            fmt="o", ms=5, label="OLS (95% CI)", capsize=3)
ax.errorbar(bayes_mean, y_pos + 0.15,
            xerr=[bayes_mean - bayes_ci[:, 0], bayes_ci[:, 1] - bayes_mean],
            fmt="s", ms=5, label="Bayesian (95% credible interval)", capsize=3)
ax.axvline(0, color="gray", linestyle="--", lw=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_names)
ax.set_xlabel("standardized coefficient")
ax.set_title("OLS vs. Bayesian linear regression: diabetes progression")
ax.legend(frameon=False)
ax.grid(axis="x")
cm_x(ax)
fig.tight_layout()
savefig_all(fig, "bayes_vs_ols_diabetes_p8")
plt.close(fig)

with open("results.txt", "w") as f:
    f.write(f"Part A acceptance rate: {accept_rate:.3f}\n")
    f.write(f"amp_mean={amp_mean:.4f} amp_ci={amp_ci}\n")
    f.write(f"x0_mean={x0_mean:.4f} x0_ci={x0_ci}\n")
    f.write(f"\nPart B acceptance rate: {reg_accept:.3f}\n")
    for j, name in enumerate(feature_names):
        f.write(f"{name}: ols={ols_coefs[j]:.4f} bayes_mean={bayes_mean[j]:.4f}\n")

print("done")