---
layout: post
title: "Bayesian Inference and Inverse Problems: From First Principles to Clinical Data"
date: 2026-05-30
tags: [bayesian-inference, mcmc, inverse-problems, uncertainty-quantification, statistics]
giscus_comments: true
published: true
---

## Motivation: the other half of uncertainty quantification

The [earlier post in this series on uncertainty quantification]({{ '/blog/2026/uncertainty-quantification-surrogate-modeling_p3/' | relative_url }}) covered the **forward** direction: determining the resulting distribution over a simulation's output, given a known probability distribution over an input. This post treats the mirror-image problem, which is arguably the more common one in practice: given noisy, real-world **observations**, what can be inferred about an unknown parameter that produced them, and, critically, **with what degree of confidence**.

That inversion is the mathematical core of an enormous range of applied problems: inferring a patient's underlying disease risk from noisy clinical measurements, inferring the unknown properties of a material from a small number of experimental tests, or, as this post returns to at the end, inferring the seafloor displacement that triggered a tsunami from a sparse network of ocean pressure sensors, in time to issue a warning before the wave arrives. In each case the same question recurs: given data that constrain the answer only partially and noisily, which values of the unknown parameter are consistent with what was observed, and with what degree of certainty.

## Two ways to answer that question

Classical statistics answers this with a **point estimate** plus a confidence interval. Ordinary least squares, for example, yields a single best-fit coefficient together with a standard error describing how much that estimate would vary were the experiment repeated many times. The claim concerns the long-run behavior of the *procedure*.

**Bayesian inference** addresses a different and arguably more directly useful question: the full **probability distribution** over the unknown parameter, given the data actually observed rather than hypothetical repetitions of the experiment. The answer is a distribution rather than a single number, termed the **posterior distribution**, and it incorporates uncertainty quantification in the most literal sense.

### Bayes' theorem as a mathematical object, not just a formula to memorize

Everything in this post follows from one identity, due originally to {% cite bayes1763essay %}. For an unknown parameter $\theta$ and observed data $D$,

$$
p(\theta \mid D) = \frac{p(D \mid \theta)\, p(\theta)}{p(D)} \;\propto\; \underbrace{p(D \mid \theta)}_{\text{likelihood}} \cdot \underbrace{p(\theta)}_{\text{prior}}.
$$

Each term is doing a specific, interpretable job:

- **$p(\theta)$, the prior**, encodes what was believed about $\theta$ *before* observing this data: physical bounds, results from previous studies, or, as in the examples below, deliberately weak and close to agnostic beliefs where prior assumptions should not dominate.
- **$p(D \mid \theta)$, the likelihood**, is the probability of seeing *this specific data* were $\theta$ the true value. At this point a physical or statistical model of the measurement process enters.
- **$p(\theta \mid D)$, the posterior**, is the answer: the full distribution of plausible $\theta$ values *given* what was actually observed, obtained by weighting the prior by how well each candidate $\theta$ explains the data.
- **$p(D)$, the evidence**, is a normalizing constant (the probability of the data, integrated over all possible $\theta$), and it is precisely the term that makes computing the posterior *directly* intractable for all but the simplest models, since it requires integrating over every possible parameter value. That intractability is the practical obstacle the remainder of this post addresses.

### A bridge from familiar ground: MAP estimation as regularized regression

Before any computational method is introduced, one connection deserves statement. The **maximum a posteriori (MAP)** estimate, the single most probable value of $\theta$ under the posterior, connects directly to a result from the optimization post in this series. Taking the log of Bayes' theorem (dropping the constant $p(D)$, which does not affect where the maximum is),

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \Big[ \log p(D \mid \theta) + \log p(\theta) \Big].
$$

If the likelihood is Gaussian and the prior on $\theta$ is also Gaussian with mean zero, this becomes *exactly* ridge-regularized least squares: the log-likelihood term is (up to constants) the familiar sum of squared residuals, and the log-prior term is exactly an $L_2$ penalty on $\theta$. Concretely, with $D = (X, y)$, $y \mid \theta \sim \mathcal{N}(X\theta, \sigma^2 I)$, and $\theta \sim \mathcal{N}(0, \tau^2 I)$,

$$
\hat{\theta}_{\text{MAP}} = \arg\min_\theta \left[ \frac{1}{2\sigma^2}\lVert y - X\theta \rVert_2^2 + \frac{1}{2\tau^2}\lVert \theta \rVert_2^2 \right]
= \arg\min_\theta \left[ \lVert y - X\theta \rVert_2^2 + \lambda \lVert \theta \rVert_2^2 \right], \quad \lambda = \frac{\sigma^2}{\tau^2},
$$

which is precisely ridge regression with regularization strength $\lambda = \sigma^2/\tau^2$. **Regularization, viewed through a Bayesian lens, is an informative prior**. The equivalence is useful, since it implies that every regularization technique from classical statistics and machine learning admits a direct Bayesian interpretation, and conversely.

## Why the posterior usually cannot be computed directly

For anything beyond a small number of textbook "conjugate prior" cases, the posterior distribution has no closed form. The normalizing constant $p(D)$ requires integrating the likelihood multiplied by the prior over every possible parameter value, which is analytically intractable outside a small number of special cases. The standard practical solution is to **sample** from the posterior instead of computing it in closed form, using **Markov Chain Monte Carlo (MCMC)**.

### The Metropolis-Hastings algorithm, from scratch

The idea, due to Metropolis and colleagues in 1953 {% cite metropolis1953equation %} and generalized by Hastings in 1970 {% cite hastings1970monte %}, is elegant: construct a random walk through parameter space that, if run sufficiently long, visits each region of the posterior with a frequency exactly proportional to its posterior probability, without ever computing the intractable normalizing constant.

The algorithm, at each step, given a current position $\theta$:

1. Propose a new candidate $\theta' = \theta + \epsilon$, where $\epsilon$ is drawn from a simple symmetric distribution (e.g. a Gaussian centered at zero).
2. Compute the acceptance ratio $r = \dfrac{p(D \mid \theta')\, p(\theta')}{p(D \mid \theta)\, p(\theta)}$ The intractable constant $p(D)$ cancels exactly, since it appears in both numerator and denominator. **This cancellation is what makes the algorithm viable**: it never requires the overall normalization of the posterior, only the *relative* posterior density at two points.
3. Accept the proposal with probability $\min(1, r)$; otherwise stay at $\theta$.

```python
def metropolis_hastings(log_post_fn, theta0, n_samples, step_sizes, rng):
    theta = np.array(theta0, dtype=float)
    current_lp = log_post_fn(*theta)
    samples = np.zeros((n_samples, len(theta)))
    n_accept = 0
    for i in range(n_samples):
        proposal = theta + rng.normal(0, step_sizes)
        proposal_lp = log_post_fn(*proposal)
        log_accept_ratio = proposal_lp - current_lp   # working in log-space
                                                        # for numerical stability
        if np.log(rng.uniform()) < log_accept_ratio:
            theta = proposal
            current_lp = proposal_lp
            n_accept += 1
        samples[i] = theta
    return samples, n_accept / n_samples
```

Working entirely in **log-space** (log-posterior, log-acceptance-ratio) rather than raw probabilities is standard practice and important in practice, not just stylistic: posterior densities routinely involve products of many small probabilities, which underflow to exactly zero in floating-point arithmetic if computed directly; sums of log-probabilities do not have this problem.

## Part A: Inferring a wave source from noisy sensor data

A physically motivated example makes this concrete, and connects directly to the [HPC post in this series]({{ '/blog/2026/hpc-domain-decomposition_p7/' | relative_url }}) and its wave-propagation simulation. Consider a simplified inverse problem: a wave pulse of unknown amplitude $A$ originates at an unknown location $x_0$, and a network of sensors at known positions records the peak amplitude and arrival time of the pulse as it passes. The task is to infer $A$ and $x_0$ **from the noisy sensor readings alone**.

The setup is a deliberately simplified version of the inverse problem underlying real tsunami early-warning systems: infer an unknown source from sparse, noisy sensor observations, fast enough to matter.

### The forward model

```python
SENSOR_POSITIONS = np.array([1.0, 2.5, 4.0, 6.0, 8.5])
WAVE_SPEED = 2.0

def forward_model(amplitude, x0):
    dist = np.abs(SENSOR_POSITIONS - x0)
    peak_amp = amplitude / (1.0 + 0.3 * dist)      # geometric spreading/damping
    arrival_time = dist / WAVE_SPEED
    return peak_amp, arrival_time
```

Synthetic "observed" data is generated from a known true source ($A=3.0$, $x_0=0.2$), corrupted with realistic Gaussian sensor noise. Recovering known parameters from synthetic data is the standard means of validating an inverse-problem pipeline: because the ground truth is known by construction, whether the inference procedure recovers it can be checked directly.

### Likelihood and prior

```python
def log_likelihood(amplitude, x0):
    pred_amp, pred_time = forward_model(amplitude, x0)
    ll_amp = -0.5 * np.sum(((obs_amp - pred_amp) / NOISE_STD_AMP) ** 2)
    ll_time = -0.5 * np.sum(((obs_time - pred_time) / NOISE_STD_TIME) ** 2)
    return ll_amp + ll_time

def log_prior(amplitude, x0):
    if not (0.0 < amplitude < 10.0):
        return -np.inf
    if not (-2.0 < x0 < 2.0):
        return -np.inf
    return 0.0   # flat (uniform) prior within physically reasonable bounds
```

The Gaussian log-likelihood here is not an arbitrary modeling choice. It follows directly from assuming independent, normally distributed measurement noise on each sensor reading, which is the standard and usually well-justified assumption for instrument noise, by the Central Limit Theorem, where the noise is the sum of many small independent error sources.

### Running the sampler

```python
samples, accept_rate = metropolis_hastings(
    log_posterior, theta0=[1.0, 0.0], n_samples=60000,
    step_sizes=[0.15, 0.1], rng=rng
)
```

```
MCMC acceptance rate: 0.446
Posterior amplitude: mean=2.948, 95% CI=[2.692, 3.213]  (true=3.0)
Posterior x0:        mean=0.224, 95% CI=[0.085, 0.362]  (true=0.2)
```

Both true parameter values fall comfortably within their respective 95% credible intervals, and the posterior means are close to the ground truth. A correctly implemented inference pipeline should produce exactly this, and it constitutes a validation of the method rather than a demonstration alone.

<p align="center">
  <img src="/assets/img/posts/mcmc_wave_source_p8.svg" alt="MCMC trace plots, posterior histograms, joint posterior, and posterior predictive check for the wave source inference problem" style="width: 100%; max-width: 1000px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 1: Top row, trace plots for both parameters, showing the position of the chain at every iteration with the true value overlaid, together with the joint posterior sample cloud. Bottom row, marginal posterior histograms and a posterior predictive check: sensor readings forward-simulated from 200 random posterior draws (gray) against the actual noisy observations (red) and the true noiseless signal (green). The gray envelope contains the observations, which is the relevant qualitative check; failure to do so would indicate that the model itself, rather than only the inferred parameters, is misspecified.*

### Reading the diagnostics like a practitioner, not just a code-runner

A few things worth understanding, since they are the actual daily practice of doing this correctly rather than superstition:

- **Acceptance rate $\approx 0.446$** sits in a reasonable range. If the proposal step size is too large, almost every proposal falls in a low-probability region and is rejected, so the acceptance rate collapses toward zero and the chain barely moves. If it is too small, *almost every* proposal is accepted, but each step covers so little ground that the chain explores the posterior very slowly. A commonly cited heuristic for random-walk Metropolis targets roughly 20–50% acceptance {% cite roberts1997weak %}, although the appropriate value depends on the dimensionality and shape of the posterior. The figure is a diagnostic to be checked rather than a target to be optimized.
- **Burn-in** (discarding the first 10,000 of 60,000 samples above) exists because the chain starts at an arbitrary initial point rather than at a sample from the posterior, so the early samples still reflect that starting point. The trace plots in Figure 1 make this directly visible at the point where the chain settles into stable, noisy variation about a fixed level rather than trending toward it.
- **The joint posterior scatter** (top-right panel) shows genuine correlation structure between $A$ and $x_0$, information that a point estimate alone would discard entirely but that carries physical significance: it identifies which *combinations* of source parameters are difficult to distinguish given the geometry of this sensor network, which is directly actionable in designing an improved sensor layout.

## Part B: Bayesian linear regression on real patient data

To ground this in a genuinely real-world dataset rather than only a synthetic inverse problem, this section turns to the **scikit-learn diabetes dataset**: 442 real patients, 10 standardized clinical measurements (age, sex, BMI, blood pressure, six blood serum measurements), and a target variable measuring disease progression one year after baseline.

### Classical OLS baseline

```python
ols = LinearRegression(fit_intercept=False).fit(X, y_c)
resid = y_c - X @ ols.coef_
sigma2_hat = np.sum(resid ** 2) / (n - d)
ols_se = np.sqrt(np.diag(sigma2_hat * np.linalg.inv(X.T @ X)))
```

### The Bayesian version

An independent, zero-mean Gaussian prior is placed on each (standardized) regression coefficient, a direct implementation of the equivalence between MAP estimation and ridge regression derived above. The procedure again uses Metropolis-Hastings, sampling here in the full 10-dimensional coefficient space at once:

```python
TAU2 = 4.0       # prior variance on each coefficient
NOISE_STD = 1.0

def log_posterior_reg(beta):
    if np.any(np.abs(beta) > 20):
        return -np.inf
    resid = y_c - X @ beta
    log_lik = -0.5 * np.sum(resid ** 2) / NOISE_STD ** 2
    log_prior_beta = -0.5 * np.sum(beta ** 2) / TAU2
    return log_lik + log_prior_beta
```

```
Regression MCMC acceptance rate: 0.362

Feature       OLS coef [95% CI]              Bayes posterior mean [95% credible interval]
age       -0.006 [-0.078, +0.066]      -0.008 [-0.114, +0.099]
sex       -0.148 [-0.222, -0.074]      -0.147 [-0.257, -0.037]
bmi       +0.321 [+0.241, +0.402]      +0.320 [+0.203, +0.436]
bp        +0.200 [+0.121, +0.279]      +0.205 [+0.087, +0.325]
s1        -0.489 [-0.993, +0.015]      -0.487 [-0.918, +0.105]
s2        +0.294 [-0.116, +0.704]      +0.294 [-0.179, +0.658]
s3        +0.062 [-0.195, +0.319]      +0.060 [-0.271, +0.342]
s4        +0.109 [-0.086, +0.305]      +0.107 [-0.161, +0.411]
s5        +0.464 [+0.256, +0.672]      +0.461 [+0.213, +0.673]
s6        +0.042 [-0.038, +0.122]      +0.043 [-0.068, +0.161]
```

<p align="center">
  <img src="/assets/img/posts/bayes_vs_ols_diabetes_p8.svg" alt="Comparison of OLS confidence intervals and Bayesian credible intervals for diabetes regression coefficients" style="width: 100%; max-width: 800px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 2: OLS point estimates with 95% confidence intervals (blue) versus Bayesian posterior means with 95% credible intervals (orange), for all 10 standardized clinical features. `bmi`, `bp` (blood pressure), and `s5` have intervals that exclude zero under both approaches, and are the features most robustly associated with disease progression in this cohort.*

### Why the estimates agree closely here, and when they would not

The OLS and Bayesian estimates land nearly on top of each other in this example, and that is not a coincidence worth glossing over: with a weak (large-variance, $\tau^2=4$) Gaussian prior and a reasonably large sample size ($n=442$), the likelihood dominates the posterior, and the Bayesian credible interval converges toward the classical confidence interval. This is a general asymptotic result, a Bayesian analogue of the Bernstein-von Mises theorem, rather than a feature specific to this dataset {% cite efron2016computer %}.

The Bayesian approach is advantageous in the regimes where this does not hold: **small sample sizes** (where the prior genuinely constrains an otherwise poorly determined estimate), **informative priors** built from real external knowledge (a previous clinical study's results feeding directly into this one, rather than starting from scratch), and whenever a **full posterior over a downstream, nonlinear quantity** is required. A Bayesian analysis yields the exact distribution of any function of the samples at negligible additional cost, by applying the function to every posterior draw, whereas a classical approach would require a separate and often approximate argument, such as the delta method, to propagate uncertainty through a nonlinear transformation. A classical, non-Bayesian alternative that provides a useful point of contrast is the bootstrap, which constructs confidence intervals by resampling the data rather than by specifying a prior {% cite efron1979bootstrap %}.

## Connecting back: a recent breakthrough in real-time Bayesian inversion

The [HPC post earlier in this series]({{ '/blog/2026/hpc-domain-decomposition_p7/' | relative_url }}) described the 2025 Gordon Bell Prize-winning tsunami early-warning system built at LLNL, running on the El Capitan exascale supercomputer {% cite llnl2025gordonbell %}. The toy wave-source inference problem in Part A above is a deliberately simplified version of exactly the computational core of that system: given real-time seafloor pressure sensor readings, rapidly infer the unknown source displacement that best explains them, and use that inferred source to forecast the resulting wave. The operational system follows precisely the forward-model-plus-Bayesian-inversion structure developed step by step in this post, at a substantially larger and more time-critical scale, where rapid inference means seconds rather than the hours a full re-simulation would require.

The substantive engineering problem that the operational system solves, and that this simplified example avoids, is the following: **Metropolis-Hastings, as implemented above, is far too slow for a real-time, life-safety application.** Sixty thousand samples of a two-parameter posterior require a fraction of a second on a laptop, whereas a full-physics tsunami source inversion at the resolution needed for an actionable warning cannot be run tens of thousands of times sequentially within the seconds available before a wave arrives. The operational system therefore precomputes a large library of full-physics forward simulations *offline*, using the exascale throughput of El Capitan and the domain-decomposed PDE solvers of the previous post in this series, and then performs the real-time inference step as a fast lookup and interpolation against that precomputed library rather than as a live MCMC run. The Bayesian inverse-problem mathematics is the same as that developed here, but the expensive forward-model evaluation is moved entirely offline so that the online inference step runs at the speed an emergency response requires {% cite tarantola2005inverse %}.

## Summary Notes and Highlights

- **Bayesian inference reframes the question**, replacing a single answer with the full distribution of plausible answers given what was actually observed. That question differs genuinely from, and is often more useful than, a classical point estimate accompanied by a confidence interval.
- **MAP estimation is regularized regression in another form**: an explicit prior is mathematically equivalent to an explicit regularization penalty, which provides a useful bridge for readers approaching the subject from optimization rather than statistics.
- **Metropolis-Hastings sidesteps the intractable normalizing constant** by working entirely with *ratios* of posterior densities, which is the single mathematical trick that makes MCMC computationally feasible at all.
- **Acceptance rate, trace plots, and burn-in are diagnostics to be checked rather than boilerplate.** A poorly tuned sampler can silently produce a plausible-looking but incorrect posterior, and these are the tools for detecting that outcome.
- **Posterior predictive checks**, which simulate data from posterior draws and compare it against what was actually observed, are among the most underused and most valuable checks in applied Bayesian work. 
- **Bayesian and classical estimates converge given sufficient data and a weak prior.** The advantage of the Bayesian approach appears in small samples, with genuinely informative priors, and when propagating uncertainty through nonlinear downstream quantities.

## References

The foundational identity underlying this entire post is due to {% cite bayes1763essay %}. The Metropolis-Hastings algorithm follows {% cite metropolis1953equation %} and its generalization by {% cite hastings1970monte %}, with practical acceptance-rate targets justified theoretically in {% cite roberts1997weak %}. {% cite gelman2013bayesian %} is the standard modern reference for Bayesian modeling, MCMC diagnostics, and posterior predictive checks, and {% cite tarantola2005inverse %} is the classical reference connecting Bayesian inference to physical inverse problems specifically. {% cite efron2016computer %} gives an especially clear, modern treatment of the Bayesian/frequentist correspondence discussed above, and {% cite efron1979bootstrap %} introduces the bootstrap as a classical non-Bayesian point of contrast. The Gordon Bell Prize-winning tsunami-forecasting system referenced throughout is documented in {% cite llnl2025gordonbell %}.

{% bibliography --cited --file blog_references %}

---

*Full, executed code for both the wave-source inversion and the diabetes regression comparison is available in [`bayesian_inference_p8.py`]({{ '/assets/code/bayesian_inference_p8.py' | relative_url }}).*