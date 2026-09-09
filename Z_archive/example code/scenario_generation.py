"""
scenario_generation.py
======================

Gaussian-copula scenario generation for the decision-focused dispatchable-feeder
pipeline, following Pinson et al. (2009). Updated for the *frozen-copula* design
settled in the project:

  * The dependence structure Σ is estimated ONCE, before training, from the
    statistically-trained baseline forecaster's errors, shrunk toward the
    identity, and then FROZEN.
  * The underlying normal draws are ALSO frozen (common random numbers), so the
    scenario cloud deforms smoothly as the marginals train rather than being
    resampled each step.
  * Generation is DIFFERENTIABLE through the marginals only: decision-loss
    gradients flow back through the trainable inverse-CDF F⁻¹_Θ into the
    forecaster, while Σ and the draws carry no gradient.

Two parts, by design:

  A. One-time estimation (NumPy, run before DFL)
       build_probit_matrix(...)         PIT + probit of historical errors
       estimate_frozen_correlation(...) pooled + shrinkage -> frozen Σ
       PredictiveMarginal, generate_scenarios_numpy(...)  (non-diff; baseline/eval)

  B. Per-step generation (PyTorch, inside the DFL loop)
       FrozenCopulaSampler              Σ + draws frozen; differentiable in the
                                        forecaster's quantile outputs.

Consumes the monotone quantiles produced by forecaster.py (levels = QUANTILE_LEVELS).
Produces prosumption scenarios p^l and forecast-error scenarios ξ = p^l − p̂^l,
with p̂^l the median forecast (consistent with the dispatchable-feeder literature).

Dependencies: numpy, scipy; torch only for Part B. (Torch path not executed in the
authoring environment; the interpolation logic was checked in NumPy separately.)
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

EPS = 1e-6
QUANTILE_LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)   # matches forecaster.py


# ========================================================================== #
# PART A — one-time estimation (NumPy)
# ========================================================================== #
class PredictiveMarginal:
    """One lead time's predictive distribution from quantile forecasts, as a
    monotone CDF (F̂) / inverse-CDF (F̂⁻¹) by piecewise-linear interpolation.
    Used for PIT during estimation and for the non-differentiable eval path."""

    def __init__(
            self,
            levels,
            values,
            lower=None,
            upper=None
            ):
        levels = np.asarray(levels, float)

        # below not rlly necessary due to softmax + eps in forecaster but doubly ensures monotonicity.
        values = np.asarray(values, float)
        assert np.all(np.diff(values) > 0), "ERROR: marginal quantile values are not strictly monotonic!"

        ## Construct all points on the CDF from the forecaster's marginal distribution.
        
        ## Step 1: Ensure top and bottom of CDF are defined properly.
        # remember: probability is y-axis (levels), value is x-axis (values)

        # need to make sure we define a bottom value (x-axis) for every CDF.
        # if no lower bound defined before:
        # lower = bottom quantile + slope of bottom two quantiles * difference between 0 and lowest quantile prob
        if lower is None:
            slope = (values[1] - values[0]) / (levels[1] - levels[0])
            lower = values[0] + slope * ( 0 - levels[0])
        
        # for upper, same but reverse
        # upper = top quantile + slope of top two quantiles * difference between 1 and top quantile prob
        if upper is None:
            slope = (values[-1] - values[-2]) / (levels[-1] - levels[-2])
            upper = values[-1] + slope * (1.0 - levels[-1])

        l = np.concatenate(([0.0], levels, [1.0]))
        v = np.concatenate(([lower], values, [upper]))

        assert np.all(np.diff(l) > 0), "ERROR: appended 0.0 or 1.0 ruined monotonicity of quantile LEVELS (check if levels already contain 0 or 1)"
        assert np.all(np.diff(v) > 0), "ERROR: extrapolated lower/upper bounds ruined monotonicity of quantile VALUES"

        # define levels and values as instance attributes
        self._l = l
        self._v = v

    def cdf(self, x):
        """
        linearly interpollated CDF. Takes prosumption input, and returns
        corresponding CDF probability in [0,1] space.
        Does this by linearly interpollating CDF based on input discrete quantiles levels and values.
        """
        return np.interp(np.asarray(x, float), self._v, self._l)

    def inv_cdf(self, u):
        """
        Takes probability from [0,1] space, and returns exact prosumption relating to that CDF probability level
        Does this by linearly interpolating the cdf based on input discrete quantiles levels and values.
        """
    
        u = np.asarray(u, float)
    
        assert np.all((u >= 0.0) & (u <= 1.0)), (
        f"Probability input 'u' must be between 0.0 and 1.0 inclusive. "
        f"Got range [{u.min()}, {u.max()}].")

        # originally clipped values, change back if I do actually want to clip them, not generate error.
        #u = np.clip(np.asarray(u, float), 0.0, 1.0)

        return np.interp(u, self._l, self._v)



def build_probit_matrix(realisations, quantiles, levels=QUANTILE_LEVELS):
    """
    PIT + probit of the baseline forecaster's errors, for estimating Σ.

    realisations : (N_days, K) realised prosumption.
    quantiles    : (N_days, K, Q) the BASELINE forecaster's quantiles for each day.
    returns X    : (N_days, K) standard-normal transformed errors.
    """
    realisations = np.asarray(realisations, float)
    quantiles = np.asarray(quantiles, float)
    N, K = realisations.shape
    X = np.empty((N, K))
    for d in range(N):
        for k in range(K):
            m = PredictiveMarginal(levels, quantiles[d, k])
            u = np.clip(m.cdf(realisations[d, k]), EPS, 1 - EPS)
            X[d, k] = norm.ppf(u)
    return X


def estimate_frozen_correlation(X, shrink="ss"):
    """
    Pooled correlation of the probit errors, shrunk toward the identity —
    the one-time, frozen Σ. X : (N_days, K). Clock-aligned windows make every
    row a sample of the same K-dim distribution, so pooling is valid; shrinkage
    handles the one-sample-per-day scarcity.
    shrink : 'ss' Schäfer–Strimmer intensity, a float in [0,1], or None.
    """
    X = np.asarray(X, float)
    N, K = X.shape
    W = X - X.mean(0, keepdims=True)
    sd = W.std(0, ddof=1); sd[sd == 0] = 1.0
    W = W / sd
    R = (W.T @ W) / (N - 1)
    np.fill_diagonal(R, 1.0)
    if shrink is None:
        return _nearest_psd_correlation(R)
    delta = _ss_delta(W, R) if shrink == "ss" else float(shrink)
    R = (1.0 - delta) * R
    np.fill_diagonal(R, 1.0)
    return _nearest_psd_correlation(R)


def generate_scenarios_numpy(marginals, corr, n_scenarios, rng=None):
    """
    Non-differentiable generation (NumPy) for the baseline / evaluation path,
    where the forecaster is fixed. marginals : list of K PredictiveMarginal.
    Returns (n_scenarios, K) prosumption scenarios.
    """
    rng = np.random.default_rng() if rng is None else rng
    K = len(marginals)
    L = _safe_cholesky(np.asarray(corr, float))
    z = L @ rng.standard_normal((K, n_scenarios))
    u = np.clip(norm.cdf(z), EPS, 1 - EPS)
    P = np.empty((n_scenarios, K))
    for k, m in enumerate(marginals):
        P[:, k] = m.ppf(u[k])
    return P


# ---- numpy linear-algebra helpers -------------------------------------------
def _to_correlation(S):
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    R = S / np.outer(d, d); np.fill_diagonal(R, 1.0); return R


def _ss_delta(W, R):
    N, K = W.shape
    WW = W * W
    sum1 = W.T @ W
    m2 = WW.T @ WW
    var_r = (N / (N - 1) ** 3) * (m2 - (sum1 ** 2) / N)
    off = ~np.eye(K, dtype=bool)
    denom = np.sum(R[off] ** 2)
    return float(np.clip(np.sum(var_r[off]) / denom, 0.0, 1.0)) if denom > 0 else 1.0


def _nearest_psd_correlation(R, eps=1e-8):
    R = (R + R.T) / 2.0
    w, V = np.linalg.eigh(R)
    R = (V * np.clip(w, eps, None)) @ V.T
    return _to_correlation(R)


def _safe_cholesky(R):
    try:
        return np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(_nearest_psd_correlation(R))


# ========================================================================== #
# PART B — differentiable generation (PyTorch, inside the DFL loop)
# ========================================================================== #
def _precompute_plan(corr, levels, n_scenarios, seed):
    """Draw z once from N(0, corr), map to uniforms, and precompute the
    piecewise-linear interpolation plan (bracket index + weight). All frozen."""
    corr = np.asarray(corr, float)
    K = corr.shape[0]
    L = _safe_cholesky(corr)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_scenarios, K)) @ L.T          # (S, K), draws frozen
    u = np.clip(norm.cdf(z), EPS, 1 - EPS)
    levels = np.asarray(levels, float)
    Q = len(levels)
    uc = np.clip(u, levels[0], levels[-1])
    idx = np.clip(np.searchsorted(levels, uc, side="right") - 1, 0, Q - 2)   # (S,K)
    w = (uc - levels[idx]) / (levels[idx + 1] - levels[idx])                 # (S,K)
    med_idx = int(np.argmin(np.abs(levels - 0.5)))
    return idx.astype(np.int64), w.astype(np.float32), med_idx


if _HAS_TORCH:

    class FrozenCopulaSampler(nn.Module):
        """
        Differentiable Gaussian-copula sampler with FROZEN Σ and FROZEN draws.

        Precomputes, once: the correlated normal draws → uniforms → the
        (bracket-index, weight) interpolation plan. During training, `prosumption`
        / `errors` gather the CURRENT forecaster quantiles through that frozen plan
        — differentiable in the quantiles (the marginals), constant in Σ/draws.

        Args
        ----
        corr        : (K, K) frozen correlation matrix (from Part A).
        levels      : (Q,) quantile levels the forecaster emits.
        n_scenarios : number of scenarios S.
        seed        : fixes the common random draws.
        """

        def __init__(self, corr, levels=QUANTILE_LEVELS, n_scenarios=500, seed=0):
            super().__init__()
            idx, w, med_idx = _precompute_plan(corr, levels, n_scenarios, seed)
            self.K = int(np.asarray(corr).shape[0])
            self.S = int(n_scenarios)
            self.med_idx = med_idx
            # frozen buffers (move with .to(device), never trained)
            self.register_buffer("idx", torch.as_tensor(idx))          # (S, K) long
            self.register_buffer("w", torch.as_tensor(w))              # (S, K) float

        def prosumption(self, quantiles):
            """
            quantiles : (K, Q) monotone quantile forecasts (torch, requires grad).
            returns   : (S, K) prosumption scenarios p^l, differentiable in quantiles.
            """
            if quantiles.dim() != 2:
                raise ValueError("expects (K, Q) for one day; loop or batch externally")
            K, Q = quantiles.shape
            qexp = quantiles.unsqueeze(0).expand(self.S, K, Q)        # (S, K, Q)
            q_lo = torch.gather(qexp, 2, self.idx.unsqueeze(-1)).squeeze(-1)        # (S,K)
            q_hi = torch.gather(qexp, 2, (self.idx + 1).unsqueeze(-1)).squeeze(-1)  # (S,K)
            return q_lo + self.w * (q_hi - q_lo)

        def errors(self, quantiles, expected=None):
            """
            Forecast-error scenarios ξ = p^l − p̂^l for the dispatch layer.
            expected : (K,) p̂^l; defaults to the median forecast (α≈0.5),
                       which keeps ξ centred and matches the DF-literature convention.
            """
            p = self.prosumption(quantiles)                          # (S, K)
            if expected is None:
                expected = quantiles[:, self.med_idx]                # (K,)
            return p - expected.unsqueeze(0)                         # (S, K)

else:

    class FrozenCopulaSampler:                                       # pragma: no cover
        def __init__(self, *a, **k):
            raise ImportError("PyTorch is required for FrozenCopulaSampler.")


# ========================================================================== #
# smoke test
# ========================================================================== #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, Q, Ndays = 24, len(QUANTILE_LEVELS), 400

    # --- fabricate a baseline forecaster's daily quantiles + realisations ---
    mu = 3 + 1.5 * np.sin(2 * np.pi * (np.arange(K) - 18) / 24)      # diurnal mean
    sd = 0.6 + 0.2 * np.cos(2 * np.pi * np.arange(K) / 24) ** 2
    quantiles = (mu[None, :, None]
                 + sd[None, :, None] * norm.ppf(QUANTILE_LEVELS)[None, None, :]
                 + 0.1 * rng.standard_normal((Ndays, K, 1)))         # (N,K,Q)
    quantiles = np.sort(quantiles, axis=-1)
    # realisations with AR(1) cross-lead-time error correlation
    rho = 0.7
    C = rho ** np.abs(np.subtract.outer(np.arange(K), np.arange(K)))
    Lc = np.linalg.cholesky(C)
    err = (Lc @ rng.standard_normal((K, Ndays))).T                  # (N,K)
    realisations = mu[None, :] + sd[None, :] * err

    # --- Part A: estimate the frozen Σ ---
    X = build_probit_matrix(realisations, quantiles)
    Sigma = estimate_frozen_correlation(X, shrink="ss")
    print("frozen sigma:", Sigma.shape,
          "| PSD:", bool(np.all(np.linalg.eigvalsh(Sigma) > -1e-8)),
          "| mean off-diag:", round(float(Sigma[~np.eye(K, dtype=bool)].mean()), 3))

    # --- Part B: differentiable sampling (only if torch present) ---
    if _HAS_TORCH:
        sampler = FrozenCopulaSampler(Sigma, n_scenarios=500, seed=0)
        q_day = torch.tensor(quantiles[0], dtype=torch.float32, requires_grad=True)  # (K,Q)
        xi = sampler.errors(q_day)                                  # (S, K)
        print("xi scenarios:", tuple(xi.shape), "| requires_grad:", xi.requires_grad)
        loss = (xi ** 2).mean()
        loss.backward()
        g = q_day.grad
        print("grad reaches quantiles:", bool(g is not None and torch.isfinite(g).all()
                                              and g.abs().sum() > 0))
    else:
        print("torch not installed here - Part B (differentiable sampler) not run.")
    print("smoke test OK")
