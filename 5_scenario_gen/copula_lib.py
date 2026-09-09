"""
copula_lib.py
=============
Core functions and classes for frozen Gaussian-copula scenario generation and
validation, extracted from the `copula_testing_neat_final` notebook so they can
be imported and reused (e.g. inside a DFL training loop).

Contents
--------
Marginals / estimation
    MarginalDistCDF               piecewise-linear CDF / inverse-CDF per lead
    build_probit_matrix           PIT + probit of forecaster errors -> probit scores
    seasonal_correlation_matrices seasonal (24x24) correlation matrices (diagnostic)
    plot_seasonal_correlations    heatmaps for the seasonal matrices

Frozen draws / sampler
    build_Z_corr                  Sobol -> correlated standard-normal frozen draws
    FrozenCopulaSampler           differentiable Gaussian-copula sampler (frozen S, draws)
    prosumption_batched           batched-over-days differentiable generation

Validation
    scenario_calibration_check    multivariate / ensemble calibration of scenarios
    quantile_reliability_check    seasonal reliability of the quantile forecasts

Module constants
    EPS   numerical guard away from the 0/1 probability boundary
    SEED  reproducibility seed (matches the notebook)

`QUANTILE_LEVELS` is imported from the `forecasting` module when available so
the default arguments resolve; callers can always pass `levels=` explicitly.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# `forecasting` must be importable (add its directory to sys.path before import).
# The notebook does this in its setup section; here we import defensively so that
# merely importing this module never hard-fails. When it is None, pass levels= explicitly.
try:
    from forecasting import QUANTILE_LEVELS
except Exception:  # pragma: no cover - depends on caller's sys.path
    QUANTILE_LEVELS = None

# Reproducibility seed (kept identical to the notebook).
SEED = 20240801

# ----------------------------------------------------------------------------- #
# Marginals and copula-correlation estimation
# ----------------------------------------------------------------------------- #
EPS = 1e-6

# NB: QUANTILE_LEVELS comes from the checkpoint (Section 2) — single source of truth,
# so build_probit_matrix's default matches exactly what the forecaster was trained on.

class MarginalDistCDF:
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

        assert np.all(np.diff(l) > 0), "ERROR: appended 0.0 or 1.0 violates monotonicity of quantile LEVELS (check if levels already contain 0 or 1)"
        assert np.all(np.diff(v) > 0), "ERROR: extrapolated lower/upper bounds violated monotonicity of quantile VALUES"

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


def build_probit_matrix(realisations, quantiles, levels=QUANTILE_LEVELS, eps = EPS):
    """
    Prob. integral transform + probit of the baseline forecaster's errors, for estimating copula covariance.

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
            # for each time-step:
            #create class that stores
            marginal_dist = MarginalDistCDF(levels, quantiles[d, k])

            # Ensure u stays at least 1e-6 away from 0 and 1
            # so norm.ppf (normal dist inv-CDF) yields finite values instead of +/-inf
            u = np.clip(marginal_dist.cdf(realisations[d, k]), eps, 1 - eps)
            X[d, k] = norm.ppf(u) # builds normalised error matrix, one value at a time.
    return X

def seasonal_correlation_matrices(X, dates=None, year=2018):
    """
    X: (N_days, 24) probit/PIT scores — daily samples, hourly variables.
    dates: optional DatetimeIndex of length N_days giving each row's date.
           If None, rows are assumed to be consecutive calendar days from Jan 1.
    Returns dict {season_name: (24,24) correlation matrix}.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if dates is None:
        # ASSUMPTION: row i = Jan 1 + i days. Only valid if no days were dropped.
        dates = pd.date_range(f"{year}-01-01", periods=n, freq="D")
    else:
        dates = pd.DatetimeIndex(dates)

    # meteorological seasons by month
    month_to_season = {12: "Winter", 1: "Winter", 2: "Winter",
                       3: "Spring", 4: "Spring", 5: "Spring",
                       6: "Summer", 7: "Summer", 8: "Summer",
                       9: "Autumn", 10: "Autumn", 11: "Autumn"}
    season = dates.month.map(month_to_season)

    order = ["Winter", "Spring", "Summer", "Autumn"]
    mats = {}
    for s in order:
        rows = X[season == s]
        # np.corrcoef with rowvar=False -> (24,24), variables = columns (hours)
        mats[s] = np.corrcoef(rows, rowvar=False)
        print(f"{s}: {rows.shape[0]} days")
    return mats


def plot_seasonal_correlations(mats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, (season, C) in zip(axes.flat, mats.items()):
        sns.heatmap(C, ax=ax, vmin=-1, vmax=1, cmap="RdBu_r",
                    square=True, cbar_kws={"shrink": 0.8},
                    xticklabels=range(24), yticklabels=range(24))
        ax.set_title(f"{season}  (copula correlation of probit scores)")
        ax.set_xlabel("hour of horizon")
        ax.set_ylabel("hour of horizon")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------- #
# Frozen correlated draws and the differentiable sampler
# ----------------------------------------------------------------------------- #
def build_Z_corr(corr_matrix, K = 24,seed = SEED,n = 64, eps = EPS):
    """
    Uses Sobol sequences to build an evenly distributed, representative Z_corr that can be frozen.
    """
    
    # 1. Initiate Sobol sequence sampler
    sampler = qmc.Sobol(d=K, scramble=True, seed=seed)

    # 2. Draw evenly spread points in [0, 1]^K
    u_samples = sampler.random(n=n)  # Shape: (n, K)

    # 3. Prevent exact 0 and 1 boundaries for norm.ppf safety
    u_samples = np.clip(u_samples, eps, 1 - eps)

    # 4. Transform uniform QMC samples to Standard Normal N(0, I) using inverse standard normal CDF
    Z = norm.ppf(u_samples)  # Shape: (n, K)

    # 5. Apply Cholesky factor of the Target Covariance Matrix
    L_corr = np.linalg.cholesky(corr_matrix)
    Z_corr = Z @ L_corr.T  # Shape: (n, K)
    return Z_corr


def _precompute_plan(Z_corr, levels, eps = EPS):
    """
    This function pre-maps where the Z_corr variables fall within the array of probability levels of the CDF quantiles.
    As Z_corr is frozen, this only has to be computed once. Doing it now speeds up computation later on using pytorch,
    as pytorch doesn't have to search which quantiles a given point sits between each time it checks.
    """

    levels = np.asarray(levels, float)
    # Add 0.0 and 1.0 probability values to the grid. This allows pytorch to extrapolate tail values later.
    l_aug = np.concatenate(([0.0], levels, [1.0]))                    # (num_quantiles + 2,)

    assert np.all(np.diff(l_aug) > 0), "Probability levels must be strictly monotonic"

    # transform Z_corr to uniform space by passing it through normal CDF, 
    # so that we can find each Z_corr value in (S,K)'s respective index
    # within the respective marginal distribution's CDF.
    u = np.clip(norm.cdf(Z_corr), eps, 1.0 - eps)                     # clip by EPS for numerical stability around boundary values
    # u represents all scenario timesteps values in uniform space.
    num_pts = l_aug.shape[0]                                          # num_pts = num_quantiles + 2

    # finds array location for each uniformly transformed scenario timestep in marginal dist's quantiles in uniform space.
    idx = np.clip(np.searchsorted(l_aug, u, side="right") - 1, 0, num_pts - 2)  # finds index of (S,K) in [0,num_quantiles]
    
    # find how far between two surrounding quantile values each scenario timestep is. 
    # w is the fraction showing how much distance u has between the quantiles below and above it
    # in uniform probability space.
    # (distance from bottom) / (distance between top and bottom)
    w = (u - l_aug[idx]) / (l_aug[idx + 1] - l_aug[idx])              # (S,K)

    ## Also need median quantiles location in index for converting scenarios to errors for the optimiser.
    # if we don't know for certain that we have a quantile at 50% probability on CDF,
    # we can find the closest match by subtracting 0.5 from all quantile values,
    # taking the absolute value (so that each value now represents the distance of the 
    # quantil from the middle) and finding the smallest value. 
    med_idx = int(np.argmin(np.abs(l_aug - 0.5)))
    med_idx_q = int(np.argmin(np.abs(levels - 0.5)))   # index into UN-augmented quantiles
    return idx.astype(np.int64), w.astype(np.float32), med_idx, med_idx_q

class FrozenCopulaSampler(nn.Module):
    """
    Differentiable Gaussian-copula sampler with FROZEN Σ and FROZEN draws.

    Precomputes ONCE, from the frozen draws and the quantile LEVELS only:
        correlated normal draws -> uniforms -> (bracket-index, weight) plan
    against the augmented level grid [0, levels..., 1].
    The plan is independent of the forecast VALUES, so it stays valid for
    every DFL step. During training, `prosumption` / `errors` interpolate the
    CURRENT forecaster quantiles through that frozen plan — differentiable in
    the quantiles (marginals), constant in Sigma/draws.

    Args
    ----
    Z_corr : (S, K) frozen pre-drawn correlated standard-normal samples.
    levels : (num_quantiles,)   quantile levels the forecaster emits (must NOT include 0 or 1).
    """

    def __init__(self, Z_corr, levels=QUANTILE_LEVELS):
        super().__init__()
        levels = np.asarray(levels, dtype=float)

        self.num_quantiles = len(levels)
        self.S, self.K = Z_corr.shape

        # calculations all done with tail-extrapolated CDF.
        idx, w, med_idx, med_idx_q = _precompute_plan(Z_corr, levels)   # pre-computes levels to speed up scenario gen
        self.med_idx = med_idx
        self.med_idx_q = med_idx_q

        # idx indexes the augmented (num_quantiles+2) value grid, so it must live in [0, num_quantiles]
        assert idx.min() >= 0 and idx.max() <= self.num_quantiles, (
            f"idx range [{idx.min()},{idx.max()}] inconsistent with augmented grid "
            f"(expected [0,{self.num_quantiles}]); is _precompute_plan using the [0,levels,1] grid?"
        )

        # frozen buffers (move with .to(device), never trained)
        self.register_buffer("idx", torch.as_tensor(idx, dtype=torch.long))     # (S, K)
        self.register_buffer("w", torch.as_tensor(w, dtype=torch.float32))      # (S, K)
        self.register_buffer("levels_t", torch.as_tensor(levels, dtype=torch.float32))  # (num_quantiles,)

    # ------------------------------------------------------------------ #
    def _augmented_values(self, quantiles):
        """
        quantiles : (K, num_quantiles) monotone forecast quantiles (requires grad).
        returns   : (K, num_quantiles+2) grid [lower, quantiles..., upper] with slope-
                    extrapolated tail endpoints, differentiable in quantiles.
        """
        if quantiles.dim() != 2:
            raise ValueError("expects (K, num_quantiles) for one day; loop or batch externally")
        K, num_quantiles = quantiles.shape
        if num_quantiles != self.num_quantiles:
            raise ValueError(f"quantiles has num_quantiles={num_quantiles}, sampler built for num_quantiles={self.num_quantiles}")

        ## ensure that all calculations are done on same device and at same precision
        lev_t = self.levels_t.to(quantiles)                                      # (num_quantiles,)
    
        ## uses same logic as in piecewise linear CDF class
        # now, instead of finding the slope between each quantile per individual timestep, 
        # we want to do it for a matrix representing an entire window of K steps.
        # lower = q0 + (q1-q0)/(l1-l0) * (0 - l0)
        slope_lo = (quantiles[:, 1] - quantiles[:, 0]) / (lev_t[1] - lev_t[0])      # (K,)
        lower = quantiles[:, 0] + slope_lo * (0.0 - lev_t[0])                    # (K,)
        # upper = q_{-1} + (q_{-1}-q_{-2})/(l_{-1}-l_{-2}) * (1 - l_{-1})
        slope_hi = (quantiles[:, -1] - quantiles[:, -2]) / (lev_t[-1] - lev_t[-2])  # (K,)
        upper = quantiles[:, -1] + slope_hi * (1.0 - lev_t[-1])                  # (K,)

        return torch.cat([lower.unsqueeze(1), quantiles, upper.unsqueeze(1)], dim=1)  # (K, num_quantiles+2)


    def _interpolate(self, v_aug):
        """Gather the frozen bracket plan through the augmented values -> (S, K)."""
        K = v_aug.shape[0]
        vexp = v_aug.unsqueeze(0).expand(self.S, K, self.num_quantiles + 2)             
        # unsqueeze takes the matrix of augment values for each quantile, 
        # and adds new dimension at front of tensor to represent scenarios.
        # expand expands this dimension so that the tensor is duplicated
        # num_scenario times across the scenario dimension.
        # now, vexp represents the quantile values for each timestep, duplicated 
        # for every scenario.
        # It is now ready to interpolate the values for each timestep for each scenario
        # based on the precalculated idx values, weights and augment values (quantile values + 0 and 1 prob values)
        # final dimensions of vexp are (S, K, num_quantiles+2)

        # for each scenario's timestep, we want to find the values of the quantiles above and below
        # the scenario's value using the precomputed idx values.
        # Since we have precomputed the distance of each value between the quantiles above and below
        # it in uniform probability space already, we know that the realised value of that scenario's
        # timestep is the value of the below quantile, plus the distance of the point between the below and above
        # quantiles in probability space, multiplied by the distance between these quantiles in value space.  
        
        v_lo = torch.gather(vexp, 2, self.idx.unsqueeze(-1)).squeeze(-1)      # (S, K)
        v_hi = torch.gather(vexp, 2, (self.idx + 1).unsqueeze(-1)).squeeze(-1)  # (S, K)
        # To find the below and above quantile values:
        # for each value:
        # - unsqueeze the precomputed idx values so they have same number of dimensions as vexp
        # - use torch.gather to look along the third dimension of vexp (the quantile value dimension)
        # - torch.gather outputs a tensor that matches the shape of the index tensor provided.
        #   Therefore, the output of gather is (S, K, 1), as it only returns the index of a SINGLE quantile.
        # - Since we don't want that final dimension and there is only one of it, we can use 
        #   squeeze to remove it.
        # - this results in an (S,K) shaped tensor for each of v_lo and v_hi,
        #   which represents the above and below quantiles for each timestep of each scenario.
        #
        # - From here, all we have to do is return the final tensor of the interpolated final value of each
        #   scenario's timesteps, based on the precomputed values!
        return v_lo + self.w * (v_hi - v_lo)

    # ------------------------------------------------------------------ #
    def prosumption(self, quantiles):
        """
        quantiles : (K, num_quantiles) monotone quantile forecasts (torch, requires grad).
        returns   : (S, K) prosumption scenarios p^l, differentiable in quantiles.

        Tails are slope-extrapolated to prob 0/1 (no clamping), matching
        MarginalDistCDF, so scenarios can exceed the outer quantiles.
        """
        v_aug = self._augmented_values(quantiles)
        return self._interpolate(v_aug)
    
    def prosumption_mean(self, quantiles):
        """Mean prosumption forecast: average of all generated scenarios. (K,)"""
        p = self.prosumption(quantiles)              # (S, K)
        return p.mean(dim=0)                          # (K,)

    def errors_mean(self, quantiles):
        """Scenario deviations from the MEAN forecast: scenario - mean.
        Has EXACTLY zero sample mean by construction."""
        p = self.prosumption(quantiles)          # (S, K)
        mean = p.mean(dim=0, keepdim=True)        # (1, K)
        return p - mean                           # (S, K), zero column-mean

    def errors_median(self, quantiles):
        """Scenario deviations from the forecast median: scenario - q_median.
        quantiles : (K, Q) ; returns (S, K), differentiable in quantiles."""
        p = self.prosumption(quantiles)                       # (S, K)
        med = quantiles[:, self.med_idx_q]                    # (K,) un-augmented median
        return p - med.unsqueeze(0)

    def mean_and_errors(self, quantiles):
        """Mean forecast (K,) and mean-centred error scenarios (S, K) from ONE
        prosumption pass. p̂E and ξ share the same scenario mean by construction,
        so E[ξ]=0 exactly and the anchor/error are guaranteed consistent."""
        p = self.prosumption(quantiles)              # (S, K)
        mean = p.mean(dim=0)                          # (K,)
        return mean, p - mean.unsqueeze(0)            # (K,), (S, K)


def prosumption_batched(sampler, quantiles_batch):
    """
    Differentiable scenario generation for a batch of days.

    sampler         : a built FrozenCopulaSampler
    quantiles_batch : (D, K, Q) monotone quantiles, requires grad
    returns         : (D, S, K) scenarios, differentiable in quantiles_batch

    Mirrors sampler._augmented_values + _interpolate but with a leading day axis.
    """
    if quantiles_batch.dim() != 3:
        raise ValueError("expects (D, K, Q)")
    D, K, Q = quantiles_batch.shape
    if Q != sampler.num_quantiles:
        raise ValueError(f"Q={Q} but sampler built for {sampler.num_quantiles}")
    if K != sampler.K:
        raise ValueError(f"K={K} but sampler built for {sampler.K}")

    lev = sampler.levels_t.to(quantiles_batch)                              # (Q,)

    # --- slope-extrapolated tail endpoints, per day & lead (differentiable) ---
    slope_lo = (quantiles_batch[:, :, 1] - quantiles_batch[:, :, 0]) / (lev[1] - lev[0])   # (D,K)
    lower = quantiles_batch[:, :, 0] + slope_lo * (0.0 - lev[0])                            # (D,K)
    slope_hi = (quantiles_batch[:, :, -1] - quantiles_batch[:, :, -2]) / (lev[-1] - lev[-2])  # (D,K)
    upper = quantiles_batch[:, :, -1] + slope_hi * (1.0 - lev[-1])                          # (D,K)

    # augmented value grid: (D, K, Q+2)
    v_aug = torch.cat([lower.unsqueeze(-1), quantiles_batch, upper.unsqueeze(-1)], dim=-1)

    # frozen plan is (S, K); expand to (D, S, K) over the batch and gather along the value axis
    idx = sampler.idx.unsqueeze(0).expand(D, sampler.S, K)                  # (D,S,K)
    w   = sampler.w.unsqueeze(0).expand(D, sampler.S, K)                    # (D,S,K)

    # v_aug -> (D, S, K, Q+2) so each scenario indexes its day's values
    vexp = v_aug.unsqueeze(1).expand(D, sampler.S, K, sampler.num_quantiles + 2)
    v_lo = torch.gather(vexp, 3, idx.unsqueeze(-1)).squeeze(-1)             # (D,S,K)
    v_hi = torch.gather(vexp, 3, (idx + 1).unsqueeze(-1)).squeeze(-1)       # (D,S,K)
    return v_lo + w * (v_hi - v_lo)                                         # (D,S,K)


# ----------------------------------------------------------------------------- #
# Validation
# ----------------------------------------------------------------------------- #
def scenario_calibration_check(
    realised, scenarios, dates=None, year=2018,
    scenario_axis=1, midday_hours=range(9, 17),
    n_display_bins=20, seed=SEED,
):
    """
    Multivariate / ensemble calibration of copula scenarios.

    realised   : (N_days, K)          realised trajectories (K=24)
    scenarios  : (N_days, S, K)        S scenarios per day  (set scenario_axis if different)
    dates      : optional DatetimeIndex length N_days; else assumes consecutive days from Jan 1
    midday_hours: hours treated as the solar window for the summary

    Produces:
      - mean-PIT heatmap  (season x hour): 0.5 = well-centred;
            < 0.5 => ensemble sits too HIGH (realised low in ensemble);
            > 0.5 => ensemble sits too LOW  (realised high in ensemble)
      - per-season multivariate (average-rank) histograms:
            flat = calibrated; U = under-dispersed; dome = over-dispersed; slope = biased
      - printed midday summary per season
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(realised, dtype=float)                    # (N, K)
    S_arr = np.asarray(scenarios, dtype=float)
    if scenario_axis != 1:
        S_arr = np.moveaxis(S_arr, scenario_axis, 1)         # -> (N, S, K)
    N, S, K = S_arr.shape
    assert y.shape == (N, K), f"realised {y.shape} vs scenarios {(N, S, K)}"

    # ---- season labels ----
    if dates is None:
        dates = pd.date_range(f"{year}-01-01", periods=N, freq="D")
    dates = pd.DatetimeIndex(dates)
    m2s = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
           6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}
    season = dates.month.map(m2s).to_numpy()
    order = ["Winter", "Spring", "Summer", "Autumn"]

    # ---- per-hour randomized rank of realised within the ensemble ----
    # rank in {0..S}; PIT = rank / S  (mean ~0.5 if centred)
    ranks_hour = np.empty((N, K), dtype=int)
    for h in range(K):
        col = S_arr[:, :, h]                                 # (N, S)
        below = (col <  y[:, [h]]).sum(axis=1)               # (N,)
        equal = (col == y[:, [h]]).sum(axis=1)               # (N,) ~0 for continuous
        ranks_hour[:, h] = below + rng.integers(0, equal + 1)
    pit_hour = ranks_hour / S                                # (N, K) in [0,1]

    # ---- multivariate average-rank of the whole 24h trajectory ----
    # robust in high-d (avoids the componentwise pre-rank degeneracy at K=24)


    mv_rank = np.empty(N, dtype=int)
    for d in range(N):
        pool = np.vstack([y[d][None, :], S_arr[d]])          # (S+1, K), row 0 = realised
        # ordinal rank within each column (ties negligible for continuous scenarios)
        uni = np.argsort(np.argsort(pool, axis=0, kind="mergesort"),
                         axis=0, kind="mergesort")           # (S+1, K), 0..S
        avg = uni.mean(axis=1)                               # (S+1,)
        obs = avg[0]
        n_less = int((avg < obs).sum())
        n_eq   = int((avg == obs).sum())
        mv_rank[d] = n_less + rng.integers(1, n_eq + 1)      # 1..S+1

    # ================= plots =================
    # (1) mean-PIT heatmap: season x hour
    heat = np.vstack([pit_hour[season == s].mean(axis=0) for s in order])   # (4, K)
    fig1, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=0.3, vmax=0.7)
    ax.set_yticks(range(4)); ax.set_yticklabels(order)
    ax.set_xticks(range(K)); ax.set_xticklabels(range(K))
    ax.set_xlabel("hour of horizon"); ax.set_title(
        "Mean ensemble-PIT of realised  (0.5 = centred;  <0.5 ensemble too high;  >0.5 ensemble too low)")
    for i in range(4):
        for h in range(K):
            ax.text(h, i, f"{heat[i,h]:.2f}", ha="center", va="center", fontsize=6)
    fig1.colorbar(im, shrink=0.8); plt.tight_layout()

    # (2) per-season multivariate rank histograms
    fig2, axes = plt.subplots(2, 2, figsize=(13, 8))
    bins = np.linspace(0.5, S + 1.5, min(n_display_bins, S + 1) + 1)
    for axh, s in zip(axes.flat, order):
        r = mv_rank[season == s]
        axh.hist(r, bins=bins, density=True, color="C0", edgecolor="white")
        axh.axhline(1.0 / (S + 1), color="k", ls="--", lw=1, label="uniform (calibrated)")
        axh.set_title(f"{s}: multivariate rank histogram  (n={ (season==s).sum() })")
        axh.set_xlabel("rank of realised trajectory within ensemble"); axh.legend(fontsize=8)
    plt.tight_layout()

    # ================= printed summary =================
    md = list(midday_hours)
    print(f"{'season':8s} {'midday meanPIT':>15s} {'midday outside-rate':>20s} "
          f"{'expected outside':>17s} {'MV reliability':>15s}")
    exp_out = 2.0 / (S + 1)
    for s in order:
        mask = season == s
        md_pit = pit_hour[mask][:, md].mean()
        # outside = realised below all or above all scenarios at a midday hour
        r_md = ranks_hour[mask][:, md]
        outside = ((r_md == 0) | (r_md == S)).mean()
        # multivariate reliability index: L1 distance of rank hist from uniform (0=perfect)
        counts, _ = np.histogram(mv_rank[mask], bins=bins, density=True)
        widths = np.diff(bins)
        ri = np.sum(np.abs(counts * widths - widths / (bins[-1] - bins[0])))
        print(f"{s:8s} {md_pit:15.3f} {outside:20.3f} {exp_out:17.3f} {ri:15.3f}")

    plt.show()
    return {"pit_hour": pit_hour, "mv_rank": mv_rank, "season": season}


def quantile_reliability_check(
    realised,
    quantiles,
    dates=None,
    year=2018,
    levels=QUANTILE_LEVELS,
    midday_hours=range(9, 17),
    n_display_bins=20,
):
    """
    Seasonal reliability check for quantile forecasts.

    Parameters
    ----------
    realised   : (N_days, K) realised trajectories
    quantiles  : (N_days, K, Q) forecast quantiles for the same days/leads
    dates      : optional DatetimeIndex of length N_days
    year       : used only when dates is None
    levels     : quantile levels, same length as Q
    midday_hours: hours used for the printed summary
    n_display_bins: ignored here, kept for API symmetry with the scenario version

    Returns
    -------
    dict with:
      - pit_hour   : (N_days, K) PIT values from the quantile CDF
      - coverage   : (4, K, Q) empirical coverage by season/hour/quantile level
      - season     : (N_days,) season labels
    """

    y = np.asarray(realised, dtype=float)              # (N, K)
    q = np.asarray(quantiles, dtype=float)            # (N, K, Q)
    levels = np.asarray(levels, dtype=float)

    N, K = y.shape
    assert q.shape[:2] == (N, K), f"realised {y.shape} vs quantiles {q.shape}"
    assert q.shape[2] == len(levels), (
        f"quantiles has {q.shape[2]} columns but levels has {len(levels)} entries"
    )

    # ---- season labels ----
    if dates is None:
        dates = pd.date_range(f"{year}-01-01", periods=N, freq="D")
    dates = pd.DatetimeIndex(dates)
    m2s = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    season = dates.month.map(m2s).to_numpy()
    order = ["Winter", "Spring", "Summer", "Autumn"]

    # ============================================================
    # 1) PIT / probability-transform reliability per hour
    #    For each day/horizon, treat the quantile forecast as a CDF
    #    and map the realised value through it.
    # ============================================================
    pit_hour = np.empty((N, K), dtype=float)

    for d in range(N):
        for h in range(K):
            marginal = MarginalDistCDF(levels, q[d, h])
            u = np.clip(marginal.cdf(y[d, h]), EPS, 1.0 - EPS)
            pit_hour[d, h] = u

    # ============================================================
    # 2) Empirical quantile coverage by season / hour / level
    #    Coverage(alpha) = mean(y <= q_alpha)
    #    Calibration implies coverage ~= alpha over time.
    # ============================================================
    coverage = np.empty((len(order), K, len(levels)), dtype=float)

    for s_idx, s in enumerate(order):
        mask = season == s
        for h in range(K):
            # q[mask, h, :] -> (n_days, Q)
            q_s = q[mask, h, :]
            y_s = y[mask, h]
            coverage[s_idx, h, :] = (y_s[:, None] <= q_s).mean(axis=0)

    # ============================================================
    # 3) Plot 1: season x hour PIT heatmap
    #    Mean PIT near 0.5 = well-centred.
    # ============================================================
    pit_heat = np.vstack([pit_hour[season == s].mean(axis=0) for s in order])  # (4, K)
    fig1, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(pit_heat, aspect="auto", cmap="RdBu_r", vmin=0.3, vmax=0.7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(order)
    ax.set_xticks(range(K))
    ax.set_xticklabels(range(K))
    ax.set_xlabel("hour of horizon")
    ax.set_title(
        "Mean PIT of realised through quantile CDF "
        "(0.5 = centred; <0.5 => realised too low for forecast distribution; >0.5 => realised too high)"
    )
    for i in range(4):
        for h in range(K):
            ax.text(h, i, f"{pit_heat[i, h]:.2f}", ha="center", va="center", fontsize=6)
    fig1.colorbar(im, shrink=0.8)
    plt.tight_layout()

    # ============================================================
    # 4) Plot 2: reliability curves per season
    #    For each season, average empirical coverage over all hours
    #    and compare to the nominal quantile levels.
    # ============================================================
    fig2, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, s in zip(axes.flat, order):
        s_idx = order.index(s)
        cov_avg = coverage[s_idx].mean(axis=0)  # average over K hours
        ax.plot(levels, cov_avg, "o-", color="C0", label="empirical coverage")
        ax.plot([0, 1], [0, 1], "k--", label="perfect reliability")
        ax.set_title(f"{s}: quantile reliability")
        ax.set_xlabel("nominal quantile level")
        ax.set_ylabel("empirical coverage")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    plt.tight_layout()

    # ============================================================
    # 5) Printed summary
    #    - midday mean PIT
    #    - mean absolute coverage bias versus nominal levels
    # ============================================================
    midday = list(midday_hours)
    print(f"{'season':8s} {'midday meanPIT':>15s} {'midday |coverage-bias|':>22s}")
    for s in order:
        s_idx = order.index(s)
        mask = season == s
        md_pit = pit_hour[mask][:, midday].mean()
        cov_bias = np.abs(coverage[s_idx, midday, :] - levels).mean()
        print(f"{s:8s} {md_pit:15.3f} {cov_bias:22.3f}")

    plt.show()

    return {
        "pit_hour": pit_hour,
        "coverage": coverage,
        "season": season,
    }
