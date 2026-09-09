from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# =====================================================================================
# dispatch_shared.py  --  infrastructure shared across every dispatch formulation
# (point-robust, full-robust, 1-stage): the two Variables containers, the one
# ProblemBundle every build_problem() returns, the solver-mechanics functions
# (make_layer / order_param_values / solve_plain / robustify_vec), and the pre-solve
# input-construction helpers (_to_t / cholesky_of_second_moment / get_prices /
# oracle_price_values / select_fc_columns / compute_box) -- none of these care which
# formulation built the cp.Problem they're handed, or whether the caller is training
# or testing.
#
# `mode` (not `price_model` + `k`) is the single selector threaded through everything
# problem-construction-related below: "single-price" | "dual-price" | "dispatchability".
# The first two carry over what `price_model` used to mean; the third replaces `k=1` --
# see dispatch_objectives.py for where the actual branching on it happens.
#
# Moved here from dispatch_wrapper.py: _to_t, cholesky_of_second_moment, PRICE_COLS,
# get_prices, oracle_price_values -- all pre-solve input shaping, used identically by
# training and testing, with no dependency on realised_breakdown's post-solve engine.
# dispatch_wrapper.py now imports _to_t from HERE (the dependency direction flipped --
# it used to be the other way around, back when compute_box first moved).
# =====================================================================================


def _to_t(x, like: torch.Tensor | None = None) -> torch.Tensor:
    """
    Converts input to torch tensor, optionally matching device and
    dtype of input to reference tensor. Especially useful to ensure
    gradient can flow through back to forecaster for realised_breakdown. 
    """
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float64))
    if like is not None:
        t = t.to(dtype=like.dtype, device=like.device)
    return t


# -------------------------------------------------------------------------------------
# Second-moment Cholesky factor, DIFFERENTIABLE (torch.linalg.cholesky, not numpy) --
# k=1's tracking term needs grad to flow xi -> Sigma_xi_chol -> the layer -> the
# forecaster. An earlier version computed this via xi.detach().cpu().numpy() +
# np.linalg.cholesky, which silently zeroed that gradient path; this is the fix.
# M = xi^T xi / N (biased /N: matches dispatch_layer's sum_squares(R @ Sigma_xi_chol)
# trace identity exactly for mean-centred xi). Symmetrise + jitter before Cholesky --
# cheap insurance against intermittent non-PD M (N=64 > T=24 so generically full rank,
# but not guaranteed).
# -------------------------------------------------------------------------------------
def cholesky_of_second_moment(xi: torch.Tensor, jitter: float = 1e-9) -> torch.Tensor:
    N = xi.shape[-2]
    M = xi.transpose(-2, -1) @ xi / N
    M = 0.5 * (M + M.transpose(-2, -1)) + jitter * torch.eye(xi.shape[-1], dtype=xi.dtype, device=xi.device)
    return torch.linalg.cholesky(M)   # lower L with L @ L^T == M; grad flows to xi


# -------------------------------------------------------------------------------------
# PLUG POINT: prices straight from the dataset columns (no computation in between).
# imb_up/imb_down are the FULL settlement price level for each direction (imb_up =
# max(da,imb), imb_down = min(da,imb)) -- NOT a premium over da. An earlier convention
# fed max(0,imb-da)/max(0,da-imb) (the premium only), which under-charged every unit of
# shortfall by exactly `da` relative to a fully-consistent settlement (pay da for the
# bid, the FULL imb price for the imbalance volume) -- an unbounded, learnable exploit
# for pl_hat under dual pricing (verified algebraically: this convention makes dual
# price's total cost match single price's already-correct signed formula exactly).
# Clipped at 0 for the rare (~0.1-1.4% of hours, clustered in the Apr-Jul 2020 COVID
# demand-collapse period) negative-price hours, to satisfy the nonneg=True Parameter
# constraint -- a minor, bounded conservatism (negative pricing is exogenous, not
# something a policy can manufacture by biasing its own forecast).
#   single -> {pi_da, pi_imb}                  (signed settlement price 'imb')
#   dual   -> {pi_da, pi_imb_up, pi_imb_down}  (full price level per direction, clipped >= 0)
# -------------------------------------------------------------------------------------
PRICE_COLS = ("da", "imb", "imb_up", "imb_down")

# NOTE: `cols` has NO default on either function below, deliberately. `get_prices`
# only interprets POSITIONS in `price_day` -- it never checks whether the values
# themselves are true settlement prices or the fc/proxy columns (that's entirely
# determined by which array the caller passes as `price_day`, e.g. via
# select_fc_columns(...) reordering fc columns into PRICE_COLS's positions). A default
# of PRICE_COLS here would silently paper over a caller passing an array whose actual
# column order doesn't match what `cols` claims -- forcing every call site to state
# `cols=PRICE_COLS` explicitly makes that assumption visible and reviewable everywhere
# a decision solve's price source could otherwise be confused, rather than leaving it
# an invisible default a future edit could drift away from unnoticed.


def get_prices(price_day, price_model, cols):
    """price_day: (T, 4) slice from WindowSet.price; columns in `cols` order."""
    price_day = np.asarray(price_day, dtype=float)
    idx = {c: i for i, c in enumerate(cols)}
    da = price_day[..., idx["da"]]
    if price_model == "single":
        return {"pi_da": da, "pi_imb": price_day[..., idx["imb"]]}   # pi_imb POSITIVE; sign is in p_imb
    if price_model == "dual":
        return {"pi_da": da,
                "pi_imb_up": np.clip(price_day[..., idx["imb_up"]], 0.0, None),     # clamp = insurance
                "pi_imb_down": np.clip(price_day[..., idx["imb_down"]], 0.0, None)}
    raise ValueError(price_model)


# -------------------------------------------------------------------------------------
# DECISION-TIME price routing: every solver call that DECIDES decision variables (the
# training/test bid-shaping solve, and the oracle's perfect-foresight solve) must be fed
# guaranteed->=0 prices, never the true (possibly negative) columns -- get_prices' own
# nonneg=True Parameter clamp already enforces this for imb_up/imb_down, but "da"/"imb"
# carry no such attribute and would otherwise pass true negative values straight through
# into a decision. da_fc/imb_fc/imb_up_fc/imb_down_fc are the real, always->=0 proxy
# columns for exactly this purpose (see 3_data_processing/power_prices.ipynb) -- true
# prices are reserved for settlement (dispatch_wrapper.realised_breakdown/realised_cost),
# never for a decision solve. `real_cols` names which four physical columns to pull, in
# PRICE_COLS-matching (da, imb, imb_up, imb_down) SLOT order -- explicit at every call
# site, no default, so a call site can never silently drift onto the true columns.
# -------------------------------------------------------------------------------------
def select_fc_columns(price_day, cols, real_cols):
    """Reorders `real_cols` (typically the four _fc columns) out of `price_day` into
    PRICE_COLS-matching positions, so an ordinary get_prices(..., cols=PRICE_COLS) call
    resolves its literal 'da'/'imb'/'imb_up'/'imb_down' lookups against proxy data --
    get_prices only interprets POSITIONS, never checks which physical columns fed it."""
    price_day = np.asarray(price_day, dtype=float)
    idx = {c: i for i, c in enumerate(cols)}
    positions = [idx[c] for c in real_cols]
    return price_day[..., positions]


def oracle_price_values(price_day, price_model, realised, cols):
    """Full param dict for solve_oracle: prices + p_d = realised prosumption (perfect foresight)."""
    d = get_prices(price_day, price_model, cols)
    d["p_d"] = np.asarray(realised, dtype=float)
    return d


def price_model_for_settlement(mode: str) -> str:
    """Maps a `mode` ("single-price"/"dual-price"/"dispatchability") onto the
    "single"/"dual" convention get_prices/realised_breakdown expect for their
    price_model argument -- these are DELIBERATELY separate axes now: `mode` selects
    which cvxpy problem was built (what the decision solve needs), this selects only
    which of realised_breakdown's two settlement branches to take.

    For "dispatchability" this choice is PROVABLY inert: realised_breakdown computes
    p_imb (dispatch_wrapper.py: `p_imb = p_g - bid`) BEFORE branching on price_model at
    all, and the dispatchability loss/metric only ever reads p_imb, never C_da/C_imb.
    "single" is picked arbitrarily for it, purely to avoid fetching pi_imb_up/
    pi_imb_down for a value that's never used -- used by both
    7_model_training/dfl_train_utils.py and 8_testing/evaluate.py, so this ONE
    arbitrary choice is made in exactly one place."""
    if mode == "single-price":
        return "single"
    if mode == "dual-price":
        return "dual"
    if mode == "dispatchability":
        return "single"
    raise ValueError(mode)


def build_layer_vals(prices, h_plus=None, h_minus=None, xi_samples=None, Sigma_xi_chol=None):
    """Assembles a cvxpy Parameter-name -> value dict from whichever pieces the caller
    has; type-agnostic (numpy arrays for a plain solve_plain call, torch tensors for a
    cvxpylayers forward pass) since it never touches the values themselves, only
    conditionally inserts them. Extra entries for Parameters the built problem doesn't
    actually declare are harmless -- the caller's own `[vals[name] for name in keys]`
    (keys = the bundle's real param names) silently ignores anything not needed."""
    vals = dict(prices)
    if h_plus is not None:
        vals["h_plus"] = h_plus
    if h_minus is not None:
        vals["h_minus"] = h_minus
    if xi_samples is not None:
        vals["xi_samples"] = xi_samples
    if Sigma_xi_chol is not None:
        vals["Sigma_xi_chol"] = Sigma_xi_chol
    return vals


@dataclass
class Variables:
    """Shared by the two 2-stage formulations (point-robust, full-robust). Declared by
    dispatch_setup.py's setup_point_robust()/setup_full_robust(); consumed by
    dispatch_objectives.py's build_objective_2stage(), which only ever reads R --
    never how it was made feasible.

    No imb_det field: it was always provably 0 (identically for point-robust's plain-
    expression p_da_bat, or enforced by full-robust's pinning equality constraint) in
    every place it was used (single-price's C_imb, dispatchability's sum_trace), so it
    contributed nothing to any objective value, argmin, or gradient -- removed rather
    than kept as an always-zero term. See dispatch_objectives.py's module docstring for
    the pinned-bid argument this removal relies on, which would need revisiting if a
    future formulation ever left the bid free."""
    p_ch_hat:  cp.Variable      # (T,)
    p_dis_hat: cp.Variable      # (T,)
    D_ch:      cp.Variable      # (T,T) RAW (pre-mask, or already-triangular for full-robust)
    D_dis:     cp.Variable      # (T,T) RAW (pre-mask, or already-triangular for full-robust)
    D_ch_eff:  cp.Expression    # (T,T) lower-triangular / non-anticipative
    D_dis_eff: cp.Expression    # (T,T) lower-triangular / non-anticipative
    p_da_bat:  cp.Expression    # (T,) PINNED bid-minus-pl_hat (plain expr. or a pinned Variable)
    R:         np.ndarray       # (T,T) net recourse matrix I + D_ch_eff - D_dis_eff
    s_hat:     cp.Expression    # (T,) nominal (xi=0) SOC trajectory
    G:         cp.Expression    # (T,T) recourse SOC-gain matrix
    h_plus:    cp.Parameter     # (T,) frozen box, upper edge
    h_minus:   cp.Parameter     # (T,) frozen box, lower edge


@dataclass
class Variables1Stage:
    """1-stage's own, much smaller set -- no D_ch/D_dis/R/G/box at all, since nothing
    here ever adapts to a realisation. Declared by dispatch_setup.py's setup_1stage();
    consumed by dispatch_objectives.py's build_objective_1stage().

    No imb_det field -- see Variables' docstring for why it was removed (always
    provably 0 here too, enforced by setup_1stage's pinning equality constraint)."""
    p_ch_hat:  cp.Variable      # (T,)
    p_dis_hat: cp.Variable      # (T,)
    p_da_bat:  cp.Variable      # (T,) PINNED via an equality constraint (see setup_1stage)
    s_hat:     cp.Expression    # (T,) SOC trajectory -- fully deterministic, no recourse term


@dataclass
class ProblemBundle:
    """Same shape for all three formulations now that `mode` has replaced the old
    (price_model, k) pair -- point-robust/full-robust no longer need a separate `k`
    field, and 1-stage no longer needs to be the odd one out without one."""
    problem: cp.Problem
    params: list
    variables: list
    param_by_name: dict
    var_by_name: dict
    mode: str


def make_layer(bundle: ProblemBundle) -> CvxpyLayer:
    return CvxpyLayer(bundle.problem, parameters=bundle.params, variables=bundle.variables)


def order_param_values(bundle: ProblemBundle, **named) -> list:
    """Return values in the bundle's parameter order, for the CvxpyLayer forward pass."""
    missing = [p.name() for p in bundle.params if p.name() not in named]
    if missing:
        raise KeyError(f"missing param values for this (mode={bundle.mode!r}) problem: "
                        f"{missing}. Expected order: {[p.name() for p in bundle.params]}")
    return [named[p.name()] for p in bundle.params]


def solve_plain(bundle: ProblemBundle, values: dict, solver=cp.GUROBI, **solver_kwargs) -> dict:
    for name, p in bundle.param_by_name.items():
        if name not in values:
            raise KeyError(f"missing param value: {name} (need {list(bundle.param_by_name)})")
        p.value = np.asarray(values[name], dtype=float)
    bundle.problem.solve(solver=solver, **solver_kwargs)
    if bundle.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"plain solve status: {bundle.problem.status}")
    out = {name: np.asarray(v.value) for name, v in bundle.var_by_name.items()}
    out["objective_no_da_const"] = float(bundle.problem.value)
    return out


def compute_box(baseline_quantiles, sampler, quantile_levels,
                box_levels=(0.15, 0.85), min_box=1e-4):
    """B1 frozen box from the BASELINE forecaster, for ONE day. Moved here from the
    former models_robust/dispatch_wrapper_robust.py -- box computation is formulation-
    agnostic (used by both point-robust and full-robust's setup), so it belongs beside
    the other shared infrastructure, not in a robust-specific module.

    baseline_quantiles: (K, Q) from the frozen reference (128) model, physical MW.
    Expressed against the baseline MEAN anchor (sampler.mean_and_errors) so that, applied
    in each model's own xi = (realised - mean) coordinate, the half-widths are identical
    everywhere (B1). Returns detached (h_plus, h_minus) -- constant across all models/days.
    Compute ONCE per test day and cache.

    box_levels default = (0.15, 0.85), locked in from h_selection_sweep.py /
    h_selection_sweep_point.py's comparison: dominates the previously-used (0.20, 0.80)
    on both the full LP-dual robust model and the point-constrained training surrogate
    (near-identical control-action cost, meaningfully lower saturation on both) --
    verified on 2018 training-period data, both price models.
    """
    q = _to_t(baseline_quantiles)
    if q.dim() != 2:
        raise ValueError("baseline_quantiles must be (K, Q) for a single day")
    base_mean, _ = sampler.mean_and_errors(baseline_quantiles)      # SAME anchor convention
    base_mean = _to_t(base_mean, like=q)
    levels = np.asarray(quantile_levels, float)
    i_lo = int(np.argmin(np.abs(levels - box_levels[0])))
    i_hi = int(np.argmin(np.abs(levels - box_levels[1])))
    q_lower, q_upper = q[:, i_lo], q[:, i_hi]
    h_plus  = q_upper - base_mean
    h_minus = base_mean - q_lower
    with torch.no_grad():
        if (h_plus < 0).any() or (h_minus < 0).any():
            warnings.warn("baseline mean outside [q_lower, q_upper] on some lead(s) "
                          "(extreme skew); clamping half-width to min_box.")
    return (torch.clamp(h_plus,  min=min_box).detach(),
            torch.clamp(h_minus, min=min_box).detach())



