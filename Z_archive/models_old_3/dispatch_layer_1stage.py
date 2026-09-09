from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# =====================================================================================
# dispatch_layer_1stage.py  --  ONE-STAGE commitment: mean forecast only, NO recourse.
#
# WHY THIS EXISTS (see pivot_context.md for the full account): the 2-stage NON-ROBUST
# LDR (dispatch_layer.py) has D_ch/D_dis as free decision variables with NO constraint
# tying their magnitude to physical realizability -- only upper_tri(.)==0 (a causality
# SHAPE constraint, not a feasibility bound) and a near-negligible Tikhonov penalty
# (GAMMA=1e-4, "just for uniqueness"). At k=1 the tracking objective rewards D_ch/D_dis
# for reducing variance with NO awareness that the resulting recourse response might be
# physically unrealizable -- the optimizer solves a fantasy where recourse is free, then
# reality (the hard clip in realised_breakdown) intervenes afterward. That's a
# structurally ill-posed recourse formulation, not merely an expensive one: it produces
# decisions the downstream physics silently invalidates 74.6% of the time (measured,
# single/k=1, current forecaster).
#
# THIS model has no recourse mechanism at all -- p_ch_hat/p_dis_hat/p_da_rel are the
# ONLY decisions, and every one of them is fully, correctly bounded by physics in the
# optimization problem itself. Consequence: realised_breakdown's clip becomes a
# mathematical no-op. With D_ch=D_dis=0, p_ch_raw = p_ch_hat + 0@xi = p_ch_hat, which is
# ALREADY guaranteed feasible by the hard constraints -- there is nothing left to clip,
# ever, by construction. The saturation/dead-gradient pathology that motivated the whole
# reserve-penalty investigation doesn't need patching here; it structurally cannot occur.
#
# TRADE-OFF, stated plainly: this model's objective can only ever be the ECONOMIC term
# (C_da + C_imb) -- there's no D_ch/D_dis to build a tracking/variance term from, so
# `k` (the economic/tracking blend) doesn't apply here at all. Training against this
# model teaches the forecaster nothing about the cost of its own predictive uncertainty
# -- only about the mean forecast's usefulness under a rigid, non-adaptive commitment.
# That's the deliberate scope: TRAIN here (cheap, structurally clean gradient), then
# TEST the resulting forecaster against the two-stage ROBUST LDR (dispatch_layer_robust.py,
# the properly-specified recourse formulation, h_plus/h_minus actually grounding the
# response) to check whether the decision-focused signal learned here transfers.
#
# Reuses dispatch_wrapper.realised_breakdown/realised_cost/per_day_metrics UNCHANGED --
# call them with D_ch=np.zeros((T,T)), D_dis=np.zeros((T,T)) (or torch.zeros for the
# differentiable path) and they compute exactly the right thing.
# =====================================================================================


@dataclass
class FixedParams1Stage:
    T_total: int
    dt: float
    eta_ch: float
    eta_dis: float
    C_ch: float
    C_dis: float
    B_max: float
    SOC0: float
    gamma: float = 1e-4   # Tikhonov coefficient (uniqueness / KKT-invertibility)


def default_fixed_params(gamma: float = 1e-4) -> FixedParams1Stage:
    """Same pinned physical configuration as the 2-stage models -- only battery/bid
    physics, no `k` (there's no tracking term this model could blend against)."""
    return FixedParams1Stage(
        T_total=24, dt=1.0, eta_ch=0.95, eta_dis=0.95, C_ch=2.0, C_dis=2.0,
        B_max=4.0, SOC0=2.0, gamma=gamma,
    )


@dataclass
class ProblemBundle:
    problem: cp.Problem
    params: list
    variables: list
    param_by_name: dict
    var_by_name: dict
    price_model: str
    # Reminder: the layer objective EXCLUDES the dropped DA constant  pi_da . pl_hat . dt.
    # Add it back downstream when reporting absolute cost / regret (same convention as
    # dispatch_layer.py).


def build_problem(fp: FixedParams1Stage, price_model: str) -> ProblemBundle:
    assert price_model in ("single", "dual")
    T, dt = fp.T_total, fp.dt

    # ---- decision variables -- NO D_ch, NO D_dis --------------------------------
    p_ch_hat  = cp.Variable(T, name="p_ch_hat")
    p_dis_hat = cp.Variable(T, name="p_dis_hat")

    # NOTE: pl_hat is NOT a cvxpy Parameter here -- with the bid pinned, the forecast
    # never enters this optimization problem; it only matters downstream in
    # realised_cost/realised_breakdown. See the matching note in dispatch_layer.py.
    params = []

    # ---- bid: PINNED, not free (matches models_old/single_price_robust_old.py's
    # p_bid = pl_hat + p_ch_hat - p_dis_hat -- the original formulation never had an
    # independent bid variable). p_da_rel is kept as a pure-variable DPP stand-in for
    # (p_da - pl_hat) so C_da = pi_da @ p_da_rel * dt doesn't multiply pi_da by a
    # pl_hat-containing expression (not DPP) -- but it's now CONSTRAINED to its correct
    # value rather than left free. A free p_da_rel let the solver arbitrage the known
    # pi_da/pi_imb spread independently of the forecast (measured: bid saturated to
    # p_min or p_max every hour, regret ~0 regardless of forecast quality). Pinning it
    # makes imb_det (below) identically zero, matching the theoretical p~imb = xi
    # derivation exactly (no recourse here, so realised imbalance is pure forecast error).
    p_da_rel = cp.Variable(T, name="p_da_rel")

    # ---- SOC: fully deterministic, no recourse term to add ------------------------
    power_flow_hat = fp.eta_ch * p_ch_hat - (1.0 / fp.eta_dis) * p_dis_hat
    s_hat = fp.SOC0 + dt * cp.cumsum(power_flow_hat)

    cons = [
        p_ch_hat >= 0, p_ch_hat <= fp.C_ch,
        p_dis_hat >= 0, p_dis_hat <= fp.C_dis,
        s_hat >= 0, s_hat <= fp.B_max,
        s_hat[T - 1] == fp.SOC0,
        p_da_rel == p_ch_hat - p_dis_hat,   # pin the bid (see note above) -- no separate bound needed
    ]

    # imb_det IS the true (only) imbalance here -- no `+ R@xi` term, since there's no
    # recourse. Deterministic by construction, not a mean-field approximation of it.
    imb_det = p_ch_hat - p_dis_hat - p_da_rel

    pi_da = cp.Parameter(T, name="pi_da")
    params.append(pi_da)
    C_da = pi_da @ p_da_rel * dt   # DA constant pi_da.pl_hat.dt dropped, added back downstream

    if price_model == "single":
        # SIGNED settlement -- exact, no epigraph needed (matches dispatch_layer.py's
        # single-price case, but here it's exact rather than an E[.] identity, since
        # there's no scenario/xi term to average over at all).
        pi_imb = cp.Parameter(T, name="pi_imb")
        params.append(pi_imb)
        C_imb = pi_imb @ imb_det * dt
    else:
        # DUAL: simpler than the 2-stage case -- imb_det is a plain (T,) vector, no
        # scenario broadcasting needed at all (no xi_samples, no /N averaging).
        pi_imb_up = cp.Parameter(T, nonneg=True, name="pi_imb_up")
        pi_imb_down = cp.Parameter(T, nonneg=True, name="pi_imb_down")
        params += [pi_imb_up, pi_imb_down]
        p_plus  = cp.Variable(T, nonneg=True, name="p_plus")
        p_minus = cp.Variable(T, nonneg=True, name="p_minus")
        cons += [p_plus - p_minus == imb_det]
        C_imb = (pi_imb_up @ p_plus + pi_imb_down @ p_minus) * dt

    penalty = fp.gamma * (cp.sum_squares(p_ch_hat) + cp.sum_squares(p_dis_hat)
                           + cp.sum_squares(p_da_rel))

    prob = cp.Problem(cp.Minimize(C_da + C_imb + penalty), cons)
    assert prob.is_dcp(dpp=True), "not DPP -- cvxpylayers will reject"

    variables = [p_ch_hat, p_dis_hat, p_da_rel]
    return ProblemBundle(
        problem=prob, params=params, variables=variables,
        param_by_name={p.name(): p for p in params},
        var_by_name={v.name(): v for v in variables},
        price_model=price_model,
    )


def make_layer(bundle: ProblemBundle) -> CvxpyLayer:
    return CvxpyLayer(bundle.problem, parameters=bundle.params, variables=bundle.variables)


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


# build_oracle / solve_oracle: reuse dispatch_layer.py's UNCHANGED -- the economic oracle
# never referenced D_ch/D_dis/k in the first place, so there's nothing 1-stage-specific
# to build. Import directly rather than duplicating:
#   from dispatch_layer import build_oracle, solve_oracle
