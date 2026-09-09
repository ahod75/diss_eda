from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# =====================================================================================
# dispatch_layer_point_robust.py  --  2-stage LDR, PINNED bid, N=64-scenario objective,
# feasibility grounded at 3 POINTS (nominal, +h_plus, -h_minus) instead of the full
# LP-duality robust counterpart over the continuous box.
#
# WHY: the non-robust dispatch_layer.py has D_ch/D_dis as free recourse matrices with NO
# constraint tying their magnitude to physical realizability -- only the LDR shape
# constraint (upper_tri==0) and a negligible Tikhonov penalty. That's the "half
# formulation" problem: the optimizer can propose recourse the downstream physics
# silently clips, with the training objective none the wiser. The full robust
# counterpart (models_robust/dispatch_layer_robust.py, via LP duality / _robustify_vec)
# fixes this exactly but is expensive (12 extra (T,T) dual-variable blocks). This file
# is a cheap middle ground: instead of the exact worst case over the whole box (which,
# for a linear function of a T-dimensional xi, is attained at a vertex requiring a
# DIFFERENT sign combination per row -- not just "all dimensions at +h_plus" or "all at
# -h_minus" simultaneously), it only checks feasibility at 3 literal points:
#   xi = 0        (nominal -- already covered by the plain p_ch_hat/p_dis_hat/s_hat bounds)
#   xi = +h_plus  (all T dimensions at their upper box edge simultaneously)
#   xi = -h_minus (all T dimensions at their lower box edge simultaneously)
# This is NOT equivalent to exact robustness over the box -- it checks 2 of the 2^T
# vertices, not the true worst case per row (which can require mixed signs across time
# steps given D_ch/D_dis's row-wise coefficients). It's a heuristic that discourages
# D_ch/D_dis from being unconstrained, at a fraction of the LP-duality cost: no dual
# variables at all, just 2 extra literal constraint blocks (Variable @ Parameter, DPP-safe).
#
# Objective is UNCHANGED from dispatch_layer.py: full N=64 xi_samples dual epigraph /
# Sigma_xi_chol tracking term -- only feasibility is cheapened, not the cost expectation.
#
# BID: pinned as a plain expression p_da_rel = p_ch_hat - p_dis_hat (not a Variable +
# equality constraint) -- same rationale as dispatch_layer.py / dispatch_layer_1stage.py
# / models_robust/dispatch_layer_robust.py, but here it's not one of the layer's output
# `variables` -- reconstruct downstream as p_ch_hat_out - p_dis_hat_out.
#
# D_ch/D_dis's lower-triangular (LDR non-anticipativity) shape is enforced via a
# constant mask (D_ch_eff = tril_mask * D_ch) rather than cp.upper_tri(D)==0, since the
# Tikhonov penalty (applied to the RAW D_ch/D_dis) already drives unused upper-triangular
# entries to exactly 0 with no separate equality constraint needed -- saves 2*T*(T-1)/2
# equality rows in the KKT system.
#
# build_oracle / solve_oracle: reuse dispatch_layer.py's UNCHANGED (same pinned-bid
# oracle, independent of how the policy's recourse feasibility is grounded).
# =====================================================================================


@dataclass
class FixedParams:
    T_total: int          # number of settlement periods in the horizon (=24)
    num_scenarios: int    # scenarios fed to the objective (=64 for BOTH train and test)
    dt: float              # length of a settlement period (=1.0 h)
    eta_ch: float          # charging efficiency
    eta_dis: float         # discharging efficiency
    C_ch: float            # max charge rate  (MW, grid-side)
    C_dis: float           # max discharge rate (MW, grid-side)
    B_max: float           # battery energy capacity (MWh)
    SOC0: float            # initial == terminal state of charge (MWh)
    k: float               # 0 = profit-max, 1 = max-dispatchable  (use {0,1} only)
    gamma: float = 1e-4    # Tikhonov coefficient (uniqueness / KKT-invertibility)


def default_fixed_params(k: float, num_scenarios: int = 64, gamma: float = 1e-4) -> FixedParams:
    """Same pinned physical configuration as dispatch_layer.py."""
    assert k in (0.0, 1.0), "this study uses k in {0, 1} only"
    return FixedParams(
        T_total=24, num_scenarios=num_scenarios, dt=1.0,
        eta_ch=0.95, eta_dis=0.95, C_ch=2.0, C_dis=2.0, B_max=4.0, SOC0=2.0,
        k=float(k), gamma=gamma,
    )


@dataclass
class ProblemBundle:
    problem: cp.Problem
    params: list
    variables: list
    param_by_name: dict
    var_by_name: dict
    price_model: str
    k: float
    # Reminder: the layer objective EXCLUDES the dropped DA constant  pi_da . pl_hat . dt.
    # Add it back downstream when reporting absolute cost / regret.


def build_problem(fp: FixedParams, price_model: str) -> ProblemBundle:
    assert price_model in ("single", "dual")
    T, N, dt = fp.T_total, fp.num_scenarios, fp.dt

    # ---- decision variables ------------------------------------------------------
    p_ch_hat  = cp.Variable(T, name="p_ch_hat")
    p_dis_hat = cp.Variable(T, name="p_dis_hat")
    D_ch      = cp.Variable((T, T), name="D_ch")
    D_dis     = cp.Variable((T, T), name="D_dis")

    # NOTE: pl_hat is NOT a cvxpy Parameter here -- with the bid pinned, the forecast
    # never enters this optimization problem; it only matters downstream in
    # realised_cost/realised_breakdown. See the matching note in dispatch_layer.py.
    h_plus  = cp.Parameter(T, nonneg=True, name="h_plus")  # frozen box (B1), point-scenario only
    h_minus = cp.Parameter(T, nonneg=True, name="h_minus")

    cons = []

    # ---- LDR non-anticipativity via a MASK, not an equality constraint -----------
    # cp.upper_tri(D)==0 costs 2*T*(T-1)/2 = 552 equality rows in the KKT system. Since
    # the Tikhonov penalty below applies to the RAW D_ch/D_dis (not the masked version),
    # any nonzero upper-triangular entry only ever adds cost with zero benefit (it's
    # unused everywhere else) -- so the optimizer drives those entries to exactly 0 on
    # its own. D_ch_eff/D_dis_eff (masked) are used in every OTHER constraint/objective
    # term; D_ch/D_dis (raw) appear only in the penalty.
    tril_mask = np.tril(np.ones((T, T)))
    D_ch_eff  = cp.multiply(tril_mask, D_ch)
    D_dis_eff = cp.multiply(tril_mask, D_dis)

    # ---- SOC affine decomposition (bid does NOT enter SOC) -----------------------
    power_flow_hat = fp.eta_ch * p_ch_hat - (1.0 / fp.eta_dis) * p_dis_hat
    s_hat = fp.SOC0 + dt * cp.cumsum(power_flow_hat)               # (T,) nominal SOC
    D_net = fp.eta_ch * D_ch_eff - (1.0 / fp.eta_dis) * D_dis_eff
    G = dt * cp.cumsum(D_net, axis=0)                              # (T,T) recourse SOC gain

    # ---- NOMINAL feasibility (xi=0 point -- cheap, plain hat-quantity bounds) ----
    cons += [p_ch_hat >= 0, p_ch_hat <= fp.C_ch]
    cons += [p_dis_hat >= 0, p_dis_hat <= fp.C_dis]
    cons += [s_hat >= 0, s_hat <= fp.B_max]
    cons += [
        s_hat[T - 1] == fp.SOC0,
        G[T - 1, :] == 0,          # recourse energy-neutral at T -- exact for ANY xi, not just the 3 points
    ]

    # ---- POINT feasibility at the box edges (xi = +h_plus, xi = -h_minus) --------
    # D_ch_eff @ xi_i / G @ xi_i : Variable @ Parameter, DPP-safe. No dual variables.
    for xi_i in (h_plus, -h_minus):
        p_ch_i  = p_ch_hat  + D_ch_eff  @ xi_i
        p_dis_i = p_dis_hat + D_dis_eff @ xi_i
        soc_i   = s_hat + G @ xi_i
        cons += [
            p_ch_i >= 0, p_ch_i <= fp.C_ch,
            p_dis_i >= 0, p_dis_i <= fp.C_dis,
            soc_i >= 0, soc_i <= fp.B_max,
        ]

    # ---- bid: PINNED, as a plain expression (not a Variable + equality constraint) --
    # p_da_rel = p_ch_hat - p_dis_hat is a pure-variable affine EXPRESSION, DPP-safe
    # wherever it's used (C_da = pi_da @ p_da_rel * dt is parameter @ variable-only-
    # expr). This drops 24 variables + 24 equality rows vs. declaring it as a Variable
    # and constraining it. NOT one of the layer's output `variables` any more --
    # reconstruct it downstream as p_ch_hat_out - p_dis_hat_out after calling the layer.
    p_da_rel = p_ch_hat - p_dis_hat

    # ---- shared imbalance building blocks -----------------------------------------
    # imb_det = p_ch_hat - p_dis_hat - p_da_rel is now LITERALLY the zero expression
    # (algebraic identity, not just true at a constrained optimum) -- cvxpy simplifies
    # it directly, no separate equality constraint needed to force it to 0.
    imb_det = p_ch_hat - p_dis_hat - p_da_rel          # (T,) == 0 identically
    R = np.eye(T) + D_ch_eff - D_dis_eff               # (T,T) recourse matrix

    params = []
    obj_terms = []

    # ============================ ECONOMIC TERM  (only if (1-k) > 0), UNCHANGED ====
    if fp.k < 1.0:
        pi_da = cp.Parameter(T, name="pi_da")
        params.append(pi_da)
        C_da = pi_da @ p_da_rel * dt

        if price_model == "single":
            # E[pi_imb . p_imb] = pi_imb . imb_det = 0 (R xi term vanishes, E[xi]=0, and
            # imb_det==0 by the pinned bid) -- no scenarios/epigraph needed.
            pi_imb = cp.Parameter(T, name="pi_imb")
            params.append(pi_imb)
            C_imb = pi_imb @ imb_det * dt
        else:
            xi_samples = cp.Parameter((N, T), name="xi_samples")
            pi_imb_up     = cp.Parameter(T, nonneg=True, name="pi_imb_up")
            pi_imb_down     = cp.Parameter(T, nonneg=True, name="pi_imb_down")
            params += [xi_samples, pi_imb_up, pi_imb_down]
            p_plus  = cp.Variable((T, N), nonneg=True, name="p_plus")
            p_minus = cp.Variable((T, N), nonneg=True, name="p_minus")

            # imb_det == 0 identically (pinned bid), so the deterministic-offset
            # broadcast term vanishes -- imb_scen is purely the recourse response.
            imb_scen = R @ xi_samples.T   # (T, N)
            cons += [p_plus - p_minus == imb_scen]
            C_imb = cp.sum(pi_imb_up @ p_plus + pi_imb_down @ p_minus) * dt / N

        obj_terms.append((1.0 - fp.k) * (C_da + C_imb))

    # ============================ TRACKING TERM  (only if k > 0), UNCHANGED ========
    if fp.k > 0.0:
        Sigma_xi_chol = cp.Parameter((T, T), name="Sigma_xi_chol")
        params.append(Sigma_xi_chol)
        sum_trace = dt**2 * (cp.sum_squares(imb_det) + cp.sum_squares(R @ Sigma_xi_chol))
        obj_terms.append(fp.k * sum_trace)

    # ============================ PENALTY  (always) =================================
    penalty = fp.gamma * (
        cp.sum_squares(p_ch_hat) + cp.sum_squares(p_dis_hat)
        + cp.sum_squares(D_ch) + cp.sum_squares(D_dis)
    )
    obj_terms.append(penalty)

    # trailing box params (stable order)
    params += [h_plus, h_minus]

    prob = cp.Problem(cp.Minimize(sum(obj_terms)), cons)
    assert prob.is_dcp(dpp=True), "not DPP -- cvxpylayers will reject"

    # p_da_rel is NOT in this list -- it's a plain expression now, not a layer output.
    # Reconstruct downstream: p_da_rel_out = p_ch_hat_out - p_dis_hat_out.
    variables = [p_ch_hat, p_dis_hat, D_ch, D_dis]
    return ProblemBundle(
        problem=prob,
        params=params,
        variables=variables,
        param_by_name={p.name(): p for p in params},
        var_by_name={v.name(): v for v in variables},
        price_model=price_model,
        k=float(fp.k),
    )


def make_layer(bundle: ProblemBundle) -> CvxpyLayer:
    return CvxpyLayer(bundle.problem, parameters=bundle.params, variables=bundle.variables)


def order_param_values(bundle: ProblemBundle, **named) -> list:
    """Return values in the bundle's parameter order, for the CvxpyLayer forward pass."""
    missing = [p.name() for p in bundle.params if p.name() not in named]
    if missing:
        raise KeyError(f"missing param values for this ({bundle.price_model}, k={bundle.k}) "
                       f"problem: {missing}. Expected order: {[p.name() for p in bundle.params]}")
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


# build_oracle / solve_oracle: reuse dispatch_layer.py's UNCHANGED -- the oracle's
# feasibility is exact (perfect foresight, no recourse), independent of how the
# policy's recourse feasibility is grounded here.
#   from dispatch_layer import build_oracle, solve_oracle


# -------------------------------------------------------------------------------------
# Second-moment Cholesky factor. Identical to dispatch_layer_robust.py's -- reuse
# directly rather than duplicating:
#   from models_robust.dispatch_layer_robust import cholesky_of_second_moment
# -------------------------------------------------------------------------------------
