from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cvxpy as cp

from dispatch_shared import Variables, Variables1Stage, ProblemBundle
from dispatch_objectives import build_objective_2stage, build_objective_1stage

# =====================================================================================
# dispatch_setup.py  --  FixedParams + the complete per-formulation problem assembly,
# ONE function per formulation (point-robust, full-robust, 1-stage). This is the
# consolidation of what used to be three separate "front door" files
# (dispatch_point_robust_modular.py / dispatch_full_robust_modular.py /
# dispatch_1stage_modular.py) -- their FixedParams/default_fixed_params/build_problem
# now live here (those three files no longer exist), merged into setup_point_robust/
# setup_full_robust/setup_1stage, which take `mode` directly and return the complete
# ProblemBundle. Everything that's
# genuinely formulation-specific (variable declarations, feasibility mechanism) is
# still a distinct, self-contained function per formulation, exactly as before -- only
# the wrapping/front-door layer has been folded in, not the underlying design.
#
# The objective (dispatch_objectives.py's build_objective_2stage/build_objective_1stage)
# stays a SEPARATE, shared module, imported here rather than duplicated -- it's the one
# piece proven identical between point-robust and full-robust, and folding it into this
# file would re-introduce the duplication risk consolidation is meant to remove.
#
# GAMMA: all three formulations now default to 1e-4 (full-robust previously defaulted
# to 1e-6, matching models_robust/dispatch_layer_robust.py's own default -- unified on
# request). Every production call site (evaluate.py, train_corner_*.py, gamma_sweep.py)
# passes gamma explicitly regardless, so this only changes behaviour for callers that
# rely on the default -- currently just 6_models/param_sweeps/'s h_selection_sweep.py.
# =====================================================================================


@dataclass
class FixedParams:
    """Shared by point-robust and full-robust -- their physical configuration is
    identical in every field, including gamma's default now that both formulations
    use 1e-4."""
    T_total: int          # number of settlement periods in the horizon (=24)
    num_scenarios: int    # scenarios fed to the objective (=64 for BOTH train and test)
    dt: float              # length of a settlement period (=1.0 h)
    eta_ch: float          # charging efficiency
    eta_dis: float         # discharging efficiency
    C_ch: float            # max charge rate  (MW, grid-side)
    C_dis: float           # max discharge rate (MW, grid-side)
    B_max: float           # battery energy capacity (MWh)
    SOC0: float            # initial == terminal state of charge (MWh)
    gamma: float = 1e-4    # Tikhonov coefficient (uniqueness / KKT-invertibility)


def default_fixed_params(num_scenarios: int = 64, gamma: float = 1e-4) -> FixedParams:
    """Default physical configuration for BOTH point-robust and full-robust."""
    return FixedParams(
        T_total=24, num_scenarios=num_scenarios, dt=1.0,
        eta_ch=0.95, eta_dis=0.95, C_ch=2.0, C_dis=2.0, B_max=4.0, SOC0=2.0,
        gamma=gamma,
    )


@dataclass
class FixedParams1Stage:
    T_total: int
    num_scenarios: int    # dual-price epigraph only (dispatch_objectives._build_dual_price_epigraph)
                           # -- mathematically inert for the 1-stage decision itself (pinned bid, no
                           # recourse means C_imb never depends on p_ch_hat/p_dis_hat regardless of N);
                           # kept so 1-stage and 2-stage dual-price share ONE epigraph implementation.
    dt: float
    eta_ch: float
    eta_dis: float
    C_ch: float
    C_dis: float
    B_max: float
    SOC0: float
    gamma: float = 1e-4   # Tikhonov coefficient (uniqueness / KKT-invertibility)


def default_fixed_params_1stage(num_scenarios: int = 64, gamma: float = 1e-4) -> FixedParams1Stage:
    """Same pinned physical configuration as the 2-stage models -- only battery/bid
    physics; there is no `k` here (no tracking term this model could build). num_scenarios
    defaults to 64, matching the 2-stage formulations' N, purely so single-price's
    epigraph-free objective and dual-price's shared _build_dual_price_epigraph reference
    the same fp.num_scenarios field either way -- see FixedParams1Stage's field comment
    for why this is inert for single-price and for the 1-stage decision itself."""
    return FixedParams1Stage(
        T_total=24, num_scenarios=num_scenarios, dt=1.0, eta_ch=0.95, eta_dis=0.95,
        C_ch=2.0, C_dis=2.0, B_max=4.0, SOC0=2.0, gamma=gamma,
    )


# =====================================================================================
# FULL-ROBUST -- exact robustness over the whole box {xi : -h_minus <= xi <= h_plus},
# via LP duality (robustify_vec) -- 12 extra (T,T) dual-variable blocks (6 calls, 2
# blocks each), vs. point-robust's 2 literal constraint blocks and zero duals.
#
# p_da_bat is a genuine cp.Variable, pinned via an equality constraint, and D_ch/D_dis's
# lower-triangular shape is enforced via cp.upper_tri(.)==0 rather than a mask -- both
# flagged elsewhere as removable-but-not-yet-removed relative to point-robust's
# simplifications. Faithful to the deployed models_robust/dispatch_layer_robust.py, not
# a fixed version of it.
# =====================================================================================


def robustify_vec(A0, A, B, T, h_plus, h_minus):
    """LP-duality robustification: A0_t + (A xi)_t <= B for all xi in
    {xi : -h_minus <= xi <= h_plus}, componentwise. Exists mu_p, mu_m >= 0 with
    mu_p - mu_m == A and mu_p @ h_plus + mu_m @ h_minus <= B - A0. Currently only used
    by full-robust's setup, but formulation-agnostic (pure LP-duality machinery) --
    lives here rather than in dispatch_setup.py so a future robust variant can reuse it
    without importing full-robust's own module."""
    mu_p = cp.Variable((T, T), nonneg=True)
    mu_m = cp.Variable((T, T), nonneg=True)
    return [
        mu_p - mu_m == A,
        mu_p @ h_plus + mu_m @ h_minus <= B - A0,
    ]


def setup_full_robust(fp: FixedParams, mode: str) -> ProblemBundle:
    T, dt = fp.T_total, fp.dt

    p_ch_hat  = cp.Variable(T, name="p_ch_hat")
    p_dis_hat = cp.Variable(T, name="p_dis_hat")
    D_ch      = cp.Variable((T, T), name="D_ch")
    D_dis     = cp.Variable((T, T), name="D_dis")
    p_da_bat  = cp.Variable(T, name="p_da_bat")

    cons = [
        cp.upper_tri(D_ch) == 0,
        cp.upper_tri(D_dis) == 0,
        p_da_bat == p_ch_hat - p_dis_hat,     # PIN the bid
    ]

    D_ch_eff, D_dis_eff = D_ch, D_dis         # already triangular once the above holds
    R = np.eye(T) + D_ch_eff - D_dis_eff

    power_flow_hat = fp.eta_ch * p_ch_hat - (1.0 / fp.eta_dis) * p_dis_hat
    s_hat = fp.SOC0 + dt * cp.cumsum(power_flow_hat)
    D_net = fp.eta_ch * D_ch_eff - (1.0 / fp.eta_dis) * D_dis_eff
    G = dt * cp.cumsum(D_net, axis=0)

    h_plus  = cp.Parameter(T, nonneg=True, name="h_plus")
    h_minus = cp.Parameter(T, nonneg=True, name="h_minus")

    v = Variables(
        p_ch_hat=p_ch_hat, p_dis_hat=p_dis_hat, D_ch=D_ch, D_dis=D_dis,
        D_ch_eff=D_ch_eff, D_dis_eff=D_dis_eff,
        p_da_bat=p_da_bat, R=R,
        s_hat=s_hat, G=G, h_plus=h_plus, h_minus=h_minus,
    )

    cons += robustify_vec( p_ch_hat,  D_ch_eff, fp.C_ch, T, h_plus, h_minus)
    cons += robustify_vec(-p_ch_hat, -D_ch_eff, 0.0,     T, h_plus, h_minus)
    cons += robustify_vec( p_dis_hat,  D_dis_eff, fp.C_dis, T, h_plus, h_minus)
    cons += robustify_vec(-p_dis_hat, -D_dis_eff, 0.0,      T, h_plus, h_minus)
    cons += robustify_vec( s_hat,  G, fp.B_max, T, h_plus, h_minus)
    cons += robustify_vec(-s_hat, -G, 0.0,       T, h_plus, h_minus)
    cons += [
        s_hat[T - 1] == fp.SOC0,
        G[T - 1, :] == 0,
    ]

    obj, obj_params, epigraph_cons = build_objective_2stage(fp, mode, v)
    cons += epigraph_cons
    params = obj_params + [h_plus, h_minus]

    prob = cp.Problem(cp.Minimize(obj), cons)
    assert prob.is_dcp(dpp=True), "not DPP -- cvxpylayers will reject"

    # p_da_bat IS a layer output here (a real Variable), unlike point-robust where it's
    # a plain expression reconstructed downstream.
    variables = [p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_bat]
    return ProblemBundle(
        problem=prob,
        params=params,
        variables=variables,
        param_by_name={p.name(): p for p in params},
        var_by_name={var.name(): var for var in variables},
        mode=mode,
    )


# =====================================================================================
# 1-STAGE -- no D_ch/D_dis, no R, no G, no box -- there is no recourse mechanism at
# all, so nothing here needs a box-relative (xi) coordinate. p_da_bat is kept as a
# pure-variable DPP stand-in for (p_da - pl_hat), pinned via an equality constraint,
# matching the same not-yet-simplified pattern as full-robust's.
# =====================================================================================
def setup_1stage(fp: FixedParams1Stage, mode: str) -> ProblemBundle:
    T, dt = fp.T_total, fp.dt

    p_ch_hat  = cp.Variable(T, name="p_ch_hat")
    p_dis_hat = cp.Variable(T, name="p_dis_hat")
    p_da_bat  = cp.Variable(T, name="p_da_bat")

    cons = [p_da_bat == p_ch_hat - p_dis_hat]   # PIN the bid

    power_flow_hat = fp.eta_ch * p_ch_hat - (1.0 / fp.eta_dis) * p_dis_hat
    s_hat = fp.SOC0 + dt * cp.cumsum(power_flow_hat)

    v = Variables1Stage(p_ch_hat=p_ch_hat, p_dis_hat=p_dis_hat, p_da_bat=p_da_bat,
                         s_hat=s_hat)

    cons += [
        p_ch_hat >= 0, p_ch_hat <= fp.C_ch,
        p_dis_hat >= 0, p_dis_hat <= fp.C_dis,
        s_hat >= 0, s_hat <= fp.B_max,
        s_hat[T - 1] == fp.SOC0,
    ]

    obj, obj_params, epigraph_cons = build_objective_1stage(fp, mode, v)
    cons += epigraph_cons

    prob = cp.Problem(cp.Minimize(obj), cons)
    assert prob.is_dcp(dpp=True), "not DPP -- cvxpylayers will reject"

    variables = [p_ch_hat, p_dis_hat, p_da_bat]
    return ProblemBundle(
        problem=prob,
        params=obj_params,
        variables=variables,
        param_by_name={p.name(): p for p in obj_params},
        var_by_name={var.name(): var for var in variables},
        mode=mode,
    )
