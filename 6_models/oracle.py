from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cvxpy as cp

from dispatch_wrapper import realised_cost

# =====================================================================================
# oracle.py  --  perfect-foresight ECONOMIC oracle. Renamed from dispatch_layer.py.
#
# This file used to also contain a full free-bid robust LDR dispatch formulation
# (FixedParams / default_fixed_params / ProblemBundle / build_problem / make_layer /
# order_param_values / solve_plain) -- superseded entirely by dispatch_setup.py's
# setup_point_robust / setup_full_robust / setup_1stage, and removed here after
# confirming zero real callers of any of it anywhere in the live pipeline (grepped
# every training/eval/sweep script, not just checked by description). What's left is
# the one thing every training and evaluation script still actually imports: the
# shared perfect-foresight economic oracle used as the common regret benchmark.
#
# `fp` below is duck-typed, not tied to a specific FixedParams class -- build_oracle is
# called with dispatch_setup.FixedParams (point-robust/full-robust corners) AND
# dispatch_setup.FixedParams1Stage (1-stage corners) at different call sites, and only
# ever needs T_total/dt/eta_ch/eta_dis/C_ch/C_dis/B_max/SOC0 to be present -- there was
# never a single correct static type for it, so no type hint is given rather than one
# that would be quietly wrong for a third of its real callers.
# =====================================================================================


# =====================================================================================
# PERFECT-FORESIGHT ORACLE (deterministic; no LDR, no robust box, no tracking).
# Bid PINNED (bid = p_d + p_ch - p_dis), matching the policy's pinned bid. Gurobi (LP).
# =====================================================================================
@dataclass
class OracleBundle:
    problem: cp.Problem
    param_by_name: dict
    var_by_name: dict
    price_model: str
    objective: str = "economic"   # the ONLY oracle in this study is economic (see build_oracle)


def build_oracle(fp, price_model: str, objective: str = "economic") -> OracleBundle:
    """Perfect-foresight ECONOMIC oracle (min C_da + C_imb), used as the COMMON benchmark
    for every instance -- all corners, baseline and DFL.

    `objective` is an explicit guard, not a variant selector. This study deliberately has
    ONE oracle: the economic one. Regret = policy_realised_cost - economic_oracle_cost is
    reported for all eight instances and read as *economic decision regret* -- true decision
    regret at k=0, and the economic price of dispatchability at k=1 (same formula, the
    interpretation shifts with k). A perfect-foresight DISPATCH oracle is intentionally NOT
    built: min sum (p_imb)^2 under clairvoyance collapses toward a forecast-accuracy gap,
    which CRPS already owns and this model-only framework scopes out. Passing anything other
    than "economic" fails loudly so no k=1 metric can be silently benchmarked against a
    dispatch oracle that does not exist here.
    """
    assert price_model in ("single", "dual")
    if objective != "economic":
        raise ValueError(
            f"build_oracle objective must be 'economic' (got {objective!r}). This study uses a "
            "single common economic oracle for all corners; there is no dispatch oracle -- k=1 "
            "dispatchability is reported on the raw absolute-deviation axis, not as regret.")
    T = fp.T_total

    p_ch  = cp.Variable(T, nonneg=True, name="p_ch")
    p_dis = cp.Variable(T, nonneg=True, name="p_dis")

    p_d   = cp.Parameter(T, name="p_d")                   # TRUE prosumption (perfect foresight)
    pi_da = cp.Parameter(T, name="pi_da")

    # bid: PINNED to the true realised position (bid = p_d + p_ch - p_dis), not a free
    # variable. A free bid here would let the oracle arbitrage the known pi_da/pi_imb
    # spread using ONLY price information -- which the policy has exactly too, prices
    # are never uncertain in this study -- not genuine value of knowing p_d. Left free,
    # it corrupts regret = raw_cost - oracle_cost into mixing "value of perfect load
    # information" with "value of an arbitrage mechanism the (now correctly pinned)
    # policy is structurally barred from" (see dispatch_setup.py's setup_1stage /
    # setup_point_robust / setup_full_robust for the matching note on the policy side).
    # Pinning both consistently means imb == 0 for the oracle by construction (perfect
    # foresight -> no reason to ever accept an imbalance position) and the oracle's
    # entire cost reduces to C_da: pure price arbitrage of p_ch/p_dis against the TRUE
    # load, the correct "best achievable given perfect information" cost.
    bid = p_d + p_ch - p_dis                              # plain solve -- no DPP constraint

    L = np.tril(np.ones((T, T)))
    soc = fp.SOC0 + fp.dt * (L @ (fp.eta_ch * p_ch - (1.0 / fp.eta_dis) * p_dis))
    cons = [
        p_ch <= fp.C_ch, p_dis <= fp.C_dis,
        soc >= 0, soc <= fp.B_max,
        soc[T - 1] == fp.SOC0,                            # hard terminal equality
    ]

    net_draw = p_d + p_ch - p_dis                         # p^g with the TRUE realisation
    imb = net_draw - bid                                  # == 0 identically (bid pinned to net_draw)
    C_da = pi_da @ bid * fp.dt

    param_by_name = {"p_d": p_d, "pi_da": pi_da}
    if price_model == "single":
        pi_imb = cp.Parameter(T, name="pi_imb")
        param_by_name["pi_imb"] = pi_imb
        C_imb = pi_imb @ imb * fp.dt                      # signed (can be revenue)
    else:
        pi_imb_up = cp.Parameter(T, nonneg=True, name="pi_imb_up")
        pi_imb_down = cp.Parameter(T, nonneg=True, name="pi_imb_down")
        param_by_name["pi_imb_up"], param_by_name["pi_imb_down"] = pi_imb_up, pi_imb_down
        # plain solve -> DPP not required -> cp.pos/cp.neg is fine and clean.
        C_imb = (pi_imb_up @ cp.pos(imb) + pi_imb_down @ cp.neg(imb)) * fp.dt

    prob = cp.Problem(cp.Minimize(C_da + C_imb), cons)
    return OracleBundle(
        problem=prob,
        param_by_name=param_by_name,
        var_by_name={"p_ch": p_ch, "p_dis": p_dis, "bid": bid},
        price_model=price_model,
        objective="economic",
    )


def solve_oracle(bundle: OracleBundle, values: dict, solver=cp.GUROBI,
                 return_decisions: bool = False, **solver_kwargs):
    """Oracle objective is the FULL cost (no dropped constant -- bid is the actual bid)."""
    for name, p in bundle.param_by_name.items():
        if name not in values:
            raise KeyError(f"missing oracle param value: {name} "
                           f"(need {list(bundle.param_by_name)})")
        p.value = np.asarray(values[name], dtype=float)
    bundle.problem.solve(solver=solver, **solver_kwargs)
    if bundle.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"oracle solve status: {bundle.problem.status}")
    cost = float(bundle.problem.value)
    if return_decisions:
        return cost, {name: np.asarray(v.value) for name, v in bundle.var_by_name.items()}
    return cost


def oracle_realised_cost(fp, bundle: OracleBundle, decision_price_values: dict, realised,
                         price_model: str, true_pi_da, true_pi_imb=None,
                         true_pi_imb_up=None, true_pi_imb_down=None, solver=cp.GUROBI,
                         **solver_kwargs) -> float:
    """Solves the oracle under `decision_price_values` -- the price information actually
    available AT DECISION TIME (the _fc/proxy columns, matching what the policy itself
    is fed -- see dispatch_shared's decision-time price routing note) -- then re-settles
    that EXACT decision (p_ch, p_dis) against the TRUE market price via
    dispatch_wrapper.realised_cost, the same engine used for the policy's own reported
    cost. Necessary for a fair regret comparison: solving with fc prices and reporting
    solve_oracle's own objective value would benchmark the policy against an oracle that
    saw a DIFFERENT (clipped/proxy) day-ahead price than it actually settles at --
    exactly the apples-to-oranges mismatch dispatch_wrapper.py warns about for the
    policy side. p_da_bat/D_ch/D_dis are reconstructed the same way a genuinely
    recourse-free decision would be (bid pinned, zero recourse) since the oracle itself
    has no D_ch/D_dis mechanism to report."""
    _, dec = solve_oracle(bundle, decision_price_values, solver=solver,
                          return_decisions=True, **solver_kwargs)
    p_ch, p_dis = dec["p_ch"], dec["p_dis"]
    p_da_bat = p_ch - p_dis
    D_zero = np.zeros((fp.T_total, fp.T_total))
    cost = realised_cost(
        fp, p_ch, p_dis, D_zero, D_zero, p_da_bat,
        realised=realised, pl_hat=realised, price_model=price_model,
        pi_da=true_pi_da, pi_imb=true_pi_imb,
        pi_imb_up=true_pi_imb_up, pi_imb_down=true_pi_imb_down,
        clip_recourse=False,
    )
    return float(cost.item())
