from __future__ import annotations
import cvxpy as cp

from dispatch_shared import Variables, Variables1Stage

# =====================================================================================
# dispatch_objectives.py  --  every objective-function variant, selected by `mode`
# instead of the old (price_model, k) pair.
#
# mode in {"single-price", "dual-price", "dispatchability"}. "single-price" and
# "dual-price" map to their respective economic objectives (what price_model used to
# select, at k=0). "dispatchability" maps to the imbalance-tracking term (what k=1
# used to select) -- price-agnostic by construction: no price Parameters are declared
# in this mode at all, matching the old k=1 behaviour where price never entered the
# problem.
#
# Dropping the old (1-k)*A + k*B weighting entirely, not just fixing it at {0,1}: this
# study never actually blended the two terms (k was always exactly 0 or exactly 1), so
# that notation implied a tunable trade-off that never existed. Each mode below is now
# a clean, standalone objective -- there's no expression left that COULD be misread as
# a blend.
# =====================================================================================

MODES_2STAGE = ("single-price", "dual-price", "dispatchability")
MODES_1STAGE = ("single-price", "dual-price")


# =====================================================================================
# Shared (T,N) epigraph LP for the dual-price piecewise imbalance cost -- used
# identically by build_objective_2stage (R = the recourse-adjusted net matrix v.R) and
# build_objective_1stage (R=None). Declares xi_samples/pi_imb_up/pi_imb_down/p_plus/
# p_minus once; the only difference between callers is whether recourse modifies the
# raw scenario deviations before they hit the epigraph.
#
# R=None (1-stage) is NOT an approximation of R=I -- 1-stage genuinely has no recourse
# mechanism to represent, so there is no D_ch/D_dis-built R for it to pass in the first
# place (see dispatch_shared.Variables1Stage, which has no R field at all).
#
# No deterministic-offset argument here: point-robust, full-robust, and 1-stage all pin
# the bid, so the deterministic imbalance p_ch_hat - p_dis_hat - p_da_bat is always 0
# (see dispatch_shared.Variables' docstring -- removed entirely as a field rather than
# carried as an always-zero one), and imb_scen is purely R @ xi_samples.T (or
# xi_samples.T itself, with no R). If some future formulation ever leaves the bid free,
# that deterministic term would need reintroducing here explicitly, not silently
# assumed away.
#
# N=64 for 1-stage is mathematically INERT for the decision this LP feeds into: with no
# deterministic offset and no recourse, imb_scen == xi_samples.T is pure data (a
# Parameter), so p_plus/p_minus/C_imb never depend on any decision variable at all -- an additive
# constant in the 1-stage objective, changing neither its argmin nor the cvxpylayers
# gradient flowing back through it. Kept anyway so build_objective_1stage and
# build_objective_2stage's dual-price branches share this ONE implementation instead
# of two divergent ones -- the forecaster still gets its real 1-stage training signal
# from realised_breakdown downstream (true realised load, not this SAA), unaffected
# either way.
#
# p_plus/p_minus deliberately receive NO gamma regularization here, unlike
# p_ch_hat/p_dis_hat/D_ch/D_dis in the callers' own penalty block -- tried adding
# gamma*(sum_squares(p_plus)+sum_squares(p_minus)) (the natural extension of the same
# bounded-conservatism trade-off) and measured it directly: at both gamma=1e-4 and
# gamma=5e-3 the "inaccurate" ECOS flag rate on this corner was completely unchanged,
# while wall-clock time got WORSE (~17-28% slower), from the extra second-order-cone
# structure the sum_squares terms add over these 2*T*N=3072 variables. Reverted --
# see conversation/session notes on the gamma_sweep dual-price investigation for the
# measurements this is based on.
# =====================================================================================
def _build_dual_price_epigraph(fp, R=None):
    T, N, dt = fp.T_total, fp.num_scenarios, fp.dt

    xi_samples  = cp.Parameter((N, T), name="xi_samples")
    pi_imb_up   = cp.Parameter(T, nonneg=True, name="pi_imb_up")
    pi_imb_down = cp.Parameter(T, nonneg=True, name="pi_imb_down")
    p_plus  = cp.Variable((T, N), nonneg=True, name="p_plus")
    p_minus = cp.Variable((T, N), nonneg=True, name="p_minus")

    imb_scen = (R @ xi_samples.T) if R is not None else xi_samples.T   # (T, N)
    epigraph_cons = [p_plus - p_minus == imb_scen]
    # Net (standard two-price) settlement, matching dispatch_wrapper.realised_breakdown's
    # identical fix: short (imb>0) pays pi_up, long (imb<0) is CREDITED at pi_down, not
    # charged again. Still DPP-compliant and bounded -- p_plus/p_minus stay pure
    # Parameter-free Variables (only p_plus-p_minus is tied to the parameter-bearing
    # imb_scen, via the equality constraint above, not by direct multiplication), and
    # substituting that equality in shows the effective coefficient on the free variable
    # p_minus is (pi_imb_up - pi_imb_down) >= 0 always (pi_imb_up = max(da,imb) >=
    # min(da,imb) = pi_imb_down by construction) -- bounded below, minimized at the
    # natural non-negative split p_plus=(imb)+, p_minus=(imb)-, same mechanism as
    # before, just yielding the net rather than the gross value there.
    C_imb = cp.sum(pi_imb_up @ p_plus - pi_imb_down @ p_minus) * dt / N

    params = [xi_samples, pi_imb_up, pi_imb_down]
    return C_imb, epigraph_cons, params


def build_objective_2stage(fp, mode: str, v: Variables):
    """Shared, unchanged, between point-robust and full-robust -- only ever consumes
    v.R, never how it was made feasible.

    Returns (objective_expression, params, epigraph_constraints). epigraph_constraints
    is only ever non-empty for "dual-price" (the p_plus/p_minus split needed to express
    a piecewise-linear cost as an LP) -- see the point-robust setup's module note on
    why that constraint belongs here, not in dispatch_setup.py."""
    if mode not in MODES_2STAGE:
        raise ValueError(f"mode must be one of {MODES_2STAGE}, got {mode!r}")
    T, N, dt = fp.T_total, fp.num_scenarios, fp.dt

    params: list = []
    epigraph_cons: list = []
    obj_terms: list = []

    if mode in ("single-price", "dual-price"):
        pi_da = cp.Parameter(T, name="pi_da")
        params.append(pi_da)
        C_da = pi_da @ v.p_da_bat * dt   # DA constant pi_da.pl_hat.dt dropped -- added back downstream

        if mode == "single-price":
            # E[pi_imb . p_imb] = pi_imb . (deterministic imbalance) = 0: the R.xi term
            # vanishes in expectation (E[xi]=0), and the deterministic imbalance
            # p_ch_hat - p_dis_hat - p_da_bat is 0 by the pinned bid.  C_imb is
            # identically 0 regardless of pi_imb's value, and pi_imb is never declared
            # as a Parameter here at all (nothing left for it to multiply). Settlement
            # still needs the real pi_imb value -- that's realised_breakdown's job, a
            # separate, non-cvxpy computation -- this is the DECISION solve only.
            C_imb = 0.0
        else:  # dual-price
            C_imb, dual_epigraph_cons, dual_params = _build_dual_price_epigraph(fp, R=v.R)
            epigraph_cons += dual_epigraph_cons
            params += dual_params

        obj_terms.append(C_da + C_imb)
    # ---- PENALTY (always present, though set to 0 for evaluation) -----------------------
    penalty = fp.gamma * (
        cp.sum_squares(v.p_ch_hat) + cp.sum_squares(v.p_dis_hat)
        + cp.sum_squares(v.D_ch) + cp.sum_squares(v.D_dis)
    )
    obj_terms.append(penalty)

    return sum(obj_terms), params, epigraph_cons


def build_objective_1stage(fp, mode: str, v: Variables1Stage):
    """1-stage's own -- single-price/C_da/penalty are NOT shared with
    build_objective_2stage; dual-price's epigraph IS, via _build_dual_price_epigraph
    (R=None here, since there's no recourse to build v.R from). There's no D_ch/D_dis
    to build a tracking term from or to penalise, so "dispatchability" is not a valid
    mode here; it raises rather than silently falling through to something else.

    Returns the same 3-tuple shape as build_objective_2stage for a consistent calling
    convention, even though there's no k/tracking branch to gate internally."""
    if mode == "dispatchability":
        raise ValueError(
            "1-stage has no recourse mechanism (no D_ch/D_dis), so there is nothing to "
            "build a dispatchability/tracking term FROM -- this mode only exists for "
            "the 2-stage formulations (point-robust, full-robust). Use 'single-price' "
            "or 'dual-price'.")
    if mode not in MODES_1STAGE:
        raise ValueError(f"mode must be one of {MODES_1STAGE} for 1-stage, got {mode!r}")

    T, dt = fp.T_total, fp.dt
    params: list = []
    epigraph_cons: list = []

    pi_da = cp.Parameter(T, name="pi_da")
    params.append(pi_da)
    C_da = pi_da @ v.p_da_bat * dt

    if mode == "single-price":
        # SIGNED settlement -- exact, no epigraph needed: the deterministic imbalance
        # p_ch_hat - p_dis_hat - p_da_bat is a plain (T,) vector here, no scenario/xi
        # term to average over (unlike the 2-stage case, where this identity only holds
        # in EXPECTATION over xi) -- and it's 0 by the pinned bid, so C_imb is
        # identically 0 regardless of pi_imb's value; pi_imb is never declared as a
        # Parameter here at all. Settlement still needs the real pi_imb value -- that's
        # realised_breakdown's job, a separate, non-cvxpy computation.
        C_imb = 0.0
    else:  # dual-price
        C_imb, dual_epigraph_cons, dual_params = _build_dual_price_epigraph(fp, R=None)
        epigraph_cons += dual_epigraph_cons
        params += dual_params

    penalty = fp.gamma * (cp.sum_squares(v.p_ch_hat) + cp.sum_squares(v.p_dis_hat)
                           + cp.sum_squares(v.p_da_bat))

    return C_da + C_imb + penalty, params, epigraph_cons
