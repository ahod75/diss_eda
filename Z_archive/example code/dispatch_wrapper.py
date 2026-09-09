"""
Bridge between the copula sampler and the robust dispatch layer, plus the realised-cost /
regret evaluation used as the DFL objective.

Two jobs:

  1. make_dispatch_inputs(...)  — turn a day's quantile forecast into the six things the
     layer consumes, all MEAN-anchored and mutually consistent:
        xi_samples : (N, T) mean-centred error scenarios          (grad -> forecaster)
        pl_hat     : (T,)   mean forecast  (used OUTSIDE the layer, in the DA cost)
        pi_da, lam_up, lam_dn : (T,) prices from the window loader
        h_plus, h_minus : (T,) robust-box half-widths, taken from specific forecast
                          quantiles but EXPRESSED AGAINST THE MEAN (detached by default)

  2. realised_cost(...) / regret(...) — the true economic cost of the layer's decision at
     the realised prosumption (differentiable in the forecaster), and the regret against a
     perfect-foresight oracle (oracle detached -> contributes value, not gradient).

Sign / anchor conventions (see the derivation):
  * p_imb = (I + D_ch - D_dis) @ xi     ('+' sign; pl_hat cancels from the imbalance)
  * feeder-side  p~_imb = p_d - p_bid   (positive = short -> upward regulation lam_up)
  * xi is mean-centred:  xi = scenario - mean,  and pl_hat = mean, so realised
    xi_real = realised - pl_hat.
"""
from __future__ import annotations
from dataclasses import dataclass
import warnings

import numpy as np
import torch
import cvxpy as cp

from dispatch_layer import FixedParams


# --------------------------------------------------------------------------
def _to_t(a, like=None, dtype=torch.double):
    """Coerce numpy/torch/scalar to a torch tensor (double by default)."""
    if isinstance(a, torch.Tensor):
        t = a.to(dtype)
    else:
        t = torch.as_tensor(np.asarray(a), dtype=dtype)
    if like is not None:
        t = t.to(like.device)
    return t


@dataclass
class DispatchInputs:
    xi_samples: torch.Tensor   # (N, T)  mean-centred error scenarios   (requires grad)
    pl_hat:     torch.Tensor   # (T,)    mean forecast                   (requires grad)
    pi_da:      torch.Tensor   # (T,)
    lam_up:     torch.Tensor   # (T,)
    lam_dn:     torch.Tensor   # (T,)
    h_plus:     torch.Tensor   # (T,)    box half-width up   (detached by default)
    h_minus:    torch.Tensor   # (T,)    box half-width down (detached by default)

    def layer_args(self):
        """Positional tensors for CvxpyLayer, in the order build_layer expects:
        [xi_samples, pi_da, lam_up, lam_dn, h_plus, h_minus]."""
        return (self.xi_samples, self.pi_da, self.lam_up, self.lam_dn,
                self.h_plus, self.h_minus)


# --------------------------------------------------------------------------
def make_dispatch_inputs(
    quantiles,                 # (K, Q) monotone quantiles for ONE day, physical MW, torch
    sampler,                   # FrozenCopulaSampler  (mean_and_errors)
    quantile_levels,           # (Q,)  the forecaster's levels (e.g. 0.05..0.95)
    pi_da,                     # (T,)  day-ahead prices        (from window loader)
    lam_up,                    # (T,)  lambda^up  >= 0          (pre-saved, from loader)
    lam_dn,                    # (T,)  lambda^dn  >= 0          (pre-saved, from loader)
    box_levels=(0.05, 0.95),   # (lower, upper) forecast quantiles defining the box edges
    box_detach=True,           # detach h_plus/h_minus so DFL cannot game the robustness margin
    min_box=1e-4,              # floor on box half-widths (guards extreme-skew negatives)
):
    """Build mean-anchored, mutually-consistent dispatch inputs from a day's quantiles.

    The box edges come from *specific quantiles* (box_levels) but are expressed as offsets
    from the MEAN, so the box lives in the same xi = (realised - mean) coordinate as the
    error scenarios and the imbalance:
        h_plus  = q_upper - mean      (how far up   xi may reach)
        h_minus = mean    - q_lower   (how far down xi may reach)
    """
    if quantiles.dim() != 2:
        raise ValueError("quantiles must be (K, Q) for a single day")

    # mean forecast and mean-centred error scenarios, from ONE prosumption pass
    mean, xi = sampler.mean_and_errors(quantiles)      # mean (K,), xi (S, K) = (N, T)
    pl_hat = mean                                       # (T,) anchor; grad kept

    # locate the box-edge quantiles in the level grid
    levels = np.asarray(quantile_levels, float)
    i_lo = int(np.argmin(np.abs(levels - box_levels[0])))
    i_hi = int(np.argmin(np.abs(levels - box_levels[1])))
    q_lower = quantiles[:, i_lo]                        # (T,)
    q_upper = quantiles[:, i_hi]                        # (T,)

    # box half-widths expressed against the MEAN
    h_plus  = q_upper - mean                            # (T,)
    h_minus = mean - q_lower                            # (T,)

    # guard: extreme lower/upper-tail skew can (rarely) push the mean outside [q_lo, q_hi],
    # which would make a half-width negative. Clamp and warn -- a non-positive half-width
    # means the box is degenerate on that side.
    with torch.no_grad():
        if (h_plus < 0).any() or (h_minus < 0).any():
            warnings.warn(
                "mean fell outside [q_lower, q_upper] on some lead(s) (extreme skew); "
                "clamping box half-width to min_box. Consider wider box_levels.")
    h_plus  = torch.clamp(h_plus,  min=min_box)
    h_minus = torch.clamp(h_minus, min=min_box)

    if box_detach:
        # robustness margin is a per-day spec derived from the forecast, but NOT a quantity
        # the decision loss should optimise away -> detach so no gradient flows through h.
        h_plus  = h_plus.detach()
        h_minus = h_minus.detach()

    return DispatchInputs(
        xi_samples=xi,                                  # (N, T), grad
        pl_hat=pl_hat,                                  # (T,),   grad
        pi_da=_to_t(pi_da,  like=xi),
        lam_up=_to_t(lam_up, like=xi),
        lam_dn=_to_t(lam_dn, like=xi),
        h_plus=_to_t(h_plus,  like=xi),
        h_minus=_to_t(h_minus, like=xi),
    )


# --------------------------------------------------------------------------
def realised_cost(
    fp: FixedParams,
    p_ch_hat, p_dis_hat, D_ch, D_dis,   # decision tensors from the layer (grad)
    realised,                           # (T,) realised prosumption for the day
    pl_hat,                             # (T,) mean forecast (same anchor as inputs)
    pi_da, lam_up, lam_dn,              # (T,) prices
    clip_recourse=False,                # clip battery actions to physical limits (kinks)
):
    """True economic cost of the layer's decision at the realised prosumption.

    Differentiable in the decision variables and pl_hat (hence in the forecaster). This is
    the DFL loss; regret() subtracts the (detached) oracle.

    Note: the pi_da @ pl_hat term that was DROPPED from the layer objective is added back
    here (it is part of the real DA cost, just not decision-relevant).
    """
    dev = p_ch_hat.device
    T = fp.T_total
    I_T = torch.eye(T, dtype=p_ch_hat.dtype, device=dev)

    realised = _to_t(realised, like=p_ch_hat)
    pl_hat   = _to_t(pl_hat,   like=p_ch_hat)
    pi_da    = _to_t(pi_da,    like=p_ch_hat)
    lam_up   = _to_t(lam_up,   like=p_ch_hat)
    lam_dn   = _to_t(lam_dn,   like=p_ch_hat)

    xi_real = realised - pl_hat                         # (T,) mean-centred realised error

    # realised recourse actions (optionally saturated to physical limits)
    p_ch_r  = p_ch_hat  + D_ch  @ xi_real               # (T,)
    p_dis_r = p_dis_hat + D_dis @ xi_real
    if clip_recourse:
        p_ch_r  = torch.clamp(p_ch_r,  min=0.0, max=fp.C_ch)
        p_dis_r = torch.clamp(p_dis_r, min=0.0, max=fp.C_dis)

    # committed day-ahead bid (here-and-now decisions; recourse is not in the bid)
    bid = pl_hat + p_ch_hat - p_dis_hat                 # (T,)

    # realised imbalance:  p_imb = (I + D_ch - D_dis) @ xi_real
    # (if clipping, recompute from the saturated actions so imbalance reflects saturation)
    if clip_recourse:
        # net realised draw minus committed bid
        net_draw = realised + p_ch_r - p_dis_r
        p_imb = net_draw - bid
    else:
        p_imb = (I_T + D_ch - D_dis) @ xi_real          # (T,)

    C_da  = (pi_da * bid).sum() * fp.dt                 # includes pi_da @ pl_hat
    C_imb = (lam_up * torch.clamp(p_imb, min=0.0)
             + lam_dn * torch.clamp(-p_imb, min=0.0)).sum() * fp.dt
    return C_da + C_imb                                 # scalar, grad -> forecaster


# --------------------------------------------------------------------------
def build_oracle(fp: FixedParams):
    """Perfect-foresight deterministic dispatch (parametric, built once).

    Knows the realised prosumption exactly, so it sets the bid = realised net draw ->
    zero imbalance (optimal, since C_imb >= 0), and optimises battery arbitrage against
    the DA price subject to the same operational limits. Returns a closure solve(p_d,
    pi_da) -> oracle economic cost (float). Detached from the forecast entirely.
    """
    T = fp.T_total
    p_ch  = cp.Variable(T, nonneg=True)
    p_dis = cp.Variable(T, nonneg=True)
    p_d    = cp.Parameter(T)
    pi_da  = cp.Parameter(T)

    L = np.tril(np.ones((T, T)))
    soc = fp.SOC0 + fp.dt * (L @ (fp.eta_ch * p_ch - (1.0 / fp.eta_dis) * p_dis))
    cons = [p_ch <= fp.C_ch, p_dis <= fp.C_dis, soc >= 0, soc <= fp.B_max]
    if fp.terminal_soc_equality:
        cons += [soc[T - 1] == fp.SOC0]
    else:
        cons += [soc[T - 1] >= fp.SOC0]

    net_draw = p_d + p_ch - p_dis                       # bid = realised net draw
    prob = cp.Problem(cp.Minimize(pi_da @ net_draw * fp.dt), cons)

    def solve(p_d_val, pi_da_val, solver=cp.CLARABEL):
        p_d.value = np.asarray(p_d_val, float)
        pi_da.value = np.asarray(pi_da_val, float)
        prob.solve(solver=solver)
        return float(prob.value)

    return solve


def regret(
    fp: FixedParams,
    p_ch_hat, p_dis_hat, D_ch, D_dis,
    realised, pl_hat, pi_da, lam_up, lam_dn,
    oracle_solve,                       # closure from build_oracle
    clip_recourse=False,
):
    """Regret = realised_cost(forecast decision) - oracle_cost.

    The oracle term is a detached constant (perfect foresight, no forecast dependence), so
    the gradient of regret w.r.t. the forecaster equals the gradient of realised_cost. Use
    realised_cost as the DFL loss; regret is the reported, interpretable metric (>= 0).
    """
    cost_fcst = realised_cost(
        fp, p_ch_hat, p_dis_hat, D_ch, D_dis,
        realised, pl_hat, pi_da, lam_up, lam_dn, clip_recourse=clip_recourse,
    )
    with torch.no_grad():
        oracle_cost = oracle_solve(_to_t(realised).cpu().numpy(),
                                   _to_t(pi_da).cpu().numpy())
    return cost_fcst - oracle_cost      # scalar; grad only through cost_fcst


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # minimal shape/logic smoke test with stand-in objects (no forecaster/cvxpy solve here)
    print("dispatch_wrapper: import OK")
    T, Q, S = 6, 19, 40
    levels = np.round(np.arange(0.05, 0.96, 0.05), 2)

    # fake monotone quantiles (K, Q)
    base = torch.linspace(1.0, 3.0, T).unsqueeze(1)
    incr = torch.linspace(-0.5, 0.5, Q).unsqueeze(0)
    quantiles = (base + incr).clone().requires_grad_(True)     # (T, Q), monotone in Q

    class _FakeSampler:
        """stand-in: scenarios = median + spread*(u-0.5), mean != median under skew."""
        def mean_and_errors(self, q):
            med = q[:, Q // 2]                                  # (T,)
            # skew the scenarios so mean != median
            draws = torch.linspace(-1, 1, S).unsqueeze(1)       # (S,1)
            spread = (q[:, -1] - q[:, 0]).unsqueeze(0)          # (1,T)
            p = med.unsqueeze(0) + 0.5 * spread * draws + 0.1 * spread * draws**2
            mean = p.mean(0)                                    # (T,)
            return mean, p - mean.unsqueeze(0)                  # (T,), (S,T)

    fp = FixedParams(T_total=T, num_scenarios=S, dt=1.0, eta_ch=0.95, eta_dis=0.95,
                     C_ch=2.0, C_dis=2.0, B_max=6.0, SOC0=3.0, gamma=1e-3)

    di = make_dispatch_inputs(
        quantiles, _FakeSampler(), levels,
        pi_da=np.array([50., 80., 120., 60., 90., 40.]),
        lam_up=np.array([20., 10., 0., 40., 5., 30.]),
        lam_dn=np.array([0., 0., 15., 0., 25., 0.]),
    )
    print("xi_samples:", tuple(di.xi_samples.shape), "| pl_hat:", tuple(di.pl_hat.shape))
    print("h_plus>=0:", bool((di.h_plus >= 0).all()),
          "| h_minus>=0:", bool((di.h_minus >= 0).all()),
          "| box detached:", not di.h_plus.requires_grad)
    print("xi mean ~0 per lead:", float(di.xi_samples.mean(0).abs().max()))

    # realised_cost differentiability (fake decisions from quantiles so grad has a path)
    p_ch_hat  = torch.zeros(T, dtype=torch.double, requires_grad=True)
    p_dis_hat = torch.zeros(T, dtype=torch.double, requires_grad=True)
    D_ch  = torch.zeros((T, T), dtype=torch.double, requires_grad=True)
    D_dis = torch.zeros((T, T), dtype=torch.double, requires_grad=True)
    realised = torch.linspace(1.2, 2.8, T, dtype=torch.double)
    c = realised_cost(fp, p_ch_hat, p_dis_hat, D_ch, D_dis, realised,
                      di.pl_hat.double(), di.pi_da, di.lam_up, di.lam_dn)
    c.backward()
    print("realised_cost:", float(c),
          "| grad reaches pl_hat via quantiles:", quantiles.grad is not None)
