"""
Robust dispatchable-feeder layer — DPP-compliant, differentiable (cvxpylayers).

Formulation (Option 1):
  * imbalance cost via the epigraph split  C_imb = lam_up @ p_plus + lam_dn @ p_minus,
    with  p_plus - p_minus == p_imb,  p_plus,p_minus >= 0.
    Single-price is the symmetric case lam_up == lam_dn; dual-price is lam_up != lam_dn.
  * quadratic tracking term DROPPED (relies on gamma-Tikhonov for strict convexity / a
    unique, differentiable solution — see the note on k below).
  * DA cost keeps only the variable part  pi_da @ (p_ch_hat - p_dis_hat) * dt
    inside the layer; the constant  pi_da @ pl_hat * dt  is added OUTSIDE (it does not
    affect the argmin and pi_da @ pl_hat would be a parameter*parameter DPP violation).

Imbalance sign (derived):  p_imb = (I + D_ch - D_dis) @ xi
  Feeder-side convention  p~_imb = p_d - p_bid  (positive = short).  The here-and-now
  battery decisions cancel against the committed bid; only the recourse survives, with a
  '+' on (D_ch - D_dis).  pl_hat (the mean forecast) cancels out of the imbalance entirely
  and appears only in the (out-of-layer) DA constant.

Scenario convention:  xi_samples is (N, T)  (N scenarios, T leads) — matches the copula
sampler's (S, K).  Internally we use xi.T -> (T, N).

Robust box:  operational limits hold for all xi in [-h_minus, +h_plus] per lead, enforced
by the standard dual certificate (h_plus/h_minus are the box half-widths, both >= 0).
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product

import numpy as np
import cvxpy as cp

try:
    from cvxpylayers.torch import CvxpyLayer
    _HAVE_CVXPYLAYERS = True
except Exception:
    _HAVE_CVXPYLAYERS = False


# --------------------------------------------------------------------------
# Fixed (structural) parameters — baked into the problem at build time.
# Everything that varies per window is a cp.Parameter, not in here.
# --------------------------------------------------------------------------
@dataclass
class FixedParams:
    T_total: int          # number of lead periods T
    num_scenarios: int    # N — number of scenarios fed to the objective
    dt: float             # length of a time step
    eta_ch: float         # charging efficiency
    eta_dis: float        # discharging efficiency
    C_ch: float           # max charge rate
    C_dis: float          # max discharge rate
    B_max: float          # battery energy capacity
    SOC0: float           # initial == terminal state of charge
    gamma: float = 1e-3   # Tikhonov coefficient (raised from 1e-6: it is now the ONLY
                          # source of strong convexity, since the quadratic is dropped)
    terminal_soc_equality: bool = True


# --------------------------------------------------------------------------
# Robust box constraint via dual certificate.
# Enforce  a0 + a^T xi <= b   for all xi in [-h_minus, +h_plus], by adding
#   mu_p, mu_m >= 0 ,  mu_p - mu_m == a ,  h_plus^T mu_p + h_minus^T mu_m <= b - a0.
# a0 : scalar expr (variable-dependent) ; a : (T,) expr ; b : scalar constant.
# h_plus / h_minus : (T,) cp.Parameter (>= 0).  Parameter*variable => affine => DPP-ok.
# --------------------------------------------------------------------------
def _robustify(a0, a, b, T, h_plus, h_minus):
    mu_p = cp.Variable(T, nonneg=True)
    mu_m = cp.Variable(T, nonneg=True)
    return [
        mu_p - mu_m == a,
        h_plus @ mu_p + h_minus @ mu_m <= b - a0,
    ]


def build_layer(fp: FixedParams):
    """Build the parametric problem ONCE and wrap it as a differentiable CvxpyLayer.

    Returns (layer, problem, param_order) where param_order documents the positional
    order the layer expects its parameter tensors in.
    """
    T = fp.T_total
    N = fp.num_scenarios

    # ---- decision variables (here-and-now + linear recourse) ----
    p_ch_hat  = cp.Variable(T, name="p_ch_hat")
    p_dis_hat = cp.Variable(T, name="p_dis_hat")
    D_ch      = cp.Variable((T, T), name="D_ch")
    D_dis     = cp.Variable((T, T), name="D_dis")

    # ---- epigraph split of the (per-scenario) imbalance ----
    p_plus  = cp.Variable((T, N), nonneg=True, name="p_plus")
    p_minus = cp.Variable((T, N), nonneg=True, name="p_minus")

    # ---- parameters (vary per window; the differentiable interface) ----
    xi_samples = cp.Parameter((N, T), name="xi_samples")   # (N, T) scenarios
    pi_da      = cp.Parameter(T, name="pi_da")
    lam_up     = cp.Parameter(T, nonneg=True, name="lam_up")   # lambda^up  (>= 0)
    lam_dn     = cp.Parameter(T, nonneg=True, name="lam_dn")   # lambda^dn  (>= 0)
    h_plus     = cp.Parameter(T, nonneg=True, name="h_plus")
    h_minus    = cp.Parameter(T, nonneg=True, name="h_minus")
    # NB: pl_hat is NOT a layer parameter — it cancels from p_imb and its only appearance
    #     (pi_da @ pl_hat) is an out-of-layer constant.  Keeping it out avoids a
    #     parameter*parameter DPP violation.

    cons = []

    # ---- non-anticipativity: D lower-triangular (strict upper == 0) ----
    strictly_upper = np.triu(np.ones((T, T)), 1).astype(bool)
    cons += [D_ch[strictly_upper] == 0, D_dis[strictly_upper] == 0]

    # ---- SOC as an affine function of xi:  p_soc(xi) = s_hat + G @ xi ----
    # s_hat[t] = SOC0 + dt * sum_{tau<=t} (eta_ch*p_ch_hat - (1/eta_dis)*p_dis_hat)_tau
    # G[t,:]   = dt * sum_{tau<=t} (eta_ch*D_ch - (1/eta_dis)*D_dis)_tau
    L = np.tril(np.ones((T, T)))
    power_flow_hat = fp.eta_ch * p_ch_hat - (1.0 / fp.eta_dis) * p_dis_hat     # (T,)
    s_hat = fp.SOC0 + fp.dt * (L @ power_flow_hat)                              # (T,)
    D_net = fp.eta_ch * D_ch - (1.0 / fp.eta_dis) * D_dis                       # (T,T)
    G = fp.dt * (L @ D_net)                                                     # (T,T)

    # ---- robust operational constraints (hold for all xi in the box) ----
    for t in range(T):
        d_ch  = D_ch[t, :]        # (T,) coefficient vector on xi for charge at period t
        d_dis = D_dis[t, :]
        g_t   = G[t, :]
        # 0 <= p_ch(xi)  <= C_ch
        cons += _robustify( p_ch_hat[t],   d_ch,  fp.C_ch, T, h_plus, h_minus)
        cons += _robustify(-p_ch_hat[t],  -d_ch,  0.0,     T, h_plus, h_minus)
        # 0 <= p_dis(xi) <= C_dis
        cons += _robustify( p_dis_hat[t],  d_dis, fp.C_dis, T, h_plus, h_minus)
        cons += _robustify(-p_dis_hat[t], -d_dis, 0.0,      T, h_plus, h_minus)
        # 0 <= p_soc(xi) <= B_max
        cons += _robustify( s_hat[t],  g_t, fp.B_max, T, h_plus, h_minus)
        cons += _robustify(-s_hat[t], -g_t, 0.0,      T, h_plus, h_minus)

    # ---- terminal SOC ----
    if fp.terminal_soc_equality:
        cons += [s_hat[T - 1] == fp.SOC0, G[T - 1, :] == 0]   # energy-neutral recourse
    else:
        cons += _robustify(-s_hat[T - 1], -G[T - 1, :], -fp.SOC0, T, h_plus, h_minus)
    
    # ---- imbalance (per scenario) and its epigraph split ----
    # p_imb = (I + D_ch - D_dis) @ xi   (derived; '+' sign, pl_hat cancels)
    I_T = np.eye(T)
    recourse = I_T + D_ch - D_dis                    # (T, T)
    p_imb = recourse @ xi_samples.T                  # (T, N)   xi is (N, T)
    cons += [p_plus - p_minus == p_imb]              # affine in xi  => DPP-ok

    # ---- objective (Option 1: no quadratic tracking term) ----
    # DA: keep only the variable part; pi_da @ pl_hat added OUTSIDE the layer.
    C_da = pi_da @ (p_ch_hat - p_dis_hat) * fp.dt
    # imbalance: sample-average of the split cost
    C_imb = cp.sum(lam_up @ p_plus + lam_dn @ p_minus) * fp.dt / N
    penalty = fp.gamma * (
        cp.sum_squares(p_ch_hat) + cp.sum_squares(p_dis_hat)
        + cp.sum_squares(D_ch) + cp.sum_squares(D_dis)
    )
    obj = cp.Minimize(C_da + C_imb + penalty)

    prob = cp.Problem(obj, cons)
    assert prob.is_dcp(dpp=True), "problem is not DPP — cvxpylayers will reject it"

    param_order = [xi_samples, pi_da, lam_up, lam_dn, h_plus, h_minus]
    variables_out = [p_ch_hat, p_dis_hat, D_ch, D_dis]

    layer = None
    if _HAVE_CVXPYLAYERS:
        layer = CvxpyLayer(prob, parameters=param_order, variables=variables_out)

    handles = dict(
        prob=prob, layer=layer,
        p_ch_hat=p_ch_hat, p_dis_hat=p_dis_hat, D_ch=D_ch, D_dis=D_dis,
        p_plus=p_plus, p_minus=p_minus,
        xi_samples=xi_samples, pi_da=pi_da, lam_up=lam_up, lam_dn=lam_dn,
        h_plus=h_plus, h_minus=h_minus,
        param_order=["xi_samples", "pi_da", "lam_up", "lam_dn", "h_plus", "h_minus"],
    )
    return handles


# --------------------------------------------------------------------------
# Vertex-enumeration oracle (correctness check, small T only).
# Exact: affine constraints on a box are tight at the vertices.
# Uses the SAME epigraph objective so objective values are directly comparable.
# --------------------------------------------------------------------------
def build_vertex(fp: FixedParams, xi_val, pi_da_val, lam_up_val, lam_dn_val,
                 h_plus_val, h_minus_val):
    T, N = fp.T_total, fp.num_scenarios
    p_ch_hat  = cp.Variable(T)
    p_dis_hat = cp.Variable(T)
    D_ch      = cp.Variable((T, T))
    D_dis     = cp.Variable((T, T))
    p_plus  = cp.Variable((T, N), nonneg=True)
    p_minus = cp.Variable((T, N), nonneg=True)

    cons = []
    strictly_upper = np.triu(np.ones((T, T)), 1).astype(bool)
    cons += [D_ch[strictly_upper] == 0, D_dis[strictly_upper] == 0]

    L = np.tril(np.ones((T, T)))
    s_hat = fp.SOC0 + fp.dt * (L @ (fp.eta_ch * p_ch_hat - (1.0 / fp.eta_dis) * p_dis_hat))
    G = fp.dt * (L @ (fp.eta_ch * D_ch - (1.0 / fp.eta_dis) * D_dis))

    corners = [np.array(c) for c in product(
        *[(h_plus_val[i], -h_minus_val[i]) for i in range(T)])]
    for xi in corners:
        p_ch_r  = p_ch_hat + D_ch @ xi
        p_dis_r = p_dis_hat + D_dis @ xi
        soc_r   = s_hat + G @ xi
        cons += [p_ch_r >= 0, p_ch_r <= fp.C_ch,
                 p_dis_r >= 0, p_dis_r <= fp.C_dis,
                 soc_r >= 0, soc_r <= fp.B_max]
        if fp.terminal_soc_equality:
            cons += [soc_r[T - 1] == fp.SOC0]
        else:
            cons += [soc_r[T - 1] >= fp.SOC0]

    I_T = np.eye(T)
    p_imb = (I_T + D_ch - D_dis) @ xi_val.T          # (T, N)
    cons += [p_plus - p_minus == p_imb]

    C_da = pi_da_val @ (p_ch_hat - p_dis_hat) * fp.dt
    C_imb = cp.sum(lam_up_val @ p_plus + lam_dn_val @ p_minus) * fp.dt / N
    penalty = fp.gamma * (cp.sum_squares(p_ch_hat) + cp.sum_squares(p_dis_hat)
                          + cp.sum_squares(D_ch) + cp.sum_squares(D_dis))
    obj = cp.Minimize(C_da + C_imb + penalty)
    return cp.Problem(obj, cons), (p_ch_hat, p_dis_hat, D_ch, D_dis)


# --------------------------------------------------------------------------
# Tests / smoke check
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, N = 6, 40
    fp = FixedParams(
        T_total=T, num_scenarios=N, dt=1.0,
        eta_ch=0.95, eta_dis=0.95, C_ch=2.0, C_dis=2.0,
        B_max=6.0, SOC0=3.0, gamma=1e-3, terminal_soc_equality=True,
    )

    # sample parameter values
    xi_val   = np.clip(rng.normal(0.0, 0.3, size=(N, T)), -0.8, 0.8)   # (N, T), ~zero-mean
    pi_da_v  = np.array([50., 80., 120., 60., 90., 40.])
    # dual-price: lam_up >= 0, lam_dn >= 0, complementary in spirit
    lam_up_v = np.array([20., 10.,  0., 40.,  5., 30.])
    lam_dn_v = np.array([ 0.,  0., 15.,  0., 25.,  0.])
    h_plus_v  = np.full(T, 0.8)
    h_minus_v = np.full(T, 0.8)

    # ---- 1) build the parametric layer + DPP check ----
    H = build_layer(fp)
    prob = H["prob"]
    print("is_dcp:", prob.is_dcp(), "| is_dpp:", prob.is_dcp(dpp=True))

    # assign parameter values and solve the *cvxpy* problem directly (no torch needed)
    H["xi_samples"].value = xi_val
    H["pi_da"].value = pi_da_v
    H["lam_up"].value = lam_up_v
    H["lam_dn"].value = lam_dn_v
    H["h_plus"].value = h_plus_v
    H["h_minus"].value = h_minus_v
    prob.solve(solver=cp.CLARABEL)
    print(f"robust layer : {prob.status}, obj = {prob.value:.8f}")

    # ---- 2) vertex oracle on the SAME data ----
    prob_v, _ = build_vertex(fp, xi_val, pi_da_v, lam_up_v, lam_dn_v, h_plus_v, h_minus_v)
    prob_v.solve(solver=cp.CLARABEL)
    print(f"vertex oracle: {prob_v.status}, obj = {prob_v.value:.8f}")
    print(f"GAP          : {abs(prob.value - prob_v.value):.3e}  <-- must be ~0")

    # ---- 3) single-price is the symmetric special case ----
    pi_imb = np.array([70., 90., 40., 100., 55., 85.])
    H["lam_up"].value = pi_imb
    H["lam_dn"].value = pi_imb          # lam_up == lam_dn  => symmetric penalty
    prob.solve(solver=cp.CLARABEL)
    print(f"single-price : {prob.status}, obj = {prob.value:.8f} "
          f"(lam_up == lam_dn == pi_imb)")


    # ---- 4) differentiability (only if cvxpylayers + torch present) ----
    if _HAVE_CVXPYLAYERS:
        import torch
        layer = H["layer"]
        xi_t   = torch.tensor(xi_val, dtype=torch.double, requires_grad=True)
        pida_t = torch.tensor(pi_da_v, dtype=torch.double)
        lu_t   = torch.tensor(lam_up_v, dtype=torch.double)
        ld_t   = torch.tensor(lam_dn_v, dtype=torch.double)
        hp_t   = torch.tensor(h_plus_v, dtype=torch.double)
        hm_t   = torch.tensor(h_minus_v, dtype=torch.double)
        p_ch, p_dis, Dch, Ddis = layer(xi_t, pida_t, lu_t, ld_t, hp_t, hm_t)
        (p_ch.sum() + p_dis.sum()).backward()
        print("d(decisions)/d(xi) finite & nonzero:",
              bool(torch.isfinite(xi_t.grad).all()) and float(xi_t.grad.abs().sum()) > 0)
    else:
        print("cvxpylayers/torch not present here — run the layer() gradient test in your env")
