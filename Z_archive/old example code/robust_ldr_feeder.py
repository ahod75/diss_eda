"""
Robust counterpart of the LDR dispatchable-feeder problem (single imbalance price).

Implements the duality-based reformulation of the semi-infinite constraints:

    a0(Theta) + a(Theta)^T xi <= b   for all xi in Xi' = {xi : H xi <= h}
      <=>  exists mu >= 0 :  H^T mu = a(Theta),   h^T mu <= b - a0(Theta)

with H = [I; -I], h = [h+; h-]  (i.e. the box  -h- <= xi <= h+ ).

`build_robust()` is the model you want. `build_vertex()` re-solves the same problem
by brute-force enumeration of the 2^T corners of the box and is the correctness
oracle -- for small T the two optimal values must agree to solver tolerance.

Requires: cvxpy, numpy.
"""
from __future__ import annotations
from itertools import product
from dataclasses import dataclass, field

import numpy as np
import cvxpy as cp


# --------------------------------------------------------------------------
# Problem data
# --------------------------------------------------------------------------
@dataclass
class Params:
    T: int
    eta_ch: float          # charging efficiency
    eta_dis: float         # discharging efficiency
    C_ch: float            # max charge rate
    C_dis: float           # max discharge rate
    B_max: float           # battery energy capacity
    SOC0: float            # initial == terminal state of charge
    pl_hat: np.ndarray     # (T,) expected prosumption
    pi_da: np.ndarray      # (T,) day-ahead price
    pi_imb: np.ndarray     # (T,) imbalance price (single price)
    h_plus: np.ndarray     # (T,) upper bound of Xi'    ( xi <= h+ )
    h_minus: np.ndarray    # (T,) lower bound magnitude ( xi >= -h- )
    xi_samples: np.ndarray # (n, T) empirical error scenarios (objective only)
    k: float = 0.5         # 0 = pure DA arbitrage, 1 = pure imbalance minimisation
    terminal_soc_equality: bool = True  # False => p_soc_T >= SOC0 (dualised instead)


@dataclass
class Vars:
    """Decision variables shared by both formulations."""
    p_ch: cp.Variable      # (T,)   here-and-now charge
    p_dis: cp.Variable     # (T,)   here-and-now discharge
    D_ch: cp.Variable      # (T,T)  LDR charge recourse  (lower triangular)
    D_dis: cp.Variable     # (T,T)  LDR discharge recourse (lower triangular)
    cons: list = field(default_factory=list)


def _make_vars(P: Params) -> Vars:
    T = P.T
    v = Vars(cp.Variable(T, name="p_ch_hat"),
             cp.Variable(T, name="p_dis_hat"),
             cp.Variable((T, T), name="D_ch"),
             cp.Variable((T, T), name="D_dis"))
    # Non-anticipativity: D[t, tau] = 0 for tau > t  (strictly-upper entries vanish).
    # row t of D is the coefficient vector acting on xi for period t.
    # Making D_ch and D_dis lower-triangular means period t only sees xi_1..xi_t.
    lower = np.tril(np.ones((T, T)), 0).astype(bool)
    v.cons += [v.D_ch[lower] == True, v.D_dis[lower] == True]
    return v


# --------------------------------------------------------------------------
# Affine-in-xi quantities (Section 7 of the write-up)
# --------------------------------------------------------------------------
def _soc_affine(P: Params, v: Vars):
    """Return (s_hat, G) with  p_soc_t(xi) = SOC0 + s_hat[t] + G[t, :] @ xi."""
    L = np.tril(np.ones((P.T, P.T)))                     # L[t, s] = 1 iff s <= t
    s_hat = L @ (P.eta_ch * v.p_ch - (1.0 / P.eta_dis) * v.p_dis)   # (T,)
    # g_t = sum_{s<=t} ( eta_ch * col_s(D_ch) - 1/eta_dis * col_s(D_dis) )
    # col_s(D) = D[:, s], so  G = ( eta_ch*D_ch - 1/eta_dis*D_dis ) @ L^T   -> (T rows = tau)
    M = P.eta_ch * v.D_ch - (1.0 / P.eta_dis) * v.D_dis  # (tau, s)
    G = (M @ L.T).T                                      # G[t, tau] = sum_{s<=t} M[tau, s]
    return s_hat, G


def _objective(P: Params, v: Vars):
    """Empirical objective (4.2): expectation over the SAMPLE set, not the box."""
    T, X = P.T, P.xi_samples
    n = X.shape[0]
    p_bid = P.pl_hat + v.p_ch - v.p_dis                       # (T,)
    # eps_i = (I + (D_ch - D_dis)^T) xi_i   ->  stack as  X @ (I + (D_ch - D_dis))
    # because (A^T x)_i stacked over rows i is  X @ A  when A = (D_ch - D_dis).
    E = X @ (np.eye(T) + (v.D_ch - v.D_dis))                  # (n, T) imbalance
    econ = P.pi_da @ p_bid + cp.sum(E @ P.pi_imb) / n
    quad = cp.sum_squares(E) / n
    return cp.Minimize((1 - P.k) * econ + P.k * quad)


# --------------------------------------------------------------------------
# Formulation A: robust counterpart via LP duality  (the real model)
# --------------------------------------------------------------------------
def _robustify(a0, a, b, P: Params, cons: list):
    """
    Enforce  a0 + a^T xi <= b  for all xi in the box, by adding the dual certificate
        mu+ , mu- >= 0 ,  mu+ - mu- == a ,  h+^T mu+ + h-^T mu- <= b - a0.
    `a0` is a scalar expression, `a` a (T,) expression, `b` a scalar constant.
    """
    T = P.T
    mu_p = cp.Variable(T, nonneg=True)
    mu_m = cp.Variable(T, nonneg=True)
    cons += [
        mu_p - mu_m == a,                                     # H^T mu = a
        P.h_plus @ mu_p + P.h_minus @ mu_m <= b - a0,         # h^T mu <= b - a0
    ]


def build_robust(P: Params):
    v = _make_vars(P)
    cons = list(v.cons)
    s_hat, G = _soc_affine(P, v)

    for t in range(P.T):
        d_ch, d_dis = v.D_ch[:, t], v.D_dis[:, t]   # columns = coefficient vectors on xi
        g_t = G[t, :]

        # (1)(2) charge rate                       0 <= p_ch_t(xi) <= C_ch
        _robustify( v.p_ch[t],  d_ch,  P.C_ch,             P, cons)
        _robustify(-v.p_ch[t], -d_ch,  0.0,                P, cons)
        # (3)(4) discharge rate                    0 <= p_dis_t(xi) <= C_dis
        _robustify( v.p_dis[t],  d_dis, P.C_dis,           P, cons)
        _robustify(-v.p_dis[t], -d_dis, 0.0,               P, cons)
        # (5)(6) state of charge                   0 <= p_soc_t(xi) <= B_max
        _robustify( s_hat[t],  g_t,  P.B_max - P.SOC0,     P, cons)
        _robustify(-s_hat[t], -g_t,  P.SOC0,               P, cons)

    # Terminal SOC (Section 10) -- an EQUALITY, so coefficient-match, don't dualise.
    if P.terminal_soc_equality:
        cons += [s_hat[P.T - 1] == 0,        # shat_T = SOC0
                 G[P.T - 1, :] == 0]         # g_T = 0 : recourse is energy-neutral
    else:
        _robustify(-s_hat[P.T - 1], -G[P.T - 1, :], 0.0, P, cons)   # p_soc_T >= SOC0

    return cp.Problem(_objective(P, v), cons), v


# --------------------------------------------------------------------------
# Formulation B: vertex enumeration  (correctness oracle, small T only)
# --------------------------------------------------------------------------
def build_vertex(P: Params):
    """Affine constraints over a box are tight at its corners, so imposing them at
    all 2^T vertices is EXACTLY equivalent to the robust counterpart. Use as a check."""
    v = _make_vars(P)
    cons = list(v.cons)
    s_hat, G = _soc_affine(P, v)

    corners = [np.array(c) for c in product(*[(P.h_plus[i], -P.h_minus[i])
                                              for i in range(P.T)])]
    for xi in corners:
        p_ch_r = v.p_ch + v.D_ch.T @ xi
        p_dis_r = v.p_dis + v.D_dis.T @ xi
        soc_r = P.SOC0 + s_hat + G @ xi
        cons += [p_ch_r >= 0, p_ch_r <= P.C_ch,
                 p_dis_r >= 0, p_dis_r <= P.C_dis,
                 soc_r >= 0, soc_r <= P.B_max]
        if P.terminal_soc_equality:
            cons += [soc_r[P.T - 1] == P.SOC0]
        else:
            cons += [soc_r[P.T - 1] >= P.SOC0]

    return cp.Problem(_objective(P, v), cons), v


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def report(P: Params, v: Vars, label: str):
    print(f"\n--- {label} ---")
    print(f"objective            : {v.p_ch.value is not None}")
    D = v.D_ch.value - v.D_dis.value
    print(f"||D_ch - D_dis||_F   : {np.linalg.norm(D):.4f}   "
          f"(-> 0 means the LDR has been squeezed out: Xi' too wide, or a degenerate objective)")
    wash = np.minimum(v.p_ch.value, v.p_dis.value)
    print(f"max simultaneous ch/dis (nominal): {wash.max():.4f}   (should be ~0)")
    # out-of-sample check on the SAMPLE scenarios (which may lie outside Xi'!)
    viol = 0
    for xi in P.xi_samples:
        soc = P.SOC0 + np.tril(np.ones((P.T, P.T))) @ (
            P.eta_ch * (v.p_ch.value + v.D_ch.value.T @ xi)
            - (1 / P.eta_dis) * (v.p_dis.value + v.D_dis.value.T @ xi))
        pc = v.p_ch.value + v.D_ch.value.T @ xi
        pd = v.p_dis.value + v.D_dis.value.T @ xi
        if (soc.min() < -1e-6 or soc.max() > P.B_max + 1e-6
                or pc.min() < -1e-6 or pc.max() > P.C_ch + 1e-6
                or pd.min() < -1e-6 or pd.max() > P.C_dis + 1e-6):
            viol += 1
    print(f"sample scenarios violating constraints: {viol}/{len(P.xi_samples)} "
          f"(non-zero => samples fall outside Xi'; widen h or accept the risk)")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 6
    P = Params(
        T=T, eta_ch=0.95, eta_dis=0.95, C_ch=2.0, C_dis=2.0, B_max=6.0, SOC0=3.0,
        pl_hat=np.array([1., 3., 4., 2., 3., 1.]),
        pi_da=np.array([50., 80., 120., 60., 90., 40.]),
        pi_imb=np.array([70., 90., 40., 100., 55., 85.]),
        h_plus=np.full(T, 0.8), h_minus=np.full(T, 0.8),
        xi_samples=np.clip(rng.normal(0.1, 0.3, size=(200, T)), -0.8, 0.8),
        k=0.5,
    )

    prob_r, v_r = build_robust(P)
    prob_r.solve(solver=cp.CLARABEL)
    print(f"robust counterpart  : {prob_r.status}, obj = {prob_r.value:.8f}  "
          f"({prob_r.size_metrics.num_scalar_variables} vars)")

    prob_v, v_v = build_vertex(P)      # 2^6 = 64 corners; do NOT try this for T=48
    prob_v.solve(solver=cp.CLARABEL)
    print(f"vertex enumeration  : {prob_v.status}, obj = {prob_v.value:.8f}  "
          f"({prob_v.size_metrics.num_scalar_variables} vars)")
    print(f"GAP                 : {abs(prob_r.value - prob_v.value):.3e}  "
          f"<-- must be ~0; if not, you have a sign/transpose bug")

    report(P, v_r, "robust solution")

    # Xi' width sweep: watch the recourse die as the box widens.
    print("\nrho   ||D||_F    objective")
    for rho in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]:
        Pr = Params(**{**P.__dict__, "h_plus": np.full(T, 0.8 * rho),
                       "h_minus": np.full(T, 0.8 * rho)})
        pr, vr = build_robust(Pr)
        pr.solve(solver=cp.CLARABEL)
        if vr.D_ch.value is None:
            print(f"{rho:<5} INFEASIBLE ({pr.status})")
        else:
            nrm = np.linalg.norm(vr.D_ch.value - vr.D_dis.value)
            print(f"{rho:<5} {nrm:<10.4f} {pr.value:.4f}")
