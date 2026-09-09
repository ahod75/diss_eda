from __future__ import annotations
from itertools import product
from dataclasses import dataclass, field, replace

import numpy as np
import cvxpy as cp


# --------------------------------------------------------------------------
# Problem data
# --------------------------------------------------------------------------
@dataclass
class Params:
    T_total: int  # number of t steps in time window T
    dt: float  # length of time step t
    eta_ch: float  # charging efficiency
    eta_dis: float  # discharging efficiency
    C_ch: float  # max charge rate
    C_dis: float  # max discharge rate
    B_max: float  # battery energy capacity
    SOC0: float  # initial == terminal state of charge
    pl_hat: np.ndarray  # (T,) expected prosumption
    pi_da: np.ndarray  # (T,) day-ahead price
    pi_imb: np.ndarray  # (T,) imbalance price (single price)
    h_plus: np.ndarray  # (T,) upper bound of Xi'    ( xi <= h+ )
    h_minus: np.ndarray  # (T,) lower bound magnitude ( xi >= -h- )
    xi_samples: np.ndarray  # (T, n) empirical error scenarios (objective only)
    k: float = 0.5  # 0 = pure DA arbitrage, 1 = pure imbalance minimisation
    gamma: float = 10**-6  # Tikhonov regularisation coefficient
    terminal_soc_equality: bool = True  # True if terminal equality. False if \le


@dataclass
class Vars:
    """Decision variables shared by both formulations."""

    p_ch_hat: cp.Variable  # (T,)   here-and-now charge
    p_dis_hat: cp.Variable  # (T,)   here-and-now discharge
    D_ch: cp.Variable  # (T,T)  LDR charge recourse  (lower triangular)
    D_dis: cp.Variable  # (T,T)  LDR discharge recourse
    cons: list = field(default_factory=list)


def _make_vars(P: Params) -> Vars:
    T = P.T_total
    v = Vars(
        cp.Variable(T, name="p_ch_hat"),
        cp.Variable(T, name="p_dis_hat"),
        cp.Variable((T, T), name="D_ch"),
        cp.Variable((T, T), name="D_dis"),
    )
    # Non-anticipativity: D[t, tau] = 0 for tau > t  (strictly-upper entries vanish).
    # row t of D is the coefficient vector acting on xi for period t.
    # Making D_ch and D_dis lower-triangular means period t only sees xi_1..xi_t.
    strictly_upper = np.triu(np.ones((T, T)), 1).astype(bool)
    v.cons += [v.D_ch[strictly_upper] == 0, v.D_dis[strictly_upper] == 0]
    return v


### NOTES ON ROBUSTIFYING PROBLEM FORMULATION
# To robustify the constraints of the model, we need to separate the random variable \xi
# from the rest of the model.
# To do this, we can represent the total SOC function as an affine function.
# To do this, we separate the day-ahead decisions (which don't rely on \xi at all) from the
# recourse actions (which depend on \xi).
# That means that:
#   e_soc_t = e_soc_t-1 + dt(eta_ch @ (p_ch_hat_t + D_ch_t @ \xi_t)
#                                      - (1/eta_dis) *
#                                      (p_dis_hat_t + D_dis_t @ \xi_t)
#   e_soc = SOC0 + dt * L @ (eta_ch @ p_ch_hat - (1/eta_dis) *
#                                           (p_dis_hat)
#           + dt * L @ (eta_ch @ D_ch - (1/eta_dis) * D_dis) @ xi
# where L is a T x T lower triangular matrix.
# (this can be coded as L = np.tril(np.ones((P.T_total, P.T_total))))
#
# The parts of the equation independent of xi can be interpreted as the intercept of the
# SOC equation, whilst the parts that relate to xi can be interpreted as the gradient.
# So:
# intercept vector (T,):        s_hat = SOC0 + dt * L @ (eta_ch*p_ch_hat - (1/eta_dis)*p_dis_hat)
# slope matrix (T,T), row t = g_t:   G = dt * L @ (eta_ch*D_ch - (1/eta_dis)*D_dis)
# then e_soc(xi) = s_hat + G @ xi    for every xi


def _soc_affine(P: Params, v: Vars):
    """Return (s_hat, G) with  p_soc_t(xi) = s_hat[t] + G[t, :] @ xi."""
    L = np.tril(np.ones((P.T_total, P.T_total)))
    s_hat = P.SOC0 + P.dt * L @ (P.eta_ch * v.p_ch_hat - (1 / P.eta_dis) * v.p_dis_hat)
    G = P.dt * L @ (P.eta_ch * v.D_ch - (1 / P.eta_dis) * v.D_dis)
    return s_hat, G


def _objective(P: Params, v: Vars):
    """Empirical objective (4.2): expectation over the SAMPLE set, not the box."""

    ## Define used vars and params
    T_total, X = P.T_total, P.xi_samples
    assert X.shape[0] == T_total, f"xi_samples must be (T, n), got {X.shape}"
    n = X.shape[1]

    # combined control variables (used in cost function)
    p_bid = P.pl_hat + v.p_ch_hat - v.p_dis_hat  # (T,)
    p_imb = (np.eye(T_total) + v.D_ch - v.D_dis) @ X  # (T, n)
    # dimensions: (T, n) = ((T, T) + (T, T) - (T, T)) @ (T, n)

    # intermediate cost function terms
    C_da = P.pi_da @ p_bid * P.dt
    C_imb = cp.sum(P.pi_imb @ p_imb) * P.dt / n

    # final cost function terms
    econ = C_da + C_imb
    quad = cp.sum_squares(p_imb * P.dt) / n
    penalty = P.gamma * (
        cp.sum_squares(v.p_ch_hat)
        + cp.sum_squares(v.p_dis_hat)
        + cp.sum_squares(v.D_ch)
        + cp.sum_squares(v.D_dis)
    )

    return cp.Minimize((1 - P.k) * econ + P.k * quad + penalty)


def _robustify(a0, a, b, P: Params, cons: list):
    """Enforce  a0 + a^T xi <= b  for all xi in the box, by adding the dual certificate
        mu+ , mu- >= 0 ,  mu+ - mu- == a ,  h+^T mu+ + h-^T mu- <= b - a0.
    `a0` is a scalar expression, `a` a (T,) expression, `b` a scalar constant.
    """
    T = P.T_total
    mu_p = cp.Variable(T, nonneg=True)
    mu_m = cp.Variable(T, nonneg=True)
    cons += [
        mu_p - mu_m == a,  # H^T mu = a
        P.h_plus @ mu_p + P.h_minus @ mu_m <= b - a0,  # h^T mu <= b - a0
    ]


def build_robust(P: Params):
    v = _make_vars(P)
    cons = list(v.cons)
    s_hat, G = _soc_affine(P, v)
    
    for t in range(P.T_total):
        d_ch, d_dis = v.D_ch[t, :], v.D_dis[t, :]  # rows = coefficient vectors on xi
        g_t = G[t, :]

        # (1)(2) charge rate                       0 <= p_ch_t(xi) <= C_ch
        _robustify(v.p_ch_hat[t], d_ch, P.C_ch, P, cons)
        _robustify(-v.p_ch_hat[t], -d_ch, 0.0, P, cons)
        # (3)(4) discharge rate                    0 <= p_dis_t(xi) <= C_dis
        _robustify(v.p_dis_hat[t], d_dis, P.C_dis, P, cons)
        _robustify(-v.p_dis_hat[t], -d_dis, 0.0, P, cons)
        # (5)(6) state of charge                   0 <= p_soc_t(xi) <= B_max
        _robustify(s_hat[t], g_t, P.B_max, P, cons)
        _robustify(-s_hat[t], -g_t, 0, P, cons)

    # Terminal SOC (Section 10) -- an EQUALITY, so coefficient-match, don't dualise.
    if P.terminal_soc_equality:
        cons += [
            s_hat[P.T_total - 1] == P.SOC0,  # s_hat = SOC0
            G[P.T_total - 1, :] == 0,
        ]  # g_T = 0 : recourse is energy-neutral
    else:
        _robustify(
            -s_hat[P.T_total - 1], -G[P.T_total - 1, :], -P.SOC0, P, cons
        )  # p_soc_T >= SOC0

    return cp.Problem(_objective(P, v), cons), v


# --------------------------------------------------------------------------
# Formulation B: vertex enumeration  (correctness oracle, small T only)
# --------------------------------------------------------------------------


def build_vertex(P: Params):
    """Oracle: impose every robust constraint at every vertex of the box.
    Exact (affine constraints on a box are tight at vertices). T <= ~10 only."""
    v = _make_vars(P)
    cons = list(v.cons)
    s_hat, G = _soc_affine(P, v)
    T = P.T_total

    corners = [
        np.array(c) for c in product(*[(P.h_plus[i], -P.h_minus[i]) for i in range(T)])
    ]

    for xi in corners:
        p_ch_r = v.p_ch_hat + v.D_ch @ xi
        p_dis_r = v.p_dis_hat + v.D_dis @ xi
        e_soc_r = s_hat + G @ xi
        cons += [
            p_ch_r >= 0,
            p_ch_r <= P.C_ch,
            p_dis_r >= 0,
            p_dis_r <= P.C_dis,
            e_soc_r >= 0,
            e_soc_r <= P.B_max,
        ]
        if P.terminal_soc_equality:
            cons += [e_soc_r[T - 1] == P.SOC0]
        else:
            cons += [e_soc_r[T - 1] >= P.SOC0]

    return cp.Problem(_objective(P, v), cons), v


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def report(P: Params, v: Vars, label: str):
    print(f"\n--- {label} ---")
    print(f"Solved            : {v.p_ch_hat.value is not None}")
    D = v.D_ch.value - v.D_dis.value
    print(
        f"||D_ch - D_dis||_F   : {np.linalg.norm(D):.4f}   "
        f"(-> 0 means the LDR has been squeezed out: Xi' too wide, or a degenerate objective)"
    )
    wash = np.minimum(v.p_ch_hat.value, v.p_dis_hat.value)
    print(f"max simultaneous ch/dis (nominal): {wash.max():.4f}   (should be ~0)")
    # out-of-sample check on the SAMPLE scenarios (which may lie outside Xi'!)

    viol = 0
    s_hat, G = _soc_affine(P, v)
    n_scen = P.xi_samples.shape[1]
    for i in range(n_scen):
        xi = P.xi_samples[:, i]
        soc = s_hat.value + G.value @ xi
        pc = v.p_ch_hat.value + v.D_ch.value @ xi
        pd = v.p_dis_hat.value + v.D_dis.value @ xi
        if (
            soc.min() < -1e-6
            or soc.max() > P.B_max + 1e-6
            or abs(soc[-1] - P.SOC0) > 1e-6
            or pc.min() < -1e-6
            or pc.max() > P.C_ch + 1e-6
            or pd.min() < -1e-6
            or pd.max() > P.C_dis + 1e-6
        ):
            viol += 1
    print(
        f"sample scenarios violating constraints: {viol}/{n_scen} "
        f"(non-zero => samples fall outside Xi'; widen h or accept the risk)"
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 6
    P = Params(
        T_total=T,
        dt=1,
        eta_ch=0.95,
        eta_dis=0.95,
        C_ch=2.0,
        C_dis=2.0,
        B_max=6.0,
        SOC0=3.0,
        pl_hat=np.array([1.0, 3.0, 4.0, 2.0, 3.0, 1.0]),
        pi_da=np.array([50.0, 80.0, 120.0, 60.0, 90.0, 40.0]),
        pi_imb=np.array([70.0, 90.0, 40.0, 100.0, 55.0, 85.0]),
        h_plus=np.full(T, 0.8),
        h_minus=np.full(T, 0.8),
        xi_samples=np.clip(rng.normal(0.1, 0.3, size=(T, 200)), -0.8, 0.8),
        k=0.5,
    )

    prob_r, v_r = build_robust(P)
    prob_r.solve(solver=cp.CLARABEL)
    print(
        f"robust counterpart  : {prob_r.status}, obj = {prob_r.value:.8f}  "
        f"({prob_r.size_metrics.num_scalar_variables} vars)"
    )

    prob_v, v_v = build_vertex(P)  # 2^6 = 64 corners; do NOT try this for T=48
    prob_v.solve(solver=cp.CLARABEL)
    print(
        f"vertex enumeration  : {prob_v.status}, obj = {prob_v.value:.8f}  "
        f"({prob_v.size_metrics.num_scalar_variables} vars)"
    )
    print(
        f"GAP                 : {abs(prob_r.value - prob_v.value):.3e}  "
        f"<-- must be ~0; if not, you have a sign/transpose bug"
    )

    report(P, v_r, "robust solution")

    # Xi' width sweep: watch the recourse die as the box widens.
    print("\nrho   ||D||_F    objective")
    for rho in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]:
        Pr = replace(P, h_plus=np.full(T, 0.8 * rho), h_minus=np.full(T, 0.8 * rho))
        pr, vr = build_robust(Pr)
        pr.solve(solver=cp.CLARABEL)
        if vr.D_ch.value is None:
            print(f"{rho:<5} INFEASIBLE ({pr.status})")
        else:
            nrm = np.linalg.norm(vr.D_ch.value - vr.D_dis.value)
            print(f"{rho:<5} {nrm:<10.4f} {pr.value:.4f}")
