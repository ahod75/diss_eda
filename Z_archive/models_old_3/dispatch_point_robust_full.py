from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# =====================================================================================
# dispatch_point_robust_full.py  --  the point-robust 3-point model, END TO END, in one
# SELF-CONTAINED file. Companion code to "Model 3. Point-Robust Two-Stage LDR
# Formulation.md". No imports from this project's other model files -- everything
# needed to go from (forecast mean, scenarios, box, prices) to a fully realised,
# true-price-settled day is defined below.
#
# This is a deliberate departure from the earlier version of this file, which imported
# the two pieces from dispatch_layer_point_robust.py / dispatch_wrapper.py instead of
# copying them. That version has zero duplication risk (one source of truth); THIS
# version has none of that protection -- if dispatch_layer_point_robust.py or
# dispatch_wrapper.py change, this file will silently drift out of sync with the actual
# training/eval pipeline. Keep that in mind if this is used for anything beyond reading
# the whole formulation in one place.
#
# Two halves, in the order execution actually happens:
#   PART 1 -- WITHIN cvxpylayers: the differentiable convex solve. Declares
#             p_ch_hat/p_dis_hat/D_ch/D_dis as cp.Variables, prices/h_plus/h_minus/
#             xi_samples/Sigma_xi_chol as cp.Parameters, and returns decisions via
#             CvxpyLayer's forward/backward (implicit differentiation of the KKT
#             system). pl_hat is NOT a Parameter here -- it cannot affect the argmin
#             (see PART 1's build_problem for the algebra), so it is dropped entirely
#             and reattached in PART 2.
#   PART 2 -- WITHOUT cvxpylayers: plain PyTorch, ordinary autograd. Takes the SOLVED
#             decisions, reconstructs the pinned bid (pl_hat + p_da_rel), applies
#             physical saturation, and prices the result at TRUE (not forecast) prices
#             to get the actually-realised C_da/C_imb/p_imb.
#   PART 3 -- the glue: one call (`solve_and_realise`) that runs PART 1 then PART 2 for
#             a single day, exactly as `dfl_loss_batch` in 7_model_training/
#             train_corner_point_robust.py does per-day inside its training loop.
# =====================================================================================


# =====================================================================================
# PART 1 -- WITHIN cvxpylayers (copied from dispatch_layer_point_robust.py)
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
    # realised_cost/realised_breakdown (PART 2).
    h_plus  = cp.Parameter(T, nonneg=True, name="h_plus")  # frozen box (B1), point-scenario only
    h_minus = cp.Parameter(T, nonneg=True, name="h_minus")

    cons = []

    # ---- LDR non-anticipativity via a MASK, not an equality constraint -----------
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
    p_da_rel = p_ch_hat - p_dis_hat

    # ---- shared imbalance building blocks -----------------------------------------
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

    params += [h_plus, h_minus]

    prob = cp.Problem(cp.Minimize(sum(obj_terms)), cons)
    assert prob.is_dcp(dpp=True), "not DPP -- cvxpylayers will reject"

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


# =====================================================================================
# PART 2 -- WITHOUT cvxpylayers (copied from dispatch_wrapper.py)
# =====================================================================================

_SAT_TOL = 1e-6
PRICE_COLS = ("da", "imb", "imb_up", "imb_down")


def cholesky_of_second_moment(xi: torch.Tensor, jitter: float = 1e-9) -> torch.Tensor:
    """DIFFERENTIABLE (T,T) Cholesky factor for k>0's Sigma_xi_chol layer parameter.
    M = xi^T xi / N (biased /N: matches sum_squares(R @ Sigma_xi_chol)'s trace identity
    exactly for mean-centred xi). Symmetrise + jitter before Cholesky -- cheap insurance
    against intermittent non-PD M (N=64 > T=24 so generically full rank, but not
    guaranteed)."""
    N = xi.shape[-2]
    M = xi.transpose(-2, -1) @ xi / N
    M = 0.5 * (M + M.transpose(-2, -1)) + jitter * torch.eye(xi.shape[-1], dtype=xi.dtype, device=xi.device)
    return torch.linalg.cholesky(M)   # lower L with L @ L^T == M; grad flows to xi


def get_prices(price_day, price_model, cols=PRICE_COLS):
    """price_day: (T, 4) slice; columns in `cols` order.
    imb_up/imb_down are the FULL settlement price level for each direction
    (imb_up=max(da,imb), imb_down=min(da,imb)) -- NOT a premium over da. Clipped at 0
    for the rare negative-price hours, to satisfy the nonneg=True Parameter constraint."""
    price_day = np.asarray(price_day, dtype=float)
    idx = {c: i for i, c in enumerate(cols)}
    da = price_day[..., idx["da"]]
    if price_model == "single":
        return {"pi_da": da, "pi_imb": price_day[..., idx["imb"]]}   # pi_imb POSITIVE; sign is in p_imb
    if price_model == "dual":
        return {"pi_da": da,
                "pi_imb_up": np.clip(price_day[..., idx["imb_up"]], 0.0, None),
                "pi_imb_down": np.clip(price_day[..., idx["imb_down"]], 0.0, None)}
    raise ValueError(price_model)


def oracle_price_values(price_day, price_model, realised, cols=PRICE_COLS):
    """Full param dict for a perfect-foresight solve: prices + p_d = realised prosumption."""
    d = get_prices(price_day, price_model, cols)
    d["p_d"] = np.asarray(realised, dtype=float)
    return d


def assert_price_consistency(price_all, cols=PRICE_COLS, tol=1e-2):
    """Run ONCE at load over the WHOLE dataset. Verifies imb_up/imb_down are genuinely
    max(da,imb)/min(da,imb)."""
    p = np.asarray(price_all, dtype=float).reshape(-1, len(cols))
    idx = {c: i for i, c in enumerate(cols)}
    da, imb = p[:, idx["da"]], p[:, idx["imb"]]
    up, dn = p[:, idx["imb_up"]], p[:, idx["imb_down"]]
    assert (up >= dn - tol).all(), "imb_up must be always >= imb_down"
    assert np.allclose(up, np.maximum(da, imb), atol=tol), "imb_up != max(da, imb)"
    assert np.allclose(dn, np.minimum(da, imb), atol=tol), "imb_down != min(da, imb)"


def _to_t(x, like: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float64))
    if like is not None:
        t = t.to(dtype=like.dtype, device=like.device)
    return t


def make_dispatch_inputs(
    quantiles,                 # (K, Q) THIS model's quantiles for one day (physical MW)
    sampler,                   # FrozenCopulaSampler (mean_and_errors); S must == num_scenarios
    price_model,               # "single" | "dual"
    k,                         # 0 | 1
    pi_da,                     # (T,)
    pi_imb=None,
    pi_imb_up=None, pi_imb_down=None,
    also_return_meta=False,
):
    """pl_hat, xi, Sigma come from THIS model's quantiles; the box (h_plus, h_minus) is
    the FROZEN baseline box, passed in separately at the call site."""
    if quantiles.dim() != 2:
        raise ValueError("quantiles must be (K, Q) for a single day")
    mean, xi = sampler.mean_and_errors(quantiles)      # mean (T,), xi (N,T); grad kept
    pl_hat = mean

    vals = {
        "pl_hat":  pl_hat,
        "pi_da":   _to_t(pi_da,   like=xi),
    }
    if price_model == "single":
        if pi_imb is None:
            raise ValueError("single price needs pi_imb")
        vals["pi_imb"] = _to_t(pi_imb, like=xi)
    elif price_model == "dual":
        if pi_imb_up is None or pi_imb_down is None:
            raise ValueError("dual price needs pi_imb_up and pi_imb_down (computed outside as max{0,.})")
        vals["xi_samples"] = xi
        vals["pi_imb_up"] = _to_t(pi_imb_up, like=xi)
        vals["pi_imb_down"] = _to_t(pi_imb_down, like=xi)
    else:
        raise ValueError(price_model)

    if k > 0.0:
        vals["Sigma_xi_chol"] = cholesky_of_second_moment(xi)

    if also_return_meta:
        return vals, {"pl_hat": pl_hat, "xi": xi}
    return vals


@dataclass
class RealisedBreakdown:
    p_ch_raw:  torch.Tensor   # (T,) raw recourse charge  = p_ch_hat  + D_ch  @ xi_real
    p_dis_raw: torch.Tensor   # (T,) raw recourse discharge
    p_ch_r:    torch.Tensor   # (T,) realised charge  (post-saturation)
    p_dis_r:   torch.Tensor   # (T,) realised discharge (post-saturation)
    soc:       torch.Tensor   # (T,) realised SOC trajectory (post-action)
    bid:       torch.Tensor   # (T,) committed DA bid = pl_hat + p_da_rel
    p_g:       torch.Tensor   # (T,) realised grid draw = realised + p_ch_r - p_dis_r
    p_imb:     torch.Tensor   # (T,) realised imbalance = p_g - bid
    C_da:      torch.Tensor   # scalar
    C_imb:     torch.Tensor   # scalar (signed single, >=0 dual)


def realised_breakdown(
    fp: FixedParams,
    p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_rel,   # the 5 decisions from the layer / solve
    realised,                                      # (T,) realised prosumption for the day
    pl_hat,                                        # (T,) SAME anchor the layer used
    price_model,                                   # "single" | "dual"
    pi_da, pi_imb=None, pi_imb_up=None, pi_imb_down=None,
    clip_recourse=True,                            # test: True; correctness gate uses False
) -> RealisedBreakdown:
    """The single source of truth for realised quantities. Used for both training AND
    testing -- for training, arguments are tensors so PyTorch's autograd can track
    gradients; for testing this is unnecessary but keeps one code path for both."""
    p_ch_hat = torch.as_tensor(p_ch_hat) if isinstance(p_ch_hat, torch.Tensor) \
           else torch.as_tensor(np.asarray(p_ch_hat, np.float64))
    p_dis_hat = _to_t(p_dis_hat, like=p_ch_hat)
    D_ch      = _to_t(D_ch, like=p_ch_hat)
    D_dis = _to_t(D_dis, like=p_ch_hat)
    p_da_rel  = _to_t(p_da_rel, like=p_ch_hat)
    realised  = _to_t(realised,  like=p_ch_hat)
    pl_hat    = _to_t(pl_hat,    like=p_ch_hat)
    pi_da     = _to_t(pi_da,     like=p_ch_hat)
    T, dt = fp.T_total, fp.dt

    xi_real = realised - pl_hat                                    # [B, 24]
    bid = pl_hat + p_da_rel                                        # [B, 24]

    xi_real_col = xi_real.unsqueeze(-1)

    p_ch_raw  = p_ch_hat  + (D_ch  @ xi_real_col).squeeze(-1)
    p_dis_raw = p_dis_hat + (D_dis @ xi_real_col).squeeze(-1)

    if clip_recourse:
        # Stratigakos-like state-dependent saturation. SOC headroom caps carry /dt (dt-correct).
        soc = _to_t(fp.SOC0, like=p_ch_hat)
        ch_list, dis_list, soc_list = [], [], []
        C_ch  = _to_t(fp.C_ch,  like=p_ch_hat)
        C_dis = _to_t(fp.C_dis, like=p_ch_hat)
        B_max = _to_t(fp.B_max, like=p_ch_hat)
        z = _to_t(0.0, like=p_ch_hat)

        for t in range(T):
            max_ch  = torch.minimum(C_ch,  (B_max - soc) / (fp.eta_ch * dt))
            max_dis = torch.minimum(C_dis, (soc * fp.eta_dis) / dt)
            pc = torch.clamp(torch.minimum(torch.clamp(p_ch_raw[..., t],  min=z), torch.clamp(max_ch,  min=z)), min=z)
            pd = torch.clamp(torch.minimum(torch.clamp(p_dis_raw[..., t], min=z), torch.clamp(max_dis, min=z)), min=z)
            soc = soc + dt * (fp.eta_ch * pc - (1.0 / fp.eta_dis) * pd)
            ch_list.append(pc); dis_list.append(pd); soc_list.append(soc)
        p_ch_r  = torch.stack(ch_list, dim=-1)
        p_dis_r = torch.stack(dis_list, dim=-1)
        soc_traj = torch.stack(soc_list, dim=-1)
    else:
        p_ch_r, p_dis_r = p_ch_raw, p_dis_raw
        net = fp.eta_ch * p_ch_r - (1.0 / fp.eta_dis) * p_dis_r
        soc_traj = fp.SOC0 + dt * torch.cumsum(net, dim=-1)

    p_g = realised + p_ch_r - p_dis_r                             # realised grid draw
    p_imb = p_g - bid                                            # = imb_det + R xi_real (in-box)

    C_da = (pi_da * bid).sum(dim=-1) * dt                              # includes pi_da . pl_hat (add-back)

    if price_model == "single":
        if pi_imb is None:
            raise ValueError("single needs pi_imb")
        C_imb = (_to_t(pi_imb, like=p_ch_hat) * p_imb).sum(dim=-1) * dt          # signed
    elif price_model == "dual":
        if pi_imb_up is None or pi_imb_down is None:
            raise ValueError("dual needs pi_imb_up, pi_imb_down")
        pi_up = _to_t(pi_imb_up, like=p_ch_hat); pi_down = _to_t(pi_imb_down, like=p_ch_hat)
        C_imb = (pi_up * torch.clamp(p_imb, min=0.0)
                 + pi_down * torch.clamp(-p_imb, min=0.0)).sum(dim=-1) * dt          # >= 0
    else:
        raise ValueError(price_model)

    return RealisedBreakdown(
        p_ch_raw=p_ch_raw, p_dis_raw=p_dis_raw, p_ch_r=p_ch_r, p_dis_r=p_dis_r,
        soc=soc_traj, bid=bid, p_g=p_g, p_imb=p_imb, C_da=C_da, C_imb=C_imb,
    )


def realised_cost(fp, *args, **kwargs) -> torch.Tensor:
    """Thin DFL-loss wrapper: total realised cost = C_da + C_imb on the SAME breakdown the
    test metrics use (so training loss and reported cost cannot drift)."""
    bd = realised_breakdown(fp, *args, **kwargs)
    return bd.C_da + bd.C_imb


def per_day_metrics(fp: FixedParams, bd: RealisedBreakdown, oracle_cost: float) -> dict:
    """The seven per-day scalars. Saturation is combined charge+discharge (per
    decision). oracle_cost is precomputed per (day, price_model)."""
    dt = fp.dt
    total_cost = float(bd.C_da + bd.C_imb)
    clip_ch  = torch.abs(bd.p_ch_raw  - bd.p_ch_r)
    clip_dis = torch.abs(bd.p_dis_raw - bd.p_dis_r)
    sat_hours = int(((clip_ch > _SAT_TOL) | (clip_dis > _SAT_TOL)).sum().item())
    sat_MWh   = float(((clip_ch + clip_dis) * dt).sum().item())
    return {
        "total_cost":     total_cost,                                  # 1
        "regret":         total_cost - float(oracle_cost),             # 2
        "abs_dev_MWh":    float((torch.abs(bd.p_imb) * dt).sum().item()),  # 3
        "C_da":           float(bd.C_da.item()),                       # 4a  (DA/IMB split...
        "C_imb":          float(bd.C_imb.item()),                      # 4b   ...sums to total_cost)
        "sat_hours":      sat_hours,                                   # 5
        "sat_MWh":        sat_MWh,                                     # 6
        "throughput_MWh": float(((bd.p_ch_r + bd.p_dis_r) * dt).sum().item()),  # 7 (grid-side)
    }


def money_plot_series(fp, bd: RealisedBreakdown, price_model,
                      pi_da, pi_imb=None, pi_imb_up=None, pi_imb_down=None) -> dict:
    """Arrays for the bid / p^g / p_imb / price plots, for one representative day."""
    to_np = lambda t: t.detach().cpu().numpy()
    out = {
        "bid":    to_np(bd.bid),
        "p_g":    to_np(bd.p_g),
        "p_imb":  to_np(bd.p_imb),
        "soc":    to_np(bd.soc),
        "p_ch_r": to_np(bd.p_ch_r),
        "p_dis_r": to_np(bd.p_dis_r),
        "pi_da":  np.asarray(pi_da, float),
    }
    if price_model == "single":
        out["pi_imb"] = np.asarray(pi_imb, float)
    else:
        out["pi_imb_up"] = np.asarray(pi_imb_up, float)
        out["pi_imb_down"] = np.asarray(pi_imb_down, float)
    return out


def regret(
    fp, p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_rel,
    realised, pl_hat, price_model, oracle_cost,
    pi_da, pi_imb=None, pi_imb_up=None, pi_imb_down=None,
    clip_recourse=True,
) -> torch.Tensor:
    """realised_cost - oracle_cost. oracle_cost is precomputed & detached upstream
    (perfect foresight) -> grad(regret) == grad(realised_cost)."""
    cost = realised_cost(
        fp, p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_rel,
        realised, pl_hat, price_model,
        pi_da=pi_da, pi_imb=pi_imb, pi_imb_up=pi_imb_up, pi_imb_down=pi_imb_down,
        clip_recourse=clip_recourse,
    )
    return cost - float(oracle_cost)


# =====================================================================================
# PART 3 -- the glue: PART 1 then PART 2, for one day
# =====================================================================================

def build_layer_vals(prices: dict, h_plus, h_minus, xi_samples=None, Sigma_xi_chol=None) -> dict:
    """Assemble this problem's cp.Parameter values, in the shape `order_param_values`
    expects. `prices` is whichever price dict (single -> {pi_da,pi_imb}; dual ->
    {pi_da,pi_imb_up,pi_imb_down}) this (price_model,k) problem actually declares."""
    vals = {**prices, "h_plus": h_plus, "h_minus": h_minus}
    if xi_samples is not None:
        vals["xi_samples"] = xi_samples
    if Sigma_xi_chol is not None:
        vals["Sigma_xi_chol"] = Sigma_xi_chol
    return vals


def setup(price_model: str, k: float, num_scenarios: int = 64, gamma: float = 1e-4):
    """WITHIN cvxpylayers, one-time setup: FixedParams + the compiled ProblemBundle +
    the differentiable CvxpyLayer. Call once per (price_model, k) corner; reuse the
    returned layer across every day/epoch -- compiling the DPP problem is the expensive
    part, solving it with new Parameter values is not."""
    fp = default_fixed_params(k=k, num_scenarios=num_scenarios, gamma=gamma)
    bundle = build_problem(fp, price_model)
    layer = make_layer(bundle)
    return fp, bundle, layer


def solve_and_realise(
    fp: FixedParams,
    bundle: ProblemBundle,
    layer,
    *,
    mean: torch.Tensor,                 # (T,) pl_hat -- THIS model's forecast mean
    xi: torch.Tensor,                   # (N,T) THIS model's scenario deviations from `mean`
    h_plus: torch.Tensor,               # (T,) frozen box, from the BASELINE forecaster
    h_minus: torch.Tensor,              # (T,)
    decision_prices: dict,              # forecast/proxy prices -- what the SOLVE sees
    settlement_prices: dict,            # TRUE prices -- what actually gets PAID
    realised: torch.Tensor,             # (T,) true realised prosumption for the day
    price_model: str,
    clip_recourse: bool = True,
) -> RealisedBreakdown:
    """The whole point-robust problem for one day, start to end.

    WITHIN cvxpylayers (PART 1): build this problem's Parameter values from `mean`/`xi`
    exactly as k and price_model gate them (xi_samples only if k<1 and dual,
    Sigma_xi_chol only if k>0), solve via `layer(*args)`, get back
    p_ch_hat/p_dis_hat/D_ch/D_dis with gradients wired to `mean`/`xi` via implicit
    differentiation of the KKT system (this is the ONLY point at which cvxpylayers'
    machinery is invoked).

    WITHOUT cvxpylayers (PART 2): reconstruct p_da_rel as a plain tensor expression
    (not a layer output), then hand everything to `realised_breakdown`, which runs
    entirely in ordinary PyTorch autograd, using `mean` directly as `pl_hat` and
    `settlement_prices` (TRUE prices) rather than `decision_prices` (the proxy).
    """
    k = fp.k

    # ---- PART 1: WITHIN cvxpylayers -------------------------------------------------
    xi_samples = xi if (k < 1.0 and price_model == "dual") else None
    Sigma_xi_chol = cholesky_of_second_moment(xi) if k > 0.0 else None

    vals = build_layer_vals(decision_prices, h_plus, h_minus,
                             xi_samples=xi_samples, Sigma_xi_chol=Sigma_xi_chol)
    args = order_param_values(bundle, **vals)
    p_ch_hat, p_dis_hat, D_ch, D_dis = layer(*args)

    # ---- PART 2: WITHOUT cvxpylayers (post-hoc, TRUE-price settlement) --------------
    p_da_rel = p_ch_hat - p_dis_hat

    bd = realised_breakdown(
        fp, p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_rel,
        realised=realised, pl_hat=mean, price_model=price_model,
        clip_recourse=clip_recourse, **settlement_prices,
    )
    return bd


def solve_and_score(
    fp: FixedParams,
    bundle: ProblemBundle,
    layer,
    *,
    mean, xi, h_plus, h_minus, decision_prices, settlement_prices, realised,
    price_model: str,
    oracle_cost: float,
    clip_recourse: bool = True,
) -> dict:
    """Convenience one-shot: solve_and_realise() then per_day_metrics() against a
    precomputed oracle_cost (perfect-foresight, detached). Returns the same
    seven-scalar dict evaluate.py reports per day."""
    bd = solve_and_realise(
        fp, bundle, layer,
        mean=mean, xi=xi, h_plus=h_plus, h_minus=h_minus,
        decision_prices=decision_prices, settlement_prices=settlement_prices,
        realised=realised, price_model=price_model, clip_recourse=clip_recourse,
    )
    return per_day_metrics(fp, bd, oracle_cost)
