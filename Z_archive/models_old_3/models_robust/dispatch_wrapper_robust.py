from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
import torch

from dispatch_layer_robust import FixedParams, cholesky_of_second_moment

# =====================================================================================
# dispatch_wrapper.py  --  the realised-metric ENGINE and the input builder.
#
#   compute_box(...)            -> frozen box half-widths from the BASELINE forecaster (B1)
#   make_dispatch_inputs(...)   -> named param-value dict for solve_plain / the layer
#   realised_breakdown(...)     -> the single source of truth for realised quantities
#   realised_cost(...)          -> thin C_da + C_imb wrapper on the breakdown (DFL loss)
#   per_day_metrics(...)        -> the seven per-day scalars
#   money_plot_series(...)      -> arrays for the bid / p^g / p_imb / price plots
#   regret(...)                 -> realised_cost - oracle_cost (arbitraging, price-aware)
#
# ALL realised quantities are post-saturation and computed against the SAME pl_hat anchor
# the layer used (xi_real = realised - pl_hat). Free bid:  bid = pl_hat + p_da_rel.
# =====================================================================================

_SAT_TOL = 1e-6

# -------------------------------------------------------------------------------------
# PLUG POINT: prices straight from the dataset columns (no computation in between).
# The dataset already stores the rectified reg costs, so the max{0,.} that would
# otherwise be an "outside the layer" step is precomputed upstream.
#   single -> {pi_da, pi_imb}          (signed settlement price 'imb')
#   dual   -> {pi_da, pi_imb_up, pi_imb_downpi_imb_up}  (rectified spread, already >= 0)
# -------------------------------------------------------------------------------------
PRICE_COLS = ("da", "imb", "imb_up", "imb_down")


def get_prices(price_day, price_model, cols=PRICE_COLS):
    """price_day: (T, 4) slice from WindowSet.price; columns in `cols` order."""
    price_day = np.asarray(price_day, dtype=float)
    idx = {c: i for i, c in enumerate(cols)}
    da = price_day[:, idx["da"]]
    if price_model == "single":
        return {"pi_da": da, "pi_imb": price_day[:, idx["imb"]]}   # pi_imb POSITIVE; sign is in p_imb
    if price_model == "dual":
        return {"pi_da": da,
                "imb_up": np.clip(price_day[:, idx["imb_up"]], 0.0, None),   # clamp = insurance
                "imb_down": np.clip(price_day[:, idx["imb_down"]], 0.0, None)}
    raise ValueError(price_model)


def oracle_price_values(price_day, price_model, realised, cols=PRICE_COLS):
    """Full param dict for solve_oracle: prices + p_d = realised prosumption (perfect foresight)."""
    d = get_prices(price_day, price_model, cols)
    d["p_d"] = np.asarray(realised, dtype=float)
    return d


def assert_price_consistency(price_all, cols=PRICE_COLS, tol=1e-2):
    """Run ONCE at load over the WHOLE dataset. Verifies the dual reg costs are the
    rectified da-imb spread, nonneg, and complementary. If the consistency check fails,
    up/down are independent series (not derived from imb) -- know that before framing
    single-vs-dual as 'same prices, two settlement rules'."""
    p = np.asarray(price_all, dtype=float).reshape(-1, len(cols))
    idx = {c: i for i, c in enumerate(cols)}
    da, imb = p[:, idx["da"]], p[:, idx["imb"]]
    up, dn = p[:, idx["imb_up"]], p[:, idx["imb_down"]]
    assert (up >= dn).all(), "imb_up must be always >= imb_down"

def _to_t(x, like: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float64))
    if like is not None:
        t = t.to(dtype=like.dtype, device=like.device)
    return t


# -------------------------------------------------------------------------------------
# BOX (B1): half-widths in xi = (realised - mean) coordinate, from the frozen BASELINE
# forecaster's quantiles. Computed ONCE per test day, cached, and passed VERBATIM to
# every (price_model, k) instance -> identical robust feasible set in xi-space.
#
# box_levels default = (0.15, 0.85), locked in from h_selection_sweep.py /
# h_selection_sweep_point.py's comparison (see h_selection_sweep.ipynb): dominates the
# previously-used (0.20, 0.80) on both the full LP-dual robust model and the
# point-constrained training surrogate (near-identical control-action cost, meaningfully
# lower saturation on both) -- verified on 2018 training-period data, both price models.
# -------------------------------------------------------------------------------------
def compute_box(baseline_quantiles, sampler, quantile_levels,
                box_levels=(0.15, 0.85), min_box=1e-4):
    """B1 frozen box from the BASELINE forecaster, for ONE day.

    baseline_quantiles: (K, Q) from the frozen reference (128) model, physical MW.
    Expressed against the baseline MEAN anchor (sampler.mean_and_errors) so that, applied
    in each model's own xi = (realised - mean) coordinate, the half-widths are identical
    everywhere (B1). Returns detached (h_plus, h_minus) -- constant across all models/days.
    Compute ONCE per test day and cache.
    """
    q = _to_t(baseline_quantiles)
    if q.dim() != 2:
        raise ValueError("baseline_quantiles must be (K, Q) for a single day")
    base_mean, _ = sampler.mean_and_errors(baseline_quantiles)      # SAME anchor convention
    base_mean = _to_t(base_mean, like=q)
    levels = np.asarray(quantile_levels, float)
    i_lo = int(np.argmin(np.abs(levels - box_levels[0])))
    i_hi = int(np.argmin(np.abs(levels - box_levels[1])))
    q_lower, q_upper = q[:, i_lo], q[:, i_hi]
    h_plus  = q_upper - base_mean
    h_minus = base_mean - q_lower
    with torch.no_grad():
        if (h_plus < 0).any() or (h_minus < 0).any():
            warnings.warn("baseline mean outside [q_lower, q_upper] on some lead(s) "
                          "(extreme skew); clamping half-width to min_box.")
    return (torch.clamp(h_plus,  min=min_box).detach(),
            torch.clamp(h_minus, min=min_box).detach())


# -------------------------------------------------------------------------------------
# INPUT BUILDER.  pl_hat, xi, Sigma come from THIS model's quantiles; the box (h_plus,
# h_minus) is the FROZEN baseline box (B1), passed in. Second moment is xi^T xi / N
# (NOT torch.cov) so the k=1 tracking equals the true SAA expected-squared-imbalance.
# Returns a NAMED dict of param values; solve_plain / order_param_values select the
# subset this (price_model, k) problem actually declares.
# -------------------------------------------------------------------------------------
def make_dispatch_inputs(
    quantiles,                 # (K, Q) THIS model's quantiles for one day (physical MW)
    sampler,                   # FrozenCopulaSampler (mean_and_errors); S must == num_scenarios
    price_model,               # "single" | "dual"
    k,                         # 0 | 1  (gates Sigma_xi_chol)
    pi_da,                     # (T,)
    h_plus, h_minus,           # (T,) FROZEN baseline box (B1) -- same for every model
    pi_imb=None,               # (T,) SIGNED single price       (single only)
    imb_up=None, imb_down=None,  # (T,) >=0 dual reg costs        (dual only; computed OUTSIDE)
    also_return_meta=False,
):
    if quantiles.dim() != 2:
        raise ValueError("quantiles must be (K, Q) for a single day")
    mean, xi = sampler.mean_and_errors(quantiles)      # mean (K,)=(T,), xi (S,K)=(N,T); grad kept
    pl_hat = mean

    vals = {
        "pl_hat":  pl_hat,
        "h_plus":  _to_t(h_plus,  like=xi),
        "h_minus": _to_t(h_minus, like=xi),
        "pi_da":   _to_t(pi_da,   like=xi),
    }
    if price_model == "single":
        if pi_imb is None:
            raise ValueError("single price needs pi_imb")
        vals["pi_imb"] = _to_t(pi_imb, like=xi)
    elif price_model == "dual":
        if imb_up is None or imb_down is None:
            raise ValueError("dual price needs pi_imb_up and pi_imb_down (computed outside as max{0,.})")
        vals["xi_samples"] = xi
        vals["pi_imb_up"] = _to_t(imb_up, like=xi)
        vals["pi_imb_down"] = _to_t(imb_down, like=xi)
    else:
        raise ValueError(price_model)

    if k > 0.0:
        # Cholesky of xi^T xi / N (biased /N). cholesky_of_second_moment expects numpy (N,T).
        L = cholesky_of_second_moment(xi.detach().cpu().numpy())
        vals["Sigma_xi_chol"] = _to_t(L, like=xi)

    if also_return_meta:
        return vals, {"pl_hat": pl_hat, "xi": xi}
    return vals


# =====================================================================================
# THE ENGINE.  One realised solve -> everything the metrics and money plot need.
# Runs in torch (serves the differentiable DFL loss AND the detached test path). At test,
# pass numpy decisions; _to_t lifts them (grad-free) and the reductions return floats.
# =====================================================================================
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

### This function is used for both training AND testing.
# For training, variables need to be passed through as tensors so that Pytorch's autograd
# system can keep track of gradients.
# This is unecessary for testing, but it makes it simpler to keep it all as one function.
# Makes it slightly slower, but keeps it simpler and neater.
def realised_breakdown(
    fp: FixedParams,
    p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_rel,   # the 5 decisions from the layer / solve
    realised,                                      # (T,) realised prosumption for the day
    pl_hat,                                        # (T,) SAME anchor the layer used
    price_model,                                   # "single" | "dual"
    pi_da, pi_imb=None, pi_imb_up=None, pi_imb_down=None,
    clip_recourse=True,                            # test: True; correctness gate uses False
) -> RealisedBreakdown:
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

    xi_real = realised - pl_hat                                    # (T,) realised error
    bid = pl_hat + p_da_rel                                        # FREE bid

    p_ch_raw  = p_ch_hat  + D_ch  @ xi_real                        # raw LDR actions
    p_dis_raw = p_dis_hat + D_dis @ xi_real

    if clip_recourse:
        # Stratigakos-like state-dependent saturation. SOC headroom caps carry /dt (dt-correct).
        soc = _to_t(fp.SOC0, like=p_ch_hat)
        ch_list, dis_list, soc_list = [], [], []
        C_ch  = _to_t(fp.C_ch,  like=p_ch_hat)
        C_dis = _to_t(fp.C_dis, like=p_ch_hat)
        B_max = _to_t(fp.B_max, like=p_ch_hat)
        z = _to_t(0.0, like=p_ch_hat)

        ## This runs for every time-step of the day, to ensure accurate tracking of battery SOC.
        for t in range(T):
            max_ch  = torch.minimum(C_ch,  (B_max - soc) / (fp.eta_ch * dt))
            max_dis = torch.minimum(C_dis, (soc * fp.eta_dis) / dt)
            pc = torch.clamp(torch.minimum(torch.clamp(p_ch_raw[t],  min=z), torch.clamp(max_ch,  min=z)), min=z)
            pd = torch.clamp(torch.minimum(torch.clamp(p_dis_raw[t], min=z), torch.clamp(max_dis, min=z)), min=z)
            soc = soc + dt * (fp.eta_ch * pc - (1.0 / fp.eta_dis) * pd)
            ch_list.append(pc); dis_list.append(pd); soc_list.append(soc)
        p_ch_r  = torch.stack(ch_list)
        p_dis_r = torch.stack(dis_list)
        soc_traj = torch.stack(soc_list)
    else:
        # no-clip: realised = raw; SOC from raw actions (used only for the in-box gate).
        p_ch_r, p_dis_r = p_ch_raw, p_dis_raw
        net = fp.eta_ch * p_ch_r - (1.0 / fp.eta_dis) * p_dis_r
        soc_traj = fp.SOC0 + dt * torch.cumsum(net, dim=0)

    ## Now accurate vectors have been produced, can aggregate them to realised values.
    p_g = realised + p_ch_r - p_dis_r                             # realised grid draw
    p_imb = p_g - bid                                            # = imb_det + R xi_real (in-box)

    C_da = (pi_da * bid).sum() * dt                              # includes pi_da . pl_hat (add-back)


    if price_model == "single":
        if pi_imb is None:
            raise ValueError("single needs pi_imb")
        C_imb = (_to_t(pi_imb, like=p_ch_hat) * p_imb).sum() * dt          # signed
    elif price_model == "dual":
        if pi_imb_up is None or pi_imb_down is None:
            raise ValueError("dual needs imb_up, imb_down")
        pi_imb_up = _to_t(pi_imb_up, like=p_ch_hat); pi_imb_down = _to_t(pi_imb_down, like=p_ch_hat)
        C_imb = (pi_imb_up * torch.clamp(p_imb, min=0.0)
                 + pi_imb_down * torch.clamp(-p_imb, min=0.0)).sum() * dt          # >= 0
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


# -------------------------------------------------------------------------------------
# THE SEVEN PER-DAY SCALARS (reductions over one breakdown).  Saturation is combined
# charge+discharge (per decision). oracle_cost is precomputed per (day, price_model).
# -------------------------------------------------------------------------------------
def per_day_metrics(fp: FixedParams, bd: RealisedBreakdown, oracle_cost: float) -> dict:
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


# -------------------------------------------------------------------------------------
# MONEY-PLOT SERIES (per representative day). The bid/p^g gap (bid is now pinned to
# pl_hat + p_ch_hat - p_dis_hat, not bounded arbitrage) is the single-vs-dual story;
# the price panel is what makes the bid behaviour readable.
# -------------------------------------------------------------------------------------
def money_plot_series(fp, bd: RealisedBreakdown, price_model,
                      pi_da, pi_imb=None, pi_imb_up=None, pi_imb_down=None) -> dict:
    to_np = lambda t: t.detach().cpu().numpy()
    out = {
        "bid":    to_np(bd.bid),
        "p_g":    to_np(bd.p_g),          # realised grid draw (post-saturation)
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


# -------------------------------------------------------------------------------------
# REGRET (corrected oracle caller: arbitraging, price-aware, bid-clamped in build_oracle).
# oracle_solve is the closure from dispatch_layer.solve_oracle wrapped per price_model.
# The oracle term is detached (perfect foresight) -> grad(regret) == grad(realised_cost).
# -------------------------------------------------------------------------------------
def regret(
    fp, p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_rel,
    realised, pl_hat, price_model, oracle_cost,
    pi_da, pi_imb=None, pi_imb_up=None, pi_imb_down=None,
    clip_recourse=True,
) -> torch.Tensor:
    cost = realised_cost(
        fp, p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_rel,
        realised, pl_hat, price_model,
        pi_da=pi_da, pi_imb=pi_imb, pi_imb_up=pi_imb_up, pi_imb_down=pi_imb_down,
        clip_recourse=clip_recourse,
    )
    return cost - float(oracle_cost)      # oracle_cost precomputed & detached upstream
