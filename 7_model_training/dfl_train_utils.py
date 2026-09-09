"""dfl_train_utils.py -- every helper function/constant shared by ALL DFL training runs.
Only architecture=="1stage" is supported now (point_robust training was removed
entirely, not merged in here -- see dispatch_setup.py's module comment; evaluation still
uses setup_full_robust directly, unaffected by this). Modes: single-price | dual-price
(NOT dispatchability -- setup_1stage/build_objective_1stage raise on it, no recourse
mechanism to build a tracking term from). train_dfl_forecasts.py is the only thing that
calls into this file (formerly train_dfl_models.py, retired -- see
train_dfl_forecasts.py's own docstring for why "balanced" was folded into its archetype
grid instead of kept separate); this file has no __main__ of its own and runs nothing on
import besides its own top-level constant/Parameter-tensor setup.

Split out of the former train_corner.py (single-corner CLI script, now removed) once
the former train_dfl_models.py took over running every corner in one process --
everything that used to live in train_corner.py's module body EXCEPT its __main__ block
lives here unchanged. All-corners-in-one-process was chosen over one-subprocess-per-corner
(the even older run_all_corners.py, also removed) for the shared-data speedup (training
data/baseline model loaded once instead of once per corner) at the cost of losing true
process isolation -- see train_dfl_forecasts.py's per-corner try/except + gc.collect for
how that tradeoff is managed now.

ORACLE REMOVED from training entirely (was: f_dfl = clamp(raw_cost - oracle_cost, 0)).
Now: economic modes (single-price/dual-price) train directly on
clamp(realised C_da + C_imb, min=0) -- no oracle solve, no fc/true price-basis mismatch
to keep straight for it. Dispatchability trains on clamp(dt**2 * sum_squares(realised
p_imb), min=0) -- always >=0 already, the clamp is a no-op there, kept only for a single
uniform code path. Both are computed directly off dispatch_wrapper.realised_breakdown's
`bd` (not via the realised_cost/realised_imbalance wrapper functions) since `bd` is
already needed here for the saturation diagnostics -- calling those wrappers too would
re-run realised_breakdown's per-timestep recourse-clipping loop a second time for
nothing. self_balanced_loss keeps its own clamp(min=0) unchanged (not switched to an
abs()-weighted version): empirically, negative total_cost is rare enough on this
dataset (0.55% of days for baseline/single/k0 on the full test set) that the lost
gradient on profitable days is an acceptable simplification.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import time
import warnings
import numpy as np
import torch

import sys
from pyprojroot import here

ROOT_DIR = here()
FORECASTING_DIR = ROOT_DIR / "4_forecasting"
DATA_DIR = ROOT_DIR / "1_data" / "processed"
COPULA_DIR = ROOT_DIR / "5_scenario_gen"
MODEL_DIR = ROOT_DIR / "6_models"
DFL_TRAIN_DIR = ROOT_DIR / "7_model_training"

sys.path.insert(0, str(FORECASTING_DIR))
sys.path.insert(0, str(COPULA_DIR))
sys.path.insert(0, str(MODEL_DIR))

from forecasting import (reindex_and_impute, build_features, make_windows,
                         normalise_hist, normalise_exo, denormalise_y, normalise_y,
                         QUANTILE_LEVELS, Baseline_Forecaster,
                         TRAIN_START, VAL_START, TEST_START,
                         HIST_COLS, FEAT_COLS, EXO_COLS)
from copula_lib import FrozenCopulaSampler
from dispatch_setup import default_fixed_params_1stage, setup_1stage
from dispatch_shared import (make_layer, get_prices, select_fc_columns,
                              cholesky_of_second_moment, price_model_for_settlement, build_layer_vals)
from dispatch_wrapper import realised_breakdown

PRICE_COLS = ["da", "imb", "imb_up", "imb_down"]
PRICE_COLS_FC = ["da_fc", "imb_fc", "imb_up_fc", "imb_down_fc"]   # all 4 real, always->=0 columns
PRICE_COLS_ALL = PRICE_COLS + PRICE_COLS_FC
ISSUE_HOUR, HORIZON, N_HIST = 9, 24, 168
device = torch.device("cpu")

GAMMA = 1e-4
TRAIN_SOLVER = "ECOS"
QUANTILE_LEVELS_TENSOR = torch.as_tensor(QUANTILE_LEVELS, dtype=torch.float32, device=device)
EPS_BALANCE = 1e-6
FALLBACK_SOLVER = "SCS"
SAT_TOL = 1e-6


@dataclass
class TrainConfig:
    architecture: str
    mode: str
    lr: float = 5e-4
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 10
    grad_clip: float = 3.0
    seed: int = 20240801


def forecast_train(model, sc, x_hist_day, x_fut_day, device, normalise_hist, denormalise_y,
                    normalise_exo):
    xh = normalise_hist(np.asarray(x_hist_day), sc)
    xh = torch.as_tensor(xh, dtype=torch.float32, device=device).unsqueeze(0)
    xf = normalise_exo(np.asarray(x_fut_day), sc)
    xf = torch.as_tensor(xf, dtype=torch.float32, device=device).unsqueeze(0)
    q_norm = model(xh, xf)
    q_phys = denormalise_y(q_norm, sc)
    return q_phys.squeeze(0).to(torch.float64), q_norm.squeeze(0)


def pinball_per_day(y_true_norm, q_norm, levels):
    e = y_true_norm.unsqueeze(-1) - q_norm
    q = levels.view(1, 1, -1)
    return torch.maximum(q * e, (q - 1.0) * e).sum(dim=(1, 2)).to(torch.float64)


def self_balanced_loss(L_base, f_dfl, eps=EPS_BALANCE):
    denom = L_base.detach() + f_dfl.detach() + eps
    alpha = f_dfl.detach() / denom
    beta = 1.0 - alpha
    return alpha * L_base + beta * f_dfl


def _combine_loss(dec, realised_s, mean_s, prices_s, q_norm_s, y_true_s, *,
            mode, fp, T, price_model_str):
    """Turns one solved decision into (self_balanced loss, L_base, f_dfl, sat_h) --
    pulled out of dfl_loss_batch as its own top-level function (it was a nested closure
    there, capturing mode/fp/T/price_model_str from the enclosing scope; now takes them
    as explicit keyword args instead). Still only ever called from dfl_loss_batch's two
    solve paths (whole-batch, and the per-day fallback), never directly -- the leading
    underscore stays to signal that.

    1stage only -- no recourse, so D_ch/D_dis are zero (structural no-op in
    realised_breakdown)."""
    L_base = pinball_per_day(y_true_s, q_norm_s, QUANTILE_LEVELS_TENSOR)
    p_ch_hat, p_dis_hat, p_da_bat = dec
    D_ch = torch.zeros((p_ch_hat.shape[0], T, T), dtype=p_ch_hat.dtype, device=p_ch_hat.device)
    D_dis = D_ch
    bd = realised_breakdown(fp, p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_bat,
                            realised=realised_s, pl_hat=mean_s, price_model=price_model_str,
                            clip_recourse=True, **prices_s)
    if mode == "dispatchability":
        raw = fp.dt ** 2 * (bd.p_imb ** 2).sum(dim=-1)   # always >=0 -- clamp below is a no-op
    else:
        raw = bd.C_da + bd.C_imb
    f_dfl = torch.clamp(raw, min=0.0)
    clip_ch = torch.abs(bd.p_ch_raw - bd.p_ch_r)
    clip_dis = torch.abs(bd.p_dis_raw - bd.p_dis_r)
    sat_h = int(((clip_ch > SAT_TOL) | (clip_dis > SAT_TOL)).sum().item())
    return self_balanced_loss(L_base, f_dfl), L_base, f_dfl, sat_h


def solve_with_retry(layer, args):
    """Returns (decisions, inaccurate). `inaccurate` is True if either solve attempt
    completed with a "Solved/Inaccurate" status (or equivalent) -- logged but NOT
    treated as a failure: diffcp only raises SolverError on hard failures (infeasible/
    unbounded/crashed), never on "solved but numerically imprecise", so this is the
    only way to see it happened at all."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            dec = layer(*args, solver_args={"solve_method": TRAIN_SOLVER})
        except cvxpylayers_solver_error():
            dec = layer(*args, solver_args={"solve_method": FALLBACK_SOLVER})
        inaccurate = any("Inaccurate" in str(w.message) for w in caught)
    return dec, inaccurate


def cvxpylayers_solver_error():
    from diffcp import SolverError
    return SolverError


def make_fp(architecture: str, gamma: float):
    if architecture == "1stage":
        return default_fixed_params_1stage(gamma=gamma)
    raise ValueError(f"only 'architecture'==\"1stage\" is supported -- got {architecture!r} "
                      f"(point_robust training was removed; evaluation still uses "
                      f"setup_full_robust directly, not this function)")


def make_bundle(architecture: str, fp, mode: str):
    if architecture == "1stage":
        return setup_1stage(fp, mode)
    raise ValueError(f"only 'architecture'==\"1stage\" is supported -- got {architecture!r} "
                      f"(point_robust training was removed; evaluation still uses "
                      f"setup_full_robust directly, not this function)")


def dfl_loss_batch(batch_indices, *, model, fp, sampler, sc, windows, layer, keys,
                   mode, device, fwd):
    realised  = np.asarray(windows.y[batch_indices], float)
    price_day = np.asarray(windows.price[batch_indices], float)
    x_hist = windows.x_hist[batch_indices]
    x_fut = windows.x_fut[batch_indices]
    B = len(batch_indices)
    T = fp.T_total

    means_list, xi_list, q_norm_list = [], [], []
    for i in range(B):
        q_phys_i, q_norm_i = forecast_train(model, sc, x_hist[i], x_fut[i], device,
                                     fwd["normalise_hist"], fwd["denormalise_y"], fwd["normalise_exo"])
        mean_i, xi_i = sampler.mean_and_errors(q_phys_i)
        means_list.append(mean_i); xi_list.append(xi_i); q_norm_list.append(q_norm_i)
    mean = torch.stack(means_list, dim=0)
    xi = torch.stack(xi_list, dim=0)
    q_norm_batch = torch.stack(q_norm_list, dim=0)

    # 1stage has no h_plus/h_minus Parameters at all -- always None.
    h_plus = h_minus = None

    price_model_str = price_model_for_settlement(mode)

    # DECISION solve: proxy (_fc) prices only, always >=0 by construction -- never let
    # this drift to the true columns, that's the whole point of the proxy.
    fc_price_day = select_fc_columns(price_day, cols=PRICE_COLS_ALL, real_cols=PRICE_COLS_FC)
    fc_prices = get_prices(fc_price_day, price_model_str, cols=PRICE_COLS)
    fc_prices_t = {kk: torch.as_tensor(np.asarray(vv, float), dtype=mean.dtype, device=device)
                for kk, vv in fc_prices.items()}

    xi_samples = xi if mode == "dual-price" else None
    Sigma_xi_chol = cholesky_of_second_moment(xi) if mode == "dispatchability" else None

    vals = build_layer_vals(fc_prices_t, h_plus=h_plus, h_minus=h_minus,
                             xi_samples=xi_samples, Sigma_xi_chol=Sigma_xi_chol)
    args = [vals[name] for name in keys]
    y_true_norm = torch.as_tensor(normalise_y(realised, sc), dtype=torch.float32, device=device)

    # SETTLEMENT: TRUE prices always. Once a decision is made, what actually gets paid
    # is the real market price, not the proxy that informed the decision.
    true_prices = get_prices(price_day, price_model_str, cols=PRICE_COLS)
    true_prices_t = {kk: torch.as_tensor(np.asarray(vv, float), dtype=mean.dtype, device=device)
                      for kk, vv in true_prices.items()}

    try:
        t0 = time.perf_counter()
        dec, inaccurate = solve_with_retry(layer, args)
        t_forward = time.perf_counter() - t0
        per_day, L_base, f_dfl, sat_h = _combine_loss(dec, realised, mean, true_prices_t, q_norm_batch, y_true_norm,
                                                 mode=mode, fp=fp, T=T,
                                                 price_model_str=price_model_str)
        return (per_day.sum(), B, sat_h, B * T,
                float(L_base.sum()), float(f_dfl.sum()), (B if inaccurate else 0), t_forward)
    except cvxpylayers_solver_error():
        pass

    total = 0.0; n_survived = 0; L_base_total = 0.0; f_dfl_total = 0.0; sat_total = 0; n_inaccurate = 0
    t_forward_total = 0.0
    for i in range(B):
        args_i = [v[i:i + 1] for v in args]
        try:
            t0 = time.perf_counter()
            dec_i, inaccurate_i = solve_with_retry(layer, args_i)
            t_forward_total += time.perf_counter() - t0
        except cvxpylayers_solver_error():
            print(f"  SKIPPING day {batch_indices[i]}")
            continue
        if inaccurate_i:
            n_inaccurate += 1
        prices_i = {kk: v[i:i + 1] for kk, v in true_prices_t.items()}
        loss_i, L_base_i, f_dfl_i, sat_i = _combine_loss(dec_i, realised[i:i + 1], mean[i:i + 1], prices_i,
                          q_norm_batch[i:i + 1], y_true_norm[i:i + 1],
                          mode=mode, fp=fp, T=T, price_model_str=price_model_str)
        total = total + loss_i.sum()
        L_base_total += float(L_base_i.sum()); f_dfl_total += float(f_dfl_i.sum())
        sat_total += sat_i
        n_survived += 1
    if n_survived == 0:
        raise RuntimeError(f"all {B} days failed with both ECOS and SCS")
    return total, n_survived, sat_total, n_survived * T, L_base_total, f_dfl_total, n_inaccurate, t_forward_total


def evaluate_regret(*, model, fp, sampler, sc, windows, layer, keys, mode, device, fwd):
    model.eval()
    tot_combined = 0.0; tot_fsurr = 0.0; tot_base = 0.0; n_ok = 0; n_skipped = 0
    tot_sat_h = 0; tot_n_h = 0; tot_inaccurate = 0; tot_t_forward = 0.0
    with torch.no_grad():
        for d in range(len(windows.delivery_start)):
            try:
                combined, n_survived, sat_h, n_h, L_base_sum, f_dfl_sum, n_inaccurate, t_forward = dfl_loss_batch(
                    [d], model=model, fp=fp, sampler=sampler, sc=sc, windows=windows,
                    layer=layer, keys=keys, mode=mode, device=device, fwd=fwd)
            except RuntimeError:
                n_skipped += 1
                continue
            tot_combined += float(combined); tot_fsurr += f_dfl_sum; tot_base += L_base_sum
            tot_sat_h += sat_h; tot_n_h += n_h; tot_inaccurate += n_inaccurate
            tot_t_forward += t_forward
            n_ok += 1

    sat_frac = tot_sat_h / tot_n_h if tot_n_h > 0 else 0.0
    return tot_combined / n_ok, tot_fsurr / n_ok, tot_base / n_ok, n_skipped, sat_frac, tot_inaccurate, tot_t_forward


def train_one_config(cfg: TrainConfig, *, model, fp, sampler, sc, train_windows, val_windows,
                     device, fwd, out_path=None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # DECISION bundle is always built as single-price, even when cfg.mode=="dual-price".
    # For 1-stage, dual-price's epigraph (p_plus/p_minus/xi_samples,
    # dispatch_objectives._build_dual_price_epigraph) is additively separable from the
    # battery decision variables: build_objective_1stage returns C_da(battery) +
    # C_imb(epigraph) + penalty(battery), and the epigraph's only constraint,
    # p_plus-p_minus==xi_samples.T, never references p_ch_hat/p_dis_hat/p_da_bat/s_hat --
    # verified directly against dispatch_objectives.py, not assumed. No shared
    # constraint + a separable objective means the KKT system is block-diagonal, so
    # cvxpylayers' implicit differentiation gives IDENTICAL forward decisions AND
    # gradients for the battery variables whether or not the epigraph is included
    # (already verified empirically too -- dual_price_gradient_mechanism.md's
    # single-vs-dual decision comparison differs by ~1e-9, solver noise only). f_dfl
    # itself never touches the epigraph either: realised_breakdown takes only the
    # battery decision + true prices. So the epigraph is provably inert for training --
    # building the decision LP without it changes nothing except solve/backward cost,
    # which is exactly the point: dual-price's LP scales with num_scenarios=64 for no
    # training-relevant reason. Mode-specific behaviour (which settlement formula
    # f_dfl uses) is entirely preserved below via cfg.mode, untouched by this.
    decision_mode = "single-price" if cfg.architecture == "1stage" else cfg.mode
    bundle = make_bundle(cfg.architecture, fp, decision_mode)
    layer = make_layer(bundle)
    keys = [p.name() for p in bundle.params]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_train = len(train_windows.delivery_start)
    best_val = float("inf"); best_state = copy.deepcopy(model.state_dict()); patience = 0
    history = []

    for epoch in range(cfg.max_epochs):
        model.train()
        order = np.random.permutation(n_train)
        grad_norms = []
        sat_hours_epoch = 0; n_hours_epoch = 0; train_inaccurate = 0
        t_forward_epoch = 0.0; t_backward_epoch = 0.0
        for start in range(0, n_train, cfg.batch_size):
            batch = order[start:start + cfg.batch_size]
            opt.zero_grad()
            try:
                batch_loss, n_survived, sat_h, n_h, _, _, n_inaccurate, t_forward = dfl_loss_batch(
                    batch.tolist(), model=model, fp=fp, sampler=sampler, sc=sc,
                    windows=train_windows, layer=layer, keys=keys,
                    mode=cfg.mode, device=device, fwd=fwd)
            except RuntimeError as e:
                print(f"  SKIPPING whole batch (start={start}): {e}")
                continue
            train_inaccurate += n_inaccurate
            t_forward_epoch += t_forward
            t0 = time.perf_counter()
            (batch_loss / n_survived).backward()
            t_backward_epoch += time.perf_counter() - t0
            if cfg.grad_clip is not None:
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                grad_norms.append(float(pre_clip_norm))
            opt.step()
            sat_hours_epoch += sat_h; n_hours_epoch += n_h

        val_combined, val_fsurr, val_base, n_skipped, val_sat_frac, val_inaccurate, val_t_forward = evaluate_regret(
            model=model, fp=fp, sampler=sampler, sc=sc, windows=val_windows,
            layer=layer, keys=keys, mode=cfg.mode, device=device, fwd=fwd)
        grad_norms_arr = np.asarray(grad_norms) if grad_norms else np.asarray([0.0])
        train_sat_frac = sat_hours_epoch / n_hours_epoch if n_hours_epoch > 0 else 0.0
        history.append({"val_combined": val_combined, "val_fsurr": val_fsurr,
                        "val_base": val_base, "n_skipped": n_skipped,
                        "grad_norm_mean": float(grad_norms_arr.mean()),
                        "grad_norm_max": float(grad_norms_arr.max()),
                        "train_sat_frac": train_sat_frac, "val_sat_frac": val_sat_frac,
                        "train_inaccurate": train_inaccurate, "val_inaccurate": val_inaccurate,
                        "t_forward_train": t_forward_epoch, "t_backward_train": t_backward_epoch,
                        "t_forward_val": val_t_forward})
        # strict improvement required (matches the baseline forecaster's own early-
        # stopping convention) -- no min_delta tolerance, models here are only ever
        # trained with one, so a separate field for it would be dead configuration.
        improved = val_fsurr < best_val
        print(f"[{cfg.architecture} {cfg.mode}] epoch {epoch:2d}  "
              f"val_fsurr={val_fsurr:.4f}  val_pinball={val_base:.4f}  "
              f"val_combined={val_combined:.4f}  skipped={n_skipped}  "
              f"grad_norm(mean/max)={grad_norms_arr.mean():.3f}/{grad_norms_arr.max():.3f}  "
              f"sat_frac(train/val)={train_sat_frac:.3f}/{val_sat_frac:.3f}  "
              f"inaccurate(train/val)={train_inaccurate}/{val_inaccurate}  "
              f"t_fwd(train/val)={t_forward_epoch:.2f}s/{val_t_forward:.2f}s  t_bwd(train)={t_backward_epoch:.2f}s  "
              f"{'*best' if improved else f'(patience {patience+1}/{cfg.patience})'}", flush=True)
        if improved:
            best_val = val_fsurr; best_state = copy.deepcopy(model.state_dict()); patience = 0
            if out_path is not None:
                torch.save({"state_dict": best_state, "cfg": cfg.__dict__,
                            "best_val_fsurr": best_val, "epoch": epoch}, out_path)
                print(f"  checkpoint saved (epoch {epoch}, val_fsurr={best_val:.4f}) -> {out_path}", flush=True)
        else:
            patience += 1
            if patience >= cfg.patience:
                print(f"  early stop at epoch {epoch} (best val_fsurr={best_val:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_val, history
