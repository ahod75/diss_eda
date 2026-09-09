"""Train ONE (price_model, k) corner to full convergence (max_epochs=50, patience=10,
min_delta=0.02 relative on val_fsurr) and save a checkpoint to 7_model_training/. Mirrors
dfl_train.ipynb's TrainConfig/train_one_config/driver-loop body exactly, but as a
standalone script so the orchestrator can invoke each corner as a FRESH process -- this
avoids the memory-accumulation/swap-thrashing issue seen when running multiple corners
sequentially inside one long-lived process.

Current state (matches notebook after the alpha-floor rollback + Sigma_xi_chol fix):
  - self_balanced_loss is the UNFLOORED paper formula (no ALPHA_MIN).
  - Early stopping / checkpoint selection keyed on val_fsurr (decision loss), not
    val_combined -- val_combined's weights self-collapse and can go numerically blind to
    calibration blowing up (see notebook cell docstrings for the single/k=0 incident).
  - k=1's tracking term uses Sigma_xi_chol (differentiable Cholesky of xi^T xi/N), not
    xi_samples -- ~16-24x faster per solve (measured both synthetically and on real data
    this session), same value by the trace identity, verified gradient still reaches the
    forecaster.

Usage: python train_corner.py <price_model: single|dual> <k: 0.0|1.0>
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
import pandas as pd
import torch
import cvxpy as cp

import sys
import pickle
import time
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
                         pinball_loss, QUANTILE_LEVELS, Baseline_Forecaster,
                         TRAIN_START, VAL_START, TEST_START,
                         HIST_COLS, FEAT_COLS, EXO_COLS)
from copula_lib import FrozenCopulaSampler
from dispatch_layer import (default_fixed_params, build_problem, make_layer,
                            build_oracle, solve_oracle)
from dispatch_wrapper import (get_prices, oracle_price_values, realised_cost, realised_breakdown,
                              cholesky_of_second_moment)
from diffcp import SolverError

PRICE_COLS = ["da", "imb", "up_reg_cost", "down_reg_cost"]
ISSUE_HOUR, HORIZON, N_HIST = 9, 24, 168
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SCEN = 64
GAMMA = 1e-4
TRAIN_SOLVER = "ECOS"
ORACLE_SOLVER = cp.GUROBI
QUANTILE_LEVELS_TENSOR = torch.as_tensor(QUANTILE_LEVELS, dtype=torch.float32, device=device)
EPS_BALANCE = 1e-6
FALLBACK_SOLVER = "SCS"


@dataclass
class TrainConfig:
    price_model: str
    k: float
    lr: float = 5e-4
    batch_size: int = 16
    max_epochs: int = 10
    patience: int = 5
    min_delta: float = 0.02
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


def build_layer_vals(mean, xi, prices, fp, device):
    vals = {"pl_hat": mean, "xi_samples": xi}
    if fp.k > 0.0:
        vals["Sigma_xi_chol"] = cholesky_of_second_moment(xi)
    for kk, vv in prices.items():
        vals[kk] = torch.as_tensor(np.asarray(vv, float), dtype=mean.dtype, device=device)
    return vals


def imbalance_loss(fp, bd):
    return (bd.p_imb ** 2).sum(dim=-1) * fp.dt


def pinball_per_day(y_true_norm, q_norm, levels):
    e = y_true_norm.unsqueeze(-1) - q_norm
    q = levels.view(1, 1, -1)
    return torch.maximum(q * e, (q - 1.0) * e).sum(dim=(1, 2)).to(torch.float64)


def self_balanced_loss(L_base, f_dfl, eps=EPS_BALANCE):
    denom = L_base.detach() + f_dfl.detach() + eps
    alpha = f_dfl.detach() / denom
    beta = 1.0 - alpha
    return alpha * L_base + beta * f_dfl


def solve_with_retry(layer, args):
    try:
        return layer(*args, solver_args={"solve_method": TRAIN_SOLVER})
    except SolverError:
        return layer(*args, solver_args={"solve_method": FALLBACK_SOLVER})


def dfl_loss_one_day(d, *, model, sampler, sc, windows, layer, keys, fp, price_model, device,
                     fwd, oracle_cost=None, return_components=False):
    realised  = np.asarray(windows.y[d], float)
    price_day = np.asarray(windows.price[d], float)
    q_phys, q_norm = forecast_train(model, sc, windows.x_hist[d], windows.x_fut[d], device,
                               fwd["normalise_hist"], fwd["denormalise_y"], fwd["normalise_exo"])
    mean, xi = sampler.mean_and_errors(q_phys)
    prices = get_prices(price_day, price_model)
    vals = build_layer_vals(mean, xi, prices, fp, device)
    args = [vals[name] for name in keys]
    dec = solve_with_retry(layer, args)

    y_true_norm = torch.as_tensor(normalise_y(realised, sc), dtype=torch.float32,
                                   device=device).unsqueeze(0)
    L_base = pinball_per_day(y_true_norm, q_norm.unsqueeze(0), QUANTILE_LEVELS_TENSOR).squeeze(0)

    if fp.k == 0.0:
        assert oracle_cost is not None
        raw_cost = realised_cost(fp, dec[0], dec[1], dec[2], dec[3], dec[4],
                             realised=realised, pl_hat=mean, price_model=price_model,
                             clip_recourse=True, **prices)
        f_dfl = torch.clamp(raw_cost - float(oracle_cost), min=0.0)
    else:
        bd = realised_breakdown(fp, dec[0], dec[1], dec[2], dec[3], dec[4],
                                realised=realised, pl_hat=mean, price_model=price_model,
                                clip_recourse=True, **prices)
        f_dfl = imbalance_loss(fp, bd)

    combined = self_balanced_loss(L_base, f_dfl)
    if return_components:
        return combined, L_base, f_dfl
    return combined


def dfl_loss_batch(batch_indices, *, model, sampler, sc, windows, layer, keys, fp, price_model,
                   device, fwd, oracle_costs=None):
    realised  = np.asarray(windows.y[batch_indices], float)
    price_day = np.asarray(windows.price[batch_indices], float)
    x_hist = windows.x_hist[batch_indices]
    x_fut = windows.x_fut[batch_indices]

    B = len(batch_indices)
    means_list, xis_list, q_norm_list = [], [], []
    for i in range(B):
        q_phys_i, q_norm_i = forecast_train(model, sc, x_hist[i], x_fut[i], device,
                                     fwd["normalise_hist"], fwd["denormalise_y"], fwd["normalise_exo"])
        mean_i, xi_i = sampler.mean_and_errors(q_phys_i)
        means_list.append(mean_i); xis_list.append(xi_i); q_norm_list.append(q_norm_i)
    mean = torch.stack(means_list, dim=0)
    xi = torch.stack(xis_list, dim=0)
    q_norm_batch = torch.stack(q_norm_list, dim=0)

    prices = get_prices(price_day, price_model)
    vals = build_layer_vals(mean, xi, prices, fp, device)
    args = [vals[name] for name in keys]
    y_true_norm = torch.as_tensor(normalise_y(realised, sc), dtype=torch.float32, device=device)

    def _combine(dec, realised_s, mean_s, prices_s, q_norm_s, y_true_s, oracle_s):
        L_base = pinball_per_day(y_true_s, q_norm_s, QUANTILE_LEVELS_TENSOR)
        sat_hours_t, n_hours_t = 0, 0   # only meaningful for k>0, where realised_breakdown's
                                         # clip is what kills f_dfl's local gradient -- k=0's
                                         # own clip use (inside realised_cost) is near-inert
                                         # given the already-established D_ch/D_dis~0 degeneracy
        if fp.k == 0.0:
            raw_cost = realised_cost(fp, dec[0], dec[1], dec[2], dec[3], dec[4],
                                realised=realised_s, pl_hat=mean_s, price_model=price_model,
                                clip_recourse=True, **prices_s)
            oracle_t = torch.as_tensor(oracle_s, dtype=raw_cost.dtype, device=raw_cost.device)
            f_dfl = torch.clamp(raw_cost - oracle_t, min=0.0)
        else:
            bd = realised_breakdown(fp, dec[0], dec[1], dec[2], dec[3], dec[4],
                                realised=realised_s, pl_hat=mean_s, price_model=price_model,
                                clip_recourse=True, **prices_s)
            f_dfl = imbalance_loss(fp, bd)
            with torch.no_grad():
                clip_ch  = (bd.p_ch_raw  - bd.p_ch_r ).abs()
                clip_dis = (bd.p_dis_raw - bd.p_dis_r).abs()
                sat_hours_t = int(((clip_ch > 1e-6) | (clip_dis > 1e-6)).sum().item())
                n_hours_t = clip_ch.numel()
        return self_balanced_loss(L_base, f_dfl), sat_hours_t, n_hours_t

    try:
        dec = solve_with_retry(layer, args)
        oracle_arr = (np.asarray([oracle_costs[d] for d in batch_indices], float)
                      if fp.k == 0.0 else None)
        per_day, sat_h, n_h = _combine(dec, realised, mean, prices, q_norm_batch, y_true_norm, oracle_arr)
        return per_day.sum(), B, sat_h, n_h
    except SolverError:
        pass

    total = 0.0; n_survived = 0; sat_h_total = 0; n_h_total = 0
    for i in range(B):
        args_i = [v[i:i+1] for v in args]
        try:
            dec_i = solve_with_retry(layer, args_i)
        except SolverError:
            print(f"  SKIPPING day {batch_indices[i]}")
            continue
        oracle_i = (np.asarray([oracle_costs[batch_indices[i]]], float) if fp.k == 0.0 else None)
        prices_i = {kk: v[i:i+1] for kk, v in prices.items()}
        loss_i, sat_h, n_h = _combine(dec_i, realised[i:i+1], mean[i:i+1], prices_i,
                          q_norm_batch[i:i+1], y_true_norm[i:i+1], oracle_i)
        total = total + loss_i.sum()
        n_survived += 1; sat_h_total += sat_h; n_h_total += n_h
    if n_survived == 0:
        raise RuntimeError(f"all {B} days failed with both ECOS and SCS")
    return total, n_survived, sat_h_total, n_h_total


def precompute_oracle_costs(windows, price_model, phys_fp):
    ob = build_oracle(phys_fp, price_model, objective="economic")
    costs = np.empty(len(windows.delivery_start))
    for d in range(len(windows.delivery_start)):
        realised  = np.asarray(windows.y[d], float)
        price_day = np.asarray(windows.price[d], float)
        v = oracle_price_values(price_day, price_model, realised)
        costs[d] = solve_oracle(ob, v, solver=ORACLE_SOLVER)
    return costs


def evaluate_regret(*, model, sampler, sc, windows, oracle_costs, layer, keys, fp,
                    price_model, device, fwd):
    model.eval()
    tot_combined = 0.0; tot_fsurr = 0.0; tot_base = 0.0; n_ok = 0; n_skipped = 0
    with torch.no_grad():
        for d in range(len(windows.delivery_start)):
            oc = float(oracle_costs[d]) if fp.k == 0.0 else None
            try:
                combined, L_base, f_dfl = dfl_loss_one_day(d, model=model, sampler=sampler,
                                        sc=sc, windows=windows, layer=layer, keys=keys, fp=fp,
                                        price_model=price_model, device=device, fwd=fwd,
                                        oracle_cost=oc, return_components=True)
            except SolverError:
                n_skipped += 1
                continue
            tot_combined += float(combined); tot_fsurr += float(f_dfl); tot_base += float(L_base)
            n_ok += 1
    return tot_combined / n_ok, tot_fsurr / n_ok, tot_base / n_ok, n_skipped


def train_one_config(cfg: TrainConfig, *, model, sampler, sc, train_windows, val_windows,
                     oracle_costs_train, oracle_costs_val, device, fwd):
    torch.manual_seed(cfg.seed)
    fp = default_fixed_params(cfg.k, num_scenarios=N_SCEN, gamma=GAMMA)
    bundle = build_problem(fp, cfg.price_model)
    layer = make_layer(bundle)
    keys = [p.name() for p in bundle.params]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_train = len(train_windows.delivery_start)
    best_val = float("inf"); best_state = copy.deepcopy(model.state_dict()); patience = 0
    history = []

    for epoch in range(cfg.max_epochs):
        model.train()
        order = np.random.permutation(n_train)
        grad_norms = []   # pre-clip total grad norm, one entry per optimiser step this epoch
        sat_hours_epoch = 0; n_hours_epoch = 0   # fraction of TRAINING (day,hour) pairs actively
                                                  # saturated -- direct evidence of how much of
                                                  # f_dfl's gradient is dead from the clip in
                                                  # realised_breakdown, vs guessing from theory
        for start in range(0, n_train, cfg.batch_size):
            batch = order[start:start + cfg.batch_size]
            opt.zero_grad()
            try:
                batch_loss, n_survived, sat_h, n_h = dfl_loss_batch(batch.tolist(), model=model,
                                    sampler=sampler, sc=sc, windows=train_windows, layer=layer,
                                    keys=keys, fp=fp, price_model=cfg.price_model, device=device,
                                    fwd=fwd, oracle_costs=oracle_costs_train)
            except RuntimeError as e:
                print(f"  SKIPPING whole batch (start={start}): {e}")
                continue
            sat_hours_epoch += sat_h; n_hours_epoch += n_h
            (batch_loss / n_survived).backward()
            if cfg.grad_clip is not None:
                # clip_grad_norm_ returns the PRE-clip total norm regardless of whether
                # clipping actually fired -- logging it directly answers "how often/how hard
                # is grad_clip biting" without needing a separate no-clip run to compare against.
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                grad_norms.append(float(pre_clip_norm))
            opt.step()

        val_combined, val_fsurr, val_base, n_skipped = evaluate_regret(
            model=model, sampler=sampler, sc=sc, windows=val_windows,
            oracle_costs=oracle_costs_val, layer=layer, keys=keys, fp=fp,
            price_model=cfg.price_model, device=device, fwd=fwd)
        grad_norms_arr = np.asarray(grad_norms) if grad_norms else np.asarray([0.0])
        n_clipped = int((grad_norms_arr > cfg.grad_clip).sum()) if cfg.grad_clip is not None else 0
        sat_frac_epoch = (sat_hours_epoch / n_hours_epoch) if n_hours_epoch else 0.0
        history.append({"val_combined": val_combined, "val_fsurr": val_fsurr,
                        "val_base": val_base, "n_skipped": n_skipped,
                        "grad_norm_mean": float(grad_norms_arr.mean()),
                        "grad_norm_max": float(grad_norms_arr.max()),
                        "n_batches": len(grad_norms_arr), "n_clipped": n_clipped,
                        "train_sat_hours": sat_hours_epoch, "train_n_hours": n_hours_epoch,
                        "train_sat_frac": sat_frac_epoch})
        # RELATIVE improvement on val_fsurr (decision loss), NOT val_combined -- see
        # notebook's TrainConfig.min_delta docstring for why.
        improved = val_fsurr < best_val * (1.0 - cfg.min_delta)
        print(f"[{cfg.price_model} k={int(cfg.k)}] epoch {epoch:2d}  "
              f"val_fsurr={val_fsurr:.4f}  val_pinball={val_base:.4f}  "
              f"val_combined={val_combined:.4f}  skipped={n_skipped}  "
              f"train_sat_frac={sat_frac_epoch:.3f}  "
              f"grad_norm(mean/max)={grad_norms_arr.mean():.3f}/{grad_norms_arr.max():.3f}  "
              f"clipped={n_clipped}/{len(grad_norms_arr)}  "
              f"{'*best' if improved else f'(patience {patience+1}/{cfg.patience})'}", flush=True)
        if improved:
            best_val = val_fsurr; best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                print(f"  early stop at epoch {epoch} (best val_fsurr={best_val:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_val, history


if __name__ == "__main__":
    pm_arg = sys.argv[1]
    k_arg = float(sys.argv[2])
    assert pm_arg in ("single", "dual")
    assert k_arg in (0.0, 1.0)

    t0 = time.time()
    print(f"CORNER_START price_model={pm_arg} k={k_arg}", flush=True)

    base  = pd.read_csv(DATA_DIR / "df_full.csv", parse_dates=["datetime"]).set_index("datetime")
    base  = reindex_and_impute(base, HIST_COLS, freq="1h", warn_gap=6)
    frame = build_features(base, feature_cols=FEAT_COLS)
    win_kw = dict(gate_aligned_only=True, issue_hour=ISSUE_HOUR, hist_cols=HIST_COLS,
                  exo_cols=EXO_COLS, target_col="prosumption", price_cols=PRICE_COLS,
                  n_hist=N_HIST, horizon=HORIZON)
    train_windows = make_windows(frame, y_range=(TRAIN_START, VAL_START - pd.Timedelta(hours=1)), **win_kw)
    val_windows   = make_windows(frame, y_range=(VAL_START,   TEST_START - pd.Timedelta(hours=1)), **win_kw)
    print(f"n_train={len(train_windows.delivery_start)}  n_val={len(val_windows.delivery_start)}", flush=True)

    ckpt = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt", weights_only=False, map_location="cpu")
    model = Baseline_Forecaster(**ckpt["model_config"]); model.load_state_dict(ckpt["state_dict"]); model.to(device)
    sc = ckpt["scaler_stats"]
    cop = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    sampler = FrozenCopulaSampler(cop["Z_corr"], cop["quantile_levels"]).to(device)
    fwd = {"normalise_hist": normalise_hist, "denormalise_y": denormalise_y, "normalise_exo": normalise_exo}
    phys_fp = default_fixed_params(0.0, num_scenarios=N_SCEN, gamma=GAMMA)

    cfg = TrainConfig(price_model=pm_arg, k=k_arg, grad_clip=5.0)

    if cfg.k == 0.0:
        print("precomputing oracle costs (train)...", flush=True)
        oracle_train = precompute_oracle_costs(train_windows, cfg.price_model, phys_fp)
        print("precomputing oracle costs (val)...", flush=True)
        oracle_val = precompute_oracle_costs(val_windows, cfg.price_model, phys_fp)
    else:
        oracle_train = None
        oracle_val = None

    model, best_val, hist = train_one_config(
        cfg, model=model, sampler=sampler, sc=sc,
        train_windows=train_windows, val_windows=val_windows,
        oracle_costs_train=oracle_train, oracle_costs_val=oracle_val,
        device=device, fwd=fwd)

    elapsed = time.time() - t0
    print(f"CORNER_DONE price_model={pm_arg} k={k_arg} best_val_fsurr={best_val:.4f} "
          f"epochs_run={len(hist)} elapsed_s={elapsed:.0f}", flush=True)

    out_path = DFL_TRAIN_DIR / f"dfl_{cfg.price_model}_k{int(cfg.k)}.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__,
                "best_val_fsurr": best_val, "val_history": hist}, out_path)
    print(f"CORNER_SAVED {out_path}", flush=True)
