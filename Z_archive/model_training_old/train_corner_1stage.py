"""Train the forecaster against the 1-STAGE dispatch model (mean forecast only, NO
recourse -- see dispatch_layer_1stage.py's module docstring and pivot_context.md for the
full justification). No `k`: there's no tracking term possible without D_ch/D_dis, so the
objective is always economic (matches the old k=0 branch's regret computation). Two
corners only (single, dual price model), not four -- k=0/k=1 doesn't apply here.

Reuses dispatch_wrapper.realised_breakdown/realised_cost/per_day_metrics UNCHANGED, via
D_ch=D_dis=torch.zeros((T,T)) -- with no recourse mechanism, p_ch_raw = p_ch_hat + 0@xi =
p_ch_hat, already feasible by the hard constraints, so the clip is a structural no-op:
this training loop cannot hit the dead-gradient-from-saturation pathology at all.

Usage: python train_corner_1stage.py <price_model: single|dual>
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import cvxpy as cp

import sys
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
from dispatch_setup import default_fixed_params_1stage as default_fixed_params, setup_1stage as build_problem
from dispatch_shared import make_layer, get_prices, oracle_price_values
from dispatch_layer import build_oracle, solve_oracle   # oracle is k/D_ch-independent, reuse as-is
from dispatch_wrapper import realised_cost, realised_breakdown

PRICE_COLS = ["da", "imb", "imb_up", "imb_down"]
# "_fc" = forecast-proxy columns (rolling-sigma + standardized-Student-t price uncertainty
# proxy, see 3_data_processing/power_prices.ipynb) -- used ONLY for the training-time
# decision solve in dfl_loss_batch. precompute_oracle_costs/oracle_price_values must keep
# using the true PRICE_COLS (their default) so regret stays anchored to real prices --
# that's the entire point of the proxy, so never pass PRICE_COLS_FC there.
PRICE_COLS_FC = ["da", "imb_fc", "imb_up_fc", "imb_down_fc"]
# built into a single window array so both `get_prices(..., cols=PRICE_COLS)` and
# `get_prices(..., cols=PRICE_COLS_FC)` can index into the same price_day slice --
# "da" isn't duplicated since PRICE_COLS_FC[0] == PRICE_COLS[0] == "da"
PRICE_COLS_ALL = PRICE_COLS + PRICE_COLS_FC[1:]
ISSUE_HOUR, HORIZON, N_HIST = 9, 24, 168
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 1e-4
TRAIN_SOLVER = "ECOS"
ORACLE_SOLVER = cp.GUROBI
QUANTILE_LEVELS_TENSOR = torch.as_tensor(QUANTILE_LEVELS, dtype=torch.float32, device=device)
EPS_BALANCE = 1e-6
FALLBACK_SOLVER = "SCS"


@dataclass
class TrainConfig:
    price_model: str
    lr: float = 5e-4
    batch_size: int = 16
    max_epochs: int = 10
    patience: int = 5
    min_delta: float = 0.0   # matches the baseline forecaster's own early-stopping (MIN_DELTA=0.0,
                              # "strict improvement required") -- absolute, not relative, so the
                              # bar for "improved" doesn't shrink as val_fsurr shrinks over training
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


def select_fc_columns(price_day):
    """get_prices() hardcodes the literal 'imb'/'imb_up'/'imb_down' keys INSIDE its
    body -- the `cols` argument only tells it where to find those literal names, not what to
    rename them to. Passing cols=PRICE_COLS_FC does not work: PRICE_COLS_FC contains
    'imb_up_fc' etc, not 'imb_up', so the internal idx["imb_up"] lookup raises
    KeyError (confirmed by testing this directly, not assumed from the signature).
    Fix without touching get_prices (still don't want the oracle's dependency touched):
    physically reorder price_day's fc columns into the same POSITIONS the default PRICE_COLS
    expects, then call get_prices with its default cols unchanged."""
    idx = {c: i for i, c in enumerate(PRICE_COLS_ALL)}
    fc_positions = [idx["da"], idx["imb_fc"], idx["imb_up_fc"], idx["imb_down_fc"]]
    return price_day[..., fc_positions]


def build_layer_vals(mean, prices, device):
    """pl_hat + price params, in bundle order. No xi/Sigma_xi_chol -- 1-stage has neither."""
    vals = {"pl_hat": mean}
    for kk, vv in prices.items():
        vals[kk] = torch.as_tensor(np.asarray(vv, float), dtype=mean.dtype, device=device)
    return vals


def dfl_loss_batch(batch_indices, *, model, fp, sampler, sc, windows, layer, keys, price_model,
                   device, fwd, oracle_costs):
    realised  = np.asarray(windows.y[batch_indices], float)
    price_day = np.asarray(windows.price[batch_indices], float)
    x_hist = windows.x_hist[batch_indices]
    x_fut = windows.x_fut[batch_indices]
    B = len(batch_indices)

    means_list, q_norm_list = [], []
    for i in range(B):
        q_phys_i, q_norm_i = forecast_train(model, sc, x_hist[i], x_fut[i], device,
                                     fwd["normalise_hist"], fwd["denormalise_y"], fwd["normalise_exo"])
        mean_i, _ = sampler.mean_and_errors(q_phys_i)   # only the mean is used -- no recourse, no xi
        means_list.append(mean_i); q_norm_list.append(q_norm_i)
    mean = torch.stack(means_list, dim=0)
    q_norm_batch = torch.stack(q_norm_list, dim=0)

    # DECISION solve: proxy prices only -- this is the whole point of the noise proxy,
    # don't let this drift to the true columns
    fc_prices = get_prices(select_fc_columns(price_day), price_model)
    vals = build_layer_vals(mean, fc_prices, device)
    args = [vals[name] for name in keys]
    y_true_norm = torch.as_tensor(normalise_y(realised, sc), dtype=torch.float32, device=device)

    # SETTLEMENT/cost evaluation: TRUE prices. Once a decision is made, what actually gets
    # paid is the real market price, not the proxy that informed the decision -- using fc
    # prices here too would compare raw_cost (at a randomly-sampled, often cheaper-or-pricier
    # proxy) against oracle_cost (at true price), an apples-to-oranges mismatch that can drive
    # f_dfl to 0 by accounting artifact alone, independent of decision quality.
    true_prices = get_prices(price_day, price_model)
    true_prices_t = {kk: torch.as_tensor(np.asarray(vv, float), dtype=mean.dtype, device=device)
                      for kk, vv in true_prices.items()}

    T = fp.T_total
    D_zero = torch.zeros((B, T, T), dtype=mean.dtype, device=device)

    def _combine(dec, realised_s, mean_s, prices_s, q_norm_s, y_true_s, oracle_s, Dz_s):
        L_base = pinball_per_day(y_true_s, q_norm_s, QUANTILE_LEVELS_TENSOR)
        raw_cost = realised_cost(fp, dec[0], dec[1], Dz_s, Dz_s, dec[2],
                            realised=realised_s, pl_hat=mean_s, price_model=price_model,
                            clip_recourse=True, **prices_s)
        oracle_t = torch.as_tensor(oracle_s, dtype=raw_cost.dtype, device=raw_cost.device)
        f_dfl = torch.clamp(raw_cost - oracle_t, min=0.0)
        sat = 0   # structurally impossible: p_ch_raw == p_ch_hat, already feasible -- no clip
        return self_balanced_loss(L_base, f_dfl), L_base, f_dfl, sat

    try:
        dec, inaccurate = solve_with_retry(layer, args)
        oracle_arr = np.asarray([oracle_costs[d] for d in batch_indices], float)
        per_day, L_base, f_dfl, sat = _combine(dec, realised, mean, true_prices_t, q_norm_batch,
                                                y_true_norm, oracle_arr, D_zero)
        return (per_day.sum(), B, sat, B * T,
                float(L_base.sum()), float(f_dfl.sum()), (B if inaccurate else 0))
    except cvxpylayers_solver_error():
        pass

    total = 0.0; n_survived = 0; L_base_total = 0.0; f_dfl_total = 0.0; n_inaccurate = 0
    for i in range(B):
        args_i = [v[i:i+1] for v in args]
        try:
            dec_i, inaccurate_i = solve_with_retry(layer, args_i)
        except cvxpylayers_solver_error():
            print(f"  SKIPPING day {batch_indices[i]}")
            continue
        if inaccurate_i:
            n_inaccurate += 1
        oracle_i = np.asarray([oracle_costs[batch_indices[i]]], float)
        prices_i = {kk: v[i:i+1] for kk, v in true_prices_t.items()}
        loss_i, L_base_i, f_dfl_i, _ = _combine(dec_i, realised[i:i+1], mean[i:i+1], prices_i,
                          q_norm_batch[i:i+1], y_true_norm[i:i+1], oracle_i, D_zero[i:i+1])
        total = total + loss_i.sum()
        L_base_total += float(L_base_i.sum()); f_dfl_total += float(f_dfl_i.sum())
        n_survived += 1
    if n_survived == 0:
        raise RuntimeError(f"all {B} days failed with both ECOS and SCS")
    return total, n_survived, 0, n_survived * T, L_base_total, f_dfl_total, n_inaccurate


def precompute_oracle_costs(windows, price_model, phys_fp):
    # true prices only (default cols=PRICE_COLS) -- the oracle/regret benchmark must never
    # see the _fc proxy columns, that would silently undo the point of the proxy
    ob = build_oracle(phys_fp, price_model, objective="economic")
    costs = np.empty(len(windows.delivery_start))
    for d in range(len(windows.delivery_start)):
        realised  = np.asarray(windows.y[d], float)
        price_day = np.asarray(windows.price[d], float)
        v = oracle_price_values(price_day, price_model, realised)
        costs[d] = solve_oracle(ob, v, solver=ORACLE_SOLVER)
    return costs


def evaluate_regret(*, model, fp, sampler, sc, windows, oracle_costs, layer, keys, price_model,
                    device, fwd):
    """Reuses dfl_loss_batch (one day per call) rather than re-deriving the same forward
    logic -- avoids duplicating/desyncing the args-ordering and tensor-conversion details
    dfl_loss_batch already gets right. dfl_loss_batch returns L_base/f_dfl sums directly
    (see its docstring-equivalent comment), so no algebraic inversion is needed."""
    model.eval()
    tot_combined = 0.0; tot_fsurr = 0.0; tot_base = 0.0; n_ok = 0; n_skipped = 0; tot_inaccurate = 0
    with torch.no_grad():
        for d in range(len(windows.delivery_start)):
            try:
                combined, n_survived, _, _, L_base_sum, f_dfl_sum, n_inaccurate = dfl_loss_batch(
                    [d], model=model, fp=fp, sampler=sampler, sc=sc, windows=windows,
                    layer=layer, keys=keys, price_model=price_model, device=device,
                    fwd=fwd, oracle_costs=oracle_costs)
            except RuntimeError:
                n_skipped += 1
                continue
            tot_combined += float(combined); tot_fsurr += f_dfl_sum; tot_base += L_base_sum
            tot_inaccurate += n_inaccurate
            n_ok += 1

    return tot_combined / n_ok, tot_fsurr / n_ok, tot_base / n_ok, n_skipped, tot_inaccurate


def train_one_config(cfg: TrainConfig, *, model, fp, sampler, sc, train_windows, val_windows,
                     oracle_costs_train, oracle_costs_val, device, fwd):
    torch.manual_seed(cfg.seed)
    bundle = build_problem(fp, f"{cfg.price_model}-price")   # 1-stage: no k, no dispatchability mode
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
        train_inaccurate = 0
        for start in range(0, n_train, cfg.batch_size):
            batch = order[start:start + cfg.batch_size]
            opt.zero_grad()
            try:
                batch_loss, n_survived, sat_h, n_h, _, _, n_inaccurate = dfl_loss_batch(batch.tolist(), model=model,
                                    fp=fp, sampler=sampler, sc=sc, windows=train_windows, layer=layer,
                                    keys=keys, price_model=cfg.price_model, device=device,
                                    fwd=fwd, oracle_costs=oracle_costs_train)
            except RuntimeError as e:
                print(f"  SKIPPING whole batch (start={start}): {e}")
                continue
            train_inaccurate += n_inaccurate
            (batch_loss / n_survived).backward()
            if cfg.grad_clip is not None:
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                grad_norms.append(float(pre_clip_norm))
            opt.step()

        val_combined, val_fsurr, val_base, n_skipped, val_inaccurate = evaluate_regret(
            model=model, fp=fp, sampler=sampler, sc=sc, windows=val_windows,
            oracle_costs=oracle_costs_val, layer=layer, keys=keys,
            price_model=cfg.price_model, device=device, fwd=fwd)
        grad_norms_arr = np.asarray(grad_norms) if grad_norms else np.asarray([0.0])
        history.append({"val_combined": val_combined, "val_fsurr": val_fsurr,
                        "val_base": val_base, "n_skipped": n_skipped,
                        "grad_norm_mean": float(grad_norms_arr.mean()),
                        "grad_norm_max": float(grad_norms_arr.max()),
                        "train_inaccurate": train_inaccurate, "val_inaccurate": val_inaccurate})
        improved = val_fsurr < best_val - cfg.min_delta
        print(f"[{cfg.price_model} 1stage] epoch {epoch:2d}  "
              f"val_fsurr={val_fsurr:.4f}  val_pinball={val_base:.4f}  "
              f"val_combined={val_combined:.4f}  skipped={n_skipped}  "
              f"grad_norm(mean/max)={grad_norms_arr.mean():.3f}/{grad_norms_arr.max():.3f}  "
              f"inaccurate(train/val)={train_inaccurate}/{val_inaccurate}  "
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
    assert pm_arg in ("single", "dual")

    t0 = time.time()
    print(f"CORNER_START price_model={pm_arg} (1-stage, no k)", flush=True)

    base  = pd.read_csv(DATA_DIR / "df_full.csv", parse_dates=["datetime"]).set_index("datetime")
    # imb_fc (the raw, pre-split proxy sample) isn't a stored column -- df_full.csv only
    # keeps the split imb_up_fc/imb_down_fc. Reconstructs exactly: since imb_up/imb_down
    # are {da, imb} as a set (whichever is larger/smaller), their sum is always da+imb,
    # so imb = imb_up + imb_down - da -- verified exactly against the true columns, and
    # the fc columns were built the same way (imb_up_fc=max(da,imb_fc) etc).
    base["imb_fc"] = base["imb_up_fc"] + base["imb_down_fc"] - base["da"]
    base  = reindex_and_impute(base, HIST_COLS, freq="1h", warn_gap=6)
    frame = build_features(base, feature_cols=FEAT_COLS, price_cols=PRICE_COLS_ALL)
    win_kw = dict(gate_aligned_only=True, issue_hour=ISSUE_HOUR, hist_cols=HIST_COLS,
                  exo_cols=EXO_COLS, target_col="prosumption", price_cols=PRICE_COLS_ALL,
                  n_hist=N_HIST, horizon=HORIZON)
    train_windows = make_windows(frame, y_range=(TRAIN_START, VAL_START - pd.Timedelta(hours=1)), **win_kw)
    val_windows   = make_windows(frame, y_range=(VAL_START, TEST_START - pd.Timedelta(hours=1)), **win_kw)
    print(f"n_train={len(train_windows.delivery_start)}  n_val={len(val_windows.delivery_start)}")

    ckpt = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt", weights_only=False, map_location="cpu")
    model = Baseline_Forecaster(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    sc = ckpt["scaler_stats"]

    import pickle
    cop = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    sampler = FrozenCopulaSampler(cop["Z_corr"], cop["quantile_levels"]).to(device)

    fp = default_fixed_params(gamma=GAMMA)
    fwd = {"normalise_hist": normalise_hist, "denormalise_y": denormalise_y, "normalise_exo": normalise_exo}

    print("precomputing oracle costs (train)...")
    oracle_train = precompute_oracle_costs(train_windows, pm_arg, fp)
    print("precomputing oracle costs (val)...")
    oracle_val = precompute_oracle_costs(val_windows, pm_arg, fp)

    cfg = TrainConfig(price_model=pm_arg)
    model, best_val, hist = train_one_config(cfg, model=model, fp=fp, sampler=sampler, sc=sc,
                                              train_windows=train_windows, val_windows=val_windows,
                                              oracle_costs_train=oracle_train, oracle_costs_val=oracle_val,
                                              device=device, fwd=fwd)

    elapsed = time.time() - t0
    print(f"\nCORNER_DONE price_model={pm_arg} best_val_fsurr={best_val:.4f} "
          f"epochs_run={len(hist)} elapsed_s={elapsed:.0f}", flush=True)
    out_path = DFL_TRAIN_DIR / f"dfl_1stage_{pm_arg}.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__,
                "best_val_fsurr": best_val, "val_history": hist}, out_path)
    print(f"CORNER_SAVED {out_path}")
