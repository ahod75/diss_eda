"""train_crps_only.py -- the missing control for the headline DFL results: fine-tunes a
FRESH copy of the baseline weights (same warm-start every DFL corner uses) on pinball loss
ALONE, no economic term, no dispatch solve, no cvxpylayers -- to isolate how much of DFL's
apparent benefit is just "more training", not "training against the real decision loss".

Deliberately NOT wired into dfl_train_utils.py's self_balanced_loss/_combine_loss path:
self_balanced_loss(L_base, f_dfl) = alpha*L_base + beta*f_dfl with alpha=f_dfl/(L_base+
f_dfl+eps) -- forcing f_dfl=0 there makes alpha=0, beta=1, so the combined loss collapses
to 0*L_base + 1*0 = 0 identically. Zero gradient reaches the model. That's why this is a
separate, minimal script rather than a flag threaded through the existing loss combiner --
correctness here means bypassing self_balanced_loss entirely, not feeding it a zero.

Price-mode-independent (pinball loss never touches price), so ONE checkpoint here serves
as the control for both single-price and dual-price evaluation -- see eval_raw.FORECASTERS.

Same training budget/hyperparameters as every DFL corner (TrainConfig's defaults:
lr=5e-4, batch_size=64, max_epochs=200, patience=10, grad_clip=3.0, seed=20240801 --
mirrors the original baseline forecaster's own training convention), so the
comparison to DFL checkpoints is "same effort, no task loss" rather than a shorter/noisier
run. Early-stops on validation pinball loss (the natural analogue of DFL's val_fsurr,
since there's no f_dfl here to stop on).

Usage: python train_crps_only.py [--seed SEED]
Seed 20240801 (the project default, matching every DFL corner's warm-start seed) writes
the unsuffixed crps_only_retrained.pt every other consumer (eval_raw.FORECASTERS,
aggregate_results.ipynb, forecast_characteristics.ipynb) already expects. Any other seed
writes crps_only_retrained_seed{SEED}.pt alongside it -- used for seed-averaging, see
9_analysis/why_dfl_helps.md Section 7.
"""
from __future__ import annotations
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch

from dfl_train_utils import (
    TRAIN_START, VAL_START, TEST_START, HIST_COLS, FEAT_COLS, EXO_COLS, PRICE_COLS_ALL,
    ISSUE_HOUR, HORIZON, N_HIST, reindex_and_impute, build_features, make_windows,
    normalise_hist, normalise_exo, normalise_y, QUANTILE_LEVELS_TENSOR, pinball_per_day,
    DATA_DIR, FORECASTING_DIR, DFL_TRAIN_DIR,
)
from forecasting import Baseline_Forecaster

LR, BATCH_SIZE, MAX_EPOCHS, PATIENCE, GRAD_CLIP, DEFAULT_SEED = 5e-4, 64, 200, 10, 3.0, 20240801
device = torch.device("cpu")


def out_path_for(seed: int):
    suffix = "" if seed == DEFAULT_SEED else f"_seed{seed}"
    return DFL_TRAIN_DIR / f"crps_only_retrained{suffix}.pt"


def forecast(model, sc, x_hist_day, x_fut_day):
    xh = torch.as_tensor(normalise_hist(np.asarray(x_hist_day), sc), dtype=torch.float32, device=device).unsqueeze(0)
    xf = torch.as_tensor(normalise_exo(np.asarray(x_fut_day), sc), dtype=torch.float32, device=device).unsqueeze(0)
    return model(xh, xf).squeeze(0)  # (K, Q) normalised


def epoch_pinball(model, sc, windows, day_ids, train: bool, opt=None):
    """One pass over day_ids. Trains (batched, grad-accumulated per BATCH_SIZE days) if
    train else just reports mean pinball loss."""
    model.train(train)
    total, n = 0.0, 0
    order = np.random.permutation(day_ids) if train else day_ids
    for start in range(0, len(order), BATCH_SIZE if train else len(order)):
        batch = order[start:start + BATCH_SIZE] if train else order
        if train:
            opt.zero_grad()
        batch_loss = 0.0
        for d in batch:
            q_norm = forecast(model, sc, windows.x_hist[d], windows.x_fut[d])
            y_true = torch.as_tensor(normalise_y(np.asarray(windows.y[d], dtype=np.float32), sc), dtype=torch.float32)
            day_loss = pinball_per_day(y_true.unsqueeze(0), q_norm.unsqueeze(0), QUANTILE_LEVELS_TENSOR).squeeze(0)
            batch_loss = batch_loss + day_loss
            total += float(day_loss)
            n += 1
        if train:
            (batch_loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    seed = args.seed
    out_path = out_path_for(seed)

    baseline_ckpt = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt", weights_only=False, map_location="cpu")
    sc = baseline_ckpt["scaler_stats"]
    model = Baseline_Forecaster(**baseline_ckpt["model_config"])
    model.load_state_dict(baseline_ckpt["state_dict"])  # fresh copy of baseline weights -- same warm-start as every DFL corner

    base_raw = reindex_and_impute(pd.read_csv(DATA_DIR / "df_full.csv", parse_dates=["datetime"]).set_index("datetime"),
                                   HIST_COLS, freq="1h", warn_gap=6)
    frame = build_features(base_raw, feature_cols=FEAT_COLS, price_cols=PRICE_COLS_ALL)
    win_kw = dict(gate_aligned_only=True, issue_hour=ISSUE_HOUR, hist_cols=HIST_COLS, exo_cols=EXO_COLS,
                  target_col="prosumption", price_cols=PRICE_COLS_ALL, n_hist=N_HIST, horizon=HORIZON)
    train_windows = make_windows(frame, y_range=(TRAIN_START, VAL_START - pd.Timedelta(hours=1)), **win_kw)
    val_windows = make_windows(frame, y_range=(VAL_START, TEST_START - pd.Timedelta(hours=1)), **win_kw)
    train_days = np.arange(len(train_windows.delivery_start))
    val_days = np.arange(len(val_windows.delivery_start))
    print(f"n_train={len(train_days)}  n_val={len(val_days)}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val, best_state, patience = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        train_pinball = epoch_pinball(model, sc, train_windows, train_days, train=True, opt=opt)
        val_pinball = epoch_pinball(model, sc, val_windows, val_days, train=False)
        improved = val_pinball < best_val
        print(f"epoch {epoch:2d}  train_pinball={train_pinball:.4f}  val_pinball={val_pinball:.4f}  "
              f"({time.time()-t0:.1f}s)  {'*best' if improved else f'(patience {patience+1}/{PATIENCE})'}", flush=True)
        if improved:
            best_val, best_state, patience = val_pinball, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"early stop at epoch {epoch} (best val_pinball={best_val:.4f})")
                break

    torch.save({"state_dict": best_state, "best_val_pinball": best_val, "seed": seed}, out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
