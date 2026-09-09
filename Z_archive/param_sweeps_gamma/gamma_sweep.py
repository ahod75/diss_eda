"""gamma_sweep.py -- profiles the 1stage differentiable surrogate corners' (the only
architecture still trained -- point_robust training was dropped, see dispatch_setup.py's
module comment) forward and backward solve across Tikhonov `gamma` values, on a fixed
16-day batch drawn from the training year.

Now that gamma is being justified as a single, uniform, cross-corner choice (applied
identically to every mode rather than tuned per corner -- see default GAMMA in
dfl_train_utils.py, used unmodified by every corner in train_dfl_archetypes.py and every
call site in evaluate.py), this sweep measures both
1stage corners directly rather than extrapolating from one.

The full LP-dual robust counterpart's backward pass is not attempted (established early in
this project as too computationally intense to run at all -- see this branch's first
commit); its own gamma sensitivity is covered separately, FORWARD-ONLY, by
gamma_sweep_full_robust.py. gamma=0 is deliberately NOT included in this sweep's GAMMAS
(unlike the full-robust sweep) -- dropping to 0 removes the only curvature dual-price has
at all, which is exactly the KKT-singularity failure mode gamma exists to prevent; there's
no safe way to differentiate through that boundary case, so it's left to the forward-only
sweep where it can't crash a backward pass.

Reuses dfl_train_utils.dfl_loss_batch directly (not a hand-rolled reimplementation of the
solve pipeline) -- guarantees the sweep tests exactly what training actually does, no
drift risk. Trade-off vs. an even more tightly-coupled version: no retain_graph=True
sharing of the forecast pass across gamma values, so each gamma value re-runs its own
forecast -- negligible cost next to the QP solve.

1stage has no h_plus/h_minus Parameters at all (no robust box), so unlike the version of
this script that also covered point_robust, no box precomputation is needed here.

Day sampling: same seeded 16-day batch as before (TrainConfig.seed, first 16 of
np.random.permutation(n_train) -- the exact shuffling call train_one_config's epoch loop
makes, made reproducible here since the live loop itself does not seed numpy).

Usage: uv run python gamma_sweep.py
"""
from __future__ import annotations
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from pyprojroot import here

ROOT_DIR = here()
MODEL_DIR = ROOT_DIR / "6_models"
PARAM_SWEEPS_DIR = MODEL_DIR / "param_sweeps"
sys.path.insert(0, str(ROOT_DIR / "7_model_training"))

from dfl_train_utils import (
    DATA_DIR, FORECASTING_DIR, COPULA_DIR,
    reindex_and_impute, build_features, make_windows,
    normalise_hist, normalise_exo, denormalise_y,
    Baseline_Forecaster, TRAIN_START, VAL_START, HIST_COLS, FEAT_COLS, EXO_COLS,
    PRICE_COLS_ALL, FrozenCopulaSampler, ISSUE_HOUR, HORIZON, N_HIST, device,
    make_fp, make_bundle, make_layer, dfl_loss_batch, TrainConfig,
)

CORNERS = [
    ("1stage", "single-price"),
    ("1stage", "dual-price"),
]
BATCH_SIZE = 16
GAMMAS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
BASELINE_GAMMA = 1e-4
SEED = TrainConfig.seed  # 20240801 -- matches what train_one_config uses (torch only; numpy unseeded there)

RESULTS_PATH = PARAM_SWEEPS_DIR / "gamma_sweep_results.pkl"


def load_data():
    base = pd.read_csv(DATA_DIR / "df_full.csv", parse_dates=["datetime"]).set_index("datetime")
    base = reindex_and_impute(base, HIST_COLS, freq="1h", warn_gap=6)
    frame = build_features(base, feature_cols=FEAT_COLS, price_cols=PRICE_COLS_ALL)
    win_kw = dict(gate_aligned_only=True, issue_hour=ISSUE_HOUR, hist_cols=HIST_COLS,
                  exo_cols=EXO_COLS, target_col="prosumption", price_cols=PRICE_COLS_ALL,
                  n_hist=N_HIST, horizon=HORIZON)
    train_windows = make_windows(frame, y_range=(TRAIN_START, VAL_START - pd.Timedelta(hours=1)), **win_kw)
    return train_windows


def load_frozen_model():
    ckpt = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt", weights_only=False, map_location="cpu")
    model = Baseline_Forecaster(**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.train()  # matches train_one_config's per-batch mode -- gradients must flow
    sc = ckpt["scaler_stats"]

    baseline_model = Baseline_Forecaster(**ckpt["model_config"])
    baseline_model.load_state_dict(ckpt["state_dict"])
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad_(False)
    return model, baseline_model, sc


def run_one(architecture, mode, gamma, batch_indices, model, sampler, sc, train_windows, fwd):
    fp = make_fp(architecture, gamma)
    bundle = make_bundle(architecture, fp, mode)
    layer = make_layer(bundle)
    keys = [p.name() for p in bundle.params]

    model.zero_grad(set_to_none=True)
    (loss_sum, n_survived, sat_h, n_h, L_base_sum, f_dfl_sum,
     n_inaccurate, t_forward) = dfl_loss_batch(
        batch_indices, model=model, fp=fp, sampler=sampler, sc=sc, windows=train_windows,
        layer=layer, keys=keys, architecture=architecture, mode=mode, device=device,
        fwd=fwd, boxes=None)

    t0 = time.perf_counter()
    (loss_sum / n_survived).backward()
    t_backward = time.perf_counter() - t0
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")))

    return {
        "architecture": architecture, "mode": mode, "gamma": gamma,
        "t_forward": t_forward, "t_backward": t_backward,
        "n_survived": n_survived, "n_inaccurate": n_inaccurate,
        "sat_frac": (sat_h / n_h) if n_h > 0 else 0.0,
        "grad_norm": grad_norm,
        "f_dfl_mean": f_dfl_sum / n_survived, "L_base_mean": L_base_sum / n_survived,
    }


def summarise(results):
    df = pd.DataFrame(results)
    for (architecture, mode), g in df.groupby(["architecture", "mode"], sort=False):
        print(f"\n=== {architecture} {mode} ===")
        base_rows = g[g.gamma == BASELINE_GAMMA]
        base_f = base_rows.iloc[0]["f_dfl_mean"] if len(base_rows) else float("nan")
        print(f"{'gamma':>8}  {'t_fwd(s)':>9}  {'t_bwd(s)':>9}  {'n_inacc':>8}  "
              f"{'grad_norm':>10}  {'sat_frac':>9}  {'f_dfl_mean':>11}  {'d_f_dfl':>9}")
        for _, r in g.sort_values("gamma").iterrows():
            d_f = r["f_dfl_mean"] - base_f
            print(f"{r['gamma']:>8.0e}  {r['t_forward']:>9.3f}  {r['t_backward']:>9.3f}  "
                  f"{r['n_inaccurate']:>8}  {r['grad_norm']:>10.2f}  {r['sat_frac']:>9.3f}  "
                  f"{r['f_dfl_mean']:>11.3f}  {d_f:>9.3f}")


if __name__ == "__main__":
    print(f"device={device}  batch_size={BATCH_SIZE}  corners={CORNERS}  gammas={GAMMAS}")

    train_windows = load_data()
    n_train = len(train_windows.delivery_start)

    np.random.seed(SEED)
    order = np.random.permutation(n_train)
    batch_indices = order[:BATCH_SIZE].tolist()
    print(f"sampled {len(batch_indices)} days (seeded={SEED}, reproducible): {batch_indices}")

    model, baseline_model, sc = load_frozen_model()
    cop = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    sampler = FrozenCopulaSampler(cop["Z_corr"], cop["quantile_levels"]).to(device)
    fwd = {"normalise_hist": normalise_hist, "denormalise_y": denormalise_y, "normalise_exo": normalise_exo}

    results = []
    for architecture, mode in CORNERS:
        print(f"\n--- corner: architecture={architecture} mode={mode} ---", flush=True)
        for gamma in GAMMAS:
            r = run_one(architecture, mode, gamma, batch_indices, model, sampler, sc, train_windows, fwd)
            results.append(r)
            print(f"  gamma={gamma:.0e}  t_fwd={r['t_forward']:.3f}s  t_bwd={r['t_backward']:.3f}s  "
                  f"n_inaccurate={r['n_inaccurate']}  grad_norm={r['grad_norm']:.2f}  "
                  f"sat_frac={r['sat_frac']:.3f}  f_dfl_mean={r['f_dfl_mean']:.3f}", flush=True)

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({"batch_indices": batch_indices, "seed": SEED, "corners": CORNERS,
                     "gammas": GAMMAS, "baseline_gamma": BASELINE_GAMMA, "results": results}, f)
    print(f"\nsaved -> {RESULTS_PATH}")
    summarise(results)
