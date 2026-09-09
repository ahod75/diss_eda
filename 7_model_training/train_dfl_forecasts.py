"""train_dfl_forecasts.py -- trains 1stage single-price/dual-price forecasters for the
balanced battery archetype (C_ch=C_dis=2.0MW, B_max=4.0MWh, duration=2h).

This is now the ONLY DFL training entry point -- train_dfl_models.py (which trained
point_robust's single-price/dispatchability/dual-price corners plus 1stage's "balanced"
pair) has been retired entirely. Checkpoints keep the UNSUFFIXED dfl_1stage_{mode}.pt
naming train_dfl_models.py used -- every evaluation/analysis script looks up checkpoints
by that exact unsuffixed name.

Only 1stage architecture is covered, and only the balanced archetype -- the earlier
short_sharp/long_slow archetype grid (and the pilot that motivated retraining per
archetype) was dropped when the project narrowed to a balanced-only, single-vs-dual-price
comparison; see git history for that pilot's findings if the archetype-sensitivity
rationale is ever needed again. point_robust/dispatchability are out of scope for this
grid, consistent with the broader pivot toward 1stage-only training (point_robust
training was removed entirely, not merged in here -- see dispatch_setup.py's module
comment).

Loads data once, per-corner try/except, gc.collect after every corner, forward/backward
timing -- no box precomputation at all, since 1stage has no h_plus/h_minus Parameters to
feed regardless of archetype.

Usage: python train_dfl_forecasts.py [--seed SEED]
Seed 20240801 (TrainConfig's default) writes the unsuffixed dfl_1stage_{mode}.pt names
every other consumer (eval_raw.FORECASTERS, aggregate_results.ipynb,
forecast_characteristics.ipynb) already expects. Any other seed writes
dfl_1stage_{mode}_seed{SEED}.pt alongside them -- used for seed-averaging, see
9_analysis/why_dfl_helps.md Section 7.
"""
from __future__ import annotations
import argparse
import gc
import time
import pickle

import pandas as pd
import torch

from dfl_train_utils import (
    FORECASTING_DIR, DATA_DIR, COPULA_DIR,
    reindex_and_impute, build_features, make_windows,
    normalise_hist, normalise_exo, denormalise_y,
    Baseline_Forecaster, TRAIN_START, VAL_START, TEST_START,
    HIST_COLS, FEAT_COLS, EXO_COLS, PRICE_COLS_ALL,
    FrozenCopulaSampler, ISSUE_HOUR, HORIZON, N_HIST, device, GAMMA,
    TrainConfig, train_one_config, DFL_TRAIN_DIR,
)
from dispatch_setup import FixedParams1Stage

LOG_DIR = DFL_TRAIN_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODES = ["single-price", "dual-price"]

# (name, C_ch, C_dis, B_max) -- SOC0 derived as 0.5*B_max. Balanced-only now (short_sharp/
# long_slow dropped -- see module docstring); kept as a single-element list rather than
# hardcoded scalars so eval_raw.py's `from train_dfl_forecasts import ARCHETYPES` import
# (its "single source of truth" for archetype physical params) keeps working unchanged.
#               columns: charge rate (MW), discharge rate (MW), capacity (MWh)
ARCHETYPES = [
    ("balanced", 2.0, 2.0, 4.0),
]


def make_fp(C_ch, C_dis, B_max, gamma=GAMMA):
    return FixedParams1Stage(T_total=24, num_scenarios=64, dt=1.0, eta_ch=0.95, eta_dis=0.95,
                              C_ch=C_ch, C_dis=C_dis, B_max=B_max, SOC0=0.5 * B_max, gamma=gamma)


def load_data():
    base  = pd.read_csv(DATA_DIR / "df_full.csv", parse_dates=["datetime"]).set_index("datetime")
    base  = reindex_and_impute(base, HIST_COLS, freq="1h", warn_gap=6)
    frame = build_features(base, feature_cols=FEAT_COLS, price_cols=PRICE_COLS_ALL)
    win_kw = dict(gate_aligned_only=True, issue_hour=ISSUE_HOUR, hist_cols=HIST_COLS,
                  exo_cols=EXO_COLS, target_col="prosumption", price_cols=PRICE_COLS_ALL,
                  n_hist=N_HIST, horizon=HORIZON)
    train_windows = make_windows(frame, y_range=(TRAIN_START, VAL_START - pd.Timedelta(hours=1)), **win_kw)
    val_windows   = make_windows(frame, y_range=(VAL_START, TEST_START - pd.Timedelta(hours=1)), **win_kw)
    return train_windows, val_windows


DEFAULT_SEED = 20240801  # matches TrainConfig's own default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    seed = args.seed

    print(f"device={device}  archetypes={ARCHETYPES}  modes={MODES}  seed={seed}")

    print("loading training data (once, shared across every corner)...")
    train_windows, val_windows = load_data()
    print(f"n_train={len(train_windows.delivery_start)}  n_val={len(val_windows.delivery_start)}")

    ckpt = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt", weights_only=False, map_location="cpu")
    sc = ckpt["scaler_stats"]

    cop = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    sampler = FrozenCopulaSampler(cop["Z_corr"], cop["quantile_levels"]).to(device)
    fwd = {"normalise_hist": normalise_hist, "denormalise_y": denormalise_y, "normalise_exo": normalise_exo}

    results = []
    for arch_name, C_ch, C_dis, B_max in ARCHETYPES:
        for mode in MODES:
            t0 = time.time()
            print(f"\nCORNER_START architecture=1stage mode={mode} archetype={arch_name} "
                  f"C_ch={C_ch} C_dis={C_dis} B_max={B_max}", flush=True)

            model = fp = cfg = trained_model = None
            try:
                model = Baseline_Forecaster(**ckpt["model_config"])
                model.load_state_dict(ckpt["state_dict"])   # FRESH copy of the baseline weights every corner
                model = model.to(device)

                fp = make_fp(C_ch, C_dis, B_max)
                cfg = TrainConfig(architecture="1stage", mode=mode, seed=seed)
                print(f"batch_size={cfg.batch_size}")
                # unsuffixed name every consumer expects when arch is balanced (the only
                # one left) and seed is the project default; ARCHETYPES being
                # balanced-only is kept as a conditional only so re-adding an archetype
                # later doesn't silently clobber the balanced checkpoint. Non-default
                # seeds get a _seed{N} suffix -- see module docstring, seed-averaging.
                arch_suffix = "" if arch_name == "balanced" else f"_{arch_name}"
                seed_suffix = "" if seed == DEFAULT_SEED else f"_seed{seed}"
                out_path = DFL_TRAIN_DIR / f"dfl_1stage_{mode}{arch_suffix}{seed_suffix}.pt"

                trained_model, best_val, hist = train_one_config(
                    cfg, model=model, fp=fp, sampler=sampler, sc=sc,
                    train_windows=train_windows, val_windows=val_windows,
                    device=device, fwd=fwd, out_path=out_path)
                elapsed = time.time() - t0
                n_epochs = len(hist)
                t_fwd_train = sum(h["t_forward_train"] for h in hist)
                t_bwd_train = sum(h["t_backward_train"] for h in hist)
                t_fwd_val = sum(h["t_forward_val"] for h in hist)
                print(f"\nCORNER_DONE archetype={arch_name} mode={mode} best_val_fsurr={best_val:.4f} "
                      f"epochs_run={n_epochs} elapsed_s={elapsed:.0f} "
                      f"t_fwd_train_s={t_fwd_train:.1f} (avg {t_fwd_train/n_epochs:.2f}/epoch)  "
                      f"t_bwd_train_s={t_bwd_train:.1f} (avg {t_bwd_train/n_epochs:.2f}/epoch)  "
                      f"t_fwd_val_s={t_fwd_val:.1f} (avg {t_fwd_val/n_epochs:.2f}/epoch)",
                      flush=True)
                torch.save({"state_dict": trained_model.state_dict(), "cfg": cfg.__dict__,
                            "best_val_fsurr": best_val, "val_history": hist,
                            "archetype": {"name": arch_name, "C_ch": C_ch, "C_dis": C_dis, "B_max": B_max}},
                           out_path)
                print(f"CORNER_SAVED {out_path}")
                results.append((arch_name, mode, "OK", elapsed))
            except Exception as e:
                elapsed = time.time() - t0
                print(f"CORNER_FAILED archetype={arch_name} mode={mode} error={e!r} "
                      f"elapsed_s={elapsed:.0f} -- moving on to the next corner", flush=True)
                results.append((arch_name, mode, f"FAILED({e!r})", elapsed))

            del model, trained_model, fp, cfg
            gc.collect()

    print("\n" + "=" * 70)
    print("PIPELINE_SUMMARY")
    for arch_name, mode, status, elapsed in results:
        print(f"  {arch_name:14s} {mode:16s} {status:30s} {elapsed:8.0f}s")


if __name__ == "__main__":
    main()
    
