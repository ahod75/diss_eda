"""gamma_sweep_full_robust.py -- forward-only gamma sensitivity sweep for the TRUE full
LP-dual robust counterpart (dispatch_setup.setup_full_robust), across the full 2018
training year, at the box level currently locked in (h_selection_sweep's BOX_LEVELS
default of (0.15, 0.85)).

Companion to gamma_sweep.py (which profiles the differentiable point_robust/1stage
surrogates' forward+backward solve): full_robust cannot be differentiated through at
reasonable cost (established at the very start of this project -- see this branch's
first commit), so its own gamma sensitivity can only be measured forward-only, the same
way evaluate.py and h_selection_sweep.py already use it -- solve_plain, no cvxpylayers,
no backward pass, no training relevance.

What this answers: how much does gamma move the TRUE decision problem's outputs
(D_ch/D_dis magnitude, economic cost, saturation) -- not a training-numerics question
(full_robust is never trained through directly), but "is the uniform gamma value
distorting the ground-truth evaluation solve". This formalises and extends, across gamma
values and all three modes, the ad hoc gamma=0-vs-1e-4 smoke test done earlier via
evaluate.py for single-price/dispatchability.

gamma=0 IS included here (unlike gamma_sweep.py's differentiable sweep, which starts at
1e-4) -- there's no backward pass to destabilise, so the boundary case is safe to include.

Runs on the TRAINING year (2018), not test -- this is hyperparameter selection, and
touching the sealed test set here would be the same mistake evaluate.py's docstring
already flags for the h-sweeps. Reuses h_selection_sweep.py's helpers (make_forecast_fn,
boxes_from_quantiles, _day_metrics) directly rather than reimplementing them, matching
h_selection_sweep_point.py's existing convention.

Modes: "single-price", "dual-price", "dispatchability" -- not the old (price_model, k)
4-corner notation. Dispatchability is price-free and settlement-convention-independent
(see h_selection_sweep.py's _assert_k1_price_free), so it is tested once, not duplicated
under both price labels.

RUN COST: ~365 days x len(GAMMAS)=6 x 3 modes = ~6.6k solves at ONE fixed box level --
similar order of magnitude to h_selection_sweep.py's own ~10.2k-solve, few-minutes run.

Usage: uv run python gamma_sweep_full_robust.py
"""
from __future__ import annotations
import sys
import pickle

import numpy as np
import pandas as pd
import torch
import cvxpy as cp
from pyprojroot import here

ROOT_DIR = here()
FORECASTING_DIR = ROOT_DIR / "4_forecasting"
DATA_DIR = ROOT_DIR / "1_data" / "processed"
COPULA_DIR = ROOT_DIR / "5_scenario_gen"
MODEL_DIR = ROOT_DIR / "6_models"
PARAM_SWEEPS_DIR = MODEL_DIR / "param_sweeps"

sys.path.insert(0, str(FORECASTING_DIR))
sys.path.insert(0, str(COPULA_DIR))
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(PARAM_SWEEPS_DIR))

from dispatch_setup import default_fixed_params, setup_full_robust as build_problem
from dispatch_shared import solve_plain, get_prices, cholesky_of_second_moment
from dispatch_wrapper import realised_breakdown
from forecasting import (reindex_and_impute, build_features, make_windows,
                          normalise_hist, normalise_exo, denormalise_y, Baseline_Forecaster,
                          HIST_COLS, FEAT_COLS, EXO_COLS)
from copula_lib import FrozenCopulaSampler

from h_selection_sweep import (
    N_SCEN, TRAIN_START, VAL_START, ISSUE_HOUR, HORIZON, N_HIST, PRICE_COLS,
    make_forecast_fn, boxes_from_quantiles, _day_metrics,
)

SOLVER = cp.GUROBI
BOX_LEVEL = (0.15, 0.85)   # locked-in default -- see evaluate.py's own BOX_LEVELS constant
MODES = ["single-price", "dual-price", "dispatchability"]
GAMMAS = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

PER_DAY_PATH = PARAM_SWEEPS_DIR / "gamma_sweep_full_robust_per_day.csv"
YEARLY_PATH = PARAM_SWEEPS_DIR / "gamma_sweep_full_robust_yearly.csv"


def run_sweep(windows, forecast_fn, sampler, quantile_levels):
    fps = {gamma: default_fixed_params(num_scenarios=N_SCEN, gamma=gamma) for gamma in GAMMAS}
    bundles = {(mode, gamma): build_problem(fps[gamma], mode) for mode in MODES for gamma in GAMMAS}

    n_days = len(windows.delivery_start)
    rows = []
    for d in range(n_days):
        day = windows.delivery_start[d]
        if (d + 1) % 50 == 0 or d == 0:
            print(f"day {d + 1} of {n_days}", flush=True)

        realised = np.asarray(windows.y[d], float)
        price_day = np.asarray(windows.price[d], float)

        quantiles = forecast_fn(windows.x_hist[d], windows.x_fut[d])
        mean, xi = sampler.mean_and_errors(quantiles)
        mean_np = mean.detach().cpu().numpy()
        xi_np = xi.detach().cpu().numpy()
        Sigma_np = cholesky_of_second_moment(xi).detach().cpu().numpy()
        boxes = boxes_from_quantiles(quantiles, mean, quantile_levels, box_levels=[BOX_LEVEL])
        h_plus, h_minus = boxes[BOX_LEVEL]

        for mode in MODES:
            price_model = "dual" if mode == "dual-price" else "single"
            prices = get_prices(price_day, price_model, cols=PRICE_COLS)
            for gamma in GAMMAS:
                fp = fps[gamma]
                bundle = bundles[(mode, gamma)]
                vals = {"pl_hat": mean_np, "Sigma_xi_chol": Sigma_np, "xi_samples": xi_np,
                        "h_plus": h_plus, "h_minus": h_minus, **prices}
                out = solve_plain(bundle, vals, solver=SOLVER)
                bd = realised_breakdown(
                    fp, out["p_ch_hat"], out["p_dis_hat"], out["D_ch"], out["D_dis"],
                    out["p_da_rel"], realised=realised, pl_hat=mean_np, price_model=price_model,
                    clip_recourse=True, **prices,
                )
                m = _day_metrics(fp, bd)
                raw_cost = float(bd.C_da + bd.C_imb) if mode != "dispatchability" else float("nan")
                rows.append({
                    "mode": mode, "gamma": gamma, "delivery_start": day,
                    "raw_cost": raw_cost,
                    "D_ch_norm": float(np.linalg.norm(out["D_ch"])),
                    "D_dis_norm": float(np.linalg.norm(out["D_dis"])),
                    **m,
                })

    per_day = pd.DataFrame(rows)
    yearly = _aggregate(per_day)
    return per_day, yearly


def _aggregate(per_day: pd.DataFrame) -> pd.DataFrame:
    metrics = ["total_charge_MWh", "total_discharge_MWh", "sat_hours", "sat_MWh",
               "D_ch_norm", "D_dis_norm"]
    g = per_day.groupby(["mode", "gamma"])
    agg = g[metrics].agg(["sum", "mean"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg["mean_raw_cost"] = g["raw_cost"].mean()
    agg["n_days"] = g.size().values
    return agg.reset_index().sort_values(["mode", "gamma"]).reset_index(drop=True)


if __name__ == "__main__":
    base = pd.read_csv(DATA_DIR / "df_full.csv", parse_dates=["datetime"]).set_index("datetime")
    base = reindex_and_impute(base, HIST_COLS, freq="1h", warn_gap=6)
    frame = build_features(base, feature_cols=FEAT_COLS, price_cols=PRICE_COLS)

    windows = make_windows(
        frame, y_range=(TRAIN_START, VAL_START - pd.Timedelta(hours=1)),
        gate_aligned_only=True, issue_hour=ISSUE_HOUR,
        hist_cols=HIST_COLS, exo_cols=EXO_COLS, target_col="prosumption",
        price_cols=PRICE_COLS, n_hist=N_HIST, horizon=HORIZON,
    )

    SEED = 20240801
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", DEVICE, "| seed:", SEED, "| modes:", MODES, "| gammas:", GAMMAS)

    forecaster_checkpoint = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt",
                                        weights_only=False, map_location="cpu")
    model_config = forecaster_checkpoint["model_config"]
    model = Baseline_Forecaster(**model_config)
    model.load_state_dict(forecaster_checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()
    sc = forecaster_checkpoint["scaler_stats"]
    QUANTILE_LEVELS = forecaster_checkpoint["quantile_levels"]

    cop_bundle = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    sampler = FrozenCopulaSampler(cop_bundle["Z_corr"], cop_bundle["quantile_levels"]).to(DEVICE)
    assert getattr(sampler, "S", N_SCEN) == N_SCEN, "sampler scenario count must equal N_SCEN"

    forecast_fn = make_forecast_fn(model, sc, DEVICE, normalise_hist, denormalise_y)

    per_day, yearly = run_sweep(windows, forecast_fn, sampler, QUANTILE_LEVELS)
    per_day.to_csv(PER_DAY_PATH, index=False)
    yearly.to_csv(YEARLY_PATH, index=False)
    print(yearly.to_string(index=False))
    print(f"\nsaved: {PER_DAY_PATH} (per-day), {YEARLY_PATH} (yearly)")
