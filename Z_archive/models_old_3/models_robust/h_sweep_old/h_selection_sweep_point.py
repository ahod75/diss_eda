"""
h-SELECTION SWEEP  --  POINT-CONSTRAINED variant.

Identical methodology to h_selection_sweep.py (see that file's docstring for the full
rationale), except it solves dispatch_layer_point_robust.py instead of
dispatch_layer_robust.py -- i.e., it measures what h_plus/h_minus actually do to the
model that will actually be TRAINED against, rather than the full LP-dual robust
counterpart used only for final testing. Run both and compare: if the elbows land in
the same place, one box suffices for both training and testing; if they diverge, the
surrogate's weaker (2-point, not exact-worst-case) feasibility guarantee needs its own
calibration.

p_da_rel is reconstructed manually (p_ch_hat - p_dis_hat) since it's a plain expression
in dispatch_layer_point_robust.py, not one of the layer's output variables.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import cvxpy as cp
import pickle

import sys
from pyprojroot import here

ROOT_DIR = here()
FORECASTING_DIR = ROOT_DIR / "4_forecasting"
DATA_DIR = ROOT_DIR / "1_data" / "processed"
COPULA_DIR = ROOT_DIR / "5_scenario_gen"
MODEL_DIR = ROOT_DIR / "6_models"
MODELS_ROBUST_DIR = MODEL_DIR / "models_robust"

sys.path.insert(0, str(FORECASTING_DIR))
sys.path.insert(0, str(COPULA_DIR))
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODELS_ROBUST_DIR))

from dispatch_layer_point_robust import default_fixed_params, build_problem, solve_plain
from dispatch_wrapper_robust import get_prices, realised_breakdown, cholesky_of_second_moment
from forecasting import (reindex_and_impute, build_features, make_windows,
                             normalise_hist, normalise_exo, denormalise_y, Baseline_Forecaster,
                             HIST_COLS, FEAT_COLS, EXO_COLS)
from copula_lib import FrozenCopulaSampler

from h_selection_sweep import (BOX_LEVELS, CORNERS, N_SCEN, MIN_BOX, SAT_TOL,
                                TRAIN_START, VAL_START, ISSUE_HOUR, HORIZON, N_HIST,
                                PRICE_COLS, make_forecast_fn, boxes_from_quantiles,
                                _day_metrics, _aggregate, _assert_k1_price_free)

SOLVER = cp.GUROBI


def run_sweep(windows, forecast_fn, sampler, quantile_levels):
    fps = {k: default_fixed_params(k, num_scenarios=N_SCEN) for k in (0.0, 1.0)}
    bundles = {(pm, k): build_problem(fps[k], pm) for pm, k in CORNERS}

    n_days = len(windows.delivery_start)
    rows = []
    for d in range(n_days):
        day = windows.delivery_start[d]
        if (d + 1) % 50 == 0 or d == 0:
            print(f"Starting day:{d + 1} of {n_days}", flush=True)

        realised = np.asarray(windows.y[d], float)
        price_day = np.asarray(windows.price[d], float)

        quantiles = forecast_fn(windows.x_hist[d], windows.x_fut[d])
        mean, xi = sampler.mean_and_errors(quantiles)
        mean_np = mean.detach().cpu().numpy()
        xi_np = xi.detach().cpu().numpy()
        Sigma = cholesky_of_second_moment(xi_np)
        boxes = boxes_from_quantiles(quantiles, mean, quantile_levels)
        for pm, k in CORNERS:
            fp = fps[k]
            bundle = bundles[(pm, k)]
            prices = get_prices(price_day, pm)
            base_vals = {"Sigma_xi_chol": Sigma, "xi_samples": xi_np, **prices}
            for (lo, hi), (h_plus, h_minus) in boxes.items():
                vals = {**base_vals, "h_plus": h_plus, "h_minus": h_minus}
                out = solve_plain(bundle, vals, solver=SOLVER)
                p_da_rel = out["p_ch_hat"] - out["p_dis_hat"]   # reconstructed, not a layer output
                bd = realised_breakdown(
                    fp, out["p_ch_hat"], out["p_dis_hat"], out["D_ch"], out["D_dis"],
                    p_da_rel, realised=realised, pl_hat=mean_np, price_model=pm,
                    clip_recourse=True, **prices,
                )
                m = _day_metrics(fp, bd)
                rows.append({"price_model": pm, "k": int(k), "box_lo": lo, "box_hi": hi,
                             "delivery_start": day, **m})

    per_day = pd.DataFrame(rows)
    _assert_k1_price_free(per_day)
    yearly = _aggregate(per_day)
    return per_day, yearly


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
    print("device:", DEVICE, "| seed:", SEED)

    forecaster_checkpoint = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt", weights_only=False, map_location="cpu")
    model_config = forecaster_checkpoint["model_config"]
    model = Baseline_Forecaster(**model_config)
    model.load_state_dict(forecaster_checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()
    sc = forecaster_checkpoint["scaler_stats"]
    QUANTILE_LEVELS = forecaster_checkpoint["quantile_levels"]

    bundle = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    Z_corr = bundle["Z_corr"]
    levels = bundle["quantile_levels"]
    sampler = FrozenCopulaSampler(Z_corr, levels).to(DEVICE)
    assert getattr(sampler, "S", N_SCEN) == N_SCEN, "sampler scenario count must equal N_SCEN"

    forecast_fn = make_forecast_fn(model, sc, DEVICE, normalise_hist, denormalise_y)

    per_day, yearly = run_sweep(windows, forecast_fn, sampler, QUANTILE_LEVELS)
    per_day_path = MODELS_ROBUST_DIR / "h_sweep_per_day_2018_point.csv"
    yearly_path = MODELS_ROBUST_DIR / "h_sweep_yearly_2018_point.csv"
    per_day.to_csv(per_day_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    print(yearly.to_string(index=False))
    print(f"\nsaved: {per_day_path} (per-day), {yearly_path} (yearly aggregate)")
