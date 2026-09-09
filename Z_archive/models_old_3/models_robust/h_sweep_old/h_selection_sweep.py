"""
h-SELECTION SWEEP  (Step 2)  --  runs over the 2018 TRAINING period only.

For every day of 2018, every box_level, and every corner {single,dual} x {k=0,1}:
  solve the robust layer with the BASELINE forecaster's inputs (B1 box from the same
  baseline), replay the REALISED 2018 prosumption through realised_breakdown, and record
      total_charge_MWh, total_discharge_MWh   (realised, post-saturation control actions)
      sat_hours, sat_MWh                        (saturation-block activation: hrs + volume)
Then aggregate to the year. Output = tidy CSVs to graph and pick the box from (visual elbow).

WHY THESE ARE THE RIGHT SIGNALS
  - control actions grow as the box widens (more recourse reserved) -> cost of a big box.
  - saturation shrinks as the box widens (realised error escapes less) -> cost of a small box.
  The elbow between them is the pick. All four metrics are PRICE-FREE (functions of decisions
  + realised error only), so at k=1 single and dual give identical decisions -> identical
  metrics; the script asserts this per day as a free correctness check.

NO oracle, NO cost, NO regret here -- h-selection does not need them.

RUN COST: ~365 days x 7 box_levels x 4 corners ~= 10.2k small QP solves (Clarabel, T=24),
plus 365 forecaster+sampler passes. Order of a few minutes. Nothing here touches val/test.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import cvxpy as cp
import pickle
import time


import sys
from pathlib import Path
from pyprojroot import here

ROOT_DIR = here()
FORECASTING_DIR = ROOT_DIR / "4_forecasting"
DATA_DIR = ROOT_DIR / "1_data" / "processed"
COPULA_DIR = ROOT_DIR / "5_scenario_gen"
MODEL_DIR = ROOT_DIR / "6_models"
MODELS_ROBUST_DIR = MODEL_DIR / "models_robust"

sys.path.insert(0, str(FORECASTING_DIR))  # make forecasting module importable
sys.path.insert(0, str(COPULA_DIR))  # make copula module importable
sys.path.insert(0, str(MODEL_DIR))  # make model modules importable
sys.path.insert(0, str(MODELS_ROBUST_DIR))  # make dispatch_layer_robust/dispatch_wrapper_robust importable

from dispatch_layer_robust import default_fixed_params, build_problem, solve_plain
from dispatch_wrapper_robust import get_prices, realised_breakdown, cholesky_of_second_moment
from forecasting import (reindex_and_impute, build_features, make_windows,
                             normalise_hist, normalise_exo, denormalise_y, Baseline_Forecaster,
                             HIST_COLS, FEAT_COLS, EXO_COLS)
from copula_lib import FrozenCopulaSampler

# ------------------------------------------------------------------ CONFIG
BOX_LEVELS = [(0.05, 0.95), (0.10, 0.90), (0.15, 0.85), (0.20, 0.80),
              (0.25, 0.75), (0.30, 0.70), (0.35, 0.65)]
CORNERS = [("single", 0.0), ("single", 1.0), ("dual", 0.0), ("dual", 1.0)]
N_SCEN = 64
DT = 1.0
MIN_BOX = 1e-4
SOLVER = cp.GUROBI                       # plain solve; significantly faster than Clarabel
SAT_TOL = 1e-6

# split boundaries (UTC); 2018 delivery blocks fully inside [TRAIN_START, VAL_START - 1h]
TRAIN_START = pd.Timestamp("2018-01-01 00:00:00+00:00")
VAL_START   = pd.Timestamp("2019-01-01 00:00:00+00:00")
ISSUE_HOUR  = 9
HORIZON     = 24
N_HIST      = 168                        # <-- set to your forecaster's lookback

PRICE_COLS = ["da", "imb", "imb_up", "imb_down"]


# ============================================================ PLUG POINT
# Return ONE day's quantiles as a torch (K=24, Q=19) tensor in PHYSICAL MW, from the
# FROZEN baseline (128) forecaster. Wire this to your Phase-1 forward and CONFIRM the
# exact model(...) call + denormalisation match your trained forecaster. Everything
# downstream assumes physical-MW monotone quantiles at the levels the sampler expects.
def make_forecast_fn(model, scaler_stats, device, normalise_hist, denormalise_y):
    model.eval()

    def forecast_fn(x_hist_day, x_fut_day):
        with torch.no_grad():
            xh = normalise_hist(np.asarray(x_hist_day), scaler_stats)      # (n_hist, C_hist)
            xh = torch.as_tensor(xh, dtype=torch.float32, device=device).unsqueeze(0)
            xf = normalise_exo(np.asarray(x_fut_day), scaler_stats)
            xf = torch.as_tensor(xf, dtype=torch.float32, device=device).unsqueeze(0)
            q_norm = model(xh, xf)                                          # (1, K, Q) normalised
            q_phys = denormalise_y(q_norm, scaler_stats)                    # affine -> physical MW
        return q_phys.squeeze(0).to(torch.float64)                         # (K, Q)
    return forecast_fn


# ============================================================ box (B1, from baseline)
def boxes_from_quantiles(quantiles, mean, quantile_levels, box_levels=BOX_LEVELS, min_box=MIN_BOX):
    """All box half-widths for one day, from the baseline quantiles + baseline mean anchor.
    Returns {box_level: (h_plus, h_minus)} as numpy. Matches dispatch_wrapper.compute_box
    but reuses the already-computed mean (one sampler pass per day)."""
    q = quantiles.detach().cpu().numpy() if torch.is_tensor(quantiles) else np.asarray(quantiles)
    m = mean.detach().cpu().numpy() if torch.is_tensor(mean) else np.asarray(mean)
    levels = np.asarray(quantile_levels, float)
    out = {}
    for (lo, hi) in box_levels:
        i_lo = int(np.argmin(np.abs(levels - lo)))
        i_hi = int(np.argmin(np.abs(levels - hi)))
        h_plus = np.clip(q[:, i_hi] - m, min_box, None)
        h_minus = np.clip(m - q[:, i_lo], min_box, None)
        out[(lo, hi)] = (h_plus, h_minus)
    return out


# ============================================================ per-day metric reduction
def _day_metrics(fp, bd):
    tn = lambda t: t.detach().cpu().numpy()
    pch_r, pdis_r = tn(bd.p_ch_r), tn(bd.p_dis_r)
    clip_ch = np.abs(tn(bd.p_ch_raw) - pch_r)
    clip_dis = np.abs(tn(bd.p_dis_raw) - pdis_r)
    return {
        "total_charge_MWh":    float((pch_r * fp.dt).sum()),
        "total_discharge_MWh": float((pdis_r * fp.dt).sum()),
        "sat_hours":           int(((clip_ch > SAT_TOL) | (clip_dis > SAT_TOL)).sum()),
        "sat_MWh":             float(((clip_ch + clip_dis) * fp.dt).sum()),
    }


# ============================================================ the sweep
def run_sweep(windows, forecast_fn, sampler, quantile_levels):
    """windows: WindowSet with x_hist, x_fut, y, price, delivery_start for 2018.
    Returns (per_day_df, yearly_df)."""
    fps = {k: default_fixed_params(k, num_scenarios=N_SCEN) for k in (0.0, 1.0)}
    bundles = {(pm, k): build_problem(fps[k], pm) for pm, k in CORNERS}

    n_days = len(windows.delivery_start)
    rows = []
    print("Beginning parameter")
    for d in range(n_days):
        day = windows.delivery_start[d]
        print(f"Starting day:{d + 1} of {n_days}")

        realised = np.asarray(windows.y[d], float)                         # (T,) realised prosumption
        price_day = np.asarray(windows.price[d], float)                    # (T, 4)

        # ONE forecaster + sampler pass per day (baseline is frozen; box-level/corner-independent)
        quantiles = forecast_fn(windows.x_hist[d], windows.x_fut[d])       # (K, Q) torch physical
        mean, xi = sampler.mean_and_errors(quantiles)                      # mean (T,), xi (N,T)
        mean_np = mean.detach().cpu().numpy()
        xi_np = xi.detach().cpu().numpy()
        Sigma = cholesky_of_second_moment(xi_np)                           # (T,T), for k=1
        boxes = boxes_from_quantiles(quantiles, mean, quantile_levels)
        for pm, k in CORNERS:
            print (f"Sweep for price: {pm}, k: {k}")
            fp = fps[k]
            bundle = bundles[(pm, k)]
            prices = get_prices(price_day, pm)                             # {pi_da, pi_imb} | {pi_da,lam_up,lam_dn}
            base_vals = {"pl_hat": mean_np, "Sigma_xi_chol": Sigma,
                         "xi_samples": xi_np, **prices}                    # superset; solve_plain selects
            for (lo, hi), (h_plus, h_minus) in boxes.items():
                vals = {**base_vals, "h_plus": h_plus, "h_minus": h_minus}
                out = solve_plain(bundle, vals, solver=SOLVER)
                bd = realised_breakdown(
                    fp, out["p_ch_hat"], out["p_dis_hat"], out["D_ch"], out["D_dis"],
                    out["p_da_rel"], realised=realised, pl_hat=mean_np, price_model=pm,
                    clip_recourse=True, **prices,
                )
                m = _day_metrics(fp, bd)
                rows.append({"price_model": pm, "k": int(k), "box_lo": lo, "box_hi": hi,
                             "delivery_start": day, **m})

    per_day = pd.DataFrame(rows)
    _assert_k1_price_free(per_day)
    yearly = _aggregate(per_day)
    return per_day, yearly


def _aggregate(per_day: pd.DataFrame) -> pd.DataFrame:
    metrics = ["total_charge_MWh", "total_discharge_MWh", "sat_hours", "sat_MWh"]
    g = per_day.groupby(["price_model", "k", "box_lo", "box_hi"])
    agg = g[metrics].agg(["sum", "mean"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg["n_days"] = g.size().values
    return agg.reset_index().sort_values(["price_model", "k", "box_lo"]).reset_index(drop=True)


def _assert_k1_price_free(per_day: pd.DataFrame, tol=1e-6):
    """At k=1, single and dual decisions are identical -> all four (price-free) metrics
    must coincide per day/box. Free correctness check embedded in the sweep."""
    metrics = ["total_charge_MWh", "total_discharge_MWh", "sat_hours", "sat_MWh"]
    k1 = per_day[per_day["k"] == 1]
    s = k1[k1.price_model == "single"].set_index(["delivery_start", "box_lo"])[metrics].sort_index()
    dd = k1[k1.price_model == "dual"].set_index(["delivery_start", "box_lo"])[metrics].sort_index()
    if len(s) and len(dd):
        gap = float(np.abs(s.values - dd.values).max())
        assert gap < tol, (f"k=1 single vs dual metrics differ by {gap:.2e} -- decisions should be "
                           f"identical at k=1; investigate the layer before trusting the sweep.")
        print(f"[check] k=1 single==dual price-free metrics agree (max gap {gap:.1e})")


# ============================================================ main (wire your artifacts)
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

    # --- load FROZEN baseline forecaster + sampler (your loaders) ---
    
        # ---- reproducibility (record SEED in the checkpoint) ----
    SEED = 20240801
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", DEVICE, "| seed:", SEED)


    # Load forecaster
    forecaster_checkpoint = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt", weights_only = False, map_location="cpu")
    model_config = forecaster_checkpoint["model_config"]
    model = Baseline_Forecaster(**model_config) 
    model.load_state_dict(forecaster_checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()
    sc = forecaster_checkpoint["scaler_stats"]
    QUANTILE_LEVELS = forecaster_checkpoint["quantile_levels"]


    # load sampler
    bundle = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    Z_corr = bundle["Z_corr"]
    levels = bundle["quantile_levels"]
    sampler = FrozenCopulaSampler(Z_corr, levels).to(DEVICE)
    assert getattr(sampler, "S", N_SCEN) == N_SCEN, "sampler scenario count must equal N_SCEN"

    forecast_fn = make_forecast_fn(model, sc, DEVICE, normalise_hist, denormalise_y)

    per_day, yearly = run_sweep(windows, forecast_fn, sampler, QUANTILE_LEVELS)
    per_day_path = MODELS_ROBUST_DIR / "h_sweep_per_day_2018.csv"
    yearly_path = MODELS_ROBUST_DIR / "h_sweep_yearly_2018.csv"
    per_day.to_csv(per_day_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    print(yearly.to_string(index=False))
    print(f"\nsaved: {per_day_path} (per-day), {yearly_path} (yearly aggregate)")
