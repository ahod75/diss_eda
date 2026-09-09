"""eval_raw.py -- raw per-day evaluation data collection (merged from the former
evaluate.py + evaluate_pairs.py; only live code carried forward, everything the old
single-archetype evaluate_model/main() pipeline needed and nothing else used has been
dropped).

Two-function design:

evaluate_one_pair (Function A): one call evaluates ONE (forecaster, price_mode,
archetype) pair over the sealed TEST set (TEST_START..TEST_END) and writes its full
per-day row set to a parquet file under RESULTS_DIR/raw_daily/.

This module is COLLECTION ONLY -- it never reads RAW_DAILY_DIR back. All querying/
aggregation (load_all_pairs, load_single_price, load_dual_price, and anything else that
reads the parquet files this module writes) lives in aggregate_results.ipynb instead.
INVARIANT the query side relies on: every file in RAW_DAILY_DIR must share the same
schema -- if evaluate_one_pair's column set ever changes, wipe and fully regenerate
RAW_DAILY_DIR rather than letting old- and new-schema files sit side by side (pyarrow's
behaviour on a schema mismatch across files in one directory read is not a graceful
merge).

run_full_evaluation_grid (Function B): loops over every valid (price_mode, archetype,
forecaster) combination and calls Function A for each. The price-mode firewall --
forecasters trained under one price structure never evaluated under the other -- is
enforced structurally by FORECASTERS' shape (a dict keyed by price_mode, each value list
containing only checkpoints trained under that mode), not by a runtime check: a
dual-price-trained checkpoint simply never appears under the "single-price" key, so
there's nothing to accidentally get wrong. "baseline" deliberately appears under BOTH
keys -- it's pure CRPS-trained, never saw either price structure, and is needed as the
reference point under both. Archetype crossing is the opposite policy and intentionally
unrestricted: every forecaster (however it was trained) is evaluated against all three
archetypes, matching what the earlier archetype_pilot2.py comparison already did
(specialist-vs-generalist only exists BECAUSE of that crossing).

oracle_costs is precomputed once per (price_mode, archetype) -- the oracle has perfect
foresight, so it never depends on which forecaster is being evaluated -- and reused
across every forecaster_id evaluated under that combination.

Column schema (locked): identity (forecaster_id, price_mode, archetype_name, day, date),
cost (total_cost, regret_v_oracle, C_da, C_imb), volume (throughput_MWh,
imbalance_volume_abs_MWh -- signed imbalance volume deliberately dropped), saturation
(sat_hours_total, sat_hours_soc, sat_hours_rate), statistical (crps, bias_MW), plus the
raw per-timestep (T,) arrays (p_ch_raw, p_dis_raw, p_ch_r, p_dis_r, soc, p_imb, mean,
realised) so any metric not yet anticipated can be derived later without re-solving.

Every checkpoint, regardless of which architecture/archetype trained it, is evaluated
through the SAME setup_full_robust decision problem -- point_robust and 1stage are both
training-time surrogates only, never the real deployment decision-maker (point_robust's
D_ch/D_dis have no EXACT feasibility guarantee at test time, only the weak gamma Tikhonov
penalty plus the 3-point check; 1stage has no recourse mechanism at all).

sat_hours_soc/sat_hours_rate: reconstructed from the returned RealisedBreakdown (soc
trajectory + pre/post-clip charge/discharge) without touching dispatch_wrapper.py's
internals -- see _saturation_breakdown's docstring. Note: charge and discharge clip
independently in this codebase (nothing enforces mutual exclusivity), so an hour can in
principle count toward BOTH sat_hours_soc and sat_hours_rate at once (e.g. charge
rate-limited, discharge SOC-limited, same hour) -- the two sub-counts are informative but
not guaranteed to partition sat_hours_total exactly.

Usage:
    python eval_raw.py                 # full test set, every (price_mode, archetype, forecaster)
    python eval_raw.py --smoke 10      # first 10 test days only (fast wiring check)
"""
from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

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
DFL_TRAIN_DIR = ROOT_DIR / "7_model_training"
RESULTS_DIR = ROOT_DIR / "8_testing" / "results"
RAW_DAILY_DIR = RESULTS_DIR / "raw_daily"

sys.path.insert(0, str(FORECASTING_DIR))
sys.path.insert(0, str(COPULA_DIR))
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(DFL_TRAIN_DIR))

from forecasting import (reindex_and_impute, build_features, make_windows,
                         normalise_hist, normalise_exo, denormalise_y, Baseline_Forecaster,
                         QUANTILE_LEVELS, TEST_START, TEST_END,
                         HIST_COLS, FEAT_COLS, EXO_COLS)
from copula_lib import FrozenCopulaSampler
# oracle: the shared perfect-foresight economic oracle -- mode/architecture-independent,
# same benchmark for every corner.
from oracle import build_oracle, oracle_realised_cost
from dispatch_wrapper import realised_breakdown
from dispatch_setup import FixedParams, setup_full_robust as build_problem
from dispatch_shared import (solve_plain, compute_box, get_prices, oracle_price_values,
                              select_fc_columns, price_model_for_settlement, build_layer_vals)
from train_dfl_forecasts import ARCHETYPES   # (name, C_ch, C_dis, B_max) -- single source of truth

PRICE_COLS = ["da", "imb", "imb_up", "imb_down"]
PRICE_COLS_FC = ["da_fc", "imb_fc", "imb_up_fc", "imb_down_fc"]   # all 4 real, always->=0 columns
PRICE_COLS_ALL = PRICE_COLS + PRICE_COLS_FC
ISSUE_HOUR, HORIZON, N_HIST = 9, 24, 168
device = torch.device("cpu")   # plain Gurobi solve; forecast forward pass is cheap on CPU too

N_SCEN = 64
# 0, not the 1e-4 training uses: gamma's only justification is cvxpylayers' implicit-
# differentiation needing an invertible KKT system at TRAINING time -- it does nothing
# for this module's plain forward Gurobi solves (no differentiation here at all). The
# mode-specific reason it used to matter anyway (dispatchability's D_ch/D_dis split
# becomes solver-arbitrary without it) is moot now that dispatchability isn't evaluated.
# The remaining theoretical risk -- single-price/dual-price's own p_ch_hat/p_dis_hat
# split, which throughput_MWh depends on, being similarly under-pinned at gamma=0 -- was
# checked empirically (gamma_sweep_full_robust.py, full gamma range including 0): both
# modes' total_charge_MWh_sum stay stable (single-price to 4 decimals, dual-price within
# ~0.3%) with no gamma at all, so it doesn't materialise here.
GAMMA = 0
BOX_LEVELS = (0.15, 0.85)   # locked in from h_selection_sweep.ipynb -- same as training
SAT_TOL = 1e-6

# Which forecaster_ids are valid under each price_mode -- the price-mode firewall lives
# in this data structure's shape, not in a runtime check (see module docstring).
# forecaster_id -> checkpoint file is f"{forecaster_id}.pt" in 7_model_training/, except
# "baseline" which loads baseline_forecaster_best.pt instead. Balanced-only now (the
# short_sharp/long_slow specialist checkpoints were retired along with the archetype
# grid -- see train_dfl_forecasts.py's docstring); ARCHETYPES (imported below) shrinking
# to one entry means the archetype loop this feeds is now degenerate too.
# "crps_only_retrained" (see train_crps_only.py) is the missing control: baseline weights
# fine-tuned on pinball loss ALONE, no economic term -- same training budget as every DFL
# corner, isolating how much of DFL's benefit is just "more training" vs the task loss
# specifically. Price-mode-independent (pinball never touches price), so it appears under
# both keys, same as "baseline".
# SEED_VARIANTS: seed-averaging campaign (9_analysis/why_dfl_helps.md Section 7) --
# baseline is deliberately NOT seed-averaged (out of scope, by request); crps_only and
# both DFL modes are, at these 4 extra seeds plus the original 20240801 (already covered
# by the unsuffixed names above) for N=5 total per model type. Naming matches
# train_crps_only.py/train_dfl_forecasts.py's own "_seed{N}" convention exactly.
SEED_VARIANTS = [20240802, 20240803, 20240804, 20240805]
FORECASTERS = {
    "single-price": (["baseline", "crps_only_retrained", "dfl_1stage_single-price"]
                      + [f"crps_only_retrained_seed{s}" for s in SEED_VARIANTS]
                      + [f"dfl_1stage_single-price_seed{s}" for s in SEED_VARIANTS]),
    "dual-price":   (["baseline", "crps_only_retrained", "dfl_1stage_dual-price"]
                      + [f"crps_only_retrained_seed{s}" for s in SEED_VARIANTS]
                      + [f"dfl_1stage_dual-price_seed{s}" for s in SEED_VARIANTS]),
}


# =====================================================================================
# SHARED HELPERS (forecast/data loading, statistical metrics, oracle) -- used by both
# evaluate_one_pair and run_full_evaluation_grid.
# =====================================================================================
def load_test_windows():
    base  = pd.read_csv(DATA_DIR / "df_full.csv", parse_dates=["datetime"]).set_index("datetime")
    base  = reindex_and_impute(base, HIST_COLS, freq="1h", warn_gap=6)
    frame = build_features(base, feature_cols=FEAT_COLS, price_cols=PRICE_COLS_ALL)
    win_kw = dict(gate_aligned_only=True, issue_hour=ISSUE_HOUR, hist_cols=HIST_COLS,
                  exo_cols=EXO_COLS, target_col="prosumption", price_cols=PRICE_COLS_ALL,
                  n_hist=N_HIST, horizon=HORIZON)
    return make_windows(frame, y_range=(TEST_START, TEST_END), **win_kw)


def forecast_quantiles(model, sc, x_hist_day, x_fut_day):
    model.eval()
    with torch.no_grad():
        xh = normalise_hist(np.asarray(x_hist_day), sc)
        xh = torch.as_tensor(xh, dtype=torch.float32, device=device).unsqueeze(0)
        xf = normalise_exo(np.asarray(x_fut_day), sc)
        xf = torch.as_tensor(xf, dtype=torch.float32, device=device).unsqueeze(0)
        q_norm = model(xh, xf)
        q_phys = denormalise_y(q_norm, sc)
    return q_phys.squeeze(0).to(torch.float64)


def crps_per_measurement(y_true, quantiles, levels=QUANTILE_LEVELS):
    """Quantile-based CRPS approximation, PHYSICAL units (MW), averaged PER HOURLY
    MEASUREMENT over the 24-hour delivery window (not summed -- CRPS is a per-measurement
    forecast-quality metric, not a physical quantity that accumulates over a day).

    CRPS(F, y) ~= 2 * integral_0^1 pinball_tau(y, F^-1(tau)) dtau, approximated here as
    2 * mean over the Q quantile LEVELS of the pinball loss at that level, per hour, then
    averaged over the K=24 hours -- the standard textbook quantile-CRPS estimator
    (properly normalised, with the 2x factor). NOT the same convention as this codebase's
    training-time "val_pinball" (an unnormalised sum over K*Q with no 2x factor, used only
    as a relative/internal training signal).

    y_true    : (K,) realised values, physical units.
    quantiles : (K, Q) forecast quantiles, physical units (same units as y_true).
    levels    : (Q,) quantile levels in (0,1).
    Returns   : scalar, physical units, averaged over the K hourly measurements.
    """
    y_true = np.asarray(y_true, float)
    quantiles = np.asarray(quantiles, float)
    levels = np.asarray(levels, float)
    e = y_true[:, None] - quantiles                        # (K, Q)
    q = levels[None, :]                                     # (1, Q)
    pinball = np.maximum(q * e, (q - 1.0) * e)               # (K, Q)
    crps_per_hour = 2.0 * pinball.mean(axis=1)                # (K,)
    return float(crps_per_hour.mean())


def fresh_baseline_model(ckpt):
    m = Baseline_Forecaster(**ckpt["model_config"])
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device)


def dfl_model_from(baseline_ckpt, dfl_ckpt):
    m = Baseline_Forecaster(**baseline_ckpt["model_config"])
    m.load_state_dict(dfl_ckpt["state_dict"])
    return m.to(device)


def precompute_test_oracle_costs(windows, mode, phys_fp, n_days=None):
    price_model = price_model_for_settlement(mode)
    ob = build_oracle(phys_fp, price_model, objective="economic")
    n = n_days if n_days is not None else len(windows.delivery_start)
    costs = np.empty(n)
    for d in range(n):
        realised  = np.asarray(windows.y[d], float)
        price_day = np.asarray(windows.price[d], float)
        fc_price_day = select_fc_columns(price_day, cols=PRICE_COLS_ALL, real_cols=PRICE_COLS_FC)
        fc_vals = oracle_price_values(fc_price_day, price_model, realised, cols=PRICE_COLS)
        true_prices = get_prices(price_day, price_model, cols=PRICE_COLS)
        costs[d] = oracle_realised_cost(
            phys_fp, ob, fc_vals, realised, price_model,
            true_pi_da=true_prices["pi_da"], true_pi_imb=true_prices.get("pi_imb"),
            true_pi_imb_up=true_prices.get("pi_imb_up"), true_pi_imb_down=true_prices.get("pi_imb_down"),
            solver=cp.GUROBI)
    return costs


# =====================================================================================
# FUNCTION A
# =====================================================================================
def _saturation_breakdown(fp, bd, tol=SAT_TOL):
    """Classifies each saturated hour as rate-limited (C_ch/C_dis was the binding term)
    or SOC-limited (energy headroom was), by recomputing the same
    min(C_ch, (B_max-soc)/(eta_ch*dt)) / min(C_dis, soc*eta_dis/dt) caps
    dispatch_wrapper.realised_breakdown's clip loop uses internally, from the SOC
    trajectory it already returns -- no change to that function needed."""
    dtype = bd.soc.dtype
    soc_before = torch.cat([torch.as_tensor([fp.SOC0], dtype=dtype), bd.soc[:-1]])
    C_ch_t = torch.as_tensor(float(fp.C_ch), dtype=dtype)
    C_dis_t = torch.as_tensor(float(fp.C_dis), dtype=dtype)
    headroom_ch = (fp.B_max - soc_before) / (fp.eta_ch * fp.dt)
    headroom_dis = (soc_before * fp.eta_dis) / fp.dt

    ch_saturated = (bd.p_ch_raw - bd.p_ch_r) > tol
    dis_saturated = (bd.p_dis_raw - bd.p_dis_r) > tol
    ch_rate_bound = ch_saturated & (C_ch_t <= headroom_ch + tol)
    ch_soc_bound = ch_saturated & ~ch_rate_bound
    dis_rate_bound = dis_saturated & (C_dis_t <= headroom_dis + tol)
    dis_soc_bound = dis_saturated & ~dis_rate_bound

    sat_total = int((ch_saturated | dis_saturated).sum().item())
    sat_rate = int((ch_rate_bound | dis_rate_bound).sum().item())
    sat_soc = int((ch_soc_bound | dis_soc_bound).sum().item())
    return sat_total, sat_soc, sat_rate


def evaluate_one_pair(model, sc, sampler, windows, baseline_model, quantile_levels, *,
                       forecaster_id: str, price_mode: str, archetype_name: str,
                       C_ch: float, C_dis: float, B_max: float,
                       oracle_costs, out_path: Path, n_days: int | None = None) -> pd.DataFrame:
    """ONE (forecaster, price_mode, archetype) pair, full test set (or first n_days for a
    smoke test). oracle_costs: (n,) array indexed by day -- perfect-foresight, so it never
    depends on `model`; the caller (Function B) computes it once per (price_mode,
    archetype) and reuses it across every forecaster evaluated under that combination."""
    fp = FixedParams(T_total=24, num_scenarios=N_SCEN, dt=1.0, eta_ch=0.95, eta_dis=0.95,
                      C_ch=C_ch, C_dis=C_dis, B_max=B_max, SOC0=0.5 * B_max, gamma=GAMMA)
    bundle = build_problem(fp, price_mode)
    price_model_str = price_model_for_settlement(price_mode)
    n = n_days if n_days is not None else len(windows.delivery_start)

    rows = []
    for d in range(n):
        realised = np.asarray(windows.y[d], float)
        price_day = np.asarray(windows.price[d], float)
        q_phys = forecast_quantiles(model, sc, windows.x_hist[d], windows.x_fut[d])
        mean, xi = sampler.mean_and_errors(q_phys)

        fc_price_day = select_fc_columns(price_day, cols=PRICE_COLS_ALL, real_cols=PRICE_COLS_FC)
        fc_prices = get_prices(fc_price_day, price_model_str, cols=PRICE_COLS)

        base_q = forecast_quantiles(baseline_model, sc, windows.x_hist[d], windows.x_fut[d])
        h_plus, h_minus = compute_box(base_q, sampler, quantile_levels, box_levels=BOX_LEVELS)

        xi_samples = xi.detach().cpu().numpy() if price_mode == "dual-price" else None
        fc_prices_np = {kk: np.asarray(vv, float) for kk, vv in fc_prices.items()}
        vals = build_layer_vals(fc_prices_np, h_plus=h_plus.detach().cpu().numpy(),
                                 h_minus=h_minus.detach().cpu().numpy(), xi_samples=xi_samples)
        mean_np = mean.detach().cpu().numpy()
        vals["pl_hat"] = mean_np

        try:
            dec = solve_plain(bundle, vals, solver=cp.GUROBI)
        except RuntimeError as e:
            print(f"    day {d} solve_plain failed, skipping: {e}")
            continue

        true_prices = get_prices(price_day, price_model_str, cols=PRICE_COLS)
        bd = realised_breakdown(fp, dec["p_ch_hat"], dec["p_dis_hat"], dec["D_ch"], dec["D_dis"],
                                 dec["p_da_bat"], realised=realised, pl_hat=mean_np,
                                 price_model=price_model_str, clip_recourse=True, **true_prices)

        sat_total, sat_soc, sat_rate = _saturation_breakdown(fp, bd)
        total_cost = float(bd.C_da + bd.C_imb)

        rows.append({
            "forecaster_id": forecaster_id, "price_mode": price_mode,
            "archetype_name": archetype_name, "day": d, "date": str(windows.delivery_start[d]),
            "total_cost": total_cost,
            "regret_v_oracle": total_cost - float(oracle_costs[d]),
            "C_da": float(bd.C_da.item()), "C_imb": float(bd.C_imb.item()),
            "throughput_MWh": float(((bd.p_ch_r + bd.p_dis_r) * fp.dt).sum().item()),
            "imbalance_volume_abs_MWh": float((torch.abs(bd.p_imb) * fp.dt).sum().item()),
            "sat_hours_total": sat_total, "sat_hours_soc": sat_soc, "sat_hours_rate": sat_rate,
            "crps": crps_per_measurement(realised, q_phys.detach().cpu().numpy()),
            "bias_MW": float((mean_np - realised).mean()),
            "p_ch_raw": bd.p_ch_raw.detach().cpu().numpy().tolist(),
            "p_dis_raw": bd.p_dis_raw.detach().cpu().numpy().tolist(),
            "p_ch_r": bd.p_ch_r.detach().cpu().numpy().tolist(),
            "p_dis_r": bd.p_dis_r.detach().cpu().numpy().tolist(),
            "soc": bd.soc.detach().cpu().numpy().tolist(),
            "p_imb": bd.p_imb.detach().cpu().numpy().tolist(),
            "mean": mean_np.tolist(),
            "realised": realised.tolist(),
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


# =====================================================================================
# FUNCTION B
# =====================================================================================
def load_forecaster(forecaster_id: str, baseline_ckpt, baseline_model):
    """"baseline" reuses the already-loaded frozen model directly (no reload, no fresh
    copy needed -- it's never trained further here). Anything else loads
    7_model_training/{forecaster_id}.pt and applies its weights on top of the baseline
    architecture via dfl_model_from."""
    if forecaster_id == "baseline":
        return baseline_model
    dfl_ckpt = torch.load(DFL_TRAIN_DIR / f"{forecaster_id}.pt", weights_only=False, map_location="cpu")
    return dfl_model_from(baseline_ckpt, dfl_ckpt)


def run_full_evaluation_grid(n_days: int | None = None):
    """Function B: every (price_mode, archetype, forecaster_id) combination FORECASTERS
    permits, via evaluate_one_pair. n_days=None -> full test set; pass a small int for a
    fast smoke check of the whole grid's wiring."""
    print("loading test windows...")
    windows = load_test_windows()
    n_total = len(windows.delivery_start)
    n = n_days if n_days is not None else n_total
    print(f"n_test={n_total}  (evaluating {n})")

    baseline_ckpt = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt", weights_only=False, map_location="cpu")
    baseline_model = fresh_baseline_model(baseline_ckpt)
    sc = baseline_ckpt["scaler_stats"]

    cop = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    sampler = FrozenCopulaSampler(cop["Z_corr"], cop["quantile_levels"])
    quantile_levels = cop["quantile_levels"]

    oracle_cache: dict[tuple[str, str], np.ndarray] = {}
    results = []
    for price_mode, forecaster_ids in FORECASTERS.items():
        for archetype_name, C_ch, C_dis, B_max in ARCHETYPES:
            fp = FixedParams(T_total=24, num_scenarios=N_SCEN, dt=1.0, eta_ch=0.95, eta_dis=0.95,
                              C_ch=C_ch, C_dis=C_dis, B_max=B_max, SOC0=0.5 * B_max, gamma=GAMMA)
            cache_key = (price_mode, archetype_name)
            if cache_key not in oracle_cache:
                print(f"\nprecomputing oracle costs (price_mode={price_mode} archetype={archetype_name})...", flush=True)
                oracle_cache[cache_key] = precompute_test_oracle_costs(windows, price_mode, fp, n_days=n)
            oracle_costs = oracle_cache[cache_key]

            for forecaster_id in forecaster_ids:
                print(f"  evaluating forecaster={forecaster_id} price_mode={price_mode} "
                      f"archetype={archetype_name} ...", flush=True)
                try:
                    model = load_forecaster(forecaster_id, baseline_ckpt, baseline_model)
                except FileNotFoundError as e:
                    print(f"    SKIPPING -- checkpoint not found: {e}")
                    results.append((forecaster_id, price_mode, archetype_name, "MISSING_CHECKPOINT", 0))
                    continue

                out_path = RAW_DAILY_DIR / f"{forecaster_id}__{price_mode}__{archetype_name}.parquet"
                df = evaluate_one_pair(model, sc, sampler, windows, baseline_model, quantile_levels,
                                        forecaster_id=forecaster_id, price_mode=price_mode,
                                        archetype_name=archetype_name, C_ch=C_ch, C_dis=C_dis, B_max=B_max,
                                        oracle_costs=oracle_costs, out_path=out_path, n_days=n)
                print(f"    saved {out_path} ({len(df)} days)")
                results.append((forecaster_id, price_mode, archetype_name, "OK", len(df)))

    print("\n" + "=" * 70)
    print("PIPELINE_SUMMARY")
    for forecaster_id, price_mode, archetype_name, status, n_rows in results:
        print(f"  {forecaster_id:40s} {price_mode:14s} {archetype_name:14s} {status:20s} n_days={n_rows}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=None,
                        help="evaluate only the first N test days (fast correctness check)")
    args = parser.parse_args()
    run_full_evaluation_grid(n_days=args.smoke)
