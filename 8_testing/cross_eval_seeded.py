"""Standalone script: seed-averaged cross-price transfer check (RQ2).

For each of the N=5 seeds already trained (20240801 original + 20240802-20240805),
cross-evaluates:
  Direction A: dual-price-trained forecaster -> SINGLE-price settlement pipeline
  Direction B: single-price-trained forecaster -> DUAL-price settlement pipeline

Mirrors cross_price_eval.ipynb's existing single-seed methodology exactly (same
evaluate_one_pair call, same archetype, same full 366-day sealed test year), just
looped over seeds. Results saved to 8_testing/results/cross_eval/ with the project's
established _seed{N} suffix convention (seed 20240801 keeps the original unsuffixed
name already on disk, not re-run).
"""
import sys, time
sys.path.insert(0, '8_testing')
sys.path.insert(0, '7_model_training')
sys.path.insert(0, '4_forecasting')
sys.path.insert(0, '5_scenario_gen')
sys.path.insert(0, '6_models')
import pickle
import torch

from eval_raw import (evaluate_one_pair, precompute_test_oracle_costs, load_test_windows,
                       fresh_baseline_model, dfl_model_from, FORECASTING_DIR, DFL_TRAIN_DIR,
                       COPULA_DIR, FrozenCopulaSampler, FixedParams, GAMMA, N_SCEN)
from pathlib import Path

CROSS_EVAL_DIR = Path("8_testing/results/cross_eval")
CROSS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
C_ch, C_dis, B_max = 2.0, 2.0, 4.0
SEEDS = [20240801, 20240802, 20240803, 20240804, 20240805]
DEFAULT_SEED = 20240801

baseline_ckpt = torch.load(FORECASTING_DIR / "baseline_forecaster_best.pt", weights_only=False, map_location="cpu")
sc = baseline_ckpt["scaler_stats"]
baseline_model = fresh_baseline_model(baseline_ckpt)

def load(fid):
    ck = torch.load(DFL_TRAIN_DIR / f"{fid}.pt", weights_only=False, map_location="cpu")
    return dfl_model_from(baseline_ckpt, ck)

cop = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
sampler = FrozenCopulaSampler(cop["Z_corr"], cop["quantile_levels"])
quantile_levels = cop["quantile_levels"]
windows = load_test_windows()

fp = FixedParams(T_total=24, num_scenarios=N_SCEN, dt=1.0, eta_ch=0.95, eta_dis=0.95,
                  C_ch=C_ch, C_dis=C_dis, B_max=B_max, SOC0=0.5 * B_max, gamma=GAMMA)
print("precomputing oracle costs for both pipelines...", flush=True)
oracle_single = precompute_test_oracle_costs(windows, "single-price", fp)
oracle_dual = precompute_test_oracle_costs(windows, "dual-price", fp)

def seed_suffix(seed):
    return "" if seed == DEFAULT_SEED else f"_seed{seed}"

for seed in SEEDS:
    ss = seed_suffix(seed)

    # Direction A: dual-price-trained -> SINGLE-price pipeline
    fid_dual = f"dfl_1stage_dual-price{ss}"
    out_a = CROSS_EVAL_DIR / f"dfl_dual-price_ON_single-price_eval__balanced{ss}.parquet"
    if out_a.exists():
        print(f"[seed {seed}] Direction A already exists, skipping: {out_a}")
    else:
        t0 = time.time()
        model = load(fid_dual)
        evaluate_one_pair(model, sc, sampler, windows, baseline_model, quantile_levels,
                           forecaster_id=f"dfl_1stage_dual-price{ss}_ON_single-price",
                           price_mode="single-price", archetype_name="balanced",
                           C_ch=C_ch, C_dis=C_dis, B_max=B_max,
                           oracle_costs=oracle_single, out_path=out_a)
        print(f"[seed {seed}] Direction A done in {time.time()-t0:.1f}s -> {out_a}")

    # Direction B: single-price-trained -> DUAL-price pipeline
    fid_single = f"dfl_1stage_single-price{ss}"
    out_b = CROSS_EVAL_DIR / f"dfl_single-price_ON_dual-price_eval__balanced{ss}.parquet"
    if out_b.exists():
        print(f"[seed {seed}] Direction B already exists, skipping: {out_b}")
    else:
        t0 = time.time()
        model = load(fid_single)
        evaluate_one_pair(model, sc, sampler, windows, baseline_model, quantile_levels,
                           forecaster_id=f"dfl_1stage_single-price{ss}_ON_dual-price",
                           price_mode="dual-price", archetype_name="balanced",
                           C_ch=C_ch, C_dis=C_dis, B_max=B_max,
                           oracle_costs=oracle_dual, out_path=out_b)
        print(f"[seed {seed}] Direction B done in {time.time()-t0:.1f}s -> {out_b}")

print("\nall seeds done")
