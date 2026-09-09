"""gamma_sweep.py -- profiles the point-robust dual/k=0 dispatch layer's forward and
backward solve across different Tikhonov `gamma` values, on a single fixed 16-day batch
drawn from the training year.

Rationale: dual/k=0 is the one corner carrying the full N=64-scenario dual-price
epigraph (see dispatch_layer_point_robust.py's docstring), so it's the worst-conditioned
of the six corners -- a gamma found safe here should be at least as safe for the other
five. Only this corner is swept; whatever gamma is chosen gets applied uniformly to
default_fixed_params for every corner.

Read-only w.r.t. the live training run: loads the FROZEN baseline_forecaster_best.pt
checkpoint (never touches dfl_point_robust_dual_k0.pt, which the live run is actively
writing to), and computes its own oracle costs / boxes for just the 16 sampled days
instead of reusing any live-run state.

Reuses train_corner_point_robust.py's own functions directly (imported through its
module namespace) rather than reimplementing dfl_loss_batch's logic -- guarantees the
sweep tests exactly what training actually does, no drift risk between the two.

Day sampling: np.random.seed(TrainConfig.seed) then np.random.permutation(n_train),
first 16 -- the exact same shuffling call train_one_config's epoch loop makes, made
reproducible (the live training loop itself does NOT seed numpy, only torch, so its
own epoch order is not reproducible -- this script fixes that just for itself).

Forecast/scenario/price pipeline (mean, xi, q_norm, h_plus/h_minus, fc & true prices) is
computed ONCE and shared across all 5 gamma values via backward(retain_graph=True) --
isolates gamma's effect on the solve with zero confound from re-sampling or the
forecaster's own stochastic paths (e.g. dropout, if any) between gamma iterations. Model
weights are the frozen baseline for all iterations -- no optimizer step is taken.

WARNING: this runs ONE batch of 16 (batch_size=16), the same scale that OOM-killed
dual/k0's training twice at batch_size=16. Check `free -h` before running this
concurrently with the live batch_size=8 training run -- do not run both at once under
memory pressure.

Usage: uv run python gamma_sweep.py
"""
from __future__ import annotations
import pickle
import time
import sys

import numpy as np
import pandas as pd
import torch
from pyprojroot import here

ROOT_DIR = here()
sys.path.insert(0, str(ROOT_DIR / "7_model_training"))

from train_corner_point_robust import (
    DATA_DIR, FORECASTING_DIR, COPULA_DIR, DFL_TRAIN_DIR,
    reindex_and_impute, build_features, make_windows,
    normalise_hist, normalise_exo, denormalise_y, normalise_y,
    Baseline_Forecaster, TRAIN_START, VAL_START, TEST_START, HIST_COLS, FEAT_COLS, EXO_COLS,
    FrozenCopulaSampler, default_fixed_params, build_problem, make_layer,
    build_oracle, solve_oracle, get_prices, oracle_price_values, realised_breakdown,
    compute_box, PRICE_COLS_ALL, ISSUE_HOUR, HORIZON, N_HIST, device,
    forecast_train, pinball_per_day, self_balanced_loss, solve_with_retry,
    select_fc_columns, build_layer_vals, QUANTILE_LEVELS_TENSOR, BOX_LEVELS, ORACLE_SOLVER,
    TrainConfig, mode_from_corner,
)

PRICE_MODEL = "dual"
K = 0.0
BATCH_SIZE = 16
GAMMAS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
BASELINE_GAMMA = 1e-4
SEED = TrainConfig.seed  # 20240801 -- matches what train_one_config uses (torch only; numpy unseeded there)

RESULTS_PATH = DFL_TRAIN_DIR / "gamma_sweep_results.pkl"


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


def precompute_oracle_and_boxes(train_windows, batch_indices, baseline_model, sc, sampler, quantile_levels):
    phys_fp = default_fixed_params(gamma=BASELINE_GAMMA)  # oracle doesn't depend on gamma or k at all
    ob = build_oracle(phys_fp, PRICE_MODEL, objective="economic")
    oracle_costs = {}
    for d in batch_indices:
        realised_d = np.asarray(train_windows.y[d], float)
        price_day_d = np.asarray(train_windows.price[d], float)
        v = oracle_price_values(price_day_d, PRICE_MODEL, realised_d)
        oracle_costs[d] = solve_oracle(ob, v, solver=ORACLE_SOLVER)

    boxes = {}
    with torch.no_grad():
        for d in batch_indices:
            xh = normalise_hist(np.asarray(train_windows.x_hist[d]), sc)
            xh = torch.as_tensor(xh, dtype=torch.float32, device=device).unsqueeze(0)
            xf = normalise_exo(np.asarray(train_windows.x_fut[d]), sc)
            xf = torch.as_tensor(xf, dtype=torch.float32, device=device).unsqueeze(0)
            q_norm = baseline_model(xh, xf)
            q_phys = denormalise_y(q_norm, sc).squeeze(0).to(torch.float64)
            h_plus, h_minus = compute_box(q_phys, sampler, quantile_levels, box_levels=BOX_LEVELS)
            boxes[d] = (h_plus.detach().cpu().numpy(), h_minus.detach().cpu().numpy())
    return oracle_costs, boxes


def build_shared_pipeline(train_windows, batch_indices, model, sc, sampler, boxes):
    """Forecast -> scenario -> price pipeline, computed ONCE and reused (via
    retain_graph=True) across every gamma value -- isolates gamma's effect with no
    resampling confound."""
    realised = np.asarray(train_windows.y[batch_indices], float)
    price_day = np.asarray(train_windows.price[batch_indices], float)
    x_hist = train_windows.x_hist[batch_indices]
    x_fut = train_windows.x_fut[batch_indices]
    B = len(batch_indices)

    means_list, xi_list, q_norm_list = [], [], []
    for i in range(B):
        q_phys_i, q_norm_i = forecast_train(model, sc, x_hist[i], x_fut[i], device,
                                             normalise_hist, denormalise_y, normalise_exo)
        mean_i, xi_i = sampler.mean_and_errors(q_phys_i)
        means_list.append(mean_i); xi_list.append(xi_i); q_norm_list.append(q_norm_i)
    mean = torch.stack(means_list, dim=0)
    xi = torch.stack(xi_list, dim=0)
    q_norm_batch = torch.stack(q_norm_list, dim=0)

    h_plus = torch.stack([torch.as_tensor(boxes[d][0], dtype=mean.dtype, device=device) for d in batch_indices])
    h_minus = torch.stack([torch.as_tensor(boxes[d][1], dtype=mean.dtype, device=device) for d in batch_indices])

    fc_prices = get_prices(select_fc_columns(price_day), PRICE_MODEL)
    fc_prices_t = {kk: torch.as_tensor(np.asarray(vv, float), dtype=mean.dtype, device=device)
                   for kk, vv in fc_prices.items()}
    true_prices = get_prices(price_day, PRICE_MODEL)
    true_prices_t = {kk: torch.as_tensor(np.asarray(vv, float), dtype=mean.dtype, device=device)
                      for kk, vv in true_prices.items()}
    y_true_norm = torch.as_tensor(normalise_y(realised, sc), dtype=torch.float32, device=device)

    return {
        "realised": realised, "mean": mean, "xi": xi, "q_norm_batch": q_norm_batch,
        "h_plus": h_plus, "h_minus": h_minus,
        "fc_prices_t": fc_prices_t, "true_prices_t": true_prices_t, "y_true_norm": y_true_norm,
    }


def run_one_gamma(gamma, batch_indices, oracle_costs, shared, model):
    fp = default_fixed_params(gamma=gamma)
    bundle = build_problem(fp, mode_from_corner(PRICE_MODEL, K))
    layer = make_layer(bundle)
    keys = [p.name() for p in bundle.params]

    vals = build_layer_vals(shared["fc_prices_t"], shared["h_plus"], shared["h_minus"],
                             xi_samples=shared["xi"], Sigma_xi_chol=None)
    args = [vals[name] for name in keys]

    # ---- FORWARD (dispatch solve) ----
    t0 = time.perf_counter()
    dec, inaccurate = solve_with_retry(layer, args)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_forward = time.perf_counter() - t0

    p_ch_hat, p_dis_hat, D_ch, D_dis = dec
    p_da_rel = p_ch_hat - p_dis_hat
    bd = realised_breakdown(fp, p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_rel,
                             realised=shared["realised"], pl_hat=shared["mean"],
                             price_model=PRICE_MODEL, clip_recourse=True, **shared["true_prices_t"])
    raw_cost = bd.C_da + bd.C_imb
    oracle_arr = np.asarray([oracle_costs[d] for d in batch_indices], float)
    oracle_t = torch.as_tensor(oracle_arr, dtype=raw_cost.dtype, device=raw_cost.device)
    f_dfl = torch.clamp(raw_cost - oracle_t, min=0.0)
    L_base = pinball_per_day(shared["y_true_norm"], shared["q_norm_batch"], QUANTILE_LEVELS_TENSOR)
    loss = self_balanced_loss(L_base, f_dfl).sum()

    # ---- BACKWARD ----
    model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    loss.backward(retain_graph=True)  # shared upstream graph is reused by later gamma values
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_backward = time.perf_counter() - t0
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")))

    return {
        "gamma": gamma, "t_forward": t_forward, "t_backward": t_backward,
        "inaccurate": bool(inaccurate), "grad_norm": grad_norm,
        "p_ch_hat": p_ch_hat.detach().cpu().numpy(), "p_dis_hat": p_dis_hat.detach().cpu().numpy(),
        "D_ch": D_ch.detach().cpu().numpy(), "D_dis": D_dis.detach().cpu().numpy(),
        "C_da": bd.C_da.detach().cpu().numpy(), "C_imb": bd.C_imb.detach().cpu().numpy(),
        "raw_cost": raw_cost.detach().cpu().numpy(), "f_dfl": f_dfl.detach().cpu().numpy(),
        "oracle_cost": oracle_arr,
    }


def summarise(results, baseline_gamma):
    baseline = next(r for r in results if r["gamma"] == baseline_gamma)
    print(f"\n{'gamma':>8}  {'t_fwd(s)':>9}  {'t_bwd(s)':>9}  {'inaccur':>8}  {'grad_norm':>10}  "
          f"{'|dp_ch|':>9}  {'|dp_dis|':>9}  {'|dD_ch|_F':>10}  {'|dD_dis|_F':>11}  "
          f"{'raw_cost':>9}  {'f_dfl':>8}  {'|df_dfl|':>9}")
    for r in results:
        d_pch = np.mean(np.linalg.norm(r["p_ch_hat"] - baseline["p_ch_hat"], axis=1))
        d_pdis = np.mean(np.linalg.norm(r["p_dis_hat"] - baseline["p_dis_hat"], axis=1))
        d_Dch = np.mean(np.linalg.norm((r["D_ch"] - baseline["D_ch"]).reshape(len(r["D_ch"]), -1), axis=1))
        d_Ddis = np.mean(np.linalg.norm((r["D_dis"] - baseline["D_dis"]).reshape(len(r["D_dis"]), -1), axis=1))
        d_fdfl = np.mean(np.abs(r["f_dfl"] - baseline["f_dfl"]))
        print(f"{r['gamma']:>8.0e}  {r['t_forward']:>9.3f}  {r['t_backward']:>9.3f}  "
              f"{str(r['inaccurate']):>8}  {r['grad_norm']:>10.2f}  "
              f"{d_pch:>9.4f}  {d_pdis:>9.4f}  {d_Dch:>10.4f}  {d_Ddis:>11.4f}  "
              f"{r['raw_cost'].mean():>9.3f}  {r['f_dfl'].mean():>8.3f}  {d_fdfl:>9.4f}")
    print(f"\ndeviation columns (|dX|) are mean-over-16-days of the PER-DAY norm of "
          f"(value at gamma) - (value at baseline gamma={baseline_gamma:.0e}), not a raw mean "
          f"difference -- avoids cancellation across days.")


if __name__ == "__main__":
    print(f"device={device}  batch_size={BATCH_SIZE}  gammas={GAMMAS}  baseline={BASELINE_GAMMA:.0e}")

    train_windows = load_data()
    n_train = len(train_windows.delivery_start)

    np.random.seed(SEED)
    order = np.random.permutation(n_train)
    batch_indices = order[:BATCH_SIZE].tolist()
    print(f"sampled {len(batch_indices)} days (seeded={SEED}, reproducible): {batch_indices}")

    model, baseline_model, sc = load_frozen_model()
    cop = pickle.load(open(COPULA_DIR / "frozen_copula.pkl", "rb"))
    sampler = FrozenCopulaSampler(cop["Z_corr"], cop["quantile_levels"]).to(device)

    print("precomputing oracle costs + boxes for the 16 sampled days...")
    oracle_costs, boxes = precompute_oracle_and_boxes(
        train_windows, batch_indices, baseline_model, sc, sampler, cop["quantile_levels"])

    print("building shared forecast/scenario/price pipeline (once, reused across all gammas)...")
    shared = build_shared_pipeline(train_windows, batch_indices, model, sc, sampler, boxes)

    results = []
    for gamma in GAMMAS:
        print(f"running gamma={gamma:.0e} ...", flush=True)
        r = run_one_gamma(gamma, batch_indices, oracle_costs, shared, model)
        results.append(r)
        print(f"  t_fwd={r['t_forward']:.3f}s  t_bwd={r['t_backward']:.3f}s  "
              f"inaccurate={r['inaccurate']}  grad_norm={r['grad_norm']:.2f}  "
              f"mean_raw_cost={r['raw_cost'].mean():.3f}  mean_f_dfl={r['f_dfl'].mean():.3f}")

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({"batch_indices": batch_indices, "seed": SEED, "gammas": GAMMAS,
                     "baseline_gamma": BASELINE_GAMMA, "results": results}, f)
    print(f"\nsaved full per-day results -> {RESULTS_PATH}")

    summarise(results, BASELINE_GAMMA)
