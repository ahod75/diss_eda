"""
Decision-focused learning (DFL) training loop.

Fine-tunes the warm-started baseline forecaster against out-of-sample dispatch REGRET,
backpropagating through: forecaster -> quantiles -> frozen copula sampler -> scenarios
-> robust dispatch layer (cvxpylayers) -> realised cost.

Fixed vs learned:
  * LEARNED : the forecaster's weights (warm-started from the pinball-trained baseline).
  * FROZEN  : the copula Sigma / Z_corr (sampler buffers), and the dispatch structure.

Discipline carried over from the pipeline contract:
  * DFL uses GATE-ALIGNED daily origins only (one window per delivery day).
  * Selection is on VALIDATION regret; the test window stays sealed.
  * Single-price is the symmetric special case of the dual-price layer (lam_up == lam_dn),
    so both are the SAME code path, switched by the price provider.

This file is a skeleton: the four PLUG points (load forecaster, load sampler, load windows,
price provider) are where your window_loader / checkpoints connect. The training mechanics,
the differentiable chain, early stopping and the regret readout are concrete.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import copy
import time

import numpy as np
import torch

from dispatch_layer import FixedParams, build_layer
from dispatch_wrapper import make_dispatch_inputs, realised_cost, build_oracle, regret


# =========================================================================
# Config
# =========================================================================
@dataclass
class DFLConfig:
    price_mode: str = "dual"        # "dual" or "single" (single => lam_up == lam_dn)
    lr: float = 1e-4                # SMALL: this is fine-tuning a warm start, not training
    max_epochs: int = 100
    patience: int = 10              # early stop on validation regret
    batch_days: int = 16            # days per optimiser step (loss accumulated over them)
    weight_decay: float = 1e-4
    dropout_in_training: bool = False  # forecaster.train() (dropout on) vs eval() during DFL
    clip_recourse: bool = False     # passed to realised_cost (physical battery saturation)
    box_levels: tuple = (0.05, 0.95)
    grad_clip: float | None = 1.0   # optional grad-norm clip (DFL gradients can be spiky)
    seed: int = 20240801
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================================
# PLUG POINTS  — connect these to your project
# =========================================================================
def load_forecaster(device):
    """Rebuild the baseline (128) model and load the warm-start weights.
    Returns a torch nn.Module in the given device, with grad enabled on its params.
    """
    # from forecasting import Baseline_Forecaster
    # ckpt = torch.load(".../baseline_forecaster_best.pt", weights_only=False, map_location="cpu")
    # model = Baseline_Forecaster(**ckpt["model_config"])
    # model.load_state_dict(ckpt["state_dict"])
    # model.to(device)
    # sc = ckpt["scaler_stats"]                 # {"mu_y":..., "sd_y":...}
    # levels = np.asarray(ckpt["quantile_levels"], float)
    # return model, sc, levels
    raise NotImplementedError("wire in the baseline checkpoint load")


def load_sampler(device):
    """Load the frozen copula bundle and build a FrozenCopulaSampler (buffers only).
    Returns the sampler (on device). Its S must equal FixedParams.num_scenarios.
    """
    # from copula import FrozenCopulaSampler   # wherever the class lives
    # bundle = pickle.load(open(".../frozen_copula.pkl", "rb"))
    # Z = bundle["Z_corr"]; levels = bundle["quantile_levels"]
    # return FrozenCopulaSampler(Z, levels).to(device)
    raise NotImplementedError("wire in the frozen copula sampler")


def load_windows():
    """Return the GATE-ALIGNED daily windows for train and validation.
    Each split is a dict of arrays aligned by day index:
        x_hist : (D, n_hist, n_hist_feat)
        x_fut  : (D, horizon, n_exo)
        y      : (D, T)      realised prosumption per delivery day
        dates  : (D,)        delivery dates (for the price provider / diagnostics)
    Test is NOT loaded here (stays sealed).
    """
    # build with make_windows(..., gate_aligned_only=True, issue_hour=9) on TRAIN and VAL
    raise NotImplementedError("wire in the gate-aligned train/val windows")


def get_prices(date, mode):
    """Return (pi_da, lam_up, lam_dn) for a delivery day, each (T,).
    lam_up/lam_dn are pre-saved in the window loader's file.
      * mode == "dual"   : the asymmetric lam_up != lam_dn as stored.
      * mode == "single" : a symmetric penalty -> lam_up == lam_dn (e.g. both = a single
                           imbalance price, or max(lam_up, lam_dn), per your single-price defn).
    """
    # pi_da, lam_up, lam_dn = window_loader.prices(date)
    # if mode == "single":
    #     lam = np.maximum(lam_up, lam_dn)      # or your defined single imbalance price
    #     lam_up = lam_dn = lam
    # return pi_da, lam_up, lam_dn
    raise NotImplementedError("wire in the price provider")


# =========================================================================
# Differentiable per-day forward
# =========================================================================
def _denormalise_quantiles(q_std, sc, device):
    """Affine, differentiable inverse-standardisation: physical = std * sd_y + mu_y."""
    sd_y = torch.as_tensor(float(sc["sd_y"]), dtype=q_std.dtype, device=device)
    mu_y = torch.as_tensor(float(sc["mu_y"]), dtype=q_std.dtype, device=device)
    return q_std * sd_y + mu_y


def forward_day(model, sampler, layer, fp, cfg, sc, levels, W, d, prices):
    """One day: forecaster -> quantiles -> inputs -> layer -> realised cost (differentiable).

    Returns the realised-cost scalar (the DFL loss contribution for this day).
    """
    dev = cfg.device
    xh = torch.as_tensor(W["x_hist"][d], dtype=torch.float32, device=dev).unsqueeze(0)
    xf = torch.as_tensor(W["x_fut"][d],  dtype=torch.float32, device=dev).unsqueeze(0)

    q_std = model(xh, xf).squeeze(0)                    # (K, Q) standardised, grad live
    quantiles = _denormalise_quantiles(q_std, sc, dev)  # (K, Q) physical MW, grad live

    pi_da, lam_up, lam_dn = prices
    di = make_dispatch_inputs(
        quantiles, sampler, levels,
        pi_da=pi_da, lam_up=lam_up, lam_dn=lam_dn,
        box_levels=cfg.box_levels, box_detach=True,
    )

    # cvxpylayers expects float64 for stable diff; cast the layer args
    args = tuple(a.double() for a in di.layer_args())
    p_ch_hat, p_dis_hat, D_ch, D_dis = layer(*args)     # decisions, grad live

    realised = W["y"][d]
    cost = realised_cost(
        fp, p_ch_hat, p_dis_hat, D_ch, D_dis,
        realised=realised, pl_hat=di.pl_hat.double(),
        pi_da=di.pi_da, lam_up=di.lam_up, lam_dn=di.lam_dn,
        clip_recourse=cfg.clip_recourse,
    )
    return cost


# =========================================================================
# Validation regret (no grad); oracle costs cached (forecaster-independent)
# =========================================================================
def precompute_oracle_costs(fp, W, cfg):
    """Oracle cost per validation day — depends only on realised data, so compute ONCE."""
    solve = build_oracle(fp)
    costs = np.empty(len(W["y"]))
    for d in range(len(W["y"])):
        pi_da, _, _ = get_prices(W["dates"][d], cfg.price_mode)
        costs[d] = solve(np.asarray(W["y"][d], float), np.asarray(pi_da, float))
    return costs


@torch.no_grad()
def validation_regret(model, sampler, layer, fp, cfg, sc, levels, W_val, oracle_costs):
    """Mean regret over the validation days (the early-stopping / selection metric)."""
    was_training = model.training
    model.eval()
    total = 0.0
    for d in range(len(W_val["y"])):
        prices = get_prices(W_val["dates"][d], cfg.price_mode)
        cost = forward_day(model, sampler, layer, fp, cfg, sc, levels, W_val, d, prices)
        total += float(cost) - float(oracle_costs[d])
    if was_training:
        model.train()
    return total / len(W_val["y"])


# =========================================================================
# Training
# =========================================================================
def train_dfl(fp: FixedParams, cfg: DFLConfig):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = cfg.device

    # ---- load pieces ----
    model, sc, levels = load_forecaster(dev)
    sampler = load_sampler(dev)
    assert sampler.S == fp.num_scenarios, \
        f"sampler S={sampler.S} != fp.num_scenarios={fp.num_scenarios}"
    W_train, W_val = load_windows()

    # ---- build the differentiable dispatch layer ONCE ----
    H = build_layer(fp)
    layer = H["layer"]
    assert layer is not None, "cvxpylayers not available — cannot build differentiable layer"

    # ---- cache oracle costs on the validation set (forecaster-independent) ----
    oracle_costs = precompute_oracle_costs(fp, W_val, cfg)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_train = len(W_train["y"])
    best_val, best_epoch, best_state = float("inf"), -1, None
    stale = 0
    t0 = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train() if cfg.dropout_in_training else model.eval()
        order = np.random.permutation(n_train)

        running = 0.0
        for start in range(0, n_train, cfg.batch_days):
            batch = order[start:start + cfg.batch_days]
            opt.zero_grad()
            batch_loss = 0.0
            for d in batch:
                prices = get_prices(W_train["dates"][d], cfg.price_mode)
                cost = forward_day(model, sampler, layer, fp, cfg,
                                   sc, levels, W_train, d, prices)
                batch_loss = batch_loss + cost
            batch_loss = batch_loss / len(batch)        # mean realised cost over the batch
            batch_loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += float(batch_loss) * len(batch)

        # ---- validation regret + early stopping (restore best) ----
        val = validation_regret(model, sampler, layer, fp, cfg, sc, levels, W_val, oracle_costs)
        train_cost = running / n_train
        print(f"epoch {epoch:3d} | train cost {train_cost:.4f} | val regret {val:.4f}"
              f" | best {best_val:.4f}@{best_epoch}")

        if val < best_val:
            best_val, best_epoch = val, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"early stop at epoch {epoch} (no val improvement for {cfg.patience})")
                break

    model.load_state_dict(best_state)                   # restore best weights
    print(f"\nDFL done in {time.time()-t0:.0f}s | best val regret {best_val:.4f} @ epoch {best_epoch}")

    return dict(model=model, best_val_regret=best_val, best_epoch=best_epoch,
                price_mode=cfg.price_mode, seed=cfg.seed)


# =========================================================================
if __name__ == "__main__":
    # Example driver. FixedParams must match your feeder/battery + sampler S.
    fp = FixedParams(
        T_total=24, num_scenarios=256, dt=1.0,
        eta_ch=0.95, eta_dis=0.95, C_ch=2.0, C_dis=2.0,
        B_max=6.0, SOC0=3.0, gamma=1e-3, terminal_soc_equality=True,
    )

    # Run BOTH price structures for the DFL-vs-baseline-under-single-vs-dual contribution.
    for mode in ["single", "dual"]:
        cfg = DFLConfig(price_mode=mode)
        print(f"\n================= DFL ({mode} price) =================")
        # out = train_dfl(fp, cfg)     # uncomment once the PLUG points are wired
        print("(wire the load_* / get_prices plug points, then uncomment train_dfl)")
