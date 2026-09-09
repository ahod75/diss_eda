"""
forecaster.py
=============

Reference implementation of a probabilistic net-load (prosumption) forecaster,
adapted from vom Scheidt et al. (2021), "Probabilistic Forecasting of Household
Loads: Effects of Distributed Energy Technologies on Forecast Quality".

This is the STATISTICAL baseline forecaster and, simultaneously, the warm-start
initialisation for the decision-focused (DFL) arm — same architecture, trained
first on pinball loss here, later fine-tuned on decision loss.

Four deliberate departures from the original notebook, each required by your setup
(see the thread for the reasoning):

  1. PyTorch, not Keras/TF.
     The downstream decision-focused layer (cvxpylayers / qpth) is PyTorch-native,
     so the forecaster must be PyTorch for decision-loss gradients to flow into it.

  2. Direct multi-horizon (K=24), not single-step (n_steps_out=1).
     You issue ONE 24-hour day-ahead vector per day (clock-aligned, midnight→
     midnight), which is what the copula correlates across lead times and what the
     dispatch problem consumes. The head emits (K=24 lead times × Q quantiles) in
     one shot, from daily origins.

  3. Monotone (non-crossing) quantiles BY CONSTRUCTION.
     The Pinson copula needs a monotone, invertible per-lead-time CDF. vom Scheidt
     train five independent sigmoid heads that can cross. Here each lead time emits
     a base quantile plus positive (softplus) increments, so quantiles are strictly
     ordered — differentiably — and give a clean CDF. Also uses more levels
     (default 19, 0.05..0.95) since 5 is too coarse for the copula.

  4. Negative-capable (linear output on standardised targets), not sigmoid on [0,1].
     Prosumption = demand − PV can be negative (export). A [0,1] sigmoid output is
     wrong here; we standardise the target (fit on train only) and use a linear
     output, then inverse-transform quantiles back to physical MWh.

Connects to scenario_generation.py: `to_predictive_marginals()` turns a day's
forecast into the PredictiveMarginal objects that module consumes.

Dependencies: torch, numpy, pandas. (Not executed in the authoring environment —
run the smoke test in __main__ in your own env; the numerical logic was checked
in NumPy separately.)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Default quantile levels — 19 levels, matching the Pinson scenario setup.
QUANTILE_LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# --------------------------------------------------------------------------- #
# 1. Feature engineering (calendar + weather), following vom Scheidt's features
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame,
                   load_col: str = "net_load",
                   weather_cols=("temp", "wind_speed", "rel_humidity")) -> pd.DataFrame:
    """
    df: DatetimeIndex (hourly), columns = [load_col, *weather_cols].
    Returns a frame with the raw net-load, plus calendar features (hour-of-day
    and weekday, both cyclical) and the weather columns. Cyclical encoding is used
    instead of one-hot to keep the exogenous vector small (matters given ~400
    daily training samples).
    """
    out = pd.DataFrame(index=df.index)
    out["net_load"] = df[load_col].astype(float)
    h = df.index.hour.values
    wd = df.index.dayofweek.values
    out["hod_sin"] = np.sin(2 * np.pi * h / 24)
    out["hod_cos"] = np.cos(2 * np.pi * h / 24)
    out["dow_sin"] = np.sin(2 * np.pi * wd / 7)
    out["dow_cos"] = np.cos(2 * np.pi * wd / 7)
    out["is_weekend"] = (wd >= 5).astype(float)
    for c in weather_cols:
        if c in df.columns:
            out[c] = df[c].astype(float)
    return out


EXO_COLS_DEFAULT = ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "is_weekend"]


# --------------------------------------------------------------------------- #
# 2. Standardiser (fit on TRAIN ONLY — no leakage)
# --------------------------------------------------------------------------- #
class Standardiser:
    """z-score per column, parameters fit on the training slice only."""
    def __init__(self):
        self.mu = {}
        self.sd = {}

    def fit(self, frame: pd.DataFrame, cols):
        for c in cols:
            self.mu[c] = float(frame[c].mean())
            self.sd[c] = float(frame[c].std() or 1.0)
        return self

    def transform(self, frame: pd.DataFrame):
        out = frame.copy()
        for c, mu in self.mu.items():
            out[c] = (frame[c] - mu) / self.sd[c]
        return out

    def inverse_target(self, y_std, col="net_load"):
        """Map standardised net-load values back to physical units (MWh)."""
        return y_std * self.sd[col] + self.mu[col]


# --------------------------------------------------------------------------- #
# 3. Daily-origin windowing (direct multi-horizon)
# --------------------------------------------------------------------------- #
def make_daily_windows(frame: pd.DataFrame, n_hist: int, horizon: int = 24,
                       issue_hour: int = 0, exo_cols=EXO_COLS_DEFAULT,
                       weather_cols=("temp", "wind_speed", "rel_humidity")):
    """
    One sample per day: at `issue_hour` (default midnight), use the preceding
    `n_hist` hours of net load as the encoder input, and predict the next
    `horizon` hourly net-load values. Horizon exogenous features (calendar +
    weather FORECAST — here we use realised weather as a proxy; swap in your
    forecast series to avoid look-ahead) are supplied for each of the K steps.

    Returns:
      x_hist : (N, n_hist, 1)         standardised net-load history
      x_exo  : (N, horizon, F_exo)    exogenous features over the horizon
      y      : (N, horizon)           standardised net-load targets
      origins: (N,) DatetimeIndex     issue timestamps (for traceability)
    """
    exo_all = [c for c in exo_cols] + [c for c in weather_cols if c in frame.columns]
    load = frame["net_load"].values
    exo = frame[exo_all].values
    idx = frame.index
    # candidate origins: rows at issue_hour with enough history and horizon ahead
    is_issue = (idx.hour == issue_hour)
    xh, xe, ys, orig = [], [], [], []
    for i in np.where(is_issue)[0]:
        if i - n_hist < 0 or i + horizon > len(frame):
            continue
        xh.append(load[i - n_hist:i])
        xe.append(exo[i:i + horizon])
        ys.append(load[i:i + horizon])
        orig.append(idx[i])
    x_hist = np.asarray(xh, dtype=np.float32)[..., None]
    x_exo = np.asarray(xe, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return x_hist, x_exo, y, pd.DatetimeIndex(orig)


# --------------------------------------------------------------------------- #
# 4. The quantile GRU (monotone, multi-horizon, negative-capable)
# --------------------------------------------------------------------------- #
class QuantileGRU(nn.Module):
    """
    Encoder: GRU over the net-load history -> summary hidden state.
    Head: for each of the K horizon steps, concatenate the summary with that
    step's exogenous features and map (via a shared MLP) to Q monotone quantiles.

    Output: quantiles of shape (B, K, Q), strictly increasing along Q by
    construction (base + cumulative softplus increments).
    """
    def __init__(self, n_exo, horizon=24, n_quantiles=len(QUANTILE_LEVELS),
                 gru_hidden=16, n_gru_layers=2, dense_hidden=32, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.gru = nn.GRU(input_size=1, hidden_size=gru_hidden,
                          num_layers=n_gru_layers, batch_first=True,
                          dropout=dropout if n_gru_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden + n_exo, dense_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, dense_hidden), nn.ReLU(),
            nn.Linear(dense_hidden, n_quantiles),
        )

    def forward(self, x_hist, x_exo):
        # x_hist: (B, L, 1);  x_exo: (B, K, F_exo)
        _, h = self.gru(x_hist)              # h: (n_layers, B, H)
        summary = h[-1]                      # (B, H)
        B, K, _ = x_exo.shape
        summary = summary.unsqueeze(1).expand(B, K, summary.shape[-1])  # (B,K,H)
        feat = torch.cat([summary, x_exo], dim=-1)     # (B, K, H+F_exo)
        raw = self.head(feat)                          # (B, K, Q)
        base = raw[..., :1]
        deltas = torch.nn.functional.softplus(raw[..., 1:])   # strictly > 0
        quant = torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)
        return quant                                   # (B, K, Q), monotone in Q


# --------------------------------------------------------------------------- #
# 5. Pinball (quantile) loss — multi-horizon, multi-quantile
# --------------------------------------------------------------------------- #
def pinball_loss(y_true, y_pred, levels):
    """
    y_true : (B, K)      standardised targets
    y_pred : (B, K, Q)   monotone quantile forecasts
    levels : (Q,) tensor
    Averaged over batch, horizon and quantiles. Summed pinball across levels
    approximates CRPS, so this training objective aligns with your reported metric.
    """
    e = y_true.unsqueeze(-1) - y_pred                   # (B, K, Q)
    q = levels.view(1, 1, -1)
    return torch.maximum(q * e, (q - 1.0) * e).mean()


# --------------------------------------------------------------------------- #
# 6. Statistical pre-training  (== baseline model AND warm-start init for DFL)
# --------------------------------------------------------------------------- #
def train_statistical(model, train_tensors, val_tensors, levels,
                      lr=1e-3, weight_decay=1e-4, n_epochs=200,
                      batch_size=32, patience=20, device="cpu", seed=0):
    """
    Trains `model` on pinball loss with early stopping on validation pinball.
    Returns the model loaded with its best (lowest val-loss) weights — this is
    both your statistical baseline and the initialisation you warm-start DFL from.

    train_tensors / val_tensors: (x_hist, x_exo, y) numpy arrays.
    """
    torch.manual_seed(seed)
    model.to(device)
    lv = torch.as_tensor(levels, dtype=torch.float32, device=device)

    def to_t(arrs):
        return [torch.as_tensor(a, dtype=torch.float32, device=device) for a in arrs]
    xh, xe, y = to_t(train_tensors)
    vxh, vxe, vy = to_t(val_tensors)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = xh.shape[0]
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)   # shuffle windows (each is self-contained)
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            loss = pinball_loss(y[b], model(xh[b], xe[b]), lv)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = pinball_loss(vy, model(vxh, vxe), lv).item()
        if vloss < best_val - 1e-6:
            best_val, best_state, wait = vloss, {k: v.detach().clone()
                                                 for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# --------------------------------------------------------------------------- #
# 7. Prediction in physical units + bridge to the copula scenario generator
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_quantiles(model, x_hist, x_exo, scaler: Standardiser, device="cpu"):
    """Return physical-unit quantiles, shape (N, K, Q)."""
    model.eval().to(device)
    xh = torch.as_tensor(x_hist, dtype=torch.float32, device=device)
    xe = torch.as_tensor(x_exo, dtype=torch.float32, device=device)
    q_std = model(xh, xe).cpu().numpy()                 # (N, K, Q)
    return scaler.inverse_target(q_std, col="net_load")  # affine, keeps monotonicity


def to_predictive_marginals(day_quantiles, levels=QUANTILE_LEVELS):
    """
    Turn one day's physical-unit forecast (K, Q) into a list of K PredictiveMarginal
    objects for scenario_generation.py. Import is local so this file stands alone.
    """
    from scenario_generation import PredictiveMarginal
    K = day_quantiles.shape[0]
    return [PredictiveMarginal(levels, day_quantiles[k]) for k in range(K)]


# --------------------------------------------------------------------------- #
# DFL HOOK (next stage — not implemented here):
#   After statistical pre-training, WARM-START the DFL model from this one:
#       dfl_model = QuantileGRU(...); dfl_model.load_state_dict(model.state_dict())
#   Then replace the pinball objective with decision loss:
#       marginals -> PredictiveMarginal CDFs -> Pinson copula scenarios (FROZEN Sigma
#       + fixed common-random draws) -> differentiable dispatch layer (cvxpylayers)
#       -> decision cost. Keep a light pinball term for calibration (see thread).
#   Only the marginals (this model's parameters) are trained; Sigma stays frozen.
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    # -------- synthetic smoke test (run in an env with torch installed) --------
    rng = np.random.default_rng(0)
    hours = pd.date_range("2018-01-01", periods=24 * 500, freq="h")
    # synthetic prosumption: diurnal + weekly + solar dip (can go negative) + noise
    h = hours.hour.values
    base = 3 + 1.5 * np.sin(2 * np.pi * (h - 18) / 24)
    solar = 2.5 * np.clip(np.sin(2 * np.pi * (h - 6) / 24), 0, None)
    net = base - solar + 0.4 * rng.standard_normal(len(hours))
    df = pd.DataFrame({"net_load": net,
                       "temp": 10 + 8 * np.sin(2 * np.pi * hours.dayofyear.values / 365) + rng.standard_normal(len(hours)),
                       "wind_speed": np.abs(rng.standard_normal(len(hours)) * 5),
                       "rel_humidity": rng.uniform(0.4, 0.9, len(hours))}, index=hours)

    feats = build_features(df)
    # chronological split: train first 400 days, val next 50, test rest
    split_cols = ["net_load", "temp", "wind_speed", "rel_humidity"]
    train_end = hours[400 * 24]
    val_end = hours[450 * 24]
    scaler = Standardiser().fit(feats.loc[:train_end], split_cols)
    fs = scaler.transform(feats)

    n_hist = 168
    tr = make_daily_windows(fs.loc[:train_end], n_hist)
    va = make_daily_windows(fs.loc[train_end:val_end], n_hist)
    te = make_daily_windows(fs.loc[val_end:], n_hist)
    print("windows  train:", tr[0].shape, "val:", va[0].shape, "test:", te[0].shape)

    n_exo = tr[1].shape[-1]
    model = QuantileGRU(n_exo=n_exo)
    model, best = train_statistical(model, tr[:3], va[:3], QUANTILE_LEVELS,
                                    n_epochs=40, patience=8)
    print(f"best val pinball: {best:.4f}")

    q = predict_quantiles(model, te[0], te[1], scaler)     # (N, K, Q) physical units
    print("test quantiles:", q.shape,
          "| monotone:", bool(np.all(np.diff(q, axis=-1) >= -1e-6)))
    marg = to_predictive_marginals(q[0]) if False else None  # needs scenario_generation.py
    print("smoke test OK")
