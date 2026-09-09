"""
forecasting_pipeline.py
=======================

pipeline ordering:

"""

from __future__ import annotations

# ============================================================================ #
# Section 0 — imports                                          (was cell 0)
# ============================================================================ #
import matplotlib.pyplot as plt
from pyprojroot import here
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
# Section 0.5 — dataclasses
# ========================================================================== #
@dataclass
class WindowSet:
    x_hist: np.ndarray | None            # (N, n_hist, C_hist) or None
    x_fut:  np.ndarray | None            # (N, horizon, F_exo) or None
    y:      np.ndarray | None            # (N, horizon)        or None
    price:  np.ndarray | None            # (N, horizon, n_price) or None
    delivery_start: pd.DatetimeIndex     # (N,) delivery-start times
    hs: pd.Timestamp                     # history start
    he: pd.Timestamp                     # history end
    ds: pd.Timestamp                     # delivery start
    de: pd.Timestamp                     # delivery end


# ============================================================================ #
# Section 1 — constants & configuration
# ============================================================================ #
# [MODIFIED] QUANTILE_LEVELS was defined identically in cells 3 and 9 — one copy.
QUANTILE_LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 19 levels
NUM_QUANTILES = len(QUANTILE_LEVELS)

HIST_COLS = ["prosumption", "solar_irrad", "panel_temp", "ambient_temp"]
FEAT_COLS = ["solar_irrad", "panel_temp", "ambient_temp"]
EXO_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "solar_irrad",
    "ambient_temp",
]

DEFAULT_MODEL_CONFIG = {
    "n_hist_features": len(HIST_COLS),
    "n_exo_features": len(EXO_COLS),
    "n_quantiles": NUM_QUANTILES,
    "gru_hidden_size": 32,
    "dense_width": 64,
    "dropout": 0.1,
    "n_gru_layers": 2,
    "eps": 1e-4,
}

# --- split boundaries --------------------------------------------------------
TRAIN_START = pd.Timestamp("2018-01-01 00:00:00+00:00")
VAL_START = pd.Timestamp("2019-01-01 00:00:00+00:00")
TEST_START = pd.Timestamp("2019-07-01 00:00:00+00:00")
TEST_END = pd.Timestamp("2020-06-30 23:00:00+00:00")

# old test split (for the month-long experiment to test if model works)
MONTH_TRAIN_START = pd.Timestamp("2018-01-01 00:00:00+00:00")
MONTH_TRAIN_END = pd.Timestamp("2018-03-31 23:00:00+00:00")
MONTH_VAL_START = pd.Timestamp("2018-04-01 00:00:00+00:00")
MONTH_VAL_END = pd.Timestamp("2018-04-30 23:00:00+00:00")


# ============================================================================ #
# Section 2 — feature engineering
# ============================================================================ #
def build_features(
    df: pd.DataFrame,
    prosumption_col: str = "prosumption",
    feature_cols: list[str] = ["solar_irrad", "panel_temp", "ambient_temp"],
    price_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    df: DatetimeIndex (hourly), columns = [load_col, *feature_cols].
    Returns a frame with the prosumption, plus calendar features (hour-of-day
    and weekday, both cyclical) and the feature columns. Cyclical encoding is used
    for hour-of-day and day-of-week, but one-hot for weekends.

    price_cols defaults to None (no price columns at all) rather than a fixed list --
    deliberately, so a caller that forgets to pass it gets a frame with NO price data
    (loud, immediate KeyError/missing-column downstream) rather than silently getting
    whatever the last price-column convention happened to be. A hardcoded default here
    already caused a real bug once (evaluate.py's load_test_windows and both h-sweep
    scripts silently pulled a stale price_cols default after the imb_up/imb_down
    rename) -- explicit callers only, from here on.
    """
    out = pd.DataFrame(index=df.index)
    out[prosumption_col] = df[prosumption_col].astype(float)
    dateindex = df.index
    # Calendar features
    hr = dateindex.hour.values
    dow = dateindex.dayofweek.values
    # mon       = dateindex.month.values
    doy = dateindex.dayofyear.values
    is_weekend = (dateindex.dayofweek >= 5).astype(int)

    # Cyclical encodings
    out["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["is_weekend"] = is_weekend
    for c in feature_cols:
        out[c] = df[c].astype(float)
    if price_cols is not None:
        for d in price_cols:
            out[d] = df[d].astype(float)

    return out


# ============================================================================ #
# Section 3 — windowing
# ============================================================================ #
def make_windows(
    frame: pd.DataFrame,

    y_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,  # keep windows whose ENTIRE delivery block falls in [y_start, y_end] inclusive
    gate_aligned_only: bool | None = None,  # False = all-hours rolling origins (baseline augmentation); True = one daily gate origin (DFL / frozen artefacts)
    issue_hour: int | None = None,  # hour-of-day the forecast is issued (gate closure) (sets offset to correct offset of 24 - issue_hour)
    lead_gap: int   | None = None,  # secondary, optional way of setting offset. last-obs -> first delivery step; default 24 - issue_hour => next-day 00:00 first delivery
    exo_cols: list[str] | None = None,
    hist_cols: list[str] | None = None,
    price_cols: list[str] | None = None,
    target_col: str | None = None,

    n_hist: int = 168,
    horizon: int = 24
) -> WindowSet: 
    
    """
    Function for creating windows for historical data, exogenous variables and price data.
    Same function used for all of these to ensure that the windows are aligned and consistent across all data types.
    Baseline forecast:
        Historical (inc. prosumption), xogenous data and target prosumption
    DFL forecast:
        Historical (inc. prosumption), exogenous data, target prosumption  and price data
    Testing forecasters:
        Historical (inc. prosumption), exogenous data, target prosumption  and price data
    Testing best-decision:
        Target prosumption and price data
    
    Window defined in relation to TARGET (delivery) window.
    A window is kept iff:
    - it has n_hist hours of history before the origin, and
    - its full delivery block [d0 .. d0+horizon-1] lies within the frame, and
    - (if y_range given) that full delivery block lies within [y_start, y_end] inclusive.

    Split cleanliness: because the ENTIRE delivery block must sit inside y_range,
    adjacent splits share zero target hours -> no target leakage across boundaries,
    with no gap arithmetic required. History may (and for early targets must) reach
    back before y_start: pass a frame that extends >= (n_hist + lead_gap) hours before
    y_start so the earliest in-period targets are not dropped for lack of history.

    y_range should be given per split, e.g.:
        train: (TRAIN_START, VAL_START  - 1h)
        val:   (VAL_START,   TEST_START - 1h)
        test:  (TEST_START,  TEST_END)
    (the -1h keeps the day-boundary exclusive so splits abut without overlapping.)
    """

    ## --- integrity checks ---
    # positional slicing assumes a sorted, gap-free, regular hourly (UTC) grid
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be sorted ascending")
    steps = frame.index.to_series().diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta(hours=1):
        raise ValueError(
            f"index must be a complete regular hourly grid (found steps: {steps}). "
            "Reindex to a full hourly UTC range before windowing."
        )

    # make sure gate_aligned_only is provided
    if gate_aligned_only is None:
        raise ValueError("must provide gate_aligned_only True/False")

    # if gate_aligned_only is True, must only provide issue_hour, not lead_gap.
    if gate_aligned_only is True and lead_gap is not None:
        raise ValueError("must provide issue_hour, not lead_gap, if gate_aligned_only is True")
    if gate_aligned_only is True and issue_hour is None:
        raise ValueError("must provide issue_hour if gate_aligned_only is True")

    # make sure lead_gap or issue_hour is provided, but not both.
    if lead_gap is None and issue_hour is None:
        raise ValueError("must provide either lead_gap or issue_hour")
    if lead_gap is not None and issue_hour is not None:
        raise ValueError("must provide either lead_gap or issue_hour, not both")

    if lead_gap is None:
        lead_gap = (
            24 - issue_hour
        )  # issue 09:00 -> first delivery = next-day 00:00 (+15..+38h)
    
    ## --- Setup internal arrays based on what the function is being asked to collect ---
    if hist_cols:
        hist = frame[hist_cols].values.astype(np.float32)  # (T, C_hist)
    if exo_cols:
        exo = frame[exo_cols].values.astype(np.float32)  # (T, F_exo)
    if target_col:
        y_all = frame[target_col].values.astype(np.float32)  # (T,)
    if price_cols:
        price_all = frame[price_cols].values.astype(np.float32)  # (T, F_price)

    idx = frame.index
    T = len(frame)

    y_lo = pd.Timestamp(y_range[0]) if y_range is not None else None
    y_hi = pd.Timestamp(y_range[1]) if y_range is not None else None

    # If gate_aligned_only, then only consider windows where the origin hour matches the issue_hour (e.g., 09:00).
    # Otherwise, consider all hours as potential origins.

    ## --- Identify candidate origins based on the issue hour and gate alignment preference ---
    candidates = (
        np.where(idx.hour == issue_hour)[0] if gate_aligned_only else np.arange(T)
    )

    ## --- Iterate over candidate origins and collect windows that meet the criteria ---
    xh, xf, ys, pr, timeind, delivery_start = [], [], [], [], [], []
    for i in candidates:
        d0 = i + lead_gap  # first delivery index
        last = d0 + horizon - 1  # last delivery index
        # history present AND full delivery block in-frame
        if i - n_hist < 0 or last >= T:
            continue
        # full delivery block inside the target range (both ends) -> no boundary crossing
        if y_lo is not None and (idx[d0] < y_lo or idx[last] > y_hi):
            continue
        if hist_cols:
            xh.append(
                hist[i - n_hist : i]
            )  # (n_hist, C_hist) up to the last complete hour before gate closure
        if exo_cols:
            xf.append(exo[d0 : d0 + horizon])  # (horizon, F_exo) over delivery window
        if target_col:
            ys.append(y_all[d0 : d0 + horizon])  # (horizon,)       over delivery window
        if price_cols:
            pr.append(price_all[d0 : d0 + horizon])  # (horizon,)       over delivery window

        timeind.append(i)
        delivery_start.append(idx[d0])

    ## --- Final checks and return the WindowSet dataclass ---
    if not timeind:
        raise ValueError(
            "no windows produced — check y_range, frame extent, or n_hist/lead_gap"
        )

    # tracking of history start/end, delivery start/end for data normalisation

    #return WindowSet dataclass, filling each variable dependent
    return WindowSet(
        x_hist = np.asarray(xh, np.float32) if hist_cols  else None,
        x_fut  = np.asarray(xf, np.float32) if exo_cols   else None,
        y      = np.asarray(ys, np.float32) if target_col else None,
        price  = np.asarray(pr, np.float32) if price_cols else None,
        delivery_start = pd.DatetimeIndex(delivery_start),
        hs = idx[min(timeind) - n_hist],
        he = idx[max(timeind) - 1],
        ds = idx[min(timeind) + lead_gap],
        de = idx[max(timeind) + lead_gap + horizon - 1],
    )

# ============================================================================ #
# Section 4 — normalisation
# ============================================================================ #
def fit_scalers(frame_slice, hist_cols, target_col="prosumption", exo_cols=None):
    """
    frame_slice : raw rows the TRAINING windows touch, e.g. frame.loc[hs:de]
                  (each hour counted once — no window double-counting).
    hist_cols   : channel order of x_hist, e.g.
                  ["prosumption", "irradiance_Wm-2", "panel_temp_C", "temp_location3"]
    exo_cols    : channel order of x_fut, e.g. EXO_COLS. Optional so old checkpoints'
                  scaler dicts (fit before exo normalisation existed) still load; when
                  omitted, normalise_exo is a no-op (mu=0, sd=1) rather than erroring.
    Returns a dict of frozen stats.
    """
    mu_hist = np.array([frame_slice[c].mean() for c in hist_cols], dtype=np.float32)
    sd_hist = np.array([frame_slice[c].std() for c in hist_cols], dtype=np.float32)
    sd_hist[sd_hist == 0] = 1.0
    # target shares the prosumption series -> same scaler (computed from same column)
    mu_y = float(frame_slice[target_col].mean())
    sd_y = float(frame_slice[target_col].std() or 1.0)
    out = {
        "hist_cols": list(hist_cols),
        "mu_hist": mu_hist,
        "sd_hist": sd_hist,
        "mu_y": mu_y,
        "sd_y": sd_y,
    }
    if exo_cols:
        # Exo columns that ALSO appear in hist_cols (solar_irrad, ambient_temp -- the raw
        # physical-unit weather channels) reuse that EXACT same mu/sd, so the model sees
        # the identical normalisation for a quantity whether it's past (hist) or future
        # (exo) weather -- no separate exo-only scaler is fit for these. Exo columns with
        # no hist_cols counterpart (hour_sin, dow_cos, is_weekend, ...) are cyclical/binary
        # calendar features already appropriately scaled; they get mu=0, sd=1 (identity) --
        # z-scoring sin/cos pairs independently would give each an empirically-different
        # mu/sd from real (non-uniform) training data and break the sin^2+cos^2=1 circular
        # pairing the cyclical encoding exists for.
        hist_index = {c: i for i, c in enumerate(hist_cols)}
        mu_exo = np.array([mu_hist[hist_index[c]] if c in hist_index else 0.0
                            for c in exo_cols], dtype=np.float32)
        sd_exo = np.array([sd_hist[hist_index[c]] if c in hist_index else 1.0
                            for c in exo_cols], dtype=np.float32)
        out["exo_cols"] = list(exo_cols)
        out["mu_exo"] = mu_exo
        out["sd_exo"] = sd_exo
    return out


def normalise_hist(x_hist, scalers):
    """x_hist: (N, L, C) aligned to scalers['hist_cols']. Per-channel z-score."""
    return ((x_hist - scalers["mu_hist"]) / scalers["sd_hist"]).astype(np.float32)


def normalise_exo(x_exo, scalers):
    """x_exo: (N, horizon, F_exo) aligned to scalers['exo_cols']. Per-channel z-score,
    where mu_exo/sd_exo (built in fit_scalers) are the SAME stats as the matching
    hist_cols channel for solar_irrad/ambient_temp, and identity (mu=0, sd=1) for the
    cyclical/binary calendar columns that shouldn't be touched (see fit_scalers).
    Falls back to a no-op if scalers has no exo stats (checkpoints fit before this existed)."""
    if "mu_exo" not in scalers:
        return np.asarray(x_exo, dtype=np.float32)
    return ((x_exo - scalers["mu_exo"]) / scalers["sd_exo"]).astype(np.float32)


def normalise_y(y, scalers):
    """y: (N, horizon) prosumption target -> standardised with the shared scaler."""
    return ((y - scalers["mu_y"]) / scalers["sd_y"]).astype(np.float32)


def denormalise_y(y_std, scalers):
    """Predictions -> physical MW. Works for point (N,K) or quantiles (N,K,Q),
    numpy or torch (scalar affine, positive scale -> monotonicity preserved)."""
    assert scalers["sd_y"] > 0
    return y_std * scalers["sd_y"] + scalers["mu_y"]


# ============================================================================ #
# Section 5 — imputation
# ============================================================================ #
def _max_consecutive_gap(mask: pd.Series) -> int:
    """Length of the longest run of True in a boolean Series."""
    if not mask.any():
        return 0
    grp = (mask != mask.shift()).cumsum()
    return int(mask.groupby(grp).sum().max())


def reindex_and_impute(
    df: pd.DataFrame, cols, freq: str = "1h", warn_gap: int = 6
) -> pd.DataFrame:
    """
    Imputation stage — sits BEFORE fit_scalers and BEFORE make_windows.

    Why here: make_windows requires a gap-free hourly grid, and fit_scalers must
    not see NaNs. Reindexing onto a complete hourly range first SURFACES any
    missing hours as explicit NaN rows; then we fill them.

    Method: *time* (linear) interpolation on the continuous `cols` — a LOCAL,
    leak-safe method. It uses only neighbouring observations, no global or
    train-derived statistic, so it cannot leak information across the
    train/val/test split. Edge NaNs are forward/back-filled. Long gaps ARE filled
    but WARNED about (fabricating many hours of a volatile series is dubious;
    prefer to drop or investigate those).

    Calendar features are NOT imputed here — they are added afterwards in
    build_features and computed from the (now complete) index, so never NaN.
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz=df.index.tz)
    df = df.reindex(full_idx)

    miss = df[cols].isna()
    total = int(miss.values.sum())
    if total:
        print(f"[impute] {total} missing values across {list(cols)} after reindex")
        for c in cols:
            n = int(miss[c].sum())
            if n:
                g = _max_consecutive_gap(miss[c])
                flag = "   <-- LONG GAP (review)" if g > warn_gap else ""
                print(f"[impute]   {c}: {n} missing, longest run {g}h{flag}")
        df[cols] = (
            df[cols].interpolate(method="time", limit_direction="both").ffill().bfill()
        )
    else:
        print("[impute] no missing values after reindex — grid already complete")

    remaining = int(df[cols].isna().sum().sum())
    assert remaining == 0, f"{remaining} NaNs remain after imputation"
    return df


# ============================================================================ #
# Section 6 — model & loss
# ============================================================================ #
class Baseline_Forecaster(nn.Module):
    def __init__(
        self,
        n_hist_features,
        n_exo_features,
        n_quantiles,
        gru_hidden_size, # 
        dense_hidden_width,
        dropout,
        horizon = 24,
        n_gru_layers = 2,
        eps = 1e-4,
    ):
        """
        n_hist_features : channels in the history window (e.g. consumption +
                          panel temp + outside temp + irradiance = 4).
        n_exo_features  : per-horizon calendar features (cyclical hour, is_weekend...).
        eps: epsilon, tiny constant added between quantiles to guarantee strict monotonicity of CDF
        """
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.eps = eps

        # ---- Encoder: two stacked GRUs; take the top layer's final state ----
        # num_layers=2 stacks GRU-1 -> GRU-2 (GRU-1 feeds its full sequence to
        # GRU-2 internally).
        self.encoder = nn.GRU(
            input_size=n_hist_features,
            hidden_size=gru_hidden_size,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )

        # ---- Decoder: shared dense head, applied at each horizon step for each batch ----
        # Applying these linears over a (B, k, H + n_exo) tensor shares the weights
        # across the all k = 24 steps by construction.
        self.decoder = nn.Sequential(
            # takes (B, K, H + n_exo) tensor
            nn.Linear(gru_hidden_size + n_exo_features, dense_hidden_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden_width, dense_hidden_width),
            nn.ReLU(),
            nn.Linear(dense_hidden_width, n_quantiles),
            # treats whatever input as a
        )

    def forward(self, x_hist, x_exo):
        """
        x_hist : (Batch, L, n_hist_features)      history (realised consumption + weather)
        x_exo  : (Batch, horizon, n_exo_features) per-horizon (time-respective) calendar
        returns: (Batch, horizon, n_quantiles)    monotone quantiles (increasing in Q)
        """

        ## Step 1: Deal with the historic data.
        # Encoder -> single history encoding (final hidden state of the top GRU)
        _, h_n = self.encoder(x_hist)  # h_n: (n_layers, B, H)
        encoding = h_n[-1]  # (B, H) # takes the final GRU layer's hidden state.

        # Reuse the encoding across all horizon steps; concat per-step calendar
        ## Step 2: Introduce calendar data, format so that it is concatenated per time-step.

        H = encoding.shape[-1]  # the length of the hidden state vector of the GRU
        B, K, _ = x_exo.shape  # K is window size (e.g. 24 hours of predictions)

        enc = encoding.unsqueeze(1).expand(B, K, H)
        # what this does, in sequence.
        # encoding: the (B,H) shaped tensor of the final GRU's hidden state. equiv to enc = encoding[:, None, :]
        # unsqueeze: inserts new size-1 dimension into the dimension at position 1, so (B,1,H)
        # expand: essentially duplicates the existing tensor along the unsqueezed dimension (doesn't fill new memory)
        # left with (B,K,H) dimensions, where every tensor for [:,n,:] is identical

        dec_in = torch.cat([enc, x_exo], dim=-1)  # (B, K, H + n_exo)
        # This appends the calendar data to the final dimension of the encoding tensor.
        # Since x_exo has dimensions (B,K), it fits in neatly across tensor.

        # Shared dense head at each step -> raw quantile values
        raw = self.decoder(dec_in)  # (B, K, Q)

        # Monotone construction: base + cumulative (softplus + eps) increments
        base = raw[..., 0:1]  # lowest quantile (0.05)
        inc = F.softplus(raw[..., 1:]) + self.eps  # strictly positive
        quantiles = torch.cat(
            [base, base + torch.cumsum(inc, dim=-1)], dim=-1
        )  # (B, K, Q)
        return quantiles  # returns all estimated quantiles for B batches over K time periods.


def pinball_loss(y_true, y_pred, levels):
    """
    y_true : (B, K)      standardised targets
    y_pred : (B, K, Q)   monotone quantile forecasts
    levels : (Q,) MUST BE TENSOR (if numpy array, breaks the code!)
    Summed pinball over levels approximates CRPS, aligning the training objective
    with the reported metric.
    """
    assert isinstance(levels, torch.Tensor), (
        f"Expected x to be a torch.Tensor, but got {type(levels).__name__}"
    )

    e = y_true.unsqueeze(-1) - y_pred  # (B, K, Q)
    # creates transformed tensor of dimensions (B,K,1) from (B,K)
    # then subtracts y_pred from it, which is of dimensions (B,K,Q)
    # this automatically broadcasts - y_pred for each quantile Q
    # results in a (B,K,Q) tensor equal to y_true - y_pred for every (B,K,Q) of y_pred

    q = levels.view(1, 1, -1)
    # the -1 means that the size of dimension is inferred from other dimensions
    # in this case, since levels is of (Q,) dimensions, q is of the dimensions (1,1,Q)
    # this means q lines up with the error tensor of (B,K,Q)
    return torch.maximum(q * e, (q - 1.0) * e).sum(dim = (1,2,)).mean()  # pinball loss, summed over all quantiles and averaged over the batch