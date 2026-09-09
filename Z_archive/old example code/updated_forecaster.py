"""
option_c_forecaster.py
======================

The Option C encoder-decoder quantile forecaster, matching the settled
architecture diagram.

  Encoder  : two stacked GRUs over the history window (nn.GRU num_layers=2 —
             GRU-1 passes its full output sequence to GRU-2, the vom Scheidt
             stacking usage). The FINAL hidden state of the top GRU is a single
             "history encoding" summarising the week of realised consumption and
             weather up to gate closure.

  Decoder  : the history encoding is REUSED across all `horizon` steps. At each
             step it is concatenated with that step's time-respective calendar
             features and passed through a SHARED dense head (the same weights
             applied at every horizon step — inherent to applying nn.Linear over
             the horizon dimension). Cross-horizon dependence is deliberately NOT
             modelled here; it is supplied downstream by the frozen copula.

  Head     : monotone quantiles by construction — base at the lowest level plus
             cumulative softplus increments (+eps for STRICT monotonicity, which
             the copula's inverse-CDF interpolation needs). Cite Brando et al.
             (2022) / Cannon (2018) for the non-crossing construction.

Trained in standardised space on pinball loss (this baseline is also the
warm-start init for the DFL phase); inverse-transform the quantiles to physical
units downstream (affine, positive scale, preserves monotonicity).

Dependencies: torch, numpy. (Not executed in the authoring environment — run in
an env with torch installed; the shape flow and monotone construction were
validated separately in NumPy.)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 19 evenly-spaced levels, base at 0.05 — copula-adequate CDF resolution.
QUANTILE_LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)


class OptionCForecaster(nn.Module):
    def __init__(self, n_hist_features, n_exo_features, horizon=24,
                 n_quantiles=len(QUANTILE_LEVELS), gru_hidden=32, n_gru_layers=2,
                 dense_hidden=64, dropout=0.1, eps=1e-4):
        """
        n_hist_features : channels in the history window (e.g. consumption +
                          panel temp + outside temp + irradiance = 4).
        n_exo_features  : per-horizon calendar features (cyclical hour, is_weekend...).
        """
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.eps = eps

        # ---- Encoder: two stacked GRUs; take the top layer's final state ----
        # num_layers=2 stacks GRU-1 -> GRU-2 (GRU-1 feeds its full sequence to
        # GRU-2 internally, i.e. return_sequences=True between the layers).
        self.encoder = nn.GRU(
            input_size=n_hist_features, hidden_size=gru_hidden,
            num_layers=n_gru_layers, batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )

        # ---- Decoder: shared dense head, applied at each horizon step ----
        # Applying these Linears over a (B, horizon, .) tensor shares the weights
        # across the 24 steps by construction.
        self.decoder = nn.Sequential(
            nn.Linear(gru_hidden + n_exo_features, dense_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, dense_hidden), nn.ReLU(),
            nn.Linear(dense_hidden, n_quantiles),
        )

    def forward(self, x_hist, x_exo):
        """
        x_hist : (B, L, n_hist_features)      history (realised consumption + weather)
        x_exo  : (B, horizon, n_exo_features) per-horizon (time-respective) calendar
        returns: (B, horizon, n_quantiles)    monotone quantiles (increasing in Q)
        """
        # Encoder -> single history encoding (final hidden state of the top GRU)
        _, h_n = self.encoder(x_hist)                 # h_n: (n_layers, B, H)
        encoding = h_n[-1]                            # (B, H)

        # Reuse the encoding across all horizon steps; concat per-step calendar
        B, K, _ = x_exo.shape
        enc = encoding.unsqueeze(1).expand(B, K, encoding.shape[-1])  # (B, K, H)
        dec_in = torch.cat([enc, x_exo], dim=-1)      # (B, K, H + n_exo)

        # Shared dense head at each step -> raw quantile values
        raw = self.decoder(dec_in)                    # (B, K, Q)

        # Monotone construction: base + cumulative (softplus + eps) increments
        base = raw[..., 0:1]                          # lowest quantile (0.05)
        inc = F.softplus(raw[..., 1:]) + self.eps     # strictly positive
        quantiles = torch.cat(
            [base, base + torch.cumsum(inc, dim=-1)], dim=-1)  # (B, K, Q)
        return quantiles


def pinball_loss(y_true, y_pred, levels):
    """
    y_true : (B, K)      standardised targets
    y_pred : (B, K, Q)   monotone quantile forecasts
    levels : (Q,) tensor
    Summed pinball over levels approximates CRPS, aligning the training objective
    with the reported metric.
    """
    e = y_true.unsqueeze(-1) - y_pred                 # (B, K, Q)
    q = levels.view(1, 1, -1)
    return torch.maximum(q * e, (q - 1.0) * e).mean()


if __name__ == "__main__":
    # ---- smoke test of shapes + monotonicity (run in an env with torch) ----
    torch.manual_seed(0)
    B, L, K = 16, 168, 24
    
    n_hist, n_exo = 4, 5                              # consumption+3 weather; calendar
    model = OptionCForecaster(n_hist_features=n_hist, n_exo_features=n_exo, horizon=K)

    x_hist = torch.randn(B, L, n_hist)
    x_exo = torch.randn(B, K, n_exo)
    q = model(x_hist, x_exo)                          # (B, K, Q)
    print("output:", tuple(q.shape))
    print("monotone (all increments > 0):", bool((torch.diff(q, dim=-1) > 0).all()))

    levels = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32)
    y = torch.randn(B, K)
    loss = pinball_loss(y, q, levels)
    loss.backward()
    g_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    print(f"pinball: {loss.item():.4f} | gradients finite & flowing: {g_ok}")
    print("smoke test OK")
