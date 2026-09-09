# Convex surrogate formulations for LDR robustness-awareness

Record of every objective-function variant tried for making the LDR recourse policy
(`D_ch`, `D_dis`) aware of forecast uncertainty *without* the hard, polyhedral/ellipsoidal
robust constraints (`mu_p`/`mu_m` box-duality, SAA scenario constraints, SOC-norm chance
constraints) that were found to cost 15-90x the non-robust baseline's solve/differentiation
time (see the GPU/CPU feasibility investigation in `gpu_feasibility_colab.ipynb` and the
session history). All timings below: batch=1, ECOS, CPU (WSL2), forward+backward through
`cvxpylayers`. Ratios are against the non-robust baseline's measured 1.65s; four-corner
estimates scale the real, completed baseline training time (~15-16hr total) by that ratio.

**Shared notation across all variants:**
- `T=24` (hours), `p_ch_hat`/`p_dis_hat` (T,) nominal charge/discharge, `D_ch`/`D_dis` (T,T)
  LDR recourse matrices (lower-triangular via `upper_tri(.)==0`), `p_da_rel` (T,) free bid
  offset, `s_hat` (T,) nominal SOC trajectory, `G = Δt·cumsum(η_ch D_ch - D_dis/η_dis)` (T,T)
  cumulative recourse SOC-gain matrix, `R = I + D_ch - D_dis` (T,T) net recourse matrix.
- `pl_hat`, `Sigma_xi_chol` (T,T, s.t. `Sigma_xi_chol @ Sigma_xi_chol.T ≈ Cov(ξ)`), `h_plus`,
  `h_minus` (T,) are the forecaster's outputs / quantile-derived box half-widths.
- `GAMMA` (1e-6) is the existing Tikhonov coefficient already in `dispatch_layer.py`.

---

## 0. Non-robust baseline (reference point, already trained and working)

```
min  Δt²(‖imb_det‖² + ‖R·Sigma_xi_chol‖²_F)                         (tracking, k>0)
   + GAMMA·(‖p_ch_hat‖² + ‖p_dis_hat‖² + ‖D_ch‖²_F + ‖D_dis‖²_F + ‖p_da_rel‖²)   (Tikhonov)
s.t. 0 ≤ p_ch_hat ≤ C_ch,  0 ≤ p_dis_hat ≤ C_dis,  0 ≤ s_hat ≤ B_max     (nominal bounds only)
     s_hat[T-1] = SOC0,  G[T-1,:] = 0,  upper_tri(D_ch) = upper_tri(D_dis) = 0
```

`D_ch`/`D_dis` are genuine linear decision rules here, but nothing in the objective or
constraints relates them to `h_plus`/`h_minus` at all — they're shaped only by tracking
accuracy and the uniform Tikhonov penalty.

**DPP:** valid. **Timing:** forward 0.33s, backward 1.32s, **total 1.65s** (1.0x).
**4-corner estimate:** ~15-16hr (actual, completed).

---

## 1. Pure vector midpoint anchor

Add one term to the baseline, using a single representative "centre" deviation vector
`h_mid = (h_plus - h_minus)/2` instead of the full covariance:

```
+ GAMMA_MID·(‖D_ch·h_mid‖² + ‖D_dis·h_mid‖²)
```

`D_ch @ h_mid` is a `(T,T)@(T,)` matrix-*vector* product — cheap, no per-row epigraph
structure, unlike anything using `Sigma_xi_chol` (a `(T,T)@(T,T)` matrix-matrix product).

**DPP:** valid. **Timing:** forward 0.39s, backward 1.66s, **total 2.05s** (1.24x).
**4-corner estimate:** ~19hr.

**Note:** cheapest uncertainty-aware option found; uses only a single representative
direction/magnitude per hour, not the full correlation structure `Sigma_xi_chol` carries.

---

## 2. Sigma-weighted recourse-smoothness anchor alone ("term 3")

```
+ GAMMA_REC·(‖D_ch·Sigma_xi_chol‖²_F + ‖D_dis·Sigma_xi_chol‖²_F)
```

Same role as (1) but using the full covariance-weighted response magnitude instead of a
single representative vector.

**DPP:** valid. **Timing:** forward 2.5s, backward 3.0s, **total 5.5s** (3.3x).
**4-corner estimate:** ~50-53hr.

---

## 3. First combined surrogate (economic/tracking + naive centre penalty + term 2 + Tikhonov)

```
+ GAMMA_TUNE · Σ_t (h_t+ + h_t-)·(ŝ_t - s_target)²          [s_target = 0.5·B_max]
+ GAMMA_REC  · (‖D_ch·Sigma_xi_chol‖²_F + ‖D_dis·Sigma_xi_chol‖²_F)     [term 2, unchanged]
```

**DPP:** valid (confirmed: `PARAMETER × cp.square(VARIABLE - CONSTANT)` canonicalizes
cleanly). **Timing:** forward 2.44s, backward 3.54s, **total 5.99s** (3.6x).

**Flaw (caught before committing):** the centre-penalty term depends only on `ŝ_t`, the
*nominal* schedule — `D_ch`/`D_dis` define deviations *around* `ŝ_t` and never appear in this
term at all. The solver could perfectly centre the nominal trajectory while `D_ch`/`D_dis`
still push the *realized* trajectory `ŝ_t + G_t·ξ` straight through a boundary the instant an
error occurs. Term 2 was doing the actual recourse-shaping work here, not this term.

Also worth noting: with this project's default parameters, `s_target = 0.5·B_max = 2.0`
happens to equal `SOC0 = 2.0` exactly, so the centre-penalty and the hard terminal
constraint (`s_hat[T-1] = SOC0`) are consistent here — but that's a property of the current
defaults, not something the formulation enforces structurally.

---

## 4. Recourse-aware fix — expected squared deviation decomposition

Penalizing `E_ξ[(s_t(ξ) - s_target)²]` where `s_t(ξ) = ŝ_t + G_t·ξ` decomposes exactly
(since `E[ξ]=0`, cross-term vanishes) into bias² + variance:

```
E_ξ[(ŝ_t + G_t·ξ - s_target)²] = (ŝ_t - s_target)² + ‖G_t·Sigma_xi_chol‖²
```

giving, per-hour and uncertainty-weighted:

```
+ GAMMA_TUNE · Σ_t w_t·[(ŝ_t - s_target)² + ‖G_t·Sigma_xi_chol‖²]        [w_t = h_t+ + h_t-]
```

**First attempt at this was NOT DPP-valid** — `w·sum(square(G@Sigma_xi_chol), axis=1)`
multiplies two *independent* parameters (`w` and `Sigma_xi_chol`) together. `G@Sigma_xi_chol`
already depends on `Sigma_xi_chol`; multiplying the result by `w` makes the term depend on
the *product* of two parameters, which DPP can never represent as affine-in-parameters,
regardless of how it's rearranged. `cvxpy` reports this indirectly as
`ParameterError: Problem contains unspecified parameters` when `is_dcp(dpp=True)` is `False`.
This is a different, more fundamental violation than "parameter × nonlinear-function-of-pure-
variables" (which is fine, see terms above) — it's "parameter × [expression already
containing a second parameter]", which is never DPP-representable no matter the rewrite.

**Fix:** precompute the weighting outside CVXPY, in torch, where both `w` and
`Sigma_xi_chol` already live as plain tensors before the `CvxpyLayer` call:

```python
weighted_Sigma = torch.sqrt(w).unsqueeze(-1) * Sigma_xi_chol   # (T,1) * (T,T) broadcast
```

passed in as a single combined `cp.Parameter`, reducing the CVXPY-side term to
`cp.sum_squares(G @ weighted_Sigma)` — single-parameter, matching the already-valid
`R @ Sigma_xi_chol` pattern exactly.

```
+ GAMMA_TUNE · [Σ_t h_t+ ·(ŝ_t - s_target)²  +  ‖G·weighted_Sigma‖²_F]
```

(mean term used `h_plus` alone in the tested version, purely to avoid threading an extra
parameter through the test; restoring `w = h_plus+h_minus` there is trivial and DPP-neutral.)

**DPP:** valid. **Timing:** forward 2.46s, backward 3.02s, **total 5.47s** (3.3x).
**4-corner estimate:** ~50-53hr.

**Assessment: best balance found.** Mathematically complete (ties both mean *and* variance
of the *realized* SOC to the actual recourse matrices, not just the nominal schedule),
cheapest of the fully recourse-aware options, no redundant terms.

---

## 5. H-centre variant — asymmetry-aware reference point

Same expected-squared-deviation logic as (4), but centred on the forecast's own asymmetry
(`h_center = (h_plus - h_minus)/2`, the quantile box's own midpoint) rather than an arbitrary
physical target like `0.5·B_max`. Since `s(ξ) - s_center = G·(ξ - ξ_center)` (the `ŝ` terms
cancel exactly), this decomposes into:

```
E_ξ[(s(ξ) - s_center)²] = ‖G·Sigma_xi_chol‖² + ‖G·h_center‖²
```

Both terms use only *one* parameter inside their own `sum_squares` (`Sigma_xi_chol` in one,
`h_center` in the other, combined by addition, not multiplication) — this avoids the DPP
violation from (4)'s first attempt without needing the torch-side precompute workaround.

### 5a. Full version (SOC-level + power-level, 6 total quadratic terms)

```
+ GAMMA_TUNE · [‖G·h_center‖² + ‖G·Sigma_xi_chol‖²_F
               + ‖D_ch·h_center‖² + ‖D_dis·h_center‖²
               + ‖D_ch·Sigma_xi_chol‖²_F + ‖D_dis·Sigma_xi_chol‖²_F]
```

**DPP:** valid. **Timing:** forward 13.29s, backward 9.99s, **total 23.28s** (14.1x).
**4-corner estimate:** ~211-226hr (~9 days).

**Why so expensive:** stacks four separate `sum_squares(matrix @ Sigma_xi_chol)`-style
matrix-*matrix* terms (existing `R@Sigma_xi_chol` + new `G@Sigma_xi_chol` +
`D_ch@Sigma_xi_chol` + `D_dis@Sigma_xi_chol`). Cost compounds *super-additively* when
stacking multiple aggregate epigraph terms — a pattern observed repeatedly this session
(see also §3's combination and the earlier "three quadratic penalties" test), not simple
per-term addition. `D_ch`/`D_dis`-level terms are also conceptually redundant with the
`G`-level ones here, since `G` is itself a `cumsum` composition of `D_ch`/`D_dis` — penalizing
`G`'s variance already indirectly discourages large, volatile recourse.

### 5b. Trimmed version (SOC-level only)

```
+ GAMMA_TUNE · [‖G·h_center‖² + ‖G·Sigma_xi_chol‖²_F]
```

**DPP:** valid. **Timing:** forward 3.89s, backward 4.39s, **total 8.28s** (5.0x).
**4-corner estimate:** ~75-80hr.

**Note:** more expensive than (4) despite using a comparable number of matrix-matrix terms —
likely because `G`'s extra `cumsum` composition (vs. referencing `D_ch`/`D_dis` directly)
adds canonicalization overhead. The conceptual advantage of `h_center` over a fixed physical
target (no arbitrary midpoint, ties directly to forecast asymmetry) currently costs more than
it buys relative to (4).

---

## 6. Reserve-headroom penalties — penalize proximity to physical bounds directly

A different family from 1-5: instead of shaping `D_ch`/`D_dis` via a bias/variance
decomposition against an uncertainty anchor, penalize the policy for planning too close to
its physical limits in the first place, so the optimizer voluntarily reserves headroom.

### 6a. Nominal-proximity penalty (no scenarios, no new parameters)

```
+ RHO · [‖pos(p_ch_hat - (C_ch - margin))‖² + ‖pos(p_dis_hat - (C_dis - margin))‖²
        + ‖pos(s_hat - (B_max - margin))‖² + ‖pos(margin - s_hat)‖²]
```

`C_ch`, `C_dis`, `B_max`, `margin` are plain constants, not `cp.Parameter`s — this term
involves **zero parameters**, so DPP is trivially satisfied (nothing to violate). Only
touches the *nominal* schedule (`p_ch_hat`, `p_dis_hat`, `s_hat`), never `D_ch`/`D_dis` or
any uncertainty quantity (`Sigma_xi_chol`, `xi_samples`, `h_plus`/`h_minus`) — it reserves a
fixed, uniform margin regardless of how much local uncertainty actually exists at a given
hour/day. Cheap and easy to justify as an auxiliary training-stability term (see the
saturation-gradient-starvation discussion in `pivot_context.md`-adjacent session notes:
`p_ch_hat`/`s_hat` are always live/differentiable, unlike the *realized* recourse action
which goes through `realised_breakdown`'s hard clip), but not "uncertainty-aware" on its own
— margin is the same on a calm winter day as a volatile summer midday.

**DPP:** valid. **Timing:** forward 0.46s, backward 2.13s, **total 2.59s** (1.6x).

**Possible refinement (not yet tested):** make `margin` itself a parameter derived from local
uncertainty (e.g. `margin_t = c·sqrt(diag(Sigma_xi_chol))[t]`, or reuse `h_plus`/`h_minus`
directly) instead of a fixed constant. `p_ch_hat - (C_ch - margin_t)` is still
variable+constant+parameter, affine in everything, no parameter multiplying a parameter — so
this stays exactly as DPP-clean at essentially the same cost, while closing the "not actually
uncertainty-aware" gap.

### 6b. Scenario-based recourse-headroom penalty (SAA-style soft penalty)

```
+ (RHO/N) · Σ over {charge, discharge, SOC} × {upper, lower} of
      ‖pos(p_ch_hat[:,None] + D_ch@xi_samples.T - C_ch)‖²   (and the other 5 symmetric terms)
```

Six terms, one per physical constraint family — maps one-to-one onto the six hard robust
constraints in `dispatch_layer_robust.py`'s `_robustify_vec` calls, evaluated as a soft
per-scenario penalty over the actual `xi_samples` draws instead of a hard worst-case-over-
the-box constraint. Conceptually the most faithful surrogate tried: unlike 1-5 (which use a
closed-form bias²+variance decomposition, implicitly only using the first two moments of
ξ), this uses the real empirical scenario draws, so it can respond to skew/tail-shape the
copula encodes that a Gaussian-moment surrogate structurally can't. DPP-valid by the same
"variable-affine-expression @ one parameter" pattern as `R@Sigma_xi_chol`.

**Implementation trap (real, worth flagging loudly):** writing this as
`cp.sum(cp.square(cp.pos(X)) + ...)` (natural, readable syntax — sum a matrix of scores) is
mathematically identical to `cp.sum_squares(cp.pos(X)) + ...` but canonicalizes completely
differently in CVXPY. `cp.sum(cp.square(X))` lowers to **one 3-dimensional SOC cone per
scalar element** of `X` (~9,200 separate cones for the six `(T,N)=(24,64)` matrices here);
`cp.sum_squares(X)` lowers to **one aggregate cone per whole-matrix term** (7 cones total,
confirmed via `problem.get_problem_data()`). Same math, catastrophically different cost —
this is the per-row-atom finding from earlier sections taken to its per-*element* extreme.
The `sum(square())` version OOM-killed the test process at 3+GB RSS before completing a
single rep; confirmed via kernel log (`dmesg`), not a guess. **Always use `sum_squares(pos(X))`
per term, never `sum(square(pos(X)) + ...)`, for any matrix-valued penalty in this codebase.**

**DPP:** valid (both versions — the DPP violation risk in this family is elsewhere, in
parameter×parameter interactions; the `sum`/`sum_squares` choice is a pure canonicalization
question, DPP-orthogonal). **Timing (`sum_squares` version, the only one that completes):**
forward 19.53s, backward 150.35s, **total 169.9s (~107x baseline)**.

**Assessment: not viable for training, even correctly implemented.** Pre-test estimate was
"north of `sigma_anchor`, nowhere near hard-constraint territory (15-90x)" — wrong by a wide
margin. Six separate large `(T,N)`-shaped cones (dimension 1538 each) compound far worse than
linearly, landing *above* the hard-constraint formulations this whole surrogate program exists
to avoid. Conceptually the best-motivated option in this family; computationally a dead end at
this scale. Would only be worth revisiting at a much smaller `N` (fewer scenarios) if the
conceptual advantage (capturing real distributional shape) is judged worth chasing further.

---

## Summary

| # | Formulation | DPP | forward | backward | total | ×baseline | 4-corner est. |
|---|---|---|---|---|---|---|---|
| 0 | non-robust baseline | ✓ | 0.33s | 1.32s | 1.65s | 1.0x | ~15-16hr (actual) |
| 1 | vector midpoint anchor | ✓ | 0.39s | 1.66s | 2.05s | 1.24x | ~19hr |
| 2 | Sigma-weighted anchor alone | ✓ | 2.5s | 3.0s | 5.5s | 3.3x | ~50-53hr |
| 3 | naive centre + term 2 (flawed) | ✓ | 2.44s | 3.54s | 5.99s | 3.6x | — (superseded) |
| 4 | **recourse-aware fix (recommended)** | ✓ | 2.46s | 3.02s | **5.47s** | 3.3x | ~50-53hr |
| 5a | h-centre, full (SOC + power level) | ✓ | 13.29s | 9.99s | 23.28s | 14.1x | ~211-226hr |
| 5b | h-centre, trimmed (SOC level only) | ✓ | 3.89s | 4.39s | 8.28s | 5.0x | ~75-80hr |
| 6a | nominal-proximity penalty (auxiliary, not uncertainty-aware) | ✓ | 0.46s | 2.13s | 2.59s | 1.6x | ~24hr |
| 6b | scenario-based recourse headroom (SAA-style, 6 terms) | ✓ | 19.53s | 150.35s | 169.9s | ~107x | not viable |

**Current recommendation: #4.** Cheapest option that is both mathematically complete (ties
recourse matrices to uncertainty via the realized, not just nominal, SOC trajectory) and free
of redundant terms. #1 is cheaper still but only shapes `D_ch`/`D_dis` via a single
representative direction, not the full covariance. #5's asymmetry-aware centring is
conceptually appealing but not yet cost-competitive with #4 in any tested variant.

**Still open, not yet tested:** `h_center` applied directly to `D_ch`/`D_dis` (matching #1's
cheap matrix-vector pattern) instead of to `G`, retaining the asymmetry-aware reference point
without `G`'s extra `cumsum`-composition overhead. Also open: calibrating `GAMMA_MID` /
`GAMMA_TUNE` / `GAMMA_REC` against the true hard-constraint model's saturation and recourse
behavior (see the staged methodology: weather retrain → `h` sweep → penalty-strength sweep →
full DFL retrain), and confirming these ratios hold at the real training batch size (16, not
the 1 used for all timings here) and on real (not synthetic) data.

### Reproducibility check

The original numbers above were single-shot measurements on a laptop (i5-8365U, 15W U-series
chip — throttles/loses turbo easily under sustained load); worth checking they weren't an
artifact of transient background load. Rebuilt all variants in one script
(`rerun_surrogate_timings.py`, not committed — scratch) and ran each 3x (after 1 discarded
warm-up rep) back-to-back:

| variant | original total | rerun mean±std (3 reps) | ratio (then → now) |
|---|---|---|---|
| baseline | 1.65s | 2.42±0.03s | 1.0x → 1.0x |
| mid_anchor | 2.05s | 2.97±0.06s | 1.24x → 1.23x |
| sigma_anchor | 5.5s | 8.39±0.74s | 3.3x → 3.5x |
| flawed_combo (§3) | 5.99s | 8.14±0.26s | 3.6x → 3.4x |
| recourse_fix (§4) | 5.47s | 7.95±0.15s | 3.3x → 3.3x |
| hcentre_full (§5a) | 23.28s | 23.50±0.27s | 14.1x → 9.7x |
| hcentre_trim (§5b) | 8.28s | 12.44±0.64s | 5.0x → 5.1x |

Absolute times ran ~1.2-1.7x higher across the board on the rerun (`load average` climbed
1.52→2.19 over the ~4.5min run; several other processes were visibly competing for the same
8 logical threads throughout; WSL2 exposes no real thermal sensors so throttling itself can't
be confirmed directly, but the CPU was reading flat base-clock (1896MHz, no turbo) rather than
its ~4.1GHz boost afterward). **Conclusion: the absolute numbers have real machine-load noise
on top of them (~±15-20% run to run), but the relative cost ordering between formulations is
reproducible** — mid_anchor cheapest (~1.2x), sigma_anchor/flawed_combo/recourse_fix cluster
tightly (~3.3-3.5x) regardless of run, hcentre_trim ~5x, hcentre_full clearly worst (~10-14x,
and its *absolute* time was nearly identical between runs, 23.28s vs 23.50s — the drop in its
*ratio* is entirely because the baseline got noisier, not because hcentre_full got cheaper).
This confirms the formulation choice (recommendation: #4) isn't resting on a load artifact.
