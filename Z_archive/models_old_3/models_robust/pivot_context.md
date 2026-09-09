# Context: why the plan changed after "train the robust model" was attempted

Briefing for picking up this project. Assumes you already know the setup up to the point
where training the original robust corners (single/dual price model × k=0/k=1, hard-constraint
LDR robust formulation) was about to be attempted. This covers everything that happened once
that attempt actually started.

## 1. Why we strayed from the original plan

The original plan was straightforward DFL training: 4 corners (single/dual price model ×
k=0 deterministic / k=1 robust), where k=1's dispatch layer used a **hard uncertainty-set
constraint** (box constraint on LDR responses, dualized via `mu_p`/`mu_m` nonneg LP duals),
differentiated end-to-end via `cvxpylayers`/`diffcp` for gradient-based training.

This turned out to be computationally intractable, on both CPU and GPU, for reasons specific
to differentiating *through* a hard robust reformulation:

- **CPU**: every hard-constraint variant tested (box/`mu_p`-`mu_m` duality, SAA scenario
  constraints, ellipsoidal/chance-constraint SOC formulations) cost roughly **15-90x** the
  non-robust baseline's forward+backward time. At real training scale (batch=16, ~23
  batches/epoch, up to 50 epochs, 4 corners), this projected to **~50 hours to ~9 days**
  depending on variant — not viable within a dissertation timeline.
- **GPU** (the natural next lever): traced and confirmed the root cause of a hard GPU
  failure (`"A linear solver returned non-finite output"` at every batch size). `diffqcp`
  (the JAX-based differentiation engine behind `cvxpylayers`' `solver=cp.CUCLARABEL` path)
  defaults its backward solve to `"jax-lu"`, a dense solve that assumes the KKT operator is
  nonsingular. It isn't: `mu_p`/`mu_m` are LP duals that sit exactly at zero over a large flat
  subspace at the optimum — a genuine degenerate/singular KKT block at a nonneg-cone
  boundary, independent of data, precision, batching, or CUDA setup (all ruled out
  individually). Patching `diffqcp`/`cvxpylayers` internals to force the more robust
  `"jax-lsmr"` solve was ruled out as a project constraint (no patching third-party
  library internals).
- Regularizing `mu_p`/`mu_m` directly (`gamma_mu * sum_squares(...)`) was tried as a
  problem-level fix — made CPU backward **1.4-1.9x slower**, not faster, and doesn't
  principledly fix the GPU singularity either.
- An ellipsoidal reformulation (`Sigma_xi_chol`-weighted SOC constraint, **no** auxiliary
  dual variables at all — removes the degenerate block at its root) was built and is ready
  to test (`gpu_feasibility_colab.ipynb`), but its GPU run has never actually been executed.
  Even on CPU, its cost (~25s/sample batch=1) is barely better than the original
  `mu_p`/`mu_m` version (~23-29s) — so even if the GPU blocker is fixed, this path is still
  fundamentally expensive; conditioning affects *whether it runs*, not fundamentally *how
  fast*.

**Conclusion**: no hard-constraint-based robust reformulation is tractable for gradient-based
DFL training within a realistic budget, regardless of the specific technique. This is a
structural property of combining LP/SOC-dual-based robust optimization with automatic
differentiation at DFL training scale — not an artifact of any one implementation choice.

## 2. What we're doing instead: convex surrogate objectives

Rather than a hard uncertainty-set constraint (with duals, degeneracy, and the cost that
comes with it), inject uncertainty-awareness into the LDR recourse matrices (`D_ch`, `D_dis`)
via **soft quadratic penalty terms in the objective** — no auxiliary variables, no dual
degeneracy, fully DPP-compliant, and empirically **~1.2-5x** the non-robust baseline's cost
(vs. 15-90x for hard constraints).

Several formulations were derived, tested for DPP-validity, and timed — full writeup with
exact math and empirical results in `convex_surrogate_formulations.md` (same directory).
Current top recommendation: the **"recourse-aware surrogate"** (~3.3x baseline), which
decomposes `E_ξ[(realized SOC - target)²]` into bias² + variance and ties both terms to the
actual `D_ch`/`D_dis` recourse matrices via `Sigma_xi_chol` and `h_plus`/`h_minus`
(quantile-derived box half-widths from the forecaster). One candidate ("h-centre") uses an
asymmetry-aware reference point instead of a fixed physical SOC target but costs more (~5-14x)
without a clear accuracy benefit yet established.

None of the penalty-strength values (`GAMMA_TUNE`/`GAMMA_MID`/`GAMMA_REC`) are calibrated —
all currently placeholders (`1e-2`) used only to test cost/DPP-validity.

## 3. Why this approach, specifically

- It's the only uncertainty-aware option found to survive contact with DFL's actual cost
  structure: the dominant driver of cost in this `cvxpylayers`/`diffcp` pipeline is the count
  of aggregate nonlinear (epigraph-inducing) atoms in the objective/constraints, not raw
  variable count — soft penalties add few/cheap aggregate atoms; hard constraints add many
  expensive ones plus dual variables.
- Since it has **no** uncertainty-set constraints or auxiliary duals at all, it should avoid
  the `jax-lu` singular-operator failure that blocked the hard-constraint GPU path entirely —
  plausible but **not yet tested on real GPU hardware**.
- A "different model"'s claim of near-instant GPU/CPU numbers for this exact surrogate
  approach (2-5s/batch on CPU, sub-second on GPU) was checked and found badly wrong — both
  internally arithmetically inconsistent, and directly contradicted by real measurement:
  batch=32 CPU costs **44-132s/batch** depending on formulation (measured directly on this
  hardware), consistent with real completed-training logs for the existing (pre-surrogate)
  k=1 corners (~29s/batch at batch=16). No real GPU number exists yet for any surrogate
  formulation — would require actually running the Colab harness.
- Considered and rejected: learning the penalty strength (`GAMMA`) end-to-end via DFL itself
  instead of manual calibration. Mechanically feasible (same DPP fix pattern already used for
  the recourse-aware surrogate), but methodologically unsound as a naive drop-in: if `GAMMA`
  is optimized against the same training loss the penalty is meant to trade off against,
  gradient descent has every incentive to collapse it toward zero (the penalty exists
  specifically to sacrifice in-sample cost for robustness — same failure mode as learning your
  own L2/KL regularization weight by minimizing the loss it regularizes). Would need a
  genuinely different, held-out/bilevel signal to avoid collapse — parked as future work, not
  pursued now.

## 4. Agreed staged methodology (mostly not yet executed)

1. Retrain the baseline forecaster with weather features (`solar_irrad`, `ambient_temp`)
   added to `EXO_COLS`, to improve forecast accuracy — `forecasting.py` is the single source
   of truth for `EXO_COLS`; downstream scripts (`train_corner.py`, `evaluate.py`,
   `h_selection_sweep.py`, `dfl_train.ipynb`) still have local hardcoded copies pending
   conversion to import from it. **Not yet implemented.**
2. **h-sweep**: find the quantile-box half-widths (`h_plus`/`h_minus`) with the best
   saturation-vs-recourse tradeoff, using only training-period (2018) data — using the sealed
   test set for this would be leakage (caught and corrected mid-session).
   `h_selection_sweep.py` currently has a confirmed bug (imports the *non-robust*
   `dispatch_layer`/`dispatch_wrapper` instead of the `_robust` versions, so it would silently
   produce identical, meaningless results at every box level) — fix identified, **on hold**
   per explicit instruction.
3. **GAMMA-sweep**: calibrate the surrogate's penalty strength against the *true*
   hard-constraint robust model's saturation/recourse behavior (a small number of forward
   solves for calibration purposes only — affordable even though full differentiable training
   through the true model is not).
4. Full DFL training on the calibrated surrogate.
5. Testing.

## 5. Scope, narrowed

- `k=0` should receive **no** surrogate treatment at all — its objective has no variance term
  to begin with (uncertainty-blind by construction), so a surrogate meant to approximate
  robust behavior doesn't apply. `k=0` corners only need retraining for the weather-`EXO_COLS`
  change.
- Development and calibration (h-sweep, GAMMA-sweep, full staged methodology) should target
  **one price model first** (single), rigorously, before replicating to dual — running the
  expensive true-model-calibration steps in parallel across both before validating the
  approach once is premature duplication.
- For the dissertation write-up itself: primary result = single price model, k=0 vs.
  k=1-with-calibrated-surrogate, done rigorously (this is the actual DFL contribution — giving
  the forecaster gradient signal about the cost of its own predictive uncertainty, cheaply
  enough to be tractable, validated against the otherwise-intractable true robust model). Dual
  price model = optional secondary section only if time remains.

## 6. Key artifacts

- `6_models/models_robust/convex_surrogate_formulations.md` — exact objective-function math,
  DPP status, and timing for every surrogate variant tried.
- `6_models/models_robust/gpu_feasibility_colab.ipynb` — Julia/CUDA/JAX GPU setup for
  `CUCLARABEL`, currently testing the ellipsoidal hard-constraint reformulation; GPU cells
  never executed. Would need its problem-construction cell swapped to a surrogate formulation
  to test the surrogate approach on GPU.
- `6_models/models_robust/dispatch_layer_robust.py` / `dispatch_wrapper_robust.py` — the
  hard-constraint (box/`mu_p`-`mu_m`) robust formulation as it stands; not what the surrogate
  work trains against.
- `6_models/dispatch_layer.py` — the non-robust baseline formulation (k=0, and the structural
  starting point the surrogates are built on top of for k=1).
- `6_models/models_robust/h_selection_sweep.py` — h-sweep script, has the import bug noted
  above (Task 2, on hold).
- `4_forecasting/forecasting.py` — canonical source for `EXO_COLS`/`HIST_COLS`/`FEAT_COLS`;
  weather-column addition still pending (Task 1).
