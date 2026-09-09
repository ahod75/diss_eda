"""
Cholesky-conditioning test: does the gradient through sum_squares(R @ chol(M)) stay finite
and match the direct scenario form  sum_squares(R @ xi.T)/N  on real 2018 days?

Both compute the SAME tracking value  Σ_t r_t^T M r_t  (M = xi^T xi / N), so their gradients
w.r.t. the forecaster (here proxied by xi) must agree in exact arithmetic. Where they DIVERGE
in floating point is where the Cholesky backward is fragile -- expected on low-min-eigenvalue
(near-degenerate scenario) days.

Run in your env (needs torch). Wire `make_day_inputs` from the testing notebook, or paste in
xi arrays. This file is torch-only; no cvxpy/solve needed -- it isolates the tracking term.
"""
import numpy as np
import torch


def tracking_via_cholesky(R, xi, jitter):
    """sum_squares(R @ chol(M + jitter*I)),  M = xi^T xi / N.  Gradient flows through chol."""
    N = xi.shape[0]
    M = xi.T @ xi / N
    M = 0.5 * (M + M.T) + jitter * torch.eye(M.shape[0], dtype=xi.dtype, device=xi.device)
    L = torch.linalg.cholesky(M)              # <-- the fragile op (its backward inherits M's cond)
    return (R @ L).pow(2).sum()


def tracking_via_scenarios(R, xi):
    """sum_squares(R @ xi.T) / N  ==  Σ_t r_t^T M r_t.  No Cholesky; matmuls only."""
    N = xi.shape[0]
    return (R @ xi.T).pow(2).sum() / N


def analytic_dtrack_dR(R, xi):
    """Closed-form gradient of the tracking term w.r.t. R:  d/dR Σ_t r_t^T M r_t = 2 R M."""
    N = xi.shape[0]
    M = xi.T @ xi / N
    return 2.0 * R @ M


def run_case(name, xi_np, jitter=1e-9, seed=0):
    torch.manual_seed(seed)
    T = xi_np.shape[1]
    # eigen-conditioning of M
    M = xi_np.T @ xi_np / xi_np.shape[0]
    w = np.linalg.eigvalsh(0.5 * (M + M.T))
    cond = w.max() / max(w.min(), 1e-30)

    # a representative recourse matrix R = I + lower-tri noise (like I + D_ch - D_dis)
    R0 = np.eye(T) + np.tril(np.random.default_rng(seed).normal(0, 0.1, (T, T)))

    results = {}
    for label, fn in [("cholesky", lambda R, xi: tracking_via_cholesky(R, xi, jitter)),
                      ("scenarios", lambda R, xi: tracking_via_scenarios(R, xi))]:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        xi = torch.tensor(xi_np, dtype=torch.float64)
        val = fn(R, xi)
        val.backward()
        g = R.grad.detach().numpy()
        results[label] = (float(val), g, np.isfinite(g).all())

    # ground-truth gradient w.r.t. R (closed form)
    R = torch.tensor(R0, dtype=torch.float64)
    xi = torch.tensor(xi_np, dtype=torch.float64)
    g_true = analytic_dtrack_dR(R, xi).numpy()

    v_c, g_c, fin_c = results["cholesky"]
    v_s, g_s, fin_s = results["scenarios"]
    val_match = np.isclose(v_c, v_s, rtol=1e-8)
    g_cs_gap  = np.abs(g_c - g_s).max() if (fin_c and fin_s) else np.inf
    g_c_err   = np.abs(g_c - g_true).max() if fin_c else np.inf
    g_s_err   = np.abs(g_s - g_true).max() if fin_s else np.inf

    print(f"[{name}]  min_eig(M)={w.min():.2e}  cond(M)={cond:.1e}")
    print(f"    value match (chol vs scen): {val_match}   ({v_c:.6f} vs {v_s:.6f})")
    print(f"    grad finite:  cholesky={fin_c}   scenarios={fin_s}")
    print(f"    max|grad_chol - grad_scen| = {g_cs_gap:.2e}")
    print(f"    max|grad - analytic(2 R M)|:  cholesky={g_c_err:.2e}   scenarios={g_s_err:.2e}")
    print()


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    T, N = 24, 64

    # (1) well-conditioned: full-rank correlated errors
    xi = rng.normal(0, 0.4, (N, T)); xi -= xi.mean(0, keepdims=True)
    run_case("well-conditioned", xi)

    # (2) near-degenerate: scenarios nearly collinear (low-rank error cloud)
    #     mimics a day where the copula draw doesn't span the error space
    base = rng.normal(0, 0.4, (N, 3))                     # only 3 effective directions
    load = rng.normal(0, 1.0, (3, T))
    xi = base @ load + 1e-3 * rng.normal(0, 1, (N, T))    # tiny full-rank floor
    xi -= xi.mean(0, keepdims=True)
    run_case("near-degenerate (rank-3)", xi)

    # (3) degenerate with the CURRENT tiny jitter vs a larger jitter -- does jitter rescue grad?
    run_case("near-degenerate, jitter 1e-9", xi, jitter=1e-9)
    run_case("near-degenerate, jitter 1e-6", xi, jitter=1e-6)

    print("READ: value should match everywhere. Where grad_chol diverges from grad_scen or")
    print("from the analytic 2RM (or goes non-finite), the Cholesky backward is the culprit.")
    print("Proxy-1 (scenarios) should track the analytic gradient regardless of conditioning.")
    print("\nTo test on YOUR data: replace the xi arrays with sampler.mean_and_errors(quantiles)[1]")
    print("for a calm day and a volatile day, and compare the two grad columns.")
