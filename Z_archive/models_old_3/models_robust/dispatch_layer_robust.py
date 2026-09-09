from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# =====================================================================================
# dispatch_layer.py  --  PINNED-BID robust LDR dispatch, single- & dual-price objectives.
#
# THREE FORMULATIONS, ONE CONSTRUCTOR:
#   * build_problem(fp, price_model)  -> ProblemBundle   (the shared skeleton)
#       - make_layer(bundle)          -> CvxpyLayer      (TRAINING, differentiable, N=64,
#                                                         cone backend via diffcp/SCS)
#       - solve_plain(bundle, vals)   -> dict            (TESTING, plain Gurobi solve, N=64)
#   * build_oracle(fp, price_model)   -> OracleBundle     (perfect-foresight, Gurobi, N/A)
#
# The ONLY thing that differs single-vs-dual is the imbalance block; everything else
# (variables, SOC, robust box, tracking, penalty) is shared verbatim.
#
# BID: pinned to p^{da} = pl_hat + p_ch_hat - p_dis_hat (matches
# models_old/single_price_robust_old.py's original formulation -- no independent bid
# variable). p_da_rel is kept as a pure-variable DPP stand-in for (p^{da} - pl_hat), so
# C_da = pi_da @ p_da_rel * dt doesn't multiply pi_da by a pl_hat-containing expression
# (not DPP) -- but it's constrained to p_ch_hat - p_dis_hat rather than left free. A free
# p_da_rel let the solver arbitrage the known pi_da/pi_imb spread independently of the
# forecast (measured: bid saturated to a bound every hour, regret ~0 regardless of
# forecast quality). Pinned, the deterministic imbalance offset
#       imb_det = (p_hat_ch - p_hat_dis - p_da_rel)  =  0  identically
# and the realised imbalance is purely recourse-driven:
#       p_imb(xi) = R xi ,   R = I + D_ch - D_dis .
# The DA cost drops the constant  pi_da . pl_hat . dt  (added back DOWNSTREAM).
# =====================================================================================



## To simplify things, going to be giving data to the optimisation problem using a dataclass.
# Means that parameters can be called by their name, e.g. T = fp.T_total
@dataclass
class FixedParams:
    T_total: int          # number of settlement periods in the horizon (=24)
    num_scenarios: int    # scenarios fed to the optimiser (=64 for BOTH train and test)
    dt: float             # length of a settlement period (=1.0 h)
    eta_ch: float         # charging efficiency
    eta_dis: float        # discharging efficiency
    C_ch: float           # max charge rate  (MW, grid-side)
    C_dis: float          # max discharge rate (MW, grid-side)
    B_max: float          # battery energy capacity (MWh)
    SOC0: float           # initial == terminal state of charge (MWh)
    k: float              # 0 = profit-max, 1 = max-dispatchable  (use {0,1} only)
    gamma: float = 1e-4   # Tikhonov coefficient (uniqueness / KKT-invertibility)


def default_fixed_params(k: float, num_scenarios: int = 64, gamma = 1e-6) -> FixedParams:
    """The pinned physical configuration for this dissertation."""
    assert k in (0.0, 1.0), "this study uses k in {0, 1} only"
    return FixedParams(
        T_total=24, num_scenarios=num_scenarios, dt=1.0,
        eta_ch=0.95, eta_dis=0.95, C_ch=2.0, C_dis=2.0, B_max=4.0, SOC0=2.0,
        k=float(k), gamma=gamma,
    )


# -------------------------------------------------------------------------------------
# Robustification helper (unchanged from the reference; the pinned bid does NOT enter
# any robust constraint -- p_da_rel appears only in the objective, via imb_det = 0).
#
# Robustifies, row-wise for t in T:   A0_t + (A xi)_t <= B   for all xi in Xi' = {xi : H xi <= h},
# with H = [I; -I], h = [h_plus; h_minus]  (i.e. -h_minus <= xi <= h_plus, componentwise).
# LP-duality counterpart:  exists mu_p, mu_m >= 0 with  mu_p - mu_m = A  and
#                          mu_p h_plus + mu_m h_minus <= B - A0.
# -------------------------------------------------------------------------------------

## Vectorising the function that creates the dual variables makes it build the model much quicker.
def _robustify_vec(A0, A, B, T, h_plus, h_minus):
    """
    Vectorized robustification.
    A0: (T,) expression
    A: (T, T) expression
    B: scalar
    """
    # Create (T, T) variables instead of looping to create T variables of size T
    mu_p = cp.Variable((T, T), nonneg=True)
    mu_m = cp.Variable((T, T), nonneg=True)

    return [
        mu_p - mu_m == A,  # Evaluates equality for the entire (T,T) matrix
        # (T,T) @ (T,) yields a (T,) vector of dot products for each time step
        mu_p @ h_plus + mu_m @ h_minus <= B - A0,
    ]

# Used to create the cvxpylayers layer.
@dataclass
class ProblemBundle:
    problem: cp.Problem
    params: list                 # ordered cp.Parameter list (order = CvxpyLayer forward order)
    variables: list              # ordered cp.Variable list returned by the layer
    param_by_name: dict
    var_by_name: dict
    price_model: str
    k: float
    # Reminder: the layer objective EXCLUDES the dropped DA constant  pi_da . pl_hat . dt.
    # Add it back downstream when reporting absolute cost / regret.


def build_problem(fp: FixedParams, price_model: str) -> ProblemBundle:
    assert price_model in ("single", "dual")
    T, N, dt = fp.T_total, fp.num_scenarios, fp.dt

    # ---- decision variables ------------------------------------------------------
    p_ch_hat  = cp.Variable(T, name="p_ch_hat")
    p_dis_hat = cp.Variable(T, name="p_dis_hat")
    D_ch      = cp.Variable((T, T), name="D_ch")
    D_dis     = cp.Variable((T, T), name="D_dis")
    # p_da_rel: pure-variable DPP stand-in for (p_da - pl_hat), PINNED below to
    # p_ch_hat - p_dis_hat (matches models_old/single_price_robust_old.py's
    # p_bid = pl_hat + p_ch_hat - p_dis_hat). Left free it lets the solver arbitrage
    # the known pi_da/pi_imb spread independently of the forecast -- see the matching
    # note in dispatch_layer.py/dispatch_layer_1stage.py. Pinned, the bid is bounded
    # automatically by p_ch_hat/p_dis_hat's own physical limits.
    p_da_rel  = cp.Variable(T, name="p_da_rel")

    # ---- always-present parameters ----------------------------------------------
    # NOTE: pl_hat is NOT a cvxpy Parameter here -- with the bid pinned, the forecast
    # never enters this optimization problem; it only matters downstream in
    # realised_cost/realised_breakdown. See the matching note in dispatch_layer.py.
    h_plus  = cp.Parameter(T, nonneg=True, name="h_plus")
    h_minus = cp.Parameter(T, nonneg=True, name="h_minus")

    cons = []
    # LDR non-anticipativity: D lower-triangular (D_{t,tau}=0 for tau>t).
    cons += [cp.upper_tri(D_ch) == 0, cp.upper_tri(D_dis) == 0]


        ## MAKE SOC EQUATION
    ### NOTES ON ROBUSTIFYING PROBLEM FORMULATION
    # To robustify the constraints of the model, we need to separate the random variable \xi
    # from the rest of the model.
    # To do this, we can represent the total SOC function as an affine function.
    # To do this, we separate the day-ahead decisions (which don't rely on \xi at all) from the
    # recourse actions (which depend on \xi).
    # That means that:
    #   e_soc_t = e_soc_t-1 + dt(eta_ch @ (p_ch_hat_t + D_ch_t @ \xi_t)
    #                                      - (1/eta_dis) *
    #                                      (p_dis_hat_t + D_dis_t @ \xi_t)
    #   e_soc = SOC0 + dt * L @ (eta_ch @ p_ch_hat - (1/eta_dis) *
    #                                           (p_dis_hat)
    #           + dt * L @ (eta_ch @ D_ch - (1/eta_dis) * D_dis) @ xi
    # where L is a T x T lower triangular matrix.
    # (this can be coded as L = np.tril(np.ones((P.T_total, P.T_total))))
    #
    # The parts of the equation independent of xi can be interpreted as the intercept of the
    # SOC equation, whilst the parts that relate to xi can be interpreted as the gradient.
    # So:
    # intercept vector (T,):        s_hat = SOC0 + dt * L @ (eta_ch*p_ch_hat - (1/eta_dis)*p_dis_hat)
    # slope matrix (T,T), row t = g_t:   G = dt * L @ (eta_ch*D_ch - (1/eta_dis)*D_dis)
    # then e_soc(xi) = s_hat + G @ xi    for every xi

    # Note: s_hat can be further broken down into SOC0 + expected_power_flow_per_t * cp.cumsum(expected_power_flow_per_t)
    # 1. Nominal State of Charge (s_hat) - 1D Vector Cumulative Sum

    # ---- SOC affine decomposition (bid does NOT enter SOC) -----------------------
    power_flow_hat = fp.eta_ch * p_ch_hat - (1.0 / fp.eta_dis) * p_dis_hat
    s_hat = fp.SOC0 + dt * cp.cumsum(power_flow_hat)               # (T,) nominal SOC
    D_net = fp.eta_ch * D_ch - (1.0 / fp.eta_dis) * D_dis
    G = dt * cp.cumsum(D_net, axis=0)                              # (T,T) recourse SOC gain

    # --- VECTORIZED SOC CONSTRAINTS (Replaces the for-loop) ---


    # (1)(2) Charge rate: 0 <= p_ch_t(xi) <= C_ch
    cons += _robustify_vec( p_ch_hat,  D_ch, fp.C_ch,           T, h_plus, h_minus)
    cons += _robustify_vec(-p_ch_hat, -D_ch, 0.0,               T, h_plus, h_minus)

    # (3)(4) Discharge rate: 0 <= p_dis_t(xi) <= C_dis
    cons += _robustify_vec( p_dis_hat,  D_dis, fp.C_dis,           T, h_plus, h_minus)
    cons += _robustify_vec(-p_dis_hat, -D_dis, 0.0,                T, h_plus, h_minus)

    # (5)(6) State of Charge: 0 <= s_t(xi) <= B_max
    cons += _robustify_vec( s_hat,  G, fp.B_max,           T, h_plus, h_minus)
    cons += _robustify_vec(-s_hat, -G, 0.0,                T, h_plus, h_minus)
    
    """
    # (1)(2) Charge rate: 0 <= p_ch_hat <= C_ch
    cons += [p_ch_hat >= 0, p_ch_hat <= fp.C_ch]

    # (3)(4) Discharge rate: 0 <= p_dis_hat <= C_dis
    cons += [p_dis_hat >= 0, p_dis_hat <= fp.C_dis]

    # (5)(6) State of Charge: 0 <= s_hat <= B_max
    cons += [s_hat >= 0, s_hat <= fp.B_max]
    """

    # terminal equality. Outside for loop.
    cons += [
        s_hat[T - 1] == fp.SOC0,  # s_hat = SOC0
        G[T - 1, :] == 0,
    ]  # g_T = 0 : recourse is energy-neutral


    # ---- bid: PINNED (p^{da} = pl_hat + p_ch_hat - p_dis_hat), no separate bound needed --
    cons += [p_da_rel == p_ch_hat - p_dis_hat]

    # ---- shared imbalance building blocks ----------------------------------------
    imb_det = p_ch_hat - p_dis_hat - p_da_rel          # (T,) variable-only; = g_hat - p^{da}
    R = np.eye(T) + D_ch - D_dis                      # (T,T) recourse matrix

    params = []
    obj_terms = []

    ## The major ball ache: DPP COMPLIANCE
    # To pass a cvxpy problem through cvxpylayers, the problem must be DPP compliant.
    # """
    # To compile a parameterized problem, CVXPY must be able to express all constraints and objectives in a canonical form (e.g., Ax < b) 
    # where the coefficients (like A and b) are strictly affine functions of the parameters. 
    # Multiplying a cp.Parameter by another cp.Parameter creates a non-linear (quadratic or bilinear) parameter dependence, which CVXPY cannot factorize.
    # """

    # ============================ ECONOMIC TERM  (only if (1-k) > 0) ===============
    
    if fp.k < 1.0:
        # NOTE: to ensure DPP compliance, constant  pi_da . pl_hat . dt  is DROPPED here
        # and MUST be added back downstream (realised_cost / reporting).
        pi_da = cp.Parameter(T, name="pi_da")
        params.append(pi_da)
        # DA cost: variable part only. 
        C_da = pi_da @ p_da_rel * dt

        if price_model == "single":
            # SIGNED settlement.  E[pi_imb . p_imb] = pi_imb . imb_det  (R xi term vanishes,
            # E[xi]=0), so NO scenarios and NO epigraph are needed in the single objective.
            # WOOHOOO!!!! makes life so much easier.
            pi_imb = cp.Parameter(T, name="pi_imb")           # signed imbalance price
            params.append(pi_imb)
            C_imb = pi_imb @ imb_det * dt
        else:

            # one way to implement: 
            # C_imb = sum(pi_imb_up @ cp.pos(p_imb) + pi_imb_down @ cp.neg(p_imb)) * dt / N
            # where p_imb = recourse_matrix @ xi_samples.T
            # However, DPP does not accept non-linear atoms (like .pos and .neg) that are not purely a variable or purely a parameter.
            # Also, multiplying an imbalance cost by p_imb means that it must be param(price) X var (rec matrix) X param (xi_samples).
            # Because of both of these reasons, it is not DPP compliant.

            # However, p_imb can be reformulated using an epigraph:
            # p_plus  = cp.Variable((T, N), nonneg=True, name="p_plus")
            # p_minus = cp.Variable((T, N), nonneg=True, name="p_minus")
            # where the difference between p_plus and p_minus equals p_imb at all timesteps.
            # From here, we have to consider the deterministic and recourse-based realisation of imbalance.
            # As the imbalance is non-symmetric for dual-price formulation, 
            # We need to work out per-scenario imbalance by broadcasting the deterministic imbalance power
            # to each scenario's recourse actions.
            # Then, we need to make sure that the epigraph is valid for all scenarios by adding the constraints.

            # DUAL no-arbitrage penalty via epigraph split (DPP-clean, removes the kink).
            xi_samples = cp.Parameter((N, T), name="xi_samples")
            pi_imb_up     = cp.Parameter(T, nonneg=True, name="pi_imb_up")
            pi_imb_down     = cp.Parameter(T, nonneg=True, name="pi_imb_down")
            params += [xi_samples, pi_imb_up, pi_imb_down]
            p_plus  = cp.Variable((T, N), nonneg=True, name="p_plus")
            p_minus = cp.Variable((T, N), nonneg=True, name="p_minus")

            # per-scenario imbalance = imb_det (broadcast) + R xi^s ; outer-product broadcast
            # of the deterministic offset is DPP-safe (var @ const).
            ones_N = np.ones((1, N))
            imb_scen = cp.reshape(imb_det, (T, 1), order='C') @ ones_N + R @ xi_samples.T   # (T, N)
            cons += [p_plus - p_minus == imb_scen]
            # math (.)^- <= 0  maps to code p_minus >= 0 :  -imb_down (.)^- = imb_down . p_minus
            C_imb = cp.sum(pi_imb_up @ p_plus + pi_imb_down @ p_minus) * dt / N

        obj_terms.append((1.0 - fp.k) * (C_da + C_imb))

    # ============================ TRACKING TERM  (only if k > 0) ===================
    if fp.k > 0.0:
        
        # To maintain DPP in this problem, I have to make sure I don't multiply any parameters by any parameters.
        # This means I can't multiply p_imb by itself as I defined originally in my equations, as p_imb contained xi_samples, a PARAMETER.
        # To get around this, I can reformulate the quadratic part of the equation using the Georghiou et. al. paper original LDR paper.
        # E(X^2) = Var(X) + (E(X))^2
        # So, if E(xi_samples) is 0: E((Recourse Matrix @ Xi samples) ^ 2) = Variance (Recourse Matrix @ Xi samples)
        # Which is the same as 
        # E[(r_t @ xi)^2] = r_t @ Sigma_xi @ r_t^T
        # This reformulation ONLY WORKS IF E()
        # where r_t is the t'th row of the recouse matrix, Sigma_xi is the second moment of xi_samples.
        # This would be:
        # sum_trace = fixed_params.dt**2 * (cp.sum_squares(imb_det) + (sum(cp.quad_form(R[t, :], Sigma_xi) for t in range(T))
        # HOWEVER:
        # apparently cvxpy doesn't like that formulation and it wouldn't be DPP, so I have to do it using the cholesky factor and using sum_squares.
        

        Sigma_xi_chol = cp.Parameter((T, T), name="Sigma_xi_chol")
        params.append(Sigma_xi_chol)
        sum_trace = dt**2 * (cp.sum_squares(imb_det) + cp.sum_squares(R @ Sigma_xi_chol))
        obj_terms.append(fp.k * sum_trace)

    # ============================ PENALTY  (always) =================================
    # p_da_rel no longer needs its own reg term -- it's pinned to p_ch_hat - p_dis_hat,
    # so their sum_squares terms already cover it.
    penalty = fp.gamma * (
        cp.sum_squares(p_ch_hat) + cp.sum_squares(p_dis_hat)
        + cp.sum_squares(D_ch) + cp.sum_squares(D_dis)
    )
    obj_terms.append(penalty)

    # trailing box params (stable order)
    params += [h_plus, h_minus]

    prob = cp.Problem(cp.Minimize(sum(obj_terms)), cons)
    assert prob.is_dcp(dpp=True), "not DPP -- cvxpylayers will reject"

    variables = [p_ch_hat, p_dis_hat, D_ch, D_dis, p_da_rel]
    return ProblemBundle(
        problem=prob,
        params=params,
        variables=variables,
        param_by_name={p.name(): p for p in params},
        var_by_name={v.name(): v for v in variables},
        price_model=price_model,
        k=float(fp.k),
    )

# -------------------------------------------------------------------------------------
# TRAINING: differentiable layer
# -------------------------------------------------------------------------------------
## NOTE: For training and testing the model stays the same
# The difference is that make_layer is used for creating the cvxpylayers layer,
# whereas solve_plain is 
def make_layer(bundle: ProblemBundle) -> CvxpyLayer:
    return CvxpyLayer(bundle.problem, parameters=bundle.params, variables=bundle.variables)



def order_param_values(bundle: ProblemBundle, **named) -> list:
    """Return values in the bundle's parameter order, for the CvxpyLayer forward pass."""
    missing = [p.name() for p in bundle.params if p.name() not in named]
    if missing:
        raise KeyError(f"missing param values for this ({bundle.price_model}, k={bundle.k}) "
                       f"problem: {missing}. Expected order: {[p.name() for p in bundle.params]}")
    return [named[p.name()] for p in bundle.params]


# -------------------------------------------------------------------------------------
# TESTING: plain forward solve with Gurobi (QP). Reuses the SAME problem object, so
# train and test provably encode the identical formulation. Objective EXCLUDES the
# dropped DA constant (fine -- we only use the DECISIONS at test).
# -------------------------------------------------------------------------------------
def solve_plain(bundle: ProblemBundle, values: dict, solver=cp.GUROBI, **solver_kwargs) -> dict:
    # Convert the bundle params from tensors or other dtypes to nparrays.
    for name, p in bundle.param_by_name.items():
        if name not in values:
            raise KeyError(f"missing param value: {name} "
                           f"(need {list(bundle.param_by_name)})")
        p.value = np.asarray(values[name], dtype=float)
    
    # Solve the problem.
    bundle.problem.solve(solver=solver, **solver_kwargs)
    if bundle.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"plain solve status: {bundle.problem.status}")
    
    # return the outputs of the solution
    out = {name: np.asarray(v.value) for name, v in bundle.var_by_name.items()}
    out["objective_no_da_const"] = float(bundle.problem.value)
    return out


# =====================================================================================
# PERFECT-FORESIGHT ORACLE  (deterministic; no LDR, no robust box, no tracking).
# Bid PINNED (bid = p_d + p_ch - p_dis), matching the policy's pinned bid. Gurobi (LP).
# =====================================================================================
@dataclass
class OracleBundle:
    problem: cp.Problem
    param_by_name: dict
    var_by_name: dict
    price_model: str
    objective: str = "economic"   # the ONLY oracle in this study is economic (see build_oracle)
 
 
def build_oracle(fp: FixedParams, price_model: str, objective: str = "economic") -> OracleBundle:
    """Perfect-foresight ECONOMIC oracle (min C_da + C_imb), used as the COMMON benchmark
    for every instance -- all four corners, baseline and DFL.
 
    `objective` is an explicit guard, not a variant selector. This study deliberately has
    ONE oracle: the economic one. Regret = policy_realised_cost - economic_oracle_cost is
    reported for all eight instances and read as *economic decision regret* -- true decision
    regret at k=0, and the economic price of dispatchability at k=1 (same formula, the
    interpretation shifts with k). A perfect-foresight DISPATCH oracle is intentionally NOT
    built: min sum (p_imb)^2 under clairvoyance collapses toward a forecast-accuracy gap,
    which CRPS already owns and this model-only framework scopes out. Passing anything other
    than "economic" fails loudly so no k=1 metric can be silently benchmarked against a
    dispatch oracle that does not exist here.
    """
    assert price_model in ("single", "dual")
    if objective != "economic":
        raise ValueError(
            f"build_oracle objective must be 'economic' (got {objective!r}). This study uses a "
            "single common economic oracle for all corners; there is no dispatch oracle -- k=1 "
            "dispatchability is reported on the raw absolute-deviation axis, not as regret.")
    T = fp.T_total
 
    p_ch  = cp.Variable(T, nonneg=True, name="p_ch")
    p_dis = cp.Variable(T, nonneg=True, name="p_dis")

    p_d   = cp.Parameter(T, name="p_d")                   # TRUE prosumption (perfect foresight)
    pi_da = cp.Parameter(T, name="pi_da")

    # bid: PINNED to the true realised position (see matching note in dispatch_layer.py's
    # build_oracle). A free bid lets the oracle arbitrage the known pi_da/pi_imb spread
    # using only price info the policy has too -- not genuine value of knowing p_d.
    bid = p_d + p_ch - p_dis                              # plain solve -- no DPP constraint

    L = np.tril(np.ones((T, T)))
    soc = fp.SOC0 + fp.dt * (L @ (fp.eta_ch * p_ch - (1.0 / fp.eta_dis) * p_dis))
    cons = [
        p_ch <= fp.C_ch, p_dis <= fp.C_dis,
        soc >= 0, soc <= fp.B_max,
        soc[T - 1] == fp.SOC0,                            # hard terminal equality
    ]

    net_draw = p_d + p_ch - p_dis                         # p^g with the TRUE realisation
    imb = net_draw - bid                                  # == 0 identically (bid pinned to net_draw)
    C_da = pi_da @ bid * fp.dt
 
    param_by_name = {"p_d": p_d, "pi_da": pi_da}
    if price_model == "single":
        pi_imb = cp.Parameter(T, name="pi_imb")
        param_by_name["pi_imb"] = pi_imb
        C_imb = pi_imb @ imb * fp.dt                      # signed (can be revenue)
    else:
        pi_imb_up = cp.Parameter(T, nonneg=True, name="pi_imb_up")
        pi_imb_down = cp.Parameter(T, nonneg=True, name="pi_imb_down")
        param_by_name["imb_up"], param_by_name["imb_down"] = pi_imb_up, pi_imb_down
        # plain solve -> DPP not required -> cp.pos/cp.neg is fine and clean.
        C_imb = (pi_imb_up @ cp.pos(imb) + pi_imb_down @ cp.neg(imb)) * fp.dt
 
    prob = cp.Problem(cp.Minimize(C_da + C_imb), cons)
    return OracleBundle(
        problem=prob,
        param_by_name=param_by_name,
        var_by_name={"p_ch": p_ch, "p_dis": p_dis, "bid": bid},
        price_model=price_model,
        objective="economic",
    )
 
 
def solve_oracle(bundle: OracleBundle, values: dict, solver=cp.GUROBI,
                 return_decisions: bool = False, **solver_kwargs):
    """Oracle objective is the FULL cost (no dropped constant -- bid is the actual bid)."""
    for name, p in bundle.param_by_name.items():
        if name not in values:
            raise KeyError(f"missing oracle param value: {name} "
                           f"(need {list(bundle.param_by_name)})")
        p.value = np.asarray(values[name], dtype=float)
    bundle.problem.solve(solver=solver, **solver_kwargs)
    if bundle.problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"oracle solve status: {bundle.problem.status}")
    cost = float(bundle.problem.value)
    if return_decisions:
        return cost, {name: np.asarray(v.value) for name, v in bundle.var_by_name.items()}
    return cost


# -------------------------------------------------------------------------------------
# Second-moment Cholesky factor.  M = xi^T xi / N  (biased /N: matches the epigraph's
# /N averaging AND makes the tracking cross term vanish EXACTLY for mean-centred xi).
# Symmetrise + jitter before Cholesky (cheap insurance vs intermittent LinAlgError).
# N here MUST equal the model's N (one draw -> one mean -> pl_hat, h, xi, M all share it).
# -------------------------------------------------------------------------------------
def cholesky_of_second_moment(xi: np.ndarray, jitter: float = 1e-9) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)              # (N, T), mean-centred
    N, T = xi.shape
    M = xi.T @ xi / N
    M = 0.5 * (M + M.T) + jitter * np.eye(T)
    return np.linalg.cholesky(M)                  # lower L with L L^T = M
