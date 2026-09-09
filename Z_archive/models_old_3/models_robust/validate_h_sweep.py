"""
Validate the torch/cvxpy-free logic of h_selection_sweep.py:
  - boxes_from_quantiles matches the intended h_plus/h_minus definition
  - _aggregate produces correct yearly sum/mean and shapes
  - _assert_k1_price_free passes on identical k=1 metrics and RAISES on a mismatch
"""
import ast
import numpy as np
import pandas as pd

import h_selection_sweep as H


def ok(name, cond): print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def check_parse():
    ast.parse(open("h_selection_sweep.py").read())
    print("  [PASS] h_selection_sweep.py parses (ast)")


def check_boxes():
    rng = np.random.default_rng(0)
    K, Q = 24, 19
    levels = np.round(np.linspace(0.05, 0.95, Q), 2)
    q = np.sort(rng.normal(0, 1, (K, Q)), axis=1)          # monotone quantiles
    mean = q[:, Q // 2] + rng.normal(0, 0.05, K)           # near-median mean
    boxes = H.boxes_from_quantiles(q, mean, levels, box_levels=[(0.05, 0.95), (0.25, 0.75)])
    hp, hm = boxes[(0.05, 0.95)]
    i_lo, i_hi = int(np.argmin(np.abs(levels - 0.05))), int(np.argmin(np.abs(levels - 0.95)))
    ok("h_plus == clip(q_upper - mean)", np.allclose(hp, np.clip(q[:, i_hi] - mean, H.MIN_BOX, None)))
    ok("h_minus == clip(mean - q_lower)", np.allclose(hm, np.clip(mean - q[:, i_lo], H.MIN_BOX, None)))
    # wider levels (0.25,0.75) give a TIGHTER box than (0.05,0.95)
    hp2, _ = boxes[(0.25, 0.75)]
    ok("(0.25,0.75) box is tighter than (0.05,0.95)", (hp2 <= hp + 1e-9).all())


def _fake_per_day(n_days=5):
    rows = []
    rng = np.random.default_rng(1)
    for pm, k in H.CORNERS:
        for (lo, hi) in H.BOX_LEVELS:
            for d in range(n_days):
                # make k=1 single and dual identical (price-free); k=0 may differ
                seed = (int(k), lo, hi, d) if k == 1 else (pm, int(k), lo, hi, d)
                r = np.random.default_rng(abs(hash(seed)) % (2**32))
                rows.append({"price_model": pm, "k": int(k), "box_lo": lo, "box_hi": hi,
                             "delivery_start": pd.Timestamp("2018-01-01") + pd.Timedelta(days=d),
                             "total_charge_MWh": r.uniform(0, 10), "total_discharge_MWh": r.uniform(0, 10),
                             "sat_hours": int(r.integers(0, 24)), "sat_MWh": r.uniform(0, 5)})
    return pd.DataFrame(rows)


def check_aggregate():
    per_day = _fake_per_day()
    yearly = H._aggregate(per_day)
    # one row per (corner, box_level) = 4 corners x 7 boxes = 28
    ok("yearly has one row per corner x box_level (28)", len(yearly) == len(H.CORNERS) * len(H.BOX_LEVELS))
    # spot-check a sum against a manual groupby
    sub = per_day[(per_day.price_model == "dual") & (per_day.k == 0) & (per_day.box_lo == 0.05)]
    row = yearly[(yearly.price_model == "dual") & (yearly.k == 0) & (yearly.box_lo == 0.05)].iloc[0]
    ok("sum matches manual", np.isclose(row["sat_MWh_sum"], sub["sat_MWh"].sum()))
    ok("mean matches manual", np.isclose(row["total_charge_MWh_mean"], sub["total_charge_MWh"].mean()))
    ok("n_days recorded", int(row["n_days"]) == len(sub))


def check_k1_consistency():
    per_day = _fake_per_day()
    try:
        H._assert_k1_price_free(per_day)      # constructed so k=1 single==dual -> should pass
        passed = True
    except AssertionError:
        passed = False
    ok("k=1 consistency PASSES on identical decisions", passed)
    # now corrupt one dual k=1 value -> must raise
    bad = per_day.copy()
    mask = (bad.k == 1) & (bad.price_model == "dual")
    bad.loc[bad[mask].index[0], "sat_MWh"] += 1.0
    try:
        H._assert_k1_price_free(bad)
        raised = False
    except AssertionError:
        raised = True
    ok("k=1 consistency RAISES on a mismatch", raised)


if __name__ == "__main__":
    print("h-sweep logic validation (numpy/pandas):")
    check_parse()
    check_boxes()
    check_aggregate()
    check_k1_consistency()
    print("Done. (forecaster + sampler + solves run in your env.)")
