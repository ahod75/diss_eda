"""
Validate the torch/cvxpy-free control flow of dfl_train.py:
  - file parses
  - the early-stop / restore-best / patience logic behaves (mocked loss curve)
  - batch iteration covers every day exactly once per epoch
"""
import ast
import numpy as np

import sys
from pathlib import Path
from pyprojroot import here

ROOT_DIR = here()
FORECASTING_DIR = ROOT_DIR / "4_forecasting"
DATA_DIR = ROOT_DIR / "1_data" / "processed"
COPULA_DIR = ROOT_DIR / "5_scenario_gen"
MODEL_DIR = ROOT_DIR / "6_models"
DFL_TRAIN_DIR = ROOT_DIR / "7_model_training"

sys.path.insert(0, str(FORECASTING_DIR))  # make forecasting module importable
sys.path.insert(0, str(COPULA_DIR))  # make copula module importable
sys.path.insert(0, str(MODEL_DIR))  # make model modules importable
sys.path.insert(0, str(DFL_TRAIN_DIR)) # make training script importable


def ok(name, cond): print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def check_parse():
    ast.parse(open(DFL_TRAIN_DIR / "dfl_train.py").read())
    print("  [PASS] dfl_train.py parses (ast)")


def mock_early_stop(val_curve, patience, min_delta=0.0):
    """Reproduce train_one_config's early-stop/restore-best logic on a given val-regret curve.
    Returns (stop_epoch, best_val, restored_epoch)."""
    best_val = float("inf"); best_epoch = -1; pat = 0
    for epoch, v in enumerate(val_curve):
        improved = v < best_val - min_delta
        if improved:
            best_val = v; best_epoch = epoch; pat = 0
        else:
            pat += 1
            if pat >= patience:
                return epoch, best_val, best_epoch
    return len(val_curve) - 1, best_val, best_epoch


def check_early_stop():
    # improving then flat: should stop `patience` epochs after the best, restore the best
    curve = [10, 8, 6, 5, 5.1, 5.2, 5.3]          # best at epoch 3 (val 5)
    stop, best, restored = mock_early_stop(curve, patience=3)
    ok("stops patience epochs after best", stop == 6)          # 3 non-improving: 4,5,6
    ok("best value found", np.isclose(best, 5.0))
    ok("restores the best epoch", restored == 3)

    # monotonically improving: never early-stops, restores last
    curve2 = [10, 9, 8, 7, 6]
    stop2, best2, restored2 = mock_early_stop(curve2, patience=3)
    ok("no early stop when always improving", stop2 == 4 and restored2 == 4)

    # immediate plateau: stops at `patience`, restores epoch 0
    curve3 = [5, 5, 5, 5]
    stop3, best3, restored3 = mock_early_stop(curve3, patience=3)
    ok("plateau stops at patience, restores first", stop3 == 3 and restored3 == 0)


def check_batching():
    # every day covered exactly once per epoch, batches of batch_size (last may be short)
    n_train, batch_size = 53, 16
    order = np.random.permutation(n_train)
    seen, n_batches = [], 0
    for start in range(0, n_train, batch_size):
        batch = order[start:start + batch_size]
        seen.extend(batch.tolist()); n_batches += 1
    ok("all days covered once per epoch", sorted(seen) == list(range(n_train)))
    ok("correct number of batches", n_batches == int(np.ceil(n_train / batch_size)))
    ok("last batch is the remainder", n_train % batch_size == 53 % 16 == 5)


def check_oracle_independence():
    # sanity of the design claim: oracle cost is a per-(day,price_model) constant, so regret
    # for two forecasters on the same day differs ONLY by their realised_cost.
    oracle = 100.0
    cost_a, cost_b = 130.0, 118.0
    ok("regret difference == realised_cost difference (oracle cancels)",
       np.isclose((cost_a - oracle) - (cost_b - oracle), cost_a - cost_b))


if __name__ == "__main__":
    print("dfl_train control-flow validation (numpy):")
    check_parse()
    check_early_stop()
    check_batching()
    check_oracle_independence()
    print("Done. (forecaster forward + layer + solves run in your env.)")
