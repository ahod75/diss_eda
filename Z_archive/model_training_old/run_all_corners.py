"""Master orchestration: runs all 6 DFL fine-tuning corners sequentially, cheapest
first. 1-stage (single, dual) first as a cheap guaranteed fallback (if the more
expensive point-constrained corners produce nothing useful, these results still
exist), then point-constrained 2-stage corners in ascending measured cost order
(single/k0 cheapest; single/k1 and dual/k1 structurally identical decisions, hence
near-identical cost; dual/k0 most expensive, driven by the N=64-scenario epigraph).

Each corner runs as a SEPARATE subprocess (own process/CUDA/memory context, matches
each script's existing CLI design) with full output streamed to its own log file in
real time, plus a start/done/failed summary line appended to pipeline.log. A failure
in one corner does NOT stop the rest -- the whole point of running cheap-first is that
a later failure shouldn't cost the earlier results.

Usage: python run_all_corners.py
"""
import subprocess
import sys
import time
from pyprojroot import here

ROOT_DIR = here()
TRAIN_DIR = ROOT_DIR / "7_model_training"
LOG_DIR = TRAIN_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
PIPELINE_LOG = LOG_DIR / "pipeline.log"
PYTHON = sys.executable

# (description, script, args, log_filename)
CORNERS = [
    ("1stage/single",          "train_corner_1stage.py",       ["single"],      "train_1stage_single.log"),
    ("1stage/dual",            "train_corner_1stage.py",       ["dual"],        "train_1stage_dual.log"),
    ("point_robust/single/k0", "train_corner_point_robust.py", ["single", "0"], "train_point_robust_single_k0.log"),
    ("point_robust/single/k1", "train_corner_point_robust.py", ["single", "1"], "train_point_robust_single_k1.log"),
    ("point_robust/dual/k1",   "train_corner_point_robust.py", ["dual", "1"],   "train_point_robust_dual_k1.log"),
    ("point_robust/dual/k0",   "train_corner_point_robust.py", ["dual", "0"],   "train_point_robust_dual_k0.log"),
]


def log_pipeline(msg: str) -> None:
    print(msg, flush=True)
    with open(PIPELINE_LOG, "a") as f:
        f.write(msg + "\n")


def main():
    log_pipeline(f"PIPELINE_START {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    results = []
    for desc, script, args, log_name in CORNERS:
        log_path = LOG_DIR / log_name
        cmd = [PYTHON, str(TRAIN_DIR / script), *args]
        log_pipeline(f"CORNER_LAUNCH {desc} cmd={' '.join(cmd)} log={log_path}")
        t0 = time.time()
        try:
            with open(log_path, "w") as logf:
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                       cwd=str(TRAIN_DIR), check=False)
            elapsed = time.time() - t0
            if proc.returncode == 0:
                log_pipeline(f"CORNER_OK {desc} elapsed_s={elapsed:.0f}")
                results.append((desc, "OK", elapsed))
            else:
                log_pipeline(f"CORNER_FAILED {desc} returncode={proc.returncode} "
                             f"elapsed_s={elapsed:.0f} -- see {log_path}")
                results.append((desc, f"FAILED(rc={proc.returncode})", elapsed))
        except Exception as e:
            elapsed = time.time() - t0
            log_pipeline(f"CORNER_EXCEPTION {desc} error={e!r} elapsed_s={elapsed:.0f}")
            results.append((desc, f"EXCEPTION({e!r})", elapsed))

    log_pipeline("PIPELINE_SUMMARY")
    for desc, status, elapsed in results:
        log_pipeline(f"  {desc:28s} {status:24s} {elapsed:8.0f}s")
    log_pipeline(f"PIPELINE_COMPLETE {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
