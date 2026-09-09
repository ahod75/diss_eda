#!/bin/bash
# Orchestrator: train all four corners to convergence via train_corner.py. Each corner is
# a FRESH python process (memory reset between corners) wrapped in a watchdog that kills
# it if RSS exceeds 6GB (WSL2 has 7.6GB RAM; earlier runs swap-thrashed when a single
# process accumulated too much state across corners).
#
# NOTE: this replaces an earlier version of this script that ran a verify-then-gate step
# against the (now abandoned) alpha/beta floor fix. Early stopping is now keyed on
# val_fsurr directly (see train_corner.py), and k=1's tracking term now uses Sigma_xi_chol
# instead of xi_samples (~16-24x faster per solve, verified against real data) -- so no
# separate verification stage is needed before launching the real run.
set -uo pipefail
cd /home/fleet75/diss_bess
VENV=.venv/bin/python
LOG_DIR=7_model_training/logs
SCRIPT_DIR=7_model_training
MAX_RSS_KB=6000000
mkdir -p "$LOG_DIR"

run_with_watchdog() {
  local logfile=$1; shift
  "$@" > "$logfile" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$rss" ] && [ "$rss" -gt "$MAX_RSS_KB" ]; then
      echo "WATCHDOG_KILL pid=$pid rss=${rss}KB exceeds ${MAX_RSS_KB}KB" | tee -a "$logfile"
      kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 137
    fi
    sleep 20
  done
  wait "$pid"
  return $?
}

echo "PIPELINE_START $(date -Iseconds)"

for pair in "single 0.0" "single 1.0" "dual 0.0" "dual 1.0"; do
  set -- $pair
  pm=$1; k=$2
  kshort=${k%.*}
  logfile="$LOG_DIR/train_${pm}_k${kshort}.log"
  echo "CORNER_START price_model=$pm k=$k $(date -Iseconds)"
  run_with_watchdog "$logfile" $VENV "$SCRIPT_DIR/train_corner.py" "$pm" "$k"
  rc=$?
  tail -8 "$logfile"
  if [ $rc -ne 0 ]; then
    echo "CORNER_FAILED price_model=$pm k=$k rc=$rc, see $logfile"
    echo "PIPELINE_ABORTED"
    exit 1
  fi
  echo "CORNER_DONE price_model=$pm k=$k $(date -Iseconds)"
done

echo "PIPELINE_COMPLETE $(date -Iseconds)"
