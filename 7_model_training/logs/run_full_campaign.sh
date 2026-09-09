#!/usr/bin/env bash
# Full retrain campaign: N=5 seeds x {CRPS-only control, DFL (single+dual-price)}.
# Sequential (safe re: memory/CPU contention), continues past any single failure
# (matches train_dfl_forecasts.py's own per-corner resilience philosophy) rather than
# aborting the whole multi-hour run over one bad seed.
cd /home/fleet75/diss_bess
SEEDS="20240801 20240802 20240803 20240804 20240805"
LOG_DIR="7_model_training/logs"

echo "CAMPAIGN_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"

for seed in $SEEDS; do
  echo ""
  echo "=== SEED $seed: crps_only_retrained ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
  .venv/bin/python3 7_model_training/train_crps_only.py --seed "$seed" \
    > "$LOG_DIR/crps_only_seed${seed}.log" 2>&1
  if [ $? -eq 0 ]; then echo "  OK"; else echo "  FAILED (seed=$seed) -- see $LOG_DIR/crps_only_seed${seed}.log"; fi

  echo "=== SEED $seed: dfl_forecasts (single+dual-price) ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
  .venv/bin/python3 7_model_training/train_dfl_forecasts.py --seed "$seed" \
    > "$LOG_DIR/dfl_forecasts_seed${seed}.log" 2>&1
  if [ $? -eq 0 ]; then echo "  OK"; else echo "  FAILED (seed=$seed) -- see $LOG_DIR/dfl_forecasts_seed${seed}.log"; fi
done

echo ""
echo "CAMPAIGN_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
