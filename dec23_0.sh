#!/bin/sh

# Baseline reproduction for atari100k (This will take a long time like 4-5 days)
SWEEP_OUTPUT=$(wandb sweep experiments/atari100k_sweep.yml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent [^ ]*' | awk '{print $3}')
# Print ID to debug
echo "Detected Sweep ID: $SWEEP_ID"
wandb agent $SWEEP_ID