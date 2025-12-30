#!/bin/sh

# Baseline reproduction for atari100k (This will take a long time like 4-5 days)
SWEEP_OUTPUT=$(wandb sweep experiments/pinpad-easy.yml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent [^ ]*' | awk '{print $NF}')
# Print ID to debug
echo "Detected Sweep ID: $SWEEP_ID"
wandb agent $SWEEP_ID