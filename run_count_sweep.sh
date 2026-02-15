#!/bin/sh

# Count-based Exploration Sweep for pinpad-easy
# This script tests episodic count-based exploration
# on pinpad-easy tasks with multiple seeds

SWEEP_OUTPUT=$(wandb sweep experiments/pinpad-easy-count-weight-sweep.yml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent [^ ]*' | awk '{print $NF}')
# Print ID to debug
echo "Detected Sweep ID: $SWEEP_ID"
wandb agent $SWEEP_ID
