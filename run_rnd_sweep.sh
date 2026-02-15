#!/bin/sh

# RND-only Exploration Sweep for pinpad-easy
# This script tests Random Network Distillation exploration
# on pinpad-easy tasks with multiple seeds

SWEEP_OUTPUT=$(wandb sweep experiments/pinpad-easy-rnd-weight-sweep.yml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent [^ ]*' | awk '{print $NF}')
# Print ID to debug
echo "Detected Sweep ID: $SWEEP_ID"
wandb agent $SWEEP_ID
