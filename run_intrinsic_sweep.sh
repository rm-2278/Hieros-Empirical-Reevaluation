#!/bin/sh

# Intrinsic Motivation Exploration Sweep for pinpad-easy
# This script runs a wandb sweep to compare different intrinsic motivation methods:
# - Full exploration suite (RND + Count + Hierarchical)
# - Tests on pinpad-easy_three and pinpad-easy_four tasks
# - Multiple seeds for statistical significance

SWEEP_OUTPUT=$(wandb sweep experiments/pinpad-easy-intrinsic-sweep.yml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent [^ ]*' | awk '{print $NF}')
# Print ID to debug
echo "Detected Sweep ID: $SWEEP_ID"
wandb agent $SWEEP_ID
