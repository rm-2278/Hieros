#!/bin/sh

# Baseline reproduction for atari100k (This will take a long time like 4-5 days)
SWEEP_OUTPUT=$(wandb sweep experiments/atari100k_sweep.yml)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb: Created sweep with ID: [^ ]*' | awk '{print $7}')
wandb agent $SWEEP_ID