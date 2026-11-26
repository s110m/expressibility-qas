#!/bin/bash
# train_cell.sh with a random start delay: several trainings initializing CUDA
# in the same second crashed with "CUDA error: unknown error" (2026-10-19 17:05).
sleep $((RANDOM % 150))
exec /c/Neural-Networks-Files/tq_mapping_wt/genome_ablation/train_cell.sh "$@"
