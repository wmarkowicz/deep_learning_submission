#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/task_A/ism_window_optimization.py"
FASTA="$ROOT/task_materials/subtaskA.fa"
OUTPUTS="$ROOT/task_A/outputs"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1

for record in seq1_broken seq_2_broken seq_3_broken
do
  "$PYTHON" "$SCRIPT" "$FASTA" \
    --record "$record" \
    --output-dir "$OUTPUTS/${record}_best_regions"
done
