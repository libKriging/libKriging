#!/usr/bin/env bash
# One-shot local GPU comparison: GPyTorch vs libKriging at n > 1000.
#
# Prereqs (built/installed once by the caller, see README.md):
#   - a Python venv with: numpy<2 scipy pandas torch(+cuda) gpytorch
#   - pylibkriging built from this tree with -DENABLE_CUDA_ITERATIVE=ON,
#     on PYTHONPATH, with its libKriging.so on LD_LIBRARY_PATH
#   - CUDA_VISIBLE_DEVICES pointing at the GPU to use (H100 here)
#
# Env knobs:
#   REPEATS (3)  QUICK ("" | --quick)  BUDGET (1800)  BACKENDS (all three)
#   OMP_NUM_THREADS (48, for the libkriging-cpu baseline)
set -euo pipefail
cd "$(dirname "$0")"

: "${REPEATS:=3}"
: "${QUICK:=}"
: "${BUDGET:=1800}"
: "${BACKENDS:=gpytorch,libkriging-gpu,libkriging-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:=48}"

echo "== datasets (repeats=$REPEATS $QUICK) =="
python make_datasets.py --repeats "$REPEATS" $QUICK

echo "== run (backends=$BACKENDS budget=${BUDGET}s) =="
python run_gpu.py --budget "$BUDGET" --backends "$BACKENDS" --out results/gpu.csv

echo "== aggregate =="
python aggregate.py --results results --out results/summary.md
echo
echo "results/summary.md and results/all.csv written."
