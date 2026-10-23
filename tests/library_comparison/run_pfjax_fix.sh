#!/usr/bin/env bash
# pfjax with the key-split fix (PFJAX_FIX_KEYSPLIT=1), same grid as run_gpu.sh.
cd "$(dirname "$0")"
VJ=${VENV_JAX:-.venv-jax}/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0} XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
       PYTHONWARNINGS=ignore PFJAX_FIX_KEYSPLIT=1 PFB_RESULTS=$PWD/results/gpu/results.jsonl
mkdir -p results/gpu
one() { timeout 1500 "$@" || echo "{\"failed\": \"$*\", \"rc\": $?}" | tee -a $PFB_RESULTS; }
for reps in 1 100; do
  export PFB_REPS=$reps
  grid="1000 10000 100000 1000000"; [ $reps != 1 ] && grid="1000 10000"
  for m in lg sde; do for J in $grid; do one $VJ bench_pfjax.py $m $J; done; done
done
echo PFJAX_FIX_DONE
