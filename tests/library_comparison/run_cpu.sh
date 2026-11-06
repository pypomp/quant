#!/usr/bin/env bash
# CPU-only libraries (particles, pypfilt): their best case. Pinned to 4 P-cores.
cd "$(dirname "$0")"
VJ=${VENV_JAX:-.venv-jax}/bin/python VT=${VENV_TORCH:-.venv-torch}/bin/python
export CUDA_VISIBLE_DEVICES= PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PFB_RESULTS=$PWD/results/cpu/results.jsonl PFB_REPS=1
mkdir -p results/cpu
PIN="taskset -c 4-7"
one() { timeout 2400 "$@" || echo "{\"failed\": \"$*\", \"rc\": $?}" | tee -a $PFB_RESULTS; }
for m in lg sde; do
  for J in 1000 10000 100000 1000000; do one $PIN $VJ bench_particles.py $m $J; done
  for J in 1000 10000 100000; do one $PIN $VJ bench_pypfilt.py $m $J; done
done
echo CPU_DONE
