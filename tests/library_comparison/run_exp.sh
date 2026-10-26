#!/usr/bin/env bash
# Kernel-launch experiments on pypomp (GPU 0): substep unroll, XLA command buffers, PRNG impl.
cd "$(dirname "$0")"
VJ=${VENV_JAX:-.venv-jax}/bin/python VT=${VENV_TORCH:-.venv-torch}/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0} XLA_PYTHON_CLIENT_PREALLOCATE=false \
       JAX_PLATFORMS=cuda PYTHONWARNINGS=ignore PFB_RESULTS=$PWD/results/exp/results.jsonl; mkdir -p results/exp
CB="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN,WHILE,CONDITIONAL --xla_gpu_graph_min_graph_size=1"
one() { timeout 1500 "$@" || echo "{\"failed\": \"$*\", \"tag\": \"$PFB_TAG\", \"rc\": $?}" | tee -a $PFB_RESULTS; }
for cfg in "base||" "unroll|1|" "cmdbuf||$CB" "unroll+cmdbuf|1|$CB" "rbg||"; do
  IFS="|" read -r tag unroll flags <<< "$cfg"
  export PFB_TAG=$tag PYPOMP_SUBSTEP_UNROLL=$unroll XLA_FLAGS="$flags"
  [ "$tag" = rbg ] && export JAX_DEFAULT_PRNG_IMPL=unsafe_rbg || unset JAX_DEFAULT_PRNG_IMPL
  for reps in 1 36; do
    export PFB_REPS=$reps
    grid="1000 10000 100000"; [ $reps != 1 ] && grid="1000 10000"
    for J in $grid; do one $VJ bench_he10.py $J; done
  done
  export PFB_REPS=1
  one $VJ bench_pypomp.py sde 1000 100000
done
echo EXP_DONE
