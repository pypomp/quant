#!/usr/bin/env bash
# GPU runs on one card (GPU=<index or UUID>, default 0), one (lib, model, J, reps) per process.
# pypomp, pfjax, cuthbert, pydpf on the toy models; then He10.
# usage: run_gpu.sh [smoke]
cd "$(dirname "$0")"
VJ=${VENV_JAX:-.venv-jax}/bin/python VT=${VENV_TORCH:-.venv-torch}/bin/python VS=${VENV_SSM:-.venv-ssm}/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0} XLA_PYTHON_CLIENT_PREALLOCATE=false \
       JAX_PLATFORMS=cuda PYTHONWARNINGS=ignore
TO=1500
if [ "$1" = smoke ]; then
  export PFB_RESULTS=/tmp/pfb_gpu_smoke.jsonl; rm -f $PFB_RESULTS
  GRID1="2000"; GRIDR="1000"; HEJ="1000"; RR=10
else
  export PFB_RESULTS=$PWD/results/gpu/results.jsonl; mkdir -p results/gpu
  GRID1="1000 10000 100000 1000000"; GRIDR="1000 10000"; HEJ="1000 10000 100000 1000000"; RR=100
fi
one() { timeout $TO "$@" || echo "{\"failed\": \"$*\", \"rc\": $?}" | tee -a $PFB_RESULTS; }
nvidia-smi -i ${GPU:-0} --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
for reps in 1 $RR; do
  export PFB_REPS=$reps
  grid=$GRID1; [ $reps != 1 ] && grid=$GRIDR
  for m in lg sde; do
    for J in $grid; do
      one $VJ bench_pypomp.py $m $J
      one $VJ bench_pfjax.py $m $J
      one $VS bench_cuthbert.py $m $J
      one $VT bench_pydpf.py $m cuda $J
    done
  done
done
for reps in 1 36; do
  export PFB_REPS=$reps
  grid=$HEJ; [ $reps != 1 ] && grid=$GRIDR
  for J in $grid; do one $VJ bench_he10.py $J; done
done
nvidia-smi -i ${GPU:-0} --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
echo GPU_DONE
