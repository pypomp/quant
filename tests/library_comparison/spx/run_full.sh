#!/usr/bin/env bash
# Full SPX pfilter workload of tests/spx/timing (36 thetas x 24 filters x J=1000), one library
# per process, one JSON line each in results/<device>/results.jsonl.
#   GPU libraries on one card:        GPU=0 ./run_full.sh gpu      (pypomp nested+flat, pfjax, pydpf)
#   CPU-only libraries:               PROCS=36 ./run_full.sh cpu   (particles, pypfilt)
#   JAX/torch libraries on CPU too:   PROCS=36 ./run_full.sh cpu-all
# PIN="nice -n 19 taskset -c 8-31" prefixes every command (shared box).
cd "$(dirname "$0")"
VJ="$PIN ${VENV_JAX:-$HOME/src/pf-bench/.venv-jax}/bin/python"
VT="$PIN ${VENV_TORCH:-$HOME/src/pf-bench/.venv-torch}/bin/python"
A="--J ${J:-1000} --nthetas 36 --nreps 24"
PROCS=${PROCS:-36}
export PYTHONWARNINGS=ignore
one() { timeout "${TO:-7200}" "$@" || echo "{\"failed\": \"$*\", \"rc\": $?}" | tee -a "$OUT"; }
case "$1" in
  gpu)
    export CUDA_VISIBLE_DEVICES=${GPU:-0} XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
    OUT=${PFB_RESULTS:-$PWD/results/gpu/results.jsonl}; mkdir -p "$(dirname "$OUT")"
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
    one $VJ bench_pypomp.py $A --device gpu
    one $VJ bench_pypomp.py $A --device gpu --layout flat
    one $VJ bench_pfjax.py $A --device gpu
    one $VT bench_pydpf.py $A --device gpu ;;
  cpu|cpu-all)
    export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
    OUT=${PFB_RESULTS:-$PWD/results/cpu/results.jsonl}; mkdir -p "$(dirname "$OUT")"
    one $VJ bench_particles.py $A --procs "$PROCS"
    one $VJ bench_pypfilt.py $A --procs "$PROCS"
    if [ "$1" = cpu-all ]; then
      one $VJ bench_pypomp.py $A
      one $VJ bench_pfjax.py $A
      one $VT bench_pydpf.py $A --procs "$PROCS"
    fi ;;
  *) echo "usage: $0 gpu|cpu|cpu-all"; exit 2 ;;
esac
echo FULL_DONE
