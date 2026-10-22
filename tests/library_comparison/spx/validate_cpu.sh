#!/usr/bin/env bash
# CPU validation at J=1000, full T, NREPS filters per theta, for two theta sets:
#   starts  = the first 3 of the 36 timing starts
#   sun2024 = the Sun (2024) near-MLE vector (rho = -0.74: a sharp test of the leverage term)
# Never touches a GPU. Results: results/validation/results.jsonl; tables: validate.py.
# usage: [SETS="starts sun2024"] [NREPS=24] [PROCS=8] [PIN="nice -n 19 taskset -c 8-31"] ./validate_cpu.sh
cd "$(dirname "$0")"
VJ="$PIN ${VENV_JAX:-$HOME/src/pf-bench/.venv-jax}/bin/python"
VT="$PIN ${VENV_TORCH:-$HOME/src/pf-bench/.venv-torch}/bin/python"
NREPS=${NREPS:-24} PROCS=${PROCS:-8} SETS=${SETS:-starts sun2024}
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONWARNINGS=ignore
export PFB_RESULTS=${PFB_RESULTS:-$PWD/results/validation/results.jsonl}
for SET in $SETS; do
  N=3; [ "$SET" = sun2024 ] && N=1
  A="--J 1000 --nthetas $N --nreps $NREPS --device cpu --theta-set $SET"
  $VJ bench_pypomp.py $A
  $VJ bench_pfjax.py $A
  $VJ bench_pfjax.py $A --resampler systematic
  $VT bench_pydpf.py $A --procs $PROCS
  $VJ bench_particles.py $A --procs $PROCS
  $VJ bench_pypfilt.py $A --procs $PROCS
done
for SET in $SETS; do $VJ validate.py "$PFB_RESULTS" --J 1000 --theta-set "$SET"; echo; done
echo VALIDATION_DONE
