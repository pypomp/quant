#!/usr/bin/env bash
# CPU correctness smoke test of every adapter at small J.
cd "$(dirname "$0")"
VJ=${VENV_JAX:-.venv-jax}/bin/python VT=${VENV_TORCH:-.venv-torch}/bin/python
export PFB_RESULTS=/tmp/pfb_smoke.jsonl CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONWARNINGS=ignore
rm -f $PFB_RESULTS
for m in ${@:-lg sde}; do
  $VJ bench_pypomp.py $m 2000
  $VJ bench_pfjax.py $m 2000
  $VJ bench_particles.py $m 2000
  $VJ bench_pypfilt.py $m 2000
  $VT bench_pydpf.py $m cpu 2000
done
echo SMOKE_DONE
