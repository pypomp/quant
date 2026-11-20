#!/usr/bin/env bash
# pfjax and cuthbert on London and the panel with pypomp's fast samplers (--samplers pypomp), on one
# RTX Pro 6000 Blackwell. Installs pypomp into .venv-ssm first if it is missing.
# usage: sbatch run_fast_samplers.sh smoke   compile check (T = 5) + full-width probe (T = 20) -> results/logs/
#        sbatch run_fast_samplers.sh full    full workloads + GPU validation -> <model>/results/gpu_rtxpro6000/
#SBATCH --job-name=libcmp-fast-samplers
#SBATCH --account=aaronabk1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/gpu-%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
ROOT=$PWD
VJ=$ROOT/.venv-jax/bin/python VS=$ROOT/.venv-ssm/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONWARNINGS=ignore
if ! $VS -c "import pypomp" 2>/dev/null; then
  V=$($VJ -c "import pypomp; print(pypomp.__version__)")
  VIRTUAL_ENV=$ROOT/.venv-ssm uv pip install -q "jax[cuda12]==$($VJ -c 'import jax; print(jax.__version__)')" "pypomp==$V" || exit 1
fi
$VS -c "import jax, pypomp, cuthbert; print('ssm: jax', jax.__version__, 'pypomp', pypomp.__version__)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
W="--J 5000 --nthetas 36 --nreps 36 --device gpu --samplers pypomp"
one() { local out=$1 to=$2; shift 2; PFB_RESULTS=$out timeout $to "$@" \
  || echo "{\"failed\": \"$*\", \"rc\": $?}" | tee -a "$out"; }
runs() {  # $1 = results file prefix, $2 = timeout, rest = extra args
  local o=$1 to=$2; shift 2
  one ${o}_pfjax.jsonl $to $VJ bench_pfjax.py "$@"
  PFJAX_FIX_KEYSPLIT=1 one ${o}_pfjax_keyfix.jsonl $to $VJ bench_pfjax.py "$@"
  one ${o}_cuthbert.jsonl $to $VS bench_cuthbert.py "$@"
}
for M in measles panel_measles; do
  cd "$ROOT/$M"
  if [ "$1" = smoke ]; then
    OUT=$ROOT/results/logs/fast-samplers-smoke-${SLURM_JOB_ID:-local}/$M; mkdir -p "$OUT"
    runs $OUT/compile 1200 --J 50 --nthetas 2 --nreps 2 --T 5 --device gpu --samplers pypomp
    runs $OUT/probe_T20 1200 $W --T 20
  elif [ "$1" = full ]; then
    OUT=$PWD/results/gpu_rtxpro6000
    runs $OUT/results_fastsamplers 3600 $W
    runs $OUT/validation_fastsamplers 1800 --J 1000 --nthetas 3 --nreps 100 --device gpu --samplers pypomp --tag validation_n100
  else
    echo "usage: $0 <smoke|full>"; exit 2
  fi
done
grep -rh failed "$ROOT"/results/logs/fast-samplers-smoke-${SLURM_JOB_ID:-local} "$ROOT"/*/results/gpu_rtxpro6000/*fastsamplers* 2>/dev/null
echo FAST_SAMPLERS_DONE
