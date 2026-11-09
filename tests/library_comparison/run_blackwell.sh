#!/usr/bin/env bash
# The ms PF workload of one model on an RTX Pro 6000 Blackwell (Great Lakes gpu-rtx6000), every
# GPU library, one run per process. Rows go to <model>/results/gpu_rtxpro6000/.
# usage: sbatch run_blackwell.sh <dacca|measles|panel_measles|spx> [smoke]
#SBATCH --job-name=libcmp-gpu
#SBATCH --account=aaronabk1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/gpu-%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
ROOT=$PWD
M=$1
VJ=$ROOT/.venv-jax/bin/python VT=$ROOT/.venv-torch/bin/python VS=$ROOT/.venv-ssm/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONWARNINGS=ignore
case $M in
  spx) A="--J 1000 --nthetas 36 --nreps 24"; L1="--layout nested"; L2="--layout flat" ;;
  dacca) A="--J 5000 --nthetas 36 --nreps 36"; L1="--layout map"; L2="--layout vmap" ;;
  measles) A="--J 5000 --nthetas 36 --nreps 36"; L1="--layout api"; L2="--layout flat" ;;
  panel_measles) A="--J 5000 --nthetas 36 --nreps 36"; L1="--chunk-size 1"; L2="--chunk-size 1" ;;
  *) echo "usage: $0 <dacca|measles|panel_measles|spx> [smoke]"; exit 2 ;;
esac
A="$A --device gpu"
cd "$ROOT/$M"
if [ "$2" = smoke ]; then
  A="$A --T 3 --nthetas 2 --nreps 2"; TO=900
  OUT=$ROOT/results/logs/smoke-$M-${SLURM_JOB_ID:-local}
else
  TO=3600; OUT=$PWD/results/gpu_rtxpro6000
fi
mkdir -p "$OUT"
one() { local f=$1; shift; PFB_RESULTS=$OUT/$f timeout $TO "$@" \
  || echo "{\"failed\": \"$*\", \"rc\": $?}" | tee -a "$OUT/$f"; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
one results.jsonl $VJ bench_pypomp.py $A $L1
[ "$L2" != "$L1" ] && one results.jsonl $VJ bench_pypomp.py $A $L2
PFB_BACKEND=cute PFB_CUTE_RNG=philox one results_pypomp_cute.jsonl $VJ bench_pypomp.py $A $L2
one results.jsonl $VJ bench_pfjax.py $A
PFJAX_FIX_KEYSPLIT=1 one results_pfjax_keyfix.jsonl $VJ bench_pfjax.py $A
one results.jsonl $VS bench_cuthbert.py $A
one results.jsonl $VT bench_pydpf.py $A
cat "$OUT"/*.jsonl | grep -c '"warm_s"' | xargs echo "rows with warm_s:"
if grep -h failed "$OUT"/*.jsonl; then [ "$2" = smoke ] && exit 1; fi
echo GPU_DONE
