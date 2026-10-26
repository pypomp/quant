#!/usr/bin/env bash
# Stand-in ptxas: save each PTX input and its ptxas -v report to $PTXAS_SAVE, then run the real ptxas unchanged.
REAL=/home/kevin/src/pf-bench/.venv-cute/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas
D=${PTXAS_SAVE:-/tmp/ptxas_save}; mkdir -p "$D"
IN=$1
if [ -f "$IN" ]; then
  n=$(date +%s%N)
  cp "$IN" "$D/$n.ptx"
  "$REAL" -v -arch=sm_86 "$IN" -o "$D/$n.cubin" 2> "$D/$n.ptxas.txt"
fi
exec "$REAL" "$@"
