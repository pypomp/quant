#!/usr/bin/env bash
# Three envs: JAX libraries (pypomp, pfjax, particles, pypfilt), PyTorch (pydpf), and cuthbert
# (JAX, same version as .venv-jax, in its own env so .venv-jax's packages stay as they are).
# pypomp comes from PyPI (PYPOMP_SPEC); PYPOMP_SRC=<checkout> installs a checkout instead.
# torch cu129 wheels: the cu130 default needs a CUDA 13 driver; cu129 includes sm_120 (Blackwell).
set -eo pipefail
cd "$(dirname "$0")"
JAX_SPEC=${JAX_SPEC:-"jax[cuda12]==0.11.2"}
PYPOMP_SPEC=${PYPOMP_SPEC:-"pypomp[cute]==1.4.0"}
[ -n "$PYPOMP_SRC" ] && PYPOMP_SPEC="-e $PYPOMP_SRC[cute]"
[ -d pfjax ] || git clone -q https://github.com/mlysy/pfjax.git
uv venv -q --python ${PY:-3.14} .venv-jax
VIRTUAL_ENV=$PWD/.venv-jax uv pip install -q "$JAX_SPEC" $PYPOMP_SPEC -e ./pfjax particles pypfilt scipy pandas
uv venv -q --python ${PY:-3.14} .venv-torch
VIRTUAL_ENV=$PWD/.venv-torch uv pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
VIRTUAL_ENV=$PWD/.venv-torch uv pip install -q pydpf particles
uv venv -q --python ${PY:-3.14} .venv-ssm
VIRTUAL_ENV=$PWD/.venv-ssm uv pip install -q "$JAX_SPEC" cuthbert==0.1.1 cuthbertlib==0.1.1 scipy pandas
# Optional: the bench-only substep-unroll toggle used by run_exp.sh (needs PYPOMP_SRC).
[ -n "$PYPOMP_SRC" ] && git -C "$PYPOMP_SRC" apply --check "$PWD/pypomp_substep_unroll_toggle.patch" 2>/dev/null \
  && git -C "$PYPOMP_SRC" apply "$PWD/pypomp_substep_unroll_toggle.patch" || true
git -C pfjax apply --check ../pfjax_keysplit_fix.patch 2>/dev/null && git -C pfjax apply ../pfjax_keysplit_fix.patch || true
