#!/usr/bin/env bash
# Three envs: JAX libraries (pypomp, pfjax, particles, pypfilt), PyTorch (pydpf), and cuthbert
# (JAX, same version as .venv-jax, in its own env so .venv-jax's packages stay as they are).
# torch cu129 wheels: the cu130 default needs a CUDA 13 driver.
set -eo pipefail
cd "$(dirname "$0")"
[ -d pypomp ] || git clone -q https://github.com/pypomp/pypomp.git
[ -d pfjax ] || git clone -q https://github.com/mlysy/pfjax.git
uv venv -q --python 3.12 .venv-jax
VIRTUAL_ENV=$PWD/.venv-jax uv pip install -q "jax[cuda12]" -e ./pypomp -e ./pfjax particles pypfilt scipy pandas
uv venv -q --python 3.12 .venv-torch
VIRTUAL_ENV=$PWD/.venv-torch uv pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
VIRTUAL_ENV=$PWD/.venv-torch uv pip install -q pydpf particles
uv venv -q --python 3.12 .venv-ssm
VIRTUAL_ENV=$PWD/.venv-ssm uv pip install -q "jax[cuda12]==0.12.0.1" cuthbert==0.1.1 cuthbertlib==0.1.1 scipy pandas
# Optional: the bench-only substep-unroll toggle used by run_exp.sh.
git -C pypomp apply --check ../pypomp_substep_unroll_toggle.patch 2>/dev/null && git -C pypomp apply ../pypomp_substep_unroll_toggle.patch || true
git -C pfjax apply --check ../pfjax_keysplit_fix.patch 2>/dev/null && git -C pfjax apply ../pfjax_keysplit_fix.patch || true
