"""Write inputs.npz: everything the five SPX ports consume, taken from pypomp itself.

Data, time grid and covariates come from the PompStruct of pypomp.models.spx()
(as built by tests/spx/model.py). The 36 parameter vectors are the starts drawn
by tests/spx/timing/run.py at RUN_LEVEL 4, reproduced with the same RNG calls:

    key = jax.random.key(MAIN_SEED); np.random.seed(MAIN_SEED)
    key, subkey = jax.random.split(key)
    starts = model.sample_starts(36, key=subkey)

usage: .venv-jax/bin/python export_inputs.py
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

HERE = os.path.dirname(os.path.abspath(__file__))
SPX_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "spx"))
sys.path.insert(0, SPX_DIR)

import importlib  # noqa: E402
import importlib.metadata as md  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

import model  # noqa: E402  (tests/spx/model.py)

spx_module = importlib.import_module("pypomp.models.spx")

NSTARTS = 36  # tests/spx/timing/run.py, RUN_LEVEL 4
S0 = 1105.0  # hard-coded initial price in pypomp.models.spx._rinit

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)
key, subkey = jax.random.split(key)
starts = model.sample_starts(NSTARTS, key=subkey)

pomp = model.spx()
s = pomp.to_struct()
param_names = list(s.param_names)
thetas = np.array(
    [[float(d[n]) for n in param_names] for d in starts.params(as_list=True)], dtype=np.float64
)

ys = np.asarray(s.ys)
cov = np.asarray(s.covars_extended)
nstep = np.asarray(s.nstep_array)
T = ys.shape[0]
assert ys.shape == (T, 1) and cov.shape == (T + 1, 1), (ys.shape, cov.shape)
assert np.all(nstep == 1), "SPX takes one process step per observation"
# The step into observation n reads covars_extended[n]: 0 for n = 0, then y_{n-1}.
assert cov[0, 0] == 0.0 and np.array_equal(cov[1:, 0], ys[:, 0])

# Same rinit as pypomp: deterministic (V_0, S0).
th0 = {n: float(v) for n, v in zip(param_names, thetas[0])}
init = spx_module._rinit(th0, jax.random.key(0), None, 0.0)
assert float(init["S"]) == S0 and float(init["V"]) == th0["V_0"]

feller = 2 * thetas[:, 1] * thetas[:, 2] - thetas[:, 3] ** 2
assert np.all(feller > 0)

# Validation-only extra: the Sun (2024) estimate used by tests/spx/loglik (rho < 0, near the
# MLE), where log-likelihood estimates have small variance and the leverage term matters.
sun = model.SUN2024_THETA.params(as_list=True)[0]
theta_sun2024 = np.array([[float(sun[n]) for n in param_names]], dtype=np.float64)

out = os.path.join(HERE, "inputs.npz")
tmp = out + ".tmp.npz"
np.savez(
    tmp,
    ys=ys[:, 0],
    times=np.asarray(s.times),
    t0=np.float64(s.t0),
    covars_extended=cov[:, 0],
    y_prev=cov[:T, 0],
    dt_array_extended=np.asarray(s.dt_array_extended),
    nstep_array=nstep,
    covar_names=np.array(pomp.covar_names),
    statenames=np.array(s.statenames),
    param_names=np.array(param_names),
    y_names=np.array(s.y_names),
    thetas=thetas,
    theta_sun2024=theta_sun2024,
    S0=np.float64(S0),
    V_floor=np.float64(spx_module._V_FLOOR),
    main_seed=np.int64(model.MAIN_SEED),
    pypomp_version=np.array(md.version("pypomp")),
    jax_version=np.array(md.version("jax")),
)
os.replace(tmp, out)
print(f"wrote {out}")
print(f"T={T} t0={s.t0} times[:3]={np.asarray(s.times)[:3]} dtype ys={ys.dtype} cov={cov.dtype}")
print("params:", param_names, "states:", list(s.statenames), "covars:", list(pomp.covar_names))
np.set_printoptions(precision=4, linewidth=140)
print("thetas[:3]:\n", thetas[:3])
