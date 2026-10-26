"""pypomp: user-defined models through the public API.

variants: default (per-particle rproc, pypomp vmaps it), vectorized (@pp.vectorized rproc)
timing:   api  = Pomp.pfilter(...) as a user calls it
          core = pypomp.functional.pfilter on the struct (no OO bookkeeping)
usage: python bench_pypomp.py MODEL J [J ...]
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import common as C
import pypomp as pp
import pypomp.functional as F

model_name = sys.argv[1]
Js = [int(j) for j in sys.argv[2:]]
device = jax.devices()[0].platform
y = C.data(model_name)
ys = pd.DataFrame({"y": y}, index=np.arange(1, C.T + 1, dtype=float))

norm_lpdf = jax.scipy.stats.norm.logpdf

if model_name == "lg":
    theta = {"a": C.A, "sx": C.SX, "sy": C.SY}
    nstep = 1

    def rinit(theta_, key, covars, t0):
        return {"x": jax.random.normal(key)}

    def rproc(X_, theta_, key, covars, t, dt):
        return {"x": theta_["a"] * X_["x"] + theta_["sx"] * jax.random.normal(key)}

    @pp.vectorized
    def rproc_vec(X_, theta_, key, covars, t, dt):
        x = X_["x"]
        return {"x": theta_["a"] * x + theta_["sx"] * jax.random.normal(key, x.shape)}

    def dmeas(Y_, X_, theta_, covars, t):
        return norm_lpdf(Y_["y"], X_["x"], theta_["sy"])

else:
    theta = {"r": C.R, "sig": C.SIG, "tau": C.TAU, "z0m": C.Z0M, "z0s": C.Z0S}
    nstep = C.NSTEP

    def rinit(theta_, key, covars, t0):
        return {"x": theta_["z0m"] + theta_["z0s"] * jax.random.normal(key)}

    def _drift(z, th):
        return th["r"] * (1 - jnp.exp(z)) - th["sig"] ** 2 / 2

    def rproc(X_, theta_, key, covars, t, dt):
        z = X_["x"]
        eps = jax.random.normal(key)
        return {"x": z + _drift(z, theta_) * dt + theta_["sig"] * jnp.sqrt(dt) * eps}

    @pp.vectorized
    def rproc_vec(X_, theta_, key, covars, t, dt):
        z = X_["x"]
        eps = jax.random.normal(key, z.shape)
        return {"x": z + _drift(z, theta_) * dt + theta_["sig"] * jnp.sqrt(dt) * eps}

    def dmeas(Y_, X_, theta_, covars, t):
        return norm_lpdf(Y_["y"], X_["x"], theta_["tau"])


for variant, rp in (("default", rproc), ("vectorized", rproc_vec)):
    model = pp.Pomp(
        ys=ys, theta=pp.PompParameters(theta), statenames=["x"], t0=0.0,
        rinit=rinit, rproc=rp, dmeas=dmeas, nstep=nstep,
    )
    struct = model.to_struct()
    th_arr = jnp.tile(model.theta.to_jax_array(model.canonical_param_names), (C.REPS, 1))
    for J in Js:
        def run_api(seed, J=J, model=model):
            model.pfilter(J=J, key=jax.random.key(seed))
            return float(model.results()["logLik"].iloc[-1])

        def run_core(seed, J=J):
            keys = jax.random.split(jax.random.key(seed), C.REPS)[:, None]
            out = F.pfilter(struct, th_arr, J, keys)
            return float(jax.block_until_ready(out["logLik"]).mean())

        runs = (("core", run_core), ("api", run_api)) if C.REPS == 1 else (("core", run_core),)
        for timing, run in runs:
            first, med, lls = C.timeit(run)
            C.record("pypomp", f"{variant}/{timing}", model_name, device, J, first, med, lls)
