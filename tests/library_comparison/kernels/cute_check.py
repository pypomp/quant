"""Stock pypomp pfilter vs the same pfilter with the CuTe interval kernel, same keys.

usage: python cute_check.py MODEL R:J [R:J ...]     (MODEL = sde | he10)
"""

import json
import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402
import cute_rproc as cr
import pypomp as pp
import pypomp.functional as F


def sde_model():
    def rinit(theta_, key, covars, t0):
        return {"x": theta_["z0m"] + theta_["z0s"] * jax.random.normal(key)}

    def rproc(X_, theta_, key, covars, t, dt):
        z = X_["x"]
        eps = jax.random.normal(key)
        return {"x": z + (theta_["r"] * (1 - jnp.exp(z)) - theta_["sig"] ** 2 / 2) * dt
                + theta_["sig"] * jnp.sqrt(dt) * eps}

    def dmeas(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(Y_["y"], X_["x"], theta_["tau"])

    ys = pd.DataFrame({"y": C.data("sde")}, index=np.arange(1, C.T + 1, dtype=float))
    theta = {"r": C.R, "sig": C.SIG, "tau": C.TAU, "z0m": C.Z0M, "z0s": C.Z0S}
    return pp.Pomp(ys=ys, theta=pp.PompParameters(theta), statenames=["x"], t0=0.0, rinit=rinit,
                   rproc=rproc, dmeas=dmeas, nstep=C.NSTEP)


def he10_model():
    theta = {"R0": 56.8, "sigma": 28.9, "gamma": 30.4, "iota": 2.9, "rho": 0.488, "sigmaSE": 0.0878,
             "psi": 0.116, "cohort": 0.557, "amplitude": 0.554, "S_0": 2.97e-02, "E_0": 5.17e-05,
             "I_0": 5.14e-05, "R_0": 9.70e-01}
    return pp.models.UKMeasles.pomp(unit="London", theta=pp.PompParameters(theta), clean=True, model="001b")


def timed(f, n=3):
    t0 = time.perf_counter()
    out = jax.block_until_ready(f(0))
    first = time.perf_counter() - t0
    ts = []
    for s in range(1, n + 1):
        t0 = time.perf_counter()
        jax.block_until_ready(f(s))
        ts.append(time.perf_counter() - t0)
    return first, statistics.median(ts), out


def main():
    name = sys.argv[1]
    model = sde_model() if name == "sde" else he10_model()
    t0 = time.perf_counter()
    s_cute = cr.cute_struct(model)
    print(f"traced step in {time.perf_counter() - t0:.1f}s", flush=True)
    s_stock = model.to_struct()
    th1 = model.theta.to_jax_array(model.canonical_param_names)[0]

    # One interval from identical particles and keys, unbatched and vmapped over 2 filters.
    J0 = 1024
    X0 = jax.vmap(lambda k: s_stock.rinit_pf(th1, jax.random.split(k, J0), None if s_stock.covars_extended is None
                  else s_stock.covars_extended[0], s_stock.t0, False))(jax.random.split(jax.random.key(9), 2))
    for i_obs in (0, 40):
        t_idx = int(np.asarray(s_stock.nstep_array)[:i_obs].sum())
        t = float(np.asarray(s_stock.times)[i_obs - 1]) if i_obs else float(s_stock.t0)
        def interval(st, X, k):
            return st.rproc_pf(X, th1, k, s_stock.covars_extended, s_stock.dt_array_extended, t, t_idx,
                               int(np.asarray(s_stock.nstep_array)[i_obs]), s_stock.accumvars, False)[0]
        ks = jax.random.split(jax.random.key(i_obs + 1), 2)
        a = jax.jit(jax.vmap(lambda X, k: interval(s_stock, X, k)))(X0, ks)
        b = jax.jit(jax.vmap(lambda X, k: interval(s_cute, X, k)))(X0, ks)
        c = jax.jit(lambda X, k: interval(s_cute, X, k))(X0[0], ks[0])
        rel = jnp.abs(a - b) / (jnp.abs(a) + 1.0)
        print(f"interval {i_obs}: vmapped max rel |dX| {float(rel.max()):.2e}, "
              f"share of particles differing > 1e-4: {float((rel.max(-1) > 1e-4).mean()):.4f}, "
              f"unbatched vs vmapped max |d| {float(jnp.abs(c - b[0]).max()):.2e}", flush=True)
    out = os.environ.get("PFB_RESULTS", "results_cute.jsonl")
    for shp in sys.argv[2:]:
        R, J = (int(x) for x in shp.split(":"))
        th = jnp.tile(th1, (R, 1))
        res = {}
        for variant, st in (("stock", s_stock), ("cute", s_cute)):
            def f(seed, st=st):
                keys = jax.random.split(jax.random.key(seed), R)[:, None]
                return F.pfilter(st, th, J, keys)["logLik"].ravel()
            first, med, ll = timed(f)
            res[variant] = np.asarray(ll)
            row = dict(model=name, variant=variant, R=R, J=J, first_s=round(first, 2), median_s=round(med, 4),
                       loglik_mean=round(float(ll.mean()), 3), loglik_sd=round(float(ll.std()), 3),
                       load=os.getloadavg()[0], device=jax.devices()[0].platform)
            print(json.dumps(row), flush=True)
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        d = np.abs(res["stock"] - res["cute"])
        print(f"same keys, per-filter |loglik stock - cute|: max {d.max():.4f}, median {np.median(d):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
