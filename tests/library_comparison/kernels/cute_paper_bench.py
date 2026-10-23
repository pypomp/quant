"""The paper's PF and IF2 workloads with pypomp's XLA rproc vs the CuTe backend (Philox).

Needs the pypomp branch cute-rproc on PYTHONPATH and nvidia-cutlass-dsl.
PF: all 36 starting parameter vectors x NREPS replicates in one flat vmap, as
pypomp.functional.pfilter runs them. IF2: Pomp.mif with the quant RW_SD, 36 searches.

usage: python cute_paper_bench.py [--if2-iters M] MODEL [MODEL ...]   (dacca | measles | spx)
"""

import importlib
import json
import os
import sys
import time
from copy import deepcopy

import jax
import jax.numpy as jnp

import pypomp as pp
import pypomp.functional as F

QUANT_TESTS = os.path.expanduser("~/OneDrive/Documents/UMich/Research/pypomp/quant/tests")
sys.path.insert(0, QUANT_TESTS)
OUT = os.environ.get("PFB_RESULTS", "results_cute_paper.jsonl")


def load(name):
    sys.path.insert(0, os.path.join(QUANT_TESTS, name))
    sys.modules.pop("model", None)
    m = importlib.import_module("model")
    sys.path.pop(0)
    if name == "dacca":
        return m.dacca(), m.fixed_starts(36), m.RW_SD, 5000, 36
    if name == "measles":
        starts = m.fixed_starts(36)
        return m.measles("London", starts, model="001b"), starts, m.RW_SD, 5000, 36
    if name == "spx":
        return m.spx(), m.sample_starts(36, jax.random.key(m.MAIN_SEED)), m.RW_SD, 1000, 24
    raise ValueError(name)


def timed(f):
    t0 = time.perf_counter()
    jax.block_until_ready(f(0))
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = jax.block_until_ready(f(1))
    return cold, time.perf_counter() - t0, out


def emit(**row):
    row["load"] = round(os.getloadavg()[0], 1)
    print(json.dumps(row), flush=True)
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def main():
    args = sys.argv[1:]
    iters = 10
    if "--if2-iters" in args:
        i = args.index("--if2-iters")
        iters = int(args[i + 1])
        del args[i:i + 2]
    print("pypomp from", pp.__file__, flush=True)
    for name in args or ["spx", "dacca", "measles"]:
        model, starts, rw_sd, J, reps = load(name)
        th = jnp.repeat(starts.to_jax_array(model.canonical_param_names), reps, axis=0)
        for backend in ("xla", "cute"):
            m = deepcopy(model)
            m.set_rproc_backend(backend)
            struct = m.to_struct()
            used = "cute" if struct.rproc_pf is not m.rproc.mechanics_pf_interp else "xla"

            def pf(seed, struct=struct):
                keys = jax.random.split(jax.random.key(seed), th.shape[0])[:, None]
                return F.pfilter(struct, th, J, keys)["logLik"]

            cold, warm, ll = timed(pf)
            emit(model=name, task="PF", backend=backend, used=used, J=J, filters=int(th.shape[0]),
                 cold_s=round(cold, 2), warm_s=round(warm, 3), loglik_median=float(jnp.median(ll)))

            def if2(seed, m=m):
                m.mif(J=J, M=iters, rw_sd=rw_sd, theta=starts, key=jax.random.key(seed))
                return jnp.asarray(m.results()["logLik"].to_numpy(dtype=float))

            cold, warm, ll = timed(if2)
            emit(model=name, task=f"IF2 M={iters}", backend=backend, used=used, J=J, searches=36,
                 cold_s=round(cold, 2), warm_s=round(warm, 3), loglik_median=float(jnp.nanmedian(ll)))


if __name__ == "__main__":
    main()
