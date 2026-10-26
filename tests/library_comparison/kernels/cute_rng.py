"""PF on the paper workloads with CuTe forced, threefry vs Philox, all filters in one batch.
usage: cute_rng.py MODEL [MODEL ...]   (run from quant's kernels/ directory)"""
import sys, warnings
from copy import deepcopy
import jax, jax.numpy as jnp
import pypomp.functional as F
from cute_paper_bench import load, timed

for name in sys.argv[1:]:
    model, starts, _, J, reps = load(name)
    th = jnp.repeat(starts.to_jax_array(model.canonical_param_names), reps, axis=0)
    for rng in ("threefry", "philox"):
        m = deepcopy(model)
        m.set_rproc_backend("cute", rng=rng)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            struct = m.to_struct()
        used = "cute" if struct.rproc_pf is not m.rproc.mechanics_pf_interp else "xla"
        why = next((str(x.message)[:120] for x in w if "XLA" in str(x.message)), "")

        def pf(seed, struct=struct):
            keys = jax.random.split(jax.random.key(seed), th.shape[0])[:, None]
            return F.pfilter(struct, th, J, keys)["logLik"]

        cold, warm, ll = timed(pf)
        print(f"{name} rng={rng} used={used} warm {warm:.2f} s cold {cold:.1f} s median loglik {float(jnp.median(ll)):.2f} {why}", flush=True)
