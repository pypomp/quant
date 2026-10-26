"""The paper's PF on the XLA rproc at 1,296 x 5,000 and at 1,295 x 4,999, where XLA can't put 4 particles in a thread.
usage: pf_odd.py MODEL [MODEL ...]   (run from quant's kernels/ directory)"""
import sys
from copy import deepcopy
import jax, jax.numpy as jnp
import pypomp.functional as F
from cute_paper_bench import load, timed
for name in sys.argv[1:]:
    model, starts, rw_sd, J0, reps = load(name)
    m = deepcopy(model); m.set_rproc_backend("xla"); struct = m.to_struct()
    th_all = jnp.repeat(starts.to_jax_array(model.canonical_param_names), reps, axis=0)
    for drop in (0, 1):
        th, J = th_all[: th_all.shape[0] - drop], J0 - drop
        def pf(seed, th=th, J=J):
            keys = jax.random.split(jax.random.key(seed), th.shape[0])[:, None]
            return F.pfilter(struct, th, J, keys)["logLik"]
        cold, warm, ll = timed(pf)
        print(f"{name} XLA PF {th.shape[0]} filters x J = {J}: warm {warm:.2f} s, cold {cold:.1f} s, median loglik {float(jnp.median(ll)):.2f}", flush=True)
