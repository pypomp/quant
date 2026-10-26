"""A PF on the XLA rproc at its usual size and with one filter and one particle fewer, where the element
count is odd and XLA can't put 4 particles in a thread. The paper's models (dacca, measles, spx) use the
paper's workloads; pypomp's built-in sir and lg run 1,296 filters x 5,000 particles at their default
parameters. usage: pf_odd.py MODEL [MODEL ...]   (run from quant's kernels/ directory)"""
import sys
from copy import deepcopy
import jax, jax.numpy as jnp
import pypomp as pp
import pypomp.functional as F
from cute_paper_bench import load, timed

BUILTIN = {"sir": lambda: pp.models.sir(), "lg": lambda: pp.models.lg(T=100)}


def build(name):
    """Return the model, the parameter set of every filter, and J."""
    if name in BUILTIN:
        model = BUILTIN[name]()
        th = model.theta.to_jax_array(model.canonical_param_names)[:1]
        return model, jnp.repeat(th, 1296, axis=0), 5000
    model, starts, _, J, reps = load(name)
    return model, jnp.repeat(starts.to_jax_array(model.canonical_param_names), reps, axis=0), J


for name in sys.argv[1:]:
    model, th_all, J0 = build(name)
    m = deepcopy(model); m.set_rproc_backend("xla"); struct = m.to_struct()
    for drop in (0, 1):
        th, J = th_all[: th_all.shape[0] - drop], J0 - drop
        def pf(seed, th=th, J=J):
            keys = jax.random.split(jax.random.key(seed), th.shape[0])[:, None]
            return F.pfilter(struct, th, J, keys)["logLik"]
        cold, warm, ll = timed(pf)
        print(f"{name} XLA PF {th.shape[0]} filters x J = {J}: warm {warm:.2f} s, cold {cold:.1f} s, median loglik {float(jnp.median(ll)):.2f}", flush=True)
