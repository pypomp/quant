"""pypomp He10 measles (London, model 001b, daily Euler steps): pfilter time vs J and replicates.

Reports time per Euler substep, to show where the GPU is latency-bound (flat in J)
vs throughput-bound (linear in J).
usage: python bench_he10.py J [J ...]
"""

import sys

import jax
import jax.numpy as jnp

import common as C
import pypomp as pp
import pypomp.functional as F

THETA = {
    "R0": 56.8, "sigma": 28.9, "gamma": 30.4, "iota": 2.9, "rho": 0.488,
    "sigmaSE": 0.0878, "psi": 0.116, "cohort": 0.557, "amplitude": 0.554,
    "S_0": 2.97e-02, "E_0": 5.17e-05, "I_0": 5.14e-05, "R_0": 9.70e-01,
}
Js = [int(j) for j in sys.argv[1:]]
device = jax.devices()[0].platform
model = pp.models.UKMeasles.pomp(unit="London", theta=pp.PompParameters(THETA), clean=True, model="001b")
struct = model.to_struct()
th_arr = jnp.tile(model.theta.to_jax_array(model.canonical_param_names), (C.REPS, 1))
n_sub = int(struct.nstep_array.sum())

for J in Js:
    def run(seed, J=J):
        keys = jax.random.split(jax.random.key(seed), C.REPS)[:, None]
        out = F.pfilter(struct, th_arr, J, keys)
        return float(jax.block_until_ready(out["logLik"]).mean())

    first, med, lls = C.timeit(run)
    C.record("pypomp", "he10/core", "he10", device, J, first, med, lls,
             n_substeps=n_sub, us_per_substep=round(med / n_sub * 1e6, 2))
