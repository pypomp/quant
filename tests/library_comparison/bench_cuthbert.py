"""cuthbert: user-defined bootstrap models; cuthbert.smc.particle_filter run by cuthbert.filter under jax.jit.

cuthbert's particle filter draws x0 from init_sample with no weight and then, at each of the
T observations, resamples, propagates and weights, so x0 is drawn at t0 as in pypomp and no
push-forward is needed. Systematic resampling at every step
(cuthbertlib.resampling.systematic.resampling, not wrapped in the ESS decorator).
The loglik is the last log_normalizing_constant of the filter states.
Runs from .venv-ssm (cuthbert is not installed in .venv-jax).
usage: python bench_cuthbert.py MODEL J [J ...]
"""

import sys

import cuthbert
import jax
import jax.numpy as jnp
from cuthbert.smc import particle_filter
from cuthbertlib.resampling import systematic

import common as C

model_name = sys.argv[1]
Js = [int(j) for j in sys.argv[2:]]
device = jax.devices()[0].platform
y = jnp.asarray(C.data(model_name), dtype=jnp.float32)
norm_lpdf = jax.scipy.stats.norm.logpdf

if model_name == "lg":
    def init_sample(key):
        return jax.random.normal(key)

    def propagate_sample(key, x, y_t):
        return C.A * x + C.SX * jax.random.normal(key)

    def log_potential(x_prev, x, y_t):
        return norm_lpdf(y_t, x, C.SY)
else:
    def init_sample(key):
        return C.Z0M + C.Z0S * jax.random.normal(key)

    def propagate_sample(key, z, y_t):
        eps = jax.random.normal(key, (C.NSTEP,))

        def body(z, e):
            return z + (C.R * (1 - jnp.exp(z)) - C.SIG**2 / 2) * C.DT + C.SIG * jnp.sqrt(C.DT) * e, None

        return jax.lax.scan(body, z, eps)[0]

    def log_potential(x_prev, x, y_t):
        return norm_lpdf(y_t, x, C.TAU)

for J in Js:
    pf_obj = particle_filter.build_filter(init_sample, propagate_sample, log_potential,
                                          n_filter_particles=J, resampling_fn=systematic.resampling)

    def pf1(key, pf_obj=pf_obj):
        init_key, filter_key = jax.random.split(key)
        states = cuthbert.filter(pf_obj, y, pf_obj.init_prepare(key=init_key), key=filter_key)
        return states.log_normalizing_constant[-1]

    pf = jax.jit(jax.vmap(pf1))

    def run(seed):
        return float(pf(jax.random.split(jax.random.key(seed), C.REPS)).mean())

    first, med, lls = C.timeit(run)
    C.record("cuthbert", "systematic", model_name, device, J, first, med, lls)
