"""pfjax: user-defined bootstrap models; pfjax.particle_filter under jax.jit.

pfjax ships only multinomial (and continuous-relaxation) resamplers, so it is
run with resample_multinomial, every step (its only mode).
usage: python bench_pfjax.py MODEL J [J ...]
"""

import sys
import jax
import jax.numpy as jnp
import pfjax
from pfjax.models.base_model import BaseModel

import common as C

model_name = sys.argv[1]
Js = [int(j) for j in sys.argv[2:]]
device = jax.devices()[0].platform
y = jnp.asarray(C.data(model_name), dtype=jnp.float32)
norm_lpdf = jax.scipy.stats.norm.logpdf


class LG(BaseModel):
    def __init__(self):
        super().__init__(bootstrap=True)

    def state_sample(self, key, x_prev, theta):
        return C.A * x_prev + C.SX * jax.random.normal(key)

    def meas_lpdf(self, y_curr, x_curr, theta):
        return norm_lpdf(y_curr, x_curr, C.SY)

    # x at the first observation time: x0 ~ N(0,1) pushed one step.
    def init_sample(self, key, y_init, theta):
        k1, k2 = jax.random.split(key)
        return self.state_sample(k2, jax.random.normal(k1), theta)

    def init_logw(self, x_init, y_init, theta):
        return self.meas_lpdf(y_init, x_init, theta)

    def pf_init(self, key, y_init, theta):
        x = self.init_sample(key, y_init, theta)
        return x, self.init_logw(x, y_init, theta)


class SDE(LG):
    def state_sample(self, key, x_prev, theta):
        eps = jax.random.normal(key, (C.NSTEP,))

        def body(z, e):
            return z + (C.R * (1 - jnp.exp(z)) - C.SIG**2 / 2) * C.DT + C.SIG * jnp.sqrt(C.DT) * e, None

        return jax.lax.scan(body, x_prev, eps)[0]

    def meas_lpdf(self, y_curr, x_curr, theta):
        return norm_lpdf(y_curr, x_curr, C.TAU)

    def init_sample(self, key, y_init, theta):
        k1, k2 = jax.random.split(key)
        return self.state_sample(k2, C.Z0M + C.Z0S * jax.random.normal(k1), theta)


model = LG() if model_name == "lg" else SDE()
theta = jnp.zeros(1)

for J in Js:
    def pf1(key, J=J):
        out = pfjax.particle_filter(model, key, y, theta, J,
                                    resampler=pfjax.particle_resamplers.resample_multinomial)
        return out["loglik"]

    pf = jax.jit(jax.vmap(pf1))

    def run(seed):
        return float(pf(jax.random.split(jax.random.key(seed), C.REPS)).mean())

    first, med, lls = C.timeit(run)
    C.record("pfjax", "multinomial", model_name, device, J, first, med, lls)
