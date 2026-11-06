"""particles (Chopin): user-defined StateSpaceModel, Bootstrap FK, systematic resampling every step.

usage: python bench_particles.py MODEL J [J ...]
"""

import sys

import numpy as np
import particles
from particles import distributions as dists
from particles import state_space_models as ssm

import common as C

model_name = sys.argv[1]
Js = [int(j) for j in sys.argv[2:]]
y = C.data(model_name)


class EulerSDE(dists.ProbDist):
    """NSTEP Euler-Maruyama substeps from xp (vectorised over particles); rvs only."""

    def __init__(self, xp):
        self.xp = xp

    def rvs(self, size=None):
        z = np.array(self.xp, dtype=float, copy=True)
        if z.ndim == 0:
            z = np.full(size, float(z))
        for _ in range(C.NSTEP):
            z = z + (C.R * (1 - np.exp(z)) - C.SIG**2 / 2) * C.DT \
                + C.SIG * np.sqrt(C.DT) * np.random.normal(size=z.shape)
        return z


class LG(ssm.StateSpaceModel):
    def PX0(self):
        # x at the first observation: x0 ~ N(0,1) pushed one step.
        return dists.Normal(loc=0.0, scale=np.sqrt(C.A**2 + C.SX**2))

    def PX(self, t, xp):
        return dists.Normal(loc=C.A * xp, scale=C.SX)

    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=C.SY)


class SDE(ssm.StateSpaceModel):
    def PX0(self):
        class Init(dists.ProbDist):
            def rvs(self, size=None):
                z0 = C.Z0M + C.Z0S * np.random.normal(size=size)
                return EulerSDE(z0).rvs()
        return Init()

    def PX(self, t, xp):
        return EulerSDE(xp)

    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=C.TAU)


model = LG() if model_name == "lg" else SDE()

for J in Js:
    def run(seed, J=J):
        np.random.seed(seed)
        alg = particles.SMC(fk=ssm.Bootstrap(ssm=model, data=y), N=J, resampling="systematic",
                            ESSrmin=1.0, store_history=False, collect=None, verbose=False)
        alg.run()
        return float(alg.logLt)

    first, med, lls = C.timeit(run, n_rep=3)
    C.record("particles", "systematic", model_name, "cpu", J, first, med, lls)
