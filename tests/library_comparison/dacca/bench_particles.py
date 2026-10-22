"""particles (Chopin): dacca as a StateSpaceModel, Bootstrap FK, systematic resampling every step.

float64 numpy. Randomness comes from numpy's global legacy generator (np.random), which is
what particles' own distributions and resamplers draw from; each filter seeds it.
particles weights the initial state, so PX0 draws x(t0) and pushes it through the first
observation interval: the filter targets p(y_1..y_T), like pypomp.
ESSrmin = 1 resamples whenever ESS < N, i.e. at every observation unless all weights are equal.
CPU only: one filter per task on a pool of --procs single-threaded workers.

usage: .venv-jax/bin/python bench_particles.py --J 5000 --nthetas 36 --nreps 36 --procs 32
"""

import dacca_common as DC

if __name__ == "__main__":
    ARGS = DC.parse_args("particles", cpu_only=True)
DC.set_cpu_env(threads=1)

import time  # noqa: E402

import numpy as np  # noqa: E402
import particles  # noqa: E402
from particles import distributions as dists  # noqa: E402
from particles import state_space_models as ssm  # noqa: E402

from dacca_numpy import Dacca  # noqa: E402

NX = len(DC.STATES)


def _to_dict(X):
    return {s: X[:, j] for j, s in enumerate(DC.STATES)}


def _to_array(x):
    return np.stack([x[s] for s in DC.STATES], axis=1)


def _normal(n):
    return np.random.normal(size=n)


class _Interval(dists.ProbDist):
    """X at the end of observation interval i, given X at its start (rvs only)."""

    dim = NX

    def __init__(self, model, i, xp=None):
        self.model, self.i, self.xp = model, i, xp

    def rvs(self, size=None):
        x = self.model.rinit(size) if self.xp is None else _to_dict(self.xp)
        return _to_array(self.model.interval(x, self.i, _normal))


class _Deaths(dists.ProbDist):
    def __init__(self, model, x):
        self.model, self.x = model, x

    def logpdf(self, y):
        return self.model.dmeas(y, self.x[:, 3], self.x[:, 7])


class DaccaSSM(ssm.StateSpaceModel):
    def __init__(self, model):
        self.model = model

    def PX0(self):
        return _Interval(self.model, 0)  # x(t0) ~ rinit, pushed to the first observation

    def PX(self, t, xp):
        return _Interval(self.model, t, xp)

    def PY(self, t, xp, x):
        return _Deaths(self.model, x)


_W = {}


def _init(J, T, theta_set):
    _W["J"], _W["inp"] = J, DC.load_inputs(T, theta_set)


def _one(task):
    i, r, seed = task
    inp = _W["inp"]
    np.random.seed(seed)
    fk = ssm.Bootstrap(ssm=DaccaSSM(Dacca(DC.theta_dict(inp, i), inp)), data=inp["ys"])
    alg = particles.SMC(fk=fk, N=_W["J"], resampling="systematic", ESSrmin=1.0,
                        store_history=False, collect=None, verbose=False)
    t0 = time.perf_counter()
    alg.run()
    return float(alg.logLt), time.perf_counter() - t0


if __name__ == "__main__":
    args = ARGS
    inp = DC.load_inputs(args.T, args.theta_set)
    cold_s, warm_s, ll, per_filter, start_s = DC.run_pool(_init, _one, args, args.T)
    DC.record("particles", args, inp, cold_s, warm_s, ll, DC.versions("particles", "numpy"),
              variant="Bootstrap/systematic", dtype="float64", resampler="systematic (ESSrmin=1)",
              filter_s_mean=round(per_filter, 4), pool_start_s=round(start_s, 3))
