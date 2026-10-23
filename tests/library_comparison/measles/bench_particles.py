"""particles (Chopin): He10 measles as a StateSpaceModel; Bootstrap FK, SMC, systematic resampling every step.

Custom ProbDists (vectorized over particles, float64, numpy Generator samplers):
PX0 draws the initial state at t0 (pypomp's rinit) and pushes it through the
first observation interval, so the filter targets p(y_1..y_T) like pypomp; PX(t, xp)
runs the Euler steps of interval t (index into the shared Euler grid); PY is He10's
discretised-normal measurement density. The model maths is he10.np_step / np_dmeas.

CPU only: one filter per task, a multiprocessing (fork) pool of --procs workers,
one BLAS/OpenMP thread each. ESSrmin=1 resamples at every observation.

    python bench_particles.py --J 1000 --nthetas 3 --nreps 10 --procs 8 [--panel]
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import he10 as H  # noqa: E402

H.single_thread_env()
args = H.parse_args("particles")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import particles  # noqa: E402
from particles import distributions as dists  # noqa: E402
from particles import state_space_models as ssm  # noqa: E402

inp = H.Inputs(args.panel, args.T)


class IntervalDist(dists.ProbDist):
    """Law of the state at observation t given the state xp at the previous one (xp None: from rinit at t0)."""

    dim = 6

    def __init__(self, mod, t, xp=None):
        self.mod, self.t, self.xp = mod, t, xp

    def rvs(self, size=None):
        m = self.mod
        if self.xp is None:
            x = H.np_rinit(m.th, m.inp.pop[m.u, 0], size)
        else:
            x = self.xp.T
        return H.np_interval(m.rng, x, self.t, m.th, m.inp, m.u).T  # (N, 6)


class MeasDist(dists.ProbDist):
    def __init__(self, x, th):
        self.x, self.th = x, th

    def logpdf(self, y):
        return H.np_dmeas(y, self.x[:, 5], self.th["rho"], self.th["psi"])


class He10SSM(ssm.StateSpaceModel):
    """kwargs: th (dict of natural-scale params), inp (he10.Inputs), u (unit index), rng (numpy Generator)."""

    def PX0(self):
        return IntervalDist(self, 0)

    def PX(self, t, xp):
        return IntervalDist(self, t, xp)

    def PY(self, t, xp, x):
        return MeasDist(x, self.th)


def run_one(task):
    i, r, u = task
    seed = H.task_seed(args.seed, i, r, u)
    np.random.seed(np.random.SeedSequence(seed).generate_state(1))  # particles' resampling uses np.random
    mod = He10SSM(th=H.np_theta(inp.thetas(args.nthetas)[i, u]), inp=inp, u=u,
                  rng=np.random.default_rng(seed))
    alg = particles.SMC(fk=ssm.Bootstrap(ssm=mod, data=list(inp.ys[u])), N=args.J,
                        resampling="systematic", ESSrmin=1.0, store_history=False,
                        collect=None, verbose=False)
    alg.run()
    return float(alg.logLt)


def run():
    tasks = H.task_list(args.nthetas, args.nreps, inp.U)
    if args.procs <= 1:
        out = [run_one(t) for t in tasks]
    else:
        with mp.get_context("fork").Pool(args.procs) as pool:
            out = pool.map(run_one, tasks, chunksize=1)
    return np.array(out).reshape(args.nthetas, args.nreps, inp.U)


if __name__ == "__main__":
    ll, tm = H.timed_calls(run, args.calls)
    cold, warm = tm.pop("cold_s"), tm.pop("warm_s", None)
    H.record(
        args, "particles", inp, ll, cold, warm, **tm,
        samplers="numpy Generator standard_gamma / poisson / binomial (exact)",
        dtype="float64",
        versions=H.versions("particles", "numpy", "scipy"),
        resampling="systematic every observation (ESSrmin=1)",
    )
