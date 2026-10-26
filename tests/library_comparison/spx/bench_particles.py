"""particles (Chopin): SPX as a StateSpaceModel; Bootstrap FK, SMC, systematic resampling at
every observation (ESSrmin=1), float64, NumPy's global RNG (what particles' samplers use).

State x = (V, S) as an (N, 2) array. The step into observation t (particles' time index) reads
y_prev = covars_extended[t], the previous return pypomp's rproc reads. particles observes its
initial state, so PX0 draws x0 = (V_0, S0) and applies one step; every library therefore
targets p(y_1..y_T).

One filter per task; --procs worker processes (OMP/MKL/OPENBLAS threads = 1). --procs 1 runs
in-process with no pool.

usage: python bench_particles.py --J 1000 --nthetas 36 --nreps 24 --procs 32
"""

import spx_common as C

args = C.parse_args("particles", cpu_only=True)

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import numpy.random as random  # noqa: E402
import particles  # noqa: E402
from particles import distributions as dists  # noqa: E402
from particles import state_space_models as ssm  # noqa: E402

INP = C.load_inputs(args.T, args.nthetas, args.theta_set)
YS = INP["ys"].astype(np.float64)
Y_PREV = INP["y_prev"].astype(np.float64)
S0 = float(INP["S0"])
VF = float(INP["V_floor"])
NAMES = [str(n) for n in INP["param_names"]]


class Move(dists.ProbDist):
    """One SPX step from xp (N, 2), given the previous return y_prev; rvs only."""

    def __init__(self, xp, y_prev, m):
        self.xp, self.y_prev, self.m = xp, y_prev, m

    def rvs(self, size=None):
        m = self.m
        V, S = self.xp[:, 0], self.xp[:, 1]
        dZ = random.normal(size=V.shape)
        dWs = (self.y_prev - m.mu + 0.5 * V) / np.sqrt(V)
        dWv = m.rho * dWs + np.sqrt(1 - m.rho**2) * dZ
        S = S + S * (m.mu + np.sqrt(np.maximum(V, VF)) * dWs)
        V = V + m.kappa * (m.theta - V) + m.xi * np.sqrt(V) * dWv
        return np.stack([np.maximum(V, VF), S], axis=1)


class Init(dists.ProbDist):
    """x at the first observation: x0 = (V_0, S0) pushed one step with y_prev = covars_extended[0]."""

    def __init__(self, m):
        self.m = m

    def rvs(self, size=None):
        x0 = np.empty((size, 2))
        x0[:, 0], x0[:, 1] = self.m.V_0, S0
        return Move(x0, Y_PREV[0], self.m).rvs()


class SPX(ssm.StateSpaceModel):
    default_params = dict.fromkeys(NAMES)

    def PX0(self):
        return Init(self)

    def PX(self, t, xp):
        return Move(xp, Y_PREV[t], self)

    def PY(self, t, xp, x):
        V = x[:, 0]
        return dists.Normal(loc=self.mu - 0.5 * V, scale=np.sqrt(V))


def run_one(task):
    theta, seed = task
    random.seed(seed)
    model = SPX(**dict(zip(NAMES, theta)))
    alg = particles.SMC(fk=ssm.Bootstrap(ssm=model, data=YS), N=args.J, resampling="systematic",
                        ESSrmin=1.0, store_history=False, collect=None, verbose=False)
    alg.run()
    return float(alg.logLt)


tasks = [
    (tuple(float(v) for v in INP["thetas"][i]), (args.seed * 1_000_003 + i * args.nreps + r) % 2**32)
    for i in range(args.nthetas) for r in range(args.nreps)
]

if __name__ == "__main__":
    pool = mp.get_context("fork").Pool(args.procs) if args.procs > 1 else None

    def run():
        out = pool.map(run_one, tasks, chunksize=1) if pool else list(map(run_one, tasks))
        return np.array(out).reshape(args.nthetas, args.nreps)

    cold, ll = C.timed(run)
    warm, ll2 = C.timed(run)
    if pool:
        pool.close()
        pool.join()
    assert np.array_equal(ll, ll2, equal_nan=True)
    C.record(
        args, "particles", len(YS), cold, warm, ll,
        C.versions("particles", "numpy", "scipy"),
        variant="Bootstrap/SMC", dtype="float64", resampler="systematic", procs=args.procs,
    )
