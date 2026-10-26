"""pypfilt: He10 measles via a scenario TOML (pypfilt_he10.He10Model + He10Obs), pypfilt.fit.

Time is counted in Euler steps (Scalar time, steps_per_unit = 1), observations sit
at the end of each interval, and the model's init() sets the state at t0 (pypomp's
rinit), so pypfilt already targets p(y_1..y_T). Resampling: "deterministic"
(systematic), threshold 1 (every observation), no regularisation. float64, numpy
Generator samplers (pypfilt's "model" PRNG stream). The marginal loglik is
accumulated in He10Obs.log_llhd (pypfilt does not report one).

history_window is finite (default 49 steps): pypfilt's default of keeping the whole
history (-1) would allocate (K+1) x J state records, about 1.8 GB per filter at J=5000.

CPU only: one filter per task, a multiprocessing (fork) pool of --procs workers,
one BLAS/OpenMP thread each.

    python bench_pypfilt.py --J 1000 --nthetas 3 --nreps 10 --procs 8 [--panel]
"""

import io
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import he10 as H  # noqa: E402

H.single_thread_env()


def add_args(p):
    p.add_argument("--history-window", type=int, default=49, help="pypfilt filter.history_window (time units = steps)")


args = H.parse_args("pypfilt", add_args)

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import pypfilt  # noqa: E402

import pypfilt_he10 as M  # noqa: E402

inp = H.Inputs(args.panel, args.T)
WORK = tempfile.mkdtemp(prefix="pfb_pypfilt_he10_")
obs_end = np.cumsum(inp.nstep)
OBS = []
for u in range(inp.U):
    path = os.path.join(WORK, f"obs-unit{u}.ssv")
    with open(path, "w") as f:
        f.write("time value\n")
        for t, v in zip(obs_end, inp.ys[u]):
            f.write(f"{int(t)} {float(v)!r}\n")
    OBS.append(path)


def toml(i, u, seed):
    th = inp.thetas(args.nthetas)[i, u]
    return f"""
[components]
model = "pypfilt_he10.He10Model"
time = "pypfilt.Scalar"
sampler = "pypfilt.sampler.LatinHypercube"
summary = "pypfilt.summary.HDF5"

[time]
start = 0.0
until = {float(inp.K)}
steps_per_unit = 1

[filter]
particles = {args.J}
prng_seed = {seed}
history_window = {args.history_window}
resample.threshold = 1.0
resample.method = "deterministic"
regularisation.enabled = false

[files]
input_directory = "{WORK}"
output_directory = "{WORK}"
delete_cache_file_before_forecast = true
delete_cache_file_after_forecast = true

[he10]
inputs = "{inp.path}"
T = {inp.T}
unit = {u}
theta = [{", ".join(repr(float(v)) for v in th)}]

[scenario.bench]
name = "bench"

[scenario.bench.observations.cases]
model = "pypfilt_he10.He10Obs"
file = "{OBS[u]}"
"""


def run_one(task):
    i, r, u = task
    seed = int(np.random.SeedSequence(H.task_seed(args.seed, i, r, u)).generate_state(1)[0] % 2**31)  # pypfilt wants int32
    (inst,) = list(pypfilt.load_instances(io.StringIO(toml(i, u, seed))))
    ctx = inst.build_context()
    M.LOGLIK[0] = 0.0
    pypfilt.fit(ctx, filename=None)
    return M.LOGLIK[0]


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
        args, "pypfilt", inp, ll, cold, warm, **tm,
        samplers="numpy Generator standard_gamma / poisson / binomial (exact)",
        dtype="float64",
        versions=H.versions("pypfilt", "numpy", "scipy"),
        resampling="deterministic (systematic) every observation, threshold 1",
        history_window=args.history_window,
    )
