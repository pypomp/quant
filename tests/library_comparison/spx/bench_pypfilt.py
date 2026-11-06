"""pypfilt: SPX model + observation model (pypfilt_spx.py) through a scenario TOML; pypfilt.fit,
"deterministic" (systematic) resampling with threshold 1 (every observation), float64, the
model PRNG pypfilt provides (ctx.component["random"]["model"]).

pypfilt does not return a marginal likelihood; the observation model's log_llhd accumulates
log sum_i w_i p(y_n | x_n^i) with the pre-update normalised weights (pypfilt_spx.LOGLIK).
pypfilt never weights its initial state (time 0), so it targets p(y_1..y_T) directly.

One filter per task, each building its own context from the TOML (natural pypfilt usage);
--procs worker processes (OMP/MKL/OPENBLAS threads = 1). --procs 1 runs in-process.

usage: python bench_pypfilt.py --J 1000 --nthetas 36 --nreps 24 --procs 32
"""

import spx_common as C

args = C.parse_args("pypfilt", cpu_only=True)

import io  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import pypfilt  # noqa: E402

import pypfilt_spx as M  # noqa: E402

INP = C.load_inputs(args.T, args.nthetas, args.theta_set)
T = len(INP["ys"])
NAMES = [str(n) for n in INP["param_names"]]
WORK = tempfile.mkdtemp(prefix="pfb-spx-pypfilt-")
OBS_FILE = os.path.join(WORK, "spx-obs.ssv")
with open(OBS_FILE, "w") as f:
    f.write("time value\n")
    for n, v in enumerate(INP["ys"].astype(np.float64), start=1):
        f.write(f"{n} {float(v)!r}\n")


def toml(theta, seed):
    params = "\n".join(f"{k} = {float(v)!r}" for k, v in zip(NAMES, theta))
    return f"""
[components]
model = "pypfilt_spx.SPXModel"
time = "pypfilt.Scalar"
sampler = "pypfilt.sampler.LatinHypercube"
summary = "pypfilt.summary.HDF5"

[time]
start = 0.0
until = {float(T)}
steps_per_unit = 1

[spx]
{params}

[filter]
particles = {args.J}
prng_seed = {seed}
history_window = -1
resample.threshold = 1.0
resample.method = "deterministic"
regularisation.enabled = false

[files]
input_directory = "{WORK}"
output_directory = "{WORK}"
delete_cache_file_before_forecast = true
delete_cache_file_after_forecast = true

[scenario.bench]
name = "bench"

[scenario.bench.observations.y]
model = "pypfilt_spx.SPXObs"
file = "{OBS_FILE}"
"""


def run_one(task):
    theta, seed = task
    (inst,) = list(pypfilt.load_instances(io.StringIO(toml(theta, seed))))
    ctx = inst.build_context()
    M.LOGLIK[0] = 0.0
    pypfilt.fit(ctx, filename=None)
    return M.LOGLIK[0]


tasks = [
    (tuple(float(v) for v in INP["thetas"][i]), 1 + args.seed * 1_000_003 + i * args.nreps + r)
    for i in range(args.nthetas) for r in range(args.nreps)
]

if __name__ == "__main__":
    pool = mp.get_context("fork").Pool(args.procs) if args.procs > 1 else None
    try:
        def run():
            out = pool.map(run_one, tasks, chunksize=1) if pool else list(map(run_one, tasks))
            return np.array(out).reshape(args.nthetas, args.nreps)

        cold, ll = C.timed(run)
        warm, ll2 = C.timed(run)
    finally:
        if pool:
            pool.close()
            pool.join()
        shutil.rmtree(WORK, ignore_errors=True)
    assert np.array_equal(ll, ll2, equal_nan=True)
    C.record(
        args, "pypfilt", T, cold, warm, ll,
        C.versions("pypfilt", "numpy", "scipy"),
        variant="fit/scenario", dtype="float64", resampler="deterministic(systematic)",
        procs=args.procs,
    )
