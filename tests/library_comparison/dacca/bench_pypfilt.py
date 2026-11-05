"""pypfilt: dacca as a pypfilt Model + Univariate obs model via a scenario TOML.

float64 numpy, numpy Generators seeded by pypfilt (prng_seed). Resampling: "deterministic"
(systematic) whenever ESS < N (threshold 1), i.e. at every observation unless all weights are
equal. pypfilt starts from x(t0) and steps it to each observation before weighting, so it
targets p(y_1..y_T) without any adjustment. pypfilt has no marginal likelihood output; the
obs model's log_llhd hook accumulates it (pypfilt_dacca.LOGLIK).
--steps ministeps (default): one pypfilt time-step per observation, the 20 Euler steps as
  pypfilt @ministeps; the whole history (T + 1 rows) is kept (history_window = -1).
--steps euler: one pypfilt time-step per Euler step (steps_per_unit = 20), as the toy adapters do.
  The full history would be 12,001 rows x J particles (~5 GB at J = 5000), so it keeps a sliding
  window (--history-window, in observation intervals). pypfilt rescans every time-step of the run
  each time the window slides (state.History.summary_window), so this mode costs O(T^2 / window)
  on top of the filter; at window = 1 that dominated (about 40 s per filter at J = 1000).
CPU only: one filter per task on a pool of --procs single-threaded workers.

usage: .venv-jax/bin/python bench_pypfilt.py --J 5000 --nthetas 36 --nreps 36 --procs 32
"""

import dacca_common as DC


def _extra(ap):
    ap.add_argument("--steps", choices=["ministeps", "euler"], default="ministeps")
    ap.add_argument("--history-window", type=int, default=None,
                    help="pypfilt filter.history_window (default: -1 for ministeps, 30 for euler)")


if __name__ == "__main__":
    ARGS = DC.parse_args("pypfilt", cpu_only=True, extra=_extra)
DC.set_cpu_env(threads=1)

import io  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pypfilt  # noqa: E402

import pypfilt_dacca as M  # noqa: E402


def write_obs(inp):
    path = os.path.join(DC.HERE, f"pypfilt-dacca-obs-T{inp['T']}.ssv")
    fd, tmp = tempfile.mkstemp(dir=DC.HERE, suffix=".ssv")
    with os.fdopen(fd, "w") as f:
        f.write("time value\n")
        for t, v in enumerate(inp["ys"], start=1):
            f.write(f"{t} {float(v)!r}\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
    return path


def toml(J, seed, theta, T, nstep, obs_file, steps, window):
    comp, spu, mini = ("pypfilt_dacca.DaccaModel", nstep, "") if steps == "euler" else \
        ("pypfilt_dacca.DaccaModelMini", 1, f"mini_steps_per_step = {nstep}")
    th = "\n".join(f"{k} = {v!r}" for k, v in theta.items())
    outdir = tempfile.gettempdir()
    return f"""
[components]
model = "{comp}"
time = "pypfilt.Scalar"
sampler = "pypfilt.sampler.LatinHypercube"
summary = "pypfilt.summary.HDF5"

[time]
start = 0.0
until = {float(T)}
steps_per_unit = {spu}
{mini}

[filter]
particles = {J}
prng_seed = {seed}
history_window = {window}
resample.threshold = 1.0
resample.method = "deterministic"
regularisation.enabled = false

[files]
input_directory = "{DC.HERE}"
output_directory = "{outdir}"
delete_cache_file_before_forecast = true
delete_cache_file_after_forecast = true

[dacca]
T = {T}

[dacca.theta]
{th}

[scenario.bench]
name = "bench"

[scenario.bench.observations.deaths]
model = "pypfilt_dacca.DeathsObs"
file = "{obs_file}"
"""


_W = {}


def _init(J, T, theta_set):
    _W["J"], _W["inp"] = J, DC.load_inputs(T, theta_set)
    _W["obs"] = os.path.join(DC.HERE, f"pypfilt-dacca-obs-T{_W['inp']['T']}.ssv")


def _one(task):
    i, r, seed = task
    inp = _W["inp"]
    cfg = toml(_W["J"], seed, DC.theta_dict(inp, i), inp["T"], inp["nstep_const"], _W["obs"], _W["steps"],
               _W["window"])
    (inst,) = list(pypfilt.load_instances(io.StringIO(cfg)))
    ctx = inst.build_context()
    M.LOGLIK[0] = 0.0
    t0 = time.perf_counter()
    pypfilt.fit(ctx, filename=None)
    return M.LOGLIK[0], time.perf_counter() - t0


if __name__ == "__main__":
    args = ARGS
    _W["steps"] = args.steps  # inherited by the forked workers
    _W["window"] = args.history_window if args.history_window is not None else \
        (-1 if args.steps == "ministeps" else 30)
    inp = DC.load_inputs(args.T, args.theta_set)
    write_obs(inp)
    cold_s, warm_s, ll, per_filter, start_s = DC.run_pool(_init, _one, args, args.T)
    DC.record("pypfilt", args, inp, cold_s, warm_s, ll, DC.versions("pypfilt", "numpy", "scipy"),
              variant=f"deterministic/{args.steps}", dtype="float64", history_window=_W["window"],
              resampler="deterministic=systematic (threshold 1)",
              filter_s_mean=round(per_filter, 4), pool_start_s=round(start_s, 3))
