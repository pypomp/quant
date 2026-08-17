"""SIR: ABC posterior over (beta1, rho), swept over the tolerance epsilon.

Runs NCHAINS independent ABC-MCMC chains for each epsilon in EPS_GRID, writing
each sweep into results/gpu/eps<epsilon>/.

What the epsilon sweep does and does not show. As epsilon shrinks the ABC
posterior converges to p(theta | s(y_obs)) -- the posterior given the *probes* --
which equals the full posterior only when the probes are sufficient. Mean, sd
and lag-1 autocorrelation are not sufficient for this model, so the limit is a
genuinely different and generally wider distribution than the grid reference.
"Distance to the reference shrinks to zero" would therefore be a false
prediction. What must hold instead: successive epsilons converge to each other,
pypomp and R reach the same limit, and that limit is no tighter than the full
posterior while still covering the truth.

The first entry of EPS_GRID is deliberately enormous. At that tolerance every
proposal is accepted and the ABC posterior must collapse onto the prior -- a
cheap, sharp check that catches sign and normalization errors in the distance.
That arm alone starts in stationarity, its target being the prior box the starts
are drawn from, so it pays no burn-in despite mixing slowly: with the proposal
stepping +/-8 across a 200-wide box its autocorrelation time is ~430, against
~25 for the tight arms.

Cost model: wall-clock is latency-bound, not FLOP-bound. An iteration is 208
observations x NSTEP=20 = 4160 sequential scan steps costing ~0.124s almost
independently of chain count. So M is the only expensive axis; buy ESS with
NCHAINS and keep M at the burn-in floor.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "SIR: ABC posterior over (beta1, rho), swept over tolerance epsilon"
# tags: [bayesian, sir, abc, gpu]
# sbatch_args:
#   job-name: "bayesian abc (pypomp)"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 30GB
#   output: "results/gpu/logs/slurm-%j.out"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:20:00" }
#   2:
#     sbatch_args: { time: "00:30:00" }
#   3:
#     sbatch_args: { time: "00:45:00" }
#   4:
#     sbatch_args: { time: "00:45:00" }
# --- END SLURM CONFIG ---
#
# Level 4 samples for ~25 min across the four arms, which run serially here;
# the rest is headroom for XLA, about 2m30s to build the ABC program on CPU.
#
# Level 2 is the calibration run: it carries the full epsilon grid at reduced M
# so acceptance and autocorrelation can be measured at the tight tolerances,
# which have never been observed -- the old level-4 job died after its first
# arm. If a tight arm mixes far worse than eps=1e6, M_ABC should become a
# per-epsilon tuple sized at ~20x that arm's autocorrelation time.

import os
import sys
import time

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if tests_dir not in sys.path:
    sys.path.append(tests_dir)
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if model_dir not in sys.path:
    sys.path.append(model_dir)

# JAX reads these at import time, so they must be set before it is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

USE_CPU = os.environ.get("USE_CPU", "false").lower() == "true"
if USE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "SLURM_CPUS_PER_TASK" in os.environ:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={os.environ['SLURM_CPUS_PER_TASK']}"
        )

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import model  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from utils import save_run  # noqa: E402

print(jax.devices())
print("Using CPU:", USE_CPU)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running abc at level {RUN_LEVEL}")

#: At 1024 chains and M=3000 the eps=1e6 arm reaches ESS ~5900, above what the
#: old M=200000 at 12 chains delivered. ESS comes from NCHAINS, which is free.
NCHAINS = (2, 8, 256, 1024)[RUN_LEVEL - 1]
M_ABC = (20, 1500, 3000, 3000)[RUN_LEVEL - 1]

#: The scaled distance sums three squared standardized probe differences, so
#: under the true model it sits around chi-square(3) doubled -- typical values
#: of a few. The leading 1e6 is the accept-everything case.
#:
#: Levels 2-4 share one grid: the ladder needs three informative rungs for
#: "successive epsilons converge" to be distinguishable from coincidence, and
#: level 2 must exercise every rung to calibrate it. eps=1.0 is dropped for
#: expected very low acceptance.
EPS_GRID = (
    (1e6,),
    (1e6, 20.0, 5.0, 2.0),
    (1e6, 20.0, 5.0, 2.0),
    (1e6, 20.0, 5.0, 2.0),
)[RUN_LEVEL - 1]

#: See the note in pmcmc/run.py: only beta1 and rho are estimated, and the
#: traces otherwise dominate the record at this chain count.
TRACE_COLS = list(model.FREE) + ["logLik", "log_prior"]
TRACE_THIN = 10

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

scale = model.probe_scale()
print("probe scale:", scale)

out_root = os.path.join("results", "gpu")
os.makedirs(out_root, exist_ok=True)

#: The precondition check: the JAX probes evaluated on the observed series.
#: run.R writes the same three numbers from pomp's own probe functions, and
#: report.qmd compares them before any ABC output is interpreted. pomp's
#: probe_acf carries an n/(n-1) correction that stats::acf does not, so this
#: agreement is measured rather than assumed.
ys = model.load_data()
y_obs = {"reports": jnp.asarray(ys["reports"].to_numpy(), dtype=float)}
probe_values = {name: float(fn(y_obs)) for name, fn in model.PROBES.items()}
print("probe values on the data:", probe_values)
pd.DataFrame(
    {"probe": list(probe_values), "value": list(probe_values.values())}
).to_csv(os.path.join(out_root, "probe_values.csv"), index=False)

key, start_key = jax.random.split(key)
starts = model.sample_starts(NCHAINS, key=start_key)
print(f"{NCHAINS} chains, M={M_ABC}, EPS_GRID={EPS_GRID}")

for eps in EPS_GRID:
    obj = model.sir_pomp(theta=starts)
    key, subkey = jax.random.split(key)

    start = time.time()
    obj.abc(
        M=M_ABC,
        probes=model.PROBES,
        epsilon=eps,
        proposal=model.proposal(),
        scale=scale,
        dprior=model.sir_dprior,
        key=subkey,
    )
    execution_time = time.time() - start

    result = obj.results_history[-1]
    acceptance = np.asarray(result.acceptance_rate, dtype=float)
    print(
        f"eps={eps:g}: {execution_time:.1f}s, "
        f"acceptance {acceptance.min():.3f}-{acceptance.max():.3f}"
    )

    out_dir = os.path.join("results", "gpu", f"eps{eps:g}")
    save_run(
        obj,
        out_dir=out_dir,
        run_config={
            "kind": "abc",
            "model": "sir",
            "RUN_LEVEL": RUN_LEVEL,
            "USE_CPU": USE_CPU,
            "MAIN_SEED": model.MAIN_SEED,
            "NCHAINS": NCHAINS,
            "M": M_ABC,
            "epsilon": eps,
            "probes": list(model.PROBES.keys()),
            "probe_scale": scale,
            "free_params": list(model.FREE),
            "rw_sd_estimation_scale": model.RW_SD,
            "prior_box": {k: list(v) for k, v in model.PRIOR_BOX.items()},
            "execution_time": execution_time,
            "platform": jax.devices()[0].platform,
            "trace_thin": TRACE_THIN,
        },
        execution_time=execution_time,
        trace_cols=TRACE_COLS,
        thin=TRACE_THIN,
    )

    pd.DataFrame(
        {
            "chain": np.arange(len(acceptance)),
            "acceptance_rate": acceptance,
            "epsilon": eps,
            "M": M_ABC,
            "execution_time": execution_time,
        }
    ).to_csv(os.path.join(out_dir, "acceptance.csv"), index=False)

print("done")
