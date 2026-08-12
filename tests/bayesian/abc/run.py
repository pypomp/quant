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
#     sbatch_args: { time: "00:45:00" }
#   3:
#     sbatch_args: { time: "02:30:00" }
#   4:
#     sbatch_args: { time: "08:00:00" }
# --- END SLURM CONFIG ---
#
# Level 1 is compile-bound rather than sampling-bound; XLA took about 2m30s to
# build the ABC program on CPU. Levels 3 and 4 should be recalibrated from a
# level-2 run, since the per-iteration cost behind them is extrapolated.

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

NCHAINS = (2, 4, 8, 12)[RUN_LEVEL - 1]
M_ABC = (20, 5000, 50000, 200000)[RUN_LEVEL - 1]

#: The scaled distance sums three squared standardized probe differences, so
#: under the true model it sits around chi-square(3) doubled -- typical values
#: of a few. The leading 1e6 is the accept-everything case.
EPS_GRID = (
    (1e6,),
    (1e6, 5.0),
    (1e6, 20.0, 5.0, 2.0),
    (1e6, 20.0, 5.0, 2.0, 1.0),
)[RUN_LEVEL - 1]

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
        },
        execution_time=execution_time,
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
