"""SIR: the reference posterior over (beta1, rho) by grid quadrature.

Neither PMCMC nor ABC has an analytic target on this model, so this job builds
one numerically: evaluate the particle-filter log-likelihood on a lattice over
the prior box, combine replicates with logmeanexp, add the flat log-prior, and
normalize. The result is the yardstick both the pmcmc/ and abc/ reports measure
against.

What this reference can and cannot do is the important part. It runs pypomp's
own particle filter on pypomp's own model, so it is *blind* to errors in the SIR
translation -- a wrong model produces a wrong reference that a wrong PMCMC would
match perfectly. It is the only absolute test that the sampler targets the
posterior implied by this model and prior. Catching model errors is the job of
the R cross-check, and of the pfilter-logLik precondition that gates it.

The work here is embarrassingly parallel, so this saturates the GPU and is
throughput-bound -- the opposite regime from pmcmc/ and abc/, which are
sequential in M and leave the GPU mostly idle.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "SIR: reference posterior over (beta1, rho) by pfilter grid quadrature"
# tags: [bayesian, sir, reference, gpu]
# sbatch_args:
#   job-name: "bayesian reference (grid)"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 30GB
#   output: "results/gpu/logs/slurm-%j.out"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:05:00" }
#   2:
#     sbatch_args: { time: "00:15:00" }
#   3:
#     sbatch_args: { time: "01:00:00" }
#   4:
#     sbatch_args: { time: "00:10:00" }
# --- END SLURM CONFIG ---

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
import model  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from utils import save_run  # noqa: E402

print(jax.devices())
print("Using CPU:", USE_CPU)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running reference grid at level {RUN_LEVEL}")

GRID_N = (4, 20, 40, 60)[RUN_LEVEL - 1]
NP_REF = (10, 500, 2000, 5000)[RUN_LEVEL - 1]
NREPS_REF = (1, 2, 3, 3)[RUN_LEVEL - 1]

#: Grid points evaluated per pfilter call. The whole lattice would fit in
#: memory, but chunking keeps peak usage flat as GRID_N grows and costs only a
#: recompile on the final short chunk.
CHUNK = (4, 100, 200, 200)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

points, beta1_axis, rho_axis = model.grid_frame(GRID_N)
n_points = len(points)
print(f"grid {GRID_N}x{GRID_N} = {n_points} points, J={NP_REF}, reps={NREPS_REF}")

obj = model.sir_pomp(theta=model.params_from_frame(points.iloc[:1]))

start = time.time()

rows = []
for lo in range(0, n_points, CHUNK):
    hi = min(lo + CHUNK, n_points)
    chunk = model.params_from_frame(points.iloc[lo:hi].reset_index(drop=True))
    key, subkey = jax.random.split(key)
    obj.pfilter(J=NP_REF, reps=NREPS_REF, theta=chunk, key=subkey)

    res = obj.results_history[-1]
    logliks = np.asarray(res.payload["logLiks"].values, dtype=float)
    # (n_theta, reps) -> one logmeanexp per grid point, with its MC error.
    logliks = logliks.reshape(hi - lo, -1)
    for i in range(hi - lo):
        v = logliks[i]
        rows.append(
            {
                "index": lo + i,
                "logLik": float(logsumexp(v) - np.log(len(v))),
                "logLik_sd": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
            }
        )
    print(f"  chunk {lo}:{hi} done ({time.time() - start:.1f}s)", flush=True)

execution_time = time.time() - start
print(f"grid complete in {execution_time:.1f}s")

bb, rr = np.meshgrid(beta1_axis, rho_axis, indexing="ij")
grid = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
grid["beta1"] = bb.ravel()
grid["rho"] = rr.ravel()

# Flat prior over the box, so the posterior is the normalized likelihood.
cell_area = float(np.diff(beta1_axis).mean() * np.diff(rho_axis).mean())
log_post = grid["logLik"].to_numpy()
log_norm = logsumexp(log_post) + np.log(cell_area)
grid["log_post"] = log_post - log_norm
grid["post"] = np.exp(grid["log_post"])

out_dir = os.path.join("results", "gpu")
os.makedirs(out_dir, exist_ok=True)
grid[["beta1", "rho", "logLik", "logLik_sd", "log_post", "post"]].to_csv(
    os.path.join(out_dir, "grid_posterior.csv"), index=False
)

save_run(
    obj,
    out_dir=out_dir,
    run_config={
        "kind": "reference",
        "model": "sir",
        "RUN_LEVEL": RUN_LEVEL,
        "USE_CPU": USE_CPU,
        "MAIN_SEED": model.MAIN_SEED,
        "GRID_N": GRID_N,
        "NP_REF": NP_REF,
        "NREPS_REF": NREPS_REF,
        "CHUNK": CHUNK,
        "free_params": list(model.FREE),
        "prior_box": {k: list(v) for k, v in model.PRIOR_BOX.items()},
        "execution_time": execution_time,
        "platform": jax.devices()[0].platform,
    },
    execution_time=execution_time,
    write_traces=False,
)

print("wrote", os.path.join(out_dir, "grid_posterior.csv"))
