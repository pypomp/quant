"""SIR: reference answers for the PMCMC and ABC tests.

Neither sampler has an analytic target on this model, so this job computes one
for each by brute force, with no Markov chain involved:

* PMCMC target: the posterior. The particle-filter log-likelihood is evaluated
  on a lattice over the prior box; with a flat prior the normalized likelihood
  is the posterior (grid quadrature).
* ABC target: the ABC posterior at tolerance eps, i.e. the prior restricted to
  parameters whose simulated probes land within eps of the observed ones.
  Rejection sampling draws from it exactly: sample theta from the prior,
  simulate once, keep it if the distance is below eps^2.

Both use pypomp's own simulator and filter, so they check the samplers (the
MCMC machinery), not the SIR translation; the R comparisons check the latter.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "SIR: reference posterior (grid) and ABC posterior (rejection) over (beta1, rho)"
# tags: [bayesian, sir, reference, gpu]
# sbatch_args:
#   job-name: "bayesian reference"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 30GB
#   output: "results/gpu/logs/slurm-%j.out"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:10:00" }
#   2:
#     sbatch_args: { time: "00:15:00" }
#   3:
#     sbatch_args: { time: "00:20:00" }
#   4:
#     sbatch_args: { time: "00:30:00" }
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

import jax
import jax.numpy as jnp
import model
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from utils import save_run

print(jax.devices())
print("Using CPU:", USE_CPU)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running reference at level {RUN_LEVEL}")

GRID_N = (4, 20, 40, 60)[RUN_LEVEL - 1]
NP_REF = (10, 500, 2000, 5000)[RUN_LEVEL - 1]
NREPS_REF = (1, 2, 3, 3)[RUN_LEVEL - 1]
CHUNK = (4, 100, 200, 200)[RUN_LEVEL - 1]

N_REJ = (10_000, 1_000_000, 5_000_000, 20_000_000)[RUN_LEVEL - 1]
REJ_CHUNK = (10_000, 250_000, 500_000, 500_000)[RUN_LEVEL - 1]
EPS_KEEP = max(e for e in model.ABC_EPS_LADDER if e < 1e3)

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

out_dir = os.path.join("results", "gpu")
os.makedirs(out_dir, exist_ok=True)

# --- Grid quadrature (PMCMC target) -------------------------------------------

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

grid_time = time.time() - start
print(f"grid complete in {grid_time:.1f}s")

bb, rr = np.meshgrid(beta1_axis, rho_axis, indexing="ij")
grid = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
grid["beta1"] = bb.ravel()
grid["rho"] = rr.ravel()

cell_area = float(np.diff(beta1_axis).mean() * np.diff(rho_axis).mean())
log_post = grid["logLik"].to_numpy()
grid["log_post"] = log_post - (logsumexp(log_post) + np.log(cell_area))
grid["post"] = np.exp(grid["log_post"])
grid[["beta1", "rho", "logLik", "logLik_sd", "log_post", "post"]].to_csv(
    os.path.join(out_dir, "grid_posterior.csv"), index=False
)

# --- Rejection ABC (ABC target) -------------------------------------------------

lo_box = jnp.asarray([model.PRIOR_BOX[p][0] for p in model.FREE])
hi_box = jnp.asarray([model.PRIOR_BOX[p][1] for p in model.FREE])
distance_fn = model.abc_distance_fn(obj)


def draw_and_distance(key):
    k_prior, k_sim = jax.random.split(key)
    free = jax.random.uniform(
        k_prior, (REJ_CHUNK, len(model.FREE)), minval=lo_box, maxval=hi_box
    )
    return free, distance_fn(free, k_sim)


print(f"rejection ABC: {N_REJ:,} prior draws, keeping distance < {EPS_KEEP}^2")
start = time.time()
kept = []
n_drawn = 0
while n_drawn < N_REJ:
    key, subkey = jax.random.split(key)
    free, dist = jax.device_get(draw_and_distance(subkey))
    keep = dist < EPS_KEEP**2
    kept.append(
        pd.DataFrame(
            {
                **{p: free[keep, i] for i, p in enumerate(model.FREE)},
                "distance": dist[keep],
            }
        )
    )
    n_drawn += REJ_CHUNK
    if (n_drawn // REJ_CHUNK) % 20 == 0:
        print(f"  {n_drawn:,} drawn ({time.time() - start:.1f}s)", flush=True)

rejection_time = time.time() - start
rejection = pd.concat(kept, ignore_index=True)
rejection.to_csv(os.path.join(out_dir, "abc_rejection.csv.gz"), index=False)
for eps in model.ABC_EPS_LADDER:
    n_acc = int((rejection["distance"] < eps**2).sum()) if eps <= EPS_KEEP else n_drawn
    print(f"  eps={eps:g}: {n_acc:,} accepted ({n_acc / n_drawn:.2e})")
print(f"rejection complete in {rejection_time:.1f}s")

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
        "N_REJ": n_drawn,
        "EPS_KEEP": EPS_KEEP,
        "free_params": list(model.FREE),
        "prior_box": {k: list(v) for k, v in model.PRIOR_BOX.items()},
        "grid_time": grid_time,
        "rejection_time": rejection_time,
        "execution_time": grid_time + rejection_time,
        "platform": jax.devices()[0].platform,
    },
    execution_time=grid_time + rejection_time,
    write_traces=False,
)

print("done")
