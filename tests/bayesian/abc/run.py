"""SIR: ABC-MCMC posterior over (beta1, rho), down a ladder of tolerances.

Runs NCHAINS chains at each epsilon in model.ABC_EPS_LADDER, writing each arm
into results/gpu/eps<epsilon>/. report.qmd compares every arm with the exact
ABC posterior from rejection sampling (reference/) and with R pomp (run.R).

Each tighter arm starts from the previous arm: candidate states are drawn from
its post-burn-in draws, each is simulated once, and the starts are drawn from
the candidates already within the new tolerance (the selection step of
ABC-SMC). Started from the prior instead, many chains at a tight tolerance
never accept a single move (31% of 1024 at eps=2 in an earlier version of this
test). The starting distribution does not affect what the chains converge to.

The first rung accepts every proposal, so its chains are a random walk over
the prior box; their stationary distribution must be the prior.

Cost is dominated by M: each iteration simulates 4160 sequential steps, so
extra chains are nearly free while extra iterations are not.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "SIR: ABC-MCMC posterior over (beta1, rho) down a tolerance ladder"
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
#     sbatch_args: { time: "00:40:00" }
#   4:
#     sbatch_args: { time: "00:40:00" }
# --- END SLURM CONFIG ---

import glob
import os
import shutil
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
from utils import save_run

print(jax.devices())
print("Using CPU:", USE_CPU)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running abc at level {RUN_LEVEL}")

NCHAINS = (2, 8, 256, 1024)[RUN_LEVEL - 1]
M_PRIOR = (20, 500, 1000, 1000)[RUN_LEVEL - 1]
M_ABC = (20, 1000, 2000, 2000)[RUN_LEVEL - 1]
EPS_LADDER = model.ABC_EPS_LADDER if RUN_LEVEL > 1 else model.ABC_EPS_LADDER[:2]

CAND_FACTOR = 4
BURN_FRAC = 0.5

TRACE_COLS = list(model.FREE) + ["distance"]
TRACE_THIN = 10

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

scale = model.probe_scale()
print("probe scale:", scale)

out_root = os.path.join("results", "gpu")
os.makedirs(out_root, exist_ok=True)
for stale in glob.glob(os.path.join(out_root, "eps*")):
    shutil.rmtree(stale)

#: Compared against pomp's values in report.qmd: different probe values would
#: mean the two languages run different algorithms.
ys = model.load_data()
y_obs = {"reports": jnp.asarray(ys["reports"].to_numpy(), dtype=float)}
probe_values = {name: float(fn(y_obs)) for name, fn in model.PROBES.items()}
print("probe values on the data:", probe_values)
pd.DataFrame(
    {"probe": list(probe_values), "value": list(probe_values.values())}
).to_csv(os.path.join(out_root, "probe_values.csv"), index=False)

distance_fn = model.abc_distance_fn(model.sir_pomp())
rng = np.random.default_rng(model.MAIN_SEED)


def next_starts(obj, M, eps, key):
    tr = obj.traces()
    post = tr[tr["iteration"] > M * BURN_FRAC]
    cand = post.iloc[rng.integers(len(post), size=NCHAINS * CAND_FACTOR)]
    free = cand[list(model.FREE)].to_numpy(dtype=float)
    dist = np.asarray(distance_fn(jnp.asarray(free), key))
    passed = free[dist < eps**2]
    print(f"  starts for eps={eps:g}: {len(passed)} of {len(free)} candidates pass")
    if len(passed) == 0:
        passed = free[np.argsort(dist)[:NCHAINS]]
    pick = passed[rng.choice(len(passed), size=NCHAINS, replace=len(passed) < NCHAINS)]
    overrides = {p: pick[:, i] for i, p in enumerate(model.FREE)}
    return model.params_from_frame(model.theta_frame(NCHAINS, overrides))


key, start_key = jax.random.split(key)
theta = model.sample_starts(NCHAINS, key=start_key)
print(f"{NCHAINS} chains, M_PRIOR={M_PRIOR}, M_ABC={M_ABC}, EPS_LADDER={EPS_LADDER}")

for rung, eps in enumerate(EPS_LADDER):
    M = M_PRIOR if rung == 0 else M_ABC
    obj = model.sir_pomp(theta=theta)
    key, subkey = jax.random.split(key)

    start = time.time()
    # Private until these tests pass (pypomp c2fbb62).
    obj._abc(
        M=M,
        probes=model.PROBES,
        epsilon=eps,
        proposal=model.abc_proposal(),
        scale=scale,
        dprior=model.sir_dprior,
        key=subkey,
    )
    execution_time = time.time() - start
    if rung + 1 < len(EPS_LADDER):
        key, sel_key = jax.random.split(key)
        theta = next_starts(obj, M, EPS_LADDER[rung + 1], sel_key)

    result = obj.results_history[-1]
    acceptance = np.asarray(result.acceptance_rate, dtype=float)
    print(
        f"eps={eps:g}: {execution_time:.1f}s, "
        f"acceptance {acceptance.min():.3f}-{acceptance.max():.3f}, "
        f"{int((acceptance == 0).sum())} chains never moved"
    )

    out_dir = os.path.join(out_root, f"eps{eps:g}")
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
            "M": M,
            "epsilon": eps,
            "rung": rung,
            "probes": list(model.PROBES.keys()),
            "probe_scale": scale,
            "free_params": list(model.FREE),
            "rw_sd_estimation_scale": model.ABC_RW_SD,
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
            "M": M,
            "execution_time": execution_time,
        }
    ).to_csv(os.path.join(out_dir, "acceptance.csv"), index=False)

print("done")
