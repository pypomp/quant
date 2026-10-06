"""SIR: PMCMC posterior over (beta1, rho), at two particle counts J.

Runs NCHAINS independent chains from dispersed starts for each J in J_GRID,
writing each into results/gpu/J<J>/. report.qmd compares them with the grid
reference (reference/), with R pomp (run.R), and with each other.

Why two J: PMCMC's stationary distribution does not depend on J (only its
mixing does), so the two posteriors must agree. The low arm is chosen for a
noisy likelihood estimate (sd ~1.3 at J=25, the efficient regime for PMMH), the
high arm for a nearly exact one (sd ~0.14 at J=2000); a sampler that mishandled
the noise would give posteriors that differ between them.

Cost is dominated by M: each iteration is 4160 sequential scan steps, so extra
chains are nearly free while extra iterations are not.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "SIR: PMCMC posterior over (beta1, rho), at two particle counts J"
# tags: [bayesian, sir, pmcmc, gpu]
# sbatch_args:
#   job-name: "bayesian pmcmc (pypomp)"
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
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "00:30:00" }
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
import model
import numpy as np
import pandas as pd
from utils import pfilter_logliks_frame, save_run

print(jax.devices())
print("Using CPU:", USE_CPU)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running pmcmc at level {RUN_LEVEL}")

NCHAINS = (2, 8, 32, 128)[RUN_LEVEL - 1]
M = (20, 1000, 2000, 2000)[RUN_LEVEL - 1]
J_GRID = ((5,), (100,), (25, 2000), (25, 2000))[RUN_LEVEL - 1]

#: pfilter logLik at the true theta, compared against R in report.qmd: the check
#: that the two SIR implementations are the same model.
NP_PRECOND = (10, 500, 2000, 2000)[RUN_LEVEL - 1]
NREPS_PRECOND = (2, 12, 360, 3600)[RUN_LEVEL - 1]
NREPS_NOISE = (2, 12, 24, 100)[RUN_LEVEL - 1]
J_NOISE_GRID = ((5,), (100,), (10, 25, 100, 2000), (10, 25, 100, 2000))[RUN_LEVEL - 1]

TRACE_COLS = list(model.FREE) + ["logLik", "log_prior"]
TRACE_THIN = 10

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

out_root = os.path.join("results", "gpu")
os.makedirs(out_root, exist_ok=True)
for stale in glob.glob(os.path.join(out_root, "J*")):
    shutil.rmtree(stale)

truth_obj = model.sir_pomp(theta=model.params_from_frame(model.theta_frame(1)))
key, pf_key = jax.random.split(key)
pf_start = time.time()
truth_obj.pfilter(J=NP_PRECOND, reps=NREPS_PRECOND, key=pf_key)
precond = pfilter_logliks_frame(truth_obj)
precond["J"] = NP_PRECOND
precond.to_csv(os.path.join(out_root, "pfilter_logliks.csv"), index=False)
print(
    f"precondition pfilter at truth: J={NP_PRECOND} reps={NREPS_PRECOND} "
    f"mean logLik {precond['logLik'].mean():.2f} "
    f"({time.time() - pf_start:.1f}s)"
)

noise_rows = []
for J in J_NOISE_GRID:
    key, nk = jax.random.split(key)
    truth_obj.pfilter(J=J, reps=NREPS_NOISE, key=nk)
    frame = pfilter_logliks_frame(truth_obj)
    frame["J"] = J
    noise_rows.append(frame)
    finite = np.isfinite(frame["logLik"])
    print(
        f"  J={J:>5}: mean logLik {frame['logLik'][finite].mean():9.2f} "
        f"sd {frame['logLik'][finite].std():6.3f} "
        f"({int((~finite).sum())} non-finite of {len(frame)})"
    )
pd.concat(noise_rows, ignore_index=True).to_csv(
    os.path.join(out_root, "pfilter_vs_J.csv"), index=False
)

key, start_key = jax.random.split(key)
starts = model.sample_starts(NCHAINS, key=start_key)
print(f"{NCHAINS} chains, M={M}, J_GRID={J_GRID}")

for J in J_GRID:
    obj = model.sir_pomp(theta=starts)
    key, subkey = jax.random.split(key)

    start = time.time()
    # Private until these tests pass (pypomp c2fbb62).
    obj._pmcmc(J=J, M=M, proposal=model.proposal(), dprior=model.sir_dprior, key=subkey)
    execution_time = time.time() - start

    result = obj.results_history[-1]
    acceptance = np.asarray(result.acceptance_rate, dtype=float)
    print(
        f"J={J}: {execution_time:.1f}s, "
        f"acceptance {acceptance.min():.3f}-{acceptance.max():.3f}"
    )

    out_dir = os.path.join(out_root, f"J{J}")
    save_run(
        obj,
        out_dir=out_dir,
        run_config={
            "kind": "pmcmc",
            "model": "sir",
            "RUN_LEVEL": RUN_LEVEL,
            "USE_CPU": USE_CPU,
            "MAIN_SEED": model.MAIN_SEED,
            "NCHAINS": NCHAINS,
            "M": M,
            "J": J,
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
            "J": J,
            "M": M,
            "execution_time": execution_time,
        }
    ).to_csv(os.path.join(out_dir, "acceptance.csv"), index=False)

print("done")
