"""SPX: distribution of IF2 parameter estimates from a global search.

Kind: estimation -- theta is free and the starting point is what varies across
replicates. The question is where IF2 lands from many random starts, and
whether that distribution matches R's.

This is the expensive SPX kind: cost is starts x iterations x particles. Wall
clock is recorded here as metadata, but it is *not* the timing measurement --
see ../timing/ for that, which is a controlled comparison at fixed, small
replicate counts.

Compared against the frozen R baseline in R_reference/ (360 replicates,
7183 s elapsed / 70 CPU-hours in R). Regenerating that is a deliberate act; see
TESTING.md.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "SPX: distribution of IF2 parameter and likelihood estimates (CPU & GPU)"
# tags: [estimation, spx, gpu, cpu]
# jobs:
#   gpu:
#     sbatch_args:
#       job-name: "spx estimation (gpu)"
#       partition: gpu
#       gpus: "v100:1"
#       cpus-per-gpu: 1
#       mem: 6GB
#       time: "00:04:00"
#       output: "results/gpu/logs/slurm-%j.out"
#   cpu:
#     sbatch_args:
#       job-name: "spx estimation (cpu)"
#       partition: standard
#       cpus-per-task: 36
#       mem: 80GB
#       time: "00:04:00"
#       output: "results/cpu/logs/slurm-%j.out"
#     env:
#       USE_CPU: "true"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:00:30" }
#   2:
#     sbatch_args: { time: "00:04:00" }
#   3:
#     sbatch_args: { time: "02:00:00" }
#   4:
#     sbatch_args: { time: "03:00:00" }
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
USE_CPU = os.environ.get("USE_CPU", "false").lower() == "true"
if USE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "SLURM_CPUS_PER_TASK" in os.environ:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={os.environ['SLURM_CPUS_PER_TASK']}"
        )

import jax  # noqa: E402
import numpy as np  # noqa: E402

import model  # noqa: E402
from utils import save_run  # noqa: E402

print(jax.devices())
print("Using CPU:", USE_CPU)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running at level {RUN_LEVEL}")

NP_FITR = (2, 1000, 1000, 1000)[RUN_LEVEL - 1]
NFITR = (2, 20, 200, 200)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 20, 120 * 3)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 1000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 24)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

key, subkey = jax.random.split(key)
starts = model.sample_starts(NREPS_FITR, key=subkey)

spx_obj = model.spx()

started = time.time()

key, subkey = jax.random.split(key)
spx_obj.mif(theta=starts, rw_sd=model.RW_SD, M=NFITR, J=NP_FITR, key=subkey)
spx_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)

execution_time = time.time() - started

print(spx_obj.results())
print(spx_obj.time())

# Label the output by the device actually used, not by the USE_CPU request:
# a job that asked for a GPU and silently fell back to CPU must not write
# its numbers into results/gpu/.
platform = jax.devices()[0].platform
out_dir = os.path.join("results", platform)
metrics = save_run(
    spx_obj,
    out_dir=out_dir,
    run_config={
        "kind": "estimation",
        "model": "spx",
        "RUN_LEVEL": RUN_LEVEL,
        "USE_CPU": USE_CPU,
        "NP_FITR": NP_FITR,
        "NFITR": NFITR,
        "NREPS_FITR": NREPS_FITR,
        "NP_EVAL": NP_EVAL,
        "NREPS_EVAL": NREPS_EVAL,
        "MAIN_SEED": model.MAIN_SEED,
    },
    execution_time=execution_time,
)

print(f"\nbest logLik: {metrics['loglik']}")
print(f"wrote {out_dir}/ (results.csv, traces.csv.gz, latest.json, history.jsonl)")
