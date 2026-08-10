"""Measles: distribution of IF2 parameter estimates from a global search.

Both halves of the comparison start from the same committed parameter vectors
(../starting_parameters.csv), so the R baseline in results/R/ is searching from
exactly the same points.
"""

# --- SLURM CONFIG ---
# importance: medium
# description: "Measles: distribution of IF2 parameter estimates from a global search"
# tags: [estimation, measles, gpu]
# sbatch_args:
#   job-name: "measles estimation"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "results/gpu/logs/slurm-%j.out"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:04:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "01:00:00" }
#   4:
#     sbatch_args: { time: "02:00:00" }
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

import jax
import model
import numpy as np
from utils import save_run_multi

print(jax.devices())

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running at level {RUN_LEVEL}")

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 10, 100, 100)[RUN_LEVEL - 1]
NSTARTS = (2, 3, 20, 360)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

starts = model.fixed_starts(NSTARTS)

objs = {}
started = time.time()

for unit in model.UNITS:
    print(f"unit {unit}: {NSTARTS} starts, M={NFITR}, J={NP_FITR}", flush=True)
    obj = model.measles(unit, theta=starts)

    key, subkey = jax.random.split(key)
    obj.mif(J=NP_FITR, M=NFITR, key=subkey, rw_sd=model.RW_SD)

    objs[unit] = obj

execution_time = time.time() - started

platform = jax.devices()[0].platform
out_dir = os.path.join("results", platform)

save_run_multi(
    objs,
    out_dir=out_dir,
    run_config={
        "kind": "estimation",
        "model": "measles",
        "RUN_LEVEL": RUN_LEVEL,
        "USE_CPU": USE_CPU,
        "MAIN_SEED": model.MAIN_SEED,
        "UNITS": model.UNITS,
        "NP_FITR": NP_FITR,
        "NFITR": NFITR,
        "NSTARTS": NSTARTS,
        "execution_time": execution_time,
    },
)

print(f"wrote {out_dir}/ (results.csv, traces.csv.gz, timings.csv, latest.json)")
