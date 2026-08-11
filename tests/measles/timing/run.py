"""Measles: wall-clock timing of mif and pfilter.

Three pypomp configurations are timed against the R baseline in
results/R/timings.csv: the GPU with pypomp's fast samplers, the GPU with stock
JAX samplers, and 36 CPU cores. The gpu/gpu_jax pair prices `pypomp.random`
specifically; the gpu/cpu pair prices the accelerator.

All four start from the same committed parameter vectors
(../starting_parameters.csv), so every configuration times the same work.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "Measles: wall-clock timing of mif and pfilter (GPU fast / GPU JAX / CPU)"
# tags: [timing, measles, gpu, cpu]
# jobs:
#   gpu:
#     sbatch_args:
#       job-name: "measles timing (gpu)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 6GB
#       output: "results/gpu/logs/slurm-%j.out"
#     env:
#       SAMPLERS: "fast"
#   gpu_jax:
#     sbatch_args:
#       job-name: "measles timing (gpu, jax samplers)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 6GB
#       output: "results/gpu_jax/logs/slurm-%j.out"
#     env:
#       SAMPLERS: "jax"
#   cpu:
#     sbatch_args:
#       job-name: "measles timing (cpu)"
#       partition: standard
#       cpus-per-task: 36
#       mem: 80GB
#       output: "results/cpu/logs/slurm-%j.out"
#     env:
#       SAMPLERS: "fast"
#       USE_CPU: "true"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:04:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "01:00:00" }
#   4:
#     sbatch_args: { time: "03:00:00" }
# --- END SLURM CONFIG ---

import json
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
import pandas as pd
from utils import run_metadata

print(jax.devices())
print("Using CPU:", USE_CPU)

SAMPLERS = os.environ.get("SAMPLERS", "fast").lower()
if SAMPLERS not in ("fast", "jax"):
    raise ValueError(f"SAMPLERS must be 'fast' or 'jax', got {SAMPLERS!r}")

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running {SAMPLERS} samplers at level {RUN_LEVEL}")

NP = (2, 500, 5000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 10, 100, 100)[RUN_LEVEL - 1]
NSTARTS = (2, 3, 36, 36)[RUN_LEVEL - 1]
NREPS = (2, 3, 36, 36)[RUN_LEVEL - 1]

UNIT = "London"

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

starts = model.fixed_starts(NSTARTS)

if SAMPLERS == "jax":
    measles_obj = model.measles_jax(starts, unit=UNIT)
else:
    measles_obj = model.measles(UNIT, theta=starts)

timings = []


def timed(phase, fn):
    started = time.time()
    fn()
    elapsed = time.time() - started
    timings.append({"phase": phase, "time_seconds": elapsed})
    print(f"  {phase}: {elapsed:.3f} s", flush=True)
    return elapsed


print("timing:")

key, subkey = jax.random.split(key)
timed(
    "mif",
    lambda: measles_obj.mif(theta=starts, rw_sd=model.RW_SD, M=NFITR, J=NP, key=subkey),
)

# First pfilter call pays JIT compilation; the second does not.
key, subkey = jax.random.split(key)
timed(
    "pfilter_cold",
    lambda: measles_obj.pfilter(J=NP, reps=NREPS, key=subkey),
)
timed("pfilter_warm", lambda: measles_obj.pfilter(J=NP, reps=NREPS))

platform = jax.devices()[0].platform
out_dir = os.path.join("results", f"{platform}_jax" if SAMPLERS == "jax" else platform)
os.makedirs(out_dir, exist_ok=True)

timings_df = pd.DataFrame(timings)
timings_df.to_csv(os.path.join(out_dir, "timings.csv"), index=False)

record = run_metadata(
    {
        "kind": "timing",
        "model": "measles",
        "RUN_LEVEL": RUN_LEVEL,
        "USE_CPU": USE_CPU,
        "SAMPLERS": SAMPLERS,
        "MAIN_SEED": model.MAIN_SEED,
        "UNIT": UNIT,
        "NP": NP,
        "NFITR": NFITR,
        "NSTARTS": NSTARTS,
        "NREPS": NREPS,
    }
)
record["timings"] = {t["phase"]: t["time_seconds"] for t in timings}

with open(os.path.join(out_dir, "latest.json"), "w") as f:
    json.dump(record, f, indent=2, default=str)
    f.write("\n")

print(f"\n{timings_df.to_string(index=False)}")
print(f"wrote {out_dir}/ (timings.csv, latest.json)")
