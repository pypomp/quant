"""SPX: wall-clock timing of mif and pfilter.

Compared against the frozen R baseline in ../estimation/results/R/timings.csv.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "SPX: wall-clock timing of mif and pfilter (CPU & GPU)"
# tags: [timing, spx, gpu, cpu]
# jobs:
#   gpu:
#     sbatch_args:
#       job-name: "spx timing (gpu)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 6GB
#       time: "00:20:00"
#       output: "results/gpu/logs/slurm-%j.out"
#     run_levels:
#       1:
#         sbatch_args: { time: "00:02:00" }
#       2:
#         sbatch_args: { time: "00:20:00" }
#       3:
#         sbatch_args: { time: "00:30:00" }
#       4:
#         sbatch_args: { time: "00:05:00" }
#   cpu:
#     sbatch_args:
#       job-name: "spx timing (cpu)"
#       partition: standard
#       cpus-per-task: 36
#       mem: 80GB
#       time: "00:30:00"
#       output: "results/cpu/logs/slurm-%j.out"
#     env:
#       USE_CPU: "true"
#     run_levels:
#       1:
#         sbatch_args: { time: "00:02:00" }
#       2:
#         sbatch_args: { time: "00:20:00" }
#       3:
#         sbatch_args: { time: "00:30:00" }
#       4:
#         sbatch_args: { time: "00:10:00" }
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

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import model  # noqa: E402
from utils import run_metadata  # noqa: E402

print(jax.devices())
print("Using CPU:", USE_CPU)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running at level {RUN_LEVEL}")

NP = (2, 1000, 1000, 1000)[RUN_LEVEL - 1]
NFITR = (2, 20, 50, 200)[RUN_LEVEL - 1]
NSTARTS = (2, 3, 3, 36)[RUN_LEVEL - 1]
NREPS = (2, 3, 3, 24)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

key, subkey = jax.random.split(key)
starts = model.sample_starts(NSTARTS, key=subkey)

spx_obj = model.spx()

timings = []


def timed(phase, fn):
    started = time.time()
    fn()
    elapsed = time.time() - started
    timings.append({"phase": phase, "time_seconds": elapsed})
    print(f"  {phase}: {elapsed:.3f} s")
    return elapsed


print("timing:")

key, subkey = jax.random.split(key)
timed(
    "mif",
    lambda: spx_obj.mif(theta=starts, rw_sd=model.RW_SD, M=NFITR, J=NP, key=subkey),
)

# First pfilter call pays JIT compilation; the second does not.
timed("pfilter_cold", lambda: spx_obj.pfilter(J=NP, reps=NREPS))
timed("pfilter_warm", lambda: spx_obj.pfilter(J=NP, reps=NREPS))

platform = jax.devices()[0].platform
out_dir = os.path.join("results", platform)
os.makedirs(out_dir, exist_ok=True)

timings_df = pd.DataFrame(timings)
timings_df.to_csv(os.path.join(out_dir, "timings.csv"), index=False)

record = run_metadata(
    {
        "kind": "timing",
        "model": "spx",
        "RUN_LEVEL": RUN_LEVEL,
        "USE_CPU": USE_CPU,
        "MAIN_SEED": model.MAIN_SEED,
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
