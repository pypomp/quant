"""Panel measles: distribution of block-IF2 parameter estimates from a global search.

Both halves of the comparison start from the same committed parameter vectors
(../starting_parameters.csv), so the R baseline in results/R/ is searching from
exactly the same points.

After the search the run continues into the panel-specific part of the
workflow -- mix-and-match across units, then a re-evaluation of the single best
combination. R panelPomp has no mix_and_match, so the frame the report compares
against pomp is the one captured before that point: results_final.csv, one row
per start. results.csv is the pruned best afterwards, a pypomp-only result.
"""

# --- SLURM CONFIG ---
# importance: medium
# description: "Panel measles: distribution of block-IF2 parameter estimates from a global search"
# tags: [estimation, panel_measles, gpu]
# sbatch_args:
#   job-name: "panel measles estimation (pypomp)"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "results/gpu/logs/slurm-%j.out"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:05:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "01:00:00" }
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

import jax
import model
import numpy as np
from utils import pfilter_logliks_frame, save_run

print(jax.devices())

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running at level {RUN_LEVEL}")

NP_FITR = (2, 500, 5000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 10, 100, 100)[RUN_LEVEL - 1]
NSTARTS = (2, 3, 36, 360)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 5000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 36, 36)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

starts = model.fixed_starts(NSTARTS)
panel_obj = model.panel_measles(starts)

print(f"{NSTARTS} starts, M={NFITR}, J={NP_FITR}", flush=True)

started = time.time()

key, subkey = jax.random.split(key)
panel_obj.mif(rw_sd=model.RW_SD, M=NFITR, J=NP_FITR, key=subkey, block=True)

key, subkey = jax.random.split(key)
panel_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)

# Everything the comparison against pomp needs is in this entry: one row per
# start with its final parameter vector. Capture it before mix-and-match and
# pruning discard the other starts.
all_starts_index = len(panel_obj.results_history) - 1
results_final = panel_obj.results(index=all_starts_index)
logliks = pfilter_logliks_frame(panel_obj, history_index=all_starts_index)

panel_obj.mix_and_match()
panel_obj.prune(n=1, refill=False)
panel_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)

execution_time = time.time() - started

panel_obj.print_summary()
print(panel_obj.time())

platform = jax.devices()[0].platform
out_dir = os.path.join("results", platform)

save_run(
    panel_obj,
    out_dir=out_dir,
    run_config={
        "kind": "estimation",
        "model": "panel_measles",
        "RUN_LEVEL": RUN_LEVEL,
        "USE_CPU": USE_CPU,
        "MAIN_SEED": model.MAIN_SEED,
        "UNITS": model.UNITS,
        "SHARED_PARAMS": model.SHARED_PARAMS,
        "NP_FITR": NP_FITR,
        "NFITR": NFITR,
        "NSTARTS": NSTARTS,
        "NP_EVAL": NP_EVAL,
        "NREPS_EVAL": NREPS_EVAL,
        "execution_time": execution_time,
    },
    execution_time=execution_time,
)

results_final.to_csv(os.path.join(out_dir, "results_final.csv"), index=False)
logliks.to_csv(os.path.join(out_dir, "pfilter_logliks.csv"), index=False)

print(
    f"wrote {out_dir}/ (results.csv, results_final.csv, pfilter_logliks.csv, "
    "traces.csv.gz, timings.csv, latest.json)"
)
