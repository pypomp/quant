"""Panel measles: distribution of the pfilter log-likelihood at the He et al. estimates.

Nothing varies across replicates except the random seed, so any systematic
difference from the pomp baseline in results/R/pfilter_logliks.csv is a
difference between the implementations rather than between the models.
"""

# --- SLURM CONFIG ---
# importance: medium
# description: "Panel measles: distribution of pfilter logLik at the He et al. (2010) estimates"
# tags: [loglik, panel_measles, gpu]
# sbatch_args:
#   job-name: "panel measles loglik (pypomp)"
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
#     sbatch_args: { time: "00:15:00" }
#   3:
#     sbatch_args: { time: "00:15:00" }
#   4:
#     sbatch_args: { time: "08:00:00" }
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

NP_EVAL = (2, 1000, 5000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 300, 300, 3600)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

panel_obj = model.panel_measles(model.mle_theta())

print(f"{NREPS_EVAL} panel pfilters with J={NP_EVAL}", flush=True)

started = time.time()
key, subkey = jax.random.split(key)
panel_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
execution_time = time.time() - started

print(f"pfilter completed in {execution_time:.2f} s")

platform = jax.devices()[0].platform
out_dir = os.path.join("results", platform)

save_run(
    panel_obj,
    out_dir=out_dir,
    run_config={
        "kind": "loglik",
        "model": "panel_measles",
        "RUN_LEVEL": RUN_LEVEL,
        "USE_CPU": USE_CPU,
        "MAIN_SEED": model.MAIN_SEED,
        "UNITS": model.UNITS,
        "SHARED_PARAMS": model.SHARED_PARAMS,
        "NP_EVAL": NP_EVAL,
        "NREPS_EVAL": NREPS_EVAL,
        "execution_time": execution_time,
    },
    execution_time=execution_time,
    write_traces=False,
)

logliks = pfilter_logliks_frame(panel_obj)
logliks.to_csv(os.path.join(out_dir, "pfilter_logliks.csv"), index=False)

print(f"\n{logliks.groupby('unit')['logLik'].describe().to_string()}")
print(f"wrote {out_dir}/ (pfilter_logliks.csv, results.csv, timings.csv, latest.json)")
