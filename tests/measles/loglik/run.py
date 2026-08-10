"""Measles: distribution of the pfilter log-likelihood at the He et al. estimates.

Run twice, once in single and once in double precision, because measles is the
model where 32-bit accumulation has been suspected of shifting the likelihood.
Compared against results/R/pfilter_logliks.csv, which pomp computes in double.
"""

# --- SLURM CONFIG ---
# importance: medium
# description: "Measles: distribution of pfilter logLik at the He et al. (2010) estimates, 32- vs 64-bit"
# tags: [loglik, measles, gpu]
# sbatch_args:
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 6GB
# jobs:
#   f32:
#     sbatch_args:
#       job-name: "measles loglik (32-bit)"
#       output: "results/f32/logs/slurm-%j.out"
#     env:
#       USE_64BIT: "false"
#   f64:
#     sbatch_args:
#       job-name: "measles loglik (64-bit)"
#       output: "results/f64/logs/slurm-%j.out"
#     env:
#       USE_64BIT: "true"
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

import jax
import numpy as np
import pandas as pd

USE_64BIT = os.environ.get("USE_64BIT", "false").lower() == "true"
if USE_64BIT:
    jax.config.update("jax_enable_x64", True)

import model
from utils import pfilter_logliks_frame, save_run_multi

print(jax.devices())
print(f"Precision: {'64-bit' if USE_64BIT else '32-bit'}")

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running at level {RUN_LEVEL}")

NP_EVAL = (2, 1000, 5000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 300, 300, 3600)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

objs = {}
loglik_frames = []

started = time.time()
for unit in model.UNITS:
    print(f"unit {unit}: {NREPS_EVAL} pfilters with J={NP_EVAL}", flush=True)
    obj = model.measles(unit, theta=model.mle_theta(unit))

    key, subkey = jax.random.split(key)
    obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)

    frame = pfilter_logliks_frame(obj)
    frame.insert(0, "unit", unit)
    loglik_frames.append(frame)
    objs[unit] = obj

    print(f"  mean logLik {frame['logLik'].mean():.2f}")

execution_time = time.time() - started

out_dir = os.path.join("results", "f64" if USE_64BIT else "f32")

save_run_multi(
    objs,
    out_dir=out_dir,
    run_config={
        "kind": "loglik",
        "model": "measles",
        "RUN_LEVEL": RUN_LEVEL,
        "USE_64BIT": USE_64BIT,
        "MAIN_SEED": model.MAIN_SEED,
        "UNITS": model.UNITS,
        "NP_EVAL": NP_EVAL,
        "NREPS_EVAL": NREPS_EVAL,
        "execution_time": execution_time,
    },
    write_traces=False,
)

logliks = pd.concat(loglik_frames, ignore_index=True)
logliks.to_csv(os.path.join(out_dir, "pfilter_logliks.csv"), index=False)

print(f"\n{logliks.groupby('unit')['logLik'].describe().to_string()}")
print(f"wrote {out_dir}/ (pfilter_logliks.csv, results.csv, timings.csv, latest.json)")
