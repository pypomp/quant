# --- SLURM CONFIG ---
# importance: medium
# sbatch_args:
#   job-name: "dacca pfilter check"
#   partition: gpu
#   gpus: "v100:1"
#   cpus-per-gpu: 1
#   time: "00:20:00"
#   mem: 6GB
#   output: "py_results/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "00:20:00" }
# --- END SLURM CONFIG ---

"""
This script estimates the distribution of the estimated logLik at the default parameter value (an estimate of the MLE).
"""

import os
import time

# Set JAX platform before importing JAX
if os.environ.get("USE_CPU", "false").lower() == "true":
    os.environ["JAX_PLATFORMS"] = "cpu"

import pickle

import jax
import numpy as np
import pypomp as pp

print(jax.devices())

MAIN_SEED = 631409
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))

NP_EVAL = (2, 1000, 5000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 300, 300, 3600)[RUN_LEVEL - 1]

print(f"Running at level {RUN_LEVEL}")

# Parameters from AK
dacca_obj = pp.models.dacca()

key, subkey = jax.random.split(key)
start_time = time.time()
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
print(f"pfilter time taken: {time.time() - start_time} seconds")

print(dacca_obj.results())

os.makedirs("py_results", exist_ok=True)
with open("py_results/dacca_results_eval.pkl", "wb") as f:
    pickle.dump(dacca_obj, f)
