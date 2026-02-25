# --- SLURM CONFIG ---
# sbatch_args:
#   job-name: "Measles pfilter check"
#   partition: gpu
#   gpus: "v100:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "results/logs/slurm-%j.out"
#   account: "ionides0"
# run_levels:
#   1:
#     sbatch_args: { time: "00:04:00" }
# --- END SLURM CONFIG ---

import os
import time

# Set JAX platform before importing JAX
if os.environ.get("USE_CPU", "false").lower() == "true":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import pickle
import jax.numpy as jnp
import pypomp as pp
import numpy as np


print(jax.devices())

MAIN_SEED = 631409
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))

NP_EVAL = (2, 1000, 5000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 300, 300, 3600)[RUN_LEVEL - 1]

print(f"Running at level {RUN_LEVEL}")

# Parameters from AK
measles_obj = pp.UKMeasles.Pomp(
    unit=["London"],
    theta={
        "R0": 56.8,
        "sigma": 28.9,
        "gamma": 30.4,
        "iota": 2.9,
        "rho": 0.488,
        "sigmaSE": 0.0878,
        "psi": 0.116,
        "cohort": 0.557,
        "amplitude": 0.554,
        "S_0": 2.97e-02,
        "E_0": 5.17e-05,
        "I_0": 5.14e-05,
        "R_0": 9.70e-01,
    },
)

key, subkey = jax.random.split(key)
measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey, CLL=True)
print(measles_obj.results(ignore_nan=True))

measles_obj.print_summary()

with open("results/measles_results.pkl", "wb") as f:
    pickle.dump(measles_obj, f)
