# --- SLURM CONFIG ---
# sbatch_args:
#   job-name: "SPX pfilter check (pypomp)"
#   partition: gpu
#   gpus: "v100:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "py_results/logs/slurm-%j.out"
#   account: "ionides0"
# --- END SLURM CONFIG ---

import os
from datetime import datetime
from importlib.metadata import version
import jax
import pickle
import pypomp as pp
import numpy as np


now = datetime.now()
print("DATE: ", now.date())
print("TIME: ", now.time())
print("pypomp version:", version("pypomp"))
print("jax version:", version("jax"))

print(jax.devices())

MAIN_SEED = 631409
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "2"))

NP_EVAL = (2, 1000, 1000, 1000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 3600)[RUN_LEVEL - 1]

print(f"Running at level {RUN_LEVEL}")


spx_obj = pp.spx()

theta = {
    "mu": 3.68e-4,
    "kappa": 3.14e-2,
    "theta": 1.12e-4,
    "xi": 2.27e-3,
    "rho": -7.38e-1,
    "V_0": 7.66e-3**2,
}

key, subkey = jax.random.split(key)
spx_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey, theta=theta)
print(spx_obj.results())

with open("py_results/spx_results_eval.pkl", "wb") as f:
    pickle.dump(spx_obj, f)
