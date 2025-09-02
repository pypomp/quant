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
dacca_obj = pp.dacca()

key, subkey = jax.random.split(key)
start_time = time.time()
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
print(f"pfilter time taken: {time.time() - start_time} seconds")

print(dacca_obj.results())

with open("dacca_results_eval.pkl", "wb") as f:
    pickle.dump(dacca_obj, f)
