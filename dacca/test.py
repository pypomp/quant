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

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 20, 100, 100)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 20, 36)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

# rho fixed at 0
RW_SD = jnp.array([0.02] * 2 + [0.0] + [0.02] * 18)
RW_SD_INIT = jnp.array([0.0] * 21)
COOLING_RATE = 0.5

dacca_obj = pp.dacca()

params_box = {k: [v * 0.5, v * 1.5] for k, v in dacca_obj.theta[0].items()}
params_box["rho"] = [0.0, 0.0]
key, subkey = jax.random.split(key)
initial_params_list = pp.Pomp.sample_params(params_box, NREPS_FITR, key=subkey)

with jax.profiler.trace("dacca_profiler"):
    key, subkey = jax.random.split(key)
    dacca_obj.pfilter(theta=initial_params_list, J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
    dacca_obj.mif(
        theta=initial_params_list,
        sigmas=RW_SD,
        sigmas_init=RW_SD_INIT,
        M=NFITR,
        a=COOLING_RATE,
        J=NP_FITR,
    )
    dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
    # dacca_obj.train(J=NP_FITR, M=NTRAIN, eta=0.2)
    # dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
    dacca_obj.prune(n=3)
    dacca_obj.mif(
        sigmas=RW_SD / 4,
        sigmas_init=RW_SD_INIT / 4,
        M=NFITR,
        a=COOLING_RATE,
        J=NP_FITR,
    )
    dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)

print(dacca_obj.time())

with open("dacca_results.pkl", "wb") as f:
    pickle.dump(dacca_obj, f)
