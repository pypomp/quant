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
init_params = np.array([2.97e-02, 5.17e-05, 5.14e-05, 9.70e-01])
init_params_T = np.log(init_params / np.sum(init_params))
measles_obj = pp.UKMeasles.Pomp(
    unit=["London"],
    theta={
        "R0": float(np.log(56.8)),
        "sigma": float(np.log(28.9)),
        "gamma": float(np.log(30.4)),
        "iota": float(np.log(2.9)),
        "rho": float(pp.logit(0.488)),
        "sigmaSE": float(np.log(0.0878)),
        "psi": float(np.log(0.116)),
        "cohort": float(pp.logit(0.557)),
        "amplitude": float(pp.logit(0.554)),
        "S_0": float(init_params_T[0]),
        "E_0": float(init_params_T[1]),
        "I_0": float(init_params_T[2]),
        "R_0": float(init_params_T[3]),
    },
)

key, subkey = jax.random.split(key)
start_time = time.time()
measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey, CLL=True)
print(measles_obj.results(ignore_nan=True))
print(f"pfilter time taken: {time.time() - start_time} seconds")

measles_obj.print_summary()


with open("measles_results_eval.pkl", "wb") as f:
    pickle.dump(measles_obj, f)
