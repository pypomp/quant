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

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "2"))

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 20, 100, 100)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 20, 36)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

RW_SD = jnp.array([0.02] * 4 + [0] * 4 + [0.02] * 5)
RW_SD_INIT = jnp.array([0.0] * 4 + [0.1] * 4 + [0.0] * 5)
COOLING_RATE = 0.5

measles_box = {
    "R0": [10, 60],
    "rho": [0.1, 0.9],
    "sigmaSE": [0.04, 0.1],
    "amplitude": [0.1, 0.6],
    "S_0": [0.01, 0.07],
    "E_0": [0.000004, 0.0001],
    "I_0": [0.000003, 0.001],
    "R_0": [0.9, 0.99],
    "sigma": [25, 100],
    "iota": [0.004, 3],
    "psi": [0.05, 3],
    "cohort": [0.1, 0.7],
    "gamma": [25, 320],
}

key, subkey = jax.random.split(key)
initial_params_list = pp.Pomp.sample_params(measles_box, NREPS_FITR, key=subkey)

transformed_params_list = []
for params in initial_params_list:
    transformed_params = {}
    for k, v in params.items():
        if k in ["R0", "sigmaSE", "sigma", "iota", "psi", "gamma"]:
            transformed_params[k] = float(np.log(v))
        elif k in ["rho", "amplitude", "cohort"]:
            transformed_params[k] = float(pp.logit(v))
        else:
            transformed_params[k] = v
    transformed_params_list.append(transformed_params)


measles_obj = pp.UKMeasles.Pomp(
    unit=["London"],
    theta=transformed_params_list,
)

print("Starting IF2")
key, subkey = jax.random.split(key)
start_time = time.time()
measles_obj.mif(
    theta=transformed_params_list,
    sigmas=RW_SD,
    sigmas_init=RW_SD_INIT,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
print(f"mif time taken: {time.time() - start_time} seconds")

start_time = time.time()
measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
print(f"pfilter time taken: {time.time() - start_time} seconds")

# start_time = time.time()
# measles_obj.train(J=NP_FITR, reps=NTRAIN, key=subkey)
# print(f"train time taken: {time.time() - start_time} seconds")

# start_time = time.time()
# measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
# print(f"pfilter time taken: {time.time() - start_time} seconds")

with open("measles_results.pkl", "wb") as f:
    pickle.dump(measles_obj, f)
