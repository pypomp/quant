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
NFITR = (2, 10, 100, 100)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 20, 36)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

# RW_SD = jnp.array([0.02] * 9 + [0.0] * 4)
# RW_SD_INIT = jnp.array([0.0] * 9 + [0.1] * 4)
DEFAULT_SD = 0.02
DEFAULT_IVP_SD = DEFAULT_SD * 12
RW_SD_DICT = {
    "R0": (DEFAULT_SD * 0.25, 0.0),
    "sigma": (DEFAULT_SD * 0.25, 0.0),
    "gamma": (DEFAULT_SD * 0.5, 0.0),
    "iota": (DEFAULT_SD, 0.0),
    "rho": (DEFAULT_SD * 0.5, 0.0),
    "sigmaSE": (DEFAULT_SD, 0.0),
    "psi": (DEFAULT_SD * 0.25, 0.0),
    "cohort": (DEFAULT_SD * 0.5, 0.0),
    "amplitude": (DEFAULT_SD * 0.5, 0.0),
    "S_0": (0.0, DEFAULT_IVP_SD),
    "E_0": (0.0, DEFAULT_IVP_SD),
    "I_0": (0.0, DEFAULT_IVP_SD),
    "R_0": (0.0, DEFAULT_IVP_SD),
}
RW_SD = jnp.array([RW_SD_DICT[k][0] for k in RW_SD_DICT.keys()])
RW_SD_INIT = jnp.array([RW_SD_DICT[k][1] for k in RW_SD_DICT.keys()])
COOLING_RATE = 0.5

measles_box = {
    "R0": [10.0, 60.0],
    "sigma": [25.0, 100.0],
    "gamma": [25.0, 320.0],
    "iota": [0.004, 3.0],
    "rho": [0.1, 0.9],
    "sigmaSE": [0.04, 0.1],
    "psi": [0.05, 3.0],
    "cohort": [0.1, 0.7],
    "amplitude": [0.1, 0.6],
    "S_0": [0.01, 0.07],
    "E_0": [0.000004, 0.0001],
    "I_0": [0.000003, 0.001],
    "R_0": [0.9, 0.99],
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

# Apply log barycentric transformation to S_0, E_0, I_0, R_0
for params in transformed_params_list:
    S = params["S_0"]
    E = params["E_0"]
    I = params["I_0"]
    R = params["R_0"]
    total = S + E + I + R

    S /= total
    E /= total
    I /= total
    R /= total

    params["S_0"] = float(np.log(S))
    params["E_0"] = float(np.log(E))
    params["I_0"] = float(np.log(I))
    params["R_0"] = float(np.log(R))


measles_obj = pp.UKMeasles.Pomp(
    unit=["London"],
    theta=transformed_params_list,
    model="001b",
    clean=True,
)

key, subkey = jax.random.split(key)
measles_obj.mif(
    theta=transformed_params_list,
    sigmas=RW_SD,
    sigmas_init=RW_SD_INIT,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
print(measles_obj.results(ignore_nan=True))

measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(measles_obj.results(ignore_nan=True))


measles_obj.print_summary()
print(measles_obj.time())

with open("measles_results.pkl", "wb") as f:
    pickle.dump(measles_obj, f)
