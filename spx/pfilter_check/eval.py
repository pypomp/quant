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

NP_EVAL = (2, 1000, 1000, 1000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 3600)[RUN_LEVEL - 1]

print(f"Running at level {RUN_LEVEL}")


def rho_transform(x):
    """Transform rho to perturbation scale"""
    return np.log((1 + x) / (1 - x))


key, subkey = jax.random.split(key)
spx_obj = pp.spx()

theta = {
    "mu": float(jnp.log(3.68e-4)),
    "kappa": float(jnp.log(3.14e-2)),
    "theta": float(jnp.log(1.12e-4)),
    "xi": float(jnp.log(2.27e-3)),
    "rho": rho_transform(-7.38e-1),
    "V_0": float(jnp.log(7.66e-3**2)),
}

start_time = time.time()
spx_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey, theta=theta)
print(f"pfilter time taken: {time.time() - start_time} seconds")

with open("spx_results_eval.pkl", "wb") as f:
    pickle.dump(spx_obj, f)
