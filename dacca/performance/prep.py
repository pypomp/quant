"""
This script prepares the environment for the performance tests.
"""

import os
from datetime import datetime
from importlib.metadata import version

# Set JAX platform before importing JAX
USE_CPU = os.environ.get("USE_CPU", "false").lower() == "true"
if USE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pypomp as pp  # noqa: E402
import numpy as np  # noqa: E402

print(jax.devices())

print("Using CPU: ", USE_CPU)
now = datetime.now()
print("DATE: ", now.date())
print("TIME: ", now.time())
print("pypomp version:", version("pypomp"))
print("jax version:", version("jax"))

MAIN_SEED = 631409
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))

NREPS_FITR = (2, 3, 20, 36)[RUN_LEVEL - 1]

print(f"Running at level {RUN_LEVEL}")

RW_SD = pp.RWSigma(
    sigmas={
        "gamma": 0.02,
        "m": 0.02,
        "rho": 0.0,
        "epsilon": 0.02,
        "omega": 0.02,
        "c": 0.02,
        "beta_trend": 0.02,
        "sigma": 0.02,
        "tau": 0.02,
        "bs1": 0.02,
        "bs2": 0.02,
        "bs3": 0.02,
        "bs4": 0.02,
        "bs5": 0.02,
        "bs6": 0.02,
        "omegas1": 0.02,
        "omegas2": 0.02,
        "omegas3": 0.02,
        "omegas4": 0.02,
        "omegas5": 0.02,
        "omegas6": 0.02,
    },
    init_names=[],
)
COOLING_RATE = 0.5

dacca_obj = pp.dacca(dt=None, nstep=20)

# params_box = {k: [v * 0.5, v * 1.5] for k, v in dacca_obj.theta[0].items()}
# params_box["rho"] = [0.0, 0.0]
# This is the params box from diffPomp
params_box = {
    "gamma": [10.0, 40.0],
    "m": [0.03, 0.60],
    "rho": [0.0, 0.0],
    "epsilon": [0.20, 30.0],
    "omega": [float(jnp.exp(-4.5)), float(jnp.exp(-4.5))],
    "c": [1.0, 1.0],
    "beta_trend": [-0.01, 0.00],
    "sigma": [1.0, 5.0],
    "tau": [0.10, 0.50],
    "bs1": [-4.0, 4.0],
    "bs2": [0.0, 8.0],
    "bs3": [-4.0, 4.0],
    "bs4": [0.0, 8.0],
    "bs5": [0.0, 8.0],
    "bs6": [0.0, 8.0],
    "omegas1": [-10.0, 0.0],
    "omegas2": [-10.0, 0.0],
    "omegas3": [-10.0, 0.0],
    "omegas4": [-10.0, 0.0],
    "omegas5": [-10.0, 0.0],
    "omegas6": [-10.0, 0.0],
}

key, subkey = jax.random.split(key)
initial_params_list = pp.Pomp.sample_params(params_box, NREPS_FITR, key=subkey)
