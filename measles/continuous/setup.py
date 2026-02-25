"""
This script sets up the continuous UK measles model.
"""

import os
from datetime import datetime
from importlib.metadata import version
import jax
import pypomp as pp
import numpy as np

print(jax.devices())

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

DEFAULT_SD = 0.02
DEFAULT_IVP_SD = DEFAULT_SD * 12
RW_SD = pp.RWSigma(
    sigmas={
        "R0": DEFAULT_SD * 0.25,
        "sigma": DEFAULT_SD * 0.25,
        "gamma": DEFAULT_SD * 0.5,
        "iota": DEFAULT_SD,
        "rho": DEFAULT_SD * 0.5,
        "sigmaSE": DEFAULT_SD,
        "psi": DEFAULT_SD * 0.25,
        "cohort": DEFAULT_SD * 0.5,
        "amplitude": DEFAULT_SD * 0.5,
        "S_0": DEFAULT_IVP_SD,
        "E_0": DEFAULT_IVP_SD,
        "I_0": DEFAULT_IVP_SD,
        "R_0": DEFAULT_IVP_SD,
    },
    init_names=["S_0", "E_0", "I_0", "R_0"],
)
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

measles_obj = pp.UKMeasles.Pomp(
    unit=["Halesworth"],
    theta=initial_params_list,
    model="003",
    clean=True,
)
