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

UNITS = [
    # "Bedwellty",
    # "Birmingham",
    # "Bradford",
    # "Bristol",
    # "Cardiff",
    # "Consett",
    # "Dalton.in.Furness",
    "Halesworth",
    "Hastings",
    "Hull",
    "Leeds",
    "Lees",
    "Liverpool",
    "London",
    "Manchester",
    # "Mold",
    # "Northwich",
    # "Nottingham",
    # "Oswestry",
    # "Sheffield",
]
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
dummy_initial_params_list = pp.Pomp.sample_params(measles_box, NREPS_FITR, key=subkey)

initial_params = pp.PanelPomp.sample_params(
    measles_box,
    n=NREPS_FITR,
    units=UNITS,
    key=subkey,
    shared_names=[],
)

# ----- Create pomp objects -----

pomp_dict = {
    unit: pp.UKMeasles.Pomp(
        unit=[unit],
        theta=dummy_initial_params_list,
        model="001b",
        clean=True,
    )
    for unit in UNITS
}

panel_measles_obj = pp.PanelPomp(
    Pomp_dict=pomp_dict,
    theta=pp.PanelParameters(theta=initial_params),
)


# ----- MIF round 1 -----
key, subkey = jax.random.split(key)
panel_measles_obj.mif(
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
print(panel_measles_obj.results(ignore_nan=False))

# ----- PFILTER round 1 -----
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(panel_measles_obj.results(ignore_nan=False))

# ----- Mix-and-match, then evaluate best model -----
panel_measles_obj.mix_and_match()
panel_measles_obj.prune(n=1, refill=False)
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)

with open("panel_measles_results.pkl", "wb") as f:
    pickle.dump(panel_measles_obj, f)

panel_measles_obj.print_summary()

results = panel_measles_obj.results(ignore_nan=False)
print(results[["unit", "unit logLik"]].groupby("unit").max())

print(panel_measles_obj.time())
