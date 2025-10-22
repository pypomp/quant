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

initial_shared, initial_unit_specific = pp.PanelPomp.sample_params(
    measles_box,
    n=NREPS_FITR,
    units=["London", "Hastings"],
    key=subkey,
    shared_names=["gamma"],
)

# Transform shared parameter DataFrames
transformed_shared = []
if initial_shared is not None:
    for df in initial_shared:
        df_t = df.copy()
        for idx, row in df.iterrows():
            transformed_row = row.copy()
            for k, v in row.items():
                if k in ["R0", "sigmaSE", "sigma", "iota", "psi", "gamma"]:
                    transformed_row[k] = float(np.log(v))
                elif k in ["rho", "amplitude", "cohort"]:
                    transformed_row[k] = float(pp.logit(v))
                else:
                    transformed_row[k] = v
            df_t.loc[idx] = transformed_row
        transformed_shared.append(df_t)
else:
    transformed_shared = None

# Transform unit-specific parameter DataFrames
transformed_unit_specific = []
if initial_unit_specific is not None:
    for df in initial_unit_specific:
        df_t = df.copy()
        for idx, row in df.iterrows():
            transformed_row = row.copy()
            for k, v in row.items():
                if k in ["R0", "sigmaSE", "sigma", "iota", "psi", "gamma"]:
                    transformed_row[k] = float(np.log(v))
                elif k in ["rho", "amplitude", "cohort"]:
                    transformed_row[k] = float(pp.logit(v))
                else:
                    transformed_row[k] = v
            df_t.loc[idx] = transformed_row
        transformed_unit_specific.append(df_t)
else:
    transformed_unit_specific = None

# Apply log barycentric transformation to S_0, E_0, I_0, R_0
# Assume each df is indexed by parameter name (S_0, E_0, etc.) and columns are units ("London", "Hastings", etc.)
if transformed_unit_specific is not None:
    for df in transformed_unit_specific:
        # Extract all vectors for S_0, E_0, I_0, R_0 per column (unit)
        if all(col in df.index for col in ["S_0", "E_0", "I_0", "R_0"]):
            S_vec = df.loc["S_0"]
            E_vec = df.loc["E_0"]
            I_vec = df.loc["I_0"]
            R_vec = df.loc["R_0"]
            total = S_vec + E_vec + I_vec + R_vec
            # Normalize and log, assign back
            df.loc["S_0"] = np.log(S_vec / total)
            df.loc["E_0"] = np.log(E_vec / total)
            df.loc["I_0"] = np.log(I_vec / total)
            df.loc["R_0"] = np.log(R_vec / total)


london = pp.UKMeasles.Pomp(
    unit=["London"],
    theta=dummy_initial_params_list,
    model="001b",
    clean=True,
)

hastings = pp.UKMeasles.Pomp(
    unit=["Hastings"],
    theta=dummy_initial_params_list,
    model="001b",
    clean=True,
)

panel_measles_obj = pp.PanelPomp(
    Pomp_dict={
        "London": london,
        "Hastings": hastings,
    },
    shared=transformed_shared,
    unit_specific=transformed_unit_specific,
)

key, subkey = jax.random.split(key)
panel_measles_obj.mif(
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
print(panel_measles_obj.results(ignore_nan=False))
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(panel_measles_obj.results(ignore_nan=False))

print(panel_measles_obj.time())

with open("panel_measles_results.pkl", "wb") as f:
    pickle.dump(panel_measles_obj, f)
