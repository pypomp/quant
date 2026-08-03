# --- SLURM CONFIG ---
# importance: medium
# sbatch_args:
#   job-name: "panel measles parameter comparison (pypomp)"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "results/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "00:04:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "01:00:00" }
#   4:
#     sbatch_args: { time: "02:00:00" }
# --- END SLURM CONFIG ---

import os
import pickle
import jax
import numpy as np
import pandas as pd
import pypomp as pp

print(jax.devices())

MAIN_SEED = 594709947
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))

NP_FITR = (2, 500, 5000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 10, 100, 100)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 36, 360)[RUN_LEVEL - 1]

print(f"Running at level {RUN_LEVEL}")
print(f"NP_FITR: {NP_FITR}, NFITR: {NFITR}, NREPS_FITR: {NREPS_FITR}")

# 4 Units to process
CHOSEN_UNITS = ["London", "Halesworth", "Hastings", "Cardiff"]
SHARED_PARAMS = ["R0", "sigma", "gamma", "sigmaSE", "cohort", "amplitude"]

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
).geometric_cooling(a=0.5)

measles_box = {
    "R0": (10.0, 60.0),
    "sigma": (25.0, 100.0),
    "gamma": (25.0, 320.0),
    "iota": (0.004, 3.0),
    "rho": (0.1, 0.9),
    "sigmaSE": (0.04, 0.1),
    "psi": (0.05, 3.0),
    "cohort": (0.1, 0.7),
    "amplitude": (0.1, 0.6),
    "S_0": (0.01, 0.07),
    "E_0": (0.000004, 0.0001),
    "I_0": (0.000003, 0.001),
    "R_0": (0.9, 0.99),
}

key, subkey = jax.random.split(key)
dummy_initial_params_list = pp.Pomp.sample_params(measles_box, NREPS_FITR, key=subkey)

initial_params = pp.PanelPomp.sample_params(
    measles_box,
    n=NREPS_FITR,
    units=CHOSEN_UNITS,
    key=subkey,
    shared_names=SHARED_PARAMS,
)

# Export starting parameters to CSV
records = []
for rep_idx, rep_dict in enumerate(initial_params._to_list()):
    if rep_dict["shared"] is not None:
        shared_df = rep_dict["shared"]
        for param in shared_df.index:
            records.append(
                {
                    "replicate": rep_idx,
                    "unit": "shared",
                    "param": param,
                    "value": float(shared_df.loc[param, "shared"]),
                }
            )
    if rep_dict["unit_specific"] is not None:
        unit_spec_df = rep_dict["unit_specific"]
        for unit in unit_spec_df.columns:
            for param in unit_spec_df.index:
                records.append(
                    {
                        "replicate": rep_idx,
                        "unit": unit,
                        "param": param,
                        "value": float(unit_spec_df.loc[param, unit]),
                    }
                )
os.makedirs("results", exist_ok=True)
pd.DataFrame(records).to_csv("starting_parameters.csv", index=False)
print("Saved starting_parameters.csv")

pomp_dict = {
    unit: pp.models.UKMeasles.Pomp(
        unit=unit,
        theta=dummy_initial_params_list,
        model="001b",
        clean=False,
    )
    for unit in CHOSEN_UNITS
}

# Override covariates with R-computed ones to match R exactly
import sys

sys.path.append("..")
from align_covariates import align_covariates

align_covariates(pomp_dict)


panel_measles_obj = pp.PanelPomp(
    Pomp_dict=pomp_dict,
    theta=pp.PanelParameters(theta=initial_params),
)

# Run Panel MIF
key, subkey = jax.random.split(key)
panel_measles_obj.mif(rw_sd=RW_SD, M=NFITR, J=NP_FITR, key=subkey, block=True)

# Extract and save results
results_df = panel_measles_obj.results()
with open("results/mif_coefs.pkl", "wb") as f:
    pickle.dump(results_df, f)

print("Saved results/mif_coefs.pkl")
