# --- SLURM CONFIG ---
# importance: medium
# sbatch_args:
#   job-name: "panel measles loglik comparison (pypomp)"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "results/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "00:04:00" }
#   2:
#     sbatch_args: { time: "00:15:00" }
#   3:
#     sbatch_args: { time: "00:15:00" }
#   4:
#     sbatch_args: { time: "08:00:00" }
# --- END SLURM CONFIG ---

import os
import pickle
from typing import Any
import jax
import numpy as np
import pandas as pd
import pypomp as pp

print(jax.devices())

MAIN_SEED = 594709947
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
NP_EVAL = (2, 1000, 5000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 300, 300, 3600)[RUN_LEVEL - 1]

print(f"Running at level {RUN_LEVEL}")
print(f"NP_EVAL: {NP_EVAL}, NREPS_EVAL: {NREPS_EVAL}")

# Units to process
CHOSEN_UNITS = ["London", "Halesworth", "Hastings", "Cardiff"]
SHARED_PARAMS = ["R0", "sigma", "gamma", "sigmaSE", "cohort", "amplitude"]
SPECIFIC_PARAMS = ["iota", "rho", "psi", "S_0", "E_0", "I_0", "R_0"]

# Load MLE parameters from CSV
mle_params = pd.read_csv("../../../../measles/R_comparison/data/AK_mles.csv")

# Extract MLEs for the chosen units
mles = mle_params.set_index("town").loc[CHOSEN_UNITS]

# Average the shared parameters across the 4 units
shared_dict = {}
for p in SHARED_PARAMS:
    shared_dict[p] = float(mles[p].mean())
shared_df = pd.DataFrame.from_dict(shared_dict, orient="index").rename(
    columns={0: "shared"}
)

# Specific parameters matrix
specific_df = mles[SPECIFIC_PARAMS].T

# Construct PanelParameters
theta_dict = {"shared": shared_df, "unit_specific": specific_df}
theta = pp.PanelParameters(theta=theta_dict)

# Construct panelPomp
dummy_initial_params_list = pp.Pomp.sample_params(
    {
        p: (
            theta_dict["shared"].loc[p, "shared"],
            theta_dict["shared"].loc[p, "shared"] + 1e-5,
        )
        for p in SHARED_PARAMS
    }
    | {
        p: (
            float(theta_dict["unit_specific"].loc[p].iloc[0]),
            float(theta_dict["unit_specific"].loc[p].iloc[0]) + 1e-5,
        )
        for p in SPECIFIC_PARAMS
    },
    1,
    key=key,
)

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
    theta=theta,
)


# Run Panel Pfilter
print(f"Running {NREPS_EVAL} panel pfilters with J={NP_EVAL}...")
key, subkey = jax.random.split(key)
import time

t_start = time.perf_counter()
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
t_pfilter = time.perf_counter() - t_start
print(f"Pfilter completed in {t_pfilter:.2f} seconds.")

# Extract logLiks
all_logliks = []
if len(panel_measles_obj.results_history) > 0:
    last_res: Any = panel_measles_obj.results_history[-1]
    logliks_da = last_res.logLiks  # coordinates: theta_idx, unit, rep

    # We only have one parameter set (theta_idx = 0)
    logliks_df = logliks_da.isel(theta_idx=0).to_dataframe(name="logLik").reset_index()

    # logliks_df has columns: unit, rep, logLik
    logliks_df["replicate"] = logliks_df["rep"] + 1

    os.makedirs("results", exist_ok=True)
    output_file = "results/pfilter_logliks_f32.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(logliks_df[["unit", "replicate", "logLik"]], f)

    print(f"Saved results to {output_file}")

    # Save timing to CSV
    timing_df = pd.DataFrame({"phase": ["pfilter"], "time_seconds": [t_pfilter]})
    timing_df.to_csv("results/pypomp_time.csv", index=False)
    print("Saved timing to results/pypomp_time.csv")
else:
    print("Warning: No results in results_history")
