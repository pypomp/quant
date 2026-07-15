# --- SLURM CONFIG ---
# importance: medium
# sbatch_args:
#   job-name: "panel measles speed comparison (pypomp)"
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
import time
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
NP_FITR = (2, 500, 5000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 10, 100, 100)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 36, 36)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 5000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 36, 36)[RUN_LEVEL - 1]

print(f"Running at level {RUN_LEVEL}")
print(f"NP_FITR: {NP_FITR}, NFITR: {NFITR}, NREPS_FITR: {NREPS_FITR}")

# 4 Units to process
CHOSEN_UNITS = ["London", "Halesworth", "Hastings", "Cardiff"]
SHARED_PARAMS = ["R0", "sigma", "gamma", "sigmaSE", "cohort", "amplitude"]
SPECIFIC_PARAMS = ["iota", "rho", "psi", "S_0", "E_0", "I_0", "R_0"]

# Load starting parameters from parameter_comparison folder
starting_params_path = "../parameter_comparison/starting_parameters.csv"
if not os.path.exists(starting_params_path):
    print(
        f"Starting parameters not found at {starting_params_path}. Re-sampling starting parameters..."
    )
    # sample and write starting parameters if they don't exist
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
    initial_params = pp.PanelPomp.sample_params(
        measles_box,
        n=NREPS_FITR,
        units=CHOSEN_UNITS,
        key=subkey,
        shared_names=SHARED_PARAMS,
    )
else:
    starting_params_df: Any = pd.read_csv(starting_params_path)
    # Slices to match replication count if different
    starting_params_df = starting_params_df[
        starting_params_df["replicate"] < NREPS_FITR
    ]

    # Reconstruct PanelParameters from the flat dataframe
    shared_data_list = []
    unit_specific_data_list = []
    for j in range(NREPS_FITR):
        rep_df = starting_params_df[starting_params_df["replicate"] == j]

        shared_rep = rep_df[rep_df["unit"] == "shared"]
        shared_vals = shared_rep.set_index("param")["value"]
        shared_data_list.append(
            pd.DataFrame(
                shared_vals.values,
                index=shared_vals.index,
                columns=pd.Index(["shared"]),
            )
        )

        specific_rep = rep_df[rep_df["unit"] != "shared"]
        specific_pivot = specific_rep.pivot(
            index="param", columns="unit", values="value"
        )
        unit_specific_data_list.append(specific_pivot[CHOSEN_UNITS])

    initial_params = pp.PanelParameters(
        theta=[
            {"shared": s_df, "unit_specific": u_df}
            for s_df, u_df in zip(shared_data_list, unit_specific_data_list)
        ]
    )

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
# The discrepancy occurs due to the available spline smoothing functions being slightly 
# different between the two languages. The difference is significant for some cities.
import sys
sys.path.append("..")
from align_covariates import align_covariates
align_covariates(pomp_dict)


panel_measles_obj = pp.PanelPomp(
    Pomp_dict=pomp_dict,
    theta=pp.PanelParameters(theta=initial_params),
)

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

# --- RUN BENCHMARK ---
print(f"\n--- Running benchmark ---")

# 1. Benchmark MIF
key, subkey = jax.random.split(key)
t_start = time.perf_counter()
panel_measles_obj.mif(J=NP_FITR, M=NFITR, key=subkey, rw_sd=RW_SD, block=True)
t_mif = time.perf_counter() - t_start
print(f"MIF completed in {t_mif:.2f} seconds.")

# 2. Benchmark Pfilter (Cold - compiles JAX)
key, subkey = jax.random.split(key)
t_start = time.perf_counter()
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
t_pfilter_cold = time.perf_counter() - t_start
print(f"Pfilter (cold) completed in {t_pfilter_cold:.2f} seconds.")

# 3. Benchmark Pfilter (Warm - pre-compiled)
key, subkey = jax.random.split(key)
t_start = time.perf_counter()
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
t_pfilter_warm = time.perf_counter() - t_start
print(f"Pfilter (warm) completed in {t_pfilter_warm:.2f} seconds.")

results_to_save = {
    "mif_time": t_mif,
    "pfilter_time": t_pfilter_cold,
    "pfilter_warm_time": t_pfilter_warm,
}

# Save results
os.makedirs("results", exist_ok=True)
output_path = "results/pypomp_speed_results.pkl"
with open(output_path, "wb") as f:
    pickle.dump(results_to_save, f)

print(f"\npypomp speed benchmark complete. Saved to {output_path}")
