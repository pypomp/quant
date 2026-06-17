# --- SLURM CONFIG ---
# importance: medium
# sbatch_args:
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 6GB
# jobs:
#   float32:
#     sbatch_args:
#       job-name: "measles loglik comparison (pypomp 32-bit)"
#       output: "results/logs/slurm-32-%j.out"
#     env:
#       USE_64BIT: "false"
#   float64:
#     sbatch_args:
#       job-name: "measles loglik comparison (pypomp 64-bit)"
#       output: "results/logs/slurm-64-%j.out"
#     env:
#       USE_64BIT: "true"
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

import jax
import numpy as np
import pandas as pd
import pypomp as pp

# Set JAX platform before JAX operations
if os.environ.get("USE_CPU", "false").lower() == "true":
    os.environ["JAX_PLATFORMS"] = "cpu"

# Configure float precision based on environment variable
USE_64BIT = os.environ.get("USE_64BIT", "false").lower() == "true"
if USE_64BIT:
    jax.config.update("jax_enable_x64", True)

print(jax.devices())

MAIN_SEED = 594709947
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
NP_EVAL = (2, 1000, 5000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 300, 300, 3600)[RUN_LEVEL - 1]

print(f"Running at level {RUN_LEVEL}")
print(f"USE_64BIT: {USE_64BIT} (Precision: {'64-bit' if USE_64BIT else '32-bit'})")

# Units to process
units = ["London", "Halesworth"]

# Load MLE parameters from CSV
mle_params = pd.read_csv("../data/AK_mles.csv")

print(f"Processing {len(units)} units: {units}")

# Initialize list to store logLiks
all_logliks = []

# Loop over units
for unit_name in units:
    print(f"Processing unit: {unit_name}")

    # Extract MLE parameters for this unit
    unit_mle = mle_params[mle_params["town"] == unit_name]

    if len(unit_mle) == 0:
        print(f"Warning: No MLE parameters found for unit: {unit_name}")
        continue

    row = unit_mle.iloc[0]

    unit_params = {
        "R0": float(row["R0"]),
        "sigma": float(row["sigma"]),
        "gamma": float(row["gamma"]),
        "iota": float(row["iota"]),
        "sigmaSE": float(row["sigmaSE"]),
        "psi": float(row["psi"]),
        "rho": float(row["rho"]),
        "cohort": float(row["cohort"]),
        "amplitude": float(row["amplitude"]),
        "S_0": float(row["S_0"]),
        "E_0": float(row["E_0"]),
        "I_0": float(row["I_0"]),
        "R_0": float(row["R_0"]),
    }

    # Create pomp object for this unit using PompParameters
    measles_obj = pp.models.UKMeasles.Pomp(
        unit=[unit_name],
        theta=pp.PompParameters(unit_params),
        model="001b",
        clean=False,
    )

    # Run pfilter
    print(f"  Running {NREPS_EVAL} pfilters with J={NP_EVAL}...")
    key, subkey = jax.random.split(key)
    measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)

    # Extract logLiks from results_history
    if len(measles_obj.results_history) > 0:
        # Get logLiks from the last pfilter result
        last_res = measles_obj.results_history[-1]
        logliks_data = getattr(last_res, "logLiks")

        # Convert xarray to numpy array and flatten
        if hasattr(logliks_data, "values"):
            logliks = logliks_data.values.flatten()
        else:
            logliks = np.array(logliks_data).flatten()

        # Store results
        for i, loglik in enumerate(logliks):
            all_logliks.append(
                {"unit": unit_name, "replicate": i + 1, "logLik": float(loglik)}
            )

        print(f"  Completed {unit_name}. Mean logLik: {np.mean(logliks):.2f}")
    else:
        print(f"  Warning: No results in results_history for {unit_name}")

# Save logLiks to file indicating precision
if all_logliks:
    os.makedirs("results/logs", exist_ok=True)
    results_df = pd.DataFrame(all_logliks)
    precision_str = "64" if USE_64BIT else "32"
    output_file = f"results/pfilter_logliks_f{precision_str}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results_df, f)

    print(f"\nResults saved to {output_file}")
    print(f"Total replicates: {len(results_df)}")
else:
    print("Warning: No results to save")
