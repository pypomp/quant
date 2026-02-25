# --- SLURM CONFIG ---
# sbatch_args:
#   job-name: "pypomp measles pfilter comparison against R"
#   partition: gpu
#   gpus: "v100:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "slurm-%j.out"
#   account: "ionides0"
# run_levels:
#   1:
#     sbatch_args: { time: "00:04:00" }
# --- END SLURM CONFIG ---

import os
import pandas as pd

# Set JAX platform before importing JAX
if os.environ.get("USE_CPU", "false").lower() == "true":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import pickle
import pypomp as pp
import numpy as np

# jax.config.update("jax_enable_x64", True)

print(jax.devices())

MAIN_SEED = 594709947
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

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

    # Create pomp object for this unit
    measles_obj = pp.UKMeasles.Pomp(
        unit=[unit_name],
        theta=unit_params,
        model="001b",
        clean=False,
    )

    # Run pfilter 3600 times
    print("  Running 3600 pfilters with J=5000...")
    key, subkey = jax.random.split(key)
    measles_obj.pfilter(J=5000, reps=3600, key=subkey)

    # Extract logLiks from results_history
    if len(measles_obj.results_history) > 0:
        # Get logLiks from the last pfilter result
        logliks_data = measles_obj.results_history[-1].logLiks

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

# Save logLiks to file
if all_logliks:
    results_df = pd.DataFrame(all_logliks)
    output_file = "pfilter_logliks.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results_df, f)

    print(f"\nResults saved to {output_file}")
    print(f"Total replicates: {len(results_df)}")
else:
    print("Warning: No results to save")
