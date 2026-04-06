"""
This script tests the performance of the panel POMP implementation, running mif and pfilter.
"""
# --- SLURM CONFIG ---
# jobs:
#   u20:
#     sbatch_args:
#       job-name: "pypomp panel measles test (20 units)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 6GB
#       output: "u20_results/logs/slurm-%j.out"
#       account: "ionides0"
#     env:
#       N_UNITS: "20"
#   u100:
#     sbatch_args:
#       job-name: "pypomp panel measles test (100 units)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 6GB
#       output: "u100_results/logs/slurm-%j.out"
#       account: "ionides0"
#     env:
#       N_UNITS: "100"
#   u200:
#     sbatch_args:
#       job-name: "pypomp panel measles test (200 units)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 12GB
#       output: "u200_results/logs/slurm-%j.out"
#       account: "ionides0"
#     env:
#       N_UNITS: "200"
#   u800:
#     sbatch_args:
#       job-name: "pypomp panel measles test (800 units)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 30GB
#       output: "u800_results/logs/slurm-%j.out"
#       account: "ionides0"
#     env:
#       N_UNITS: "800"
# run_levels:
#   1:
#     sbatch_args: { time: "00:10:00" }
#   2:
#     sbatch_args: { time: "01:00:00" }
#   3:
#     sbatch_args: { time: "12:00:00" }
#   4:
#     sbatch_args: { time: "12:00:00" }
# --- END SLURM CONFIG ---

import importlib.util
import os
import pickle
import sys
import time

import jax
import numpy as np
import pypomp as pp
import session_info

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if tests_dir not in sys.path:
    sys.path.append(tests_dir)

import utils

session_info.show(dependencies=True)

print(jax.devices())

MAIN_SEED = 631409
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))

NP_FITR = (2, 500, 5000, 10000)[RUN_LEVEL - 1]
NFITR = (2, 10, 100, 200)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 36, 12)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 5000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 36, 36)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

units_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../units.py"))
spec = importlib.util.spec_from_file_location("units", units_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {units_path}")
units = importlib.util.module_from_spec(spec)
spec.loader.exec_module(units)

N_UNITS = int(os.environ.get("N_UNITS", 20))
CHOSEN_UNITS = units.UNITS[:N_UNITS]

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

SHARED_PARAMS = []
print("Shared parameters: ", SHARED_PARAMS)

key, subkey = jax.random.split(key)
dummy_initial_params_list = pp.Pomp.sample_params(measles_box, NREPS_FITR, key=subkey)

initial_params = pp.PanelPomp.sample_params(
    measles_box,
    n=NREPS_FITR,
    units=CHOSEN_UNITS,
    key=subkey,
    shared_names=SHARED_PARAMS,
)

# ----- Create pomp objects -----

pomp_dict = {
    unit: pp.UKMeasles.Pomp(
        unit=[unit],
        theta=dummy_initial_params_list,
        model="001b",
        clean=True,
    )
    for unit in CHOSEN_UNITS
}

panel_measles_obj = pp.PanelPomp(
    Pomp_dict=pomp_dict,
    theta=pp.PanelParameters(theta=initial_params),
)
# ----- MIF and PFILTER -----
start_time = time.time()

# ----- MIF round 1 -----
key, subkey = jax.random.split(key)
panel_measles_obj.mif(
    rw_sd=RW_SD, M=NFITR, a=COOLING_RATE, J=NP_FITR, key=subkey, vmap_chunk_size=16
)

# ----- PFILTER round 1 -----
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)

# ----- Mix-and-match, then evaluate best model -----
panel_measles_obj.mix_and_match()
panel_measles_obj.prune(n=1, refill=False)
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)

# ---- Save results ----

out_dir = "u" + str(N_UNITS) + "_results"

with open(f"{out_dir}/panel_measles_results.pkl", "wb") as f:
    pickle.dump(panel_measles_obj, f)

panel_measles_obj.print_summary()

results = panel_measles_obj.results(ignore_nan=False)
print(results[["unit", "unit logLik"]].groupby("unit").max())

print(panel_measles_obj.time())

execution_time = time.time() - start_time
print(f"Total execution time: {execution_time:.2f} seconds")

# ---- Save performance history ----
run_config = {
    "test": "panel_measles",
    "N_UNITS": N_UNITS,
    "RUN_LEVEL": RUN_LEVEL,
    "partition": os.environ.get("SLURM_JOB_PARTITION", "local"),
}

metrics = utils.get_pomp_metrics(
    panel_measles_obj,
    execution_time=execution_time,
    run_config=run_config,
    history_index=-2,
)
utils.append_history(metrics, f"{out_dir}/performance_history.jsonl")
print(f"Performance metrics saved to {out_dir}/performance_history.jsonl")
