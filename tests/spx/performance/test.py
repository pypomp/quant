"""
This script is used to test the speed, parameter estimation, and likelihood estimation of the pypomp package on the S&P 500 dataset using fairly typical algorithmic parameters. It can test both GPU and CPU performance. The results are meant to be compared against the R version of the script.

Because the SPX model uses simple random number generation (it samples from a normal distribution) and has just one rproc step per observation, it is more sensitive to certain kinds of overhead in mif and pfilter, which makes it a useful measure of whether this overhead has grown too much.

Particular points of comparison:

- Execution time.
- Parameter and log likelihood traces.
- Empirical distribution of the parameter estimates.
- Empirical distribution of the log likelihood estimates.
"""

# --- SLURM CONFIG ---
# jobs:
#   gpu:
#     sbatch_args:
#       partition: gpu
#       gpus: "v100:1"
#       cpus-per-gpu: 1
#       mem: 6GB
#       time: "00:04:00"
#       output: "gpu_results/logs/slurm-%j.out"
#   cpu:
#     sbatch_args:
#       partition: standard
#       cpus-per-task: 36
#       mem: 80GB
#       time: "00:04:00"
#       output: "cpu_results/logs/slurm-%j.out"
#     env:
#       USE_CPU: "true"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:00:30" }
#   2:
#     sbatch_args: { time: "00:04:00" }
#   3:
#     sbatch_args: { time: "02:00:00" }
#   4:
#     sbatch_args: { time: "03:00:00" }
# --- END SLURM CONFIG ---

import os
import sys

import session_info

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if tests_dir not in sys.path:
    sys.path.append(tests_dir)


# Set JAX platform before importing JAX
USE_CPU = os.environ.get("USE_CPU", "false").lower() == "true"
if USE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "SLURM_CPUS_PER_TASK" in os.environ:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={os.environ['SLURM_CPUS_PER_TASK']}"
        )

import pickle  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pypomp as pp  # noqa: E402

session_info.show(dependencies=True)

print(jax.devices())

print("Using CPU: ", USE_CPU)

MAIN_SEED = 631409
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))

NP_FITR = (2, 1000, 1000, 1000)[RUN_LEVEL - 1]
NFITR = (2, 20, 200, 200)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 20, 120 * 3)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 1000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 24)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

COOLING_RATE = 0.5
RW_SD = pp.RWSigma(
    sigmas={
        "mu": 0.02,
        "kappa": 0.02,
        "theta": 0.02,
        "xi": 0.02,
        "rho": 0.02,
        "V_0": 0.1,
    },
    init_names=["V_0"],
)

sp500_box = {
    "mu": [1e-6, 1e-4],
    "kappa": [1e-8, 0.1],
    "theta": [0.000075, 0.0002],
    "xi": [1e-8, 1e-2],
    "rho": [1e-8, 1],
    "V_0": [1e-10, 1e-4],
}

key, subkey = jax.random.split(key)
initial_params_list = pp.Pomp.sample_params(sp500_box, NREPS_FITR, key=subkey)

# implement Feller's condition
for params in initial_params_list:
    params["xi"] = float(
        np.random.uniform(
            low=0,
            high=np.sqrt(params["kappa"] * params["theta"] * 2),
        )
    )

spx_obj = pp.spx()

# ----- MIF and PFILTER -----
key, subkey = jax.random.split(key)
spx_obj.mif(
    theta=initial_params_list,
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
print(spx_obj.results())
spx_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(spx_obj.results())

print(spx_obj.time())

out_dir = "cpu_results" if USE_CPU else "gpu_results"

with open(f"{out_dir}/spx_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(spx_obj, f)
