# --- SLURM CONFIG ---
# jobs:
#   gpu:
#     sbatch_args:
#       job-name: "pypomp measles quant test (gpu)"
#       partition: gpu
#       gpus: "v100:1"
#       cpus-per-gpu: 1
#       mem: 6GB
#       output: "gpu_results/logs/slurm-%j.out"
#       account: "ionides0"
#   cpu:
#     sbatch_args:
#       job-name: "pypomp measles quant test (cpu)"
#       partition: standard
#       cpus-per-task: 1
#       mem: 6GB
#       output: "cpu_results/logs/slurm-%j.out"
#       account: "ionides0"
#       time: "00:04:00"
#     env:
#       USE_CPU: "true"
# --- END SLURM CONFIG ---

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set JAX platform before importing JAX
USE_CPU = os.environ.get("USE_CPU", "false").lower() == "true"
if USE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "SLURM_CPUS_PER_TASK" in os.environ:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={os.environ['SLURM_CPUS_PER_TASK']}"
        )

import jax
import pickle
import jax.numpy as jnp
import pypomp as pp
import numpy as np
import time

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

run_config = {
    "RUN_LEVEL": RUN_LEVEL,
    "NP_FITR": NP_FITR,
    "NFITR": NFITR,
    "NTRAIN": NTRAIN,
    "NREPS_FITR": NREPS_FITR,
    "NP_EVAL": NP_EVAL,
    "NREPS_EVAL": NREPS_EVAL
}

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
    model="001b",
    clean=True,
)

key, subkey = jax.random.split(key)

start_time = time.time()

measles_obj.mif(
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
measles_obj.prune(n=1, refill=False)
measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, CLL=True)

measles_obj.print_summary()
print(measles_obj.time())

out_dir = "cpu_results" if USE_CPU else "gpu_results"

with open(f"{out_dir}/measles_results.pkl", "wb") as f:
    pickle.dump(measles_obj, f)