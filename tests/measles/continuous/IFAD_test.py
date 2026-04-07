"""
Tests pypomp on the continuous measles model using IFAD. This is meant to be compared with the IF2 results to see which one maximizes the logLik better.
"""

# --- SLURM CONFIG ---
# sbatch_args:
#   job-name: "pypomp continuous measles test (IFAD)"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 12GB
#   output: "IFAD_results/logs/slurm-%j.out"
#   account: "ionides0"
# run_levels:
#   1:
#     sbatch_args: { time: "00:03:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "00:20:00" }
#   4:
#     sbatch_args: { time: "00:30:00" }
# --- END SLURM CONFIG ---

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import pickle
import sys

import session_info

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if tests_dir not in sys.path:
    sys.path.append(tests_dir)

import jax  # noqa: E402
from setup import (  # noqa: E402
    COOLING_RATE,
    RUN_LEVEL,
    RW_SD,
    key,
    measles_obj,
)

session_info.show(dependencies=True)

NFITR = (2, 20, 100, 300)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 50)[RUN_LEVEL - 1]
NP_FITR = (2, 3, 20, 10000)[RUN_LEVEL - 1]
NP_EVAL = (2, 20, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

DEFAULT_ETA = 0.01
eta = {
    "R0": DEFAULT_ETA,
    "sigma": DEFAULT_ETA,
    "gamma": DEFAULT_ETA,
    "iota": DEFAULT_ETA,
    "rho": DEFAULT_ETA / 8,
    "sigmaSE": DEFAULT_ETA,
    "psi": DEFAULT_ETA,
    "cohort": DEFAULT_ETA,
    "amplitude": DEFAULT_ETA,
    "S_0": DEFAULT_ETA,
    "E_0": DEFAULT_ETA,
    "I_0": DEFAULT_ETA,
    "R_0": DEFAULT_ETA,
}

key, subkey = jax.random.split(key)
measles_obj.mif(
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
measles_obj.train(
    J=NP_FITR,
    M=NTRAIN,
    eta=eta,
    optimizer="Adam",
    n_monitors=1,
    alpha=0.97,
    alpha_cooling=1.0,
    eta_cooling=1.0,
)
measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
measles_obj.prune(n=1, refill=False)
measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, CLL=True)

measles_obj.print_summary()
print(measles_obj.time())

with open(f"IFAD_results/measles_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(measles_obj, f)
