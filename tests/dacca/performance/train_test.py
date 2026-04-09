"""
This script runs the performance test for IF2 + train (IFAD) as the only fitting algorithm. The goal is to see how well IFAD can maximize the likelihood, as well as check how fast it runs. This can be compared with the IFAD Dacca script.

This script can use either the GPU or CPU.

The Dacca model has a very fast rproc, but many steps between observations, so this test can help determine if the overhead from interpolation steps is too high.
"""

# --- SLURM CONFIG ---
# sbatch_args:
#   job-name: "pypomp dacca test (train)"
#   partition: gpu
#   gpus: "v100:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "train_results/logs/slurm-%j.out"
#   account: "ionides0"
# run_levels:
#   1:
#     sbatch_args: { time: "00:01:00" }
#   2:
#     sbatch_args: { time: "00:30:00" }
#   3:
#     sbatch_args: { time: "00:30:00" }
#   4:
#     sbatch_args: { time: "00:30:00" }
# --- END SLURM CONFIG ---

import pickle

import jax
from prep import (
    COOLING_RATE,
    RUN_LEVEL,
    RW_SD,
    dacca_obj,
    initial_params_list,
    key,
    subkey,
)

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 5, 100, 100)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

DEFAULT_ETA = 0.2
eta = {
    "gamma": DEFAULT_ETA,
    "epsilon": DEFAULT_ETA,
    "rho": 0.0,
    "m": DEFAULT_ETA,
    "c": 0.0,
    "beta_trend": DEFAULT_ETA,
    **{f"bs{i + 1}": DEFAULT_ETA for i in range(6)},
    "sigma": DEFAULT_ETA,
    "tau": DEFAULT_ETA,
    "omega": DEFAULT_ETA,
    **{f"omegas{i + 1}": DEFAULT_ETA for i in range(6)},
}

# MIF step
key, subkey = jax.random.split(key)
dacca_obj.mif(
    theta=initial_params_list,
    rw_sd=RW_SD,
    M=60,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
print(dacca_obj.results())

# Train step
dacca_obj.train(J=NP_FITR, M=40, eta=eta, optimizer="Adam", n_monitors=1)
print(dacca_obj.results())

# # PFILTER round 2
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

# Re-evaluate top fit to account for sample max luck
dacca_obj.prune(n=1, refill=False)
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

dacca_obj.print_summary()
print(dacca_obj.time())

with open(f"train_results/dacca_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(dacca_obj, f)
