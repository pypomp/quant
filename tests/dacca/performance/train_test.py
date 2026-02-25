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

"""
This script runs the performance test for IF2 + train.
"""

import pickle
from prep import (
    dacca_obj,
    key,
    subkey,
    initial_params_list,
    RW_SD,
    COOLING_RATE,
    RUN_LEVEL,
)
import jax

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 5, 100, 100)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

eta = {
    "gamma": 0.2,
    "epsilon": 0.2,
    "rho": 0.0,
    "m": 0.2,
    "c": 0.0,
    "beta_trend": 0.2,
    **{f"bs{i + 1}": 0.2 for i in range(6)},
    "sigma": 0.2,
    "tau": 0.2,
    "omega": 0.2,
    **{f"omegas{i + 1}": 0.2 for i in range(6)},
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

# PFILTER round 1
# dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
# print(dacca_obj.results())

# # Prune step
# dacca_obj.prune(n=10, refill=True)

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
