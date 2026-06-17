"""
This script runs the performance test for IF2 + train (IFAD) as the only fitting algorithm. The goal is to see how well IFAD can maximize the likelihood, as well as check how fast it runs. This can be compared with the IFAD Dacca script.

This script can use either the GPU or CPU.

The Dacca model has a very fast rproc, but many steps between observations, so this test can help determine if the overhead from interpolation steps is too high.
"""

# --- SLURM CONFIG ---
# importance: high
# sbatch_args:
#   job-name: "pypomp dacca test (train)"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 10GB
#   output: "train_results/logs/slurm-%j.out"
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

import numpy as np
import pypomp as pp
from prep import (
    RUN_LEVEL,
    RW_SD,
    dacca_obj,
    initial_params_list,
    subkey,
)

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 5, 100, 175)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 175)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]
warmup = (1, 5, 10, 10)[RUN_LEVEL - 1]


def w(v):
    if v == 0.0:
        return 0.0
    return np.concatenate(
        [np.linspace(v * 0.1, v, warmup), np.full(NTRAIN - warmup, v)]
    )


DEFAULT_ETA = 0.1
DEFAULT_IVP_ETA = DEFAULT_ETA / 2
eta = pp.LearningRate(
    {
        "gamma": w(DEFAULT_ETA * 0.5),
        "epsilon": w(DEFAULT_ETA),
        "rho": 0.0,
        "m": w(DEFAULT_ETA),
        "c": 0.0,
        "alpha": 0.0,
        "delta": 0.0,
        "beta_trend": w(DEFAULT_ETA * 0.5),
        **{f"bs{i + 1}": w(DEFAULT_ETA) for i in range(6)},
        "sigma": w(DEFAULT_ETA * 0.5),
        "tau": w(DEFAULT_ETA * 0.5),
        **{f"omegas{i + 1}": w(DEFAULT_ETA) for i in range(6)},
        "S_0": w(DEFAULT_IVP_ETA),
        "I_0": w(DEFAULT_IVP_ETA),
        "Y_0": 0.0,
        "R1_0": w(DEFAULT_IVP_ETA),
        "R2_0": w(DEFAULT_IVP_ETA),
        "R3_0": w(DEFAULT_IVP_ETA),
    }
).cosine_decay(final_factor=0.05, M=NTRAIN)

# MIF step
dacca_obj.mif(
    theta=initial_params_list,
    rw_sd=RW_SD,
    M=NFITR,
    J=NP_FITR,
    key=subkey,
)
print(dacca_obj.results())

# Train step
dacca_obj.train(
    J=NP_FITR,
    M=NTRAIN,
    eta=eta,
    optimizer=pp.Adam(),
    n_monitors=1,
)
print(dacca_obj.results())

# PFILTER round 2
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
