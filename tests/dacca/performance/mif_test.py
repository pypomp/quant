"""
This script runs the performance test for IF2 as the only fitting algorithm. The goal is to see how well IF2 can maximize the likelihood, as well as check how fast it runs. This can be compared with the IFAD Dacca script.

This script can use either the GPU or CPU.

The Dacca model has a very fast rproc, but many steps between observations, so this test can help determine if the overhead from interpolation steps is too high.
"""

# --- SLURM CONFIG ---
# importance: high
# sbatch_args:
#   job-name: "pypomp dacca test (mif)"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "mif_results/logs/slurm-%j.out"
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

from prep import (
    RUN_LEVEL,
    RW_SD,
    dacca_obj,
    initial_params_list,
    subkey,
)

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 5, 100, 650)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

# MIF round 1
dacca_obj.mif(
    theta=initial_params_list,
    rw_sd=RW_SD,
    M=NFITR,
    J=NP_FITR,
    key=subkey,
)
print(dacca_obj.results())

# PFILTER round 1
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

# Re-evaluate top fit to account for sample max luck
dacca_obj.prune(n=1, refill=False)
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

dacca_obj.print_summary()
print(dacca_obj.time())

with open(f"mif_results/dacca_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(dacca_obj, f)
