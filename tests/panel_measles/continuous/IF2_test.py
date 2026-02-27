# --- SLURM CONFIG ---
# sbatch_args:
#   job-name: "pypomp panel continuous measles test (IF2)"
#   partition: gpu
#   gpus: "v100:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "IF2_results/logs/slurm-%j.out"
#   account: "ionides0"
# run_levels:
#   1:
#     sbatch_args: { time: "00:03:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "03:00:00" }
#   4:
#     sbatch_args: { time: "03:00:00" }
# --- END SLURM CONFIG ---

import jax
import pickle
from setup import (
    key,
    RW_SD,
    COOLING_RATE,
    panel_measles_obj,
    RUN_LEVEL,
)

NFITR = (2, 20, 200, 200)[RUN_LEVEL - 1]
NP_FITR = (2, 3, 20, 10000)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 10000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

key, subkey = jax.random.split(key)
panel_measles_obj.mif(
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
panel_measles_obj.prune(n=1)
panel_measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, CLL=True)

panel_measles_obj.print_summary()
print(panel_measles_obj.time())

with open(f"IF2_results/panel_measles_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(panel_measles_obj, f)
