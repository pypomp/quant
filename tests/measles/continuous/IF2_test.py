"""
Tests pypomp on the continuous measles model using IF2. This is meant to be compared with the IFAD results to see which one maximizes the logLik better.
"""

# --- SLURM CONFIG ---
# sbatch_args:
#   job-name: "pypomp contiunous measles test (IF2)"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 30GB
#   output: "IF2_results/logs/slurm-%j.out"
#   account: "ionides0"
# run_levels:
#   1:
#     sbatch_args: { time: "00:03:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "00:20:00" }
#   4:
#     sbatch_args: { time: "00:10:00" }
# --- END SLURM CONFIG ---

import os
import pickle
import sys

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if tests_dir not in sys.path:
    sys.path.append(tests_dir)

import jax
import session_info
import utils
from setup import (
    COOLING_RATE,
    RUN_LEVEL,
    RW_SD,
    key,
    measles_obj,
)

session_info.show(dependencies=True)

NFITR = (2, 20, 200, 300)[RUN_LEVEL - 1]
NP_FITR = (2, 3, 20, 10000)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (36, 5, 24, 36)[RUN_LEVEL - 1]

key, subkey = jax.random.split(key)
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

with open(f"IF2_results/measles_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(measles_obj, f)

# ---- Save performance history ----

run_config = {
    "test": "continuous measles IF2",
    "partition": os.environ.get("SLURM_JOB_PARTITION", "local"),
}

metrics = utils.get_pomp_metrics(measles_obj, run_config=run_config, history_index=-2)
utils.append_history(metrics, "IF2_results/performance_history.jsonl")
print("Performance metrics saved to IF2_results/performance_history.jsonl")
