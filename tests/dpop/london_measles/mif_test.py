"""
Standalone IF2 baseline for the single-unit London measles DPOP benchmark.
"""

# --- SLURM CONFIG ---
# sbatch_args:
#   job-name: "pypomp dpop london measles IF2"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 20GB
#   output: "mif_results/logs/slurm-%j.out"
#   account: "ionides0"
#   time: "02:00:00"
# env:
#   RUN_LEVEL: "4"
# run_levels:
#   1:
#     sbatch_args: { time: "00:05:00", mem: "6GB" }
#   4:
#     sbatch_args: { time: "02:00:00", mem: "20GB" }
# --- END SLURM CONFIG ---

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import pickle

from prep import RUN_LEVEL, RW_SD, key, panel

NFITR = (2, 5, 100, 650)[RUN_LEVEL - 1]
NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]
COOLING_RATE_IF2 = 0.8

panel.mif(rw_sd=RW_SD, M=NFITR, a=COOLING_RATE_IF2, J=NP_FITR, key=key)

panel.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(panel.results())

panel.prune(n=1, refill=False)
panel.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(panel.results())

panel.print_summary()
print(panel.time())

os.makedirs("mif_results", exist_ok=True)
with open(f"mif_results/measles_london_mif_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(panel, f)
