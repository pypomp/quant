# --- SLURM CONFIG ---
# sbatch_args:
#   job-name: "pypomp continuous measles test (IFAD)"
#   partition: gpu_mig40
#   gpus: "nvidia_a100_80gb_pcie_3g.40gb:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "IFAD_results/logs/slurm-%j.out"
#   account: "ionides0"
# run_levels:
#   1:
#     sbatch_args: { time: "00:03:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "00:30:00" }
#   4:
#     sbatch_args: { time: "01:00:00" }
# --- END SLURM CONFIG ---

import jax
import pickle
from setup import (
    key,
    RW_SD,
    COOLING_RATE,
    measles_obj,
    RUN_LEVEL,
)

NFITR = (2, 20, 100, 100)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
NP_FITR = (2, 3, 20, 10000)[RUN_LEVEL - 1]
NP_EVAL = (2, 20, 1000, 10000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

DEFAULT_ETA = 0.02
eta = {
    "R0": DEFAULT_ETA,
    "sigma": DEFAULT_ETA,
    "gamma": DEFAULT_ETA,
    "iota": DEFAULT_ETA,
    "rho": DEFAULT_ETA,
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
# measles_obj.mif(
#     rw_sd=RW_SD,
#     M=NFITR,
#     a=COOLING_RATE,
#     J=NP_FITR,
#     key=subkey,
# )
measles_obj.train(
    J=NP_FITR, M=NTRAIN, eta=eta, optimizer="Adam", n_monitors=1, key=subkey
)
measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
# measles_obj.prune(n=1, refill=False)
# measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, CLL=True)

measles_obj.print_summary()
print(measles_obj.time())

with open(f"IFAD_results/measles_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(measles_obj, f)
