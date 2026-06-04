"""
DPOP search for the single-unit London measles benchmark.
"""

# --- SLURM CONFIG ---
# jobs:
#   alpha08_80:
#     sbatch_args:
#       job-name: "pypomp dpop london measles 80"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 32GB
#       output: "dpop_results/logs/slurm-%j.out"
#       account: "ionides0"
#       time: "01:00:00"
#     env:
#       RUN_LEVEL: "4"
#       ALPHA: "0.8"
#       NFITR_WARM: "80"
#       NTRAIN: "80"
#       NP_FITR: "5000"
#       NP_EVAL: "5000"
#       NREPS_EVAL: "36"
#       DEFAULT_ETA: "0.002"
#       RHO_ETA: "0.0005"
#       COS_FINAL: "0.05"
#   alpha08_175:
#     sbatch_args:
#       job-name: "pypomp dpop london measles 175"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 32GB
#       output: "dpop_results/logs/slurm-%j.out"
#       account: "ionides0"
#       time: "02:00:00"
#     env:
#       RUN_LEVEL: "4"
#       ALPHA: "0.8"
#       NFITR_WARM: "175"
#       NTRAIN: "175"
#       NP_FITR: "5000"
#       NP_EVAL: "5000"
#       NREPS_EVAL: "36"
#       DEFAULT_ETA: "0.001"
#       RHO_ETA: "0.00025"
#       COS_FINAL: "0.05"
# --- END SLURM CONFIG ---

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import pickle

import jax
import pypomp as pp

from prep import ALPHA, BETA1, COOLING_RATE, RUN_LEVEL, RW_SD, key, panel


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


NFITR_WARM = env_int("NFITR_WARM", 80)
NTRAIN = env_int("NTRAIN", 80)
NP_FITR = env_int("NP_FITR", 5000)
NP_EVAL = env_int("NP_EVAL", 5000)
NREPS_EVAL = env_int("NREPS_EVAL", 36)
DEFAULT_ETA = env_float("DEFAULT_ETA", 0.002)
RHO_ETA = env_float("RHO_ETA", DEFAULT_ETA / 4)
COS_FINAL = env_float("COS_FINAL", 0.05)

base_eta = {
    name: DEFAULT_ETA
    for name in [
        "R0",
        "sigma",
        "gamma",
        "iota",
        "sigmaSE",
        "psi",
        "cohort",
        "amplitude",
        "S_0",
        "E_0",
        "I_0",
        "R_0",
    ]
}
base_eta["rho"] = RHO_ETA
eta = pp.LearningRate(base_eta).cosine_decay(final_factor=COS_FINAL, M=NTRAIN)

print(
    "DPOP London measles "
    f"ALPHA={ALPHA} RUN_LEVEL={RUN_LEVEL} BETA1={BETA1} "
    f"warm={NFITR_WARM} dpop={NTRAIN} J={NP_FITR} "
    f"evalJ={NP_EVAL} evalReps={NREPS_EVAL} "
    f"eta={DEFAULT_ETA} rho_eta={RHO_ETA} cosine_final={COS_FINAL}"
)

k_mif, k_dpop = jax.random.split(key)

panel.mif(rw_sd=RW_SD, M=NFITR_WARM, a=COOLING_RATE, J=NP_FITR, key=k_mif)
print(panel.results())

panel.dpop_train(
    J=NP_FITR,
    M=NTRAIN,
    eta=eta,
    optimizer="Adam",
    beta1=BETA1,
    alpha=ALPHA,
    process_weight_state="logw",
    key=k_dpop,
)
print(panel.results())

panel.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(panel.results())

panel.prune(n=1, refill=False)
panel.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(panel.results())

panel.print_summary()
print(panel.time())

os.makedirs("dpop_results", exist_ok=True)
eta_tag = str(DEFAULT_ETA).replace(".", "p")
out = (
    "dpop_results/"
    f"measles_london_dpop_rl{RUN_LEVEL}_alpha{ALPHA}_"
    f"J{NP_FITR}_M{NFITR_WARM}p{NTRAIN}_eta{eta_tag}.pkl"
)
with open(out, "wb") as f:
    pickle.dump(panel, f)
print(f"saved {out}")
