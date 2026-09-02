"""Dhaka: IF2 alone against IF2 followed by gradient training (IFAD).

Both arms start from the same points and are given a comparable wall-clock
budget, so the question the test answers is which algorithm reaches the higher
likelihood for that budget -- not which is faster.

The arm is selected by the ALGORITHM environment variable and writes into
results/<algorithm>/.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "Dhaka: IF2 vs IFAD likelihood maximization at a matched budget"
# tags: [algorithms, dacca, gpu]
# jobs:
#   if2:
#     sbatch_args:
#       job-name: "dacca algorithms (if2)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 6GB
#       output: "results/if2/logs/slurm-%j.out"
#     env:
#       ALGORITHM: "if2"
#   ifad:
#     sbatch_args:
#       job-name: "dacca algorithms (ifad)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 10GB
#       output: "results/ifad/logs/slurm-%j.out"
#     env:
#       ALGORITHM: "ifad"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:01:00" }
#   2:
#     sbatch_args: { time: "00:30:00" }
#   3:
#     sbatch_args: { time: "00:30:00" }
#   4:
#     sbatch_args: { time: "00:18:00" }
# --- END SLURM CONFIG ---

# JAX must be configured before it is imported, so imports below sit after
# the environment setup. A file-level directive keeps this from tripping E402
# even after the editor reorganises the import block.

import os
import sys
import time

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if tests_dir not in sys.path:
    sys.path.append(tests_dir)
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if model_dir not in sys.path:
    sys.path.append(model_dir)

# JAX reads these at import time, so they must be set before it is imported.
USE_CPU = os.environ.get("USE_CPU", "false").lower() == "true"
if USE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "SLURM_CPUS_PER_TASK" in os.environ:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={os.environ['SLURM_CPUS_PER_TASK']}"
        )

import jax
import model
import numpy as np
import pypomp as pp
from utils import pfilter_logliks_frame, save_run

print(jax.devices())
print("Using CPU:", USE_CPU)

ALGORITHM = os.environ.get("ALGORITHM", "if2").lower()
if ALGORITHM not in ("if2", "ifad"):
    raise ValueError(f"ALGORITHM must be 'if2' or 'ifad', got {ALGORITHM!r}")

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running {ALGORITHM} at level {RUN_LEVEL}")

NSTARTS = (2, 3, 20, 100)[RUN_LEVEL - 1]
NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

# IF2 alone gets the iterations that IFAD spends on mif and train combined,
# plus the extra it takes for the two arms to cost about the same.
if ALGORITHM == "if2":
    NFITR = (2, 5, 100, 1000)[RUN_LEVEL - 1]
    NTRAIN = 0
else:
    NFITR = (2, 5, 100, 175)[RUN_LEVEL - 1]
    NTRAIN = (2, 20, 40, 175)[RUN_LEVEL - 1]

WARMUP = (1, 5, 10, 10)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

key, subkey = jax.random.split(key)
starts = model.sample_starts(NSTARTS, key=subkey)

dacca_obj = model.dacca()


def warmed(v):
    """Ramp a learning rate from a tenth of `v` up to `v` over WARMUP iterations."""
    if v == 0.0:
        return 0.0
    return np.concatenate(
        [np.linspace(v * 0.1, v, WARMUP), np.full(NTRAIN - WARMUP, v)]
    )


DEFAULT_ETA = 0.1
DEFAULT_IVP_ETA = DEFAULT_ETA / 2

started = time.time()

key, subkey = jax.random.split(key)
dacca_obj.mif(
    theta=starts,
    rw_sd=model.RW_SD,
    M=NFITR,
    J=NP_FITR,
    key=subkey,
)
print(dacca_obj.results())

if ALGORITHM == "ifad":
    eta = pp.LearningRate(
        {
            "gamma": warmed(DEFAULT_ETA * 0.5),
            "epsilon": warmed(DEFAULT_ETA),
            "rho": 0.0,
            "m": warmed(DEFAULT_ETA),
            "c": 0.0,
            "alpha": 0.0,
            "delta": 0.0,
            "beta_trend": warmed(DEFAULT_ETA * 0.5),
            **{f"bs{i + 1}": warmed(DEFAULT_ETA) for i in range(6)},
            "sigma": warmed(DEFAULT_ETA * 0.5),
            "tau": warmed(DEFAULT_ETA * 0.5),
            **{f"omegas{i + 1}": warmed(DEFAULT_ETA) for i in range(6)},
            "S_0": warmed(DEFAULT_IVP_ETA),
            "I_0": warmed(DEFAULT_IVP_ETA),
            "Y_0": 0.0,
            "R1_0": warmed(DEFAULT_IVP_ETA),
            "R2_0": warmed(DEFAULT_IVP_ETA),
            "R3_0": warmed(DEFAULT_IVP_ETA),
        }
    ).cosine_decay(final_factor=0.05, M=NTRAIN)

    dacca_obj.train(
        J=NP_FITR,
        M=NTRAIN,
        eta=eta,
        optimizer=pp.Adam(),
        n_monitors=1,
    )
    print(dacca_obj.results())

dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

# Everything the report needs about the search is in this entry: one row per
# start, with its evaluated logLik and its final parameter vector. Capture it
# before pruning drops the other starts.
all_starts_index = len(dacca_obj.results_history) - 1
results_final = dacca_obj.results(index=all_starts_index)
logliks = pfilter_logliks_frame(dacca_obj, history_index=all_starts_index)

# The sample maximum is optimistic, so the top fit is re-evaluated on its own.
dacca_obj.prune(n=1, refill=False)
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

execution_time = time.time() - started

dacca_obj.print_summary()
print(dacca_obj.time())

out_dir = os.path.join("results", ALGORITHM)
save_run(
    dacca_obj,
    out_dir=out_dir,
    run_config={
        "kind": "algorithms",
        "model": "dacca",
        "ALGORITHM": ALGORITHM,
        "RUN_LEVEL": RUN_LEVEL,
        "USE_CPU": USE_CPU,
        "MAIN_SEED": model.MAIN_SEED,
        "NSTARTS": NSTARTS,
        "NFITR": NFITR,
        "NTRAIN": NTRAIN,
        "NP_FITR": NP_FITR,
        "NP_EVAL": NP_EVAL,
        "NREPS_EVAL": NREPS_EVAL,
        "execution_time": execution_time,
        "platform": jax.devices()[0].platform,
    },
    execution_time=execution_time,
)

results_final.to_csv(os.path.join(out_dir, "results_final.csv"), index=False)
logliks.to_csv(os.path.join(out_dir, "pfilter_logliks.csv"), index=False)

res = dacca_obj.results()
if "logLik" in res.columns:
    print(f"\nbest logLik: {res['logLik'].max():.4f}")
print(
    f"wrote {out_dir}/ (results.csv, results_final.csv, pfilter_logliks.csv, "
    "traces.csv.gz, timings.csv, latest.json)"
)
