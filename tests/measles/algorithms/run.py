"""Measles: IF2 alone against IF2 followed by gradient training (IFAD).

Both arms start from the same points and are given a comparable wall-clock
budget, so the question the test answers is which algorithm reaches the higher
likelihood for that budget -- not which is faster.

This kind uses the continuous-time model (003) rather than the 001b model the
R comparisons use: gradient training needs a differentiable measurement model,
which the discrete variant does not provide. There is no R counterpart.

The arm is selected by the ALGORITHM environment variable and writes into
results/<algorithm>/.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "Measles: IF2 vs IFAD likelihood maximization at a matched budget (continuous model)"
# tags: [algorithms, measles, gpu]
# jobs:
#   if2:
#     sbatch_args:
#       job-name: "measles algorithms (if2)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 30GB
#       output: "results/if2/logs/slurm-%j.out"
#     env:
#       ALGORITHM: "if2"
#   ifad:
#     sbatch_args:
#       job-name: "measles algorithms (ifad)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 30GB
#       output: "results/ifad/logs/slurm-%j.out"
#     env:
#       ALGORITHM: "ifad"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:03:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "00:30:00" }
#   4:
#     sbatch_args: { time: "00:30:00" }
# --- END SLURM CONFIG ---

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
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

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

NSTARTS = (2, 3, 20, 36)[RUN_LEVEL - 1]
NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

# IF2 alone gets the iterations that IFAD spends on mif and train combined,
# plus the extra it takes for the two arms to cost about the same.
if ALGORITHM == "if2":
    NFITR = (2, 20, 200, 350)[RUN_LEVEL - 1]
    NTRAIN = 0
else:
    NFITR = (2, 20, 100, 300)[RUN_LEVEL - 1]
    NTRAIN = (2, 20, 40, 50)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

key, subkey = jax.random.split(key)
starts = model.sample_starts(NSTARTS, key=subkey)

measles_obj = model.measles_continuous(starts)

WARMUP = (1, 5, 10, 10)[RUN_LEVEL - 1]

DEFAULT_ETA = 0.01


def warmed(v):
    """Ramp a learning rate from a tenth of `v` up to `v` over WARMUP iterations.

    Passing a schedule rather than a scalar is also what keeps pypomp's
    `LearningRate._canonicalize` on its 2-D path; the scalar path indexes a
    1-D JAX array with a Python list and raises.
    """
    if v == 0.0:
        return 0.0
    return np.concatenate(
        [np.linspace(v * 0.1, v, WARMUP), np.full(NTRAIN - WARMUP, v)]
    )


started = time.time()

key, subkey = jax.random.split(key)
measles_obj.mif(
    theta=starts,
    rw_sd=model.CONTINUOUS_RW_SD,
    M=NFITR,
    J=NP_FITR,
    key=subkey,
)
print(measles_obj.results())

if ALGORITHM == "ifad":
    # rho is the reporting probability; it moves on a much finer scale than
    # the rate parameters, so it gets an eighth of the step size.
    eta = pp.LearningRate(
        {
            name: warmed(DEFAULT_ETA / 8 if name == "rho" else DEFAULT_ETA)
            for name in model.BOX
        }
    ).cosine_decay(final_factor=0.05, M=NTRAIN)

    measles_obj.train(
        J=NP_FITR,
        M=NTRAIN,
        eta=eta,
        optimizer=pp.Adam(),
        n_monitors=1,
    )
    print(measles_obj.results())

measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(measles_obj.results())

# Everything the report needs about the search is in this entry: one row per
# start, with its evaluated logLik and its final parameter vector. Capture it
# before pruning drops the other starts.
all_starts_index = len(measles_obj.results_history) - 1
results_final = measles_obj.results(index=all_starts_index)
logliks = pfilter_logliks_frame(measles_obj, history_index=all_starts_index)

# The sample maximum is optimistic, so the top fit is re-evaluated on its own.
measles_obj.prune(n=1, refill=False)
measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, CLL=True)
print(measles_obj.results())

execution_time = time.time() - started

measles_obj.print_summary()
print(measles_obj.time())

out_dir = os.path.join("results", ALGORITHM)
save_run(
    measles_obj,
    out_dir=out_dir,
    run_config={
        "kind": "algorithms",
        "model": "measles",
        "variant": "003 (continuous)",
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

res = measles_obj.results()
if "logLik" in res.columns:
    print(f"\nbest logLik: {res['logLik'].max():.4f}")
print(
    f"wrote {out_dir}/ (results.csv, results_final.csv, pfilter_logliks.csv, "
    "traces.csv.gz, timings.csv, latest.json)"
)
