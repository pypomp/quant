"""SPX: distribution of the pfilter log-likelihood at a fixed parameter vector.

Kind: loglik -- theta is pinned to the Sun (2024) estimates and only the
replicate seed varies. The question is whether pypomp's particle filter
produces the *right distribution* of likelihood estimates, not whether it can
find good parameters.

Cost is driven purely by the replicate count, and the precision of the
comparison scales as 1/sqrt(reps): going from 3600 replicates to 225 costs 4x
in the standard error of the mean for 16x less compute. Level 3 is the routine
setting; level 4 reproduces the 3600-replicate archival run that the frozen R
baseline uses.

Compared against R_reference/pfilter_logliks.csv (3600 R replicates).
"""

# --- SLURM CONFIG ---
# importance: medium
# description: "SPX: distribution of pfilter logLik at the Sun (2024) estimates"
# tags: [loglik, spx, gpu]
# sbatch_args:
#   job-name: "spx loglik check"
#   partition: gpu
#   gpus: "v100:1"
#   cpus-per-gpu: 1
#   mem: 6GB
#   output: "results/logs/slurm-%j.out"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:02:00" }
#   2:
#     sbatch_args: { time: "00:10:00" }
#   3:
#     sbatch_args: { time: "00:30:00" }
#   4:
#     sbatch_args: { time: "02:00:00" }
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

import jax  # noqa: E402
import numpy as np  # noqa: E402

import model  # noqa: E402
from utils import pfilter_logliks_frame, save_run  # noqa: E402

print(jax.devices())

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "2"))
print(f"Running at level {RUN_LEVEL}")

NP_EVAL = (2, 1000, 1000, 1000)[RUN_LEVEL - 1]
# Level 3 is the routine setting; level 4 matches the archival R baseline.
NREPS_EVAL = (2, 100, 225, 3600)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

spx_obj = model.spx()

started = time.time()
key, subkey = jax.random.split(key)
spx_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey, theta=model.SUN2024_THETA)
execution_time = time.time() - started

print(spx_obj.results())

# Label the output by the device actually used, matching the other SPX kinds.
platform = jax.devices()[0].platform
out_dir = os.path.join("results", platform)

# theta is fixed, so there are no per-iteration traces worth writing.
metrics = save_run(
    spx_obj,
    out_dir=out_dir,
    run_config={
        "kind": "loglik",
        "model": "spx",
        "RUN_LEVEL": RUN_LEVEL,
        "NP_EVAL": NP_EVAL,
        "NREPS_EVAL": NREPS_EVAL,
        "MAIN_SEED": model.MAIN_SEED,
        "theta": "sun2024",
    },
    execution_time=execution_time,
    write_traces=False,
)

# The per-replicate log-likelihoods are the point of this kind, and what the
# frozen R baseline holds. results.csv carries only the aggregated logmeanexp,
# which sits ~sd^2/2 above the mean and is not comparable to R's per-replicate
# numbers.
logliks = pfilter_logliks_frame(spx_obj)
logliks.to_csv(os.path.join(out_dir, "pfilter_logliks.csv"), index=False)
print(
    f"\npfilter_logliks.csv: {len(logliks)} replicates, "
    f"mean {logliks['logLik'].mean():.4f}, sd {logliks['logLik'].std():.4f}"
)

print(f"logLik stats: {metrics['loglik_stats']}")
print(
    f"wrote {out_dir}/ (pfilter_logliks.csv, results.csv, latest.json, history.jsonl)"
)
