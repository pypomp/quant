"""Dhaka: distribution of the pfilter log-likelihood at a fixed parameter vector.

The parameter vector is the published MLE that pomp ships as dacca()'s default,
so this is a direct check that pypomp's particle filter agrees with pomp's.
Compared against results/R/pfilter_logliks.csv.
"""

# --- SLURM CONFIG ---
# importance: medium
# description: "Dhaka: distribution of pfilter logLik at the published MLE"
# tags: [loglik, dacca, gpu]
# sbatch_args:
#   job-name: "dacca loglik check"
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
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "00:20:00" }
#   4:
#     sbatch_args: { time: "00:35:00" }
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

import jax
import model
import numpy as np
from utils import pfilter_logliks_frame, save_run

print(jax.devices())

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running at level {RUN_LEVEL}")

NP_EVAL = (2, 1000, 5000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 300, 300, 3600)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

dacca_obj = model.dacca()

started = time.time()
key, subkey = jax.random.split(key)
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
execution_time = time.time() - started
print(f"pfilter time taken: {execution_time:.2f} seconds")

print(dacca_obj.results())

platform = jax.devices()[0].platform
out_dir = os.path.join("results", platform)

save_run(
    dacca_obj,
    out_dir=out_dir,
    run_config={
        "kind": "loglik",
        "model": "dacca",
        "RUN_LEVEL": RUN_LEVEL,
        "MAIN_SEED": model.MAIN_SEED,
        "NP_EVAL": NP_EVAL,
        "NREPS_EVAL": NREPS_EVAL,
        "execution_time": execution_time,
    },
    execution_time=execution_time,
    write_traces=False,
)

logliks = pfilter_logliks_frame(dacca_obj)
logliks.to_csv(os.path.join(out_dir, "pfilter_logliks.csv"), index=False)
print(
    f"\npfilter_logliks.csv: {len(logliks)} replicates, "
    f"mean {logliks['logLik'].mean():.4f}, sd {logliks['logLik'].std():.4f}"
)
print(f"wrote {out_dir}/ (pfilter_logliks.csv, results.csv, timings.csv, latest.json)")
