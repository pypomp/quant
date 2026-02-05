"""
This script runs the performance test for IF2 + train.
"""

import pickle
from prep import (
    dacca_obj,
    key,
    subkey,
    initial_params_list,
    RW_SD,
    COOLING_RATE,
    RUN_LEVEL,
)
import jax

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 5, 100, 100)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

# options = jax.profiler.ProfileOptions()
# options.gpu_max_activity_api_events = 1028 * 1024 * 4
# with jax.profiler.trace("dacca_profiler", profiler_options=options):

# MIF step
key, subkey = jax.random.split(key)
dacca_obj.mif(
    theta=initial_params_list,
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
print(dacca_obj.results())

# PFILTER round 1
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

# Prune step
dacca_obj.prune(n=10, refill=True)

# Train step
dacca_obj.train(J=NP_FITR, M=NTRAIN, eta=0.2)
print(dacca_obj.results())

# PFILTER round 2
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

# Re-evaluate top fit to account for sample max luck
dacca_obj.prune(n=1, refill=False)
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

dacca_obj.print_summary()
print(dacca_obj.time())

with open(f"train_results/dacca_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(dacca_obj, f)
