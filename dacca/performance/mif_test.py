"""
This script runs the performance test for IF2 as the only fitting algorithm. It can use either the CPU or GPU.
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
    USE_CPU,
)
import jax

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 5, 100, 250)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

# options = jax.profiler.ProfileOptions()
# options.gpu_max_activity_api_events = 1028 * 1024 * 4
# with jax.profiler.trace("dacca_profiler", profiler_options=options):

# MIF round 1
key, subkey = jax.random.split(key)
dacca_obj.mif(
    theta=initial_params_list,
    rw_sd=RW_SD,
    M=60 + 40 * 3 + 50,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
print(dacca_obj.results())

# PFILTER round 1
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

# Prune step
# dacca_obj.prune(n=10, refill=True)
# dacca_obj.train(J=NP_FITR, M=NTRAIN, eta=0.2)

# MIF round 2
# RW_SD.cool(0.25)
# dacca_obj.mif(
#     rw_sd=RW_SD,
#     M=NFITR,
#     a=COOLING_RATE,
#     J=NP_FITR,
#     key=subkey,
# )
# print(dacca_obj.results())

# # PFILTER round 2
# dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
# print(dacca_obj.results())

# Re-evaluate top fit to account for sample max luck
dacca_obj.prune(n=1, refill=False)
dacca_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(dacca_obj.results())

dacca_obj.print_summary()
print(dacca_obj.time())

# Save results
out_dir = "cpu_results" if USE_CPU else "gpu_results"
with open(f"{out_dir}/dacca_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(dacca_obj, f)
