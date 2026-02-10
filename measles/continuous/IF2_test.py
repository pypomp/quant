import jax
import pickle
from setup import (
    key,
    RW_SD,
    COOLING_RATE,
    measles_obj,
    RUN_LEVEL,
)

NFITR = (2, 20, 200, 200)[RUN_LEVEL - 1]
NP_FITR = (2, 3, 20, 10000)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 10000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

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
