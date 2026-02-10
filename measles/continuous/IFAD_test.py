import jax
import pickle
from setup import (
    key,
    RW_SD,
    COOLING_RATE,
    measles_obj,
    RUN_LEVEL,
)

NFITR = (2, 20, 200, 100)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
NP_FITR = (2, 3, 20, 10000)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 10000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]

DEFAULT_ETA = 0.0
eta = {
    "mu": 0.0,
    "alpha": 0.0,
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
measles_obj.mif(
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
measles_obj.train(J=NP_FITR, M=NTRAIN, eta=eta, optimizer="SGD", n_monitors=1)
measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
# measles_obj.prune(n=1, refill=False)
# measles_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, CLL=True)

measles_obj.print_summary()
print(measles_obj.time())

with open(f"IFAD_results/measles_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(measles_obj, f)
