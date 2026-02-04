import os
import time

# Set JAX platform before importing JAX
if os.environ.get("USE_CPU", "false").lower() == "true":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import pickle
import jax.numpy as jnp
import pypomp as pp
import numpy as np

print(jax.devices())

MAIN_SEED = 631409
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))

NP_FITR = (2, 500, 1000, 5000)[RUN_LEVEL - 1]
NFITR = (2, 5, 100, 100)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 20, 36)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 36)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

RW_SD = pp.RWSigma(
    sigmas={
        "gamma": 0.02,
        "m": 0.02,
        "rho": 0.0,
        "epsilon": 0.02,
        "omega": 0.02,
        "c": 0.02,
        "beta_trend": 0.02,
        "sigma": 0.02,
        "tau": 0.02,
        "bs1": 0.02,
        "bs2": 0.02,
        "bs3": 0.02,
        "bs4": 0.02,
        "bs5": 0.02,
        "bs6": 0.02,
        "omegas1": 0.02,
        "omegas2": 0.02,
        "omegas3": 0.02,
        "omegas4": 0.02,
        "omegas5": 0.02,
        "omegas6": 0.02,
    },
    init_names=[],
)
COOLING_RATE = 0.5

dacca_obj = pp.dacca(dt=None, nstep=20)

# params_box = {k: [v * 0.5, v * 1.5] for k, v in dacca_obj.theta[0].items()}
# params_box["rho"] = [0.0, 0.0]
params_box = {
    "gamma": [10.0, 40.0],
    "m": [0.03, 0.60],
    "rho": [0.0, 0.0],
    "epsilon": [0.20, 30.0],
    "omega": [float(jnp.exp(-4.5)), float(jnp.exp(-4.5))],
    "c": [1.0, 1.0],
    "beta_trend": [-0.01, 0.00],
    "sigma": [1.0, 5.0],
    "tau": [0.10, 0.50],
    "bs1": [-4.0, 4.0],
    "bs2": [0.0, 8.0],
    "bs3": [-4.0, 4.0],
    "bs4": [0.0, 8.0],
    "bs5": [0.0, 8.0],
    "bs6": [0.0, 8.0],
    "omegas1": [-10.0, 0.0],
    "omegas2": [-10.0, 0.0],
    "omegas3": [-10.0, 0.0],
    "omegas4": [-10.0, 0.0],
    "omegas5": [-10.0, 0.0],
    "omegas6": [-10.0, 0.0],
}

key, subkey = jax.random.split(key)
initial_params_list = pp.Pomp.sample_params(params_box, NREPS_FITR, key=subkey)


# options = jax.profiler.ProfileOptions()
# options.gpu_max_activity_api_events = 1028 * 1024 * 4
# with jax.profiler.trace("dacca_profiler", profiler_options=options):

# MIF round 1
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
# dacca_obj.train(J=NP_FITR, M=NTRAIN, eta=0.2)

# MIF round 2
RW_SD.cool(0.25)
dacca_obj.mif(
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
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

with open("dacca_results.pkl", "wb") as f:
    pickle.dump(dacca_obj, f)
