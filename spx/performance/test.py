import os
from datetime import datetime
from importlib.metadata import version

# Set JAX platform before importing JAX
USE_CPU = os.environ.get("USE_CPU", "false").lower() == "true"
if USE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "SLURM_CPUS_PER_TASK" in os.environ:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={os.environ['SLURM_CPUS_PER_TASK']}"
        )

import jax  # noqa: E402
import pickle  # noqa: E402
import pypomp as pp  # noqa: E402
import numpy as np  # noqa: E402

print(jax.devices())

print("Using CPU: ", USE_CPU)
now = datetime.now()
print("DATE: ", now.date())
print("TIME: ", now.time())
print("pypomp version:", version("pypomp"))
print("jax version:", version("jax"))

print(jax.devices())

MAIN_SEED = 631409
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))

NP_FITR = (2, 1000, 1000, 1000)[RUN_LEVEL - 1]
NFITR = (2, 20, 200, 200)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 20, 120 * 3)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000, 1000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24, 24)[RUN_LEVEL - 1]
NTRAIN = (2, 20, 40, 40)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

COOLING_RATE = 0.5
RW_SD = pp.RWSigma(
    sigmas={
        "mu": 0.02,
        "kappa": 0.02,
        "theta": 0.02,
        "xi": 0.02,
        "rho": 0.02,
        "V_0": 0.01,
    },
    init_names=["V_0"],
)

sp500_box = {
    "mu": [1e-6, 1e-4],
    "kappa": [1e-8, 0.1],
    "theta": [0.000075, 0.0002],
    "xi": [1e-8, 1e-2],
    "rho": [1e-8, 1],
    "V_0": [1e-10, 1e-4],
}

key, subkey = jax.random.split(key)
initial_params_list = pp.Pomp.sample_params(sp500_box, NREPS_FITR, key=subkey)

# implement Feller's condition
for params in initial_params_list:
    params["xi"] = float(
        np.random.uniform(
            low=0,
            high=np.sqrt(params["kappa"] * params["theta"] * 2),
        )
    )

spx_obj = pp.spx()

key, subkey = jax.random.split(key)
spx_obj.mif(
    theta=initial_params_list,
    rw_sd=RW_SD,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
)
print(spx_obj.results())
spx_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(spx_obj.results())

print(spx_obj.time())

out_dir = "cpu_results" if USE_CPU else "gpu_results"

with open(f"{out_dir}/spx_results_rl{RUN_LEVEL}.pkl", "wb") as f:
    pickle.dump(spx_obj, f)
