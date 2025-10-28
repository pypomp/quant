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
import psutil
import threading


def monitor_cpu(interval=1.0):
    def _monitor():
        while True:
            usages = psutil.cpu_percent(interval=interval, percpu=True)
            core_stats = " | ".join(f"Core {i}: {u:.1f}%" for i, u in enumerate(usages))
            print(f"[CPU] {core_stats}")

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()


# monitor_cpu()
# time.sleep(10)
# monitor_vram()
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
spx_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL)
print(spx_obj.results())

# start_time = time.time()
# spx_obj.train(J=NP_FITR, M=NTRAIN, key=subkey)
# print(f"train time taken: {time.time() - start_time} seconds")

# start_time = time.time()
# spx_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
# print(f"pfilter time taken: {time.time() - start_time} seconds")

print(spx_obj.time())

with open("spx_results.pkl", "wb") as f:
    pickle.dump(spx_obj, f)
