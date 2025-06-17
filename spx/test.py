import os

# Must be done before importing jax
# os.environ["XLA_FLAGS"] = (
#     "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8 "
# )
import time
import jax

# jax.config.update("jax_num_cpu_devices", 8)
import jax.numpy as jnp
import pypomp as pp
import pypomp.spx as pps
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
print(jax.devices())

MAIN_SEED = 631409
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

RUN_LEVEL = 2

NP_FITR = (2, 1000, 1000)[RUN_LEVEL - 1]
NFITR = (2, 20, 200)[RUN_LEVEL - 1]
NREPS_FITR = (2, 3, 120)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 1000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 24)[RUN_LEVEL - 1]
print(f"Running at level {RUN_LEVEL}")

RW_SD = jnp.array([0.02, 0.02, 0.02, 0.02, 0.02, 0])
RW_SD_INIT = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
COOLING_RATE = 0.5


def rho_transform(lst):
    """Transform rho to perturbation scale"""
    out = [np.log((1 + x) / (1 - x)) for x in lst]
    return out


sp500_box = {
    "mu": [1e-6, 1e-4],
    "kappa": [1e-8, 0.1],
    "theta": [0.000075, 0.0002],
    "xi": [1e-8, 1e-2],
    "rho": [-0.9, 0.9],
    "V_0": [1e-6, 1e-4],
}

key, subkey = jax.random.split(key)
initial_params_list = pp.Pomp.sample_params(sp500_box, NREPS_FITR, key=subkey)

transformed_params_list = []
for params in initial_params_list:
    transformed_params = {}
    for k, v in params.items():
        if k == "rho":
            transformed_params[k] = float(rho_transform([v])[0])
        else:
            transformed_params[k] = float(jnp.log(v))
    transformed_params_list.append(transformed_params)

# implement Feller's condition
for params in transformed_params_list:
    params["xi"] = float(
        np.random.uniform(
            low=0,
            high=np.log(np.sqrt(np.exp(params["kappa"]) * np.exp(params["theta"]) * 2)),
        )
    )

spx = pps.spx()

print("Starting IF2")
key, subkey = jax.random.split(key)
start_time = time.time()
spx.mif(
    theta=transformed_params_list,
    sigmas=RW_SD,
    sigmas_init=RW_SD_INIT,
    M=NFITR,
    a=COOLING_RATE,
    J=NP_FITR,
    key=subkey,
    n_monitors=0,
)
print(spx.results[-1]["logLiks"][0].mean())
print(f"mif time taken: {time.time() - start_time} seconds")

start_time = time.time()
spx.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey)
print(spx.results[-1]["logLiks"][0].mean())
print(f"pfilter time taken: {time.time() - start_time} seconds")
pass
