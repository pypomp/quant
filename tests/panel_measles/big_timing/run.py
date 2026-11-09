"""Panel measles: full 1422-unit, 36-start fit timed on the CuTe backend (Philox RNG).

Reruns pyMeasles lab/model_001b/search_005_u1422_timing at --units=1422 --reps 36
(model 001b, all parameters unit-specific, same box, seed and stage sequence:
mif -> pfilter -> mix_and_match/prune -> pfilter with CLL) and compares the
per-stage wall clock with that run's, which used pypomp 1.0.8 on XLA.

Level 3 is the full-size problem cut to 5 mif iterations and 4 evaluation reps,
and projects the level 4 runtime from its own rates.
"""

# --- SLURM CONFIG ---
# importance: low
# description: "Panel measles: 1422-unit x 36-start block-IF2 fit timed on CuTe (philox)"
# tags: [timing, panel_measles, gpu, cute, big]
# jobs:
#   gpu_cute:
#     sbatch_args:
#       job-name: "panel measles big timing (gpu, cute)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 48GB
#       output: "results/gpu_cute/logs/slurm-%j.out"
#     env:
#       BACKEND: "cute"
#       CUTE_RNG: "philox"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:10:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "01:30:00" }
#   4:
#     sbatch_args: { time: "2-00:00:00" }
# --- END SLURM CONFIG ---

import json
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
import numpy as np
import pandas as pd
import pypomp as pp
from units import UNITS
from utils import apply_backend, backend_suffix, run_metadata

print(jax.devices())

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running at level {RUN_LEVEL}")

N_UNITS = (5, 20, 1422, 1422)[RUN_LEVEL - 1]
NP_FITR = (2, 1000, 10000, 10000)[RUN_LEVEL - 1]
NFITR = (2, 5, 5, 200)[RUN_LEVEL - 1]
NREPS_FITR = (2, 6, 36, 36)[RUN_LEVEL - 1]
NP_EVAL = (2, 1000, 5000, 5000)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 4, 36)[RUN_LEVEL - 1]

FULL_NFITR = 200
FULL_NREPS_EVAL = 36

# pyMeasles search_005_u1422_timing, results/2026-10-05_134449_u1422_L4_d001_reps36
REFERENCE = {
    "source": "pyMeasles lab/model_001b/search_005_u1422_timing "
    "(2026-10-05_134449_u1422_L4_d001_reps36)",
    "pypomp_version": "1.0.8",
    "backend": "xla",
    "gpu": "NVIDIA RTX PRO 6000 Blackwell (gl1805)",
    "slurm_job_id": "63275037",
    "gpu_peak_gb": 31.17,
    "stage_wall_s": {
        "build": 48.97,
        "mif": 127866.61,
        "pfilter_eval1": 8933.12,
        "mix_and_match_prune": 0.32,
        "pfilter_final": 338.17,
        "total": 137188.32,
    },
}

MAIN_SEED = 631409
key = jax.random.key(MAIN_SEED)
np.random.seed(MAIN_SEED)

DEFAULT_SD = 0.02
DEFAULT_IVP_SD = DEFAULT_SD * 12
COOLING_RATE = 0.5

RW_SD = pp.RWSigma(
    sigmas={
        "R0": DEFAULT_SD * 0.25,
        "sigma": DEFAULT_SD * 0.25,
        "gamma": DEFAULT_SD * 0.5,
        "iota": DEFAULT_SD,
        "rho": DEFAULT_SD * 0.5,
        "sigmaSE": DEFAULT_SD,
        "psi": DEFAULT_SD * 0.25,
        "cohort": DEFAULT_SD * 0.5,
        "amplitude": DEFAULT_SD * 0.5,
        "S_0": DEFAULT_IVP_SD,
        "E_0": DEFAULT_IVP_SD,
        "I_0": DEFAULT_IVP_SD,
        "R_0": DEFAULT_IVP_SD,
    },
    init_names=["S_0", "E_0", "I_0", "R_0"],
).geometric_cooling(COOLING_RATE)

BOX = {
    "R0": (10.0, 60.0),
    "sigma": (25.0, 100.0),
    "gamma": (25.0, 320.0),
    "iota": (0.004, 3.0),
    "rho": (0.1, 0.9),
    "sigmaSE": (0.04, 0.1),
    "psi": (0.05, 3.0),
    "cohort": (0.1, 0.7),
    "amplitude": (0.1, 0.6),
    "S_0": (0.01, 0.07),
    "E_0": (0.000004, 0.0001),
    "I_0": (0.000003, 0.001),
    "R_0": (0.9, 0.99),
}

active_units = UNITS[:N_UNITS]
print(f"Using {len(active_units)} units")

key, subkey = jax.random.split(key)
initial_params = pp.PanelPomp.sample_params(
    BOX, n=NREPS_FITR, units=active_units, key=subkey, shared_names=[]
)

stage_wall_s: dict[str, float] = {}


def timed(stage, fn):
    started = time.perf_counter()
    out = fn()
    stage_wall_s[stage] = time.perf_counter() - started
    print(f"  {stage}: {stage_wall_s[stage]:.1f} s", flush=True)
    return out


t_script = time.perf_counter()
print("timing:")

panel_obj = timed(
    "build",
    lambda: apply_backend(
        pp.models.UKMeasles.panel_pomp(
            units=active_units, theta=initial_params, model="001b", clean=True
        )
    ),
)

key, subkey = jax.random.split(key)
timed("mif", lambda: panel_obj.mif(J=NP_FITR, M=NFITR, rw_sd=RW_SD, key=subkey))

key, subkey = jax.random.split(key)
timed(
    "pfilter_eval1",
    lambda: panel_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey),
)
eval1 = panel_obj.results()
eval1_lls = eval1.groupby("theta_idx")["shared logLik"].first().tolist()


def _mix_and_prune():
    panel_obj.mix_and_match()
    panel_obj.prune(n=1, refill=False)


timed("mix_and_match_prune", _mix_and_prune)

key, subkey = jax.random.split(key)
timed(
    "pfilter_final",
    lambda: panel_obj.pfilter(J=NP_EVAL, reps=NREPS_EVAL, key=subkey, CLL=True),
)
stage_wall_s["total"] = time.perf_counter() - t_script

final = panel_obj.results()
final_ll = float(final["shared logLik"].iloc[0])
gpu_peak_gb = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e9
print(f"eval1 logLiks: {eval1_lls}")
print(f"final logLik: {final_ll:.2f}")
print(f"GPU peak memory: {gpu_peak_gb:.1f} GB")

projected_full_s = None
if N_UNITS == 1422 and NP_FITR == 10000 and NREPS_FITR == 36:
    s = stage_wall_s
    eval_scale = FULL_NREPS_EVAL / NREPS_EVAL
    projected_full_s = (
        s["build"]
        + s["mif"] * FULL_NFITR / NFITR
        + (s["pfilter_eval1"] + s["pfilter_final"]) * eval_scale
    )
    print(f"projected full-run time: {projected_full_s / 3600:.1f} h")

platform = jax.devices()[0].platform
out_dir = os.path.join("results", platform + backend_suffix())
os.makedirs(out_dir, exist_ok=True)
tag = f"rl{RUN_LEVEL}"

timings_df = pd.DataFrame(
    [{"stage": k, "time_seconds": v} for k, v in stage_wall_s.items()]
)
if RUN_LEVEL == 4:
    timings_df["reference_seconds"] = timings_df["stage"].map(REFERENCE["stage_wall_s"])
    timings_df["speedup"] = timings_df["reference_seconds"] / timings_df["time_seconds"]
timings_df.to_csv(os.path.join(out_dir, f"timings_{tag}.csv"), index=False)
final.to_csv(os.path.join(out_dir, f"final_{tag}.csv.gz"), index=False)

record = run_metadata(
    {
        "kind": "timing",
        "model": "panel_measles_001b",
        "RUN_LEVEL": RUN_LEVEL,
        "MAIN_SEED": MAIN_SEED,
        "N_UNITS": N_UNITS,
        "NP_FITR": NP_FITR,
        "NFITR": NFITR,
        "NREPS_FITR": NREPS_FITR,
        "NP_EVAL": NP_EVAL,
        "NREPS_EVAL": NREPS_EVAL,
    }
)
record["stage_wall_s"] = stage_wall_s
record["gpu_peak_gb"] = gpu_peak_gb
record["eval1_logliks"] = eval1_lls
record["final_loglik"] = final_ll
record["projected_full_seconds"] = projected_full_s
record["reference"] = REFERENCE

with open(os.path.join(out_dir, f"latest_{tag}.json"), "w") as f:
    json.dump(record, f, indent=2, default=str)
    f.write("\n")

print(f"\n{timings_df.to_string(index=False)}")
print(f"wrote {out_dir}/ (timings_{tag}.csv, final_{tag}.csv.gz, latest_{tag}.json)")
