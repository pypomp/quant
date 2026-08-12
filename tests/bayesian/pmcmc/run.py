"""SIR: PMCMC posterior over (beta1, rho), swept over the particle count J.

Runs NCHAINS independent chains from dispersed starts for each J in J_GRID,
writing each sweep into results/gpu/J<J>/.

The J sweep is the point. PMCMC is exact-approximate: the stationary
distribution of the chain does not depend on J, only its mixing does. So the
posteriors from every J must coincide while acceptance rate and ESS/second do
not -- a correctness check that needs no external reference. Note that the
classic bug this would catch (recomputing the likelihood denominator each
iteration rather than carrying the accepted estimate forward) is not present in
pypomp: _pmcmc_step threads ll_cur through the scan carry. Treat this as a
regression guard, not a bug hunt.

Cost model: pmcmc is a lax.scan over M with a full particle filter inside each
step, vmapped only over chains. Parallelism is NCHAINS * J and nothing else, so
wall-clock is linear in M and nearly flat in J until J is large. Prefer generous
J; it is close to free and improves mixing.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "SIR: PMCMC posterior over (beta1, rho), swept over particle count J"
# tags: [bayesian, sir, pmcmc, gpu]
# sbatch_args:
#   job-name: "bayesian pmcmc (pypomp)"
#   partition: gpu-rtx6000
#   gpus: "rtx_pro_6000_blackwell:1"
#   cpus-per-gpu: 1
#   mem: 30GB
#   output: "results/gpu/logs/slurm-%j.out"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:20:00" }
#   2:
#     sbatch_args: { time: "00:40:00" }
#   3:
#     sbatch_args: { time: "02:30:00" }
#   4:
#     sbatch_args: { time: "06:00:00" }
# --- END SLURM CONFIG ---
#
# Level 1 is not sampling-bound, it is compile-bound: XLA takes several minutes
# to build jit__pmcmc_internal for this model (measured at 4m33s on CPU) because
# the scan over M nests a full particle-filter scan. The budget above leaves room
# for that. Levels 3 and 4 should be recalibrated from a level-2 run -- the
# sampling estimates behind them are extrapolated, not measured.

import os
import sys
import time

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if tests_dir not in sys.path:
    sys.path.append(tests_dir)
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if model_dir not in sys.path:
    sys.path.append(model_dir)

# JAX reads these at import time, so they must be set before it is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

USE_CPU = os.environ.get("USE_CPU", "false").lower() == "true"
if USE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "SLURM_CPUS_PER_TASK" in os.environ:
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={os.environ['SLURM_CPUS_PER_TASK']}"
        )

import jax  # noqa: E402
import model  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from utils import pfilter_logliks_frame, save_run  # noqa: E402

print(jax.devices())
print("Using CPU:", USE_CPU)

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running pmcmc at level {RUN_LEVEL}")

NCHAINS = (2, 4, 8, 12)[RUN_LEVEL - 1]
M = (20, 2000, 20000, 50000)[RUN_LEVEL - 1]
J_GRID = ((5,), (100,), (100, 500, 2000), (100, 500, 2000))[RUN_LEVEL - 1]

#: The precondition check: pfilter logLik at the true theta, to be compared
#: against the same quantity from R. This gates everything else -- if the two
#: model implementations disagree here, no posterior comparison downstream means
#: anything, and the grid reference cannot detect the problem because it shares
#: this implementation.
NP_PRECOND = (10, 500, 2000, 2000)[RUN_LEVEL - 1]
NREPS_PRECOND = (2, 12, 24, 24)[RUN_LEVEL - 1]

key = jax.random.key(model.MAIN_SEED)
np.random.seed(model.MAIN_SEED)

out_root = os.path.join("results", "gpu")
os.makedirs(out_root, exist_ok=True)

truth_obj = model.sir_pomp(theta=model.params_from_frame(model.theta_frame(1)))
key, pf_key = jax.random.split(key)
pf_start = time.time()
truth_obj.pfilter(J=NP_PRECOND, reps=NREPS_PRECOND, key=pf_key)
precond = pfilter_logliks_frame(truth_obj)
precond["J"] = NP_PRECOND
precond.to_csv(os.path.join(out_root, "pfilter_logliks.csv"), index=False)
print(
    f"precondition pfilter at truth: J={NP_PRECOND} reps={NREPS_PRECOND} "
    f"mean logLik {precond['logLik'].mean():.2f} "
    f"({time.time() - pf_start:.1f}s)"
)

key, start_key = jax.random.split(key)
starts = model.sample_starts(NCHAINS, key=start_key)
print(f"{NCHAINS} chains, M={M}, J_GRID={J_GRID}")

for J in J_GRID:
    obj = model.sir_pomp(theta=starts)
    key, subkey = jax.random.split(key)

    start = time.time()
    obj.pmcmc(J=J, M=M, proposal=model.proposal(), dprior=model.sir_dprior, key=subkey)
    execution_time = time.time() - start

    result = obj.results_history[-1]
    acceptance = np.asarray(result.acceptance_rate, dtype=float)
    print(
        f"J={J}: {execution_time:.1f}s, "
        f"acceptance {acceptance.min():.3f}-{acceptance.max():.3f}"
    )

    out_dir = os.path.join("results", "gpu", f"J{J}")
    save_run(
        obj,
        out_dir=out_dir,
        run_config={
            "kind": "pmcmc",
            "model": "sir",
            "RUN_LEVEL": RUN_LEVEL,
            "USE_CPU": USE_CPU,
            "MAIN_SEED": model.MAIN_SEED,
            "NCHAINS": NCHAINS,
            "M": M,
            "J": J,
            "free_params": list(model.FREE),
            "rw_sd_estimation_scale": model.RW_SD,
            "prior_box": {k: list(v) for k, v in model.PRIOR_BOX.items()},
            "execution_time": execution_time,
            "platform": jax.devices()[0].platform,
        },
        execution_time=execution_time,
    )

    # save_run does not capture acceptance, which is the diagnostic the J sweep
    # exists to show alongside the posteriors.
    pd.DataFrame(
        {
            "chain": np.arange(len(acceptance)),
            "acceptance_rate": acceptance,
            "J": J,
            "M": M,
            "execution_time": execution_time,
        }
    ).to_csv(os.path.join(out_dir, "acceptance.csv"), index=False)

print("done")
