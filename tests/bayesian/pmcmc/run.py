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

The grid is chosen by log-likelihood noise, not by size: that bug inflates the
posterior in proportion to the variance of the loglik estimate, so the sweep
sees it only if the arms differ in that variance. sd(logLik) is 0.132 at J=2000
and scales like 1/sqrt(J), so {10, 2000} spans a ~14x range where the old
{100, 500, 2000} spanned ~4x, all of it effectively exact.

Cost model: wall-clock is latency-bound, not FLOP-bound. Each iteration is 208
observations x NSTEP=20 = 4160 sequential scan steps at ~35us of dispatch
overhead, costing ~0.17s regardless of J or chain count (J=500 cost 6.5% more
than J=100). So M is the only expensive axis; buy ESS with NCHAINS instead.
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
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "00:30:00" }
#   4:
#     sbatch_args: { time: "00:30:00" }
# --- END SLURM CONFIG ---
#
# Level 4 samples for only ~12 minutes across both J arms; the rest of the
# budget is headroom for XLA, which takes several minutes to build
# jit__pmcmc_internal (4m33s on CPU) since the scan over M nests a pfilter scan.

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

#: M's floor is burn-in and split-R-hat validity, ~10-20x the autocorrelation
#: time of ~25; M=2000 sits at 80x. ESS comes from NCHAINS, which is free.
NCHAINS = (2, 8, 32, 128)[RUN_LEVEL - 1]
M = (20, 1000, 2000, 2000)[RUN_LEVEL - 1]
J_GRID = ((5,), (100,), (10, 2000), (10, 2000))[RUN_LEVEL - 1]

#: The precondition check: pfilter logLik at the true theta, to be compared
#: against the same quantity from R. This gates everything else -- if the two
#: model implementations disagree here, no posterior comparison downstream means
#: anything, and the grid reference cannot detect the problem because it shares
#: this implementation.
NP_PRECOND = (10, 500, 2000, 2000)[RUN_LEVEL - 1]
NREPS_PRECOND = (2, 12, 24, 24)[RUN_LEVEL - 1]

#: Written to its own file, not pfilter_logliks.csv: report.qmd pools every row
#: of that one against R, so mixing noise levels in would corrupt the check.
J_NOISE_GRID = ((5,), (100,), (10, 25, 100, 2000), (10, 25, 100, 2000))[RUN_LEVEL - 1]

#: At this chain count the traces dominate the record. Only beta1 and rho are
#: estimated, and thinning by 10 against IACT ~25 discards nothing.
TRACE_COLS = list(model.FREE) + ["logLik", "log_prior"]
TRACE_THIN = 10

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

# Loglik noise vs J, which sets whether the J sweep can see anything at all.
# Also the degeneracy check on the low arm: a -inf or wildly inflated sd there
# means the chain is stuck rather than informative, and the arm should be raised.
noise_rows = []
for J in J_NOISE_GRID:
    key, nk = jax.random.split(key)
    truth_obj.pfilter(J=J, reps=NREPS_PRECOND, key=nk)
    frame = pfilter_logliks_frame(truth_obj)
    frame["J"] = J
    noise_rows.append(frame)
    finite = np.isfinite(frame["logLik"])
    print(
        f"  J={J:>5}: mean logLik {frame['logLik'][finite].mean():9.2f} "
        f"sd {frame['logLik'][finite].std():6.3f} "
        f"({int((~finite).sum())} non-finite of {len(frame)})"
    )
pd.concat(noise_rows, ignore_index=True).to_csv(
    os.path.join(out_root, "pfilter_vs_J.csv"), index=False
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
            "trace_thin": TRACE_THIN,
        },
        execution_time=execution_time,
        trace_cols=TRACE_COLS,
        thin=TRACE_THIN,
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
