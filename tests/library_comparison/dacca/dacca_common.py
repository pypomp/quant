"""Shared CLI, inputs and result recording for the dacca adapters (numpy only, no JAX/torch).

Model (pypomp's dacca, King et al. 2008), per Euler step of size dt with covariates
(trend, dpopdt, pop, seas1..6) at the start of the step and eps ~ N(0, 1):

  beta  = exp(beta_trend * trend + seas . bs)       omega = exp(seas . omegas)
  dw    = eps * sqrt(dt)                            effI  = (I / pop) ** alpha
  inf   = (omega + (beta + sigma * dw / dt) * effI) * S
  S  += (dpopdt + delta*pop - inf - delta*S + 3*epsilon*R3 + rho*Y) dt
  I  += (c*inf - m*I - delta*I - gamma*I) dt
  Y  += ((1-c)*inf - delta*Y - rho*Y) dt
  R1 += (gamma*I - 3*epsilon*R1 - delta*R1) dt   (R2, R3 likewise, fed by R1, R2)
  Mn += m*I dt
  count += any of (S, I, Y, Mn, R1, R2, R3) < 0;  then every compartment is clipped at 0.
All rates use the values at the start of the step. Mn is reset to 0 at the start of each
observation interval (pypomp accumvar); count is never reset.

rinit (at t0, pop = covars[0]): (S, I, Y, R1, R2, R3) = pop * (S_0, I_0, Y_0, R1_0, R2_0, R3_0) / sum,
Mn = count = 0.

dmeas: v = tau * Mn; log(1e-18) if v is not finite or count > 0, else
logaddexp(log N(y; Mn, v + 1e-18), log(1e-18)).
"""

import argparse
import importlib.metadata as md
import json
import os
import platform
import socket
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.environ.get("PFB_DACCA_INPUTS", os.path.join(HERE, "inputs.npz"))
TOL = 1.0e-18
LTOL = float(np.log(TOL))
STATES = ["S", "I", "Y", "Mn", "R1", "R2", "R3", "count"]
MODEL = "dacca"
LOADAVG_START = os.getloadavg()  # the box is shared: keep the load next to every timing


def parse_args(lib, cpu_only=False, extra=None):
    ap = argparse.ArgumentParser(description=f"dacca bootstrap filters with {lib}")
    ap.add_argument("--J", type=int, default=5000, help="particles per filter")
    ap.add_argument("--nthetas", type=int, default=36, help="parameter vectors (first n of fixed_starts)")
    ap.add_argument("--nreps", type=int, default=36, help="independent filters per parameter vector")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--T", type=int, default=None, help="use only the first T observations")
    ap.add_argument("--procs", type=int, default=1, help="CPU worker processes (CPU-only libs) / torch threads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theta-set", choices=["starts", "published"], default="starts",
                    help="starts: fixed_starts(36) (the paper's workload); published: pypomp's built-in "
                         "dacca parameters (one vector, near the MLE; for validation)")
    if extra is not None:
        extra(ap)
    args = ap.parse_args()
    if cpu_only and args.device != "cpu":
        ap.error(f"{lib} has no GPU backend")
    if args.theta_set == "published" and args.nthetas != 1:
        ap.error("--theta-set published has a single parameter vector: use --nthetas 1")
    return args


def set_cpu_env(threads=1):
    """Hide GPUs and pin BLAS/OpenMP thread pools (call before importing numpy users)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(threads)


def load_inputs(T=None, theta_set="starts"):
    """Inputs truncated to the first T observations, plus per-interval blocks.

    theta_set="published" replaces thetas with pypomp's built-in parameter vector.

    Adds: nstep_const, cov0 (covariates at t0), cov_blocks (T, nstep, C) and dt_blocks (T, nstep):
    row k of block i is Euler step k of observation interval i.
    """
    with np.load(INPUTS, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if theta_set == "published":
        d["thetas"] = d["theta_published"]
    T = len(d["ys"]) if T is None else int(T)
    n = int(d["nstep"][:T].sum())
    d["ys"], d["times"], d["nstep"] = d["ys"][:T], d["times"][:T], d["nstep"][:T]
    d["dt"], d["covars"] = d["dt"][:n], d["covars"][: n + 1]
    ns = np.unique(d["nstep"])
    if ns.size != 1:
        raise ValueError("adapters assume the same number of Euler steps in every interval")
    K = int(ns[0])
    d["T"], d["nstep_const"] = T, K
    d["cov0"] = d["covars"][0]
    d["cov_blocks"] = d["covars"][:n].reshape(T, K, -1)
    d["dt_blocks"] = d["dt"].reshape(T, K)
    d["statenames"] = [str(s) for s in d["statenames"]]
    d["paramnames"] = [str(s) for s in d["paramnames"]]
    d["covarnames"] = [str(s) for s in d["covarnames"]]
    assert d["statenames"] == STATES, d["statenames"]
    assert list(d["covarnames"][:3]) == ["trend", "dpopdt", "pop"]
    return d


def theta_dict(inp, i):
    return {k: float(v) for k, v in zip(inp["paramnames"], inp["thetas"][i])}


def versions(*pkgs):
    out = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def results_path(device):
    p = os.environ.get("PFB_RESULTS")
    if not p:
        p = os.path.join(HERE, "results", device, "results.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    return p


def summarise(ll):
    """ll: (nthetas, nreps) -> per-theta mean and sd (ddof=1 when nreps > 1)."""
    ll = np.asarray(ll, dtype=np.float64)
    sd = ll.std(axis=1, ddof=1) if ll.shape[1] > 1 else np.zeros(ll.shape[0])
    return [round(float(x), 4) for x in ll.mean(axis=1)], [round(float(x), 4) for x in sd]


def record(lib, args, inp, cold_s, warm_s, ll, vers, **extra):
    mean, sd = summarise(ll)
    row = dict(
        lib=lib, model=MODEL, device=args.device, J=args.J, nthetas=args.nthetas, nreps=args.nreps,
        T=inp["T"], nstep=inp["nstep_const"], cold_s=round(cold_s, 4),
        warm_s=None if warm_s is None else round(warm_s, 4),
        loglik_by_theta=mean, loglik_sd_by_theta=sd, versions=vers, seed=args.seed,
        procs=args.procs, theta_set=args.theta_set,
        loadavg_start=[round(x, 1) for x in LOADAVG_START],
        loadavg_end=[round(x, 1) for x in os.getloadavg()],
        cpus=_cpus(),
        host=socket.gethostname(), python=platform.python_version(),
        when=time.strftime("%Y-%m-%dT%H:%M:%S"), **extra,
    )
    path = results_path(args.device)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return row


def _cpus():
    """The CPUs this process may run on, e.g. "8-31" (taskset), or None."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    c = sorted(os.sched_getaffinity(0))
    return f"{c[0]}-{c[-1]}" if c == list(range(c[0], c[-1] + 1)) else ",".join(map(str, c))


def task_seed(seed, call, i, r):
    """A 31-bit seed per (call, theta, rep) (pypfilt's prng_seed must fit in int32)."""
    return int(np.random.SeedSequence([seed, call, i, r]).generate_state(1)[0] >> 1)


def run_pool(worker_init, worker_fn, args, inp_T):
    """Run nthetas x nreps single filters on a pool of args.procs processes, twice.

    worker_fn((i, r, seed)) -> (loglik, seconds). Returns cold_s, warm_s, ll (from the
    first call), mean per-filter seconds (first call), pool start-up seconds.
    """
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    pool = ctx.Pool(args.procs, initializer=worker_init, initargs=(args.J, inp_T, args.theta_set))
    pool.map(_noop, range(args.procs))  # make sure every worker has run its initializer
    start_s = time.perf_counter() - t0
    out = []
    try:
        for call in (0, 1):
            tasks = [(i, r, task_seed(args.seed, call, i, r))
                     for i in range(args.nthetas) for r in range(args.nreps)]
            t0 = time.perf_counter()
            res = pool.map(worker_fn, tasks, chunksize=1)
            out.append((time.perf_counter() - t0, res))
    finally:
        pool.close()
        pool.join()
    (cold_s, res0), (warm_s, _) = out
    ll = np.array([x[0] for x in res0]).reshape(args.nthetas, args.nreps)
    per_filter = float(np.mean([x[1] for x in res0]))
    return cold_s, warm_s, ll, per_filter, start_s


def _noop(_):
    time.sleep(0.05)
    return None
