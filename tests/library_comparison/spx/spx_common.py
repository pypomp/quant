"""Shared CLI, inputs and result recording for the SPX cross-library benchmark.

Only the standard library is imported at module level, so a bench script can
call parse_args() (which sets the device and thread environment variables)
before it imports numpy, JAX or torch.

Model (pypomp.models.spx, Breto 2014), one process step per observation:
    step into observation n uses y_prev = covars_extended[n] (the previous
    return; 0 for the first step) and a N(0,1) draw dZ:
        dWs = (y_prev - mu + V/2) / sqrt(V)
        dWv = rho dWs + sqrt(1 - rho^2) dZ
        S  <- S + S (mu + sqrt(max(V, V_floor)) dWs)
        V  <- max(V + kappa (theta - V) + xi sqrt(V) dWv, V_floor)
    y_n ~ N(mu - V/2, sqrt(V));   x0 = (V_0, S0) at t0, deterministic.
"""

import argparse
import json
import math
import os
import platform
import socket
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "inputs.npz")
MODEL = "spx"


def parse_args(lib, cpu_only=False, extra=None, argv=None):
    p = argparse.ArgumentParser(description=f"SPX bootstrap particle filter benchmark: {lib}")
    p.add_argument("--J", type=int, default=1000, help="particles per filter")
    p.add_argument("--nthetas", type=int, default=36, help="parameter vectors (first n of the 36 starts)")
    p.add_argument("--nreps", type=int, default=24, help="independent filters per parameter vector")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--T", type=int, default=None, help="use only the first T observations (quick checks)")
    p.add_argument("--procs", type=int, default=1, help="worker processes (CPU-only libraries)")
    p.add_argument("--seed", type=int, default=0, help="base random seed")
    p.add_argument("--theta-set", choices=["starts", "sun2024"], default="starts",
                   help="starts = the 36 timing starts; sun2024 = one near-MLE vector (validation)")
    if extra is not None:
        extra(p)
    args = p.parse_args(argv)
    if cpu_only and args.device != "cpu":
        p.error(f"{lib} is a CPU-only library")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    if cpu_only:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[v] = "1"
    return args


def check_jax_device(args, jax):
    """Fail instead of silently timing the CPU when --device gpu finds no GPU (and vice versa)."""
    plat = jax.devices()[0].platform
    if (args.device == "gpu") != (plat == "gpu"):
        raise SystemExit(f"--device {args.device} but JAX runs on {jax.devices()}")


def load_inputs(T=None, nthetas=None, theta_set="starts"):
    """inputs.npz as a dict, optionally truncated to the first T observations.

    theta_set="sun2024" replaces `thetas` with the single Sun (2024) vector.
    """
    import numpy as np

    with np.load(INPUTS, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if theta_set == "sun2024":
        d["thetas"] = d["theta_sun2024"]
    if T is not None:
        if not 1 <= T <= len(d["ys"]):
            raise ValueError(f"--T must be in 1..{len(d['ys'])}")
        for k in ("ys", "times", "dt_array_extended", "nstep_array", "y_prev"):
            d[k] = d[k][:T]
        d["covars_extended"] = d["covars_extended"][: T + 1]
    if nthetas is not None:
        if not 1 <= nthetas <= len(d["thetas"]):
            raise ValueError(f"--nthetas must be in 1..{len(d['thetas'])}")
        d["thetas"] = d["thetas"][:nthetas]
    return d


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out


def _clean(x):
    x = float(x)
    return x if math.isfinite(x) else None


def results_path(device):
    return os.environ.get("PFB_RESULTS") or os.path.join(HERE, "results", device, "results.jsonl")


def record(args, lib, T, cold_s, warm_s, ll, versions, **extra):
    """Append one JSON line. ll: (nthetas, nreps) log-likelihoods."""
    import numpy as np

    ll = np.asarray(ll, dtype=np.float64).reshape(args.nthetas, args.nreps)
    sd = ll.std(axis=1, ddof=1) if args.nreps > 1 else np.full(args.nthetas, np.nan)
    row = dict(
        lib=lib, model=MODEL, device=args.device, J=args.J, nthetas=args.nthetas, nreps=args.nreps,
        T=int(T), cold_s=round(cold_s, 4), warm_s=round(warm_s, 4),
        loglik_by_theta=[_clean(round(v, 4)) for v in ll.mean(axis=1)],
        loglik_sd_by_theta=[_clean(round(v, 4)) for v in sd],
        versions=versions, seed=args.seed, theta_set=args.theta_set, **extra,
        loglik=[[_clean(round(v, 4)) for v in r] for r in ll],
        host=socket.gethostname(), python=platform.python_version(),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    path = results_path(args.device)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(row)
    with open(path, "a") as f:
        f.write(line + "\n")
    short = {k: v for k, v in row.items() if k != "loglik"}
    print(json.dumps(short), flush=True)
    return row


def versions(*names):
    import importlib.metadata as md

    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out
