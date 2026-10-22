"""Shared pieces of the He10 measles cross-library benchmark (London and the 4-town panel).

Every library reads the same `inputs.npz` (written by export_inputs.py from the
pypomp model quant uses), so they all see identical data, Euler grid,
covariates and parameters. This module holds what the adapters share and that
does not need JAX or torch: the CLI, input loading, result recording, and the
NumPy (float64) version of the model used by particles and pypfilt.

Model (pypomp's model 001b, He et al. 2010 without alpha/mu), per Euler step k
of size dt[k], with covariates pop[k], birthrate[k]:
  br   = cohort*birthrate/dt + (1-cohort)*birthrate  on the cohort step, else (1-cohort)*birthrate
  seas = 1 + amplitude*0.2411/0.7589 in term time, else 1 - amplitude
  beta = R0 * seas * (1 - exp(-(gamma+mu) dt)) / dt,   mu = 0.02
  foi  = beta (I + iota) / pop
  dW   ~ Gamma(shape dt/sigmaSE^2, scale sigmaSE^2)          (white noise)
  births ~ Poisson(br dt)
  S, E, I each lose Euler-multinomial exits with rates (foi dW/dt, mu), (sigma, mu), (gamma, mu)
  W += (dW - dt)/sigmaSE;  C += I->R transitions;  R = pop - S - E - I
W and C are reset to 0 at the start of every observation interval (pypomp accumvars).
The term-time and cohort indicators are taken from inputs.npz, where they were
evaluated with pypomp's own float32 time arithmetic (see export_inputs.py).
Measurement: y ~ discretised Normal(rho C, rho C (1 - rho + psi^2 rho C)), He10's dmeasure with tol 1e-18.
"""

import argparse
import datetime
import json
import os
import platform
import resource
import socket
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.join(os.path.dirname(HERE), "panel_measles")
MU = 0.02
TOL = 1.0e-18
STATES = ("S", "E", "I", "R", "W", "C")
# Kernel parameter order (pypomp model_001b.param_names).
PARAMS = ("R0", "sigma", "gamma", "iota", "rho", "sigmaSE", "psi", "cohort",
          "amplitude", "S_0", "E_0", "I_0", "R_0")
P = {name: i for i, name in enumerate(PARAMS)}
# Diagnostic switch: PFB_DMEAS_F32=1 makes the NumPy dmeas round its CDFs to float32 like
# pypomp's (see README, "float32 measurement tail"). Off for the benchmark.
DMEAS_F32 = os.environ.get("PFB_DMEAS_F32", "") not in ("", "0")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ----------------------------------------------------------------------------- CLI

def parse_args(lib, add_args=None, argv=None):
    p = argparse.ArgumentParser(description=f"He10 measles bootstrap PF benchmark: {lib}")
    p.add_argument("--panel", action="store_true", help="4-town panel instead of London")
    p.add_argument("--J", type=int, default=1000, help="particles per filter")
    p.add_argument("--nthetas", type=int, default=3, help="parameter vectors (first n of the 36 fixed starts)")
    p.add_argument("--nreps", type=int, default=10, help="independent filters per parameter vector")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--T", type=int, default=None, help="use only the first T observations (quick checks)")
    p.add_argument("--procs", type=int, default=1, help="worker processes (CPU-only libraries)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calls", type=int, choices=[1, 2], default=2,
                   help="2: time a cold and a warm call of the full workload; 1: cold only")
    p.add_argument("--tag", default="", help="free-form label stored with the result")
    if add_args is not None:
        add_args(p)
    a = p.parse_args(argv)
    if a.device == "cpu":
        # Must happen before JAX or torch is imported: never touch a GPU on a CPU run.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    return a


def single_thread_env():
    """One BLAS/OpenMP thread per worker process (set before numpy is imported)."""
    for v in THREAD_VARS:
        os.environ[v] = "1"


# ----------------------------------------------------------------------------- inputs

def inputs_path(panel):
    return os.path.join(PANEL_DIR if panel else HERE, "inputs.npz")


class Inputs:
    """inputs.npz, optionally truncated to the first T observations.

    Shapes: U units, T observations, K = sum(nstep) Euler steps.
      ys (U, T), times (T,), t0, nstep (T,), step0 (T,) first global step of each interval,
      dt (K,), pop / birthrate (U, K+1) (index k = covariate used by Euler step k; index 0 also
      feeds rinit), term / cohort_step (K,) bool, first_step (K,) bool (accumvars reset),
      theta (n, U, 13) natural scale in PARAMS order.
    """

    def __init__(self, panel=False, T=None, path=None):
        path = path or inputs_path(panel)
        d = np.load(path, allow_pickle=False)
        self.path = path
        self.model = str(d["model"])
        self.units = [str(u) for u in d["unit_names"]]
        names = [str(n) for n in d["param_names"]]
        theta = np.asarray(d["theta"], dtype=np.float64)
        self.theta = theta[..., [names.index(n) for n in PARAMS]]
        nstep = np.asarray(d["nstep"], dtype=np.int64)
        T_full = len(nstep)
        T = T_full if T is None else min(int(T), T_full)
        K = int(nstep[:T].sum())
        self.T, self.K, self.T_full = T, K, T_full
        self.ys = np.asarray(d["ys"], dtype=np.float64)[:, :T]
        self.times = np.asarray(d["times"], dtype=np.float64)[:T]
        self.t0 = float(d["t0"])
        self.nstep = nstep[:T]
        self.step0 = np.concatenate([[0], np.cumsum(self.nstep)[:-1]]).astype(np.int64)
        self.dt = np.asarray(d["dt"], dtype=np.float64)[:K]
        self.pop = np.asarray(d["pop"], dtype=np.float64)[:, :K + 1]
        self.birthrate = np.asarray(d["birthrate"], dtype=np.float64)[:, :K + 1]
        self.term = np.asarray(d["term"], dtype=bool)[:K]
        self.cohort_step = np.asarray(d["cohort_step"], dtype=bool)[:K]
        self.first_step = np.zeros(K, dtype=bool)
        self.first_step[self.step0] = True
        self.U = len(self.units)

    def thetas(self, n):
        if n > self.theta.shape[0]:
            raise ValueError(f"{n} thetas requested, inputs hold {self.theta.shape[0]}")
        return self.theta[:n]

    def padded(self, dtype=np.float32):
        """Per-interval arrays (T, M), M = max nstep, padded (mask False) for ragged intervals."""
        M = int(self.nstep.max())
        idx = self.step0[:, None] + np.arange(M)[None, :]
        mask = np.arange(M)[None, :] < self.nstep[:, None]
        idx_c = np.where(mask, idx, 0)
        out = dict(
            mask=mask,
            dt=np.where(mask, self.dt[idx_c], 1.0).astype(dtype),
            term=np.where(mask, self.term[idx_c], False),
            coh=np.where(mask, self.cohort_step[idx_c], False),
            pop=np.where(mask[None], self.pop[:, idx_c], 1.0).astype(dtype),      # (U, T, M)
            br=np.where(mask[None], self.birthrate[:, idx_c], 0.0).astype(dtype),  # (U, T, M)
        )
        return out


# ----------------------------------------------------------------------------- results

def results_file(panel, device):
    env = os.environ.get("PFB_RESULTS")
    if env:
        return env
    return os.path.join(PANEL_DIR if panel else HERE, "results", device, "results.jsonl")


def summarize(ll):
    """ll: (nthetas, nreps, U) per-unit filter logliks -> panel totals per rep, then mean/sd per theta."""
    ll = np.asarray(ll, dtype=np.float64)
    tot = ll.sum(axis=2)
    n = tot.shape[1]
    mean = tot.mean(axis=1)
    sd = tot.std(axis=1, ddof=1) if n > 1 else np.full(tot.shape[0], np.nan)
    return tot, mean, sd


def cpu_seconds():
    """CPU time of this process and its reaped children (user + system), all threads."""
    me = resource.getrusage(resource.RUSAGE_SELF)
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    return me.ru_utime + me.ru_stime + ch.ru_utime + ch.ru_stime


def timed_calls(run, calls):
    """Run the full workload `calls` times. Returns logliks of the first call and
    dict(cold_s, warm_s, cold_cpu_s, warm_cpu_s, loadavg_start, loadavg_end)."""
    t = dict(loadavg_start=[round(v, 1) for v in os.getloadavg()])
    lls = []
    for name in ("cold", "warm")[:calls]:
        w0, c0 = time.perf_counter(), cpu_seconds()
        lls.append(run())
        t[f"{name}_s"] = time.perf_counter() - w0
        t[f"{name}_cpu_s"] = round(cpu_seconds() - c0, 3)
    t["loadavg_end"] = [round(v, 1) for v in os.getloadavg()]
    if calls == 2 and not np.array_equal(lls[0], lls[1]):
        t["warm_differs"] = True  # same seeds, so this only happens with nondeterministic kernels
    return lls[0], t


def _num(v, nd=4):
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


def record(args, lib, inp, ll, cold_s, warm_s, samplers, dtype, versions, **extra):
    tot, mean, sd = summarize(ll)
    row = dict(
        lib=lib,
        model=inp.model,
        device=args.device,
        J=args.J,
        nthetas=args.nthetas,
        nreps=args.nreps,
        units=inp.U,
        T=inp.T,
        T_full=inp.T_full,
        n_euler_steps=inp.K,
        n_filters=int(np.prod(np.shape(ll))),
        cold_s=None if cold_s is None else round(float(cold_s), 4),
        warm_s=None if warm_s is None else round(float(warm_s), 4),
        loglik_by_theta=[_num(v) for v in mean],
        loglik_sd_by_theta=[_num(v) for v in sd],
        samplers=samplers,
        dtype=dtype,
        versions=versions,
        seed=args.seed,
        procs=args.procs,
        tag=args.tag,
        host=socket.gethostname(),
        python=platform.python_version(),
        when=datetime.datetime.now().isoformat(timespec="seconds"),
        **extra,
    )
    if DMEAS_F32 and dtype == "float64":
        row["dmeas_f32_emulation"] = True
    if inp.U > 1:
        row["loglik_by_theta_unit"] = np.round(np.asarray(ll, dtype=np.float64).mean(axis=1), 3).tolist()
    # Every filter's loglik, (nthetas, nreps, units), for validation; skipped for very large workloads.
    if np.size(ll) <= 20000:
        row["loglik"] = [[[_num(v, 3) for v in rep] for rep in th] for th in np.asarray(ll)]
    path = results_file(args.panel, args.device)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return row


def versions(*pkgs):
    import importlib.metadata as md
    out = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def task_list(nthetas, nreps, U):
    """(theta, rep, unit) for every independent filter, theta-major."""
    return [(i, r, u) for i in range(nthetas) for r in range(nreps) for u in range(U)]


def task_seed(seed, i, r, u):
    return [int(seed), 1000003, int(i), int(r), int(u)]


# ----------------------------------------------------------------------------- NumPy model (float64)

def np_theta(row):
    return {name: float(row[P[name]]) for name in PARAMS}


def np_rinit(th, pop0, J):
    m = pop0 / (th["S_0"] + th["E_0"] + th["I_0"] + th["R_0"])
    x = np.zeros((6, J))
    x[0] = np.round(m * th["S_0"])
    x[1] = np.round(m * th["E_0"])
    x[2] = np.round(m * th["I_0"])
    x[3] = np.round(m * th["R_0"])
    return x  # rows S, E, I, R, W, C


def np_step(rng, S, E, I, W, C, th, k, inp, u):
    """One Euler step (global index k) for unit u, vectorized over particles. Returns S, E, I, R, W, C."""
    dt = inp.dt[k]
    pop = inp.pop[u, k]
    b = inp.birthrate[u, k]
    coh = th["cohort"]
    br = coh * b / dt + (1 - coh) * b if inp.cohort_step[k] else (1 - coh) * b
    seas = 1.0 + th["amplitude"] * 0.2411 / 0.7589 if inp.term[k] else 1.0 - th["amplitude"]
    beta = th["R0"] * seas * (1.0 - np.exp(-(th["gamma"] + MU) * dt)) / dt
    J = S.shape[0]
    foi = beta * (I + th["iota"]) / pop
    s2 = th["sigmaSE"] ** 2
    dw = rng.standard_gamma(dt / s2, J) * s2
    births = rng.poisson(br * dt, J)
    r0 = np.empty((3, J))
    r0[0] = foi * dw / dt
    r0[1] = th["sigma"]
    r0[2] = th["gamma"]
    rs = r0 + MU
    scale = (1.0 - np.exp(-rs * dt)) / rs
    p0 = r0 * scale
    p1 = MU * scale
    n = np.empty((3, J), dtype=np.int64)
    n[0], n[1], n[2] = S, E, I
    x0 = rng.binomial(n, np.clip(p0, 0.0, 1.0))
    prem = 1.0 - p0
    q = np.clip(p1 / np.where(prem > 0.0, prem, 1.0), 0.0, 1.0)
    x1 = rng.binomial(n - x0, q)
    S = S + births - x0[0] - x1[0]
    E = E + x0[0] - x0[1] - x1[1]
    I = I + x0[1] - x0[2] - x1[2]
    R = pop - S - E - I
    W = W + (dw - dt) / th["sigmaSE"]
    C = C + x0[2]
    return S, E, I, R, W, C


def np_interval(rng, x, i, th, inp, u):
    """Advance x (6, J) through observation interval i (W and C reset first)."""
    S, E, I, W = x[0], x[1], x[2], np.zeros(x.shape[1])
    C = np.zeros(x.shape[1])
    k0 = inp.step0[i]
    for k in range(k0, k0 + inp.nstep[i]):
        S, E, I, R, W, C = np_step(rng, S, E, I, W, C, th, k, inp, u)
    return np.stack([S, E, I, R, W, C])


def np_dmeas(y, C, rho, psi):
    """log He10 measurement density, vectorized over particles (float64)."""
    from scipy.special import ndtr

    m = rho * C
    v = m * (1.0 - rho + psi ** 2 * m)
    sd = np.sqrt(v) + TOL
    if DMEAS_F32:  # diagnostic only: round as pypomp's float32 dmeas does
        f = np.float32
        m32, sd32, y32 = m.astype(f), sd.astype(f), f(y)
        upper = ndtr(((y32 + f(0.5)) - m32) / sd32).astype(f)
        lower = ndtr(((y32 - f(0.5)) - m32) / sd32).astype(f)
        lik = ((upper - lower if y > TOL else upper) + f(TOL)).astype(np.float64)
    else:
        upper = ndtr((y + 0.5 - m) / sd)
        lower = ndtr((y - 0.5 - m) / sd)
        lik = (upper - lower if y > TOL else upper) + TOL
    lik = np.where(C < 0, 0.0, lik)
    if np.isnan(y):
        lik = np.ones_like(lik)
    with np.errstate(divide="ignore"):
        return np.log(lik)
