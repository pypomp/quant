"""Shared models, data, exact likelihoods and timing for the cross-library PF benchmark.

Two models, both observed at t = 1..T with the latent state starting at t0 = 0:

lg   x0 ~ N(0, 1);  x_t = A x_{t-1} + SX * eps;  y_t ~ N(x_t, SY^2)        (1 step / obs)
sde  z0 ~ N(Z0M, Z0S^2); Euler-Maruyama, NSTEP substeps of dt = 1/NSTEP per obs:
       z += (R * (1 - exp(z)) - SIG^2 / 2) dt + SIG sqrt(dt) eps          (log-scale logistic)
     y_t ~ N(z_t, TAU^2)

Libraries that observe the initial state (particles, pfjax, pydpf) get an
initial sampler that draws x0 and then applies one observation interval of
dynamics, so every library targets the same likelihood p(y_1..y_T).
"""

import json
import os
import statistics
import time

import numpy as np

T = 100
# lg
A, SX, SY = 0.9, 1.0, 1.0
# sde
R, SIG, TAU, NSTEP, Z0M, Z0S = 1.0, 0.3, 0.2, 10, float(np.log(0.2)), 0.2
DT = 1.0 / NSTEP

REPS = int(os.environ.get("PFB_REPS", "1"))  # filters run in parallel
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("PFB_RESULTS", os.path.join(HERE, "results.jsonl"))


def data(model: str) -> np.ndarray:
    """Fixed simulated data, shape (T,)."""
    rng = np.random.default_rng(20261022)
    y = np.empty(T)
    if model == "lg":
        x = rng.normal(0.0, 1.0)
        for t in range(T):
            x = A * x + SX * rng.normal()
            y[t] = x + SY * rng.normal()
    elif model == "sde":
        z = rng.normal(Z0M, Z0S)
        for t in range(T):
            for _ in range(NSTEP):
                z += (R * (1 - np.exp(z)) - SIG**2 / 2) * DT + SIG * np.sqrt(DT) * rng.normal()
            y[t] = z + TAU * rng.normal()
    else:
        raise ValueError(model)
    return y


def kalman_loglik(y: np.ndarray) -> float:
    m, P, ll = 0.0, 1.0, 0.0
    for yt in y:
        m, P = A * m, A * A * P + SX**2
        S = P + SY**2
        ll += -0.5 * (np.log(2 * np.pi * S) + (yt - m) ** 2 / S)
        K = P / S
        m, P = m + K * (yt - m), (1 - K) * P
    return float(ll)


def exact(model: str) -> float | None:
    return kalman_loglik(data(model)) if model == "lg" else None


def timeit(run, n_rep: int = 5):
    """run(seed) -> loglik.  Returns (first-call seconds, median seconds, logliks)."""
    t0 = time.perf_counter()
    ll0 = run(0)
    first = time.perf_counter() - t0
    times, lls = [], [ll0]
    for s in range(1, n_rep + 1):
        t0 = time.perf_counter()
        lls.append(run(s))
        times.append(time.perf_counter() - t0)
    return first, statistics.median(times), lls


def record(lib, variant, model, device, J, first, med, lls, **extra):
    ex = exact(model)
    if os.environ.get("PFB_TAG"):
        variant = f"{variant}+{os.environ['PFB_TAG']}"
    row = dict(
        lib=lib, variant=variant, model=model, device=device, J=J, reps=REPS,
        first_s=round(first, 4), median_s=round(med, 5),
        loglik_mean=round(float(np.mean(lls)), 3), loglik_sd=round(float(np.std(lls)), 3),
        exact=None if ex is None else round(ex, 3), **extra,
    )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
