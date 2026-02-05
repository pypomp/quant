"""
This script tests the speed of the fast_approx_rpoisson, fast_approx_rbinom, and fast_approx_rgamma functions in pypomp.random. They are compared against the corresponding functions in jax.random.
"""

import os

USE_CPU = os.environ.get("USE_CPU", "false").lower() == "true"
if USE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"

import pypomp.random as ppr  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from importlib.metadata import version  # noqa: E402

print("Using CPU: ", USE_CPU)

now = datetime.now()
print("DATE: ", now.date())
print("TIME: ", now.time())
print("pypomp version:", version("pypomp"))
print("jax version:", version("jax"))


def poissoninvf_performance():
    # Prepare parameters
    n = 1_000_000
    key = jax.random.key(42)
    lam = jnp.array([0.01, 0.2, 1.0, 8.0, 10.0, 12.0, 50.0, 100.0], dtype=jnp.float32)
    lam_samples = jnp.repeat(lam, n // len(lam))
    reps = 20

    # Warmup to trigger JITs
    key1, key2 = jax.random.split(key)
    _ = ppr.fast_approx_rpoisson(key1, lam_samples).block_until_ready()
    _ = jax.random.poisson(key2, lam_samples).block_until_ready()

    # Run fast_approx_rpoisson reps times
    t_pp_total = 0.0
    for i in range(reps):
        key1, _ = jax.random.split(key1)
        t0 = time.time()
        _ = ppr.fast_approx_rpoisson(key1, lam_samples).block_until_ready()
        t1 = time.time()
        t_pp_total += t1 - t0
    avg_t_pp = t_pp_total / reps

    # Run jax.random.poisson reps times
    t_jax_total = 0.0
    for i in range(reps):
        key2, _ = jax.random.split(key2)
        t0 = time.time()
        _ = jax.random.poisson(key2, lam_samples).block_until_ready()
        t1 = time.time()
        t_jax_total += t1 - t0
    avg_t_jax = t_jax_total / reps

    print(
        f"pp.fast_approx_rpoisson: {avg_t_pp:.4f} seconds (avg over {reps} runs, {n} samples each)"
    )
    print(
        f"jax.random.poisson: {avg_t_jax:.4f} seconds (avg over {reps} runs, {n} samples each)"
    )
    print(f"ratio: {avg_t_jax / avg_t_pp:.4f}")
    pass


def binominvf_performance():
    # Prepare parameters
    n = 1_000_000
    key = jax.random.key(43)
    trials = jnp.array([1, 5, 20, 50, 100, 200], dtype=jnp.float32)
    p = jnp.array([0.01, 0.2, 0.5, 0.7, 0.9, 0.99], dtype=jnp.float32)
    trial_grid, p_grid = jnp.meshgrid(trials, p, indexing="ij")
    trial_flat = trial_grid.reshape(-1)
    p_flat = p_grid.reshape(-1)
    n_repeat = n // len(trial_flat)
    n_samples = n_repeat * len(trial_flat)  # actual total count (may be just under n)
    trial_samples = jnp.tile(trial_flat, n_repeat)
    p_samples = jnp.tile(p_flat, n_repeat)
    reps = 20

    # Warmup to trigger JITs
    key1, key2 = jax.random.split(key)
    _ = ppr.fast_approx_rbinom(key1, trial_samples, p_samples).block_until_ready()
    _ = jax.random.binomial(key2, trial_samples, p_samples).block_until_ready()

    # Run fast_approx_rbinom reps times
    t_pp_total = 0.0
    for i in range(reps):
        key1, _ = jax.random.split(key1)
        t0 = time.time()
        _ = ppr.fast_approx_rbinom(key1, trial_samples, p_samples).block_until_ready()
        t1 = time.time()
        t_pp_total += t1 - t0
    avg_t_pp = t_pp_total / reps

    # Run jax.random.binomial reps times
    t_jax_total = 0.0
    for i in range(reps):
        key2, _ = jax.random.split(key2)
        t0 = time.time()
        _ = jax.random.binomial(key2, trial_samples, p_samples).block_until_ready()
        t1 = time.time()
        t_jax_total += t1 - t0
    avg_t_jax = t_jax_total / reps

    print(
        f"fast_approx_rbinom: {avg_t_pp:.4f} seconds (avg over {reps} runs, {n_samples} samples each)"
    )
    print(
        f"jax.random.binomial: {avg_t_jax:.4f} seconds (avg over {reps} runs, {n_samples} samples each)"
    )
    print(f"ratio: {avg_t_jax / avg_t_pp:.4f}")
    pass


def gammainvf_performance():
    # Prepare parameters
    n = 1_000_000
    key = jax.random.key(44)
    alpha = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0], dtype=jnp.float32)
    alpha_samples = jnp.repeat(alpha, n // len(alpha))
    reps = 20

    # Warmup to trigger JITs
    key1, key2 = jax.random.split(key)
    t0 = time.time()
    _ = ppr.fast_approx_rgamma(key1, alpha_samples).block_until_ready()
    t1 = time.time()
    print(f"Warmup ppr.fast_approx_rgamma: {t1 - t0:.4f} seconds")

    t2 = time.time()
    _ = jax.random.gamma(key2, alpha_samples).block_until_ready()
    t3 = time.time()
    print(f"Warmup jax.random.gamma: {t3 - t2:.4f} seconds")

    # Run fast_approx_rgamma reps times
    t_pp_total = 0.0
    for i in range(reps):
        key1, _ = jax.random.split(key1)
        t0 = time.time()
        _ = ppr.fast_approx_rgamma(key1, alpha_samples).block_until_ready()
        t1 = time.time()
        t_pp_total += t1 - t0
    avg_t_pp = t_pp_total / reps

    # Run jax.random.gamma reps times
    t_jax_total = 0.0
    for i in range(reps):
        key2, _ = jax.random.split(key2)
        t0 = time.time()
        _ = jax.random.gamma(key2, alpha_samples).block_until_ready()
        t1 = time.time()
        t_jax_total += t1 - t0
    avg_t_jax = t_jax_total / reps

    print(
        f"fast_approx_rgamma: {avg_t_pp:.4f} seconds (avg over {reps} runs, {n} samples each)"
    )
    print(
        f"jax.random.gamma: {avg_t_jax:.4f} seconds (avg over {reps} runs, {n} samples each)"
    )
    print(f"ratio: {avg_t_jax / avg_t_pp:.4f}")


print("--------------------------------")
print("Poisson performance comparison")
poissoninvf_performance()
print("--------------------------------")
print("Binomial performance comparison")
binominvf_performance()
print("--------------------------------")
print("Gamma performance comparison")
gammainvf_performance()
