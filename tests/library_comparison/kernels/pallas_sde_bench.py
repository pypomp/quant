"""Fused Pallas propagation kernel vs pypomp's XLA substep loop, on the toy SDE.

Each variant runs the same bootstrap particle filter (T = 100 observations,
10 Euler substeps each) for R filters of J particles:

  xla         pypomp's structure: lax.fori_loop over substeps, elementwise step
  xla-unroll  the same loop with unroll=True
  pallas      one Triton kernel per observation interval runs all substeps
              for a block of particles, state kept in registers

The only difference is how the propagation is compiled. All variants draw
their normals from the same integer hash of (seed, observation, substep,
particle), and share dmeasure, weight normalization and pypomp's systematic
resampler.

usage: python pallas_sde_bench.py [--interpret] [--check] [R:J ...]
"""

import json
import os
import statistics
import sys
import time

INTERPRET = "--interpret" in sys.argv
if INTERPRET:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import triton as plgpu  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402
from pypomp.core.algorithms.helpers import _normalize_weights, _resample  # noqa: E402

U32 = jnp.uint32
Y = jnp.asarray(C.data("sde"), jnp.float32)
SQRT_DT = float(C.DT) ** 0.5


# ---- shared RNG and model --------------------------------------------------------------

def mix(x):
    x = (x ^ (x >> 16)) * U32(0x45D9F3B)
    x = (x ^ (x >> 16)) * U32(0x45D9F3B)
    return x ^ (x >> 16)


def normal(seed, i, k, j):
    """Standard normal for particle j at substep k of interval i (Box-Muller on two hashes)."""
    base = mix(seed ^ mix(i * U32(0x9E3779B9) + k * U32(0x85EBCA6B)))
    h1 = mix(base + j * U32(0x27D4EB2F))
    h2 = mix(h1 ^ U32(0x5BD1E995))
    u1 = ((h1 >> 8).astype(jnp.float32) + 0.5) * (1.0 / (1 << 24))
    u2 = ((h2 >> 8).astype(jnp.float32) + 0.5) * (1.0 / (1 << 24))
    return jnp.sqrt(-2.0 * jnp.log(u1)) * jnp.cos(2.0 * jnp.pi * u2)


def step(z, eps):
    """One Euler-Maruyama substep of the log-logistic SDE (the user's rproc)."""
    return z + (C.R * (1.0 - jnp.exp(z)) - C.SIG**2 / 2) * C.DT + C.SIG * SQRT_DT * eps


# ---- propagation over one observation interval: three ways ------------------------------

def make_xla(unroll):
    def propagate(z, seed, i):  # z: (N,) flat over R filters x J particles
        j = jax.lax.iota(U32, z.shape[0])

        def body(k, z):
            return step(z, normal(seed, i, jnp.asarray(k).astype(U32), j))

        return jax.lax.fori_loop(0, C.NSTEP, body, z, unroll=unroll)
    return propagate


def make_pallas(block, num_warps):
    def kernel(si_ref, z_ref, o_ref):
        seed, i = si_ref[0], si_ref[1]
        j = (pl.program_id(0) * block + jax.lax.broadcasted_iota(jnp.int32, (block,), 0)).astype(U32)

        def body(k, z):
            return step(z, normal(seed, i, jnp.asarray(k).astype(U32), j))

        o_ref[...] = jax.lax.fori_loop(0, C.NSTEP, body, z_ref[...])

    def propagate(z, seed, i):
        n = z.shape[0]
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((n,), jnp.float32),
            grid=(n // block,),
            in_specs=[pl.BlockSpec((2,), lambda b: (0,)), pl.BlockSpec((block,), lambda b: (b,))],
            out_specs=pl.BlockSpec((block,), lambda b: (b,)),
            compiler_params=plgpu.CompilerParams(num_warps=num_warps),
            interpret=INTERPRET,
        )(jnp.stack([seed, i]), z)
    return propagate


# ---- the particle filter, identical around the propagation -----------------------------

def make_pf(propagate, R, J):
    N = R * J
    obs_i = jnp.arange(C.T, dtype=U32)

    def pf(seed_int):
        seed = U32(seed_int) if isinstance(seed_int, int) else seed_int.astype(U32)
        j = jax.lax.iota(U32, N)
        z = C.Z0M + C.Z0S * normal(seed, U32(C.T + 7), U32(0), j)
        rkey = jax.random.key(0)

        def obs_step(carry, xs):
            z, lw, ll = carry
            i, y = xs
            z = propagate(z, seed, i)
            w = lw + jax.scipy.stats.norm.logpdf(y, z, C.TAU).reshape(R, J)
            nw, ll_t = jax.vmap(_normalize_weights)(w)
            keys = jax.random.split(jax.random.fold_in(jax.random.fold_in(rkey, seed), i), R)
            idx = jax.vmap(_resample)(nw, keys)                      # (R, J)
            z = jnp.take_along_axis(z.reshape(R, J), idx, axis=1).reshape(N)
            lw = jnp.full((R, J), -jnp.log(J), jnp.float32)
            return (z, lw, ll + ll_t), None

        lw0 = jnp.full((R, J), -jnp.log(J), jnp.float32)
        (_, _, ll), _ = jax.lax.scan(obs_step, (z, lw0, jnp.zeros(R)), (obs_i, Y))
        return ll  # (R,) log-likelihoods

    return jax.jit(pf)


def time_pf(pf, n=5):
    t0 = time.perf_counter()
    ll = jax.block_until_ready(pf(jnp.uint32(0)))
    first = time.perf_counter() - t0
    ts = []
    for s in range(1, n + 1):
        t0 = time.perf_counter()
        jax.block_until_ready(pf(jnp.uint32(s)))
        ts.append(time.perf_counter() - t0)
    return first, statistics.median(ts), ll


def check():
    """One interval from the same particles: all three variants must agree."""
    N = 4096
    z = C.Z0M + C.Z0S * normal(U32(1), U32(99), U32(0), jax.lax.iota(U32, N))
    ref = make_xla(1)(z, U32(3), U32(5))
    for name, prop in (("xla-unroll", make_xla(True)), ("pallas", make_pallas(256, 4))):
        d = float(jnp.max(jnp.abs(prop(z, U32(3), U32(5)) - ref)))
        print(f"check {name:10s} max |dz| after one interval vs xla: {d:.2e}")
    for name, prop in (("xla", make_xla(1)), ("pallas", make_pallas(256, 4))):
        ll = make_pf(prop, 4, 1024)(jnp.uint32(0))
        print(f"check {name:10s} loglik (4 filters, J=1024): {[round(float(x), 3) for x in ll]}")


def main():
    if "--check" in sys.argv:
        check()
        return
    out = os.environ.get("PFB_RESULTS", os.path.join(C.HERE, "results_pallas_sde.jsonl"))
    shapes = [a for a in sys.argv[1:] if ":" in a] or ["1:1024", "1:16384", "1:131072", "1:1048576", "100:1024"]
    for shp in shapes:
        R, J = (int(x) for x in shp.split(":"))
        variants = [("xla", make_xla(1)), ("xla-unroll", make_xla(True))]
        variants += [(f"pallas/b{b}w{w}", make_pallas(b, w))
                     for b, w in ((128, 4), (256, 4), (512, 4), (1024, 8)) if (R * J) % b == 0]
        for name, prop in variants:
            first, med, ll = time_pf(make_pf(prop, R, J))
            row = dict(variant=name, R=R, J=J, first_s=round(first, 3), median_ms=round(med * 1e3, 3),
                       us_per_substep=round(med / (C.T * C.NSTEP) * 1e6, 2),
                       loglik_mean=round(float(jnp.mean(ll)), 3), device=jax.devices()[0].platform)
            print(json.dumps(row), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
