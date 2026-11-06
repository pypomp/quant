"""Seven-state update plus a negativity check, as any(stack) vs an OR chain, at 1296 x 5000."""
import glob, os, re, sys, time
import jax, jax.numpy as jnp
xs = [jax.random.normal(jax.random.key(i), (1296, 5000)) for i in range(8)]

def update(xs):
    S, I, Y, M, R1, R2, R3, cnt = xs
    new = (S + (R3 - S) * 0.01, I + (S - I) * 0.01, Y - Y * 0.01, M + I * 0.01,
           R1 + (I - R1) * 0.01, R2 + (R1 - R2) * 0.01, R3 + (R2 - R3) * 0.01)
    return new, cnt

@jax.jit
def stacked(*xs):
    new, cnt = update(xs)
    cnt = cnt + jnp.any(jnp.stack(new) < 0, axis=0).astype(cnt.dtype)
    return tuple(jnp.maximum(v, 0.0) for v in new) + (cnt,)

@jax.jit
def ored(*xs):
    new, cnt = update(xs)
    neg = new[0] < 0
    for v in new[1:]:
        neg = neg | (v < 0)
    cnt = cnt + neg.astype(cnt.dtype)
    return tuple(jnp.maximum(v, 0.0) for v in new) + (cnt,)

for f in (stacked, ored):
    jax.block_until_ready(f(*xs))
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); jax.block_until_ready(f(*xs)); ts.append(time.perf_counter() - t0)
    gb = 16 * 1296 * 5000 * 4 / 1e9
    print(f"{f.__name__}: {1e3 * min(ts):.3f} ms, {gb / min(ts):.0f} GB/s effective")
