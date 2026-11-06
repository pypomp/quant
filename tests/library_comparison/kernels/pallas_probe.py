"""Can a fused Pallas substep kernel host an unmodified user rproc?

Builds the kernel pypomp would use: a grid over blocks of particles, and inside
it pypomp's own loop over the substeps of one observation interval, calling the
user's single-step rproc. Lowers and compiles it for CUDA with the Pallas Triton
backend (Triton needs a visible GPU for its compute capability, but nothing is
executed), then does the same for every primitive of each rproc on its own, to
list exactly which ops have no Pallas GPU lowering.

usage: python pallas_probe.py [--run]   (--run also executes the kernels)
"""

import json
import os
import sys
import traceback

RUN = "--run" in sys.argv

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import triton as plgpu  # noqa: E402

from pypomp.models.measles import model_001b as he10  # noqa: E402

BLOCK = 128
NSTEP = 7  # He10 London: daily steps between weekly observations
NSTEP_PAD = 8
SDE = dict(r=1.0, sig=0.3)


def pad_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def short(e):
    lines = [l for l in str(e).splitlines() if l.strip()]
    msg = next((l for l in lines if "rimitive" in l or "mplemented" in l or "not supported" in l.lower()
                or "power of 2" in l or "Unsupported" in l), lines[0] if lines else repr(e))
    return f"{type(e).__name__}: {msg.strip()[:240]}"


def lower(fn, *args):
    """Lower and compile for the visible GPU (Triton -> PTX); with --run, also execute."""
    if RUN:
        return jax.block_until_ready(jax.jit(fn)(*args))
    return jax.jit(fn).lower(*args).compile()


def call(kernel, out_shapes, n_blocks, *args):
    return pl.pallas_call(
        kernel,
        out_shape=out_shapes,
        grid=(n_blocks,),
        in_specs=[pl.BlockSpec(a.shape, lambda i, nd=a.ndim: (0,) * nd) for a in args],
        out_specs=[pl.BlockSpec(o.shape, lambda i, nd=len(o.shape): (0,) * nd) for o in out_shapes],
        compiler_params=plgpu.CompilerParams(num_warps=4),
    )(*args)


# ---- the user's step functions, exactly as written --------------------------------

he10_states = ["S", "E", "I", "R", "W", "C"]
he10_params = list(he10.param_names)
he10_covars = ["pop", "birthrate"]


def sde_rproc(X_, theta_, key, covars, t, dt):  # per-particle, as in bench_pypomp.py
    z = X_["x"]
    eps = jax.random.normal(key)
    return {"x": z + (theta_["r"] * (1 - jnp.exp(z)) - theta_["sig"] ** 2 / 2) * dt
            + theta_["sig"] * jnp.sqrt(dt) * eps}


def he10_step(X, th, key, cov, t, dt):  # @vectorized: one key per step, (BLOCK,) arrays
    return he10.rproc(X, th, key, cov, t, dt)


def sde_step(X, th, key, cov, t, dt):  # pypomp splits one key per particle, then vmaps
    keys = jax.random.split(key, X["x"].shape[0])
    return jax.vmap(sde_rproc, in_axes=(0, None, 0, None, None, None))(X, th, keys, cov, t, dt)


CASES = {
    "he10": dict(step=he10_step, states=he10_states, params=he10_params, covars=he10_covars),
    "sde": dict(step=sde_step, states=["x"], params=["r", "sig"], covars=["dummy"]),
}

HE10_THETA = dict(R0=56.8, sigma=28.9, gamma=30.4, iota=2.9, rho=0.488, sigmaSE=0.0878, psi=0.116,
                  cohort=0.557, amplitude=0.554, S_0=2.97e-2, E_0=5.17e-5, I_0=5.14e-5, R_0=0.97)


def interval_kernel(case, rng):
    """Kernel for one observation interval: pypomp's substep loop around the user's step."""
    c = CASES[case]
    ns, npar, ncov = len(c["states"]), len(c["params"]), len(c["covars"])

    def kernel(x_ref, th_ref, cov_ref, dt_ref, t0_ref, seed_ref, o_ref):
        X = {n: x_ref[i, :] for i, n in enumerate(c["states"])}
        th = {n: th_ref[i] for i, n in enumerate(c["params"])}
        base = rng.key((seed_ref[0], seed_ref[1]), pl.program_id(0))

        def body(k, carry):
            X, t = carry
            cov = {n: cov_ref[k, j] for j, n in enumerate(c["covars"])}
            dt = dt_ref[k]
            X = c["step"](X, th, rng.fold(base, k), cov, t, dt)
            return X, t + dt

        X, _ = jax.lax.fori_loop(0, NSTEP, body, (X, t0_ref[0]))
        for i, n in enumerate(c["states"]):
            o_ref[i, :] = X[n]

    x = jnp.ones((pad_pow2(ns), BLOCK), jnp.float32) * 1000.0
    th = jnp.ones((pad_pow2(npar),), jnp.float32) * 0.5
    if case == "he10":
        th = th.at[:npar].set(jnp.array([HE10_THETA[p] for p in c["params"]], jnp.float32))
    cov = jnp.ones((NSTEP_PAD, pad_pow2(ncov)), jnp.float32) * 3e6
    dt = jnp.full((NSTEP_PAD,), 1 / 365.25, jnp.float32)
    t0 = jnp.full((1,), 1950.0, jnp.float32)
    seed = jnp.array([0, 42], jnp.uint32)
    out = jax.ShapeDtypeStruct(x.shape, jnp.float32)
    return lambda *a: call(kernel, [out], 4, *a), (x, th, cov, dt, t0, seed)


class JaxRandom:
    """The user's jax.random calls, on threefry keys, as pypomp passes them today."""
    name = "jax.random (threefry)"

    @staticmethod
    def key(seed, pid):
        return jax.random.fold_in(jax.random.wrap_key_data(jnp.stack(seed)), pid)

    @staticmethod
    def fold(key, k):
        return jax.random.fold_in(key, k)


class HashRandom(JaxRandom):
    """jax.random.uniform/normal replaced by an integer counter hash (a stand-in for a
    Pallas-lowerable PRNG implementation); everything else in the rproc unchanged."""
    name = "integer-hash uniforms"

    @staticmethod
    def key(seed, pid):
        return (seed[0] ^ seed[1]) * jnp.uint32(0x27D4EB2F) ^ pid.astype(jnp.uint32)

    @staticmethod
    def fold(key, k):
        return key * jnp.uint32(0x85EBCA6B) ^ jnp.asarray(k, jnp.uint32)


def hash_bits(seed, shape):
    x = jax.lax.broadcasted_iota(jnp.uint32, shape, len(shape) - 1) if shape else jnp.uint32(0)
    for d in range(len(shape) - 1):
        x = x + jax.lax.broadcasted_iota(jnp.uint32, shape, d) * jnp.uint32(0x632BE5AB)
    x = (x * jnp.uint32(0x9E3779B9)) ^ seed
    x = (x ^ (x >> 16)) * jnp.uint32(0x45D9F3B)
    return x ^ (x >> 16)


def patch_random_with_hash():
    """Swap jax.random.uniform/normal/split/fold_in/key handling for integer-hash versions."""
    def key_seed(key):
        return key if not hasattr(key, "dtype") or key.dtype == jnp.uint32 else jax.random.key_data(key)[-1]

    def uniform(key, shape=(), dtype=jnp.float32, minval=0.0, maxval=1.0):
        bits = hash_bits(jnp.asarray(key_seed(key), jnp.uint32).reshape(()), tuple(shape))
        u = ((bits >> 8).astype(jnp.float32) + 0.5) * (1.0 / (1 << 24))
        return minval + (maxval - minval) * u

    def normal(key, shape=(), dtype=jnp.float32):
        u1, u2 = uniform(key, shape), uniform(jnp.uint32(key_seed(key)) ^ jnp.uint32(0x5bd1e995), shape)
        return jnp.sqrt(-2.0 * jnp.log(u1)) * jnp.cos(2.0 * jnp.pi * u2)  # Box-Muller

    def split(key, num=2):
        return (jnp.asarray(key_seed(key), jnp.uint32) * jnp.uint32(0x85EBCA6B)
                + jax.lax.iota(jnp.uint32, num) * jnp.uint32(0xC2B2AE35))

    def fold_in(key, data):
        return jnp.asarray(key_seed(key), jnp.uint32) * jnp.uint32(0x27D4EB2F) ^ jnp.asarray(data, jnp.uint32)

    def wrap_key_data(d):
        return d[-1] ^ d[0]

    jax.random.uniform, jax.random.normal = uniform, normal
    jax.random.split, jax.random.fold_in, jax.random.wrap_key_data = split, fold_in, wrap_key_data


# ---- per-primitive probe -------------------------------------------------------------

CONTAINERS = {"pjit", "jit", "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr", "closed_call",
              "remat", "checkpoint"}


def sub_jaxprs(eqn):
    for p in eqn.params.values():
        for q in (p if isinstance(p, (tuple, list)) else (p,)):
            j = getattr(q, "jaxpr", None)
            if j is not None:
                yield j if hasattr(j, "eqns") else j.jaxpr


def collect(jaxpr, seen):
    for e in jaxpr.eqns:
        sig = (e.primitive.name, tuple(str(getattr(v.aval, "dtype", "")) for v in e.invars))
        if e.primitive.name not in CONTAINERS and sig not in seen:
            seen[sig] = e
        for j in sub_jaxprs(e):
            collect(j, seen)
    return seen


def probe_primitive(eqn):
    from jax.extend.core import Literal
    ins = [v for v in eqn.invars if not isinstance(v, Literal)]
    if any(jnp.issubdtype(v.aval.dtype, jax.dtypes.prng_key) for v in ins) or \
       any(jnp.issubdtype(v.aval.dtype, jax.dtypes.prng_key) for v in eqn.outvars):
        return "skipped (typed PRNG key; covered by the RNG probes)"
    shape = lambda a: a.shape if a.shape else (1,)  # noqa: E731
    args = [jnp.ones(shape(v.aval), v.aval.dtype) for v in ins]
    outs = [jax.ShapeDtypeStruct(shape(v.aval), v.aval.dtype) for v in eqn.outvars]

    def kernel(*refs):
        in_refs, out_refs = refs[: len(ins)], refs[len(ins):]
        it = iter(in_refs)
        vals = []
        for v in eqn.invars:
            if isinstance(v, Literal):
                vals.append(v.val)
            else:
                r = next(it)
                vals.append(r[0] if not v.aval.shape else r[...])
        res = eqn.primitive.bind(*vals, **eqn.params)
        res = res if eqn.primitive.multiple_results else [res]
        for r, o, v in zip(out_refs, res, eqn.outvars):
            if v.aval.shape:
                r[...] = o
            else:
                r[0] = o

    try:
        lower(lambda *a: call(kernel, outs, 1, *a), *args)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return short(e)


def rng_probes():
    """The RNG calls pypomp's rprocs make, each alone in a kernel."""
    res = {}
    snippets = {
        "wrap_key_data + fold_in": lambda s: jax.random.key_data(
            jax.random.fold_in(jax.random.wrap_key_data(s), 3)).astype(jnp.float32)[:1],
        "uniform (16, BLOCK)": lambda s: jax.random.uniform(jax.random.wrap_key_data(s), (16, BLOCK))[0, :1],
        "uniform (11, BLOCK) [He10's shape]": lambda s: jax.random.uniform(
            jax.random.wrap_key_data(s), (11, BLOCK))[0, :1],
        "normal (BLOCK,)": lambda s: jax.random.normal(jax.random.wrap_key_data(s), (BLOCK,))[:1],
        "split(key, BLOCK) + per-particle normal [default rproc]": lambda s: jax.vmap(
            lambda k: jax.random.normal(k))(jax.random.split(jax.random.wrap_key_data(s), BLOCK))[:1],
    }
    for name, fn in snippets.items():
        def kernel(s_ref, o_ref, fn=fn):
            o_ref[...] = fn(s_ref[...])
        try:
            lower(lambda s: call(kernel, [jax.ShapeDtypeStruct((1,), jnp.float32)], 1, s),
                  jnp.array([0, 42], jnp.uint32))
            res[name] = "ok"
        except Exception as e:  # noqa: BLE001
            res[name] = short(e)
    return res


def main():
    report = {"jax": jax.__version__, "backend": "pallas triton", "mode": "lower+compile+run" if RUN else "lower+compile"}
    print(f"jax {jax.__version__}; {report['mode']}\n")

    # 1. Whole interval kernel, user code unchanged.
    report["whole_kernel"] = {}
    for case in CASES:
        fn, args = interval_kernel(case, JaxRandom)
        try:
            lower(fn, *args)
            r = "ok"
        except Exception as e:  # noqa: BLE001
            r = short(e)
            if os.environ.get("PROBE_TRACE"):
                traceback.print_exc()
        report["whole_kernel"][f"{case} / {JaxRandom.name}"] = r
        print(f"[whole] {case:4s} / {JaxRandom.name}: {r}")

    # 2. RNG on its own.
    report["rng"] = rng_probes()
    for k, v in report["rng"].items():
        print(f"[rng]   {k}: {v}")

    # 3. Every primitive of each rproc on its own.
    report["primitives"] = {}
    for case, c in CASES.items():
        X = {n: jnp.ones((BLOCK,), jnp.float32) for n in c["states"]}
        th = {n: jnp.float32(0.5) for n in c["params"]}
        cov = {n: jnp.float32(3e6) for n in c["covars"]}
        jx = jax.make_jaxpr(c["step"])(X, th, jax.random.key(0), cov, jnp.float32(1950.0), jnp.float32(1 / 365.25))
        prims = collect(jx.jaxpr, {})
        rows = {}
        for (name, dts), eqn in sorted(prims.items()):
            rows[f"{name}{list(dts)}"] = probe_primitive(eqn)
        report["primitives"][case] = rows
        bad = {k: v for k, v in rows.items() if v != "ok"}
        print(f"\n[prims] {case}: {len(rows)} distinct primitives, {len(rows) - len(bad)} lower")
        for k, v in bad.items():
            print(f"        {k}: {v}")

    # 4. Whole kernel again with RNG swapped for an integer hash: what else blocks?
    patch_random_with_hash()
    for case in CASES:
        fn, args = interval_kernel(case, HashRandom)
        try:
            lower(fn, *args)
            r = "ok"
        except Exception as e:  # noqa: BLE001
            r = short(e)
            if os.environ.get("PROBE_TRACE"):
                traceback.print_exc()
        report["whole_kernel"][f"{case} / {HashRandom.name}"] = r
        print(f"\n[whole] {case:4s} / {HashRandom.name}: {r}")

    out = os.environ.get("PROBE_OUT", f"pallas_probe_{report["mode"].replace("+", "_")}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
