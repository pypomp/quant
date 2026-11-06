"""Replace a pypomp model's `rprocess_interp` with one CuTe kernel per observation interval.

`cute_struct(model)` returns the model's PompStruct with `rproc_pf` swapped, so pypomp's
own `pypomp.functional.pfilter` runs unchanged around it: dmeasure, weights,
resampling and the key schedule stay pypomp's. Each CUDA thread owns one particle of
one filter, keeps its state in registers, and runs all `nstep` Euler steps of the
interval by executing the user's rproc, translated from its jaxpr (cute_translate.py).

Random numbers are the ones pypomp would draw: the substep keys are split exactly as
`_time_interp` splits them, a default (per-particle) rproc gets
`split(substep_key, J)[j]`, and a @vectorized rproc gets element (s, j) of its
(..., J) draw. So for the same key the filter should match stock pypomp up to
float rounding in the transcendental functions.

Several filters (pypomp vmaps over parameter sets and replicates) go into one launch:
a custom_vmap rule folds every batch level into the kernel's particle axis.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

import cutlass
import cutlass.cute as cute
from cutlass.jax import cutlass_call

import cute_translate as ct


def trace_step(model):
    """jaxpr of one Euler step for one particle: (x, theta, key_data, covars, t, dt) -> x."""
    rp = model.rproc
    snames = list(rp.statenames)
    vec = bool(rp._is_vectorized)
    ns, d = len(snames), len(rp.param_names)
    nc = len(rp.covar_names)

    def f(x, th, kd, cov, t, dt):
        X = {n: (x[i][None] if vec else x[i]) for i, n in enumerate(snames)}
        theta = rp._prepare_theta(th, False)
        covd = rp._unpack_covars(cov) if nc else {}
        res = rp._call_original(X_=X, theta_=theta, key=jax.random.wrap_key_data(kd), covars=covd, t=t, dt=dt)
        return jnp.stack([jnp.asarray(res[n], jnp.float32).reshape(()) for n in snames])

    f32 = jnp.float32
    cj = jax.make_jaxpr(f)(jnp.zeros(ns, f32), jnp.ones(d, f32), jnp.zeros(2, jnp.uint32),
                           jnp.ones(max(nc, 1), f32)[:nc], f32(0.0), f32(0.01))
    return cj, vec


def _thread(cj, vec, J, ns, nth, nc, nstep, acc, X, TH, KEYS, TAB, OUT, g):
    """Everything one thread does (plain Python, run at trace time inside the kernel)."""
    b = g // J
    j = (g - b * J).to(cutlass.Uint32)
    ctx = ct.Ctx(vectorized_J=J if vec else None, particle=j)
    xs = [X[s, g] for s in range(ns)]
    for a in acc:
        xs[a] = cutlass.Float32(0.0)
    th = np.empty(nth, dtype=object)
    for k in range(nth):
        th[k] = TH[b, k]
    for k in range(nstep):
        k0, k1 = KEYS[b, k, 0], KEYS[b, k, 1]
        if not vec:  # pypomp's step_pf: split(substep_key, J)[j]
            k0, k1 = ct.threefry2x32(k0, k1, np.uint32(0), j)
        cov = np.empty(nc, dtype=object)
        for c in range(nc):
            cov[c] = TAB[k, 2 + c]
        (out,) = ct.interpret(cj, [ct.obj(xs), th, ct.obj([k0, k1]), cov, ct.obj(TAB[k, 0]),
                                   ct.obj(TAB[k, 1])], ctx)
        xs = [ct.as_dsl(out[s], np.float32) for s in range(ns)]
    for s in range(ns):
        OUT[s, g] = xs[s]


def _load(X, TH, g, b, ns, nth, acc):
    xs = [X[s, g] for s in range(ns)]
    for a in acc:
        xs[a] = cutlass.Float32(0.0)
    th = np.empty(nth, dtype=object)
    for k in range(nth):
        th[k] = TH[b, k]
    return tuple(xs), th


def _step_once(cj, vec, J, ns, nc, xs, th, KEYS, TAB, b, j, k):
    """One Euler step for one particle; k may be a runtime loop index."""
    ctx = ct.Ctx(vectorized_J=J if vec else None, particle=j)
    k0, k1 = KEYS[b, k, 0], KEYS[b, k, 1]
    if not vec:
        k0, k1 = ct.threefry2x32(k0, k1, np.uint32(0), j)
    cov = np.empty(nc, dtype=object)
    for c in range(nc):
        cov[c] = TAB[k, 2 + c]
    (out,) = ct.interpret(cj, [ct.obj(list(xs)), th, ct.obj([k0, k1]), cov, ct.obj(TAB[k, 0]),
                               ct.obj(TAB[k, 1])], ctx)
    return tuple(ct.as_dsl(out[s], np.float32) for s in range(ns))


def _store(OUT, g, xs):
    for s, x in enumerate(xs):
        OUT[s, g] = x


_DYN_SRC = """
import cutlass
import cutlass.cute as cute
from cute_rproc import _load, _step_once, _store


def make(cj, vec, J, ns, nth, nc, NSTEP, acc, block):
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, OUT: cute.Tensor,
               N: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * block + tidx
        if g < N:
            b = g // J
            j = (g - b * J).to(cutlass.Uint32)
            xs0, th = _load(X, TH, g, b, ns, nth, acc)
            {xs} = xs0
            for k in range(NSTEP):
                {xs} = _step_once(cj, vec, J, ns, nc, ({xs}), th, KEYS, TAB, b, j, k)
            _store(OUT, g, ({xs}))

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               OUT: cute.Tensor):
        N = cute.size(X, mode=[1])
        kernel(X, TH, KEYS, TAB, OUT, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                block=[block, 1, 1], stream=stream)

    return launch
"""


def build_launcher_dynamic(cj, vec, J, ns, nth, nc, nstep, acc, block=128):
    """The interval kernel with a runtime loop over the substeps (state in named registers).

    CuTe's AST preprocessor turns `for k in range(...)` into a device loop with
    loop-carried variables, so the step's code is emitted once instead of nstep times.
    The source needs one variable per state, so it is generated per model.
    """
    src = _DYN_SRC.replace("{xs}", "".join(f"x{s}, " for s in range(ns)).rstrip())
    path = os.path.join(tempfile.gettempdir(), f"cute_rproc_dyn_{hashlib.md5(src.encode()).hexdigest()[:10]}.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make(cj, vec, J, ns, nth, nc, nstep, acc, block)


def build_launcher(cj, vec, J, ns, nth, nc, nstep, acc, block=128):
    # A runtime loop over substeps by default: on He10 it compiles in ~12 s instead of
    # ~74 s and runs faster than the unrolled kernel. CUTE_UNROLL=1 unrolls it.
    if not os.environ.get("CUTE_UNROLL"):
        return build_launcher_dynamic(cj, vec, J, ns, nth, nc, nstep, acc, block)
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, OUT: cute.Tensor,
               N: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * block + tidx
        if g < N:
            _thread(cj, vec, J, ns, nth, nc, nstep, acc, X, TH, KEYS, TAB, OUT, g)

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               OUT: cute.Tensor):
        N = cute.size(X, mode=[1])
        kernel(X, TH, KEYS, TAB, OUT, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                block=[block, 1, 1], stream=stream)

    return launch


def cute_struct(model, block=128):
    """model.to_struct() with rproc_pf replaced by the CuTe interval kernel."""
    struct = model.to_struct()
    rp = model.rproc
    if rp.nstep is None:
        raise NotImplementedError("needs a fixed number of Euler steps per interval")
    nstep = int(rp.nstep)
    cj, vec = trace_step(model)
    ns, nth, nc = len(rp.statenames), len(rp.param_names), len(rp.covar_names)
    acc = tuple(int(a) for a in (struct.accumvars or ()))
    launchers = {}

    def launcher(J):
        if J not in launchers:
            launchers[J] = build_launcher(cj, vec, J, ns, nth, nc, nstep, acc, block)
        return launchers[J]

    @jax.custom_batching.custom_vmap
    def core(Xp, th, sk, tab):
        # Xp (..., J, ns); th (..., d); sk (..., nstep, 2); tab (..., nstep, 2 + nc)
        lead, J = Xp.shape[:-2], Xp.shape[-2]
        B = math.prod(lead)
        X = Xp.reshape(B * J, ns).T
        TH = th.reshape(B, nth)
        SK = sk.reshape(B, nstep, 2)
        TAB = tab.reshape(B, nstep, 2 + nc)[0]  # time and covariates are shared by all filters
        out = cutlass_call(launcher(J), output_shape_dtype=jax.ShapeDtypeStruct(X.shape, jnp.float32))(
            X, TH, SK, TAB)
        return out.T.reshape(lead + (J, ns))

    @core.def_vmap
    def _rule(axis_size, in_batched, Xp, th, sk, tab):
        args = [a if bat else jnp.broadcast_to(a, (axis_size,) + a.shape)
                for a, bat in zip((Xp, th, sk, tab), in_batched)]
        return core(*args), True

    def rproc_interp(X_, theta_, keys, covars_extended, dt_array_extended, t, t_idx, nstep_dynamic,
                     accumvars, should_trans):
        if should_trans:
            raise NotImplementedError("should_trans (IF2 parameter scale) is not wired up")
        if jnp.ndim(keys) != 0:
            keys = keys[0]
        subkeys, ts, dts, covs = [], [], [], []
        tk = t
        for k in range(nstep):  # the same key chain and clock as pypomp's _time_interp
            keys, sub = jax.random.split(keys)
            subkeys.append(jax.random.key_data(sub))
            dt = dt_array_extended[t_idx + k]
            ts.append(tk)
            dts.append(dt)
            if nc:
                covs.append(covars_extended[t_idx + k])
            tk = tk + dt
        tab = jnp.concatenate([jnp.stack([jnp.asarray(ts, jnp.float32), jnp.asarray(dts, jnp.float32)], 1)]
                              + ([jnp.stack(covs).astype(jnp.float32)] if nc else []), axis=1)
        out = core(X_.astype(jnp.float32), jnp.asarray(theta_, jnp.float32), jnp.stack(subkeys), tab)
        return out, t_idx + nstep

    return struct._replace(rproc_pf=rproc_interp)
