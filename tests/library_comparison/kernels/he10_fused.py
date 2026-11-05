"""He10 (London, model 001b) particle filter: one fused Pallas kernel per observation
interval vs pypomp's XLA substep loop, with the same step code and the same RNG.

What had to change for Pallas Triton to compile the He10 step (everything else is
pypomp's own code, called as is):
  - RNG: 11 hashed uniform streams per particle and substep, in place of one (11, J)
    jax.random.uniform draw that is then sliced (Triton needs power-of-2 sizes and
    cannot slice in-kernel arrays).
  - The gamma sampler's 3-row power product is written out as three terms.
  - jax.scipy.special.ndtri -> sqrt(2) * erf_inv(2p - 1), jnp.polyval -> Horner, and
    lax.cond / lax.switch inside pypomp.random -> elementwise where (what vmap of a
    cond computes anyway). ndtri was the source of `polynomial`, `optimization_barrier`
    and the integer-predicate `select_n`.

Variants, all running the same filter (dmeasure, normalization and pypomp's systematic
resampler are shared XLA code):
  stock   pypomp as shipped (threefry jax.random, unpatched samplers), flat vmap layout
  xla     the patched step in pypomp's structure: lax.fori_loop over the 7 daily steps
  pallas  the patched step, all 7 steps of an interval in one kernel per particle block

usage: python he10_fused.py [--check] [R:J ...]   (PALLAS_INTERPRET=1 runs the kernel on CPU)
"""

import json
import os
import statistics
import sys
import time
import types

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

import pypomp as pp
import pypomp.functional as F
from pypomp.core.algorithms.helpers import _normalize_weights, _resample
from pypomp.models.measles import _samplers as smp
from pypomp.models.measles import model_001b as he10
from pypomp.random import binom as rbinom
from pypomp.random import gamma as rgamma
from pypomp.random import poisson as rpoisson

THETA = {
    "R0": 56.8, "sigma": 28.9, "gamma": 30.4, "iota": 2.9, "rho": 0.488,
    "sigmaSE": 0.0878, "psi": 0.116, "cohort": 0.557, "amplitude": 0.554,
    "S_0": 2.97e-02, "E_0": 5.17e-05, "I_0": 5.14e-05, "R_0": 9.70e-01,
}
U32 = jnp.uint32
NSTREAM = smp.N_GAMMA_UNIFORMS + 7  # 11 uniforms per particle per daily step

model = pp.models.UKMeasles.pomp(unit="London", theta=pp.PompParameters(THETA), clean=True, model="001b")
struct = model.to_struct()
STATES = list(struct.statenames)
COVARS = list(model.covar_names)
PNAMES = list(model.canonical_param_names)
NSTEP_ARR = np.asarray(struct.nstep_array)
assert (NSTEP_ARR == NSTEP_ARR[0]).all(), "prototype assumes a fixed number of steps per interval"
NSTEP = int(NSTEP_ARR[0])
T = int(struct.ys.shape[0])
DT = np.asarray(struct.dt_array_extended, np.float32)[: T * NSTEP]
TSTEP = (float(struct.t0) + np.concatenate([[0.0], np.cumsum(DT.astype(np.float64))[:-1]])).astype(np.float32)
COV = np.asarray(struct.covars_extended, np.float32)[: T * NSTEP]
POP, BR = COV[:, COVARS.index("pop")], COV[:, COVARS.index("birthrate")]
# Per-interval tables (T, 4, 8): rows t, dt, pop, birthrate; columns the 7 steps, padded to 8.
TAB = np.zeros((T, 4, 8), np.float32)
for r, a in enumerate((TSTEP, DT, POP, BR)):
    TAB[:, r, :NSTEP] = a.reshape(T, NSTEP)
TAB = jnp.asarray(TAB)
YS = jnp.asarray(struct.ys)
TIMES = jnp.asarray(struct.times, jnp.float32)
INTERPRET = bool(os.environ.get("PALLAS_INTERPRET"))
TH = jnp.asarray(np.array([THETA[n] for n in PNAMES], np.float32))
ACC = [STATES.index(n) for n in he10.accumvars]


# ---- Pallas-friendly versions of the few constructs that block lowering ----------------

def _ndtri(p):
    return jnp.sqrt(2.0).astype(p.dtype) * jax.lax.erf_inv(2.0 * p - 1.0)


def _polyval(p, x, **_):
    c = [float(v) for v in np.asarray(p)]
    out = jnp.full_like(x, c[0])
    for v in c[1:]:
        out = out * x + v
    return out


def _cond(pred, true_fun, false_fun, *operands, operand=None):
    ops = operands if operands else (operand,)
    return jax.tree.map(lambda a, b: jnp.where(pred, a, b), true_fun(*ops), false_fun(*ops))


def _switch(index, branches, *operands):
    if isinstance(index, (int, np.integer)):
        return branches[int(index)](*operands)
    return jax.lax.switch(index, branches, *operands)


FOLDABLE = {"neg", "abs", "sign", "floor", "ceil", "round", "exp", "exp2", "log", "log1p", "expm1",
            "sqrt", "rsqrt", "cbrt", "integer_pow", "pow", "add", "sub", "mul", "div", "rem", "max",
            "min", "square", "erf_inv", "erf", "tanh", "logistic", "sin", "cos", "convert_element_type",
            "eq", "ne", "lt", "le", "gt", "ge", "and", "or", "not", "xor", "select_n", "stop_gradient"}


def patch_triton_const_folding():
    """Evaluate elementwise ops whose inputs are all scalar constants while lowering.

    Pallas Triton passes constant operands to lowering rules as numpy values. Binary
    rules accept them, but some unary rules do not (neg of a constant fails in
    gammainv's -half * eta2), so those ops are computed eagerly instead."""
    from jax._src.pallas.triton import lowering as tl

    def fold(prim, rule):
        def wrapped(ctx, *args, **params):
            if args and all(isinstance(a, (np.ndarray, np.generic, int, float, bool)) and np.size(a) == 1
                            for a in args):
                out = prim.bind(*[jnp.asarray(a) for a in args], **params)
                return np.asarray(out)
            return rule(ctx, *args, **params)
        return wrapped

    for prim, rule in list(tl.triton_lowering_rules.items()):
        if prim.name in FOLDABLE:
            tl.triton_lowering_rules[prim] = fold(prim, rule)


def patch_samplers():
    jsp_special.ndtri = _ndtri
    jnp.polyval = _polyval
    for mod in (rgamma, rpoisson, rbinom):
        shim = types.SimpleNamespace(**{k: getattr(jax.lax, k) for k in dir(jax.lax) if not k.startswith("_")})
        shim.cond, shim.switch = _cond, _switch
        if hasattr(mod, "lax"):
            mod.lax = shim
        if hasattr(mod, "jsp_special"):
            mod.jsp_special = types.SimpleNamespace(**{**vars(jsp_special), "ndtri": _ndtri})
        if hasattr(mod, "ndtri"):  # gamma.py: from jax.scipy.special import ndtri
            mod.ndtri = _ndtri
        if hasattr(mod, "norm"):  # binom.py: norm.ppf
            mod.norm = types.SimpleNamespace(
                **{**vars(mod.norm), "ppf": lambda q, loc=0.0, scale=1.0: loc + scale * _ndtri(q)})


# ---- shared RNG and step --------------------------------------------------------------

def mix(x):
    x = (x ^ (x >> 16)) * U32(0x45D9F3B)
    x = (x ^ (x >> 16)) * U32(0x45D9F3B)
    return x ^ (x >> 16)


def uniforms(seed, i, k, j):
    base = mix(seed ^ mix(i * U32(0x9E3779B9) + k * U32(0x85EBCA6B)))
    hj = mix(base + j * U32(0x27D4EB2F))
    out = []
    for s in range(NSTREAM):
        h = mix(hj ^ U32((0x5BD1E995 * (s + 1)) & 0xFFFFFFFF))
        out.append(((h >> 8).astype(jnp.float32) + 0.5) * (1.0 / (1 << 24)))
    return out


def uniforms_threefry(seed, i, k, n):
    """The same 11 streams from jax.random (threefry), as stock pypomp draws them."""
    key = jax.random.fold_in(jax.random.fold_in(jax.random.key(seed), i), k)
    u = jax.random.uniform(key, (NSTREAM, n))
    return [u[s] for s in range(NSTREAM)]


def gamma_draw(us, alpha):
    """smp.gamma with its (3, J) power product written out."""
    us = [jnp.clip(u, 1e-7, 1.0 - 1e-7) for u in us]
    alpha = jnp.broadcast_to(alpha, us[0].shape)
    x = rgamma.gammainv(us[0], alpha + smp.GAMMA_ADJ)
    for m, u in enumerate(us[1:]):
        x = x * jnp.power(u, 1.0 / (alpha + (smp.GAMMA_ADJ - 1 - m)))
    return x


def step(X, th, t, dt, pop, birthrate, us):
    """model_001b.rproc, with its uniforms passed in as 11 separate arrays."""
    S, E, I, W, C = X["S"], X["E"], X["I"], X["W"], X["C"]
    R0, sigma, gamma, iota = th["R0"], th["sigma"], th["gamma"], th["iota"]
    sigmaSE, cohort, amplitude = th["sigmaSE"], th["cohort"], th["amplitude"]
    mu = 0.02
    t_mod = t - jnp.floor(t)
    is_cohort_time = jnp.abs(t_mod - 251.0 / 365.0) < 0.5 * dt
    br = jnp.where(is_cohort_time, cohort * birthrate / dt + (1 - cohort) * birthrate, (1 - cohort) * birthrate)
    t_days = t_mod * 365.25
    in_term = (((t_days >= 7) & (t_days <= 100)) | ((t_days >= 115) & (t_days <= 199))
               | ((t_days >= 252) & (t_days <= 300)) | ((t_days >= 308) & (t_days <= 356)))
    seas = jnp.where(in_term, 1.0 + amplitude * 0.2411 / 0.7589, 1 - amplitude)
    beta = R0 * seas * (1.0 - jnp.exp(-(gamma + mu) * dt)) / dt
    foi = beta * (I + iota) / pop
    ug, u = us[: smp.N_GAMMA_UNIFORMS], us[smp.N_GAMMA_UNIFORMS:]
    dw = gamma_draw(ug, dt / sigmaSE**2) * sigmaSE**2
    births = smp.poisson(u[0], br * dt)
    tS0, tS1 = smp.euler_exits(u[1], u[2], S, foi * dw / dt, mu, dt)
    tE0, tE1 = smp.euler_exits(u[3], u[4], E, sigma, mu, dt)
    tI0, tI1 = smp.euler_exits(u[5], u[6], I, gamma, mu, dt)
    S = S + births - tS0 - tS1
    E = E + tS0 - tE0 - tE1
    I = I + tE0 - tI0 - tI1
    return {"S": S, "E": E, "I": I, "R": pop - S - E - I, "W": W + (dw - dt) / sigmaSE, "C": C + tI0}


def theta_dict(th):
    return {n: th[k] for k, n in enumerate(PNAMES)}


# ---- propagation over one interval: XLA loop or one Pallas kernel ----------------------

def make_propagate_xla(rng="hash"):
    def propagate(X, th, tab, seed, i):  # X: (6, N)
        n = X.shape[1]
        j = jax.lax.iota(U32, n)
        thd = theta_dict(th)

        def body(k, Xd):
            kk = jnp.asarray(k).astype(U32)
            us = uniforms(seed, i, kk, j) if rng == "hash" else uniforms_threefry(seed, i, kk, n)
            return step(Xd, thd, tab[0, k], tab[1, k], tab[2, k], tab[3, k], us)

        Xd = jax.lax.fori_loop(0, NSTEP, body, {s: X[r] for r, s in enumerate(STATES)})
        return jnp.stack([Xd[s] for s in STATES])
    return propagate


propagate_xla = make_propagate_xla()


def make_propagate_pallas(block, num_warps):
    def kernel(si_ref, th_ref, tab_ref, x_ref, o_ref):
        seed, i = si_ref[0], si_ref[1]
        j = (pl.program_id(0) * block + jax.lax.broadcasted_iota(jnp.int32, (block,), 0)).astype(U32)
        thd = {n: th_ref[k] for k, n in enumerate(PNAMES)}

        def body(k, Xd):
            us = uniforms(seed, i, jnp.asarray(k).astype(U32), j)
            cov = (tab_ref[0, k], tab_ref[1, k], tab_ref[2, k], tab_ref[3, k])
            return step(Xd, thd, *cov, us)

        Xd = jax.lax.fori_loop(0, NSTEP, body, {s: x_ref[r, :] for r, s in enumerate(STATES)})
        for r, s in enumerate(STATES):
            o_ref[r, :] = Xd[s]

    def propagate(X, th, tab, seed, i):
        rows, n = 8, X.shape[1]
        Xp = jnp.zeros((rows, n), jnp.float32).at[: len(STATES)].set(X)
        thp = jnp.zeros((16,), jnp.float32).at[: len(PNAMES)].set(th)
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((rows, n), jnp.float32),
            grid=(n // block,),
            in_specs=[pl.BlockSpec((2,), lambda b: (0,)), pl.BlockSpec((16,), lambda b: (0,)),
                      pl.BlockSpec((4, 8), lambda b: (0, 0)), pl.BlockSpec((rows, block), lambda b: (0, b))],
            out_specs=pl.BlockSpec((rows, block), lambda b: (0, b)),
            compiler_params=plgpu.CompilerParams(num_warps=num_warps),
            interpret=INTERPRET,
        )(jnp.stack([seed, i]), thp, tab, Xp)
        return out[: len(STATES)]
    return propagate


# ---- the particle filter around it (shared) --------------------------------------------

def make_pf(propagate, R, J):
    N = R * J
    dmeas = struct.dmeas_pf  # (y, X (J, nstates), theta, covars, t, should_trans) -> (J,)
    rinit = struct.rinit_pf

    def pf(seed):
        seed = seed.astype(U32)
        keys0 = jax.random.split(jax.random.fold_in(jax.random.key(1), seed), N)
        X0 = rinit(TH, keys0, jnp.asarray(struct.covars_extended[0]), struct.t0, False)  # (N, 6)
        rkey = jax.random.fold_in(jax.random.key(2), seed)

        def obs_step(carry, xs):
            X, ll = carry  # X: (6, N)
            i, y, tab = xs
            X = X.at[jnp.array(ACC)].set(0.0)
            X = propagate(X, TH, tab, seed, i.astype(U32))
            cov_t = jnp.asarray(struct.covars_extended)[(i + 1) * NSTEP]
            t_obs = TIMES[i]
            lw = dmeas(y, X.T, TH, cov_t, t_obs, False).reshape(R, J)
            nw, ll_t = jax.vmap(_normalize_weights)(lw - jnp.log(J))
            idx = jax.vmap(_resample)(nw, jax.random.split(jax.random.fold_in(rkey, i), R))
            X = jnp.take_along_axis(X.reshape(len(STATES), R, J), idx[None], axis=2).reshape(len(STATES), N)
            return (X, ll + ll_t), None

        (_, ll), _ = jax.lax.scan(obs_step, (X0.T, jnp.zeros(R)), (jnp.arange(T), YS, TAB))
        return ll
    return jax.jit(pf)


def make_stock(R, J):
    th = jnp.tile(model.theta.to_jax_array(PNAMES), (R, 1))

    def pf(seed):
        keys = jax.random.split(jax.random.key(seed), R)[:, None]
        return F.pfilter(struct, th, J, keys)["logLik"].ravel()
    return pf


def time_it(pf, n=3):
    t0 = time.perf_counter()
    ll = jax.block_until_ready(pf(jnp.uint32(0)))
    first = time.perf_counter() - t0
    ts = []
    for s in range(1, n + 1):
        t0 = time.perf_counter()
        jax.block_until_ready(pf(jnp.uint32(s)))
        ts.append(time.perf_counter() - t0)
    return first, statistics.median(ts), ll


def main():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Stock pypomp first, before the samplers are patched.
    shapes = [a for a in sys.argv[1:] if ":" in a] or ["1:5120", "36:5120", "1296:5120"]
    check = "--check" in sys.argv
    out = os.environ.get("PFB_RESULTS", "results_he10_fused.jsonl")
    rows = []

    def emit(variant, R, J, first, med, ll):
        row = dict(variant=variant, R=R, J=J, first_s=round(first, 2), median_s=round(med, 4),
                   us_per_step=round(med / (T * NSTEP) * 1e6, 2),
                   loglik_mean=round(float(jnp.mean(ll)), 2), loglik_sd=round(float(jnp.std(ll)), 2),
                   device=jax.devices()[0].platform, load=os.getloadavg()[0])
        print(json.dumps(row), flush=True)
        rows.append(row)
        if not check:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")

    if check:
        shapes = ["8:1024"]
    for shp in shapes:
        R, J = (int(x) for x in shp.split(":"))
        emit("stock", R, J, *time_it(make_stock(R, J)))
        # The explicit 11-stream step with pypomp's samplers as shipped.
        emit("xla/hash/stock-samplers", R, J, *time_it(make_pf(make_propagate_xla("hash"), R, J)))
        emit("xla/threefry/stock-samplers", R, J, *time_it(make_pf(make_propagate_xla("threefry"), R, J)))
    patch_samplers()
    patch_triton_const_folding()
    if check:
        n = 1024
        X = jnp.stack([jnp.full((n,), v, jnp.float32) for v in (1e5, 300.0, 300.0, 2.9e6, 0.0, 0.0)])
        a = propagate_xla(X, TH, TAB[100], U32(3), U32(100))
        b = make_propagate_pallas(256, 4)(X, TH, TAB[100], U32(3), U32(100))
        rel = jnp.max(jnp.abs(a - b) / (jnp.abs(a) + 1.0))
        print(f"check: one interval, max relative |dX| pallas vs xla = {float(rel):.2e}")
    for shp in shapes:
        R, J = (int(x) for x in shp.split(":"))
        emit("xla", R, J, *time_it(make_pf(propagate_xla, R, J)))
        emit("xla/threefry", R, J, *time_it(make_pf(make_propagate_xla("threefry"), R, J)))
        for b, w in ((128, 4), (256, 4), (512, 8)):
            if (R * J) % b == 0:
                emit(f"pallas/b{b}w{w}", R, J, *time_it(make_pf(make_propagate_pallas(b, w), R, J)))


if __name__ == "__main__":
    main()
