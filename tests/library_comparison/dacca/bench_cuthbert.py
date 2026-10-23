"""cuthbert: dacca as cuthbert.smc.particle_filter functions, run by cuthbert.filter under jax.jit, float32.

cuthbert's model is three functions: init_sample(key), propagate_sample(key, x_prev,
model_inputs) and log_potential(x_prev, x, model_inputs). They get no time and no theta, so
theta is closed over (the filter is built inside the per-filter function, as cuthbert's
examples do) and each observation's model_inputs carry its interval's covariate block and step
sizes: {deaths (T,), cov (T, nstep, C), dt (T, nstep)}. propagate_sample runs the nstep Euler
steps of that interval (one lax.scan).
cuthbert draws x(t0) with no weight and weights only after each propagation, so the filter
targets p(y_1..y_T) with rinit at t0, as pypomp does.
Resampling: cuthbertlib's systematic resampling at every observation (no ESS decorator).
The loglik is the last log_normalizing_constant of the states cuthbert.filter returns (XLA
drops the unused particle history; peak memory does not grow with T).
All nthetas x nreps filters run as one jax.vmap (or lax.map over --chunk sized vmaps).
Runs from .venv-ssm (cuthbert is not installed in .venv-jax).

usage: .venv-ssm/bin/python bench_cuthbert.py --J 5000 --nthetas 36 --nreps 36 --device gpu
"""

import os
import time

import dacca_common as DC


def _extra(ap):
    ap.add_argument("--chunk", type=int, default=0, help="filters per vmap (0 = all at once)")


args = DC.parse_args("cuthbert", extra=_extra)
if args.device == "cpu":
    DC.set_cpu_env(threads=os.cpu_count())
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import cuthbert  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from cuthbert.smc import particle_filter  # noqa: E402
from cuthbertlib.resampling import systematic  # noqa: E402

assert jax.devices()[0].platform == args.device, jax.devices()
inp = DC.load_inputs(args.T, args.theta_set)
P = {k: i for i, k in enumerate(inp["paramnames"])}
norm_lpdf = jax.scipy.stats.norm.logpdf
TOL, LTOL = DC.TOL, DC.LTOL
COV0 = jnp.asarray(inp["cov0"], dtype=jnp.float32)


def rinit(theta):
    ivp = jnp.stack([theta[P[k]] for k in ("S_0", "I_0", "Y_0", "R1_0", "R2_0", "R3_0")])
    S, I, Y, R1, R2, R3 = COV0[2] * ivp / jnp.sum(ivp)
    z = jnp.zeros_like(S)
    return jnp.stack([S, I, Y, z, R1, R2, R3, z])


def interval(key, x, theta, cov, dts):
    """One observation interval for one particle: x (8,), cov (nstep, C), dts (nstep,)."""
    th = {k: theta[i] for k, i in P.items()}
    bs = jnp.stack([th[f"bs{i}"] for i in range(1, 7)])
    om = jnp.stack([th[f"omegas{i}"] for i in range(1, 7)])
    neps = 3.0 * th["epsilon"]
    eps = jax.random.normal(key, dts.shape)

    def euler(carry, inp_k):
        S, I, Y, Mn, R1, R2, R3, cnt = carry
        c, dt, e = inp_k
        dpopdt, pop, seas = c[1], c[2], c[3:9]
        beta = jnp.exp(th["beta_trend"] * c[0] + jnp.dot(seas, bs))
        omega = jnp.exp(jnp.dot(seas, om))
        dw = e * jnp.sqrt(dt)
        effI = (I / pop) ** th["alpha"]
        inf = (omega + (beta + th["sigma"] * dw / dt) * effI) * S
        pass0, pass1, pass2, pass3 = th["gamma"] * I, neps * R1, neps * R2, neps * R3
        disease = th["m"] * I
        d = th["delta"]
        S = S + (dpopdt + d * pop - inf - d * S + pass3 + th["rho"] * Y) * dt
        I = I + (th["c"] * inf - disease - d * I - pass0) * dt
        Y = Y + ((1 - th["c"]) * inf - d * Y - th["rho"] * Y) * dt
        R1n = R1 + (pass0 - pass1 - d * R1) * dt
        R2n = R2 + (pass1 - pass2 - d * R2) * dt
        R3n = R3 + (pass2 - pass3 - d * R3) * dt
        Mn = Mn + disease * dt
        new = (S, I, Y, Mn, R1n, R2n, R3n)
        cnt = cnt + jnp.any(jnp.stack(new) < 0).astype(cnt.dtype)
        return tuple(jnp.maximum(v, 0.0) for v in new) + (cnt,), None

    S, I, Y, _, R1, R2, R3, cnt = x  # Mn (accumulator) restarts at 0
    out, _ = jax.lax.scan(euler, (S, I, Y, jnp.zeros_like(S), R1, R2, R3, cnt), (cov, dts, eps))
    return jnp.stack(out)


def dmeas(deaths, x, theta):
    Mn, cnt = x[3], x[7]
    v = theta[P["tau"]] * Mn
    ll = jnp.logaddexp(norm_lpdf(deaths, Mn, v + TOL), LTOL)
    return jnp.where(jnp.isfinite(v) & (cnt <= 0), ll, LTOL)


model_inputs = {
    "deaths": jnp.asarray(inp["ys"], dtype=jnp.float32),
    "cov": jnp.asarray(inp["cov_blocks"]),
    "dt": jnp.asarray(inp["dt_blocks"]),
}
nth, nrep, J = args.nthetas, args.nreps, args.J
thetas = jnp.repeat(jnp.asarray(inp["thetas"][:nth], dtype=jnp.float32), nrep, axis=0)


def pf1(theta, key):
    pf_obj = particle_filter.build_filter(
        init_sample=lambda k: rinit(theta),
        propagate_sample=lambda k, x, mi: interval(k, x, theta, mi["cov"], mi["dt"]),
        log_potential=lambda x_prev, x, mi: dmeas(mi["deaths"], x, theta),
        n_filter_particles=J,
        resampling_fn=systematic.resampling,
    )
    init_key, filter_key = jax.random.split(key)
    states = cuthbert.filter(pf_obj, model_inputs, pf_obj.init_prepare(key=init_key), key=filter_key)
    return states.log_normalizing_constant[-1]


@jax.jit
def workload(key):
    keys = jax.random.split(key, nth * nrep)
    if args.chunk and args.chunk < nth * nrep:
        ll = jax.lax.map(lambda a: pf1(*a), (thetas, keys), batch_size=args.chunk)
    else:
        ll = jax.vmap(pf1)(thetas, keys)
    return ll.reshape(nth, nrep)


def call(c):
    t0 = time.perf_counter()
    ll = np.asarray(jax.block_until_ready(workload(jax.random.fold_in(jax.random.key(args.seed), c))))
    return time.perf_counter() - t0, ll


cold_s, ll = call(0)
warm_s, _ = call(1)
DC.record("cuthbert", args, inp, cold_s, warm_s, ll, DC.versions("cuthbert", "cuthbertlib", "jax", "jaxlib"),
          variant="filter/systematic", dtype="float32", resampler="systematic", chunk=args.chunk,
          jax_device=str(jax.devices()[0]))
