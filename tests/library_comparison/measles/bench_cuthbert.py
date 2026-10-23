"""cuthbert: He10 measles as cuthbert.smc.particle_filter functions, run by cuthbert.filter under jit(vmap) over filters.

cuthbert's model is three per-particle functions (cuthbert vmaps them over particles with one
PRNG key each): init_sample(key), propagate_sample(key, x_prev, model_inputs) and
log_potential(x_prev, x, model_inputs). They get no time and no theta, so theta and the unit's
initial population are closed over (the filter is built inside the per-filter function, as
cuthbert's examples do), and each observation's model_inputs carry the case count plus the
Euler grid and covariates of the interval that ends at it (dt, pop, birthrate, term-time and
cohort indicators, nstep), as in the pfjax adapter. cuthbert draws x(t0) with pypomp's rinit and
no weight, and weights only after each propagation, so the filter targets p(y_1..y_T) as pypomp
does.

Samplers: exact jax.random.gamma / poisson / binomial, as a JAX user writes the model (and as the
pfjax adapter's default does); pypomp uses its inverse-CDF samplers.
Resampling: cuthbertlib's systematic resampling at every observation (no ESS decorator).
The loglik is the last log_normalizing_constant of the states cuthbert.filter returns.
float32. All nthetas*nreps(*units) filters are one jax.vmap; --chunk C runs them as lax.map over
vmapped chunks of C to bound memory. Runs from .venv-ssm (cuthbert is not installed in .venv-jax).

    python bench_cuthbert.py --J 1000 --nthetas 3 --nreps 10 --device cpu [--panel]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import he10 as H  # noqa: E402


def add_args(p):
    p.add_argument("--chunk", type=int, default=0, help="filters per vmapped chunk (0 = all at once)")


args = H.parse_args("cuthbert", add_args)

import cuthbert  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from cuthbert.smc import particle_filter  # noqa: E402
from cuthbertlib.resampling import systematic  # noqa: E402

H.check_device(args, jax.default_backend())

I = H.P  # parameter index in the kernel order
MU = H.MU
TOL = H.TOL


def euler_probs(r0, dt):
    rs = r0 + MU
    scale = (1.0 - jnp.exp(-rs * dt)) / rs
    return r0 * scale, MU * scale


def step(key, x, th, dt, pop, b, term, coh):
    """One Euler step for one particle with exact jax.random samplers."""
    cohort, amp = th[I["cohort"]], th[I["amplitude"]]
    br = jnp.where(coh, cohort * b / dt + (1 - cohort) * b, (1 - cohort) * b)
    seas = jnp.where(term, 1.0 + amp * 0.2411 / 0.7589, 1 - amp)
    beta = th[I["R0"]] * seas * (1.0 - jnp.exp(-(th[I["gamma"]] + MU) * dt)) / dt
    S, E, Ii, R, W, C = x
    foi = beta * (Ii + th[I["iota"]]) / pop
    s2 = th[I["sigmaSE"]] ** 2
    kg, kp, kb0, kb1 = jax.random.split(key, 4)
    dw = jax.random.gamma(kg, dt / s2) * s2
    births = jax.random.poisson(kp, br * dt).astype(jnp.float32)
    r0 = jnp.stack([foi * dw / dt, th[I["sigma"]], th[I["gamma"]]])
    p0, p1 = euler_probs(r0, dt)
    n = jnp.stack([S, E, Ii])
    x0 = jax.random.binomial(kb0, n, jnp.clip(p0, 0.0, 1.0))
    prem = 1.0 - p0
    x1 = jax.random.binomial(kb1, n - x0, jnp.clip(p1 / jnp.where(prem > 0.0, prem, 1.0), 0.0, 1.0))
    S = S + births - x0[0] - x1[0]
    E = E + x0[0] - x0[1] - x1[1]
    Ii = Ii + x0[1] - x0[2] - x1[2]
    R = pop - S - E - Ii
    W = W + (dw - dt) / th[I["sigmaSE"]]
    C = C + x0[2]
    return jnp.stack([S, E, Ii, R, W, C])


def rinit(theta, pop0):
    m = pop0 / (theta[I["S_0"]] + theta[I["E_0"]] + theta[I["I_0"]] + theta[I["R_0"]])
    S, E, Ii, R = (jnp.round(m * theta[I[k]]) for k in ("S_0", "E_0", "I_0", "R_0"))
    return jnp.stack([S, E, Ii, R, 0.0 * S, 0.0 * S])


def interval(key, x, mi, theta, nstep_fixed):
    """Euler steps over the interval ending at the observation in mi (W, C reset first)."""
    x = x.at[4].set(0.0).at[5].set(0.0)

    def body(j, carry):
        x, key = carry
        key, sub = jax.random.split(key)
        x = step(sub, x, theta, mi["dt"][j], mi["pop"][j], mi["br"][j], mi["term"][j], mi["coh"][j])
        return x, key

    n = nstep_fixed if nstep_fixed is not None else mi["nstep"]
    return jax.lax.fori_loop(0, n, body, (x, key))[0]


def dmeas(y, C, th):
    rho, psi = th[I["rho"]], th[I["psi"]]
    m = rho * C
    v = m * (1.0 - rho + psi ** 2 * m)
    sd = jnp.sqrt(v) + TOL
    upper = jax.scipy.stats.norm.cdf(y + 0.5, m, sd)
    lower = jax.scipy.stats.norm.cdf(y - 0.5, m, sd)
    lik = jnp.where(y > TOL, upper - lower, upper) + TOL
    lik = jnp.where(C < 0, 0.0, lik)
    lik = jnp.where(jnp.isnan(y), 1.0, lik)
    return jnp.log(lik)


inp = H.Inputs(args.panel, args.T)
pad = inp.padded(np.float32)
n, R, U, J = args.nthetas, args.nreps, inp.U, args.J
Y = jnp.asarray(inp.ys, dtype=jnp.float32)  # (U, T)
POP, BR = jnp.asarray(pad["pop"]), jnp.asarray(pad["br"])  # (U, T, M)
shared = dict(dt=jnp.asarray(pad["dt"]), term=jnp.asarray(pad["term"]), coh=jnp.asarray(pad["coh"]),
              nstep=jnp.asarray(inp.nstep, dtype=jnp.int32))
NSTEP_FIXED = int(inp.nstep[0]) if np.all(inp.nstep == inp.nstep[0]) else None


def pf1(key, theta, u):
    model_inputs = dict(y=Y[u], pop=POP[u], br=BR[u], **shared)
    pop0 = POP[u, 0, 0]  # covariate at Euler step 0, which also feeds rinit
    pf_obj = particle_filter.build_filter(
        init_sample=lambda k: rinit(theta, pop0),
        propagate_sample=lambda k, x, mi: interval(k, x, mi, theta, NSTEP_FIXED),
        log_potential=lambda x_prev, x, mi: dmeas(mi["y"], x[5], theta),
        n_filter_particles=J,
        resampling_fn=systematic.resampling,
    )
    init_key, filter_key = jax.random.split(key)
    states = cuthbert.filter(pf_obj, model_inputs, pf_obj.init_prepare(key=init_key), key=filter_key)
    return states.log_normalizing_constant[-1]


tasks = H.task_list(n, R, U)
N = len(tasks)
thetas = jnp.asarray(np.stack([inp.thetas(n)[i, u] for i, r, u in tasks]), dtype=jnp.float32)
units = jnp.asarray([u for i, r, u in tasks], dtype=jnp.int32)
keys = jax.random.split(jax.random.key(args.seed), N)
C = args.chunk if 0 < args.chunk < N else N
nb = -(-N // C)
padn = nb * C - N
if padn:
    thetas = jnp.concatenate([thetas, jnp.repeat(thetas[-1:], padn, 0)])
    units = jnp.concatenate([units, jnp.repeat(units[-1:], padn, 0)])
    keys = jnp.concatenate([keys, jnp.repeat(keys[-1:], padn, 0)])
vpf = jax.vmap(pf1)


@jax.jit
def run_all(keys, thetas, units):
    if nb == 1:
        return vpf(keys, thetas, units)
    xs = (keys.reshape(nb, C), thetas.reshape(nb, C, -1), units.reshape(nb, C))
    return jax.lax.map(lambda a: vpf(*a), xs).reshape(-1)


def run():
    out = np.asarray(jax.block_until_ready(run_all(keys, thetas, units)))[:N]
    return out.reshape(n, R, U)


ll, tm = H.timed_calls(run, args.calls)
cold, warm = tm.pop("cold_s"), tm.pop("warm_s", None)

H.record(
    args, "cuthbert", inp, ll, cold, warm, **tm,
    samplers="jax.random.gamma/poisson/binomial (exact)",
    dtype="float32",
    versions=dict(H.versions("cuthbert", "cuthbertlib", "jax", "jaxlib"), jax_backend=jax.default_backend()),
    resampling="systematic every observation (cuthbertlib.resampling.systematic)",
    chunk=C,
    jax_device=str(jax.devices()[0]),
)
