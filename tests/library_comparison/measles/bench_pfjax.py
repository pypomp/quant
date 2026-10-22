"""pfjax: He10 measles as a pfjax BaseModel, pfjax.particle_filter under jit(vmap) over filters.

The model is written per particle, the way pfjax expects (pfjax vmaps it over
particles with one PRNG key each). The data pytree y_meas carries, for each
observation, the case count plus the Euler grid and covariates of the interval
that ends at it (dt, pop, birthrate, term-time and cohort indicators, nstep), so
the time-inhomogeneous process needs no hidden time index. pf_step / pf_init
are overridden to pass that interval to the state sampler; pf_init draws the
initial state at t0 (pypomp's rinit) and pushes it through the first interval,
so the filter targets p(y_1..y_T) like pypomp.

Samplers (--samplers):
  jax     exact jax.random.gamma / poisson / binomial (as in quant's model_001b_jax.py)
  pypomp  pypomp.random's inverse-CDF samplers (via pypomp.models.measles._samplers, the
          same helpers model 001b uses), on per-particle uniforms; separates sampler cost
          from framework cost.
Resampling: pfjax's resample_multinomial, every observation (pfjax ships no systematic resampler).
float32. All nthetas*nreps(*units) filters are one jax.vmap; --chunk C runs them as lax.map
over vmapped chunks of C to bound memory. Note pfjax splits per-particle keys with
`key, *subkeys = split(key, J+1)`, so its jaxpr (and compile time) grows with J.

    python bench_pfjax.py --J 1000 --nthetas 3 --nreps 10 --device cpu [--panel] [--samplers pypomp]
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import he10 as H  # noqa: E402


def add_args(p):
    p.add_argument("--samplers", choices=["jax", "pypomp"], default="jax")
    p.add_argument("--chunk", type=int, default=0, help="filters per vmapped chunk (0 = all at once)")


args = H.parse_args("pfjax", add_args)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pfjax  # noqa: E402
from pfjax.models.base_model import BaseModel  # noqa: E402

if args.samplers == "pypomp":
    from pypomp.models.measles import _samplers as smp  # noqa: E402  (import outside any trace)

I = H.P  # parameter index in the kernel order
MU = H.MU
TOL = H.TOL


def euler_probs(r0, dt):
    rs = r0 + MU
    scale = (1.0 - jnp.exp(-rs * dt)) / rs
    return r0 * scale, MU * scale


def step_common(th, dt, pop, b, term, coh):
    cohort, amp = th[I["cohort"]], th[I["amplitude"]]
    br = jnp.where(coh, cohort * b / dt + (1 - cohort) * b, (1 - cohort) * b)
    seas = jnp.where(term, 1.0 + amp * 0.2411 / 0.7589, 1 - amp)
    beta = th[I["R0"]] * seas * (1.0 - jnp.exp(-(th[I["gamma"]] + MU) * dt)) / dt
    return br, beta


def finish(x, th, pop, dt, dw, births, x0, x1):
    S, E, Ii, R, W, C = x
    S = S + births - x0[0] - x1[0]
    E = E + x0[0] - x0[1] - x1[1]
    Ii = Ii + x0[1] - x0[2] - x1[2]
    R = pop - S - E - Ii
    W = W + (dw - dt) / th[I["sigmaSE"]]
    C = C + x0[2]
    return jnp.stack([S, E, Ii, R, W, C])


def step_jax(key, x, th, dt, pop, b, term, coh):
    """One Euler step for one particle with exact jax.random samplers."""
    br, beta = step_common(th, dt, pop, b, term, coh)
    S, E, Ii = x[0], x[1], x[2]
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
    return finish(x, th, pop, dt, dw, births, x0, x1)


def step_pypomp(key, x, th, dt, pop, b, term, coh):
    """One Euler step for one particle with pypomp.random's inverse-CDF samplers (as in model 001b)."""
    br, beta = step_common(th, dt, pop, b, term, coh)
    S, E, Ii = x[0:1], x[1:2], x[2:3]  # shape (1,): smp helpers expect a particle axis
    foi = beta * (Ii + th[I["iota"]]) / pop
    s2 = th[I["sigmaSE"]] ** 2
    u = jax.random.uniform(key, (smp.N_GAMMA_UNIFORMS + 7, 1))
    u_gamma, u = u[: smp.N_GAMMA_UNIFORMS], u[smp.N_GAMMA_UNIFORMS:]
    dw = smp.gamma(u_gamma, dt / s2) * s2
    births = smp.poisson(u[0], br * dt)
    tS0, tS1 = smp.euler_exits(u[1], u[2], S, foi * dw / dt, MU, dt)
    tE0, tE1 = smp.euler_exits(u[3], u[4], E, th[I["sigma"]], MU, dt)
    tI0, tI1 = smp.euler_exits(u[5], u[6], Ii, th[I["gamma"]], MU, dt)
    x0 = jnp.stack([tS0[0], tE0[0], tI0[0]])
    x1 = jnp.stack([tS1[0], tE1[0], tI1[0]])
    return finish(x, th, pop, dt, dw[0], births[0], x0, x1)


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


class He10(BaseModel):
    """State x = [S, E, I, R, W, C]; theta = 13 natural-scale parameters (he10.PARAMS order)."""

    def __init__(self, step, nstep_fixed=None):
        super().__init__(bootstrap=True)
        self.step = step
        self.nstep_fixed = nstep_fixed  # static substep count when every interval has the same nstep

    def rinit(self, theta, pop0):
        m = pop0 / (theta[I["S_0"]] + theta[I["E_0"]] + theta[I["I_0"]] + theta[I["R_0"]])
        S, E, Ii, R = (jnp.round(m * theta[I[k]]) for k in ("S_0", "E_0", "I_0", "R_0"))
        return jnp.stack([S, E, Ii, R, 0.0 * S, 0.0 * S])

    def interval(self, key, x, yc, theta):
        """Euler steps over the interval ending at observation yc (W, C reset first)."""
        x = x.at[4].set(0.0).at[5].set(0.0)

        def body(j, carry):
            x, key = carry
            key, sub = jax.random.split(key)
            x = self.step(sub, x, theta, yc["dt"][j], yc["pop"][j], yc["br"][j], yc["term"][j], yc["coh"][j])
            return x, key

        n = self.nstep_fixed if self.nstep_fixed is not None else yc["nstep"]
        return jax.lax.fori_loop(0, n, body, (x, key))[0]

    def meas_lpdf(self, y_curr, x_curr, theta):
        return dmeas(y_curr["y"], x_curr[5], theta)

    def pf_init(self, key, y_init, theta):
        x = self.interval(key, self.rinit(theta, y_init["pop"][0]), y_init, theta)
        return x, self.meas_lpdf(y_init, x, theta)

    def pf_step(self, key, x_prev, y_curr, theta):
        x = self.interval(key, x_prev, y_curr, theta)
        return x, self.meas_lpdf(y_curr, x, theta)


inp = H.Inputs(args.panel, args.T)
pad = inp.padded(np.float32)
n, R, U, J = args.nthetas, args.nreps, inp.U, args.J
Y = jnp.asarray(inp.ys, dtype=jnp.float32)  # (U, T)
POP, BR = jnp.asarray(pad["pop"]), jnp.asarray(pad["br"])  # (U, T, M)
shared = dict(dt=jnp.asarray(pad["dt"]), term=jnp.asarray(pad["term"]), coh=jnp.asarray(pad["coh"]),
              nstep=jnp.asarray(inp.nstep, dtype=jnp.int32))
model = He10(step_jax if args.samplers == "jax" else step_pypomp,
             nstep_fixed=int(inp.nstep[0]) if np.all(inp.nstep == inp.nstep[0]) else None)


def pf1(key, theta, u):
    y_meas = dict(y=Y[u], pop=POP[u], br=BR[u], **shared)
    out = pfjax.particle_filter(model, key, y_meas, theta, J,
                                resampler=pfjax.particle_resamplers.resample_multinomial)
    return out["loglik"]


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
    args, "pfjax" if args.samplers == "jax" else "pfjax+pypomp_samplers", inp, ll, cold, warm, **tm,
    samplers=("jax.random.gamma/poisson/binomial (exact)" if args.samplers == "jax"
              else "pypomp.random inverse-CDF (binominv/poissoninv/gammainv) on per-particle uniforms"),
    dtype="float32",
    versions=dict(H.versions("pfjax", "jax", "jaxlib", "pypomp"), jax_backend=jax.default_backend(),
                  pfjax_keysplit_fix=bool(os.environ.get("PFJAX_FIX_KEYSPLIT"))),
    resampling="multinomial every observation (pfjax.particle_resamplers.resample_multinomial)",
    chunk=C,
)
