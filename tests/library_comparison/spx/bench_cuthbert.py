"""cuthbert: SPX as cuthbert.smc.particle_filter functions, run by cuthbert.filter, jit(vmap) over all filters (float32).

cuthbert's model functions get no time and no theta: theta is closed over (the filter is built
inside the per-filter function, as cuthbert's examples do) and the leverage covariate rides in
the model inputs, model_inputs[n] = {y: y_n, y_prev: y_prev_n}, where y_prev_n = covars_extended[n]
is what pypomp's rproc reads on the step into observation n.
cuthbert draws x0 = (V_0, S0) at t0 with no weight and weights only after each propagation, so
the filter targets p(y_1..y_T) as pypomp does.
Resampling: cuthbertlib's systematic resampling at every observation (no ESS decorator).
The loglik is the last log_normalizing_constant of the states cuthbert.filter returns.

All nthetas*nreps filters go through one jax.vmap; --chunk N runs them N at a time with
jax.lax.map (for memory). Runs from .venv-ssm (cuthbert is not installed in .venv-jax).

usage: python bench_cuthbert.py --J 1000 --nthetas 36 --nreps 24 --device gpu [--chunk 0]
"""

import spx_common as C


def _extra(p):
    p.add_argument("--chunk", type=int, default=0, help="filters per lax.map chunk (0 = all in one vmap)")


args = C.parse_args("cuthbert", extra=_extra)

import cuthbert  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from cuthbert.smc import particle_filter  # noqa: E402
from cuthbertlib.resampling import systematic  # noqa: E402

C.check_jax_device(args, jax)
inp = C.load_inputs(args.T, args.nthetas, args.theta_set)
T = len(inp["ys"])
S0 = float(inp["S0"])
VF = float(inp["V_floor"])
norm_lpdf = jax.scipy.stats.norm.logpdf


def init_sample(theta):
    """x0 = (V_0, S0), deterministic. theta = (mu, kappa, theta, xi, rho, V_0)."""
    return jnp.stack([theta[5], jnp.asarray(S0, theta.dtype)])


def propagate_sample(key, x_prev, mi, theta):
    mu, kappa, th, xi, rho = theta[0], theta[1], theta[2], theta[3], theta[4]
    V, S = x_prev[0], x_prev[1]
    dZ = jax.random.normal(key)
    dWs = (mi["y_prev"] - mu + 0.5 * V) / jnp.sqrt(V)
    dWv = rho * dWs + jnp.sqrt(1 - rho**2) * dZ
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, VF)) * dWs)
    V = V + kappa * (th - V) + xi * jnp.sqrt(V) * dWv
    return jnp.stack([jnp.maximum(V, VF), S])


def log_potential(x, mi, theta):
    V = x[0]
    return norm_lpdf(mi["y"], theta[0] - 0.5 * V, jnp.sqrt(V))


model_inputs = {"y": jnp.asarray(inp["ys"], dtype=jnp.float32),
                "y_prev": jnp.asarray(inp["y_prev"], dtype=jnp.float32)}

J, nth, nrep = args.J, args.nthetas, args.nreps
B = nth * nrep
th_all = jnp.repeat(jnp.asarray(inp["thetas"], dtype=jnp.float32), nrep, axis=0)  # (B, 6)
keys = jax.random.split(jax.random.key(args.seed), B)


def pf1(theta, key):
    pf_obj = particle_filter.build_filter(
        init_sample=lambda k: init_sample(theta),
        propagate_sample=lambda k, x, mi: propagate_sample(k, x, mi, theta),
        log_potential=lambda x_prev, x, mi: log_potential(x, mi, theta),
        n_filter_particles=J,
        resampling_fn=systematic.resampling,
    )
    init_key, filter_key = jax.random.split(key)
    states = cuthbert.filter(pf_obj, model_inputs, pf_obj.init_prepare(key=init_key), key=filter_key)
    return states.log_normalizing_constant[-1]


vpf = jax.vmap(pf1)
chunk = args.chunk if 0 < args.chunk < B else B
if chunk == B:
    pf = jax.jit(vpf)
else:
    if B % chunk:
        raise SystemExit(f"--chunk must divide nthetas*nreps = {B}")

    @jax.jit
    def pf(th, k):
        return jax.lax.map(lambda a: vpf(*a), (th.reshape(-1, chunk, th.shape[1]), k.reshape(-1, chunk))).reshape(-1)


def run():
    return np.asarray(jax.block_until_ready(pf(th_all, keys))).reshape(nth, nrep)


cold, ll = C.timed(run)
warm, ll2 = C.timed(run)
assert np.array_equal(ll, ll2, equal_nan=True)
C.record(
    args, "cuthbert", T, cold, warm, ll,
    C.versions("cuthbert", "cuthbertlib", "jax", "jaxlib"),
    variant="filter/vmap" + (f"/chunk{chunk}" if chunk < B else ""),
    dtype="float32", resampler="systematic", jax_device=str(jax.devices()[0]),
)
