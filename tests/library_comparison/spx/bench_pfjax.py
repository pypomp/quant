"""pfjax: SPX as a pfjax BaseModel; pfjax.particle_filter, jit(vmap) over all filters (float32).

The leverage covariate rides in the measurement pytree: y_meas[n] = (y_n, y_prev_n), where
y_prev_n = covars_extended[n] is what pypomp's rproc reads on the step into observation n.
pfjax observes its initial state, so pf_init draws x0 = (V_0, S0) and applies one step before
weighting by y_0; every library therefore targets p(y_1..y_T).
pfjax ships only multinomial resampling (every observation). Its particle_filter unpacks one
key per particle in Python (key, *subkeys = split(...)), so trace/compile time grows with J.

All nthetas*nreps filters go through one jax.vmap; --chunk N runs them N at a time with
jax.lax.map (for memory). --resampler systematic swaps in a user-written systematic resampler
(validation diagnostic only: it separates resampling variance from model math).

usage: python bench_pfjax.py --J 1000 --nthetas 36 --nreps 24 --device gpu [--chunk 0]
"""

import spx_common as C


def _extra(p):
    p.add_argument("--chunk", type=int, default=0, help="filters per lax.map chunk (0 = all in one vmap)")
    p.add_argument("--resampler", choices=["multinomial", "systematic"], default="multinomial",
                   help="multinomial = pfjax's own (benchmark); systematic = diagnostic")


args = C.parse_args("pfjax", extra=_extra)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pfjax  # noqa: E402
from pfjax.models.base_model import BaseModel  # noqa: E402

C.check_jax_device(args, jax)
inp = C.load_inputs(args.T, args.nthetas, args.theta_set)
T = len(inp["ys"])
S0 = float(inp["S0"])
VF = float(inp["V_floor"])
norm_lpdf = jax.scipy.stats.norm.logpdf


class SPX(BaseModel):
    """theta = (mu, kappa, theta, xi, rho, V_0); x = (V, S); y = (y, y_prev)."""

    def __init__(self):
        super().__init__(bootstrap=True)

    def _move(self, key, x_prev, y_prev, theta):
        mu, kappa, th, xi, rho = theta[0], theta[1], theta[2], theta[3], theta[4]
        V, S = x_prev[0], x_prev[1]
        dZ = jax.random.normal(key)
        dWs = (y_prev - mu + 0.5 * V) / jnp.sqrt(V)
        dWv = rho * dWs + jnp.sqrt(1 - rho**2) * dZ
        S = S + S * (mu + jnp.sqrt(jnp.maximum(V, VF)) * dWs)
        V = V + kappa * (th - V) + xi * jnp.sqrt(V) * dWv
        return jnp.stack([jnp.maximum(V, VF), S])

    def meas_lpdf(self, y_curr, x_curr, theta):
        V = x_curr[0]
        return norm_lpdf(y_curr[0], theta[0] - 0.5 * V, jnp.sqrt(V))

    def pf_step(self, key, x_prev, y_curr, theta):
        x = self._move(key, x_prev, y_curr[1], theta)
        return x, self.meas_lpdf(y_curr, x, theta)

    def pf_init(self, key, y_init, theta):
        x0 = jnp.stack([theta[5], jnp.asarray(S0, theta.dtype)])
        x = self._move(key, x0, y_init[1], theta)
        return x, self.meas_lpdf(y_init, x, theta)


model = SPX()
y_meas = jnp.asarray(np.stack([inp["ys"], inp["y_prev"]], axis=1), dtype=jnp.float32)

J, nth, nrep = args.J, args.nthetas, args.nreps
B = nth * nrep
th_all = jnp.repeat(jnp.asarray(inp["thetas"], dtype=jnp.float32), nrep, axis=0)  # (B, 6)
keys = jax.random.split(jax.random.key(args.seed), B)


def resample_systematic(key, x_particles_prev, logw):
    """Diagnostic only: systematic resampling in pfjax's resampler interface.

    A float32 cumsum is neither monotone nor flat across zero-weight particles, so (as pypomp
    does) the cumulative weights are made monotone and flat across them, the points are scaled
    to the actual total, and indices are capped at the last positive-weight particle. Without
    this, a collapsed weight vector can select zero-weight particles.
    """
    n = logw.size
    prob = jnp.exp(logw - jax.scipy.special.logsumexp(logw))
    pos = prob > 0
    cum = jax.lax.cummax(jnp.where(pos, jnp.cumsum(prob), -jnp.inf))
    u = (jax.random.uniform(key, dtype=logw.dtype) + jnp.arange(n)) / n * cum[-1]
    last = n - 1 - jnp.argmax(pos[::-1])
    anc = jnp.minimum(jnp.searchsorted(cum, u), last)
    return {"x_particles": x_particles_prev[anc], "ancestors": anc}


RESAMPLER = (pfjax.particle_resamplers.resample_multinomial if args.resampler == "multinomial"
             else resample_systematic)


def pf1(theta, key):
    return pfjax.particle_filter(model, key, y_meas, theta, J, resampler=RESAMPLER)["loglik"]


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
    args, "pfjax", T, cold, warm, ll,
    C.versions("pfjax", "jax", "jaxlib"),
    variant="particle_filter/vmap" + (f"/chunk{chunk}" if chunk < B else "")
    + ("" if args.resampler == "multinomial" else "/DIAGNOSTIC-systematic"),
    dtype="float32", resampler=args.resampler, jax_device=str(jax.devices()[0]),
)
