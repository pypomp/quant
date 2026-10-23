"""pypomp: pypomp.models.spx() through pypomp.functional.pfilter (float32, systematic, thresh=0).

layout nested (default): thetas (nthetas, d), keys (nthetas, nreps); pypomp vmaps over
    thetas and lax.maps over the reps, as Pomp.pfilter(reps=...) in tests/spx/timing/run.py does.
layout flat: thetas repeated to (nthetas*nreps, d), keys (nthetas*nreps, 1), so every filter
    runs in one vmap.
The call is wrapped in jax.jit (struct closed over), so a warm call is a single dispatch.

usage: python bench_pypomp.py --J 1000 --nthetas 36 --nreps 24 --device gpu [--layout flat]
"""

import spx_common as C


def _extra(p):
    p.add_argument("--layout", choices=["nested", "flat"], default="nested")


args = C.parse_args("pypomp", extra=_extra)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pypomp as pp  # noqa: E402
import pypomp.functional as F  # noqa: E402

C.check_jax_device(args, jax)
inp = C.load_inputs(args.T, args.nthetas, args.theta_set)
full = C.load_inputs()
T = len(inp["ys"])

struct = pp.models.spx().to_struct()
# The struct must be exactly what the other libraries read from inputs.npz.
assert list(struct.param_names) == list(full["param_names"])
assert np.array_equal(np.asarray(struct.ys)[:, 0], full["ys"])
assert np.array_equal(np.asarray(struct.covars_extended)[:, 0], full["covars_extended"])
assert np.array_equal(np.asarray(struct.dt_array_extended), full["dt_array_extended"])
assert np.array_equal(np.asarray(struct.nstep_array), full["nstep_array"])
struct = struct._replace(
    ys=struct.ys[:T],
    times=struct.times[:T],
    dt_array_extended=struct.dt_array_extended[:T],
    nstep_array=struct.nstep_array[:T],
    covars_extended=struct.covars_extended[: T + 1],
)

J, nth, nrep = args.J, args.nthetas, args.nreps
thetas = jnp.asarray(inp["thetas"], dtype=jnp.float32)
keys = jax.random.split(jax.random.key(args.seed), nth * nrep)

if args.layout == "nested":
    th_in, keys_in = thetas, keys.reshape(nth, nrep)
else:
    th_in, keys_in = jnp.repeat(thetas, nrep, axis=0), keys.reshape(nth * nrep, 1)

pf = jax.jit(lambda th, k: F.pfilter(struct, th, J, k)["logLik"])


def run():
    return np.asarray(jax.block_until_ready(pf(th_in, keys_in))).reshape(nth, nrep)


cold, ll = C.timed(run)
warm, ll2 = C.timed(run)
assert np.array_equal(ll, ll2, equal_nan=True)
C.record(
    args, "pypomp", T, cold, warm, ll,
    C.versions("pypomp", "jax", "jaxlib"),
    variant=f"functional/{args.layout}", dtype="float32", resampler="systematic",
    jax_device=str(jax.devices()[0]),
)
