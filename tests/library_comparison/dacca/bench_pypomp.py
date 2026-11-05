"""pypomp: quant's dacca model through pypomp.functional.pfilter on the model struct.

Systematic resampling at every observation (thresh = 0, pypomp's default), float32.
--layout vmap: all nthetas x nreps filters vmapped at once (theta tiled, one key each).
--layout map: pypomp's own batching (Pomp.pfilter): vmap over thetas, lax.map over the reps.
--layout auto (default): vmap on GPU, map on CPU. On CPU (J = 1000, 100 filters of one theta)
  map took 25 s and vmap 65 s warm; on GPU, run both and keep the faster.

usage: .venv-jax/bin/python bench_pypomp.py --J 5000 --nthetas 36 --nreps 36 --device gpu
"""

import os
import sys
import time

import dacca_common as DC


def _extra(ap):
    ap.add_argument("--layout", choices=["auto", "vmap", "map"], default="auto")


args = DC.parse_args("pypomp", extra=_extra)
if args.device == "cpu":
    DC.set_cpu_env(threads=os.cpu_count())
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pypomp.functional as F  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(DC.HERE, "..", "..", "dacca")))
import model as dacca_model  # noqa: E402  (quant tests/dacca/model.py)

assert jax.devices()[0].platform == args.device, jax.devices()
if args.layout == "auto":
    args.layout = "vmap" if args.device == "gpu" else "map"
inp = DC.load_inputs(args.T, args.theta_set)
T, n = inp["T"], inp["dt"].size

struct = dacca_model.dacca().to_struct()
struct = struct._replace(
    ys=struct.ys[:T], times=struct.times[:T], nstep_array=struct.nstep_array[:T],
    dt_array_extended=struct.dt_array_extended[:n], covars_extended=struct.covars_extended[: n + 1],
)
# Same arrays as everyone else gets from inputs.npz.
assert list(struct.param_names) == inp["paramnames"]
assert np.array_equal(np.asarray(struct.covars_extended), inp["covars"])
assert np.array_equal(np.asarray(struct.dt_array_extended), inp["dt"])
assert np.array_equal(np.asarray(struct.ys)[:, 0], inp["ys"].astype(np.float32))

thetas = jnp.asarray(inp["thetas"][: args.nthetas], dtype=jnp.float32)
nth, nrep, J = args.nthetas, args.nreps, args.J

if args.layout == "vmap":
    th_in = jnp.repeat(thetas, nrep, axis=0)  # (nth * nrep, d)

    @jax.jit
    def workload(key):
        keys = jax.random.split(key, nth * nrep).reshape(nth * nrep, 1)
        return F.pfilter(struct, th_in, J, keys)["logLik"].reshape(nth, nrep)
else:
    th_in = thetas

    @jax.jit
    def workload(key):
        keys = jax.random.split(key, nth * nrep).reshape(nth, nrep)
        return F.pfilter(struct, th_in, J, keys)["logLik"]


def call(c):
    t0 = time.perf_counter()
    ll = np.asarray(jax.block_until_ready(workload(jax.random.fold_in(jax.random.key(args.seed), c))))
    return time.perf_counter() - t0, ll


cold_s, ll = call(0)
warm_s, _ = call(1)
DC.record(
    "pypomp", args, inp, cold_s, warm_s, ll, DC.versions("pypomp", "jax", "jaxlib"),
    variant=f"functional.pfilter/{args.layout}", dtype="float32", resampler="systematic",
    env_toggles={k: os.environ.get(k) for k in ("PYPOMP_SUBSTEP_UNROLL",)},
)
