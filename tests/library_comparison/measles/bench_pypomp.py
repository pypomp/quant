"""pypomp: He10 measles through pypomp.functional.pfilter / panel_pfilter (bootstrap PF).

The model is pypomp's own UKMeasles model 001b, built exactly as quant's
timing scripts build it (tests/measles/model.py, tests/panel_measles/model.py),
so it runs with pypomp's default fast samplers (pypomp.random inverse-CDF
binomial, Poisson and gamma inside a @vectorized rproc), float32, systematic
resampling at every observation (thresh=0). The struct's arrays are checked
against inputs.npz, so pypomp sees the same inputs as the other libraries.

Layouts (London):
  flat  every (theta, rep) filter is one vmapped lane: thetas (N, P), keys (N, 1)
  api   Pomp.pfilter's layout: thetas vmapped, reps run by lax.map: thetas (n, P), keys (n, reps)
Panel: PanelPomp.pfilter's layout, thetas (n*reps, U, P), keys (n*reps, U); --chunk-size units
are vmapped together (PanelPomp's default is 1, i.e. units in sequence; default here: all U).

    python bench_pypomp.py --J 1000 --nthetas 3 --nreps 10 --device cpu [--panel] [--layout api]
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import he10 as H  # noqa: E402


def add_args(p):
    p.add_argument("--layout", choices=["flat", "api"], default="flat")
    p.add_argument("--chunk-size", type=int, default=0, help="panel: units per vmapped chunk (0 = all units)")


args = H.parse_args("pypomp", add_args)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pypomp.functional as F  # noqa: E402

import export_inputs as X  # noqa: E402

inp = H.Inputs(args.panel, args.T)
T, K = inp.T, inp.K

if args.panel:
    PM = X.load_quant_model("panel_measles")
    obj = PM.panel_measles(PM.fixed_starts(36))
    s = obj.to_struct()
    s = s._replace(
        ys_per_unit=s.ys_per_unit[:, :T], times=s.times[:T], nstep_array=s.nstep_array[:T],
        dt_array_extended=s.dt_array_extended[:K], covars_per_unit=s.covars_per_unit[:, :K + 1],
    )
    canon = list(obj.unit_objects[inp.units[0]].canonical_param_names)
    ys_chk, cov_chk = np.asarray(s.ys_per_unit)[:, :, 0], np.asarray(s.covars_per_unit)
    assert list(s.unit_names) == inp.units
else:
    M = X.load_quant_model("measles")
    obj = M.measles("London", theta=M.fixed_starts(36))
    s = obj.to_struct()
    s = s._replace(
        ys=s.ys[:T], times=s.times[:T], nstep_array=s.nstep_array[:T],
        dt_array_extended=s.dt_array_extended[:K], covars_extended=s.covars_extended[:K + 1],
    )
    canon = list(obj.canonical_param_names)
    ys_chk, cov_chk = np.asarray(s.ys)[None, :, 0], np.asarray(s.covars_extended)[None]

# Same inputs as every other library.
cn = list(obj.unit_objects[inp.units[0]].covars.columns) if args.panel else list(obj.covars.columns)
assert np.array_equal(np.asarray(s.dt_array_extended, dtype=np.float64), inp.dt)
assert np.array_equal(np.asarray(s.nstep_array), inp.nstep)
assert np.array_equal(ys_chk.astype(np.float64), inp.ys)
assert np.array_equal(cov_chk[:, :, cn.index("pop")].astype(np.float64), inp.pop)
assert np.array_equal(cov_chk[:, :, cn.index("birthrate")].astype(np.float64), inp.birthrate)

n, R, U, J = args.nthetas, args.nreps, inp.U, args.J
th = inp.thetas(n)[..., [H.PARAMS.index(c) for c in canon]].astype(np.float32)  # (n, U, P) canonical order
assert np.array_equal(th.astype(np.float64), inp.thetas(n)[..., [H.PARAMS.index(c) for c in canon]])
key = jax.random.key(args.seed)

if args.panel:
    cs = args.chunk_size or U
    thetas = jnp.asarray(np.repeat(th, R, axis=0))  # (n*R, U, P), PanelPomp.pfilter's repeat
    keys = jax.random.split(key, n * R * U).reshape(n * R, U)

    def run():
        out = F.panel_pfilter(s, thetas, J, keys, chunk_size=cs)
        return np.asarray(jax.block_until_ready(out["logLik"])).reshape(n, R, U)

    layout = f"panel(n*reps vmapped, chunk_size={cs})"
elif args.layout == "flat":
    thetas = jnp.asarray(np.repeat(th[:, 0], R, axis=0))  # (n*R, P)
    keys = jax.random.split(key, n * R)[:, None]

    def run():
        out = F.pfilter(s, thetas, J, keys)
        return np.asarray(jax.block_until_ready(out["logLik"])).reshape(n, R, 1)

    layout = "flat(n*reps vmapped)"
else:
    thetas = jnp.asarray(th[:, 0])  # (n, P)
    keys = jax.random.split(key, n * R).reshape(n, R)

    def run():
        out = F.pfilter(s, thetas, J, keys)
        return np.asarray(jax.block_until_ready(out["logLik"])).reshape(n, R, 1)

    layout = "api(thetas vmapped, reps lax.map)"

ll, tm = H.timed_calls(run, args.calls)
cold, warm = tm.pop("cold_s"), tm.pop("warm_s", None)

H.record(
    args, "pypomp", inp, ll, cold, warm, **tm,
    samplers="pypomp.random fast inverse-CDF (binominv/poissoninv/gammainv), model 001b @vectorized rproc",
    dtype="float32",
    versions=dict(H.versions("pypomp", "jax", "jaxlib"), jax_backend=jax.default_backend()),
    layout=layout,
    resampling="systematic every observation (thresh=0)",
)
