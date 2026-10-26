"""Profile one He10 London observation interval (7 Euler steps) at the paper workload.

usage: python profile_he10.py {stock,unroll,cute} OUTDIR
stock/unroll use the bench pypomp (PYPOMP_SUBSTEP_UNROLL=1 for unroll); cute needs the
CuTe branch on PYTHONPATH. Writes a perfetto trace under OUTDIR and prints a JSON summary.
"""
import glob, gzip, json, os, sys, time
from collections import Counter, defaultdict
import jax, jax.numpy as jnp, numpy as np
import pypomp as pp

variant, out = sys.argv[1], sys.argv[2]
R, J, I_OBS, CALLS = 1296, 5000, 3, 3
THETA = {"R0": 56.8, "sigma": 28.9, "gamma": 30.4, "iota": 2.9, "rho": 0.488, "sigmaSE": 0.0878,
         "psi": 0.116, "cohort": 0.557, "amplitude": 0.554, "S_0": 2.97e-02, "E_0": 5.17e-05,
         "I_0": 5.14e-05, "R_0": 9.70e-01}
model = pp.models.UKMeasles.pomp(unit="London", theta=pp.PompParameters(THETA), clean=True, model="001b")
if variant == "cute":
    model.set_rproc_backend("cute")
s = model.to_struct()
if variant == "cute":
    assert s.rproc_pf is not model.rproc.mechanics_pf_interp, "CuTe not chosen"
th = model.theta.to_jax_array(model.canonical_param_names)[0]
cov = s.covars_extended
X0 = jax.vmap(lambda k: s.rinit_pf(th, jax.random.split(k, J), None if cov is None else cov[0], s.t0, False))(
    jax.random.split(jax.random.key(9), R))
nsteps = np.asarray(s.nstep_array)
nstep, t_idx, t = int(nsteps[I_OBS]), int(nsteps[:I_OBS].sum()), float(np.asarray(s.times)[I_OBS - 1])
keys = jax.random.split(jax.random.key(5), R)
f = jax.jit(jax.vmap(lambda X, k: s.rproc_pf(X, th, k, cov, s.dt_array_extended, t, t_idx, nstep, s.accumvars, False)[0]))
for _ in range(2):
    f(X0, keys).block_until_ready()
walls = []
for _ in range(5):
    t0 = time.perf_counter(); f(X0, keys).block_until_ready(); walls.append(time.perf_counter() - t0)
wall = float(np.median(walls))
os.makedirs(out, exist_ok=True)
with jax.profiler.trace(out, create_perfetto_trace=True):
    for _ in range(CALLS):
        f(X0, keys).block_until_ready()

# --- parse the perfetto trace --------------------------------------------------------
path = sorted(glob.glob(os.path.join(out, "**", "*.trace.json.gz"), recursive=True))[-1]
tr = json.load(gzip.open(path))
events = tr["traceEvents"] if isinstance(tr, dict) else tr
pname = {e["pid"]: e["args"]["name"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"}
tname = {(e["pid"], e["tid"]): e["args"]["name"] for e in events if e.get("ph") == "M" and e.get("name") == "thread_name"}
gpu = [e for e in events if e.get("ph") == "X" and "GPU" in pname.get(e["pid"], "") ]
kern, copies = [], []
for e in gpu:
    thr = tname.get((e["pid"], e["tid"]), "")
    n = e["name"]
    if "Memcpy" in n or "memcpy" in n.lower() or "Memcpy" in thr:
        copies.append(e)
    elif "Stream" in thr or "stream" in thr.lower():
        kern.append(e)
span_lo = min(e["ts"] for e in kern); span_hi = max(e["ts"] + e["dur"] for e in kern)
busy = sum(e["dur"] for e in kern) / CALLS / 1e6
by = Counter(); cnt = Counter()
for e in kern:
    by[e["name"][:70]] += e["dur"] / CALLS / 1e6; cnt[e["name"][:70]] += 1
d2h = [e for e in copies if "DtoH" in e["name"] or "D2H" in e["name"] or "Device to Host" in e["name"]]
# gaps between consecutive kernels on the busiest stream within one call
streams = defaultdict(list)
for e in kern:
    streams[(e["pid"], e["tid"])].append(e)
main = max(streams.values(), key=len)
main.sort(key=lambda e: e["ts"])
gaps = [b["ts"] - (a["ts"] + a["dur"]) for a, b in zip(main, main[1:])]
gaps = [g for g in gaps if g < 5e4]  # drop the gaps between the traced calls
summary = {
    "variant": variant, "R": R, "J": J, "nstep": nstep,
    "wall_ms_per_step": round(wall / nstep * 1e3, 3),
    "gpu_busy_ms_per_step": round(busy / nstep * 1e3, 3),
    "kernels_per_step": round(len(kern) / CALLS / nstep, 1),
    "d2h_copies_per_step": round(len(d2h) / CALLS / nstep, 1),
    "gap_ms_per_step": round(sum(gaps) / 1e3 / CALLS / nstep, 3),
    "median_gap_us": round(float(np.median(gaps)), 1) if gaps else None,
    "top_kernels_ms_per_step": [(n, round(v / nstep * 1e3, 3), round(cnt[n] / CALLS / nstep, 2)) for n, v in by.most_common(8)],
    "trace": path,
}
print(json.dumps(summary))
