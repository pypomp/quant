"""One He10 London interval (1,296 x 5,000) on pypomp's XLA rproc: time, top kernels with registers and
launch shape, and the compiled kernels' cubins. usage: xla_regs.py OUTDIR   (XLA_FLAGS from the environment)"""
import glob, os, re, shutil, struct, subprocess, sys, time
import jax, jax.numpy as jnp, numpy as np
import pypomp as pp
from jax._src.lib import _profile_data
from jax.experimental import serialize_executable
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
R, J, I_OBS = int(os.environ.get("R", 1296)), int(os.environ.get("J", 5000)), 3
THETA = {"R0": 56.8, "sigma": 28.9, "gamma": 30.4, "iota": 2.9, "rho": 0.488, "sigmaSE": 0.0878,
         "psi": 0.116, "cohort": 0.557, "amplitude": 0.554, "S_0": 2.97e-02, "E_0": 5.17e-05,
         "I_0": 5.14e-05, "R_0": 9.70e-01}
model = pp.models.UKMeasles.pomp(unit="London", theta=pp.PompParameters(THETA), clean=True, model="001b")
model.set_rproc_backend("xla")
s = model.to_struct()
th = model.theta.to_jax_array(model.canonical_param_names)[0]
cov = s.covars_extended
X0 = jax.vmap(lambda k: s.rinit_pf(th, jax.random.split(k, J), None if cov is None else cov[0], s.t0, False))(
    jax.random.split(jax.random.key(9), R))
nsteps = np.asarray(s.nstep_array)
nstep, t_idx, t = int(nsteps[I_OBS]), int(nsteps[:I_OBS].sum()), float(np.asarray(s.times)[I_OBS - 1])
keys = jax.random.split(jax.random.key(5), R)
f = jax.jit(jax.vmap(lambda X, k: s.rproc_pf(X, th, k, cov, s.dt_array_extended, t, t_idx, nstep, s.accumvars, False)[0]))
compiled = f.lower(X0, keys).compile()
res = compiled(X0, keys).block_until_ready()
walls = []
for _ in range(7):
    t0 = time.perf_counter(); compiled(X0, keys).block_until_ready(); walls.append(time.perf_counter() - t0)
d = f"{out}/trace"; shutil.rmtree(d, ignore_errors=True)
with jax.profiler.trace(d):
    compiled(X0, keys).block_until_ready()
pd = _profile_data.ProfileData.from_file(sorted(glob.glob(f"{d}/**/*.xplane.pb", recursive=True))[-1])
agg = {}
for plane in pd.planes:
    if "GPU" not in plane.name:
        continue
    for line in plane.lines:
        for ev in line.events:
            det = next((str(v) for k, v in ev.stats if k == "kernel_details"), "")
            a = agg.setdefault(ev.name, [0.0, 0, det]); a[0] += ev.duration_ns / 1e6; a[1] += 1
tot = sum(a[0] for a in agg.values())
print(f"nstep {nstep}, {1e3 * np.median(walls):.1f} ms per interval, {sum(a[1] for a in agg.values())} kernels, out_mean {float(jnp.nanmean(res)):.1f}")
for name, (ms, n, det) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:6]:
    g = re.search(r"grid:(\d+),(\d+),(\d+) block:(\d+),(\d+),(\d+)", det)
    thr = np.prod([int(x) for x in g.groups()]) if g else 0
    regs = re.search(r"regs:(\d+)", det); occ = re.search(r"occ_pct:([\d.]+)", det)
    print(f"  {ms:6.1f} ms {100 * ms / tot:4.1f}% x{n:<3d} regs {regs.group(1) if regs else '?':>3} occ {float(occ.group(1)) if occ else 0:5.1f}%"
          f"  elems/thread {R * J / thr if thr else 0:4.1f}  {name[:60]}")
blob = serialize_executable.serialize(compiled)[0]
n = 0
for m in re.finditer(rb"\x7fELF", blob):
    o = m.start()
    if blob[o + 4] != 2:
        continue
    shoff, = struct.unpack_from("<Q", blob, o + 0x28); shentsize, shnum = struct.unpack_from("<HH", blob, o + 0x3A)
    size = shoff + shentsize * shnum
    if 0 < size < len(blob) - o:
        open(f"{out}/k{n}.cubin", "wb").write(blob[o:o + size]); n += 1
ptx = re.findall(rb"\.version [\d.]+.*?(?=\x00)", blob, re.S)
for i, p in enumerate(ptx):
    open(f"{out}/k{i}.ptx", "wb").write(p)
print(f"extracted {n} cubins, {len(ptx)} ptx from a {len(blob)}-byte executable")
