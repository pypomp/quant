"""Per-kernel time, launches and bytes moved for one XLA interval, from xla_regs.py's trace and HLO dump.
usage: xla_bytes.py OUTDIR NSTEP"""
import glob, re, sys
from collections import defaultdict
import numpy as np
from jax._src.lib import _profile_data
out, nstep = sys.argv[1], int(sys.argv[2])
BYTES = {"f32": 4, "s32": 4, "u32": 4, "pred": 1, "f64": 8, "s64": 8, "u64": 8, "s8": 1, "u8": 1, "f16": 2, "bf16": 2}
def nbytes(shape_str):
    return sum(BYTES.get(t, 4) * int(np.prod([int(x) for x in d.split(",") if x] or [1]))
               for t, d in re.findall(r"(\w+)\[([\d,]*)\]", shape_str))
hlo = [p for p in glob.glob(f"{out}/dump/*after_optimizations.txt") if "loop_add_subtract_fusion" in open(p).read()][0]
comps, name = {}, None
for line in open(hlo):
    m = re.match(r"^(?:ENTRY )?%?([\w.\-]+) (\(.*\)) -> (.*) \{$", line)
    if m:
        name = m.group(1); comps[name] = {"params": m.group(2), "lines": []}; continue
    if line.startswith("}"):
        name = None; continue
    if name:
        comps[name]["lines"].append(line.strip())
fus = {}
for c in comps.values():
    for l in c["lines"]:
        m = re.match(r"%([\w.\-]+) = (.*?) fusion\(.*calls=%?([\w.\-]+)", l)
        if m:
            callee = comps[m.group(3)]
            fus[m.group(1)] = (nbytes(callee["params"]), nbytes(m.group(2)))
pd = _profile_data.ProfileData.from_file(sorted(glob.glob(f"{out}/trace/**/*.xplane.pb", recursive=True))[-1])
agg = defaultdict(lambda: [0.0, 0])
for plane in pd.planes:
    if "GPU" not in plane.name:
        continue
    for line in plane.lines:
        for ev in line.events:
            a = agg[ev.name]; a[0] += ev.duration_ns / 1e6; a[1] += 1
tot_ms = sum(a[0] for a in agg.values()); tot_b = 0.0; n = sum(a[1] for a in agg.values())
rows = []
for k, (ms, cnt) in agg.items():
    rb, wb = fus.get(k, (0, 0))
    b = (rb + wb) * cnt; tot_b += b
    rows.append((ms, cnt, rb, wb, b, k))
print(f"interval {tot_ms:.1f} ms GPU, {n} kernels ({n / nstep:.0f} per step), {tot_b / 1e9:.2f} GB moved "
      f"({tot_b / 1e9 / nstep:.2f} GB per step), {tot_b / 1e9 / (tot_ms / 1e3):.0f} GB/s average")
for ms, cnt, rb, wb, b, k in sorted(rows, reverse=True)[:14]:
    bw = b / 1e9 / (ms / 1e3) if ms else 0
    print(f"  {ms:5.2f} ms x{cnt:<3d} reads {rb / 1e6:6.0f} MB writes {wb / 1e6:5.0f} MB per launch, {bw:4.0f} GB/s  {k[:55]}")
small = [r for r in rows if r[0] / max(r[1], 1) < 0.2]
print(f"kernels under 0.2 ms each: {sum(r[1] for r in small)} launches, {sum(r[0] for r in small):.1f} ms, {sum(r[4] for r in small) / 1e9:.2f} GB")
