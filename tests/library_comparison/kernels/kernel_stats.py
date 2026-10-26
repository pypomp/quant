"""Per-kernel time, registers and occupancy from the profile_he10.py traces in /tmp/prof_he10/{stock,unroll,cute}."""
import glob, re
from collections import defaultdict
from jax._src.lib import _profile_data
for v in ("stock", "unroll", "cute"):
    path = sorted(glob.glob(f"/tmp/prof_he10/{v}/**/*.xplane.pb", recursive=True))[-1]
    pd = _profile_data.ProfileData.from_file(path)
    agg = defaultdict(lambda: [0.0, 0, ""])
    for plane in pd.planes:
        if "GPU" not in plane.name:
            continue
        for line in plane.lines:
            for ev in line.events:
                details = ""
                for name, val in ev.stats:
                    if name in ("kernel_details",):
                        details = str(val)
                a = agg[ev.name[:60]]
                a[0] += ev.duration_ns / 3e6  # ms per interval (3 traced calls)
                a[1] += 1
                if details:
                    a[2] = details
    print(f"== {v}")
    for n, (ms, c, d) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:4]:
        print(f"   {ms:7.2f} ms/interval  {n}")
        print(f"        {d[:240]}")
