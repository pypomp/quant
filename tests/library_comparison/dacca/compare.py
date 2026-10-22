"""Compare each library's per-theta mean loglik with pypomp's (latest matching rows of a results file).

usage: python compare.py RESULTS.jsonl [--J 1000] [--T 600] [--theta-set starts|published]
Prints mean +/- se per (lib, theta) and z = (mean - pypomp) / sqrt(se^2 + se_pypomp^2).
"""

import argparse
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--J", type=int, default=None)
ap.add_argument("--T", type=int, default=None)
ap.add_argument("--theta-set", default="starts")
a = ap.parse_args()

rows = {}
with open(a.path) as f:
    for line in f:
        r = json.loads(line)
        if "lib" not in r or (a.J and r["J"] != a.J) or (a.T and r["T"] != a.T) \
                or r.get("theta_set", "starts") != a.theta_set:
            continue
        rows[r.get("variant", r["lib"]) if r["lib"] in ("pypfilt", "pypomp") else r["lib"]] = r

ref = next((r for k, r in rows.items() if k.startswith("functional.pfilter")), None)
if ref is None:
    raise SystemExit("no pypomp row")


def se(r, i):
    return r["loglik_sd_by_theta"][i] / math.sqrt(r["nreps"])


print(f"J={ref['J']} T={ref['T']} nreps={ref['nreps']}")
for name, r in rows.items():
    cells = []
    for i, m in enumerate(r["loglik_by_theta"]):
        s = se(r, i)
        z = (m - ref["loglik_by_theta"][i]) / math.sqrt(s * s + se(ref, i) ** 2)
        cells.append(f"{m:10.2f} +/- {s:5.2f} (z={z:+.2f})")
    print(f"{r['lib']:9s} {name:32s} " + " | ".join(cells))
