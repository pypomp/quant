"""Loglik agreement table from a results.jsonl: per library and theta, mean +- se and z against references.

    python compare.py results/cpu/results.jsonl [--tag validation] [--ref pypomp] [--ref2 pfjax]

Uses the latest row per library among rows with the given tag. z = (mean - ref) / sqrt(se^2 + se_ref^2).
"""

import argparse
import json
import math


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--tag", default=None)
    p.add_argument("--ref", default="pypomp")
    p.add_argument("--ref2", default="pfjax")
    a = p.parse_args()
    rows = {}
    with open(a.path) as f:
        for line in f:
            r = json.loads(line)
            if a.tag is not None and r.get("tag") != a.tag:
                continue
            rows[r["lib"]] = r
    if not rows:
        raise SystemExit("no rows")

    def se(r, i):
        sd = r["loglik_sd_by_theta"][i]
        return float("nan") if sd is None else sd / math.sqrt(r["nreps"])

    refs = [x for x in (a.ref, a.ref2) if x in rows]
    head = f"{'lib':24s} {'J':>5s} {'reps':>4s} {'T':>4s} theta  {'mean':>10s} {'se':>7s}" + "".join(
        f" {'z vs ' + x:>14s}" for x in refs)
    print(head)
    for lib, r in sorted(rows.items(), key=lambda kv: (kv[0] not in refs, kv[0])):
        for i, m in enumerate(r["loglik_by_theta"]):
            line = f"{lib:24s} {r['J']:5d} {r['nreps']:4d} {r['T']:4d} {i:5d}  {m:10.2f} {se(r, i):7.2f}"
            for x in refs:
                q = rows[x]
                if x == lib or q["T"] != r["T"] or i >= len(q["loglik_by_theta"]):
                    line += f" {'':>14s}"
                    continue
                z = (m - q["loglik_by_theta"][i]) / math.sqrt(se(r, i) ** 2 + se(q, i) ** 2)
                line += f" {z:14.2f}"
            print(line)


if __name__ == "__main__":
    main()
