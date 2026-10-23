"""Compare each library's per-theta mean log-likelihood with pypomp's (same J, T, thetas).

For every (lib, variant) the latest row in the results file is used. Two comparisons:
  raw:       mean_lib - mean_pypomp, in combined Monte Carlo standard errors
             sqrt(sd_lib^2/n_lib + sd_pp^2/n_pp);
  corrected: the same for mean + sd^2/2, which removes the -sigma^2/2 bias that log L-hat
             carries when it is roughly normal, so resamplers with different variance
             (pfjax multinomial vs systematic) are compared on log L itself.
The corrected statistic's SE adds the uncertainty of sd^2/2 (normal approximation).

Collapsed filters (log-lik more than COLLAPSE below that theta's median) are counted and left
out of the mean and sd. At the poorly fitting timing starts (rho ~ +0.85) a large negative return
can push every particle's V to the 1e-12 floor; the next nonzero return then costs ~-1e8 or more.
This happens in float32 and float64 libraries alike, rarely and at random.

usage: python validate.py [results.jsonl] [--J 1000] [--T 3523] [--theta-set sun2024]
"""

import argparse
import json
import math
import os
import statistics

COLLAPSE = 1000.0

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("path", nargs="?", default=os.path.join(HERE, "results", "cpu", "results.jsonl"))
p.add_argument("--J", type=int, default=None)
p.add_argument("--T", type=int, default=None)
p.add_argument("--theta-set", default="starts")
a = p.parse_args()

rows = {}
with open(a.path) as f:
    for line in f:
        r = json.loads(line)
        if r.get("model") != "spx" or (a.J and r["J"] != a.J) or (a.T and r["T"] != a.T):
            continue
        if r.get("theta_set", "starts") != a.theta_set:
            continue
        rows[(r["lib"], r.get("variant"))] = r  # latest wins

ref = [r for (lib, _), r in rows.items() if lib == "pypomp"]
if not ref:
    raise SystemExit("no pypomp row")
ref = ref[-1]
mixed = {k: (r["J"], r["T"]) for k, r in rows.items() if (r["J"], r["T"]) != (ref["J"], ref["T"])}
if mixed:
    raise SystemExit(f"rows differ from pypomp's (J, T) = {(ref['J'], ref['T'])}: {mixed}; pass --J and --T")
print(f"theta_set={a.theta_set} J={ref['J']} T={ref['T']} reference: pypomp {ref['variant']} (nreps={ref['nreps']})\n")


def stats(r, i):
    lls = [x if x is not None else -math.inf for x in r["loglik"][i]]
    med = statistics.median(lls)
    good = [x for x in lls if x > med - COLLAPSE]
    n = len(good)
    m = statistics.fmean(good)
    sd = statistics.stdev(good) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    # var(s^2) ~ 2 s^4 / (n - 1) for normal data, so se(s^2/2) ~ s^2 / sqrt(2 (n - 1)).
    se_corr = math.sqrt(se**2 + (sd**2 / math.sqrt(2 * (n - 1))) ** 2) if n > 1 else 0.0
    return m, sd, se, m + sd**2 / 2, se_corr, len(lls) - n


def zscore(diff, se, se0):
    """diff in combined standard errors, or None when both are zero."""
    d = math.hypot(se, se0)
    return diff / d if d > 0 else None


def fmt(z):
    return "n/a" if z is None else f"{z:+.2f}"


print("| lib (variant) | theta | collapsed | mean +- se | sd | diff vs pypomp | z raw | z corrected |")
print("|---|---|---|---|---|---|---|---|")
worst = {}
for (lib, var), r in sorted(rows.items()):
    n_th = min(r["nthetas"], ref["nthetas"])
    for i in range(n_th):
        m, sd, se, mc, sec, nc = stats(r, i)
        m0, sd0, se0, mc0, sec0, _ = stats(ref, i)
        is_ref = lib == "pypomp" and var == ref["variant"]
        z, zc = (0.0, 0.0) if is_ref else (zscore(m - m0, se, se0), zscore(mc - mc0, sec, sec0))
        if z is not None:
            worst[(lib, var)] = max(worst.get((lib, var), 0.0), abs(z))
        print(f"| {lib} ({var}) | {i} | {nc}/{r['nreps']} | {m:.2f} +- {se:.2f} | {sd:.2f} | {m - m0:+.2f} | {fmt(z)} | {fmt(zc)} |")
print()
for k, v in worst.items():
    print(f"{k[0]:10s} {k[1]:40s} max |z raw| = {v:.2f}")
