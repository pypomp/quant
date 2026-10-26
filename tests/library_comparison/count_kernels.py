"""Count GPU kernels per loop iteration in compiled pypomp pfilter HLO (He10 and the bench SDE).

For each while loop in the optimized HLO, counts the kernel-launching instructions
(fusions, custom calls, copies, sorts, scatters, ...) in its body computation.
usage: python count_kernels.py
"""

import re
import sys

import jax
import jax.numpy as jnp

import pypomp as pp
import pypomp.functional as F


THETA = {
    "R0": 56.8, "sigma": 28.9, "gamma": 30.4, "iota": 2.9, "rho": 0.488,
    "sigmaSE": 0.0878, "psi": 0.116, "cohort": 0.557, "amplitude": 0.554,
    "S_0": 2.97e-02, "E_0": 5.17e-05, "I_0": 5.14e-05, "R_0": 9.70e-01,
}
KERNEL_OPS = ("fusion", "custom-call", "copy-start", "copy(", "sort(", "scatter(", "gather(",
              "reduce(", "cumsum", "rng-bit-generator", "dynamic-update-slice(", "convolution(")


def computations(hlo):
    comps, name, body = {}, None, []
    for line in hlo.splitlines():
        m = re.match(r"^(?:ENTRY )?%?([\w.\-]+) .*\{$", line)
        if m and not line.startswith(" "):
            name, body = m.group(1), []
            continue
        if line.startswith("}") and name:
            comps[name] = body
            name = None
            continue
        if name:
            body.append(line)
    return comps


def report(label, fn, *args):
    hlo = jax.jit(fn).lower(*args).compile().as_text()
    comps = computations(hlo)
    print(f"\n== {label}: {len(comps)} computations")
    for cname, body in comps.items():
        for line in body:
            m = re.search(r"while\(.*body=%?([\w.\-]+)", line)
            if m:
                b = m.group(1)
                kern = [l for l in comps.get(b, []) if any(k in l for k in KERNEL_OPS)]
                inner = [re.search(r"body=%?([\w.\-]+)", l).group(1) for l in comps.get(b, []) if "while(" in l]
                trip = re.search(r'known_trip_count[^0-9]*(\d+)', line)
                print(f"  in {cname[:40]}: while body {b}: {len(comps.get(b, []))} instrs, "
                      f"{len(kern)} kernel-like, trip={trip.group(1) if trip else '?'}, inner={inner}")
                if len(kern) < 60:
                    for l in kern:
                        print("      ", l.strip()[:110])


model = pp.models.UKMeasles.pomp(unit="London", theta=pp.PompParameters(THETA), clean=True, model="001b")
struct = model.to_struct()
th = model.theta.to_jax_array(model.canonical_param_names)
keys = jax.random.key(0)[None, None]
report("He10 J=1000", lambda th, k: F.pfilter(struct, th, 1000, k)["logLik"], th, keys)
