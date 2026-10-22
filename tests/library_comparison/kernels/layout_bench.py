"""Pomp.pfilter on the paper's PF workloads: replicates in parallel vs one at a time.

Needs the pypomp branch pfilter-vmap-reps on PYTHONPATH. reps_batch_size=1 reproduces
the old layout (vmap over parameter sets, replicates in sequence); None runs every
replicate in parallel. Parameters are each model's 36 fixed starting vectors.

usage: python layout_bench.py [dacca|measles|spx ...]
"""

import json
import os
import sys
import time

import jax

QUANT_TESTS = os.path.expanduser("~/OneDrive/Documents/UMich/Research/pypomp/quant/tests")
sys.path.insert(0, QUANT_TESTS)


def load(name):
    sys.path.insert(0, os.path.join(QUANT_TESTS, name))
    import importlib

    model = importlib.import_module("model")
    sys.path.pop(0)
    sys.modules.pop("model", None)
    if name == "dacca":
        return model.dacca(), model.fixed_starts(36), 5000, 36
    if name == "measles":
        starts = model.fixed_starts(36)
        return model.measles("London", starts, model="001b"), starts, 5000, 36
    if name == "spx":
        starts = model.sample_starts(36, jax.random.key(model.MAIN_SEED))
        return model.spx(), starts, 1000, 24
    raise ValueError(name)


def main():
    import pypomp

    print("pypomp from", pypomp.__file__, flush=True)
    out = os.environ.get("PFB_RESULTS", "results_layout.jsonl")
    for name in sys.argv[1:] or ["spx", "dacca", "measles"]:
        obj, starts, J, reps = load(name)
        for batch in (1, None):
            times = []
            for call in range(2):
                t0 = time.perf_counter()
                obj.pfilter(J=J, reps=reps, theta=starts, key=jax.random.key(call), reps_batch_size=batch)
                jax.block_until_ready(obj.results_history[-1].logLiks.values)
                times.append(time.perf_counter() - t0)
            ll = obj.results_history[-1].logLiks.values
            row = dict(model=name, layout="replicates sequential (old)" if batch == 1 else "replicates parallel (new)",
                       J=J, nthetas=36, reps=reps, cold_s=round(times[0], 3), warm_s=round(times[1], 3),
                       loglik_mean=float(ll.mean()), load=os.getloadavg()[0], device=jax.devices()[0].platform)
            print(json.dumps(row), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
