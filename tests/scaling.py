"""Shared driver for the GPU scaling tests (dacca/scaling, measles/scaling).

Each grid point runs in its own subprocess so XLA's allocator starts fresh and
peak VRAM is per-configuration. The driver must not initialise JAX (importing
the model does) until the workers are done, or it holds the GPU and they fail.
mif is run twice with the same M: the first call pays compilation, the second
gives the steady-state per-iteration cost.
"""

import contextlib
import json
import os
import subprocess
import sys
import threading
import time

import pandas as pd

COLUMNS = [
    "scaling_type",
    "J",
    "chains",
    "M",
    "reps",
    "total_particles_mif",
    "total_particles_pfilter",
    "mif_cold_seconds",
    "mif_warm_seconds",
    "mif_per_iter_seconds",
    "pfilter_cold_seconds",
    "pfilter_warm_seconds",
    "peak_vram_mif_mb",
    "peak_vram_pfilter_mb",
    "peak_vram_mb",
]


def _device_stats():
    import jax

    with contextlib.suppress(Exception):
        stats = jax.devices()[0].memory_stats()
        if isinstance(stats, dict) and "bytes_in_use" in stats:
            return stats
    return None


class VramPoller:
    def __init__(self, interval_sec=0.01):
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = None
        self.peak = 0

    def _sample(self):
        stats = _device_stats()
        if stats is not None:
            self.peak = max(self.peak, stats["bytes_in_use"])
        return stats

    def _poll(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_sec)

    def __enter__(self):
        if self._sample() is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._sample()

    @property
    def peak_mb(self):
        return self.peak / 1024**2 if self._thread is not None else None


def _timed(fn):
    t0 = time.time()
    fn()
    return time.time() - t0


def run_one(spec, build):
    import jax
    import model

    seed = model.MAIN_SEED

    J, chains, M, reps = spec["J"], spec["chains"], spec["M"], spec["reps"]
    if spec["index"] == 0:
        print("Detected devices:", jax.devices())
    key = jax.random.fold_in(jax.random.key(seed), spec["index"])
    k_build, k_mif1, k_mif2, k_pf1, k_pf2 = jax.random.split(key, 5)

    obj, theta, rw_sd = build(chains, k_build)

    with VramPoller() as vp_mif:
        mif_cold = _timed(
            lambda: obj.mif(theta=theta, rw_sd=rw_sd, M=M, J=J, key=k_mif1)
        )
        mif_warm = _timed(
            lambda: obj.mif(theta=theta, rw_sd=rw_sd, M=M, J=J, key=k_mif2)
        )
    with VramPoller() as vp_pf:
        pf_cold = _timed(lambda: obj.pfilter(J=J, reps=reps, key=k_pf1))
        pf_warm = _timed(lambda: obj.pfilter(J=J, reps=reps, key=k_pf2))

    stats = _device_stats()
    peak = stats.get("peak_bytes_in_use") if stats else None

    print(
        f"[{spec['scaling_type']}] J={J} chains={chains}: "
        f"mif cold {mif_cold:.2f} s, warm {mif_warm:.2f} s ({mif_warm / M:.4f} s/iter); "
        f"pfilter cold {pf_cold:.2f} s, warm {pf_warm:.2f} s"
    )
    return {
        "scaling_type": spec["scaling_type"],
        "J": J,
        "chains": chains,
        "M": M,
        "reps": reps,
        "total_particles_mif": J * chains,
        "total_particles_pfilter": J * chains * reps,
        "mif_cold_seconds": mif_cold,
        "mif_warm_seconds": mif_warm,
        "mif_per_iter_seconds": mif_warm / M,
        "pfilter_cold_seconds": pf_cold,
        "pfilter_warm_seconds": pf_warm,
        "peak_vram_mif_mb": vp_mif.peak_mb,
        "peak_vram_pfilter_mb": vp_pf.peak_mb,
        "peak_vram_mb": peak / 1024**2 if peak is not None else None,
    }


def main(
    *,
    model_name,
    build,
    run_level,
    particle_j_grid,
    particle_fixed_chains,
    chain_grid,
    chain_fixed_j,
    M,
    reps,
    extra_config=None,
):
    if "--worker" in sys.argv:
        spec = json.loads(sys.argv[sys.argv.index("--worker") + 1])
        print(f"RESULT_JSON {json.dumps(run_one(spec, build))}")
        sys.exit(0)

    print(f"Running {model_name} scaling benchmark at level {run_level}")
    specs = [
        {"scaling_type": "particles", "J": J, "chains": particle_fixed_chains}
        for J in particle_j_grid
    ] + [
        {"scaling_type": "chains", "J": chain_fixed_j, "chains": c} for c in chain_grid
    ]
    for i, spec in enumerate(specs):
        spec.update(index=i, M=M, reps=reps)

    script = os.path.abspath(sys.argv[0])
    results = []
    for spec in specs:
        proc = subprocess.run(
            [sys.executable, script, "--worker", json.dumps(spec)],
            capture_output=True,
            text=True,
            check=False,
        )
        result = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON "):
                result = json.loads(line[len("RESULT_JSON ") :])
            else:
                print(line)
        sys.stdout.flush()
        if proc.returncode != 0 or result is None:
            tail = "\n".join(proc.stderr.splitlines()[-20:])
            print(f"  [ERROR] spec {spec['index']} exited {proc.returncode}:\n{tail}")
            result = {
                **{c: None for c in COLUMNS},
                "scaling_type": spec["scaling_type"],
                "J": spec["J"],
                "chains": spec["chains"],
                "M": M,
                "reps": reps,
                "total_particles_mif": spec["J"] * spec["chains"],
                "total_particles_pfilter": spec["J"] * spec["chains"] * reps,
                "error": f"exit {proc.returncode}: {tail}",
            }
        results.append(result)

    import jax
    import model
    from utils import run_metadata

    out_dir = os.path.join("results", jax.devices()[0].platform)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "scaling.csv"), index=False)

    record = run_metadata(
        {
            "kind": "scaling",
            "model": model_name,
            "job": "gpu",
            "RUN_LEVEL": run_level,
            "MAIN_SEED": model.MAIN_SEED,
            "PARTICLE_J_GRID": particle_j_grid,
            "PARTICLE_FIXED_CHAINS": particle_fixed_chains,
            "CHAIN_GRID": chain_grid,
            "CHAIN_FIXED_J": chain_fixed_j,
            "NFITR": M,
            "NREPS_EVAL": reps,
            **(extra_config or {}),
        }
    )
    record["scaling"] = df.to_dict(orient="records")
    with open(os.path.join(out_dir, "latest.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)
        f.write("\n")

    print(f"\n{df.drop(columns='error', errors='ignore').to_string(index=False)}")
    print(f"wrote {out_dir}/ (scaling.csv, latest.json)")


def cost_table(df, sort_by):
    """Per-iteration cost per particle, the quantity that shows saturation."""
    df = df.sort_values(sort_by).reset_index(drop=True)
    per_part = df["mif_per_iter_seconds"] / df["total_particles_mif"] * 1e9
    marginal = (
        df["mif_per_iter_seconds"].diff() / df["total_particles_mif"].diff() * 1e9
    )
    pf_per_part = df["pfilter_warm_seconds"] / df["total_particles_pfilter"] * 1e9
    return pd.DataFrame(
        {
            "J": df["J"],
            "Chains": df["chains"],
            "J × chains": df["total_particles_mif"].map("{:,}".format),
            "IF2 compile + M iters (s)": df["mif_cold_seconds"].round(2),
            "IF2 per iter (s)": df["mif_per_iter_seconds"].round(4),
            "IF2 ns / particle / iter": per_part.round(1),
            "IF2 marginal ns / particle": marginal.round(1).fillna("—"),
            "pfilter warm (s)": df["pfilter_warm_seconds"].round(2),
            "pfilter ns / particle": pf_per_part.round(2),
            "Peak VRAM (MB)": df["peak_vram_mb"].round(0),
        }
    )
