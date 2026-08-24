"""SPX: scaling of MIF and pfilter over particle counts and chain counts on GPU.

This benchmark evaluates execution speed and VRAM memory scaling on GPU:
1. Scaling particles: varying J with a fixed number of chains.
2. Scaling chains: varying chains with a fixed number of particles.

Jobs:
- gpu: Scaling benchmark recording throughput and dynamic VRAM usage per configuration.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "SPX: scaling over particles and chains on GPU (runtime & memory)"
# tags: [scaling, spx, gpu]
# jobs:
#   gpu:
#     sbatch_args:
#       job-name: "spx scaling (gpu)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 16GB
#       time: "00:40:00"
#       output: "results/gpu/logs/slurm-%j.out"
#     env:
#       XLA_PYTHON_CLIENT_PREALLOCATE: "false"
#     run_levels:
#       1:
#         sbatch_args: { time: "00:02:00" }
#       2:
#         sbatch_args: { time: "00:05:00" }
#       3:
#         sbatch_args: { time: "00:25:00" }
#       4:
#         sbatch_args: { time: "00:40:00" }
# --- END SLURM CONFIG ---

import json
import os
import subprocess
import sys
import threading
import time
from typing import cast

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if tests_dir not in sys.path:
    sys.path.append(tests_dir)
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if model_dir not in sys.path:
    sys.path.append(model_dir)

import pandas as pd


class VramPoller:
    """Background thread that periodically samples live VRAM bytes_in_use."""

    def __init__(self, interval_sec: float = 0.01):
        import jax

        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self.peak_bytes = 0
        self._device = None
        self._thread = None
        try:
            d = jax.devices()[0]
            if hasattr(d, "memory_stats"):
                stats = d.memory_stats()
                if isinstance(stats, dict) and "bytes_in_use" in stats:
                    self._device = d
                    self.peak_bytes = stats.get("bytes_in_use", 0)
        except Exception:
            pass

    def _poll(self):
        while not self._stop_event.is_set():
            try:
                if self._device is not None and hasattr(self._device, "memory_stats"):
                    stats = self._device.memory_stats()
                    if isinstance(stats, dict) and "bytes_in_use" in stats:
                        b = stats["bytes_in_use"]
                        if b > self.peak_bytes:
                            self.peak_bytes = b
            except Exception:
                pass
            self._stop_event.wait(self.interval_sec)

    def start(self):
        if self._device is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> float | None:
        if self._device is None:
            return None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        try:
            if hasattr(self._device, "memory_stats"):
                stats = self._device.memory_stats()
                if isinstance(stats, dict) and "bytes_in_use" in stats:
                    b = stats["bytes_in_use"]
                    if b > self.peak_bytes:
                        self.peak_bytes = b
        except Exception:
            pass
        return self.peak_bytes / (1024**2)


def get_vram_bytes_in_use() -> float | None:
    """Query current active VRAM allocated by JAX buffers on the primary device in MB."""
    import jax

    try:
        d = jax.devices()[0]
        if hasattr(d, "memory_stats"):
            stats = d.memory_stats()
            if isinstance(stats, dict) and "bytes_in_use" in stats:
                return stats["bytes_in_use"] / (1024**2)
    except Exception:
        pass
    return None


def get_peak_vram_mb() -> float | None:
    """Query peak VRAM allocated by JAX buffers on the primary device in MB."""
    import jax

    try:
        d = jax.devices()[0]
        if hasattr(d, "memory_stats"):
            stats = d.memory_stats()
            if isinstance(stats, dict) and "peak_bytes_in_use" in stats:
                return stats["peak_bytes_in_use"] / (1024**2)
    except Exception:
        pass
    return None


def run_one(spec: dict) -> dict:
    import jax
    import numpy as np
    import model

    scaling_type = spec["scaling_type"]
    J = spec["J"]
    chains = spec["chains"]
    M = spec["M"]
    reps = spec["reps"]
    idx = spec.get("index", 0)

    # Per-spec determinism derived from position
    if idx == 0:
        print("Detected devices:", jax.devices())
    np.random.seed(model.MAIN_SEED + idx)
    config_key = jax.random.fold_in(jax.random.key(model.MAIN_SEED), idx)
    subkey_starts, subkey_mif, subkey_pf1, subkey_pf2 = jax.random.split(config_key, 4)

    dim_label = "Particle Scaling" if scaling_type == "particles" else "Chain Scaling"
    if scaling_type == "particles":
        print(f"\n[{dim_label}] Running J={J}...")
    else:
        print(f"\n[{dim_label}] Running chains={chains}...")

    starts = model.sample_starts(chains, key=subkey_starts)
    spx_obj = model.spx()

    baseline_vram_mb = get_vram_bytes_in_use()

    # MIF
    poller_mif = VramPoller().start()
    t0 = time.time()
    spx_obj.mif(theta=starts, rw_sd=model.RW_SD, M=M, J=J, key=subkey_mif)
    mif_time = time.time() - t0
    peak_vram_mif_mb = poller_mif.stop()

    if scaling_type == "particles":
        print(
            f"  mif ({M} iters, J={J}): {mif_time:.3f} s ({mif_time / M:.4f} s/iter)"
        )
    else:
        print(
            f"  mif ({M} iters, chains={chains}): {mif_time:.3f} s ({mif_time / M:.4f} s/iter)"
        )

    # Cold pfilter
    poller_pf = VramPoller().start()
    t0 = time.time()
    spx_obj.pfilter(J=J, reps=reps, key=subkey_pf1)
    pf_cold_time = time.time() - t0
    peak_vram_pfilter_mb = poller_pf.stop()

    if scaling_type == "particles":
        print(f"  pfilter_cold (J={J}, reps={reps}): {pf_cold_time:.3f} s")
    else:
        print(f"  pfilter_cold (chains={chains}, reps={reps}): {pf_cold_time:.3f} s")

    # Warm pfilter
    t0 = time.time()
    spx_obj.pfilter(J=J, reps=reps, key=subkey_pf2)
    pf_warm_time = time.time() - t0

    if scaling_type == "particles":
        print(f"  pfilter_warm (J={J}, reps={reps}): {pf_warm_time:.3f} s")
    else:
        print(f"  pfilter_warm (chains={chains}, reps={reps}): {pf_warm_time:.3f} s")

    peak_vram_mb = get_peak_vram_mb()

    if peak_vram_mb is not None:
        sub = []
        if baseline_vram_mb is not None:
            sub.append(f"baseline: {baseline_vram_mb:.1f} MB")
        if peak_vram_mif_mb is not None:
            sub.append(f"mif: {peak_vram_mif_mb:.1f} MB")
        if peak_vram_pfilter_mb is not None:
            sub.append(f"pf: {peak_vram_pfilter_mb:.1f} MB")
        if sub:
            print(f"  VRAM peak: {peak_vram_mb:.1f} MB ({', '.join(sub)})")
        else:
            print(f"  VRAM peak: {peak_vram_mb:.1f} MB")

    entry = {
        "scaling_type": scaling_type,
        "J": J,
        "chains": chains,
        "M": M,
        "reps": reps,
        "total_particles_mif": J * chains,
        "total_particles_pfilter": J * chains * reps,
        "mif_time_seconds": mif_time,
        "mif_per_iter_seconds": mif_time / M,
        "pfilter_cold_seconds": pf_cold_time,
        "pfilter_warm_seconds": pf_warm_time,
        "baseline_vram_mb": baseline_vram_mb,
        "peak_vram_mif_mb": peak_vram_mif_mb,
        "peak_vram_pfilter_mb": peak_vram_pfilter_mb,
        "peak_vram_mb": peak_vram_mb,
    }

    return entry


# Worker execution guard
if "--worker" in sys.argv:
    worker_idx = sys.argv.index("--worker")
    if worker_idx + 1 >= len(sys.argv):
        print("Error: --worker requires a JSON spec argument", file=sys.stderr)
        sys.exit(1)
    spec_json = sys.argv[worker_idx + 1]
    spec = json.loads(spec_json)
    result = run_one(spec)
    print(f"RESULT_JSON {json.dumps(result)}")
    sys.exit(0)

# =========================================================================
# Driver Mode
# =========================================================================
RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
print(f"Running SPX scaling benchmark at level {RUN_LEVEL}")

# 1. Particle scaling grid (varying J, fixed chains)
PARTICLE_J_GRID = (
    [100, 200],  # Level 1: smoke test
    [500, 1000],  # Level 2: quick check
    [1000, 2500, 5000],  # Level 3: medium
    [5000, 10000, 15000, 20000, 30000, 40000],  # Level 4: user target
)[RUN_LEVEL - 1]

PARTICLE_FIXED_CHAINS = (2, 10, 30, 120)[RUN_LEVEL - 1]

# 2. Chain scaling grid (varying chains, fixed J)
CHAIN_GRID = (
    [2, 4],  # Level 1: smoke test
    [10, 20],  # Level 2: quick check
    [50, 100, 200],  # Level 3: medium
    [250, 500, 750, 1000, 1500, 2000],  # Level 4: user target
)[RUN_LEVEL - 1]

CHAIN_FIXED_J = (100, 500, 1000, 1000)[RUN_LEVEL - 1]

# Shared parameters for both experiments
NFITR = (2, 5, 20, 50)[RUN_LEVEL - 1]
NREPS_EVAL = (2, 5, 10, 36)[RUN_LEVEL - 1]

# Build full list of specs up front
specs = []
idx = 0
for J in PARTICLE_J_GRID:
    specs.append({
        "index": idx,
        "scaling_type": "particles",
        "J": J,
        "chains": PARTICLE_FIXED_CHAINS,
        "M": NFITR,
        "reps": NREPS_EVAL,
    })
    idx += 1

for n_chains in CHAIN_GRID:
    specs.append({
        "index": idx,
        "scaling_type": "chains",
        "J": CHAIN_FIXED_J,
        "chains": n_chains,
        "M": NFITR,
        "reps": NREPS_EVAL,
    })
    idx += 1

results = []
current_scaling_type = None

for spec in specs:
    if spec["scaling_type"] != current_scaling_type:
        current_scaling_type = spec["scaling_type"]
        if current_scaling_type == "particles":
            print(
                f"\n--- Part 1: Particle Scaling (chains={PARTICLE_FIXED_CHAINS}, M={NFITR}, reps={NREPS_EVAL}) ---"
            )
        else:
            print(
                f"\n--- Part 2: Chain Scaling (J={CHAIN_FIXED_J}, M={NFITR}, reps={NREPS_EVAL}) ---"
            )

    # Isolated subprocess per grid point with fresh XLA allocator
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ,
    )

    stdout_lines = []
    result_json = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            result_json = line[len("RESULT_JSON "):].strip()
        else:
            stdout_lines.append(line)

    if stdout_lines:
        print("\n".join(stdout_lines))

    if proc.returncode != 0 or result_json is None:
        stderr_tail = (
            "\n".join(proc.stderr.splitlines()[-20:])
            if proc.stderr
            else "(no stderr)"
        )
        print(
            f"  [ERROR] Worker failed for spec index {spec.get('index')} (exit code {proc.returncode}):\n{stderr_tail}"
        )
        entry = {
            "scaling_type": spec["scaling_type"],
            "J": spec["J"],
            "chains": spec["chains"],
            "M": spec["M"],
            "reps": spec["reps"],
            "total_particles_mif": spec["J"] * spec["chains"],
            "total_particles_pfilter": spec["J"] * spec["chains"] * spec["reps"],
            "mif_time_seconds": None,
            "mif_per_iter_seconds": None,
            "pfilter_cold_seconds": None,
            "pfilter_warm_seconds": None,
            "baseline_vram_mb": None,
            "peak_vram_mif_mb": None,
            "peak_vram_pfilter_mb": None,
            "peak_vram_mb": None,
            "error": f"Exit code {proc.returncode}: {stderr_tail}",
        }
        results.append(entry)
    else:
        try:
            entry = json.loads(result_json)
        except Exception as e:
            print(f"  [ERROR] Failed to parse worker RESULT_JSON: {e}")
            entry = {
                "scaling_type": spec["scaling_type"],
                "J": spec["J"],
                "chains": spec["chains"],
                "M": spec["M"],
                "reps": spec["reps"],
                "total_particles_mif": spec["J"] * spec["chains"],
                "total_particles_pfilter": spec["J"] * spec["chains"] * spec["reps"],
                "mif_time_seconds": None,
                "mif_per_iter_seconds": None,
                "pfilter_cold_seconds": None,
                "pfilter_warm_seconds": None,
                "baseline_vram_mb": None,
                "peak_vram_mif_mb": None,
                "peak_vram_pfilter_mb": None,
                "peak_vram_mb": None,
                "error": f"JSON parse error: {e}",
            }
        results.append(entry)

# =========================================================================
# Output and Provenance
# =========================================================================
import model
from utils import run_metadata
import jax

platform_name = jax.devices()[0].platform
out_dir = os.path.join("results", platform_name)
os.makedirs(out_dir, exist_ok=True)

df_results = pd.DataFrame(results)

df_particles = cast(
    pd.DataFrame, df_results[df_results["scaling_type"] == "particles"].copy()
)
df_chains = cast(
    pd.DataFrame, df_results[df_results["scaling_type"] == "chains"].copy()
)

df_particles.to_csv(os.path.join(out_dir, "particle_scaling.csv"), index=False)
df_chains.to_csv(os.path.join(out_dir, "chain_scaling.csv"), index=False)
df_results.to_csv(os.path.join(out_dir, "scaling.csv"), index=False)

record = run_metadata(
    {
        "kind": "scaling",
        "model": "spx",
        "job": "gpu",
        "RUN_LEVEL": RUN_LEVEL,
        "MAIN_SEED": model.MAIN_SEED,
        "PARTICLE_J_GRID": PARTICLE_J_GRID,
        "PARTICLE_FIXED_CHAINS": PARTICLE_FIXED_CHAINS,
        "CHAIN_GRID": CHAIN_GRID,
        "CHAIN_FIXED_J": CHAIN_FIXED_J,
        "NFITR": NFITR,
        "NREPS_EVAL": NREPS_EVAL,
    }
)
record["particle_scaling"] = df_particles.to_dict(orient="records")
record["chain_scaling"] = df_chains.to_dict(orient="records")

with open(os.path.join(out_dir, "latest.json"), "w") as f:
    json.dump(record, f, indent=2, default=str)
    f.write("\n")

print(f"\n--- Particle Scaling Summary ---\n{df_particles.to_string(index=False)}")
print(f"\n--- Chain Scaling Summary ---\n{df_chains.to_string(index=False)}")
print(
    f"\nwrote {out_dir}/ (particle_scaling.csv, chain_scaling.csv, scaling.csv, latest.json)"
)


