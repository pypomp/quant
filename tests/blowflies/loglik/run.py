"""Blowflies1: pfilter log-likelihood distribution at the R default parameters.

Only replicate seeds vary; parameters remain at the R defaults.
"""

# --- SLURM CONFIG ---
# importance: medium
# description: "Blowflies1: fixed-parameter pfilter likelihood validation"
# tags: [loglik, blowflies, cpu, gpu]
# jobs:
#   cpu:
#     sbatch_args:
#       partition: standard
#       cpus-per-task: 1
#       mem: 6GB
#       time: "00:05:00"
#       output: "results/cpu/logs/slurm-%j.out"
#     env:
#       JAX_PLATFORMS: "cpu"
#   gpu:
#     sbatch_args:
#       partition: gpu
#       gpus: "v100:1"
#       cpus-per-gpu: 1
#       mem: 6GB
#       time: "00:05:00"
#       output: "results/gpu/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "00:05:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "01:00:00" }
#   4:
#     sbatch_args: { time: "02:00:00" }
# --- END SLURM CONFIG ---

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
import jax
import model
import numpy as np
from utils import pfilter_logliks_frame, save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Override results/<platform>; use a temporary directory for smoke tests",
    )
    parser.add_argument(
        "--observations",
        type=int,
        default=192,
        help="Full series by default; a prefix requires --out-dir",
    )
    args = parser.parse_args()
    level = int(os.environ.get("RUN_LEVEL", "1"))
    if level not in (1, 2, 3, 4):
        parser.error("RUN_LEVEL must be 1, 2, 3, or 4")
    if args.observations != 192 and args.out_dir is None:
        parser.error("A shortened diagnostic requires an explicit --out-dir")
    particles = (64, 1000, 5000, 5000)[level - 1]
    reps = (2, 20, 40, 100)[level - 1]
    platform = jax.devices()[0].platform
    root = HERE / "results" / "smoke" if level == 1 else HERE / "results"
    out_dir = args.out_dir or root / platform
    obj = model.blowflies(n_observations=args.observations)
    started = time.perf_counter()
    obj.pfilter(
        J=particles, reps=reps, key=jax.random.key(model.MAIN_SEED), CLL=True, ESS=True
    )
    np.asarray(obj.results_history[-1].logLiks)
    elapsed = time.perf_counter() - started
    save_run(
        obj,
        out_dir=str(out_dir),
        write_traces=False,
        run_config={
            "kind": "loglik",
            "model": "blowflies",
            "RUN_LEVEL": level,
            "MAIN_SEED": model.MAIN_SEED,
            "NP_EVAL": particles,
            "NREPS_EVAL": reps,
            "NOBS": args.observations,
            "theta": model.Parameters().as_r_dict(),
            "USE_64BIT": jax.config.x64_enabled,
            "SAMPLERS": "jax",
            "data_sha256": hashlib.sha256(model.DATA_PATH.read_bytes()).hexdigest(),
            "model_source_commit": model.UPSTREAM_COMMIT,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), HERE.parent / "model.py")
            },
            "execution_time_including_compilation": elapsed,
            "optimization_performed": False,
            "pathwise_ad_supported": False,
        },
    )
    frame = pfilter_logliks_frame(obj)
    frame.to_csv(out_dir / "pfilter_logliks.csv", index=False)
    print(obj.results())
    print(
        f"wrote {out_dir}/: {len(frame)} replicates, {elapsed:.2f}s including compilation"
    )


if __name__ == "__main__":
    main()
