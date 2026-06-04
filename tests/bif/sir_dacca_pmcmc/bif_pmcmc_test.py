"""
BIF versus PMCMC benchmark runner for SIR and Dacca examples.

This script delegates the model-specific implementations to the BIF paper
repository, while keeping the quant-test job configuration next to the report.
Set BIF_REPO to the local checkout of pypomp/bif_private if it is not at the
default path below.
"""

# --- SLURM CONFIG ---
# jobs:
#   sir_j1000:
#     sbatch_args:
#       job-name: "pypomp bif sir pmcmc"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 24GB
#       output: "results/logs/slurm-%j.out"
#       account: "ionides0"
#       time: "01:00:00"
#     env:
#       TARGET: "sir"
#       RUN_NAME: "sir-T100-fourparam-gpu-J1000"
#   dacca_j1000:
#     sbatch_args:
#       job-name: "pypomp bif dacca pmcmc"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 24GB
#       output: "results/logs/slurm-%j.out"
#       account: "ionides0"
#       time: "01:00:00"
#     env:
#       TARGET: "dacca"
#       RUN_NAME: "dacca-T100-fourparam-gpu-J1000"
# --- END SLURM CONFIG ---

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_BIF_REPO = Path(os.environ.get("BIF_REPO", "/Users/mac/git/bif"))
TARGET = os.environ.get("TARGET", "all").lower()
RUN_NAME = os.environ.get("RUN_NAME")


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def sir_cmd(bif_repo: Path) -> list[str]:
    outdir = ROOT / "raw_results" / "sir_bif_pmcmc"
    cmd = [
        sys.executable,
        str(bif_repo / "code" / "sir_bif_pmcmc.py"),
        "--outdir",
        str(outdir),
        "--T",
        "100",
        "--J",
        "1000",
        "--active-params",
        "gamma",
        "beta1",
        "beta2",
        "beta3",
        "--pmcmc-chains",
        "4",
        "--pmcmc-nmcmc-sweep",
        "1000",
        "--pmcmc-nmcmc-final",
        "5000",
        "--proposal-scales",
        "2",
        "--pmcmc-final-scale",
        "2",
        "--bif-M-grid",
        "100",
        "150",
        "--bif-perturb-grid",
        "0.02",
        "0.03",
        "--bif-starts",
        "4",
    ]
    if RUN_NAME:
        cmd.extend(["--run-name", RUN_NAME])
    return cmd


def dacca_cmd(bif_repo: Path) -> list[str]:
    outdir = ROOT / "raw_results" / "dacca_bif_pmcmc"
    cmd = [
        sys.executable,
        str(bif_repo / "code" / "dacca_bif_pmcmc.py"),
        "--outdir",
        str(outdir),
        "--T",
        "100",
        "--J",
        "1000",
        "--active-params",
        "gamma",
        "m",
        "epsilon",
        "sigma",
        "--pmcmc-chains",
        "4",
        "--pmcmc-nmcmc-sweep",
        "1000",
        "--pmcmc-nmcmc-final",
        "5000",
        "--proposal-scales",
        "4",
        "6",
        "--pmcmc-final-scale",
        "6",
        "--bif-M-grid",
        "150",
        "--bif-perturb-grid",
        "0.07",
        "0.1",
        "--bif-starts",
        "8",
    ]
    if RUN_NAME:
        cmd.extend(["--run-name", RUN_NAME])
    return cmd


def main() -> None:
    bif_repo = DEFAULT_BIF_REPO.expanduser().resolve()
    if not (bif_repo / "code" / "sir_bif_pmcmc.py").exists():
        raise FileNotFoundError(f"BIF_REPO does not look like the BIF repo: {bif_repo}")

    if TARGET in ("sir", "all"):
        run(sir_cmd(bif_repo))
    if TARGET in ("dacca", "all"):
        run(dacca_cmd(bif_repo))


if __name__ == "__main__":
    main()
