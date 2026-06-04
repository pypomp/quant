"""Sync compact BIF benchmark summaries from the BIF paper repository."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BIF_REPO = Path(os.environ.get("BIF_REPO", "/Users/mac/git/bif")).expanduser()

RUNS = {
    "sir-T100-fourparam-gpu-J1000-20260605": {
        "source": BIF_REPO
        / "code"
        / "results"
        / "sir_bif_pmcmc"
        / "sir-T100-fourparam-gpu-J1000-20260605",
        "summary_prefix": "four_param",
        "rank": "four_param_bif_rank.csv",
        "figures": [
            "four_param_meeting_summary.csv",
            "four_param_runtime_comparison.png",
            "four_param_posterior_marginals.png",
            "four_param_interval_comparison.png",
            "four_param_pmcmc_diagnostics.png",
        ],
    },
    "dacca-T100-fourparam-gpu-J1000-20260604": {
        "source": BIF_REPO
        / "code"
        / "results"
        / "dacca_bif_pmcmc"
        / "dacca-T100-fourparam-gpu-J1000-20260604",
        "summary_prefix": "dacca",
        "rank": "dacca_bif_rank.csv",
        "figures": [
            "dacca_meeting_summary.csv",
            "dacca_runtime_comparison.png",
            "dacca_posterior_marginals.png",
            "dacca_interval_comparison.png",
            "dacca_pmcmc_diagnostics.png",
            "dacca_observed_data.png",
        ],
    },
}

TOP_LEVEL = [
    "config.json",
    "done.json",
    "selected_pmcmc_scale.json",
    "pmcmc_metrics.csv",
    "bif_metrics.csv",
    "posterior_summaries.csv",
]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied {src} -> {dst}")


def main() -> None:
    for run_name, spec in RUNS.items():
        src_dir = spec["source"]
        dst_dir = ROOT / "results" / run_name
        for name in TOP_LEVEL:
            copy_file(src_dir / name, dst_dir / name)
        copy_file(src_dir / spec["rank"], dst_dir / spec["rank"])
        for name in spec["figures"]:
            copy_file(src_dir / "figures" / "main" / name, dst_dir / "figures" / "main" / name)


if __name__ == "__main__":
    main()
