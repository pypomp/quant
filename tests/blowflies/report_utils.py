"""Reuse Aaron's quant report styling; keep blowflies data checks local."""

import json
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacca import report_utils as _style
load_timing_data = _style.load_timing_data
theme_premium = _style.theme_premium
scale_color_premium = _style.scale_color_premium
scale_fill_premium = _style.scale_fill_premium


def build_settings_comparison_html(runs):
    """Reuse the shared table without inferring core count from JAX devices."""
    runs = deepcopy(runs)
    for run in runs.values():
        if run["cfg"].get("RUN_LEVEL") is None:
            run["cfg"]["RUN_LEVEL"] = "not recorded (historical)"
    html = _style.build_settings_comparison_html(runs)
    for run in runs.values():
        meta = run["meta"]
        hw, slurm = meta.get("hardware", {}) or {}, meta.get("slurm", {}) or {}
        devices = meta.get("devices", [])
        if not hw.get("cores") and not slurm.get("cpus") and any("cpu" in d.lower() for d in devices):
            inferred = f"CPU ({len(devices)} host devices / {len(devices)} cores)"
            html = html.replace(inferred, "CPU (core count not recorded)")
    return html


def load_record(directory):
    directory = Path(directory)
    paths = (directory / "latest.json", directory / "pfilter_logliks.csv")
    if not all(path.exists() for path in paths):
        return None, None, f"Missing record: {directory}"
    meta = json.loads(paths[0].read_text())
    frame = pd.read_csv(paths[1])
    if "logLik" not in frame or frame.empty:
        return meta, None, f"No per-replicate log likelihoods: {directory}"
    values = frame["logLik"].to_numpy(dtype=float)
    cfg = meta.get("run_config", {})
    if len(values) != cfg.get("NREPS_EVAL"):
        return meta, None, f"Replicate count does not match latest.json: {directory}"
    if not np.isfinite(values).all():
        return meta, None, f"Non-finite likelihoods in {directory}; density comparison omitted"
    return meta, values, None


def comparison_warning(r_meta, p_meta):
    """Require matching work and model before comparing two distributions."""
    r_cfg, p_cfg = r_meta.get("run_config", {}), p_meta.get("run_config", {})
    required = ("theta", "NP_EVAL", "NOBS", "model_source_commit", "data_sha256")
    mismatch = [key for key in required if key not in r_cfg or key not in p_cfg or r_cfg[key] != p_cfg[key]]
    if mismatch:
        return "Comparison omitted: missing or different " + ", ".join(mismatch)
    return None
