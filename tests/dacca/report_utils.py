import json
import logging
import os
import sys

tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if tests_dir not in sys.path:
    sys.path.append(tests_dir)

from utils import (
    load_timing_data,
    build_settings_comparison_html,
    build_timing_comparison_df,
    build_cold_vs_warm_df,
)

import numpy as np
import pandas as pd
from plotnine import (
    element_line,
    element_rect,
    element_text,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_minimal,
)

# Hide noisy JAX CUDA log messages
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

theme_premium = theme_minimal(base_size=11) + theme(
    plot_title=element_text(weight="bold", size=13, color="#2c3e50", margin={"b": 10}),
    plot_subtitle=element_text(size=10, color="#7f8c8d", margin={"b": 15}),
    axis_title=element_text(weight="bold", size=10, color="#34495e"),
    axis_text=element_text(size=9, color="#2c3e50"),
    legend_title=element_text(weight="bold", size=9, color="#34495e"),
    legend_text=element_text(size=9, color="#2c3e50"),
    legend_position="bottom",
    strip_background=element_rect(fill="#f8f9fa", color="none"),
    strip_text=element_text(weight="bold", size=9, color="#2c3e50"),
    panel_grid_major=element_line(color="#eaeded"),
    panel_grid_minor=element_line(color="#f4f6f6"),
)

color_palette = {
    "IF2": "#3498db",
    "IFAD": "#1abc9c",
    "if2": "#3498db",
    "ifad": "#1abc9c",
    "mif": "#3498db",
    "train": "#1abc9c",
    "python (GPU)": "#1abc9c",
    "python (CPU)": "#3498db",
    "python": "#1abc9c",
    "R": "#e74c3c",
}


def scale_color_premium():
    return scale_color_manual(values=color_palette)


def scale_fill_premium():
    return scale_fill_manual(values=color_palette)


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def format_metadata(data):
    if not data:
        return "Metadata unavailable\n"
    return (
        f"pypomp version: {data.get('pypomp_version', 'N/A')}\n"
        f"JAX version:    {data.get('jax_version', 'N/A')}\n"
        f"Git SHA:        {data.get('quant_git_sha', 'N/A')}\n"
        f"Devices:        {', '.join(data.get('devices', []))}\n"
        f"Timestamp:      {data.get('timestamp', 'N/A')}\n"
    )


def format_r_metadata(data):
    if not data:
        return "Metadata unavailable\n"
    hw = data.get("hardware", {}) or {}
    return (
        f"R version:      {data.get('r_version', 'N/A')}\n"
        f"pomp version:   {data.get('pomp_version', 'N/A')}\n"
        f"Git SHA:        {data.get('quant_git_sha', 'N/A')}\n"
        f"Node / CPU:     {hw.get('nodelist', 'N/A')} / {hw.get('cpu_model', 'N/A')}\n"
        f"Cores:          {hw.get('cores', 'N/A')}\n"
        f"Timestamp:      {data.get('timestamp', 'N/A')}\n"
    )


def load_traces(path, label):
    """Read a traces.csv.gz into the long form the report plots.

    Returns None when the file is absent: the Python traces are gitignored (they
    turn over every run), so a clean checkout renders the report without them.
    """
    if not os.path.exists(path):
        return None
    traces = pd.read_csv(path).rename(
        columns={
            "theta_idx": "rep",
            "replicate": "rep",
            "iteration": "iter",
            "loglik": "logLik",
        }
    )
    traces["source"] = label
    return traces


def thin(traces, max_iters=150):
    """Keep at most `max_iters` evenly spaced iterations, plus the last one.

    A full Dhaka trace is 650 iterations x 100 starts x 28 parameters; drawing
    every point costs more memory than the render has and changes nothing that
    is visible at this figure size.
    """
    iters = np.sort(traces["iter"].unique())
    if len(iters) <= max_iters:
        return traces
    step = int(np.ceil(len(iters) / max_iters))
    keep = set(iters[::step].tolist()) | {iters[-1]}
    return traces[traces["iter"].isin(keep)]


def to_long(traces, max_iters=150):
    traces = thin(traces, max_iters)
    id_cols = [
        c for c in ["rep", "iter", "logLik", "method", "source"] if c in traces.columns
    ]
    param_cols = [c for c in traces.columns if c not in id_cols and c != "se"]
    long = traces.melt(
        id_vars=id_cols,
        value_vars=param_cols,
        var_name="quantity",
        value_name="param_value",
    )
    long["param_value"] = long["param_value"].astype(np.float32)
    long["quantity"] = long["quantity"].astype("category")
    return long


def nav_bar(current):
    pages = [
        ("algorithms", "IF2 vs IFAD", "../algorithms/report.html"),
        ("timing", "Timing & Throughput", "../timing/report.html"),
        ("loglik", "Likelihood Evaluation", "../loglik/report.html"),
    ]
    parts = []
    for key, title, url in pages:
        if key == current:
            parts.append(f"<strong><a href='{url}'>{title}</a></strong>")
        else:
            parts.append(f"<a href='{url}'>{title}</a>")
    return " &nbsp;|&nbsp; ".join(parts)
