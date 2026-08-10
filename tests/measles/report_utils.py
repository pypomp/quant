import logging
import os

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
from scipy.special import logit

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
    "pypomp": "#1abc9c",
    "pypomp (GPU)": "#1abc9c",
    "pypomp (CPU)": "#3498db",
    "pypomp (GPU, JAX samplers)": "#9b59b6",
    "pypomp (32-bit)": "#1abc9c",
    "pypomp (64-bit)": "#3498db",
    "R": "#e74c3c",
    "IF2": "#3498db",
    "IFAD": "#1abc9c",
}


def scale_color_premium():
    return scale_color_manual(values=color_palette)


def scale_fill_premium():
    return scale_fill_manual(values=color_palette)


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
        f"Node / cores:   {hw.get('nodelist', 'N/A')} / {hw.get('cores', 'N/A')}\n"
        f"Timestamp:      {data.get('timestamp', 'N/A')}\n"
    )


def load_json(path):
    if not os.path.exists(path):
        return {}
    import json

    with open(path) as f:
        return json.load(f)


# The scales the model is actually estimated on. Comparing raw values would put
# most of the mass of a rate parameter in a tail and hide any real difference.
LOG_PARAMS = ["R0", "sigma", "gamma", "iota", "sigmaSE", "psi"]
LOGIT_PARAMS = ["rho", "cohort", "amplitude", "S_0", "E_0", "I_0", "R_0"]


def to_estimation_scale(df, param_col="param", value_col="value"):
    """Log/logit-transform a long parameter frame, relabelling as it goes."""
    out = df.copy()
    values = out[value_col].to_numpy(dtype=float)
    params = out[param_col].to_numpy()

    log_mask = np.isin(params, LOG_PARAMS)
    logit_mask = np.isin(params, LOGIT_PARAMS)

    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(
            log_mask, np.log(np.where(values > 0, values, np.nan)), values
        )
        values = np.where(
            logit_mask,
            logit(
                np.clip(
                    np.where((values > 0) & (values < 1), values, np.nan),
                    1e-12,
                    1 - 1e-12,
                )
            ),
            values,
        )

    out[value_col] = values
    labels = {p: f"log({p})" for p in LOG_PARAMS}
    labels.update({p: f"logit({p})" for p in LOGIT_PARAMS})
    out[param_col] = out[param_col].replace(labels)
    return out


def wide_results_to_long(df, source, id_cols=("unit",)):
    """pypomp `results()` (wide, one column per parameter) as a long frame."""
    drop = ["logLik", "se", "theta_idx", "replicate", "method", "iteration"]
    keep_ids = [c for c in id_cols if c in df.columns]
    work = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    long = work.melt(id_vars=keep_ids, var_name="param", value_name="value")
    long["source"] = source
    return long


def r_coefs_to_long(df, source="R", id_cols=("unit",)):
    """The R `mif_coefs.csv` long frame, renamed onto the shared column names."""
    keep_ids = [c for c in id_cols if c in df.columns]
    out = df[list(keep_ids) + ["coef", "names"]].rename(
        columns={"coef": "value", "names": "param"}
    )
    out["source"] = source
    return out


def summarize(df, by, value_col="logLik"):
    return (
        df.groupby(by)[value_col]
        .agg(n="count", mean="mean", sd="std", min="min", max="max")
        .reset_index()
    )


def logmeanexp(x):
    """pomp's logmeanexp: the log of the mean likelihood, not the mean logLik."""
    from scipy.special import logsumexp

    x = np.asarray(x, dtype=float)
    return float(logsumexp(x) - np.log(len(x)))


def read_if_exists(path, **kwargs):
    """Read a results CSV, or return None so a report can note it is missing."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, **kwargs)


def missing_note(label, path):
    return (
        f"<div class='alert alert-warning'><strong>{label} not found</strong> "
        f"at <code>{path}</code>. Re-run the test to regenerate it.</div>"
    )


def load_traces(path, label):
    """Read a traces.csv.gz into the long form the reports plot.

    Returns None when the file is absent: the Python traces are gitignored
    (they turn over every run), so a clean checkout renders without them.
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

    Drawing every point of a long measles trace costs more memory than the
    render has and changes nothing visible at this figure size.
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
        c
        for c in ["unit", "rep", "iter", "logLik", "method", "source"]
        if c in traces.columns
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
        ("loglik", "Likelihood Evaluation", "../loglik/report.html"),
        ("estimation", "Parameter Estimation", "../estimation/report.html"),
        ("timing", "Timing & Throughput", "../timing/report.html"),
        ("algorithms", "IF2 vs IFAD", "../algorithms/report.html"),
    ]
    parts = []
    for key, title, url in pages:
        if key == current:
            parts.append(f"<strong><a href='{url}'>{title}</a></strong>")
        else:
            parts.append(f"<a href='{url}'>{title}</a>")
    return " &nbsp;|&nbsp; ".join(parts)
