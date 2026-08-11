import json
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
    "R": "#e74c3c",
    "R panelPomp": "#e74c3c",
    "panelPomp": "#e74c3c",
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
        f"R version:        {data.get('r_version', 'N/A')}\n"
        f"pomp version:     {data.get('pomp_version', 'N/A')}\n"
        f"panelPomp version:{data.get('panelPomp_version', 'N/A')}\n"
        f"Git SHA:          {data.get('quant_git_sha', 'N/A')}\n"
        f"Node / cores:     {hw.get('nodelist', 'N/A')} / {hw.get('cores', 'N/A')}\n"
        f"Timestamp:        {data.get('timestamp', 'N/A')}\n"
    )


# The scales the model is actually estimated on. Comparing raw values would put
# most of the mass of a rate parameter in a tail and hide any real difference.
LOG_PARAMS = ["R0", "sigma", "gamma", "iota", "sigmaSE", "psi"]
LOGIT_PARAMS = ["rho", "cohort", "amplitude", "S_0", "E_0", "I_0", "R_0"]

SHARED_PARAMS = ["R0", "sigma", "gamma", "sigmaSE", "cohort", "amplitude"]
SPECIFIC_PARAMS = ["iota", "rho", "psi", "S_0", "E_0", "I_0", "R_0"]


def to_estimation_scale(df, param_col="param", value_col="value"):
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
    drop = [
        "logLik",
        "se",
        "theta_idx",
        "replicate",
        "method",
        "iteration",
        "shared logLik",
        "shared logLik se",
        "unit logLik",
        "unit logLik se",
    ]
    keep_ids = [c for c in id_cols if c in df.columns]
    work = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    long = work.melt(id_vars=keep_ids, var_name="param", value_name="value")
    long["source"] = source
    return long


def split_shared_specific(df):
    """Collapse pypomp's per-unit repetition of the shared parameters.

    A panel `results()` frame carries the shared parameters on every unit's
    row. The report wants them once, under `unit == "shared"`, so that the
    density of a shared parameter is not silently drawn four times over.
    """
    shared = df[df["param"].isin(SHARED_PARAMS)].copy()
    shared["unit"] = "shared"
    shared = shared.drop_duplicates()

    specific = df[df["param"].isin(SPECIFIC_PARAMS)].copy()
    return pd.concat([shared, specific], ignore_index=True)


def r_coefs_to_long(df, source="R panelPomp"):
    """The R `mif_coefs.csv` frame, already long, tagged with its source."""
    out = df[["unit", "param", "value"]].copy()
    out["source"] = source
    return out


def normalize_timings(df):
    """One column naming for both halves' `timings.csv`.

    The R helper writes `phase,time_seconds`; pypomp's `save_run` derives its
    table from `pomp_obj.time()`, which names the same two things `method` and
    `time`.
    """
    if df is None:
        return None
    return df.rename(columns={"method": "phase", "time": "time_seconds"})


def platform_dir(base="results", prefer=("gpu", "cpu"), marker="latest.json"):
    """The pypomp results directory to read, whichever backend produced it.

    `save_run` writes into `results/<jax platform>`, so the same script leaves
    its output under `gpu` on the cluster and `cpu` after a local smoke run.
    Existence of the directory is not enough to choose it -- the test runner
    creates `results/<arm>/logs/` before a job starts, so an arm that has never
    produced a result still has a directory. Selection is on the marker file
    the run itself writes. Falls back to the first preference so a missing-file
    note names the directory that was expected.
    """
    for name in prefer:
        path = os.path.join(base, name)
        if os.path.exists(os.path.join(path, marker)):
            return path
    return os.path.join(base, prefer[0])


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
    """Keep at most `max_iters` evenly spaced iterations, plus the last one."""
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
    ]
    parts = []
    for key, title, url in pages:
        if key == current:
            parts.append(f"<strong><a href='{url}'>{title}</a></strong>")
        else:
            parts.append(f"<a href='{url}'>{title}</a>")
    return " &nbsp;|&nbsp; ".join(parts)
