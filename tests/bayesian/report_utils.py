"""Shared report helpers for the Bayesian tests.

Mirrors tests/measles/report_utils.py -- same theme, same load-or-note-missing
idiom -- and adds the MCMC diagnostics these reports need. arviz is deliberately
not a dependency: requirements.txt pins an exact pypomp version and is the
mechanism by which these runs reproduce, so a new transitive dependency tree is
a disproportionate cost for the numpy below.
"""

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
from scipy.stats import norm, rankdata

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
    "R pomp": "#e74c3c",
    "reference": "#34495e",
    "grid reference": "#34495e",
    "prior": "#95a5a6",
    "truth": "#f39c12",
}


def scale_color_premium():
    return scale_color_manual(values=color_palette)


def scale_fill_premium():
    return scale_fill_manual(values=color_palette)


def nav_bar(current):
    pages = [
        ("reference", "Reference Posterior", "../reference/report.html"),
        ("pmcmc", "PMCMC", "../pmcmc/report.html"),
        ("abc", "ABC", "../abc/report.html"),
    ]
    parts = []
    for key, title, url in pages:
        if key == current:
            parts.append(f"<strong><a href='{url}'>{title}</a></strong>")
        else:
            parts.append(f"<a href='{url}'>{title}</a>")
    return " &nbsp;|&nbsp; ".join(parts)


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


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


def format_metadata(data):
    if not data:
        return "Metadata unavailable\n"
    cfg = data.get("run_config", {})
    slurm = data.get("slurm", {}) or {}
    lines = [
        f"- **Run at**: {data.get('timestamp', 'unknown')}",
        f"- **pypomp**: {data.get('pypomp_version', 'unknown')}"
        f" / **JAX**: {data.get('jax_version', 'unknown')}",
        f"- **Commit**: `{str(data.get('quant_git_sha', 'unknown'))[:10]}`",
        f"- **Devices**: {data.get('devices', 'unknown')}",
    ]
    if slurm.get("gpu_type"):
        lines.append(f"- **GPU**: {slurm['gpu_type']}")
    if cfg:
        knobs = ", ".join(f"`{k}`={v}" for k, v in cfg.items())
        lines.append(f"- **Config**: {knobs}")
    return "\n".join(lines) + "\n"


def format_r_metadata(data):
    if not data:
        return "Metadata unavailable\n"
    cfg = data.get("run_config", {})
    hw = data.get("hardware", {}) or {}
    lines = [
        f"- **Run at**: {data.get('timestamp', 'unknown')}",
        f"- **pomp**: {data.get('pomp_version', 'unknown')}"
        f" / **R**: {data.get('r_version', 'unknown')}",
        f"- **Commit**: `{str(data.get('quant_git_sha', 'unknown'))[:10]}`",
    ]
    if hw.get("cpu_model"):
        lines.append(f"- **CPU**: {hw['cpu_model']} ({hw.get('cores', '?')} cores)")
    if cfg:
        knobs = ", ".join(f"`{k}`={v}" for k, v in cfg.items())
        lines.append(f"- **Config**: {knobs}")
    return "\n".join(lines) + "\n"


# --- MCMC trace handling ----------------------------------------------------


def load_mcmc_traces(path, label):
    """Read an MCMC traces.csv.gz into the long form these reports plot.

    Returns None when absent, so a clean checkout renders without the (large,
    gitignored) trace files.
    """
    if not os.path.exists(path):
        return None
    traces = pd.read_csv(path).rename(
        columns={"theta_idx": "chain", "iteration": "iter", "loglik": "logLik"}
    )
    traces["source"] = label
    return traces


def drop_burnin(traces, frac=0.5):
    """Discard the first `frac` of each chain."""
    cutoff = traces["iter"].max() * frac
    return traces[traces["iter"] > cutoff]


def chains_array(traces, param):
    """A (n_chains, n_draws) array for one parameter, for the diagnostics."""
    wide = traces.pivot_table(index="chain", columns="iter", values=param)
    return wide.to_numpy(dtype=float)


# --- Diagnostics (Vehtari et al. 2021) --------------------------------------


def _rank_normalize(chains):
    """Rank-normalize across all draws, then map to normal scores."""
    flat = chains.reshape(-1)
    ranks = rankdata(flat)
    z = norm.ppf((ranks - 3.0 / 8.0) / (len(flat) - 0.25))
    return z.reshape(chains.shape)


def _split(chains):
    """Split each chain in half, doubling the chain count."""
    m, n = chains.shape
    half = n // 2
    if half < 2:
        return chains
    return np.concatenate([chains[:, :half], chains[:, n - half :]], axis=0)


def _rhat_plain(chains):
    m, n = chains.shape
    if m < 2 or n < 2:
        return float("nan")
    W = np.mean(np.var(chains, axis=1, ddof=1))
    if W == 0 or not np.isfinite(W):
        return float("nan")
    B = n * np.var(np.mean(chains, axis=1), ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def split_rhat(chains):
    """Rank-normalized split-R-hat. Values above ~1.01 indicate non-convergence."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2 or chains.shape[1] < 4:
        return float("nan")
    return _rhat_plain(_rank_normalize(_split(chains)))


def _autocov(x):
    """Biased autocovariance of a 1-D series, via FFT."""
    n = len(x)
    x = x - x.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(f * np.conjugate(f), nfft)[:n]
    return ac / n


def _ess(chains):
    """Effective sample size via Geyer's initial positive sequence."""
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    if n < 4:
        return float("nan")

    acov = np.array([_autocov(c) for c in chains])
    W = np.mean(acov[:, 0]) * n / (n - 1)
    if W == 0 or not np.isfinite(W):
        return float("nan")

    if m > 1:
        B = n * np.var(np.mean(chains, axis=1), ddof=1)
        var_hat = (n - 1) / n * W + B / n
    else:
        var_hat = W

    rho = 1.0 - (W - acov.mean(axis=0)) / var_hat
    rho[0] = 1.0

    # Sum successive pairs while they stay positive, enforcing monotonicity.
    pair_total = 0.0
    prev = np.inf
    k = 0
    while 2 * k + 1 < n:
        p = rho[2 * k] + rho[2 * k + 1]
        if p < 0:
            break
        p = min(p, prev)
        pair_total += p
        prev = p
        k += 1

    tau = 2.0 * pair_total - 1.0
    if tau <= 0:
        return float("nan")
    return float(m * n / tau)


def bulk_ess(chains):
    """Rank-normalized bulk effective sample size."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2 or chains.shape[1] < 4:
        return float("nan")
    return _ess(_rank_normalize(_split(chains)))


def ks_stat(x, y):
    """Two-sample Kolmogorov-Smirnov statistic."""
    from scipy.stats import ks_2samp

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return float(ks_2samp(x, y).statistic)


def ks_vs_grid(samples, axis, density):
    """KS distance between MCMC samples and a gridded marginal density.

    The reference posterior is a density on a lattice, not a sample, so the
    usual two-sample statistic does not apply: this compares the empirical CDF
    of `samples` against the CDF obtained by integrating `density` over `axis`.
    """
    samples = np.sort(np.asarray(samples, dtype=float))
    axis = np.asarray(axis, dtype=float)
    density = np.asarray(density, dtype=float)
    if len(samples) == 0 or len(axis) < 2:
        return float("nan")

    widths = np.gradient(axis)
    cdf = np.cumsum(density * widths)
    if cdf[-1] <= 0:
        return float("nan")
    cdf = cdf / cdf[-1]

    ref_at = np.interp(samples, axis, cdf)
    emp = np.arange(1, len(samples) + 1) / len(samples)
    return float(np.max(np.abs(emp - ref_at)))


def posterior_summary(traces, params, label):
    """Mean, sd, and a 90% credible interval per parameter."""
    rows = []
    for p in params:
        if p not in traces.columns:
            continue
        v = traces[p].to_numpy(dtype=float)
        rows.append(
            {
                "source": label,
                "parameter": p,
                "mean": np.mean(v),
                "sd": np.std(v, ddof=1),
                "q05": np.quantile(v, 0.05),
                "q95": np.quantile(v, 0.95),
                "split_rhat": split_rhat(chains_array(traces, p)),
                "bulk_ess": bulk_ess(chains_array(traces, p)),
            }
        )
    return pd.DataFrame(rows)


def grid_marginals(grid, param, other):
    """Marginalize the reference posterior surface onto one axis."""
    g = grid.groupby(param, as_index=False)["post"].sum()
    widths = np.gradient(np.sort(grid[other].unique()))
    g["post"] = g["post"] * float(np.mean(widths))
    axis = g[param].to_numpy(dtype=float)
    dens = g["post"].to_numpy(dtype=float)
    area = np.sum(dens * np.gradient(axis))
    if area > 0:
        dens = dens / area
    return axis, dens
