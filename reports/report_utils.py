"""Helpers shared by the cross-model reports in this directory."""

import logging
import os
import sys

import numpy as np
from IPython.display import HTML, display
from plotnine import (
    element_line,
    element_rect,
    element_text,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_minimal,
)
from scipy.special import logsumexp

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "tests"))

from models import MODELS
from utils import (
    build_cold_vs_warm_df,
    build_settings_comparison_html,
    build_timing_comparison_df,
    load_timing_data,
)

__all__ = [
    "MODELS",
    "build_cold_vs_warm_df",
    "build_timing_comparison_df",
]

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
    "R": "#e74c3c",
    "pypomp": "#1abc9c",
    "pypomp (32-bit)": "#9b59b6",
    "R (36 cores)": "#e74c3c",
    "pypomp (GPU)": "#1abc9c",
    "pypomp (GPU, JAX samplers)": "#9b59b6",
    "pypomp (CPU, 36 cores)": "#3498db",
    "SPX": "#1abc9c",
    "Dhaka cholera": "#3498db",
    "Measles": "#e67e22",
    "Panel measles": "#9b59b6",
}


def scale_color_premium():
    return scale_color_manual(values=color_palette)


def scale_fill_premium():
    return scale_fill_manual(values=color_palette)


def nav_bar(current):
    pages = [
        ("loglik", "Likelihood Evaluation", "loglik.html"),
        ("timing", "Timing & Throughput", "timing.html"),
        ("scaling", "Scaling", "scaling.html"),
        ("index", "All reports", "../index.html"),
    ]
    parts = []
    for key, title, url in pages:
        link = f"<a href='{url}'>{title}</a>"
        parts.append(f"<strong>{link}</strong>" if key == current else link)
    display(
        HTML(f"<div style='margin-bottom: 20px;'>{' &nbsp;|&nbsp; '.join(parts)}</div>")
    )


def models(kind):
    return [m for m, spec in MODELS.items() if kind in spec]


def results_dir(*parts):
    """A path under the repository root, relative to the working directory
    (Quarto runs each report from this directory)."""
    return os.path.relpath(os.path.join(ROOT, *parts))


def load_runs(model, kind):
    spec = MODELS[model][kind]
    arms = spec.get("arms", {"pypomp (GPU)": ""})
    return load_timing_data(
        {label: results_dir(spec["dir"], sub) for label, sub in arms.items()}
    )


def show_missing(runs):
    """Warn about every arm without results; runs is {model: runs}."""
    for model, model_runs in runs.items():
        for arm, r in model_runs.items():
            if not r["available"]:
                display(
                    HTML(
                        "<div class='alert alert-warning'><strong>Missing results "
                        f"for {MODELS[model]['label']}, {arm}:</strong> Expected at "
                        f"<code>{r['dir']}</code>.</div>"
                    )
                )


def logmeanexp(x):
    """pomp's logmeanexp: the log of the mean likelihood, not the mean logLik."""
    x = np.asarray(x, dtype=float)
    return logsumexp(x) - np.log(len(x))


def raw_html(html):
    """Markdown for an HTML block, for cells with `output: asis`."""
    return f"\n```{{=html}}\n{html}\n```\n"


def heading(model, level=3, prefix=""):
    return f"\n{'#' * level} {MODELS[model]['label']} {{#{prefix}{model}}}\n"


def settings_sections(runs):
    """Markdown for the settings appendix, one subsection per model."""
    return "".join(
        heading(model, prefix="settings-")
        + raw_html(
            build_settings_comparison_html(model_runs, is_panel=MODELS[model]["panel"])
        )
        for model, model_runs in runs.items()
    )
