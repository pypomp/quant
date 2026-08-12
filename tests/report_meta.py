"""The run metadata every report opens with.

Each test writes a `latest.json` next to its results holding the run
configuration (particle counts, starting points, replicates) and the
provenance of the run (versions, hardware, commit). This renders those files
as one table, so a reader knows what produced the numbers before seeing them.
"""

import html
import json
import os

RESULTS_DIR = "results"
MARKER = "latest.json"

# Result subdirectory -> the name the reports use for that arm.
ARM_LABELS = {
    "gpu": "pypomp (GPU)",
    "gpu_jax": "pypomp (GPU, JAX samplers)",
    "cpu": "pypomp (CPU)",
    "f32": "pypomp (32-bit)",
    "f64": "pypomp (64-bit)",
    "if2": "IF2",
    "ifad": "IFAD",
    "R": "R pomp",
}

ARM_ORDER = ["gpu", "gpu_jax", "cpu", "f64", "f32", "if2", "ifad", "R"]

CONFIG_LABELS = {
    "kind": "Test",
    "model": "Model",
    "variant": "Variant",
    "ALGORITHM": "Algorithm",
    "RUN_LEVEL": "Run level",
    "UNIT": "Unit",
    "UNITS": "Units",
    "SHARED_PARAMS": "Shared parameters",
    "NP": "Particles (J)",
    "NP_FITR": "Particles, fitting (J)",
    "NP_EVAL": "Particles, evaluation (J)",
    "NFITR": "IF2 iterations (M)",
    "NTRAIN": "IFAD training iterations",
    "NSTARTS": "Starting points",
    "NREPS": "Replicates",
    "NREPS_EVAL": "Evaluation replicates",
    "SAMPLERS": "Samplers",
    "USE_CPU": "CPU only",
    "USE_64BIT": "64-bit precision",
    "platform": "Platform",
    "MAIN_SEED": "Seed",
    "execution_time": "Script runtime",
    # The BIF/PMCMC benchmark keeps its own config vocabulary.
    "run_name": "Run",
    "stage": "Stage",
    "T": "Time steps",
    "seed": "Seed",
    "active_params": "Estimated parameters",
    "J": "Particles (J)",
    "bif_J": "BIF particles",
    "bif_starts": "BIF starting points",
    "bif_M_grid": "BIF iteration grid",
    "bif_perturb_grid": "BIF perturbation grid",
    "pmcmc_chains": "PMCMC chains",
    "pmcmc_nmcmc_sweep": "PMCMC scale-sweep iterations",
    "pmcmc_nmcmc_final": "PMCMC iterations",
}

CONFIG_ORDER = list(CONFIG_LABELS)

# The R helpers name the same quantities in their own style. Fold them onto the
# pypomp key so both arms land on one row instead of two half-empty ones.
CONFIG_ALIASES = {
    "Np": "NP",
    "Nmif": "NFITR",
    "Nstarts": "NSTARTS",
    "Nreps_global": "NSTARTS",
    "Nreps_eval": "NREPS_EVAL",
}

DURATION_KEYS = ("execution_time", "elapsed_sec", "runtime")

PROVENANCE_KEYS = {
    "timestamp",
    "run_config",
    "results_history",
    "timings",
    "devices",
    "slurm",
    "hardware",
    "pypomp_version",
    "jax_version",
    "r_version",
    "pomp_version",
    "panelPomp_version",
    "quant_git_sha",
}


def load_json(path):
    """Read a JSON file, or return `{}` so a report can render without it."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def discover_arms(base=RESULTS_DIR, marker=MARKER):
    """The result directories that hold a completed run, in report order.

    Selection is on the marker file rather than the directory, because the
    test runner creates `results/<arm>/logs/` before a job starts: an arm that
    has never produced a result still has a directory.
    """
    if not os.path.isdir(base):
        return {}
    names = [
        n for n in os.listdir(base) if os.path.exists(os.path.join(base, n, marker))
    ]
    names.sort(
        key=lambda n: (ARM_ORDER.index(n) if n in ARM_ORDER else len(ARM_ORDER), n)
    )
    return {ARM_LABELS.get(n, n): os.path.join(base, n) for n in names}


def _fmt_duration(seconds):
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _fmt_value(key, value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
        if len(items) > 8:
            return ", ".join(items[:8]) + f", … ({len(items)} total)"
        return ", ".join(items) if items else "—"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt_value(k, v)}" for k, v in value.items())
    if key in DURATION_KEYS and isinstance(value, (int, float)):
        return _fmt_duration(float(value))
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value)


def _run_config(meta):
    """The run configuration, whatever shape the test wrote it in."""
    cfg = meta.get("run_config")
    if not isinstance(cfg, dict):
        cfg = {k: v for k, v in meta.items() if k not in PROVENANCE_KEYS}
    return {CONFIG_ALIASES.get(k, k): v for k, v in cfg.items()}


def _method_calls(meta):
    """What each algorithm was actually called with, per `results_history`.

    This is the authoritative particle count: `run_config` records what the
    script intended, `results_history` what the call received.
    """
    history = meta.get("results_history")
    if isinstance(history, dict):
        history = next(iter(history.values()), [])
    if not isinstance(history, list):
        return None

    calls = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        cfg = entry.get("config") or {}
        shown = [f"{k}={_fmt_value(k, cfg[k])}" for k in ("J", "M", "reps") if k in cfg]
        call = entry.get("method", "?")
        if shown:
            call += " (" + ", ".join(shown) + ")"
        if call not in calls:
            calls.append(call)
    return "; ".join(calls) if calls else None


def _is_panel(meta):
    """Whether this is a panel run, for naming the R package that produced it.

    Runs made before `model` was recorded fall back to the directory the report
    renders from, which quarto sets to `tests/<model>/<test>`.
    """
    if str(_run_config(meta).get("model", "")).startswith("panel"):
        return True
    return any(p.startswith("panel") for p in os.getcwd().split(os.sep))


def _software(meta):
    parts = []
    if meta.get("r_version"):
        parts.append(f"R {meta['r_version']}")
        if meta.get("pomp_version"):
            parts.append(f"pomp {meta['pomp_version']}")
        # The R session loads panelPomp whether or not the test is a panel one.
        if meta.get("panelPomp_version") and _is_panel(meta):
            parts.append(f"panelPomp {meta['panelPomp_version']}")
    else:
        if meta.get("pypomp_version"):
            parts.append(f"pypomp {meta['pypomp_version']}")
        if meta.get("jax_version"):
            parts.append(f"JAX {meta['jax_version']}")
    return ", ".join(parts) or None


def _hardware(meta):
    hw = meta.get("hardware") or {}
    slurm = meta.get("slurm") or {}
    parts = []

    if hw:
        parts += [str(v) for v in (hw.get("cpu_model"), hw.get("nodelist")) if v]
        if hw.get("cores"):
            parts.append(f"{hw['cores']} cores")
    else:
        devices = meta.get("devices") or []
        gpu = slurm.get("gpu_type") or slurm.get("gpus")
        if gpu:
            parts.append(f"{gpu} x{len(devices)}" if len(devices) > 1 else str(gpu))
        elif devices:
            parts.append(f"{len(devices)} x CPU device")
    return ", ".join(parts) or None


def _slurm(meta):
    slurm = meta.get("slurm") or {}
    if not slurm.get("job_id"):
        return None
    detail = slurm.get("partition")
    return f"{slurm['job_id']} ({detail})" if detail else str(slurm["job_id"])


PROVENANCE_ROWS = [
    ("Method calls", _method_calls),
    ("Software", _software),
    ("Hardware", _hardware),
    ("SLURM job", _slurm),
    ("Commit", lambda m: (m.get("quant_git_sha") or "")[:10] or None),
    ("Run at", lambda m: m.get("timestamp")),
]


def _resolve_np(configs):
    """Fold R's undifferentiated `Np` into whichever particle count it means.

    A test that only evaluates has no fitting particle count to confuse it
    with, and vice versa, so the two arms belong on one row.
    """
    keys = {k for cfg in configs.values() for k in cfg}
    if "NP" not in keys:
        return configs
    if "NP_EVAL" in keys and not {"NP_FITR", "NFITR"} & keys:
        target = "NP_EVAL"
    elif "NP_FITR" in keys and "NP_EVAL" not in keys:
        target = "NP_FITR"
    else:
        return configs
    return {
        arm: {
            (target if k == "NP" and target not in cfg else k): v
            for k, v in cfg.items()
        }
        for arm, cfg in configs.items()
    }


def _rows(metas):
    """(label, [value per arm]) for every field any arm recorded."""
    configs = _resolve_np({arm: _run_config(meta) for arm, meta in metas.items()})
    keys = {k for cfg in configs.values() for k in cfg}
    ordered = [k for k in CONFIG_ORDER if k in keys]
    ordered += sorted(k for k in keys if k not in CONFIG_ORDER)

    rows = []
    for key in ordered:
        label = CONFIG_LABELS.get(key, key)
        values = [
            _fmt_value(key, configs[arm][key]) if key in configs[arm] else "—"
            for arm in metas
        ]
        if any(v != "—" for v in values):
            rows.append((label, values))

    provenance = []
    for label, fn in PROVENANCE_ROWS:
        values = [fn(meta) for meta in metas.values()]
        if any(v is not None for v in values):
            provenance.append((label, [v if v is not None else "—" for v in values]))
    return rows, provenance


def run_metadata_html(
    sources=None, base=RESULTS_DIR, marker=MARKER, title="Run metadata"
):
    """An HTML table of the run behind each arm of the report.

    `sources` maps a column label to the directory (or file) holding its
    metadata JSON; when omitted the arms are discovered under `base`.
    """
    if sources is None:
        sources = discover_arms(base, marker)
    if not sources:
        return (
            "<div class='alert alert-warning'><strong>No run metadata found</strong> "
            f"under <code>{html.escape(str(base))}</code>. Re-run the test to "
            "generate it.</div>"
        )

    metas = {}
    for label, path in sources.items():
        if os.path.isdir(path):
            path = os.path.join(path, marker)
        meta = load_json(path)
        if meta:
            if label == ARM_LABELS["R"] and _is_panel(meta):
                label = "R panelPomp"
            metas[label] = meta
    if not metas:
        return (
            "<div class='alert alert-warning'><strong>No run metadata found</strong> "
            f"for {html.escape(', '.join(sources))}.</div>"
        )

    config_rows, provenance_rows = _rows(metas)
    ncol = len(metas) + 1

    out = ["<table class='table table-sm' style='width:auto; font-size:0.9em;'>"]
    if title:
        out.append(
            f"<caption style='caption-side:top; font-weight:bold; padding-bottom:0.4em;'>"
            f"{html.escape(title)}</caption>"
        )
    out.append("<thead><tr><th></th>")
    out += [f"<th>{html.escape(str(a))}</th>" for a in metas]
    out.append("</tr></thead><tbody>")

    for section, rows in (("", config_rows), ("Provenance", provenance_rows)):
        if section and rows:
            out.append(
                f"<tr><th colspan='{ncol}' style='padding-top:0.9em;'>"
                f"{section}</th></tr>"
            )
        for label, values in rows:
            out.append(
                "<tr><th scope='row' style='font-weight:normal;'>"
                f"{html.escape(str(label))}</th>"
            )
            # A value the arms agree on spans them, so the cells that differ are
            # the ones that stand out.
            if len(values) > 1 and len(set(values)) == 1:
                out.append(
                    f"<td colspan='{len(values)}'>{html.escape(str(values[0]))}</td>"
                )
            else:
                out += [f"<td>{html.escape(str(v))}</td>" for v in values]
            out.append("</tr>")

    out.append("</tbody></table>")
    return "".join(out)


def show_run_metadata(
    sources=None, base=RESULTS_DIR, marker=MARKER, title="Run metadata"
):
    """Render `run_metadata_html` into the report."""
    from IPython.display import HTML, display

    display(HTML(run_metadata_html(sources, base, marker, title)))
