"""Automated checks: computed at render time, shown atop each report, and
summarized in the README by scripts/update_readme_checks.py."""

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm

PASS_Z = 3.0
FAIL_Z = 5.0
P_PASS = 2 * norm.sf(PASS_Z)
P_FAIL = 2 * norm.sf(FAIL_Z)
MIN_RUN_LEVEL = 3

CAVEAT = (
    "These checks are computed from the results each time the report is "
    "rendered. They only compare summary statistics, so the plots can show "
    "problems that the checks miss. Visual review lists what the plots should "
    "show."
)

_BADGE = {
    "PASS": "#27ae60",
    "CHECK": "#e67e22",
    "FAIL": "#c0392b",
    "n/a": "#95a5a6",
    "info": "#7f8c8d",
}


def p_verdict(p, m):
    """Verdict for a p-value, Bonferroni-adjusted over m tests."""
    if not np.isfinite(p):
        return "n/a"
    p_adj = min(1.0, p * m)
    if p_adj >= P_PASS:
        return "PASS"
    return "CHECK" if p_adj >= P_FAIL else "FAIL"


def loglik_checks(py, r, compared, m, verdict=True):
    """Mean-difference z test and two-sample KS test for two samples of
    log-likelihoods. With verdict=False the rows are reported as info."""
    py = np.asarray(py, dtype=float)
    r = np.asarray(r, dtype=float)
    if len(py) < 2 or len(r) < 2:
        return [
            {
                "check": c,
                "compared": compared,
                "result": "missing results",
                "verdict": "n/a",
            }
            for c in ("Mean log-likelihood", "Distribution (KS)")
        ]
    diff = py.mean() - r.mean()
    se = float(np.sqrt(py.var(ddof=1) / len(py) + r.var(ddof=1) / len(r)))
    z = diff / se
    d, p_ks = ks_2samp(py, r)
    return [
        {
            "check": "Mean log-likelihood",
            "compared": compared,
            "result": f"difference {diff:+.3f} nats (SE {se:.3f}), z = {z:+.2f}",
            "verdict": p_verdict(2 * norm.sf(abs(z)), m) if verdict else "info",
        },
        {
            "check": "Distribution (KS)",
            "compared": compared,
            "result": f"D = {d:.4f}, p = {p_ks:.3g}",
            "verdict": p_verdict(p_ks, m) if verdict else "info",
        },
    ]


def run_level(runs):
    """Lowest RUN_LEVEL among the runs from ru.load_timing_data, or None."""
    levels = [r["cfg"].get("RUN_LEVEL") for r in runs.values() if r["available"]]
    levels = [int(lv) for lv in levels if lv is not None]
    return min(levels) if levels else None


def gate_smoke(rows, level):
    """Below MIN_RUN_LEVEL the results are a smoke test and carry no verdict."""
    if level is None or level >= MIN_RUN_LEVEL:
        return rows
    note = f"smoke test (RUN_LEVEL {level})"
    return [{**r, "verdict": "n/a", "result": f"{r['result']}; {note}"} for r in rows]


def worst(rows):
    verdicts = [r["verdict"] for r in rows]
    for v in ("FAIL", "CHECK", "PASS"):
        if v in verdicts:
            return v
    return "n/a"


def badge(v):
    return (
        f"<span style='background:{_BADGE.get(v, '#95a5a6')};color:white;"
        f"padding:2px 8px;border-radius:4px;font-weight:bold'>{v}</span>"
    )


def table_html(rows, m):
    df = pd.DataFrame(rows)[["check", "compared", "result", "verdict"]]
    df["verdict"] = df["verdict"].map(badge)
    df.columns = ["Check", "Compared", "Result", "Verdict"]
    rule = (
        f"PASS when |z| < {PASS_Z:g}, CHECK when {PASS_Z:g} ≤ |z| < {FAIL_Z:g}, "
        f"and FAIL when |z| ≥ {FAIL_Z:g}. KS p-values use the equivalent "
        f"thresholds. Both are Bonferroni-adjusted over the {m} tests that "
        "receive a verdict."
    )
    return (
        f"<p><em>{CAVEAT}</em></p>"
        + df.to_html(
            classes="table table-striped table-hover", index=False, escape=False
        )
        + f"<p><small>{rule}</small></p>"
    )


def save(rows, level, path="checks.json", **meta):
    out = {"verdict": worst(rows), "run_level": level, **meta, "checks": rows}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
