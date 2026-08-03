"""Evaluate a test's expect.yaml against the results it just produced.

This is the automatic layer underneath the by-eye review. It does not replace
looking at the report -- trace shapes and density plausibility still need a
human -- but it takes the checks that *are* mechanical (is the likelihood where
it should be, does the distribution match R, did anything get slower) and makes
them fail loudly instead of silently.

Run it after a test, or as the last step of the job so that a failed
expectation shows up as a FAILED SLURM job and reaches the mail you already
have configured:

    python scripts/check_expectations.py tests/spx/loglik
    python scripts/check_expectations.py tests/spx/loglik --results results
    python scripts/check_expectations.py tests/spx --all

Exit status is 0 if everything passed or was skipped, 1 if anything failed.

Check types:
  scalar             a statistic of a column is within [min, max]
  fraction_above     the fraction of a column above a threshold is >= min_fraction
  mean_vs_reference  |mean - reference mean| <= max_abs_diff
  ks_test            two-sample KS against a reference column, p >= min_p_value
  timing             a phase is under max_seconds, and/or under
                     max_ratio_vs_history times the median of past runs
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Skip(Exception):
    """Raised when a check cannot apply, e.g. not enough history yet."""


def resolve_results_dir(test_dir, results_dir):
    """Find the directory holding this run's outputs.

    Tests write into results/<platform>/ so that a GPU run and a CPU run of the
    same test do not overwrite each other. Accept either that or a flat
    results/, and prefer gpu over cpu when both are present.
    """
    base = os.path.join(test_dir, results_dir)
    if not os.path.isdir(base):
        return None

    def looks_like_results(d):
        return os.path.exists(os.path.join(d, "latest.json")) or glob.glob(
            os.path.join(d, "*.csv")
        )

    if looks_like_results(base):
        return base

    subdirs = sorted(
        d
        for d in glob.glob(os.path.join(base, "*"))
        if os.path.isdir(d) and looks_like_results(d)
    )
    if not subdirs:
        return None
    for preferred in ("gpu", "cpu"):
        for d in subdirs:
            if os.path.basename(d) == preferred:
                return d
    return subdirs[0]


def _load(test_dir, results_dir, name):
    """Load a results file from the resolved results dir, else the test dir."""
    for candidate in (
        os.path.join(results_dir, name) if results_dir else None,
        os.path.join(test_dir, name),
    ):
        if candidate and os.path.exists(candidate):
            return pd.read_csv(candidate)
    where = results_dir or os.path.join(test_dir, "results")
    raise Skip(f"{name} not found under {os.path.relpath(where, REPO_ROOT)}")


def _reference(test_dir, spec):
    ref = pd.read_csv(os.path.join(test_dir, spec["reference_csv"]))
    return ref[spec.get("reference_column", spec["column"])].dropna()


def check_scalar(test_dir, results_dir, spec):
    col = _load(test_dir, results_dir, spec["source"])[spec["column"]].dropna()
    stat = spec.get("statistic", "mean")
    value = getattr(col, stat)()
    lo, hi = spec.get("min"), spec.get("max")
    ok = (lo is None or value >= lo) and (hi is None or value <= hi)
    bound = " and ".join(
        s
        for s in (
            f">= {lo}" if lo is not None else "",
            f"<= {hi}" if hi is not None else "",
        )
        if s
    )
    return ok, f"{stat}({spec['column']}) = {value:.4f}, required {bound}"


def check_fraction_above(test_dir, results_dir, spec):
    col = _load(test_dir, results_dir, spec["source"])[spec["column"]].dropna()
    frac = float((col > spec["threshold"]).mean())
    ok = frac >= spec["min_fraction"]
    return ok, (
        f"{frac:.3f} of {len(col)} replicates above {spec['threshold']}, "
        f"required >= {spec['min_fraction']}"
    )


def check_mean_vs_reference(test_dir, results_dir, spec):
    col = _load(test_dir, results_dir, spec["source"])[spec["column"]].dropna()
    ref = _reference(test_dir, spec)
    diff = abs(col.mean() - ref.mean())
    ok = diff <= spec["max_abs_diff"]
    return ok, (
        f"mean {col.mean():.4f} (n={len(col)}) vs reference {ref.mean():.4f} "
        f"(n={len(ref)}), |diff| = {diff:.4f}, allowed {spec['max_abs_diff']}"
    )


def check_ks_test(test_dir, results_dir, spec):
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        raise Skip("scipy not available")
    col = _load(test_dir, results_dir, spec["source"])[spec["column"]].dropna()
    ref = _reference(test_dir, spec)
    if len(col) < 2:
        raise Skip(f"only {len(col)} replicate(s); KS needs a sample")
    stat, p = ks_2samp(col, ref)
    ok = p >= spec["min_p_value"]
    return ok, (
        f"KS D = {stat:.4f}, p = {p:.4g} (n={len(col)} vs {len(ref)}), "
        f"required p >= {spec['min_p_value']}"
    )


def check_timing(test_dir, results_dir, spec):
    timings = _load(test_dir, results_dir, spec["source"])
    row = timings[timings["phase"] == spec["phase"]]
    if row.empty:
        raise Skip(f"phase {spec['phase']!r} not in {spec['source']}")
    seconds = float(row["time_seconds"].iloc[0])

    if "max_seconds" in spec:
        ok = seconds <= spec["max_seconds"]
        return ok, (
            f"{spec['phase']} took {seconds:.2f} s, ceiling {spec['max_seconds']} s"
        )

    # History-relative regression check.
    history_path = os.path.join(results_dir, "history.jsonl") if results_dir else ""
    if not history_path or not os.path.exists(history_path):
        raise Skip("no history.jsonl yet")
    past = []
    with open(history_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = (rec.get("timings") or {}).get(spec["phase"])
            if value is not None:
                past.append(float(value))
    # The current run is the last entry; compare against everything before it.
    past = past[:-1]
    need = spec.get("min_history", 3)
    if len(past) < need:
        raise Skip(f"{len(past)} past run(s), need {need}")
    median = pd.Series(past).median()
    ratio = seconds / median if median else float("inf")
    ok = ratio <= spec["max_ratio_vs_history"]
    return ok, (
        f"{spec['phase']} took {seconds:.2f} s vs median {median:.2f} s of "
        f"{len(past)} past runs, ratio {ratio:.2f}, allowed "
        f"{spec['max_ratio_vs_history']}"
    )


CHECKS = {
    "scalar": check_scalar,
    "fraction_above": check_fraction_above,
    "mean_vs_reference": check_mean_vs_reference,
    "ks_test": check_ks_test,
    "timing": check_timing,
}


def run_test_dir(test_dir, results_dir):
    """Evaluate one test directory. Returns (passed, failed, skipped, rows)."""
    expect_path = os.path.join(test_dir, "expect.yaml")
    if not os.path.exists(expect_path):
        return 0, 0, 0, []

    with open(expect_path) as f:
        expect = yaml.safe_load(f) or {}

    rel = os.path.relpath(test_dir, REPO_ROOT)
    rows = []
    resolved = resolve_results_dir(test_dir, results_dir)

    # A run level below the test's floor is a smoke test, not a measurement.
    run_level = None
    latest = os.path.join(resolved, "latest.json") if resolved else ""
    if latest and os.path.exists(latest):
        with open(latest) as f:
            run_level = (json.load(f).get("run_config") or {}).get("RUN_LEVEL")
    floor = expect.get("min_run_level")
    if floor is not None and run_level is not None and run_level < floor:
        rows.append(
            (
                rel,
                "(all)",
                "SKIP",
                f"run level {run_level} below min_run_level {floor}",
            )
        )
        return 0, 0, 1, rows

    passed = failed = skipped = 0
    for spec in expect.get("checks", []):
        name = spec.get("name", spec["type"])
        fn = CHECKS.get(spec["type"])
        if fn is None:
            rows.append((rel, name, "SKIP", f"unknown check type {spec['type']!r}"))
            skipped += 1
            continue
        try:
            ok, detail = fn(test_dir, resolved, spec)
        except Skip as e:
            rows.append((rel, name, "SKIP", str(e)))
            skipped += 1
            continue
        except Exception as e:
            rows.append((rel, name, "FAIL", f"{type(e).__name__}: {e}"))
            failed += 1
            continue
        rows.append((rel, name, "PASS" if ok else "FAIL", detail))
        passed += ok
        failed += not ok

    return passed, failed, skipped, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="test directory, or a parent with --all")
    parser.add_argument(
        "--all",
        action="store_true",
        help="evaluate every expect.yaml beneath the target",
    )
    parser.add_argument(
        "--results",
        default="results",
        help="results subdirectory to read (default: results)",
    )
    parser.add_argument(
        "--report",
        help="write a JSON summary here (default: <test_dir>/<results>/check_report.json)",
    )
    args = parser.parse_args()

    target = os.path.abspath(args.target)
    if args.all:
        dirs = sorted(
            os.path.dirname(p)
            for p in glob.glob(
                os.path.join(target, "**", "expect.yaml"), recursive=True
            )
        )
    else:
        dirs = [target]

    if not dirs:
        print(f"no expect.yaml found under {args.target}")
        return 0

    total = [0, 0, 0]
    all_rows = []
    for d in dirs:
        p, f_, s, rows = run_test_dir(d, args.results)
        total[0] += p
        total[1] += f_
        total[2] += s
        all_rows.extend(rows)

    width = max((len(r[1]) for r in all_rows), default=10)
    current = None
    for test, name, status, detail in all_rows:
        if test != current:
            print(f"\n{test}")
            current = test
        mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "SKIP": " skip "}[status]
        print(f"  [{mark}] {name:<{width}}  {detail}")

    print(f"\n{total[0]} passed, {total[1]} failed, {total[2]} skipped")

    if len(dirs) == 1:
        resolved = resolve_results_dir(dirs[0], args.results)
        report_path = args.report or os.path.join(
            resolved or os.path.join(dirs[0], args.results), "check_report.json"
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(
                {
                    "passed": total[0],
                    "failed": total[1],
                    "skipped": total[2],
                    "checks": [
                        {"test": t, "name": n, "status": s, "detail": d}
                        for t, n, s, d in all_rows
                    ],
                },
                f,
                indent=2,
                default=str,
            )
            f.write("\n")

    return 1 if total[1] else 0


if __name__ == "__main__":
    sys.exit(main())
