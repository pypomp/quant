"""Freeze R/pomp reference results as committed CSVs.

The R baselines in this repo are expensive to produce (one of them is budgeted
36 hours of wall clock) but they only change when `pomp` changes. Rather than
re-running them, we extract the small tidy data frames the reports actually use
into `R_reference/` directories, commit those, and record provenance in a
`MANIFEST.json` alongside.

The `.rds`/`.rda` originals stay gitignored and are not needed to render a
report. Re-run this only after bumping `pomp` and regenerating the R results.

Usage:
    python scripts/freeze_r_results.py freeze            # write R_reference/ + manifests
    python scripts/freeze_r_results.py freeze --only spx # one entry
    python scripts/freeze_r_results.py check             # verify committed CSVs vs manifests
    python scripts/freeze_r_results.py list              # show the spec and what's available
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the (gitignored) R outputs are read from. Defaults to this checkout, but
# `--source-root` lets you freeze from another one -- e.g. writing into a git
# worktree that by definition does not carry the ignored source files.
SOURCE_ROOT = REPO_ROOT


def source_path(rel):
    return os.path.join(SOURCE_ROOT, rel)


# Version of pomp/panelPomp the frozen data is attributed to. This is read from
# renv.lock, which records what the project pins, and is the best available
# evidence for baselines generated before this script existed. Runs frozen from
# here on should have it confirmed by the producing job's own log.
RENV_LOCK_NAME = "renv.lock"


# Each entry describes one frozen baseline.
#   source:    the R output file, relative to repo root (gitignored)
#   out_dir:   where the frozen CSVs go (committed)
#   produced_by: the R script that generates `source`
#   extractor: "pyreadr" for plain data frames, "rscript" when the .rda holds
#              S4 pomp objects that need pomp loaded to read
#   outputs:   {R object name -> csv filename}. `None` is pyreadr's key for an
#              unnamed single-object .rds.
SPEC = [
    {
        "name": "measles/parameter_comparison",
        "source": "tests/measles/R_comparison/parameter_comparison/results/mif_coefs.rds",
        "out_dir": "tests/measles/R_comparison/parameter_comparison/R_reference",
        "produced_by": "tests/measles/R_comparison/parameter_comparison/measles.R",
        "extractor": "pyreadr",
        "outputs": {None: "mif_coefs.csv"},
    },
    {
        "name": "measles/logLik_comparison",
        "source": "tests/measles/R_comparison/logLik_comparison/results/pfilter_logliks_f64.rds",
        "out_dir": "tests/measles/R_comparison/logLik_comparison/R_reference",
        "produced_by": "tests/measles/R_comparison/logLik_comparison/measles.R",
        "extractor": "pyreadr",
        "outputs": {None: "pfilter_logliks_f64.csv"},
    },
    {
        "name": "measles/speed_comparison",
        "source": "tests/measles/R_comparison/speed_comparison/results/r_pomp_timings.csv",
        "extra_sources": [
            "tests/measles/R_comparison/speed_comparison/results/r_pomp_results.csv"
        ],
        "out_dir": "tests/measles/R_comparison/speed_comparison/R_reference",
        "produced_by": "tests/measles/R_comparison/speed_comparison/measles.R",
        "extractor": "copy",
        "outputs": {
            "r_pomp_timings.csv": "r_pomp_timings.csv",
            "r_pomp_results.csv": "r_pomp_results.csv",
        },
    },
    {
        "name": "panel_measles/parameter_comparison",
        "source": "tests/panel_measles/R_comparison/mixed/parameter_comparison/results/mif_coefs.rds",
        "out_dir": "tests/panel_measles/R_comparison/mixed/parameter_comparison/R_reference",
        "produced_by": "tests/panel_measles/R_comparison/mixed/parameter_comparison/panel_measles.R",
        "extractor": "pyreadr",
        "outputs": {None: "mif_coefs.csv"},
    },
    {
        "name": "panel_measles/logLik_comparison",
        "source": "tests/panel_measles/R_comparison/mixed/logLik_comparison/results/pfilter_logliks_f64.rds",
        "extra_sources": [
            "tests/panel_measles/R_comparison/mixed/logLik_comparison/results/r_pomp_time.csv"
        ],
        "out_dir": "tests/panel_measles/R_comparison/mixed/logLik_comparison/R_reference",
        "produced_by": "tests/panel_measles/R_comparison/mixed/logLik_comparison/panel_measles.R",
        "extractor": "pyreadr",
        "outputs": {None: "pfilter_logliks_f64.csv"},
        "copy_also": {"r_pomp_time.csv": "r_pomp_time.csv"},
    },
    {
        "name": "panel_measles/speed_comparison",
        "source": "tests/panel_measles/R_comparison/mixed/speed_comparison/results/r_pomp_timings.csv",
        "out_dir": "tests/panel_measles/R_comparison/mixed/speed_comparison/R_reference",
        "produced_by": "tests/panel_measles/R_comparison/mixed/speed_comparison/panel_measles.R",
        "extractor": "copy",
        "outputs": {"r_pomp_timings.csv": "r_pomp_timings.csv"},
    },
    {
        "name": "dacca/loglik",
        "source": "tests/dacca/pfilter_check/R_results/dacca_results_eval.rda",
        "out_dir": "tests/dacca/pfilter_check/R_reference",
        "produced_by": "tests/dacca/pfilter_check/eval.R",
        "extractor": "pyreadr",
        "outputs": {"L_box": "pfilter_logliks.csv", "t_pfilter": "timings.csv"},
        "rename": {"L_box": "logLik", "t_pfilter": "seconds", "0": "logLik"},
        "proc_time": ["timings.csv"],
    },
    {
        "name": "spx/loglik",
        "source": "tests/spx/loglik/R_results/spx_results_eval.rda",
        "out_dir": "tests/spx/loglik/R_reference",
        "produced_by": "tests/spx/loglik/run.R",
        "extractor": "pyreadr",
        "outputs": {"L.box": "pfilter_logliks.csv", "t.box": "timings.csv"},
        "rename": {"L.box": "logLik", "t.box": "seconds", "0": "logLik"},
        "proc_time": ["timings.csv"],
    },
    {
        "name": "spx/estimation",
        "source": "tests/spx/estimation/R_results/search360_hidden/1d_global_search360.rda",
        "out_dir": "tests/spx/estimation/R_reference",
        "produced_by": "tests/spx/estimation/run.R",
        "extractor": "rscript",
        "script": "scripts/extract_spx_search360.R",
        "outputs": {
            "logliks": "pfilter_logliks.csv",
            "traces": "mif_traces.csv.gz",
            "timings": "timings.csv",
        },
    },
]

# The names R's proc_time vector carries, in order.
PROC_TIME_NAMES = ["user.self", "sys.self", "elapsed", "user.child", "sys.child"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_versions():
    """Read the R/pomp versions the project pins, from renv.lock."""
    out = {"R": None, "pomp": None, "panelPomp": None}
    try:
        lock_path = os.path.join(REPO_ROOT, RENV_LOCK_NAME)
        if not os.path.exists(lock_path):
            lock_path = source_path(RENV_LOCK_NAME)
        with open(lock_path) as f:
            lock = json.load(f)
        out["R"] = lock.get("R", {}).get("Version")
        for pkg in ("pomp", "panelPomp"):
            if pkg in lock.get("Packages", {}):
                out[pkg] = lock["Packages"][pkg].get("Version")
    except Exception as e:
        print(f"  warning: could not read renv.lock ({e})")
    return out


def _tidy_proc_time(df):
    """Turn R's 5-element proc_time vector into a labelled two-column frame."""
    import pandas as pd

    vals = df.iloc[:, 0].tolist()
    names = PROC_TIME_NAMES[: len(vals)]
    return pd.DataFrame({"stage": names, "seconds": vals})


def extract_pyreadr(entry, out_dir):
    """Read plain R data frames with pyreadr and write them as CSV."""
    import pyreadr

    written = {}
    src = source_path(entry["source"])
    data = pyreadr.read_r(src)

    for obj_name, csv_name in entry["outputs"].items():
        if obj_name not in data:
            available = [repr(k) for k in data]
            raise KeyError(
                f"{entry['name']}: object {obj_name!r} not in {entry['source']}; "
                f"found {', '.join(available)}"
            )
        df = data[obj_name].copy()

        if csv_name in entry.get("proc_time", []):
            df = _tidy_proc_time(df)
        else:
            # pyreadr leaves single-column frames with a positional column name
            # and carries R rownames into the index; neither is worth freezing.
            rename = entry.get("rename", {})
            df.columns = [str(rename.get(str(c), c)) for c in df.columns]
            df = df.reset_index(drop=True)

        dest = os.path.join(out_dir, csv_name)
        df.to_csv(dest, index=False)
        written[csv_name] = df
        print(f"  {entry['source']} :: {obj_name!r} -> {csv_name}  ({len(df)} rows)")

    for src_name, csv_name in entry.get("copy_also", {}).items():
        src_path = os.path.join(os.path.dirname(src), src_name)
        if not os.path.exists(src_path):
            print(f"  warning: {src_path} missing, skipped")
            continue
        shutil.copyfile(src_path, os.path.join(out_dir, csv_name))
        import pandas as pd

        written[csv_name] = pd.read_csv(os.path.join(out_dir, csv_name))
        print(f"  {src_name} -> {csv_name} (copied)")

    return written


def extract_copy(entry, out_dir):
    """Some R scripts already write CSV; freezing is just a copy."""
    import pandas as pd

    written = {}
    sources = [entry["source"]] + entry.get("extra_sources", [])
    by_basename = {os.path.basename(p): p for p in sources}

    for src_name, csv_name in entry["outputs"].items():
        rel = by_basename.get(src_name)
        if rel is None:
            raise KeyError(f"{entry['name']}: no source listed for {src_name}")
        src_path = source_path(rel)
        dest = os.path.join(out_dir, csv_name)
        shutil.copyfile(src_path, dest)
        written[csv_name] = pd.read_csv(dest)
        print(f"  {rel} -> {csv_name}  ({len(written[csv_name])} rows)")

    return written


def extract_rscript(entry, out_dir):
    """Delegate to R for .rda files holding S4 pomp objects.

    pyreadr cannot read these, and `traces()` needs pomp's class definitions.
    R_LIBS_USER may point at a scratch library holding pomp if the project's
    renv library is unavailable.
    """
    import pandas as pd

    script = os.path.join(REPO_ROOT, entry["script"])
    cmd = [
        "Rscript",
        script,
        source_path(entry["source"]),
        out_dir,
    ]
    print(f"  running {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{entry['name']}: Rscript failed\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    print("   " + proc.stdout.strip().replace("\n", "\n   "))

    # Record the pomp actually used to read the objects. It need not match what
    # produced them -- extraction only deserialises, it does not recompute.
    for line in proc.stdout.splitlines():
        if line.startswith("pomp version used for extraction:"):
            entry["_extracted_with_pomp"] = line.split(":", 1)[1].strip()

    written = {}
    for csv_name in entry["outputs"].values():
        dest = os.path.join(out_dir, csv_name)
        if not os.path.exists(dest):
            raise FileNotFoundError(f"{entry['name']}: R script did not write {dest}")
        written[csv_name] = pd.read_csv(dest)
    return written


EXTRACTORS = {
    "pyreadr": extract_pyreadr,
    "copy": extract_copy,
    "rscript": extract_rscript,
}


def freeze_entry(entry, versions):
    src = source_path(entry["source"])
    if not os.path.exists(src):
        print(f"[skip] {entry['name']}: source missing ({entry['source']})")
        return False

    out_dir = os.path.join(REPO_ROOT, entry["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"[freeze] {entry['name']}")

    written = EXTRACTORS[entry["extractor"]](entry, out_dir)

    all_sources = [entry["source"]] + entry.get("extra_sources", [])
    manifest = {
        "name": entry["name"],
        "frozen_at": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "frozen_by": "scripts/freeze_r_results.py",
        "produced_by": entry["produced_by"],
        "quant_git_sha": git_sha(),
        # Attribution, not measurement: these are the versions renv.lock pins,
        # not a reading taken from the job that produced the source file.
        "pinned_versions_from_renv_lock": versions,
        # Set only for entries extracted through R; deserialising the objects
        # does not recompute them, so this need not match the pinned version.
        "extracted_with_pomp": entry.get("_extracted_with_pomp"),
        "sources": [
            {
                "path": p,
                "sha256": sha256(source_path(p)),
                "bytes": os.path.getsize(source_path(p)),
                "mtime": datetime.datetime.fromtimestamp(
                    os.path.getmtime(source_path(p))
                )
                .replace(microsecond=0)
                .isoformat(),
            }
            for p in all_sources
            if os.path.exists(source_path(p))
        ],
        "files": {
            name: {
                "rows": int(len(df)),
                "columns": [str(c) for c in df.columns],
                "sha256": sha256(os.path.join(out_dir, name)),
            }
            for name, df in sorted(written.items())
        },
    }

    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return True


def git_sha():
    try:
        return subprocess.run(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def cmd_freeze(args):
    versions = pinned_versions()
    print(f"renv.lock pins: R {versions['R']}, pomp {versions['pomp']}\n")
    done = skipped = 0
    for entry in SPEC:
        if args.only and args.only not in entry["name"]:
            continue
        if freeze_entry(entry, versions):
            done += 1
        else:
            skipped += 1
    print(f"\nfroze {done} entr{'y' if done == 1 else 'ies'}, skipped {skipped}")
    return 0


def cmd_check(args):
    """Verify committed CSVs still match the checksums in their manifests.

    Cheap enough for CI, and catches a report reading a file someone edited by
    hand or a partially-regenerated freeze.
    """
    problems = []
    checked = 0
    for entry in SPEC:
        out_dir = os.path.join(REPO_ROOT, entry["out_dir"])
        manifest_path = os.path.join(out_dir, "MANIFEST.json")
        if not os.path.exists(manifest_path):
            problems.append(f"{entry['name']}: no MANIFEST.json (never frozen?)")
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        for name, meta in manifest["files"].items():
            path = os.path.join(out_dir, name)
            if not os.path.exists(path):
                problems.append(f"{entry['name']}: missing {name}")
                continue
            actual = sha256(path)
            if actual != meta["sha256"]:
                problems.append(
                    f"{entry['name']}: {name} changed "
                    f"(manifest {meta['sha256'][:12]}, actual {actual[:12]})"
                )
            checked += 1

    if problems:
        print(f"FAIL: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: {checked} frozen file(s) match their manifests")
    return 0


def cmd_list(args):
    versions = pinned_versions()
    print(f"renv.lock pins: R {versions['R']}, pomp {versions['pomp']}\n")
    for entry in SPEC:
        src = source_path(entry["source"])
        src_state = (
            f"{os.path.getsize(src) // 1024} KB" if os.path.exists(src) else "MISSING"
        )
        frozen = os.path.exists(
            os.path.join(REPO_ROOT, entry["out_dir"], "MANIFEST.json")
        )
        print(f"{entry['name']:38} source: {src_state:>10}   frozen: {frozen}")
        print(f"{'':38} {entry['source']}")
        print(f"{'':38} -> {entry['out_dir']}/")
    return 0


def main():
    global SOURCE_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=REPO_ROOT,
        help=(
            "checkout to read the gitignored R outputs from "
            "(default: this checkout). Useful when freezing into a worktree."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_freeze = sub.add_parser("freeze", help="extract and write R_reference/ CSVs")
    p_freeze.add_argument("--only", help="substring match against entry name")
    p_freeze.set_defaults(func=cmd_freeze)

    p_check = sub.add_parser("check", help="verify frozen CSVs against manifests")
    p_check.set_defaults(func=cmd_check)

    p_list = sub.add_parser("list", help="show the freeze spec")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    SOURCE_ROOT = os.path.abspath(args.source_root)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
