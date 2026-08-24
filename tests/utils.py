import datetime
import json
import logging
import os
import pickle
import subprocess
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_sha() -> str | None:
    """The quant commit a run was produced from, so a regression can be bisected."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _gpu_type() -> str | None:
    """Detect GPU hardware description if running on GPU."""
    try:
        for d in jax.devices():
            if getattr(d, "platform", None) == "gpu" or (
                hasattr(d, "device_kind") and d.device_kind.lower() != "cpu"
            ):
                return getattr(d, "device_kind", str(d))
    except (RuntimeError, OSError) as exc:
        logger.debug("Failed to detect GPU via JAX: %s", exc)
    for env_var in ("SLURM_JOB_GRES", "SLURM_GPUS"):
        val = os.environ.get(env_var)
        if val:
            return val
    return None


def run_metadata(run_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance for a single run.

    Everything here is needed to answer "which pypomp, on what hardware, from
    which commit" months later, when a number in the history looks wrong.
    """
    try:
        devices = [str(d) for d in jax.devices()]
    except (RuntimeError, OSError):
        devices = []

    slurm_info = {
        k: os.environ.get(v)
        for k, v in {
            "job_id": "SLURM_JOB_ID",
            "partition": "SLURM_JOB_PARTITION",
            "cpus": "SLURM_CPUS_PER_TASK",
            "gres": "SLURM_JOB_GRES",
            "gpus": "SLURM_GPUS",
        }.items()
        if os.environ.get(v) is not None
    }

    gpu_type = _gpu_type()
    if gpu_type is not None:
        slurm_info["gpu_type"] = gpu_type

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "pypomp_version": _package_version("pypomp"),
        "jax_version": _package_version("jax"),
        "quant_git_sha": _git_sha(),
        "devices": devices,
        "slurm": slurm_info,
        "run_config": run_config or {},
    }


#: Columns that identify a trace row rather than carry a value. Kept whenever
#: `trace_cols` subsets the traces, so the result stays joinable and groupable.
TRACE_INDEX_COLS = ("theta_idx", "replicate", "unit", "iteration", "method")


def _slim_traces(
    traces: pd.DataFrame,
    trace_cols: Sequence[str] | None = None,
    thin: int = 1,
) -> pd.DataFrame:
    """Subset trace columns and keep every `thin`-th iteration.

    Thinning is by `iteration` value, not row position, since the long format
    repeats each index once per chain. Iteration 0 always survives.
    """
    if not isinstance(traces, pd.DataFrame) or traces.empty:
        return traces

    if trace_cols is not None:
        keep = [c for c in TRACE_INDEX_COLS if c in traces.columns]
        missing = [c for c in trace_cols if c not in traces.columns]
        if missing:
            print(f"  note: trace columns not found, skipped: {missing}")
        keep += [c for c in trace_cols if c in traces.columns and c not in keep]
        traces = cast(pd.DataFrame, traces[keep])

    if thin > 1 and "iteration" in traces.columns:
        traces = cast(pd.DataFrame, traces[traces["iteration"] % thin == 0])

    return traces


def save_run(
    pomp_obj: Any,
    out_dir: str,
    run_config: dict[str, Any] | None = None,
    execution_time: float | None = None,
    pickle_name: str = "fitted.pkl",
    write_traces: bool = True,
    trace_cols: Sequence[str] | None = None,
    thin: int = 1,
) -> dict[str, Any]:
    """Persist one run: pickle as fallback, text as the record.

    Writes into `out_dir`:
      <pickle_name>   the whole fitted object (gitignored fallback)
      results.csv     parameter estimates + logLik
      traces.csv.gz   per-iteration traces, when the object has them
      timings.csv     per-method wall clock
      latest.json     provenance and algorithmic configuration for this run

    The CSV/JSON outputs are the committed record; the pickle exists so a run
    can be reopened if something not captured here turns out to matter.

    `latest.json` deliberately holds only what the CSVs cannot: which pypomp,
    on what hardware, from which commit, with which knobs. Per-run drift is
    tracked by git -- these files are committed, so `git log -p <latest.json>`
    is the history.

    `trace_cols` and `thin` exist for many-chain runs, where the traces dominate
    the record: a fit estimating 2 of 13 parameters carries 11 constant columns.
    Both default to off, so callers that do not ask keep the full traces.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, pickle_name), "wb") as f:
        pickle.dump(pomp_obj, f)

    def _try_write(name, fn):
        try:
            df = fn()
            if isinstance(df, pd.DataFrame) and not df.empty:
                path = os.path.join(out_dir, name)
                df.to_csv(path, index=False)
                return len(df)
        except (AttributeError, TypeError, ValueError, OSError, RuntimeError) as e:
            print(f"  note: could not write {name}: {type(e).__name__}: {e}")
        return None

    _try_write("results.csv", pomp_obj.results)
    _try_write("timings.csv", pomp_obj.time)
    if write_traces:
        _try_write(
            "traces.csv.gz",
            lambda: _slim_traces(pomp_obj.traces(), trace_cols, thin),
        )

    metrics = get_pomp_metrics(pomp_obj, run_config=run_config)
    metrics.update(run_metadata(run_config))

    with open(os.path.join(out_dir, "latest.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
        f.write("\n")

    return metrics


def save_run_multi(
    pomp_objs: dict[str, Any],
    out_dir: str,
    run_config: dict[str, Any] | None = None,
    group_column: str = "unit",
    pickle_name: str = "fitted.pkl",
    write_traces: bool = True,
) -> dict[str, Any]:
    """`save_run` for a test that fits one object per group.

    Measles fits each town separately, so there is no single object whose
    `results()` is the run. The tables written here are the per-group tables
    concatenated with a leading `group_column`, so a report reads the same
    filenames it would for a single-object run and simply groups by that
    column. The pickle holds the whole mapping.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, pickle_name), "wb") as f:
        pickle.dump(pomp_objs, f)

    def _try_write(name, method):
        frames = []
        for group, obj in pomp_objs.items():
            try:
                df = getattr(obj, method)()
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                print(f"  note: could not build {name} for {group}: {e}")
                continue
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                df.insert(0, group_column, group)
                frames.append(df)
        if not frames:
            return None
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(os.path.join(out_dir, name), index=False)
        return len(out)

    _try_write("results.csv", "results")
    _try_write("timings.csv", "time")
    if write_traces:
        _try_write("traces.csv.gz", "traces")

    metrics = run_metadata(run_config)
    metrics["results_history"] = {
        group: get_pomp_metrics(obj).get("results_history", [])
        for group, obj in pomp_objs.items()
    }

    with open(os.path.join(out_dir, "latest.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
        f.write("\n")

    return metrics


def pfilter_logliks_frame(pomp_obj: Any, history_index: int = -1):
    """Per-replicate pfilter log-likelihoods as a tidy frame.

    `results()` reports one aggregated row per parameter set -- a logmeanexp
    and its standard error -- which is the right summary but the wrong thing
    to compare against R's per-replicate baseline. logmeanexp sits above the
    mean by roughly sd^2/2 (about 2.1 nats for SPX), so comparing it to a mean
    looks like a 2-nat bias that is not there.

    Returns columns theta_idx, replicate, logLik, matching the shape of the
    frozen R references.

    A PanelPomp carries a unit dimension too, and gains a `unit` column.
    pypomp names that dimension `unit` on most result types and `ll_unit` on
    the ones where it would collide with the parameter frame's own `unit`, so
    both spellings are accepted.
    """
    entry = pomp_obj.results_history[history_index]
    logliks = entry.logLiks

    dims = tuple(getattr(logliks, "dims", ()))
    unit_dim = next((d for d in dims if d in ("unit", "ll_unit")), None)
    rep_dims = [d for d in dims if d not in ("theta_idx", unit_dim)]

    if unit_dim is not None and rep_dims:
        df = logliks.to_dataframe(name="logLik").reset_index()
        df = df.rename(columns={unit_dim: "unit"})
        if "theta_idx" not in df.columns:
            df["theta_idx"] = 0
        # The rep coordinate is 0-based; the R references count from 1.
        df["replicate"] = df[rep_dims[0]].astype(int) + 1
        return df[["theta_idx", "unit", "replicate", "logLik"]].reset_index(drop=True)

    arr = np.asarray(logliks)
    if arr.ndim == 1:
        arr = arr[None, :]

    rows = []
    for theta_idx in range(arr.shape[0]):
        for replicate in range(arr.shape[1]):
            rows.append(
                {
                    "theta_idx": theta_idx,
                    "replicate": replicate + 1,
                    "logLik": float(arr[theta_idx, replicate]),
                }
            )
    return pd.DataFrame(rows)


def _to_python(val):
    if isinstance(val, (jnp.ndarray, np.ndarray)):
        if val.size == 1:
            return float(val)
        return val.tolist()
    return val


def _json_safe(val, _depth: int = 0):
    """Recursively coerce a value into something json.dumps can handle.

    A shallow isinstance check is not enough: pypomp's Result.config is a dict
    whose values include objects like RWSigma, which passes an `isinstance(v,
    dict)` test at the top level and then fails to serialise from inside.
    """
    if _depth > 6:
        return str(val)
    val = _to_python(val)
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    if isinstance(val, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_safe(v, _depth + 1) for v in val]
    return str(val)


def _entry_fields(entry: Any) -> dict[str, Any]:
    """Field mapping for one results_history entry.

    pypomp's Result objects expose their contents as properties rather than in
    __dict__, so reading __dict__ alone silently yields nothing and the
    algorithmic configuration of every run is lost. Fall back to the public
    attribute names when __dict__ is empty.
    """
    if isinstance(entry, dict):
        return entry

    data = getattr(entry, "__dict__", None) or {}
    if data:
        return data

    out = {}
    for name in dir(entry):
        if name.startswith("_"):
            continue
        try:
            value = getattr(entry, name)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            continue
        if callable(value):
            continue
        out[name] = value
    return out


def get_pomp_metrics(
    pomp_obj: Any,
    run_config: dict[str, Any] | None = None,
    **_kwargs,
) -> dict[str, Any]:
    """
    Summarise a run for `latest.json`: provenance and the algorithmic
    configuration pypomp actually executed.

    Saved JSON fields:
    - timestamp: Execution ISO timestamp.
    - pypomp_version: Version of the installed pypomp package.
    - jax_version: Version of the installed jax package.
    - quant_git_sha: Git commit SHA of the quant repository.
    - devices: List of JAX backend devices (e.g. CUDA/CPU).
    - slurm: SLURM job details (job_id, partition, cpus, gres, gpus, gpu_type).
    - run_config: Specified script metadata and run parameters.
    - results_history: Executed algorithmic configuration per entry in results_history.

    Args:
        pomp_obj: The Pomp or PanelPomp object to extract data from.
        run_config (dict, optional): Metadata for the run (e.g., N_UNITS, RUN_LEVEL).
    """
    metrics = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "run_config": run_config or {},
    }

    # Extract algorithmic configuration from results_history
    try:
        if hasattr(pomp_obj, "results_history"):
            history_list = pomp_obj.results_history
            metrics["results_history"] = []

            for entry in history_list:
                parsed_entry = {}
                data = _entry_fields(entry)

                for k, v in data.items():
                    # Skip bulky arrays and things recorded elsewhere.
                    if k.startswith("_") or k in {
                        "logLiks",
                        "shared_traces",
                        "unit_traces",
                        "traces_da",
                        "traces",
                        "theta",
                        "key",
                        "timestamp",
                        "rw_sd",
                        "CLL",
                        "ESS",
                        "payload",
                        "filter_mean",
                        "prediction_mean",
                    }:
                        continue

                    parsed_entry[k] = _json_safe(v)

                metrics["results_history"].append(parsed_entry)
        else:
            metrics["results_history"] = []
    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
        metrics["results_history"] = {"error": str(e)}

    return metrics


# --- Timing Benchmark Reporting Helpers ---


def load_timing_data(platform_dirs: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Load timings.csv and latest.json for each platform/configuration.

    Args:
        platform_dirs: dict mapping display label -> results directory path.

    Returns:
        dict mapping display label -> {
            "phases": dict[phase_name, seconds],
            "cfg": run_config dict,
            "meta": full latest.json dict,
            "dir": results directory path,
            "available": bool,
        }
    """
    runs = {}
    for label, d in platform_dirs.items():
        timing_path = os.path.join(d, "timings.csv")
        json_path = os.path.join(d, "latest.json")

        phases = {}
        if os.path.exists(timing_path):
            try:
                tm = pd.read_csv(timing_path)
                if "phase" in tm.columns and "time_seconds" in tm.columns:
                    phases = dict(zip(tm["phase"], tm["time_seconds"]))
                elif "stage" in tm.columns and "seconds" in tm.columns:
                    phases = dict(zip(tm["stage"], tm["seconds"]))
            except Exception as e:
                logger.warning("Error reading %s: %s", timing_path, e)

        meta = {}
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except Exception as e:
                logger.warning("Error reading %s: %s", json_path, e)

        cfg = meta.get("run_config", {}) or {}
        runs[label] = {
            "phases": phases,
            "cfg": cfg,
            "meta": meta,
            "dir": d,
            "available": bool(phases or meta),
        }
    return runs


def _extract_algorithmic_settings(
    runs: dict[str, dict[str, Any]], is_panel: bool = False
):
    """Extract rows for algorithmic & workload settings."""
    rows = []

    def _find_val(r, cfg_keys, rh_method=None, rh_keys=None):
        cfg = r.get("cfg", {}) or {}
        for k in cfg_keys:
            if k in cfg and cfg[k] is not None:
                return cfg[k]
        meta = r.get("meta", {}) or {}
        rh = meta.get("results_history", [])
        if isinstance(rh, list):
            for entry in rh:
                if not isinstance(entry, dict):
                    continue
                if rh_method and entry.get("method") != rh_method:
                    continue
                conf = entry.get("config", {}) or {}
                if isinstance(conf, dict) and rh_keys:
                    for rk in rh_keys:
                        if rk in conf and conf[rk] is not None:
                            return conf[rk]
        elif isinstance(rh, dict):
            for group_entries in rh.values():
                if isinstance(group_entries, list):
                    for entry in group_entries:
                        if not isinstance(entry, dict):
                            continue
                        if rh_method and entry.get("method") != rh_method:
                            continue
                        conf = entry.get("config", {}) or {}
                        if isinstance(conf, dict) and rh_keys:
                            for rk in rh_keys:
                                if rk in conf and conf[rk] is not None:
                                    return conf[rk]
        return None

    # Run level
    run_levels = [r["cfg"].get("RUN_LEVEL", "—") for r in runs.values()]
    rows.append(("Run Level", run_levels))

    # Starting points
    def get_starts(r):
        val = _find_val(r, ["NSTARTS", "Nstarts", "Nreps_global"])
        return (
            f"{val:,}"
            if isinstance(val, int)
            else (str(val) if val is not None else "—")
        )

    if any(
        _find_val(r, ["NSTARTS", "Nstarts", "Nreps_global"]) is not None
        for r in runs.values()
    ):
        rows.append(
            ("Starting Searches ($N_{starts}$)", [get_starts(r) for r in runs.values()])
        )

    # Optimization iterations
    def get_iters(r):
        val = _find_val(r, ["NFITR", "Nmif", "M"], rh_method="mif", rh_keys=["M"])
        return (
            f"{val:,}"
            if isinstance(val, int)
            else (str(val) if val is not None else "—")
        )

    if any(
        _find_val(r, ["NFITR", "Nmif", "M"], rh_method="mif", rh_keys=["M"]) is not None
        for r in runs.values()
    ):
        iter_label = (
            "MPIF Iterations ($N_{iter}$)"
            if is_panel
            else "IF2 Iterations ($N_{iter}$)"
        )
        rows.append((iter_label, [get_iters(r) for r in runs.values()]))

    # Training iterations (IFAD)
    def get_train_iters(r):
        val = _find_val(r, ["NTRAIN"], rh_method="train", rh_keys=["M"])
        return (
            f"{val:,}"
            if isinstance(val, int)
            else (str(val) if val is not None else "—")
        )

    if any(
        _find_val(r, ["NTRAIN"], rh_method="train", rh_keys=["M"]) is not None
        for r in runs.values()
    ):
        rows.append(
            (
                "Training Iterations ($N_{train}$)",
                [get_train_iters(r) for r in runs.values()],
            )
        )

    # Particles for estimation
    def get_fit_particles(r):
        val = _find_val(
            r, ["NP_FITR", "NP", "Np"], rh_method="mif", rh_keys=["J", "Np", "NP"]
        )
        return (
            f"{val:,}"
            if isinstance(val, int)
            else (str(val) if val is not None else "—")
        )

    has_fit_particles = any(
        _find_val(r, ["NP_FITR"]) is not None
        or (
            _find_val(r, ["NFITR", "Nmif", "M"], rh_method="mif", rh_keys=["M"])
            is not None
            and _find_val(r, ["NP", "Np"], rh_method="mif", rh_keys=["J", "Np", "NP"])
            is not None
        )
        for r in runs.values()
    )
    if has_fit_particles:
        part_label = (
            "MPIF Particles / Unit ($N_{p,fit}$)"
            if is_panel
            else "IF2 Particles ($N_p$)"
        )
        rows.append((part_label, [get_fit_particles(r) for r in runs.values()]))

    # Eval particles
    def get_eval_particles(r):
        val = _find_val(
            r, ["NP_EVAL", "NP", "Np"], rh_method="pfilter", rh_keys=["J", "Np", "NP"]
        )
        return (
            f"{val:,}"
            if isinstance(val, int)
            else (str(val) if val is not None else "—")
        )

    has_eval = is_panel or any(
        _find_val(
            r, ["NP_EVAL", "NP", "Np"], rh_method="pfilter", rh_keys=["J", "Np", "NP"]
        )
        is not None
        for r in runs.values()
    )
    if has_eval:
        eval_label = (
            "Pfilter Particles / Unit ($N_{p,eval}$)"
            if is_panel
            else "Pfilter Particles ($N_{p,eval}$)"
        )
        rows.append((eval_label, [get_eval_particles(r) for r in runs.values()]))

    # Evaluation replicates
    def get_reps(r):
        val = _find_val(
            r,
            ["NREPS_EVAL", "NREPS", "Nreps_eval", "Nreps"],
            rh_method="pfilter",
            rh_keys=["reps", "replicates"],
        )
        return (
            f"{val:,}"
            if isinstance(val, int)
            else (str(val) if val is not None else "—")
        )

    if any(
        _find_val(
            r,
            ["NREPS_EVAL", "NREPS", "Nreps_eval", "Nreps"],
            rh_method="pfilter",
            rh_keys=["reps", "replicates"],
        )
        is not None
        for r in runs.values()
    ):
        rows.append(
            ("Evaluation Replicates ($N_{reps}$)", [get_reps(r) for r in runs.values()])
        )

    # Floating-Point Precision
    if any("USE_64BIT" in r["cfg"] for r in runs.values()):

        def get_precision(label, r):
            if "USE_64BIT" in r["cfg"]:
                return (
                    "64-bit (float64)" if r["cfg"]["USE_64BIT"] else "32-bit (float32)"
                )
            if label.startswith("R") or (
                "r " in label.lower()
                and "pomp" in label.lower()
                and "pypomp" not in label.lower()
            ):
                return "64-bit (double)"
            return "—"

        rows.append(
            (
                "Floating-Point Precision",
                [get_precision(label, r) for label, r in runs.items()],
            )
        )

    # Units
    has_units = any(
        r["cfg"].get("UNIT") or r["cfg"].get("UNITS") or r["cfg"].get("units")
        for r in runs.values()
    )
    if has_units:

        def get_units(cfg):
            u = cfg.get("UNIT") or cfg.get("UNITS") or cfg.get("units")
            if isinstance(u, list):
                return f"{', '.join(u)} ({len(u)} units)"
            return str(u) if u is not None else "—"

        rows.append(("Unit(s)", [get_units(r["cfg"]) for r in runs.values()]))

    # Samplers (if measles / present)
    has_samplers = any(r["cfg"].get("SAMPLERS") for r in runs.values())
    if has_samplers:

        def get_samplers(label, r):
            if label.startswith("R") or (
                "r " in label.lower()
                and "pomp" in label.lower()
                and "pypomp" not in label.lower()
            ):
                return "Compiled C snippets"
            s = r["cfg"].get("SAMPLERS")
            if s == "fast":
                return "fast (pypomp.random)"
            elif s == "jax":
                return "stock JAX (jax.random)"
            return str(s) if s else "—"

        rows.append(
            (
                "Sampler Implementation",
                [get_samplers(label, r) for label, r in runs.items()],
            )
        )

    # Shared parameters (panel)
    has_shared = any(r["cfg"].get("SHARED_PARAMS") for r in runs.values())
    if has_shared:

        def get_shared(cfg):
            sp = cfg.get("SHARED_PARAMS")
            if isinstance(sp, list):
                return f"{', '.join(sp)} ({len(sp)} params)"
            return str(sp) if sp is not None else "—"

        rows.append(
            ("Shared Parameters", [get_shared(r["cfg"]) for r in runs.values()])
        )

    # Random seed
    def get_seed(cfg):
        s = cfg.get("MAIN_SEED", cfg.get("seed"))
        return str(s) if s is not None else "—"

    rows.append(("Random Seed", [get_seed(r["cfg"]) for r in runs.values()]))

    return rows


def _extract_software_settings(runs: dict[str, dict[str, Any]], is_panel: bool = False):
    """Extract rows for software & environment."""
    rows = []

    # Framework / package version
    def get_framework(label, meta):
        if is_panel and meta.get("panelPomp_version"):
            pomp_v = meta.get("pomp_version")
            return f"panelPomp {meta['panelPomp_version']}" + (
                f" (pomp {pomp_v})" if pomp_v else ""
            )
        elif "pomp_version" in meta and meta.get("pomp_version"):
            return f"pomp {meta['pomp_version']}"
        elif "pypomp_version" in meta and meta.get("pypomp_version"):
            return f"pypomp {meta['pypomp_version']}"
        return "—"

    rows.append(
        (
            "Pomp Framework",
            [get_framework(label, r["meta"]) for label, r in runs.items()],
        )
    )

    # Backend runtime
    def get_backend(label, meta):
        if "jax_version" in meta and meta.get("jax_version"):
            return f"JAX {meta['jax_version']}"
        elif "r_version" in meta and meta.get("r_version"):
            return f"R {meta['r_version']}"
        return "—"

    rows.append(
        (
            "Backend / Engine",
            [get_backend(label, r["meta"]) for label, r in runs.items()],
        )
    )

    # Git SHA
    def get_git(meta):
        sha = meta.get("quant_git_sha")
        return f"<code>{sha[:7]}</code>" if sha else "—"

    rows.append(("Quant Git Commit", [get_git(r["meta"]) for r in runs.values()]))

    # Timestamp
    def get_time(meta):
        ts = meta.get("timestamp")
        if not ts:
            return "—"
        try:
            return ts.replace("T", " ")[:19]
        except Exception:
            return str(ts)

    rows.append(("Run Timestamp", [get_time(r["meta"]) for r in runs.values()]))

    return rows


def _extract_hardware_settings(runs: dict[str, dict[str, Any]]):
    """Extract rows for hardware & compute."""
    rows = []

    # Compute device
    def get_device(label, meta, cfg):
        slurm = meta.get("slurm", {}) or {}
        hw = meta.get("hardware", {}) or {}
        devices = meta.get("devices", [])

        if slurm.get("gpu_type"):
            return f"{slurm['gpu_type']} (1 GPU)"
        if any("cuda" in d.lower() for d in devices):
            return "CUDA GPU"
        if slurm.get("gpus"):
            return f"GPU ({slurm['gpus']})"

        if hw.get("cpu_model"):
            cores = hw.get("cores", "—")
            model = hw.get("cpu_model", "CPU")
            return f"{model} ({cores} cores)"
        if devices and any("cpu" in d.lower() for d in devices):
            cpus = slurm.get("cpus", len(devices))
            return f"CPU ({len(devices)} host devices / {cpus} cores)"
        if slurm.get("cpus"):
            return f"CPU ({slurm['cpus']} cores)"
        return "CPU"

    rows.append(
        (
            "Compute Device",
            [get_device(label, r["meta"], r["cfg"]) for label, r in runs.items()],
        )
    )

    # Slurm Partition
    def get_partition(meta):
        slurm = meta.get("slurm", {}) or {}
        return slurm.get("partition", "—")

    rows.append(("Slurm Partition", [get_partition(r["meta"]) for r in runs.values()]))

    # Slurm Job ID
    def get_job_id(meta):
        slurm = meta.get("slurm", {}) or {}
        return slurm.get("job_id", "—")

    rows.append(("Slurm Job ID", [get_job_id(r["meta"]) for r in runs.values()]))

    return rows


def build_settings_comparison_html(
    runs: dict[str, dict[str, Any]], is_panel: bool = False
) -> str:
    """Generate a side-by-side HTML comparison table for settings across runs."""
    import html

    headers = list(runs.keys())

    sections = [
        (
            "Algorithmic & Workload Settings",
            _extract_algorithmic_settings(runs, is_panel),
        ),
        ("Software & Environment", _extract_software_settings(runs, is_panel)),
        ("Hardware & Compute", _extract_hardware_settings(runs)),
    ]

    html_parts = [
        '<div class="table-responsive">',
        '<table class="table table-striped table-hover align-middle" style="margin-bottom: 25px;">',
        '  <thead class="table-light">',
        "    <tr>",
        '      <th style="width: 28%; font-weight: bold;">Setting / Parameter</th>',
    ]
    for h in headers:
        html_parts.append(f'      <th style="font-weight: bold;">{html.escape(h)}</th>')
    html_parts.extend(
        [
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]
    )

    for section_title, rows in sections:
        html_parts.append(
            '    <tr style="background-color: #eaeded; font-weight: bold; border-top: 2px solid #bdc3c7;">'
        )
        html_parts.append(
            f'      <td colspan="{len(headers) + 1}" style="padding: 8px 12px; color: #2c3e50; font-size: 0.95rem;">{section_title}</td>'
        )
        html_parts.append("    </tr>")

        for label, values in rows:
            html_parts.append("    <tr>")
            html_parts.append(
                f'      <td style="font-weight: 500; color: #34495e; padding-left: 18px;">{label}</td>'
            )
            for val in values:
                html_parts.append(f"      <td>{val}</td>")
            html_parts.append("    </tr>")

    html_parts.extend(["  </tbody>", "</table>", "</div>"])
    return "\n".join(html_parts)


def build_timing_comparison_df(
    runs: dict[str, dict[str, Any]],
    is_panel: bool = False,
    baseline_key: str | None = None,
) -> pd.DataFrame:
    """Build the timing comparison DataFrame.

    Privileges cold start pfilter (`pfilter_cold`) for overall speedup and throughput.
    """
    if baseline_key is None:
        for k in runs:
            if k.startswith("R") or (
                "r " in k.lower() and "pomp" in k.lower() and "pypomp" not in k.lower()
            ):
                baseline_key = k
                break

    base = runs.get(baseline_key) if baseline_key is not None else None
    base_mif = base["phases"].get("mif", np.nan) if base else np.nan

    def get_pf_cold(r):
        if not r or not r.get("phases"):
            return np.nan
        ph = r["phases"]
        if "pfilter_cold" in ph:
            return ph["pfilter_cold"]
        if "pfilter" in ph:
            return ph["pfilter"]
        if "pfilter_warm" in ph:
            return ph["pfilter_warm"]
        return np.nan

    base_pf = get_pf_cold(base)
    base_total = (
        base_mif + base_pf
        if not np.isnan(base_mif) and not np.isnan(base_pf)
        else np.nan
    )

    r_cores = 36
    if base and base.get("meta"):
        r_cores = base["meta"].get("hardware", {}).get("cores", 36)

    def work(cfg, base_cfg=None):
        def _val(keys, default=1.0):
            for k in keys:
                if k in cfg and cfg[k] is not None:
                    return float(cfg[k])
            if base_cfg:
                for k in keys:
                    if k in base_cfg and base_cfg[k] is not None:
                        return float(base_cfg[k])
            return default

        return {
            "starts": _val(["NSTARTS", "Nstarts"]),
            "iters": _val(["NFITR", "Nmif"]),
            "particles": _val(["NP_FITR", "NP", "Np"]),
            "eval_particles": _val(["NP_EVAL", "NP", "Np"]),
            "reps": _val(["NREPS_EVAL", "NREPS", "Nreps_eval", "Nreps"]),
        }

    b_work = work(base["cfg"]) if base else None

    rows = []
    opt_col = "MPIF (s)" if is_panel else "IF2 (s)"
    opt_sp_col = "MPIF Speedup" if is_panel else "IF2 Speedup"

    for label, r in runs.items():
        if not r["available"]:
            rows.append(
                {
                    "Configuration": label,
                    opt_col: "not run",
                    opt_sp_col: "—",
                    "Pfilter (s)": "not run",
                    "Pfilter Speedup": "—",
                    "Total (s)": "not run",
                    "Total Speedup": "—",
                    "Throughput (vs 1 R CPU core)": "—",
                }
            )
            continue

        mif = r["phases"].get("mif", np.nan)
        pf = get_pf_cold(r)
        total = mif + pf if not np.isnan(mif) and not np.isnan(pf) else np.nan

        if label == baseline_key or base is None:
            mif_sp_str = pf_sp_str = tot_sp_str = (
                "1.00x" if label == baseline_key else "—"
            )
            tp_str = f"{r_cores:.2f}x" if label == baseline_key else "—"
        else:
            w = work(r["cfg"], base_cfg=base["cfg"])
            mif_scale = (
                (
                    (w["starts"] / b_work["starts"])
                    * (w["iters"] / b_work["iters"])
                    * (w["particles"] / b_work["particles"])
                )
                if b_work
                else 1.0
            )
            pf_scale = (
                (
                    (w["starts"] / b_work["starts"])
                    * (w["reps"] / b_work["reps"])
                    * (w["eval_particles"] / b_work["eval_particles"])
                )
                if b_work
                else 1.0
            )

            scaled_base_mif = base_mif * mif_scale
            scaled_base_pf = base_pf * pf_scale
            scaled_base_tot = scaled_base_mif + scaled_base_pf

            mif_sp = scaled_base_mif / mif if (mif == mif and mif > 0) else np.nan
            pf_sp = scaled_base_pf / pf if (pf == pf and pf > 0) else np.nan
            tot_sp = (
                scaled_base_tot / total if (total == total and total > 0) else np.nan
            )

            mif_sp_str = f"{mif_sp:.2f}x" if not np.isnan(mif_sp) else "—"
            pf_sp_str = f"{pf_sp:.2f}x" if not np.isnan(pf_sp) else "—"
            tot_sp_str = f"{tot_sp:.2f}x" if not np.isnan(tot_sp) else "—"
            tp_str = f"{tot_sp * r_cores:.2f}x" if not np.isnan(tot_sp) else "—"

        fmt = lambda s: "—" if (s != s or np.isnan(s)) else f"{s:.1f}s ({s / 60:.2f}m)"
        rows.append(
            {
                "Configuration": label,
                opt_col: fmt(mif),
                opt_sp_col: mif_sp_str,
                "Pfilter (s)": fmt(pf),
                "Pfilter Speedup": pf_sp_str,
                "Total (s)": fmt(total),
                "Total Speedup": tot_sp_str,
                "Throughput (vs 1 R CPU core)": tp_str,
            }
        )

    return pd.DataFrame(rows)


def build_cold_vs_warm_df(runs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build the cold vs warm pfilter breakdown DataFrame."""
    rows = []
    for label, r in runs.items():
        if not r["available"]:
            continue
        ph = r["phases"]
        cold = ph.get("pfilter_cold", np.nan)
        warm = ph.get("pfilter_warm", np.nan)

        if np.isnan(cold) and not np.isnan(warm):
            cold = warm
        if np.isnan(warm) and not np.isnan(cold):
            warm = cold

        overhead = cold - warm if not np.isnan(cold) and not np.isnan(warm) else np.nan

        fmt = lambda s: "—" if (s != s or np.isnan(s)) else f"{s:.2f}s"
        rows.append(
            {
                "Configuration": label,
                "Pfilter Cold (s)": fmt(cold),
                "Pfilter Warm (s)": fmt(warm),
                "Compilation Overhead (s)": fmt(overhead)
                if (not np.isnan(overhead) and overhead >= 0.005)
                else ("0.00s" if (not np.isnan(overhead) and overhead >= 0) else "—"),
            }
        )
    return pd.DataFrame(rows)
