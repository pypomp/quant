import datetime
import json
import logging
import os
import pickle
import subprocess
from importlib.metadata import PackageNotFoundError, version
from typing import Any

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


def save_run(
    pomp_obj: Any,
    out_dir: str,
    run_config: dict[str, Any] | None = None,
    execution_time: float | None = None,
    pickle_name: str = "fitted.pkl",
    write_traces: bool = True,
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
        _try_write("traces.csv.gz", pomp_obj.traces)

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
    """
    entry = pomp_obj.results_history[history_index]
    arr = np.asarray(entry.logLiks)
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
