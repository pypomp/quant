import datetime
import json
import os
import pickle
import subprocess
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def _package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_sha() -> Optional[str]:
    """The quant commit a run was produced from, so a regression can be bisected."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def run_metadata(run_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Provenance for a single run.

    Everything here is needed to answer "which pypomp, on what hardware, from
    which commit" months later, when a number in the history looks wrong.
    """
    try:
        devices = [str(d) for d in jax.devices()]
    except Exception:
        devices = []

    return {
        "timestamp": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "pypomp_version": _package_version("pypomp"),
        "jax_version": _package_version("jax"),
        "quant_git_sha": _git_sha(),
        "devices": devices,
        "slurm": {
            k: os.environ.get(v)
            for k, v in {
                "job_id": "SLURM_JOB_ID",
                "partition": "SLURM_JOB_PARTITION",
                "cpus": "SLURM_CPUS_PER_TASK",
            }.items()
            if os.environ.get(v) is not None
        },
        "run_config": run_config or {},
    }


def save_run(
    pomp_obj: Any,
    out_dir: str,
    run_config: Optional[Dict[str, Any]] = None,
    execution_time: Optional[float] = None,
    pickle_name: str = "fitted.pkl",
    write_traces: bool = True,
) -> Dict[str, Any]:
    """Persist one run: pickle as fallback, text as the record.

    Writes into `out_dir`:
      <pickle_name>   the whole fitted object (gitignored fallback)
      results.csv     parameter estimates + logLik
      traces.csv.gz   per-iteration traces, when the object has them
      timings.csv     per-method wall clock
      latest.json     metrics for this run, pretty-printed and diffable
      history.jsonl   one appended line per run, for tracking drift over time

    The CSV/JSON outputs are the committed record; the pickle exists so a run
    can be reopened if something not captured here turns out to matter.
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
        except Exception as e:
            print(f"  note: could not write {name}: {type(e).__name__}: {e}")
        return None

    _try_write("results.csv", pomp_obj.results)
    _try_write("timings.csv", pomp_obj.time)
    if write_traces:
        _try_write("traces.csv.gz", pomp_obj.traces)

    metrics = get_pomp_metrics(
        pomp_obj, execution_time=execution_time, run_config=run_config
    )
    metrics.update(run_metadata(run_config))

    with open(os.path.join(out_dir, "latest.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
        f.write("\n")
    append_history(metrics, os.path.join(out_dir, "history.jsonl"))

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
    arr = np.asarray(getattr(entry, "logLiks"))
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


def append_history(metrics: Dict[str, Any], filepath: str):
    """Appends a dictionary of metrics to a JSONL file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, "a") as f:
        # jsonl format is one JSON object per line.
        # default=str so that an unforeseen object type can never lose a run's
        # record at the very last step, after all the compute has been paid for.
        f.write(json.dumps(metrics, default=str) + "\n")


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


def _entry_fields(entry: Any) -> Dict[str, Any]:
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
        except Exception:
            continue
        if callable(value):
            continue
        out[name] = value
    return out


def get_pomp_metrics(
    pomp_obj: Any,
    execution_time: Optional[float] = None,
    run_config: Optional[Dict[str, Any]] = None,
    history_index: int = -1,
):
    """
    Extracts logLik, top 5 estimates per parameter, unit logliks,
    algorithmic parameters, and summary statistics.

    Args:
        pomp_obj: The Pomp or PanelPomp object to extract data from.
        execution_time (float, optional): Total wall-clock time for the run.
        run_config (dict, optional): Metadata for the run (e.g., N_UNITS, RUN_LEVEL).
        history_index (int, optional): The index in results_history to use for
            summary statistics. Defaults to -1 (the most recent result).
    """
    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "execution_time": execution_time,
        "run_config": run_config or {},
    }

    # Extract logLik top 5 estimates and descriptive stats
    metrics["loglik"] = None
    metrics["loglik_stats"] = {}
    metrics["top_5_estimates"] = {}

    try:
        if hasattr(pomp_obj, "results") and callable(pomp_obj.results):
            df = pomp_obj.results(index=history_index)

            if not isinstance(df, pd.DataFrame):
                metrics["top_5_estimates"] = {
                    "error": f"results() returned {type(df)}, expected pandas.DataFrame"
                }
                return metrics

            if "logLik" in df.columns:
                df_sorted = df.sort_values(by="logLik", ascending=False)

                # Descriptive statistics for logLik over all parameter sets
                desc = df["logLik"].describe()
                metrics["loglik_stats"] = {
                    "min": float(desc.loc["min"]),
                    "25%": float(desc.loc["25%"]),
                    "median": float(desc.loc["50%"]),
                    "75%": float(desc.loc["75%"]),
                    "max": float(desc.loc["max"]),
                    "mean": float(desc.loc["mean"]),
                }

                # Best loglik overall
                metrics["loglik"] = float(df_sorted.iloc[0]["logLik"])

                # Top 5 estimates for each parameter
                top_5_df = df_sorted.head(5)
                top_5_theta = {}
                for col in top_5_df.columns:
                    if col not in ["logLik", "se"]:
                        top_5_theta[col] = top_5_df[col].tolist()

                metrics["top_5_estimates"] = top_5_theta
            else:
                metrics["top_5_estimates"] = {
                    "error": "logLik column not found in results()"
                }
        else:
            metrics["top_5_estimates"] = {
                "error": "results() method not found on pomp_obj"
            }
    except Exception as e:
        metrics["top_5_estimates"] = {"error": str(e)}

    # Extract method times
    try:
        if hasattr(pomp_obj, "time"):
            time_df = pomp_obj.time()
            if isinstance(time_df, pd.DataFrame):
                metrics["method_times"] = time_df.to_dict(orient="records")
            else:
                metrics["method_times"] = None
        else:
            metrics["method_times"] = None
    except Exception as e:
        metrics["method_times"] = {"error": str(e)}

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
    except Exception as e:
        metrics["results_history"] = {"error": str(e)}

    return metrics
