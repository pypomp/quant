import datetime
import json
import os
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd


def append_history(metrics: Dict[str, Any], filepath: str):
    """Appends a dictionary of metrics to a JSONL file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, "a") as f:
        # jsonl format is one JSON object per line
        f.write(json.dumps(metrics) + "\n")


def _to_python(val):
    if isinstance(val, (jnp.ndarray, np.ndarray)):
        if val.size == 1:
            return float(val)
        return val.tolist()
    return val


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
                data = (
                    entry if isinstance(entry, dict) else getattr(entry, "__dict__", {})
                )

                for k, v in data.items():
                    # Skip large/non-serializable objects and private attributes
                    if k.startswith("_") or k in {
                        "logLiks",
                        "shared_traces",
                        "unit_traces",
                        "traces_da",
                        "theta",
                        "key",
                        "timestamp",
                        "rw_sd",
                        "CLL",
                        "ESS",
                        "filter_mean",
                        "prediction_mean",
                    }:
                        continue

                    val = _to_python(v)
                    # Include only json-serializable basic types or lists/dicts
                    if isinstance(val, (int, float, str, bool, type(None), list, dict)):
                        parsed_entry[k] = val

                metrics["results_history"].append(parsed_entry)
        else:
            metrics["results_history"] = []
    except Exception as e:
        metrics["results_history"] = {"error": str(e)}

    return metrics
