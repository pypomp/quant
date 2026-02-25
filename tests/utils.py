import json
import os
import datetime
import subprocess
import jax

def get_system_info():
    """Gathers system and package information for the run."""
    
    # Try to get pypomp commit hash if installed from git
    try:
        # Check pip show for typical location or use git
        result = subprocess.run(
            ["pip", "show", "pypomp"], 
            capture_output=True, text=True
        )
        pypomp_version = "unknown"
        for line in result.stdout.split('\n'):
            if line.startswith("Version:"):
                pypomp_version = line.split(":", 1)[1].strip()
    except Exception:
        pypomp_version = "error"
        
    try:
        jax_version = jax.__version__
        devices = str(jax.devices())
    except Exception:
        jax_version = "error"
        devices = "error"
        
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "jax_version": jax_version,
        "jax_devices": devices,
        "pypomp_version": pypomp_version
    }

def append_history(metrics, filepath):
    """Appends a dictionary of metrics to a JSONL file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    with open(filepath, "a") as f:
        # jsonl format is one JSON object per line
        f.write(json.dumps(metrics) + "\n")


import jax.numpy as jnp
import numpy as np

def _to_python(val):
    if isinstance(val, (jnp.ndarray, np.ndarray)):
        if val.size == 1:
            return float(val)
        return val.tolist()
    return val

def get_pomp_metrics(pomp_obj, execution_time=None, run_config=None):
    """
    Extracts logLik, top 5 estimates per parameter, unit logliks,
    and returns a history dictionary.
    """
    metrics = {
        "execution_time": execution_time,
        "run_config": run_config or {},
    }
    
    # Try to extract logLik (total) and top 5 estimates per parameter
    metrics["loglik"] = None
    metrics["top_5_estimates"] = {}
    
    try:
        if hasattr(pomp_obj, 'results') and callable(pomp_obj.results):
            df = pomp_obj.results()
            if 'logLik' in df.columns:
                df_sorted = df.sort_values(by='logLik', ascending=False)
                
                # Best loglik overall 
                metrics["loglik"] = float(df_sorted.iloc[0]['logLik'])
                
                # Top 5 estimates for each parameter
                top_5_df = df_sorted.head(5)
                top_5_theta = {}
                for col in top_5_df.columns:
                    if col not in ['logLik', 'se']:
                        top_5_theta[col] = top_5_df[col].tolist()
                        
                metrics["top_5_estimates"] = top_5_theta
            else:
                metrics["top_5_estimates"] = {"error": "logLik column not found in results()"}
        else:
            metrics["top_5_estimates"] = {"error": "results() method not found on pomp_obj"}
    except Exception as e:
        metrics["top_5_estimates"] = {"error": str(e)}
        
    # Extract method times
    try:
        if hasattr(pomp_obj, 'time'):
            time_df = pomp_obj.time()
            metrics["method_times"] = time_df.to_dict(orient="records")
        else:
            metrics["method_times"] = None
    except Exception as e:
        metrics["method_times"] = {"error": str(e)}
        
    return metrics
