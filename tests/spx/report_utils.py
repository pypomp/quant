import logging
import numpy as np
import pandas as pd
from plotnine import (
    theme_minimal, theme, element_text, element_rect, element_line,
    scale_color_manual, scale_fill_manual
)

# Hide noisy JAX CUDA log messages
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

theme_premium = (
    theme_minimal(base_size=11)
    + theme(
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
        panel_grid_minor=element_line(color="#f4f6f6")
    )
)

color_palette = {
    "python (GPU)": "#1abc9c",
    "python (CPU)": "#3498db",
    "R": "#e74c3c",
    "python": "#1abc9c"
}

def scale_color_premium():
    return scale_color_manual(values=color_palette)

def scale_fill_premium():
    return scale_fill_manual(values=color_palette)

def format_metadata(data):
    if not data:
        return "Metadata unavailable\n"
    return (
        f"pypomp version: {data.get('pypomp_version', 'N/A')}\n"
        f"JAX version:    {data.get('jax_version', 'N/A')}\n"
        f"Git SHA:        {data.get('quant_git_sha', 'N/A')}\n"
        f"Devices:        {', '.join(data.get('devices', []))}\n"
        f"Timestamp:      {data.get('timestamp', 'N/A')}\n"
    )

def load_results_and_traces(csv_path, traces_path):
    res_df = pd.read_csv(csv_path)
    ll_frame = (
        pd.DataFrame(
            {
                "LL": res_df["logLik"],
                "sd": res_df["se"] if "se" in res_df.columns else 0.0,
            }
        )
        .sort_values(by="LL", ascending=False)
        .reset_index(drop=True)
    )
    traces = pd.read_csv(traces_path)
    return ll_frame, traces

def process_and_transform_traces(df, language):
    df_work = df.copy() if "param_value" in df.columns else df
    if "se" in df_work.columns:
        df_work = df_work.drop(columns=["se"])
    if "theta_idx" in df_work.columns or "iteration" in df_work.columns or "replicate" in df_work.columns:
        df_work = df_work.rename(columns={"theta_idx": "rep", "replicate": "rep", "iteration": "iter"})

    if "param_value" not in df_work.columns:
        id_vars = [c for c in ["iter", "rep", "logLik", "method"] if c in df_work.columns]
        val_vars = [c for c in df_work.columns if c not in id_vars]
        df_work = df_work.melt(id_vars=id_vars, value_vars=val_vars, var_name="quantity", value_name="param_value")

    val = df_work["param_value"].values
    qty = df_work["quantity"].values

    res = val.copy()
    pos_mask = np.isin(qty, ["mu", "kappa", "theta", "xi", "V_0"])
    val_pos = val[pos_mask]
    pos_valid = val_pos > 0
    pos_res = np.full_like(val_pos, np.nan, dtype=np.float64)
    pos_res[pos_valid] = np.log(val_pos[pos_valid])
    res[pos_mask] = pos_res

    rho_mask = (qty == "rho")
    val_rho = val[rho_mask]
    rho_valid = np.abs(val_rho) < 1
    rho_res = np.full_like(val_rho, np.nan, dtype=np.float64)
    rho_res[rho_valid] = np.log((1 + val_rho[rho_valid]) / (1 - val_rho[rho_valid]))
    res[rho_mask] = rho_res

    ll_mask = np.isin(qty, ["logLik", "loglik"])
    res[ll_mask] = val[ll_mask]

    df_work["value_T"] = res.astype(np.float32)
    df_work["param_value"] = df_work["param_value"].astype(np.float32)
    df_work["iter"] = df_work["iter"].astype(np.int16)
    df_work["rep"] = df_work["rep"].astype(np.int16)
    df_work["quantity"] = df_work["quantity"].astype("category")
    if "method" in df_work.columns:
        df_work["method"] = df_work["method"].astype("category")
    df_work["language"] = pd.Categorical([language] * len(df_work), categories=["python (GPU)", "python (CPU)", "R"])

    return df_work

def nav_bar(current):
    pages = [
        ("estimation", "Estimation", "../estimation/report.html"),
        ("timing", "Timing & Throughput", "../timing/report.html"),
        ("loglik", "Likelihood Evaluation", "../loglik/report.html")
    ]
    parts = []
    for key, title, url in pages:
        if key == current:
            parts.append(f"<strong><a href='{url}'>{title}</a></strong>")
        else:
            parts.append(f"<a href='{url}'>{title}</a>")
    return " &nbsp;|&nbsp; ".join(parts)

