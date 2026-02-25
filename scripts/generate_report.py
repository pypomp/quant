import json
import glob
import os
import matplotlib.pyplot as plt
import datetime

def load_history(filepath):
    history = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line))
    return history

def generate_report():
    os.makedirs("results", exist_ok=True)
    history_files = glob.glob("results/*_history.jsonl")
    
    report_md = "# Pypomp Standardization Test Report\n\n"
    report_md += f"Generated on: {datetime.datetime.now().isoformat()}\n\n"
    
    for hf in history_files:
        model_name = os.path.basename(hf).replace("_history.jsonl", "")
        data = load_history(hf)
        
        if not data:
            continue
            
        report_md += f"## Model: {model_name}\n\n"
        
        # Plot LogLik History
        valid_logliks = [d for d in data if d.get("loglik") is not None]
        if valid_logliks:
            times = [datetime.datetime.fromisoformat(d["timestamp"]) for d in valid_logliks]
            logliks = [d["loglik"] for d in valid_logliks]
            
            plt.figure(figsize=(10, 5))
            plt.plot(times, logliks, marker='o', linestyle='-')
            plt.title(f"{model_name} Final LogLikelihood Over Time")
            plt.xlabel("Run Timestamp")
            plt.ylabel("LogLik")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = f"results/{model_name}_loglik_history.png"
            plt.savefig(plot_path)
            plt.close()
            
            report_md += f"![LogLik History](./{model_name}_loglik_history.png)\n\n"
            
        # Table of recent runs
        report_md += "### Recent Runs (Last 5)\n\n"
        report_md += "| Timestamp | Pypomp Version | JAX Version | Run Level | Exec Time (s) | LogLik | Method Times |\n"
        report_md += "|---|---|---|---|---|---|---|\n"
        
        for d in data[-5:]:
            ts = d.get("timestamp", "N/A")[:19] # Truncate microseconds
            p_ver = d.get("pypomp_version", "N/A")
            j_ver = d.get("jax_version", "N/A")
            rl = d.get("run_config", {}).get("RUN_LEVEL", "N/A")
            et = f"{d.get('execution_time', 0):.2f}"
            ll = f"{d.get('loglik', 0):.2f}" if d.get('loglik') is not None else "N/A"
            
            method_times = d.get("method_times", [])
            if isinstance(method_times, list) and method_times:
                mt_str = ", ".join([f"{mt.get('method', 'unk')}: {mt.get('time', 0):.1f}s" for mt in method_times])
            else:
                mt_str = "N/A"
            
            report_md += f"| {ts} | {p_ver} | {j_ver} | {rl} | {et} | {ll} | {mt_str} |\n"
            
        report_md += "\n---\n"
        
    with open("results/report.md", "w") as f:
        f.write(report_md)
        
    print("Report generated at results/report.md")

if __name__ == "__main__":
    generate_report()
