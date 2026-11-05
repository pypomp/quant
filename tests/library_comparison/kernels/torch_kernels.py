"""GPU kernels of a pydpf adapter run under torch.profiler. usage: torch_kernels.py STEPS ADAPTER [adapter args...]
STEPS is the number of Euler steps across both calls (cold + warm), for per-step counts."""
import os, runpy, sys
import torch
from torch.profiler import ProfilerActivity, profile
steps, adapter = int(sys.argv[1]), sys.argv[2]
sys.argv = [adapter] + sys.argv[3:]
sys.path.insert(0, os.path.dirname(os.path.abspath(adapter)))
os.environ["PFB_RESULTS"] = "/tmp/torch_kernels_discard.jsonl"
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    runpy.run_path(adapter, run_name="__main__")
def dev(e):
    return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
rows = sorted([(e.key, e.count, dev(e)) for e in prof.key_averages() if dev(e) > 0], key=lambda r: -r[2])
tot, n = sum(r[2] for r in rows), sum(r[1] for r in rows)
print(f"GPU {tot / 1e3:.0f} ms, {n} kernels, {n / steps:.1f} kernels per Euler step, {tot / n:.1f} us per kernel")
for k, c, t in rows[:20]:
    print(f"{t / 1e3:9.1f} ms {100 * t / tot:5.1f}% {c / steps:6.2f}/step {t / c:7.1f} us  {k[:80]}")
