"""Speed and accuracy of pypomp.random's fast samplers against jax.random and scipy.

PART=speed times each sampler over a range of draw counts N (the gpu and cpu
jobs). PART=accuracy computes Wasserstein distances, quantile miss rates and
the density / Q-Q summaries the report plots. Each job writes its own JSON into
results/, with run_metadata() provenance, so the jobs can run concurrently.
"""

# --- SLURM CONFIG ---
# importance: high
# description: "Speed and accuracy of pypomp.random fast samplers vs jax.random / scipy.stats"
# tags: [samplers, performance, accuracy, gpu, cpu]
# jobs:
#   gpu:
#     sbatch_args:
#       job-name: "pypomp sampler speed (gpu)"
#       partition: gpu-rtx6000
#       gpus: "rtx_pro_6000_blackwell:1"
#       cpus-per-gpu: 1
#       mem: 32GB
#       time: "00:20:00"
#       output: "results/logs/slurm-gpu-%j.out"
#     env:
#       PART: "speed"
#   cpu:
#     sbatch_args:
#       job-name: "pypomp sampler speed (cpu)"
#       partition: standard
#       cpus-per-task: 1
#       mem: 16GB
#       time: "00:30:00"
#       output: "results/logs/slurm-cpu-%j.out"
#     env:
#       PART: "speed"
#       USE_CPU: "true"
#   accuracy:
#     sbatch_args:
#       job-name: "pypomp sampler accuracy"
#       partition: standard
#       cpus-per-task: 1
#       mem: 16GB
#       time: "00:30:00"
#       output: "results/logs/slurm-accuracy-%j.out"
#     env:
#       PART: "accuracy"
#       USE_CPU: "true"
#
# run_levels:
#   1:
#     sbatch_args: { time: "00:10:00" }
#   4:
#     sbatch_args: { time: "01:30:00" }
# --- END SLURM CONFIG ---

import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

USE_CPU = os.environ.get("USE_CPU", "false").lower() == "true"
if USE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import pypomp.random as ppr
from scipy import stats
from utils import run_metadata

RUN_LEVEL = int(os.environ.get("RUN_LEVEL", "1"))
PART = os.environ.get("PART", "speed")
PLATFORM = "cpu" if USE_CPU else "gpu"
L = RUN_LEVEL - 1

N_GRID = {
    "gpu": (
        [10**3, 10**4],
        [10**k for k in range(3, 7)],
        [10**k for k in range(3, 8)],
        [10**k for k in range(3, 9)],
    )[L],
    "cpu": (
        [10**3, 10**4],
        [10**k for k in range(3, 6)],
        [10**k for k in range(3, 7)],
        [10**k for k in range(3, 8)],
    )[L],
}[PLATFORM]
WORK = (10**5, 10**6, 10**7, 10**7)[L]
N_ACC = (10**4, 10**5, 10**6, 10**6)[L]
GRID_POINTS = (5, 30, 30, 30)[L]
QQ_PROBS = np.concatenate(
    [[1e-6, 1e-5, 1e-4], np.linspace(0.001, 0.999, 500), [1 - 1e-4, 1 - 1e-5, 1 - 1e-6]]
)

os.makedirs("results", exist_ok=True)
print("JAX devices:", jax.devices(), flush=True)
print(f"PART={PART} PLATFORM={PLATFORM} RUN_LEVEL={RUN_LEVEL}", flush=True)


def fill(n, values):
    return np.resize(np.asarray(values), n)


def mesh(a, b):
    ga, gb = np.meshgrid(a, b, indexing="ij")
    return ga.ravel(), gb.ravel()


BIN_MIX = mesh(
    [10, 40, 100, 250], [0.01, 0.039, 0.041, 0.099, 0.101, 0.499, 0.501, 0.99]
)
NB_MIX = mesh([1.0, 10.0, 100.0], [0.1, 0.4998, 0.5003, 0.7138, 0.7148, 0.9])

# name -> (pypomp sampler, jax sampler or None, regime label -> args(n))
SPEED = {
    "poisson": (
        ppr.fast_poisson,
        jax.random.poisson,
        {
            "lambda=0.01": lambda n: (fill(n, [0.01]).astype(np.float32),),
            "lambda=3.99 (below pypomp switch)": lambda n: (
                fill(n, [3.99]).astype(np.float32),
            ),
            "lambda=4.01 (above pypomp switch)": lambda n: (
                fill(n, [4.01]).astype(np.float32),
            ),
            "lambda=9.99 (below JAX switch)": lambda n: (
                fill(n, [9.99]).astype(np.float32),
            ),
            "lambda=10.01 (above JAX switch)": lambda n: (
                fill(n, [10.01]).astype(np.float32),
            ),
            "lambda=100": lambda n: (fill(n, [100.0]).astype(np.float32),),
            "mixed": lambda n: (
                fill(n, [0.01, 3.99, 4.01, 9.99, 10.01, 20.0, 50.0, 100.0]).astype(
                    np.float32
                ),
            ),
        },
    ),
    "binomial": (
        ppr.fast_binomial,
        jax.random.binomial,
        {
            **{
                f"n=100, p={p} ({where})": (
                    lambda n, p=p: (
                        fill(n, [100]).astype(np.int32),
                        fill(n, [p]).astype(np.float32),
                    )
                )
                for p, where in [
                    (0.039, "below pypomp switch"),
                    (0.041, "above pypomp switch"),
                    (0.099, "below JAX switch"),
                    (0.101, "above JAX switch"),
                    (0.499, "below flip"),
                    (0.501, "above flip"),
                ]
            },
            "mixed": lambda n: (
                fill(n, BIN_MIX[0]).astype(np.int32),
                fill(n, BIN_MIX[1]).astype(np.float32),
            ),
        },
    ),
    "gamma": (
        ppr.fast_gamma,
        jax.random.gamma,
        {
            "alpha=0.99 (below JAX switch)": lambda n: (
                fill(n, [0.99]).astype(np.float32),
            ),
            "alpha=1.01 (above JAX switch)": lambda n: (
                fill(n, [1.01]).astype(np.float32),
            ),
            "mixed": lambda n: (
                fill(n, [0.1, 0.5, 0.99, 1.01, 2.0, 10.0, 50.0, 100.0]).astype(
                    np.float32
                ),
            ),
        },
    ),
    "nbinomial": (
        lambda k, size, p: ppr.fast_nbinomial(k, size, p=p),
        None,
        {
            **{
                f"size=10, mu={mu:g}": (
                    lambda n, mu=mu: (
                        fill(n, [10.0]).astype(np.float32),
                        fill(n, [10.0 / (10.0 + mu)]).astype(np.float32),
                    )
                )
                for mu in [3.99, 4.01, 9.99, 10.01]
            },
            "mixed": lambda n: (
                fill(n, NB_MIX[0]).astype(np.float32),
                fill(n, NB_MIX[1]).astype(np.float32),
            ),
        },
    ),
}


MIN_SECONDS = 0.1
MAX_REPS = 20000


def time_sampler(fn, key, args, n):
    def scanned(keys, *a):
        def body(c, k):
            return c + jnp.sum(fn(k, *a).astype(jnp.float32)), None

        return jax.lax.scan(body, jnp.float32(0.0), keys)[0]

    args = [jax.device_put(a) for a in args]

    def build(reps):
        keys = jax.random.split(key, reps)
        compiled = jax.jit(scanned).lower(keys, *args).compile()
        compiled(keys, *args).block_until_ready()
        return keys, compiled

    def run(keys, compiled):
        t0 = time.perf_counter()
        compiled(keys, *args).block_until_ready()
        return time.perf_counter() - t0

    # One call must last MIN_SECONDS so launch/sync overhead is negligible.
    reps = int(np.clip(WORK // n, 10, 1000))
    keys, compiled = build(reps)
    elapsed = run(keys, compiled)
    if elapsed < MIN_SECONDS and reps < MAX_REPS:
        reps = int(min(MAX_REPS, np.ceil(reps * 1.2 * MIN_SECONDS / elapsed)))
        keys, compiled = build(reps)
        elapsed = run(keys, compiled)
    best = min(
        [elapsed] + [run(keys, compiled) for _ in range(2 if elapsed < 1.0 else 0)]
    )
    return best / reps, reps


def speed():
    key = jax.random.key(42)
    rows = []
    for dist, (pp_fn, jax_fn, regimes) in SPEED.items():
        for regime, make_args in regimes.items():
            for n in N_GRID:
                args = make_args(n)
                for sampler, fn in (("pypomp", pp_fn), ("jax", jax_fn)):
                    if fn is None:
                        continue
                    key, sub = jax.random.split(key)
                    t, reps = time_sampler(fn, sub, args, n)
                    rows.append(
                        {
                            "dist": dist,
                            "regime": regime,
                            "n": n,
                            "sampler": sampler,
                            "reps": reps,
                            "seconds": t,
                        }
                    )
                    print(
                        f"{dist:10s} {regime:36s} N={n:>10d} {sampler:6s} {t * 1e3:10.4f} ms",
                        flush=True,
                    )
    return rows


def wasserstein(key):
    out = {}
    grids = {
        "poisson": ("lambda", np.geomspace(0.1, 200.0, GRID_POINTS)),
        "binomial": ("p", np.linspace(0.01, 0.99, GRID_POINTS)),
        "gamma": ("alpha", np.geomspace(0.1, 100.0, GRID_POINTS)),
        "nbinomial": ("p", np.linspace(0.05, 0.95, GRID_POINTS)),
    }
    for dist, (pname, values) in grids.items():
        rows = []
        for v in values:
            key, k_pp, k_r1, k_r2 = jax.random.split(key, 4)
            full = jnp.full((N_ACC,), v, dtype=jnp.float32)
            if dist == "poisson":
                pp = ppr.fast_poisson(k_pp, full)
                r1, r2 = jax.random.poisson(k_r1, full), jax.random.poisson(k_r2, full)
                sd = np.sqrt(v)
            elif dist == "binomial":
                nn = jnp.full((N_ACC,), 100, dtype=jnp.int32)
                pp = ppr.fast_binomial(k_pp, nn, full)
                r1, r2 = (
                    jax.random.binomial(k_r1, nn, full),
                    jax.random.binomial(k_r2, nn, full),
                )
                sd = np.sqrt(100 * v * (1 - v))
            elif dist == "gamma":
                pp = ppr.fast_gamma(k_pp, full)
                r1, r2 = jax.random.gamma(k_r1, full), jax.random.gamma(k_r2, full)
                sd = np.sqrt(v)
            else:
                size = jnp.full((N_ACC,), 20.0, dtype=jnp.float32)
                pp = ppr.fast_nbinomial(k_pp, size, p=full)
                seed = int(jax.random.randint(k_r1, (), 0, 2**31 - 1))
                rng = np.random.default_rng(seed)
                r1 = stats.nbinom.rvs(20.0, v, size=N_ACC, random_state=rng)
                r2 = stats.nbinom.rvs(20.0, v, size=N_ACC, random_state=rng)
                sd = np.sqrt(20.0 * (1 - v)) / v
            w_pp = float(stats.wasserstein_distance(np.asarray(pp), np.asarray(r1)))
            w_ref = float(stats.wasserstein_distance(np.asarray(r2), np.asarray(r1)))
            rows.append(
                {"value": float(v), "w_pp": w_pp, "w_ref": w_ref, "sd": float(sd)}
            )
        fixed = {"binomial": {"n": 100}, "nbinomial": {"size": 20}}.get(dist, {})
        out[dist] = {"param": pname, "fixed": fixed, "rows": rows}
        print(f"wasserstein {dist} done", flush=True)
    return out


def miss_rates(key):
    u = jax.random.uniform(key, (N_ACC,), minval=0.0001, maxval=0.9999)
    un = np.asarray(u)
    poisson = []
    for lam in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        approx = np.asarray(ppr.poissoninv(u, lam))
        poisson.append(
            {"lam": lam, "rate": float(np.mean(approx != stats.poisson.ppf(un, lam)))}
        )
    binomial = []
    for n in [10, 100, 1000]:
        for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            approx = np.asarray(ppr.binominv(u, n, p))
            binomial.append(
                {
                    "n": n,
                    "p": p,
                    "rate": float(np.mean(approx != stats.binom.ppf(un, n, p))),
                }
            )
    print("miss rates done", flush=True)
    return {"n_queries": N_ACC, "poisson": poisson, "binomial": binomial}


def round_list(x, sig=6):
    return [float(f"{v:.{sig}g}") for v in np.asarray(x, dtype=float)]


def discrete_panel(pp, ref=None, pmf=None):
    pp = np.asarray(pp).astype(np.int64)
    lo, hi = int(pp.min()), int(pp.max())
    if ref is not None:
        ref = np.asarray(ref).astype(np.int64)
        lo, hi = min(lo, int(ref.min())), max(hi, int(ref.max()))
    x = np.arange(lo, hi + 1)
    panel = {
        "x": x.tolist(),
        "pp": round_list(np.bincount(pp - lo, minlength=x.size) / pp.size),
        "pp_q": round_list(np.quantile(pp, QQ_PROBS, method="inverted_cdf")),
    }
    if ref is not None:
        panel["ref"] = round_list(np.bincount(ref - lo, minlength=x.size) / ref.size)
        panel["ref_q"] = round_list(np.quantile(ref, QQ_PROBS, method="inverted_cdf"))
    else:
        panel["ref"] = round_list(pmf(x))
    return panel


def continuous_panel(pp, ref, log10):
    pp, ref = np.asarray(pp, dtype=float), np.asarray(ref, dtype=float)
    if log10:
        pp, ref = np.log10(pp[pp > 0]), np.log10(ref[ref > 0])
    lo, hi = min(pp.min(), ref.min()), max(pp.max(), ref.max())
    if not log10:
        hi = min(hi, max(np.quantile(pp, 0.9999), np.quantile(ref, 0.9999)))
    edges = np.linspace(lo, hi, 101)
    return {
        "log10": log10,
        "edges": round_list(edges),
        "pp": round_list(np.histogram(pp, edges, density=True)[0]),
        "ref": round_list(np.histogram(ref, edges, density=True)[0]),
        "pp_q": round_list(np.quantile(pp, QQ_PROBS)),
        "ref_q": round_list(np.quantile(ref, QQ_PROBS)),
    }


def distributions(key):
    out = {"poisson": [], "binomial": [], "gamma": [], "nbinomial": []}
    for lam in [0.0001, 0.1, 1.0, 4.0, 4.01, 8.0, 15, 19.9, 20.1, 25, 30, 100.0, 500.0]:
        key, k1, k2 = jax.random.split(key, 3)
        full = jnp.full((N_ACC,), lam, dtype=jnp.float32)
        out["poisson"].append(
            dict(
                label=f"λ = {lam:g}",
                **discrete_panel(
                    ppr.fast_poisson(k1, full), jax.random.poisson(k2, full)
                ),
            )
        )
    for n in [3, 20, 100, 2000]:
        for p in [0.02 / 365.25, 0.01, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99]:
            key, k1, k2 = jax.random.split(key, 3)
            nn = jnp.full((N_ACC,), n, dtype=jnp.int32)
            pp_arr = jnp.full((N_ACC,), p, dtype=jnp.float32)
            out["binomial"].append(
                dict(
                    label=f"n = {n}, p = {p:.3g}",
                    **discrete_panel(
                        ppr.fast_binomial(k1, nn, pp_arr),
                        jax.random.binomial(k2, nn, pp_arr),
                    ),
                )
            )
    for alpha in [0.01, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0]:
        key, k1, k2 = jax.random.split(key, 3)
        full = jnp.full((N_ACC,), alpha, dtype=jnp.float32)
        out["gamma"].append(
            dict(
                label=f"α = {alpha:g}",
                **continuous_panel(
                    ppr.fast_gamma(k1, full),
                    jax.random.gamma(k2, full),
                    log10=alpha < 0.1,
                ),
            )
        )
    for size in [1.0, 5.0, 20.0, 100.0]:
        for p in [0.1, 0.5, 0.9]:
            key, k1 = jax.random.split(key)
            s_arr = jnp.full((N_ACC,), size, dtype=jnp.float32)
            p_arr = jnp.full((N_ACC,), p, dtype=jnp.float32)
            panel = discrete_panel(
                ppr.fast_nbinomial(k1, s_arr, p=p_arr),
                pmf=lambda x, size=size, p=p: stats.nbinom.pmf(x, size, p),
            )
            panel["ref_q"] = round_list(stats.nbinom.ppf(QQ_PROBS, size, p))
            out["nbinomial"].append(dict(label=f"size = {size:g}, p = {p:g}", **panel))
    print("distributions done", flush=True)
    return out


t_start = time.time()
config = {"RUN_LEVEL": RUN_LEVEL, "PART": PART}
if PART == "speed":
    config.update(N_GRID=N_GRID, WORK=WORK)
    result = {"rows": speed()}
    path = f"results/speed_{PLATFORM}_rl{RUN_LEVEL}.json"
elif PART == "accuracy":
    config.update(N_ACC=N_ACC, GRID_POINTS=GRID_POINTS)
    k_w, k_m, k_d = jax.random.split(jax.random.key(200), 3)
    result = {
        "wasserstein": wasserstein(k_w),
        "miss_rates": miss_rates(k_m),
        "distributions": distributions(k_d),
        "qq_probs": QQ_PROBS.tolist(),
    }
    path = f"results/accuracy_rl{RUN_LEVEL}.json"
else:
    raise ValueError(f"unknown PART={PART!r}")

meta = run_metadata(config)
meta["platform"] = PLATFORM
meta["elapsed_s"] = round(time.time() - t_start, 1)
with open(path, "w") as f:
    json.dump(dict(meta=meta, **result), f)
print(f"wrote {path} in {meta['elapsed_s']} s", flush=True)
