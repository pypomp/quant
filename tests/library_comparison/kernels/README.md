# Where kernel work would speed pypomp up

Measurements on one RTX 3090 (kevtan-linux), JAX 0.12. The CPU was busy during these
runs (load average about 40 on 32 threads), which slows kernel launches, so the
small-J numbers are probably inflated, more so for the XLA paths that launch more kernels.

## He10: per-step overhead dominates at the paper's workload sizes

`results/xla_overhead_experiments.jsonl` (from `../bench_he10.py`), microseconds per daily
Euler step for one London filter (or 36 filters):

| Particles in flight | stock | `fori_loop(unroll=True)` | XLA command buffers |
|---|---|---|---|
| 1,000 | 55 | 52 | 57 |
| 10,000 | 50 | 97 | 60 |
| 100,000 | 98 | 100 | 115 |
| 36 x 1,000 | 63 | 155 | 65 |

Time per step is flat up to about 36,000 particles in flight: the GPU is waiting on
kernel launches (about 15-20 kernels per Euler step in the compiled HLO). Unrolling the
substep loop makes He10 slower and raises compile time from about 18 s to 95-140 s;
command buffers make no measurable difference.

## A fused propagation kernel (Pallas, Triton backend)

`pallas_sde_bench.py` runs the same bootstrap filter for the toy SDE (T = 100, 10 Euler
steps per observation) with the propagation compiled three ways: pypomp's XLA
`fori_loop`, the same loop unrolled, and one Pallas kernel per observation interval
that runs all substeps for a block of particles. All three use the same hashed RNG
and the same dmeasure, normalization and systematic resampler, and give bit-identical
log-likelihoods on the GPU. Wall time per filter run (`results/pallas_sde_gpu.jsonl`):

| R filters x J | XLA loop | XLA unrolled | Pallas (best block) | Pallas speedup over XLA loop |
|---|---|---|---|---|
| 1 x 1,024 | 9.1 ms | 5.2 ms | 3.8 ms | 2.4x |
| 1 x 16,384 | 17.4 ms | 5.4 ms | 5.8 ms | 3.0x |
| 1 x 131,072 | 17.6 ms | 8.4 ms | 7.3 ms | 2.4x |
| 1 x 1,048,576 | 32.1 ms | 35.2 ms | 21.4 ms | 1.5x |
| 100 x 1,024 | 38.9 ms | 6.0 ms | 5.9 ms | 6.5x |
| 36 x 4,096 | 20.2 ms | 9.3 ms | 7.1 ms | 2.9x |

For this cheap step, unrolling gets most of the way, unlike He10. With the propagation
fused, the remaining time at small J is the per-observation XLA code (dmeasure,
normalization, resampling, gather).

## Can the kernel host an unmodified user rproc? (`pallas_probe.py`)

`results/pallas_probe_lowering.json`: lowering and compiling each rproc, whole and one
primitive at a time, with the Pallas Triton backend.

- `jax.random` does not lower (`random_wrap` has no Pallas Triton rule), so a fused kernel
  needs its own counter-based RNG.
- Toy SDE: with RNG replaced by an integer hash, the whole interval kernel compiles; all
  of its other primitives lower, including `erf_inv`.
- He10: 36 of 55 primitives lower. The rest fail because Triton needs power-of-2 array
  sizes (the step draws an (11, J) block of uniforms; the gamma sampler multiplies 3 rows),
  because slicing an in-kernel array (`u[:4]`) has no lowering, and because `polynomial`
  (`jnp.polyval` in `pypomp.random`), `optimization_barrier` and an integer-predicate
  `select_n` have no lowering rules.

## CuTe: the user's rproc, unmodified, in one kernel per interval

`cute_translate.py` walks the jaxpr of a single-particle rproc step inside a CuTe DSL
kernel: one thread per particle, and small per-particle arrays (He10's 11 uniforms, the
gamma sampler's 3 rows) become registers, so none of Triton's shape rules apply.
`cute_rproc.cute_struct(model)` swaps the model's `rprocess_interp` for that kernel, so
pypomp's own `pfilter` (dmeasure, weights, resampling, key schedule) runs around it, and
all filters of a vmapped call go into one launch. Random numbers are pypomp's threefry
stream, computed per thread: `split(key, J)[j]` and element `(s, j)` of a `(k, J)` draw.
Unsupported primitives raise with their name so a caller can fall back to XLA. The
substep loop is a runtime loop, which on He10 compiles in about 12 s instead of 74 s.

Same keys, one observation interval from the same particles
(`results/cute_vs_stock_one_interval.txt`): the toy SDE matches stock pypomp to 9e-7
relative; He10 matches to 1e-7 except for 1 particle in 2,048 at the first interval (a
binomial draw that rounds to the other side of a CDF step). pypomp's He10 rproc and its
`binominv`, `poissoninv`, `gammainv` and `ndtri` run as written.

Full pfilter with the same keys, warm (`results/cute_vs_stock_gpu.jsonl`):

| Model | R filters x J | stock pypomp | CuTe kernel | speedup | compile (stock / CuTe) |
|---|---|---|---|---|---|
| He10 London | 1 x 5,120 | 221 ms | 97 ms | 2.3x | 16 s / 11 s |
| He10 London | 36 x 5,120 | 1.56 s | 0.46 s | 3.4x | 31 s / 17 s |
| He10 London | 1,296 x 5,120 | 46.8 s | 14.8 s | 3.2x | 74 s / 30 s |
| toy SDE | 1 x 1,024 | 22 ms | 18 ms | 1.2x | |
| toy SDE | 1 x 16,384 | 20 ms | 12 ms | 1.7x | |
| toy SDE | 100 x 1,024 | 26 ms | 17 ms | 1.5x | |
| toy SDE | 1 x 1,048,576 | 49 ms | 45 ms | 1.1x | |

The CPU was saturated during these runs (load average 72-88 on 32 threads).

## He10 Pallas prototype (`he10_fused.py`)

The He10 step with its uniforms drawn as 11 hashed streams, the gamma product written
out, and `ndtri`, `polyval` and `lax.cond` in `pypomp.random` replaced, so Pallas Triton
can compile it; Pallas Triton also needed its scalar-constant lowering patched. Not a
drop-in (different RNG), but the same model: warm times (`results/he10_pallas_gpu.jsonl`)
were 64 ms (1 x 5,120), 0.29 s (36 x 5,120) and 9.6 s (1,296 x 5,120), against stock's
218 ms, 1.56 s and 46.7 s. It is faster than the CuTe drop-in mainly because a hash is
much cheaper than threefry; the CuTe kernel keeps pypomp's exact random numbers.

## Replicate layout (pypomp branch `pfilter-vmap-reps`, `layout_bench.py`)

`Pomp.pfilter(reps=...)` runs the replicates of each parameter set one after another.
The branch runs them in parallel by default (`reps_batch_size=None`; 1 restores the old
behaviour). On the paper's PF workloads, warm (`results/pfilter_layout_gpu.jsonl`):

| Model | old (sequential reps) | new (parallel reps) |
|---|---|---|
| S&P 500 (36 x 24 x 1,000) | 4.0 s | 0.75 s |
| Cholera (36 x 36 x 5,000) | 9.9 s | 7.8 s |
| London measles (36 x 36 x 5,000) | 46.0 s | 39.8 s |

Run from this folder's parent: `python kernels/pallas_sde_bench.py [--interpret] [--check]`,
`python kernels/pallas_probe.py` (needs a visible GPU; `--run` also executes),
`python kernels/cute_check.py {sde,he10} R:J ...` (needs `nvidia-cutlass-dsl`), and
`PYTHONPATH=<pypomp checkout on pfilter-vmap-reps> python kernels/layout_bench.py`.
