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

Run from this folder's parent: `python kernels/pallas_sde_bench.py [--interpret] [--check]`
and `python kernels/pallas_probe.py` (needs a visible GPU; `--run` also executes).
