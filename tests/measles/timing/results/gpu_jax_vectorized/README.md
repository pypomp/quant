# gpu_jax with a vectorized rproc (archived)

`model_001b_jax.py` from commit 9deadf7: rproc decorated with `@vectorized`, like
pypomp's 001b, with per-compartment `jax.random.binomial` exits instead of the
sequential multinomial. Same job config as `../gpu_jax` (level 4, London,
Np = 5000, 36 starts, 100 IF2 iterations, 36 reps), same GPU (RTX PRO 6000
Blackwell), pypomp 1.0.8, JAX 0.11.2.

| phase        | vmapped rproc (`../gpu_jax`, 1.0.6) | vectorized rproc (here, 1.0.8) |
|--------------|------------------------------------:|-------------------------------:|
| mif          | 1470 s                              | 2037 s (+39%)                  |
| pfilter_warm | 519 s                               | 733 s (+41%)                   |

With stock `jax.random` samplers the vectorized rproc is slower, while it is
faster with `pypomp.random`. The test was reverted to the vmapped rproc so
gpu vs gpu_jax compares each sampler set in its fastest form.
