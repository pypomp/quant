import pypomp as pp
import jax
from timeit import timeit
from datetime import datetime

now = datetime.now()
print("DATE: ", now.date())
print("TIME: ", now.time())


pulls = 10000 * 36

key = jax.random.key(123)
param_key, time_key = jax.random.split(key)
many_keys = jax.random.split(key, pulls)

print("--------------------------------")
##--- POISSON SAMPLER ---##
lams = jax.random.uniform(param_key, shape=(pulls,), minval=0, maxval=50)

### Warmup
pp.random.fast_approx_rpoisson(time_key, lam=lams)
jax.random.poisson(time_key, lam=lams)

### Benchmarking
time_taken1 = timeit(
    lambda: pp.random.fast_approx_rpoisson(time_key, lam=lams),
    number=100,
)
print(f"fast approx poisson time taken: {time_taken1} seconds")

time_taken2 = timeit(lambda: jax.random.poisson(time_key, lam=lams), number=100)
print(f"jax.random.poisson time taken: {time_taken2} seconds")

print(
    f"ratio of jax.random.poisson to fast approx poisson: {time_taken2 / time_taken1}"
)
print("--------------------------------")

##--- GAMMA SAMPLER ---##
alphas = jax.random.uniform(param_key, shape=(pulls,), minval=0, maxval=50)

### Warmup
pp.random.fast_approx_rgamma(time_key, alpha=alphas)
jax.random.gamma(time_key, a=alphas)

### Benchmarking
time_taken1 = timeit(
    lambda: pp.random.fast_approx_rgamma(time_key, alpha=alphas),
    number=100,
)
print(f"fast approx gamma time taken: {time_taken1} seconds")

time_taken2 = timeit(
    lambda: jax.random.gamma(time_key, a=alphas),
    number=100,
)
print(f"jax.random.gamma time taken: {time_taken2} seconds")

print(f"ratio of jax.random.gamma to fast approx gamma: {time_taken2 / time_taken1}")
print("--------------------------------")

##--- BINOMIAL SAMPLER ---##
ns = (
    jax.random.uniform(param_key, shape=(pulls,), minval=0, maxval=50)
    .round()
    .astype(int)
)
ps = jax.random.uniform(param_key, shape=(pulls,), minval=0, maxval=1)

### Warmup
pp.random.fast_approx_rbinom(time_key, n=ns, p=ps)
jax.random.binomial(time_key, n=ns, p=ps)

### Benchmarking
time_taken1 = timeit(
    lambda: pp.random.fast_approx_rbinom(time_key, n=ns, p=ps),
    number=100,
)
print(f"fast approx binomial time taken: {time_taken1} seconds")

time_taken2 = timeit(
    lambda: jax.random.binomial(time_key, n=ns, p=ps),
    number=100,
)
print(f"jax.random.binomial time taken: {time_taken2} seconds")

print(
    f"ratio of jax.random.binomial to fast approx binomial: {time_taken2 / time_taken1}"
)
