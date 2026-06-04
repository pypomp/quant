# quant: Quantitative tests of pypomp

We are concerned with accuracy tests to make sure that the code gives correct answers (up to small Monte Carlo error) in situations where this is knowable.
We are also concerned with performance benchmark tests. 
This can involve measuring time, memory requirement, or iterations to convergence for a maximization algorithm.
Put together, we call these __quantitative tests__, or simply __quant tests__, to distinguish them from unit tests.
Our unit tests, in [pypomp:pypomp/test](https://github.com/pypomp/pypomp/tree/main/test), are quick tests designed to check that the code is not broken, and to ensure we are told if numerical results change.
The quant tests sit in their own repository, since they are not necessarily run often.
Some of them are of an exploratory nature, such as code profiling investigations.

The quant tests also provide additional examples of pypomp, focused on technical issues that extend the
 simpler examples in [pypomp:tutorials](https://pypomp.github.io/tutorials)

## Existing Quantitative Tests

Below is a list of quantitative test reports available in this repository:

### 1. Dacca Cholera Model
* **[Dacca Report (Local Search)](tests/dacca/local_search/report.html)** (`tests/dacca/local_search`): Compares Iterated Filtering (IF2 on GPU) and Iterated Filtering with Automatic Differentiation (IFAD/train) on a local parameter search.
* **[Dacca Report](tests/dacca/performance/report.html)** (`tests/dacca/performance`): Analyzes the performance, runtime, and parameter convergence of IF2 versus IFAD for the Dacca cholera model.

### 2. Measles Model
* **[LogLik Comparison: Pypomp vs R](tests/measles/R_comparison/report.html)** (`tests/measles/R_comparison`): Verifies log-likelihood evaluation accuracy by comparing `pypomp` (both exact and approximate implementations) against R's `pomp` package.
* **[Parameter Comparison: Pypomp vs R](tests/measles/R_comparison/parameter_comparison/report.html)** (`tests/measles/R_comparison/parameter_comparison`): Compares the parameter estimates obtained via `pypomp` versus R's `pomp`.
* **[Continuous Measles Report](tests/measles/continuous/report.html)** (`tests/measles/continuous`): Compares parameter and log-likelihood traces, densities, and runtimes of IF2 and IFAD on a continuous measles model.
* **[Measles Report](tests/measles/performance/report.html)** (`tests/measles/performance`): Benchmarks parameter estimation via Iterated Filtering (IF2) on a discrete measles model.

### 3. Panel Measles Model (Multi-Unit / Spatiotemporal)
* **[Continuous Panel Measles Report](tests/panel_measles/continuous/report.html)** (`tests/panel_measles/continuous`): Evaluates multi-unit panel model fitting comparing IF2 and IFAD on a continuous panel measles model.
* **[Panel Measles Report](tests/panel_measles/performance/report.html)** (`tests/panel_measles/performance`): Benchmarks spatiotemporal/panel parameter estimation via IF2 on a panel measles dataset.

### 4. Random Number Generators (RNG)
* **[Random Number Generators Benchmark & Comparison](tests/samplers/test.html)** (`tests/samplers`): Benchmarks the execution speed and validates the statistical accuracy of `pypomp`'s fast approximate inverse CDF samplers (`fast_poisson`, `fast_binomial`, `fast_gamma`, `fast_nbinomial`) against `jax.random` and `scipy.stats`.

### 5. SPX (S&P 500) Model
* **[SPX Report](tests/spx/report.html)** (`tests/spx`): Compares parameter estimation traces and log-likelihood estimates on the SPX index dataset using `pypomp` (CPU/GPU) and R's `pomp`.
