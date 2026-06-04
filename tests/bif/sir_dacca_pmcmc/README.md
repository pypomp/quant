# BIF versus PMCMC benchmarks

This quantitative test compares BIF against PMCMC on two small but useful
benchmarks:

* a four-parameter SIR model with `T=100`;
* the first 100 observations of the Dacca cholera model, fitting four
  parameters.

The benchmark follows the quant-test layout used by
`tests/dpop/london_measles/`.

* `bif_pmcmc_test.py` runs the SIR and Dacca BIF/PMCMC scripts from the BIF
  paper repository.
* `sync_results.py` copies the small report-ready summaries and figures from a
  completed BIF run into this quant directory.
* `report.qmd` summarizes runtime, posterior comparisons, and diagnostics.

Large PMCMC traces and BIF sample clouds remain in the BIF paper repository.
The quant report stores only compact CSV/JSON summaries and presentation
figures.
