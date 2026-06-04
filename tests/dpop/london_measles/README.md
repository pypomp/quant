# DPOP London measles benchmark

This quantitative test compares a DPOP global search against an IF2 baseline
on the single-unit London measles model. It follows the report organization
used in the Dacca IFAD/DMOP benchmarks:

* `prep.py` builds the shared London `PanelPomp` object and common tuning.
* `mif_test.py` runs the standalone IF2 baseline.
* `dpop_test.py` runs an IF2 warm start followed by DPOP training.
* `report.qmd` summarizes likelihood results, runtimes, and trace behavior.

The stored CSV summaries in `plot_exports/` are small enough to keep under
version control. Large pickled result objects are written to `mif_results/` and
`dpop_results/`; these are ignored by the repository by default.

The main level-4 comparison currently uses 100 starting points, `J=5000`, and
36 pfilter evaluation replicates. Two DPOP schedules are recorded:

* `DPOP-80+80`: 80 warm-start IF2 iterations and 80 DPOP iterations.
* `DPOP-175+175`: 175 warm-start IF2 iterations and 175 DPOP iterations.

The standalone IF2 reference uses 650 IF2 iterations.
