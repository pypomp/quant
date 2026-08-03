#' SPX: R pomp baseline for the pfilter log-likelihood distribution.
#'
#' Kind: loglik. Evaluates the particle filter repeatedly at a fixed theta (the
#' Sun 2024 estimates) to characterise the distribution of the likelihood
#' estimate. The R counterpart of run.py.
#'
#' You should not normally need to run this; R_reference/pfilter_logliks.csv
#' holds the frozen result. Re-freeze after running with:
#'     python scripts/freeze_r_results.py freeze --only spx/loglik
#'
#' Note: the previous version of this script (pfilter_check/eval.R) read
#' "data/SPX.csv" and wrote to "spx/pfilter_check/R_results/...", both of which
#' assumed the repository root as the working directory -- but the runner cd's
#' into the script's own directory. It also read Sys.getenv("run_level") in
#' lower case while the runner exports RUN_LEVEL. Those are fixed here.

# --- SLURM CONFIG ---
# importance: low
# description: "SPX: R pomp baseline for the pfilter logLik distribution"
# tags: [loglik, spx, r-pomp, cpu]
# sbatch_args:
#   job-name: "spx loglik check (R)"
#   partition: standard
#   nodes: 1
#   ntasks-per-node: 36
#   cpus-per-task: 1
#   mem-per-cpu: 2GB
#   output: "R_results/logs/slurm-%j.out"
#   time: "01:00:00"
# setup: |
#   module load R/4.4.0
# --- END SLURM CONFIG ---

library(doParallel)
library(foreach)
library(doRNG)

source("../model.R")

cores <- as.numeric(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = NA))
if (is.na(cores)) {
  cores <- detectCores()
}
registerDoParallel(cores)
registerDoRNG(34118892)

run_level <- as.numeric(Sys.getenv("RUN_LEVEL", unset = "2"))

Np <- switch(run_level, 100, 1000, 1000, 1000)
# Level 3 is the routine setting; level 4 is the archival 3600-replicate run
# that the committed R_reference baseline came from.
Nreps_eval <- switch(run_level, 4, 100, 225, 3600)

sp500.filt <- spx_filt()

dir.create("R_results", showWarnings = FALSE, recursive = TRUE)

stew(file = "R_results/spx_results_eval.rda", {
  t.box <- system.time({
    L.box <- foreach(
      i = 1:Nreps_eval,
      .packages = "pomp",
      .combine = rbind,
      .options.multicore = list(set.seed = TRUE)
    ) %dopar%
      {
        logLik(pfilter(sp500.filt, params = SPX_SUN2024_THETA, Np = Np))
      }
  })
})
