#' SPX: R pomp baseline for the distribution of IF2 parameter estimates.
#'
#' Kind: estimation. The R counterpart of run.py, and the source of the frozen
#' baseline in R_reference/.
#'
#' You should not normally need to run this. The frozen CSVs in R_reference/
#' are what the report reads, and they only need regenerating when pomp
#' changes -- this job cost 7183 s elapsed / 70 CPU-hours the last time it ran.
#' After running it, re-freeze with:
#'     python scripts/freeze_r_results.py freeze --only spx/estimation

# --- SLURM CONFIG ---
# importance: low
# description: "SPX: R pomp baseline for IF2 parameter and likelihood estimates"
# tags: [estimation, spx, r-pomp, cpu]
# sbatch_args:
#   job-name: "spx estimation (R)"
#   partition: standard
#   nodes: 1
#   ntasks-per-node: 36
#   cpus-per-task: 1
#   mem-per-cpu: 2GB
#   output: "R_results/logs/slurm-%j.out"
#   time: "03:00:00"
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

run_level <- as.numeric(Sys.getenv("RUN_LEVEL", unset = "1"))

Np <- switch(run_level, 100, 200, 500, 1000)
Nmif <- switch(run_level, 10, 25, 50, 200)
Nreps_eval <- switch(run_level, 4, 7, 10, 24)
Nreps_global <- switch(run_level, 10, 15, 20, 120 * 3)

sp500.filt <- spx_filt()
global_starts <- spx_starts(Nreps_global)

dir.create("R_results", showWarnings = FALSE, recursive = TRUE)

stew(file = "R_results/spx_results.rda", {
  t.box <- system.time({
    t.if.box <- system.time({
      if.box <- foreach(
        i = 1:Nreps_global,
        .packages = "pomp",
        .combine = c,
        .options.multicore = list(set.seed = TRUE)
      ) %dopar%
        {
          mif2(
            sp500.filt,
            Nmif = Nmif,
            rw.sd = SPX_RW_SD,
            cooling.fraction.50 = SPX_COOLING_FRACTION_50,
            Np = Np,
            params = unlist(global_starts[i, ])
          )
        }
    })
    t.L.box <- system.time({
      L.box <- foreach(
        i = 1:Nreps_global,
        .packages = "pomp",
        .combine = rbind,
        .options.multicore = list(set.seed = TRUE)
      ) %dopar%
        {
          replicate(
            Nreps_eval,
            logLik(pfilter(
              sp500.filt,
              params = coef(if.box[[i]]),
              Np = Np
            ))
          ) |>
            logmeanexp(se = TRUE)
        }
    })
  })
})
