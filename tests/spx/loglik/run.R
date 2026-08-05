#' SPX: R pomp baseline for the pfilter log-likelihood distribution.
#'
#' You should not normally need to run this; results/R/pfilter_logliks.csv
#' holds the committed result.

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
#   output: "results/R/logs/slurm-%j.out"
#   time: "01:00:00"
# setup: |
#   module load R/4.4.0
# --- END SLURM CONFIG ---

library(doParallel)
library(foreach)
library(doRNG)

source("../../utils.R")
source("../model.R")

cores <- as.numeric(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = NA))
if (is.na(cores)) {
  cores <- detectCores()
}
registerDoParallel(cores)
registerDoRNG(34118892)

run_level <- as.numeric(Sys.getenv("RUN_LEVEL", unset = "2"))

Np <- switch(run_level, 100, 1000, 1000, 1000)
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

save_run(
  out_dir = file.path("results", "R"),
  tables = list(
    pfilter_logliks.csv = data.frame(logLik = as.numeric(L.box)),
    timings.csv = proc_time_frame(t.box)
  ),
  run_config = list(
    kind = "loglik",
    model = "spx",
    RUN_LEVEL = run_level,
    Np = Np,
    Nreps_eval = Nreps_eval
  )
)
