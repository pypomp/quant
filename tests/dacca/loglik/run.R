#' Dhaka: R pomp baseline for the pfilter log-likelihood distribution.
#'
#' You should not normally need to run this; results/R/pfilter_logliks.csv
#' holds the committed result.

# --- SLURM CONFIG ---
# importance: low
# description: "Dhaka: R pomp baseline for the pfilter logLik distribution"
# tags: [loglik, dacca, r-pomp, cpu]
# sbatch_args:
#   job-name: "dacca loglik check (R)"
#   partition: standard
#   nodes: 1
#   ntasks-per-node: 36
#   cpus-per-task: 1
#   mem-per-cpu: 2GB
#   output: "results/R/logs/slurm-%j.out"
#   time: "02:00:00"
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
registerDoRNG(631409)
set.seed(631409)

run_level <- as.numeric(Sys.getenv("RUN_LEVEL", unset = "1"))

Np <- switch(run_level, 2, 1000, 5000, 5000)
Nreps_eval <- switch(run_level, 2, 300, 300, 3600)

cat(sprintf("Running at level %d\n", run_level))
cat(sprintf("Np: %d, Nreps_eval: %d\n", Np, Nreps_eval))

# dacca() carries the published MLE as its default params, so none are passed.
dacca_model <- dacca_obj()

dir.create(file.path("results", "R"), showWarnings = FALSE, recursive = TRUE)

stew(file = file.path("results", "R", "dacca_results_eval.rda"), {
  t_pfilter <- system.time({
    L.box <- foreach(
      i = 1:Nreps_eval,
      .packages = "pomp",
      .combine = c,
      .options.multicore = list(set.seed = TRUE)
    ) %dopar%
      {
        logLik(pfilter(dacca_model, Np = Np))
      }
  })
})

print(logmeanexp(L.box, se = TRUE))

save_run(
  out_dir = file.path("results", "R"),
  tables = list(
    pfilter_logliks.csv = data.frame(logLik = as.numeric(L.box)),
    timings.csv = proc_time_frame(t_pfilter)
  ),
  run_config = list(
    kind = "loglik",
    model = "dacca",
    RUN_LEVEL = run_level,
    Np = Np,
    Nreps_eval = Nreps_eval
  )
)
