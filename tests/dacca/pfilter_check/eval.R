# --- SLURM CONFIG ---
# importance: low
# sbatch_args:
#   job-name: "dacca pfilter check (R pomp)"
#   partition: standard
#   nodes: 1
#   ntasks-per-node: 36
#   cpus-per-task: 1
#   mem-per-cpu: 2GB
#   output: "results/R/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "02:00:00" }
# setup: |
#   module load R/4.4.0
# command: |
#   R CMD BATCH --no-restore --no-save eval.R results/R/logs/eval.Rout
# --- END SLURM CONFIG ---

#' This script estimates the distribution of the estimated logLik at the default parameter value.
#' The results are meant to be compared against the Python version of the script.

library(pomp)
library(doParallel)
library(foreach)
library(doRNG)

source("../../utils.R")

# Create results directories
dir.create("results/R/logs", recursive = TRUE, showWarnings = FALSE)

cores <- as.numeric(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = NA))
run_level <- as.numeric(Sys.getenv("RUN_LEVEL", unset = "1"))

if (is.na(cores)) {
  cores <- detectCores()
}
registerDoParallel(cores)
registerDoRNG(631409)
set.seed(631409)

NP_EVAL <- c(2, 1000, 5000, 5000)[run_level]
NREPS_EVAL <- c(2, 300, 300, 3600)[run_level]

cat(sprintf("Running at level %d\n", run_level))
cat(sprintf("NP_EVAL: %d, NREPS_EVAL: %d\n", NP_EVAL, NREPS_EVAL))

# Load the dacca model
dacca_obj <- dacca()

start_time <- Sys.time()
stew(file = "R_results/dacca_results_eval.rda", {
  t_pfilter <- system.time({
    L_box <- foreach(
      i = 1:NREPS_EVAL,
      .packages = "pomp",
      .combine = c,
      .options.multicore = list(set.seed = TRUE)
    ) %dopar%
      {
        logLik(pfilter(dacca_obj, Np = NP_EVAL))
      }
  })
})
end_time <- Sys.time()
cat(sprintf(
  "pfilter time taken: %f seconds\n",
  as.numeric(difftime(end_time, start_time, units = "secs"))
))

res <- logmeanexp(L_box, se = TRUE)
coefs <- coef(dacca_obj)

results_df <- data.frame(
  logLik = res["est"],
  se = res["se"],
  t(coefs)
)
rownames(results_df) <- "0"

print(results_df)

save_run(
  out_dir = file.path("results", "R"),
  tables = list(
    pfilter_logliks.csv = data.frame(logLik = as.numeric(L_box)),
    timings.csv = proc_time_frame(t_pfilter)
  ),
  run_config = list(
    kind = "loglik",
    model = "dacca",
    RUN_LEVEL = run_level,
    NP_EVAL = NP_EVAL,
    NREPS_EVAL = NREPS_EVAL
  )
)
