#' Measles: distribution of the pfilter log-likelihood at the He et al. estimates,
#' using R pomp. The baseline that run.py is compared against.

# --- SLURM CONFIG ---
# importance: low
# description: "Measles: distribution of pfilter logLik at the He et al. (2010) estimates (R pomp)"
# tags: [loglik, measles, r-pomp, cpu]
# sbatch_args:
#   job-name: "measles loglik (R)"
#   partition: standard
#   nodes: 1
#   ntasks-per-node: 36
#   cpus-per-task: 1
#   mem-per-cpu: 2GB
#   output: "results/R/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "00:02:00" }
#   2:
#     sbatch_args: { time: "00:30:00" }
#   3:
#     sbatch_args: { time: "01:00:00" }
#   4:
#     sbatch_args: { time: "01:30:00" }
# setup: |
#   module load R/4.4.0
# command: |
#   R CMD BATCH --no-restore --no-save run.R results/R/logs/run.Rout
# --- END SLURM CONFIG ---

library(doParallel)
library(foreach)
library(doRNG)

source("../../utils.R")
source("../model.R")

run_level <- as.numeric(Sys.getenv("RUN_LEVEL", unset = "1"))
NP_EVAL <- c(2, 1000, 5000, 5000)[run_level]
NREPS_EVAL <- c(2, 300, 300, 3600)[run_level]

cores <- measles_cores()
registerDoParallel(cores)
registerDoRNG(MEASLES_MAIN_SEED)

cat(sprintf("Running at level %d\n", run_level))
cat(sprintf(
  "Np: %d, Nreps: %d, cores: %d\n",
  NP_EVAL,
  NREPS_EVAL,
  cores
))

pomp_objects <- measles_objects()

dir.create("results/R/logs", recursive = TRUE, showWarnings = FALSE)

t_pf <- system.time({
  all_logliks <- list()

  for (unit_name in MEASLES_UNITS) {
    cat(sprintf(
      "unit %s: %d pfilters with Np = %d\n",
      unit_name,
      NREPS_EVAL,
      NP_EVAL
    ))
    pomp_obj <- pomp_objects[[unit_name]]
    unit_params <- measles_mle(unit_name)

    unit_logliks <- foreach(
      i = 1:NREPS_EVAL,
      .packages = "pomp",
      .combine = c,
      .options.multicore = list(set.seed = TRUE)
    ) %dopar%
      {
        logLik(pfilter(pomp_obj, params = unit_params, Np = NP_EVAL))
      }

    all_logliks[[unit_name]] <- data.frame(
      unit = unit_name,
      replicate = 1:NREPS_EVAL,
      logLik = unit_logliks
    )

    cat(sprintf("  mean logLik %.2f\n", mean(unit_logliks)))
  }

  pfilter_logliks <- do.call(rbind, all_logliks)
})

save_run(
  out_dir = file.path("results", "R"),
  tables = list(
    pfilter_logliks.csv = pfilter_logliks,
    timings.csv = proc_time_frame(t_pf)
  ),
  run_config = list(
    kind = "loglik",
    model = "measles",
    RUN_LEVEL = run_level,
    NP_EVAL = NP_EVAL,
    NREPS_EVAL = NREPS_EVAL,
    UNITS = MEASLES_UNITS
  ),
  raw = pfilter_logliks,
  raw_name = "pfilter_logliks.rds"
)
