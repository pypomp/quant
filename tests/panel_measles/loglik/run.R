#' Panel measles: distribution of the pfilter log-likelihood at the He et al.
#' estimates using R panelPomp. The baseline run.py is measured against.

# --- SLURM CONFIG ---
# importance: low
# description: "Panel measles: distribution of pfilter logLik at the He et al. (2010) estimates (R panelPomp)"
# tags: [loglik, panel_measles, r-pomp, cpu]
# sbatch_args:
#   job-name: "panel measles loglik (R panelPomp)"
#   partition: standard
#   nodes: 1
#   ntasks-per-node: 36
#   cpus-per-task: 1
#   mem-per-cpu: 2GB
#   output: "results/R/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "00:05:00" }
#   2:
#     sbatch_args: { time: "00:30:00" }
#   3:
#     sbatch_args: { time: "02:00:00" }
#   4:
#     sbatch_args: { time: "03:00:00" }
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

cores <- panel_cores()
registerDoParallel(cores)
registerDoRNG(PANEL_MAIN_SEED)

cat(sprintf("Running at level %d\n", run_level))
cat(sprintf("Np: %d, Nreps: %d, cores: %d\n", NP_EVAL, NREPS_EVAL, cores))

panel_obj <- panel_measles_objects()
mles <- panel_measles_mles()
shared(panel_obj) <- mles$shared
specific(panel_obj) <- mles$specific

t_pf <- system.time({
  all_logliks <- foreach(
    i = 1:NREPS_EVAL,
    .packages = c("pomp", "panelPomp"),
    .combine = rbind,
    .options.multicore = list(set.seed = TRUE)
  ) %dopar%
    {
      ull <- unitLogLik(pfilter(panel_obj, Np = NP_EVAL))
      data.frame(
        unit = names(ull),
        replicate = i,
        logLik = as.numeric(ull)
      )
    }
})

timings_df <- data.frame(
  phase = "pfilter",
  time_seconds = as.numeric(t_pf["elapsed"])
)

cat(sprintf("\npfilter: %.2f s\n", timings_df$time_seconds[1]))

save_run(
  out_dir = file.path("results", "R"),
  tables = list(
    pfilter_logliks.csv = all_logliks,
    timings.csv = timings_df
  ),
  run_config = list(
    kind = "loglik",
    model = "panel_measles",
    RUN_LEVEL = run_level,
    NP_EVAL = NP_EVAL,
    NREPS_EVAL = NREPS_EVAL,
    units = PANEL_UNITS
  )
)
