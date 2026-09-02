#' Panel measles: wall-clock timing of block-IF2 and pfilter using R
#' panelPomp. The baseline the pypomp configurations in run.py are measured
#' against.
#'
#' R has no JIT step, so its single pfilter timing is the warm one; the report
#' shows it against both of pypomp's.

# --- SLURM CONFIG ---
# importance: low
# description: "Panel measles: wall-clock timing of block-IF2 and pfilter (R panelPomp)"
# tags: [timing, panel_measles, r-pomp, cpu]
# sbatch_args:
#   job-name: "panel measles timing (R panelPomp)"
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
#     sbatch_args: { time: "03:30:00" }
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
NP_FITR <- c(2, 500, 5000, 5000)[run_level]
NFITR <- c(2, 10, 100, 100)[run_level]
NSTARTS <- c(2, 3, 36, 36)[run_level]
NP_EVAL <- c(2, 1000, 5000, 5000)[run_level]
NREPS_EVAL <- c(2, 5, 36, 36)[run_level]

cores <- panel_cores()
registerDoParallel(cores)
registerDoRNG(PANEL_MAIN_SEED)

cat(sprintf("Running at level %d\n", run_level))
cat(sprintf(
  "Np: %d, Nmif: %d, Nstarts: %d, Nreps: %d, cores: %d\n",
  NP_FITR,
  NFITR,
  NSTARTS,
  NREPS_EVAL,
  cores
))

panel_obj <- panel_measles_objects()
starts <- panel_measles_starts(NSTARTS)

cat("\nPhase 1: timing mif...\n")
t_mif <- system.time({
  mif_out <- foreach(
    i = 1:NSTARTS,
    .packages = c("pomp", "panelPomp"),
    .options.multicore = list(set.seed = TRUE)
  ) %dopar%
    {
      mif2(
        panel_obj,
        shared.start = starts[[i]]$shared,
        specific.start = starts[[i]]$specific,
        Np = NP_FITR,
        Nmif = NFITR,
        rw.sd = PANEL_RW_SD,
        cooling.fraction.50 = PANEL_COOLING_FRACTION_50,
        block = TRUE
      )
    }
})

cat("Phase 2: timing pfilter...\n")
t_pf <- system.time({
  foreach(
    i = 1:NREPS_EVAL,
    .packages = c("pomp", "panelPomp"),
    .options.multicore = list(set.seed = TRUE)
  ) %dopar%
    {
      pfilter(mif_out[[1]], Np = NP_EVAL)
    }
})

timings_df <- data.frame(
  phase = c("mif", "pfilter_warm"),
  time_seconds = c(as.numeric(t_mif["elapsed"]), as.numeric(t_pf["elapsed"]))
)

cat(sprintf(
  "\nmif: %.2f s | pfilter: %.2f s\n",
  timings_df$time_seconds[1],
  timings_df$time_seconds[2]
))

save_run(
  out_dir = file.path("results", "R"),
  tables = list(timings.csv = timings_df),
  run_config = list(
    kind = "timing",
    model = "panel_measles",
    RUN_LEVEL = run_level,
    NP_FITR = NP_FITR,
    NFITR = NFITR,
    NSTARTS = NSTARTS,
    NP_EVAL = NP_EVAL,
    NREPS_EVAL = NREPS_EVAL,
    units = PANEL_UNITS
  )
)
