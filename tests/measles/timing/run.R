#' Measles: wall-clock timing of mif and pfilter using R pomp. The baseline the
#' pypomp configurations in run.py are measured against.

# --- SLURM CONFIG ---
# importance: high
# description: "Measles: wall-clock timing of mif and pfilter (R pomp)"
# tags: [timing, measles, r-pomp, cpu]
# sbatch_args:
#   job-name: "measles timing (R)"
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
#     sbatch_args: { time: "04:00:00" }
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
NP <- c(2, 500, 5000, 5000)[run_level]
NFITR <- c(2, 10, 100, 100)[run_level]
NSTARTS <- c(2, 3, 36, 36)[run_level]
NREPS <- c(2, 3, 36, 36)[run_level]

UNIT <- "London"

cores <- measles_cores()
registerDoParallel(cores)
registerDoRNG(MEASLES_MAIN_SEED)

cat(sprintf("Running at level %d\n", run_level))
cat(sprintf(
  "Np: %d, Nmif: %d, Nstarts: %d, Nreps: %d, cores: %d\n",
  NP,
  NFITR,
  NSTARTS,
  NREPS,
  cores
))

measles_obj <- measles_objects(UNIT)[[UNIT]]
starts <- measles_starts(NSTARTS)

dir.create("results/R/logs", recursive = TRUE, showWarnings = FALSE)

t_mif <- system.time({
  mifs <- foreach(
    i = 1:NSTARTS,
    .packages = "pomp",
    .options.multicore = list(set.seed = TRUE)
  ) %dopar%
    {
      mif2(
        measles_obj,
        params = unlist(starts[i, ]),
        Np = NP,
        Nmif = NFITR,
        rw.sd = MEASLES_RW_SD,
        cooling.fraction.50 = MEASLES_COOLING_FRACTION_50
      )
    }
})

# NREPS filter evaluations per starting point, matching reps = NREPS in run.py.
t_pf <- system.time({
  pf_logliks <- foreach(
    idx = 1:(NSTARTS * NREPS),
    .packages = "pomp",
    .combine = rbind,
    .options.multicore = list(set.seed = TRUE)
  ) %dopar%
    {
      rep_id <- (idx - 1) %/% NREPS + 1
      data.frame(
        replicate = rep_id,
        logLik = logLik(pfilter(mifs[[rep_id]], Np = NP))
      )
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

# The estimates the timed work actually produced, so the report can check that
# the faster configurations are arriving at the same answer.
coefs <- do.call(rbind, lapply(seq_len(NSTARTS), function(i) {
  cf <- coef(mifs[[i]])
  data.frame(replicate = i, names = names(cf), coef = as.numeric(cf))
}))

ll_summary <- aggregate(logLik ~ replicate, data = pf_logliks, FUN = mean)
names(ll_summary)[2] <- "mean_logLik"
ll_sd <- aggregate(logLik ~ replicate, data = pf_logliks, FUN = sd)
ll_summary$sd_logLik <- ll_sd$logLik

results_df <- merge(coefs, ll_summary, by = "replicate")
results_df$unit <- UNIT

save_run(
  out_dir = file.path("results", "R"),
  tables = list(
    timings.csv = timings_df,
    results.csv = results_df
  ),
  run_config = list(
    kind = "timing",
    model = "measles",
    RUN_LEVEL = run_level,
    UNIT = UNIT,
    NP = NP,
    NFITR = NFITR,
    NSTARTS = NSTARTS,
    NREPS = NREPS
  )
)
