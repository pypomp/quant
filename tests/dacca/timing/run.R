#' Dhaka: wall-clock timing of mif and pfilter using R pomp.

# --- SLURM CONFIG ---
# importance: high
# description: "Dhaka: wall-clock timing of mif and pfilter (R pomp)"
# tags: [timing, dacca, r-pomp, cpu]
# sbatch_args:
#   job-name: "dacca timing (R)"
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
#     sbatch_args: { time: "01:00:00" }
#   4:
#     sbatch_args: { time: "02:00:00" }
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
registerDoRNG(631409L)

run_level <- as.numeric(Sys.getenv("RUN_LEVEL", unset = "1"))

Np <- switch(run_level, 2, 500, 5000, 5000)
Nmif <- switch(run_level, 2, 10, 100, 100)
Nstarts <- switch(run_level, 2, 3, 36, 36)
Nreps <- switch(run_level, 2, 3, 36, 36)

cat(sprintf("Running at level %d\n", run_level))
cat(sprintf(
  "Np: %d, Nmif: %d, Nstarts: %d, Nreps: %d, cores: %d\n",
  Np,
  Nmif,
  Nstarts,
  Nreps,
  cores
))

dacca_model <- dacca_obj()
starts <- dacca_starts(Nstarts)

t_mif <- system.time({
  if.box <- foreach(
    i = 1:Nstarts,
    .packages = "pomp",
    .combine = c,
    .options.multicore = list(set.seed = TRUE)
  ) %dopar%
    {
      mif2(
        dacca_model,
        params = unlist(starts[i, ]),
        Np = Np,
        Nmif = Nmif,
        cooling.fraction.50 = DACCA_COOLING_FRACTION_50,
        rw.sd = DACCA_RW_SD
      )
    }
})

# Nreps filter evaluations per starting point, matching reps = Nreps in run.py.
t_pf <- system.time({
  L.box <- foreach(
    i = 1:Nstarts,
    .packages = "pomp",
    .combine = c,
    .options.multicore = list(set.seed = TRUE)
  ) %dopar%
    {
      replicate(
        Nreps,
        logLik(pfilter(dacca_model, params = unlist(starts[i, ]), Np = Np))
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

save_run(
  out_dir = file.path("results", "R"),
  tables = list(timings.csv = timings_df),
  run_config = list(
    kind = "timing",
    model = "dacca",
    RUN_LEVEL = run_level,
    NP = Np,
    NFITR = Nmif,
    NSTARTS = Nstarts,
    NREPS = Nreps
  )
)
