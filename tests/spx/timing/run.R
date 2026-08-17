#' SPX: wall-clock timing of mif and pfilter using R pomp.

# --- SLURM CONFIG ---
# importance: low
# description: "SPX: wall-clock timing of mif and pfilter (R pomp)"
# tags: [timing, spx, r-pomp, cpu]
# sbatch_args:
#   job-name: "spx timing (R)"
#   partition: standard
#   nodes: 1
#   ntasks-per-node: 36
#   cpus-per-task: 1
#   mem-per-cpu: 2GB
#   output: "results/R/logs/slurm-%j.out"
#   time: "00:20:00"
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

run_level <- as.numeric(Sys.getenv("RUN_LEVEL", unset = "1"))

Np <- switch(run_level, 2, 1000, 1000, 1000)
Nmif <- switch(run_level, 2, 20, 50, 200)
Nstarts <- switch(run_level, 2, 3, 3, 36)
Nreps_eval <- switch(run_level, 2, 3, 3, 24)

sp500.filt <- spx_filt()
global_starts <- spx_starts(Nstarts)

t_mif <- system.time({
  if.box <- foreach(
    i = 1:Nstarts,
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

t_pf <- system.time({
  L.box <- foreach(
    i = 1:Nstarts,
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
      )
    }
})

mif_sec <- as.numeric(t_mif["elapsed"])
pf_sec <- as.numeric(t_pf["elapsed"])

timings_df <- data.frame(
  phase = c("mif", "pfilter_warm"),
  time_seconds = c(mif_sec, pf_sec)
)

save_run(
  out_dir = file.path("results", "R"),
  tables = list(timings.csv = timings_df),
  run_config = list(
    kind = "timing",
    model = "spx",
    RUN_LEVEL = run_level,
    Np = Np,
    Nmif = Nmif,
    Nstarts = Nstarts,
    Nreps_eval = Nreps_eval
  )
)
