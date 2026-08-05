#' SPX: R pomp baseline for the distribution of IF2 parameter estimates.
#'
#' Only needs regenerating when pomp changes; it writes results/R/ itself.

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
#   output: "results/R/logs/slurm-%j.out"
#   time: "03:00:00"
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

# L.box is a Nreps_global x 2 matrix of (est, se).
logliks <- as.data.frame(L.box)
colnames(logliks) <- c("logLik", "se")
logliks$replicate <- seq_len(nrow(logliks))
logliks <- logliks[, c("replicate", "logLik", "se")]

traces_df <- do.call(rbind, lapply(seq_along(if.box), function(i) {
  tr <- as.data.frame(traces(if.box[[i]]))
  tr$iteration <- seq_len(nrow(tr)) - 1L # iteration 0 is the starting value
  tr$replicate <- i
  tr
}))
id_cols <- c("replicate", "iteration")
traces_df <- traces_df[, c(id_cols, setdiff(colnames(traces_df), id_cols))]

proc_rows <- function(tv, label) {
  data.frame(
    stage = label,
    metric = names(tv),
    seconds = as.numeric(tv),
    stringsAsFactors = FALSE
  )
}
timings <- rbind(
  proc_rows(t.if.box, "mif"),
  proc_rows(t.L.box, "pfilter"),
  proc_rows(t.box, "total")
)

save_run(
  out_dir = file.path("results", "R"),
  tables = list(
    pfilter_logliks.csv = logliks,
    mif_traces.csv.gz = traces_df,
    timings.csv = timings
  ),
  run_config = list(
    kind = "estimation",
    model = "spx",
    RUN_LEVEL = run_level,
    Np = Np,
    Nmif = Nmif,
    Nreps_eval = Nreps_eval,
    Nreps_global = Nreps_global
  )
)
