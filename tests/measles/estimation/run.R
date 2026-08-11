#' Measles: distribution of IF2 parameter estimates from a global search, using
#' R pomp. The baseline that run.py is compared against.

# --- SLURM CONFIG ---
# importance: low
# description: "Measles: distribution of IF2 parameter estimates from a global search (R pomp)"
# tags: [estimation, measles, r-pomp, cpu]
# sbatch_args:
#   job-name: "measles estimation (R)"
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
#     sbatch_args: { time: "04:00:00" }
#   4:
#     sbatch_args: { time: "18:00:00" }
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
NP_FITR <- c(2, 500, 1000, 5000)[run_level]
NFITR <- c(2, 10, 100, 100)[run_level]
NSTARTS <- c(2, 3, 20, 360)[run_level]

cores <- measles_cores()
registerDoParallel(cores)
registerDoRNG(MEASLES_MAIN_SEED)

cat(sprintf("Running at level %d\n", run_level))
cat(sprintf(
  "Np: %d, Nmif: %d, Nstarts: %d, cores: %d\n",
  NP_FITR,
  NFITR,
  NSTARTS,
  cores
))

pomp_objects <- measles_objects()
starts <- measles_starts(NSTARTS)

dir.create("results/R/logs", recursive = TRUE, showWarnings = FALSE)

t_mif <- system.time({
  all_coefs <- list()
  all_traces <- list()

  for (unit_name in MEASLES_UNITS) {
    cat(sprintf("unit %s: %d starts\n", unit_name, NSTARTS))
    pomp_obj <- pomp_objects[[unit_name]]

    unit_mifs <- foreach(
      i = 1:NSTARTS,
      .packages = "pomp",
      .combine = list,
      .multicombine = TRUE,
      .options.multicore = list(set.seed = TRUE)
    ) %dopar%
      {
        mif2(
          pomp_obj,
          params = unlist(starts[i, ]),
          Np = NP_FITR,
          Nmif = NFITR,
          rw.sd = MEASLES_RW_SD,
          cooling.fraction.50 = MEASLES_COOLING_FRACTION_50
        )
      }

    unit_coefs <- unlist(lapply(unit_mifs, coef))
    all_coefs[[unit_name]] <- data.frame(
      unit = unit_name,
      replicate = rep(1:NSTARTS, each = length(MEASLES_PARAM_NAMES)),
      coef = unit_coefs,
      names = names(unit_coefs)
    )

    unit_traces_list <- lapply(seq_along(unit_mifs), function(i) {
      tr <- as.data.frame(traces(unit_mifs[[i]]))
      tr$iteration <- seq_len(nrow(tr)) - 1L
      tr$replicate <- i
      tr$unit <- unit_name
      tr
    })
    all_traces[[unit_name]] <- do.call(rbind, unit_traces_list)
  }

  mif_coefs <- do.call(rbind, all_coefs)
  mif_traces <- do.call(rbind, all_traces)
})

save_run(
  out_dir = file.path("results", "R"),
  tables = list(
    mif_coefs.csv = mif_coefs,
    mif_traces.csv.gz = mif_traces,
    timings.csv = proc_time_frame(t_mif)
  ),
  run_config = list(
    kind = "estimation",
    model = "measles",
    RUN_LEVEL = run_level,
    NP_FITR = NP_FITR,
    NFITR = NFITR,
    NSTARTS = NSTARTS,
    UNITS = MEASLES_UNITS
  ),
  raw = mif_coefs,
  raw_name = "mif_coefs.rds"
)
