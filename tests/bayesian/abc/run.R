#' SIR: ABC posterior over (beta1, rho) using R pomp, swept over epsilon.
#'
#' For ABC this is the primary correctness check, not corroboration. ABC has no
#' exact target of its own -- its limit is the posterior given the probes, which
#' the grid reference does not describe -- so agreement with a mature
#' independent implementation at identical probes, scale, epsilon, prior and
#' data is the strongest available statement.
#'
#' All chains and all epsilons are written into one results/R/traces.csv.gz with
#' an `epsilon` column, rather than the per-sweep subdirectories run.py uses.

# --- SLURM CONFIG ---
# importance: high
# description: "SIR: ABC posterior over (beta1, rho) swept over epsilon (R pomp)"
# tags: [bayesian, sir, abc, r-pomp, cpu]
# sbatch_args:
#   job-name: "bayesian abc (R)"
#   partition: standard
#   nodes: 1
#   ntasks-per-node: 12
#   cpus-per-task: 1
#   mem-per-cpu: 2GB
#   output: "results/R/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "00:10:00" }
#   2:
#     sbatch_args: { time: "00:30:00" }
#   3:
#     sbatch_args: { time: "01:30:00" }
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
NCHAINS <- c(2, 4, 8, 12)[run_level]
NABC <- c(20, 5000, 50000, 200000)[run_level]

# Must match EPS_GRID in run.py: report.qmd compares the two arm by arm.
EPS_GRID <- list(
  c(1e6),
  c(1e6, 20.0, 5.0, 2.0),
  c(1e6, 20.0, 5.0, 2.0),
  c(1e6, 20.0, 5.0, 2.0)
)[[run_level]]

cat("run level", run_level, ": chains", NCHAINS, "Nabc", NABC, "\n")
cat("epsilon grid:", paste(EPS_GRID, collapse = ", "), "\n")

cores <- as.integer(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = "2"))
registerDoParallel(cores = cores)

obj <- bayes_sir()
starts <- bayes_starts(NCHAINS)

scale_df <- read.csv(BAYES_SCALE_PATH)
scale_vec <- setNames(scale_df$scale, scale_df$probe)
cat("probe scale:", paste(names(scale_vec), signif(scale_vec, 4),
                          sep = "=", collapse = ", "), "\n")

#' The precondition check: pomp's own probe values on the observed series.
#' run.py writes the same three numbers computed in pure JAX, and report.qmd
#' compares them before any ABC output is interpreted.
pb_obs <- probe(obj, probes = bayes_probes(), nsim = 50)
probe_values <- data.frame(
  probe = names(bayes_probes()),
  value = as.numeric(pb_obs@datvals)
)
cat("probe values on the data:\n")
print(probe_values)

t_start <- proc.time()

grid <- expand.grid(chain = seq_len(NCHAINS), epsilon = EPS_GRID)

chains <- foreach(
  row = seq_len(nrow(grid)),
  .packages = c("pomp"),
  .combine = rbind
) %dorng% {
  i <- grid$chain[row]
  eps <- grid$epsilon[row]

  p <- coef(obj)
  for (nm in BAYES_FREE) p[[nm]] <- starts[[nm]][i]

  fit <- abc(
    obj,
    Nabc = NABC,
    probes = bayes_probes(),
    scale = scale_vec,
    epsilon = eps,
    params = p,
    proposal = mvn_diag_rw(BAYES_RW_SD)
  )

  tr <- as.data.frame(traces(fit))
  tr$chain <- i
  tr$epsilon <- eps
  tr$iteration <- seq_len(nrow(tr)) - 1L
  tr$acceptance_rate <- fit@accepts / NABC
  tr
}

elapsed <- proc.time() - t_start
cat("elapsed:", elapsed[["elapsed"]], "seconds\n")

keep <- c("chain", "epsilon", "iteration", BAYES_FREE, "acceptance_rate")
traces_df <- chains[, intersect(keep, colnames(chains))]

acceptance_df <- unique(traces_df[, c("chain", "epsilon", "acceptance_rate")])
acceptance_df$Nabc <- NABC

save_run(
  out_dir = file.path("results", "R"),
  tables = list(
    traces.csv.gz = traces_df,
    acceptance.csv = acceptance_df,
    probe_values.csv = probe_values,
    timings.csv = proc_time_frame(elapsed)
  ),
  run_config = list(
    kind = "abc",
    model = "sir",
    RUN_LEVEL = run_level,
    NCHAINS = NCHAINS,
    Nabc = NABC,
    epsilon_grid = EPS_GRID,
    probes = names(scale_vec),
    probe_scale = as.list(scale_vec),
    free_params = BAYES_FREE,
    rw_sd_natural_scale = as.list(BAYES_RW_SD),
    execution_time = elapsed[["elapsed"]]
  )
)

cat("done\n")
