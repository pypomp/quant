#' SIR: ABC-MCMC posterior over (beta1, rho) using R pomp, down the same
#' warm-started tolerance ladder as run.py.
#'
#' Each chain runs the ladder in order, starting each rung where the previous
#' one ended. All chains and rungs go into one results/R/traces.csv.gz with an
#' `epsilon` column.

# --- SLURM CONFIG ---
# importance: medium
# description: "SIR: ABC-MCMC posterior over (beta1, rho) down a tolerance ladder (R pomp)"
# tags: [bayesian, sir, abc, r-pomp, cpu]
# sbatch_args:
#   job-name: "bayesian abc (R)"
#   partition: standard
#   nodes: 1
#   ntasks-per-node: 36
#   cpus-per-task: 1
#   mem-per-cpu: 2GB
#   output: "results/R/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "00:10:00" }
#   2:
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "00:45:00" }
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
NCHAINS <- c(2, 4, 36, 36)[run_level]
NABC <- c(20, 1000, 20000, 40000)[run_level]
TRACE_THIN <- c(1, 1, 10, 20)[run_level]

# Must match ABC_EPS_LADDER in model.py.
EPS_LADDER <- c(1e6, 4.0, 2.5, 1.5)
if (run_level == 1) EPS_LADDER <- EPS_LADDER[1:2]

cat("run level", run_level, ": chains", NCHAINS, "Nabc", NABC, "thin", TRACE_THIN, "\n")
cat("epsilon ladder:", paste(EPS_LADDER, collapse = ", "), "\n")

cores <- as.integer(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = "2"))
registerDoParallel(cores = cores)

obj <- bayes_sir()
starts <- bayes_starts(NCHAINS)

scale_df <- read.csv(BAYES_SCALE_PATH)
scale_vec <- setNames(scale_df$scale, scale_df$probe)
cat(
  "probe scale:",
  paste(names(scale_vec), signif(scale_vec, 4), sep = "=", collapse = ", "),
  "\n"
)

pb_obs <- probe(obj, probes = bayes_probes(), nsim = 10)
probe_values <- data.frame(
  probe = names(bayes_probes()),
  value = as.numeric(pb_obs@datvals)
)
print(probe_values)

#' Starts for the next rung: candidates from this rung's post-burn-in draws,
#' kept if one fresh simulation is within the next tolerance. Mirrors
#' next_starts() in run.py.
next_starts <- function(rung, eps) {
  post <- rung[rung$iteration > NABC * BURN_FRAC, BAYES_FREE]
  cand <- post[sample.int(nrow(post), NCHAINS * CAND_FACTOR, replace = TRUE), ]
  dist <- bayes_abc_distance(obj, cand, scale_vec)
  passed <- cand[dist < eps^2, , drop = FALSE]
  cat("  starts for eps", eps, ":", nrow(passed), "of", nrow(cand), "candidates pass\n")
  if (nrow(passed) == 0) passed <- cand[order(dist)[seq_len(NCHAINS)], ]
  passed[sample.int(nrow(passed), NCHAINS, replace = nrow(passed) < NCHAINS), ]
}

CAND_FACTOR <- 4
BURN_FRAC <- 0.5

t_start <- proc.time()

rungs <- list()
for (eps in EPS_LADDER) {
  rung <- foreach(
    i = seq_len(NCHAINS),
    .packages = c("pomp"),
    .combine = rbind
  ) %dorng% {
    p <- coef(obj)
    for (nm in BAYES_FREE) p[[nm]] <- starts[[nm]][i]
    fit <- abc(
      obj,
      Nabc = NABC,
      probes = bayes_probes(),
      scale = scale_vec,
      epsilon = eps,
      params = p,
      proposal = mvn_diag_rw(BAYES_ABC_RW_SD)
    )
    tr <- as.data.frame(traces(fit))
    tr$chain <- i
    tr$epsilon <- eps
    tr$iteration <- seq_len(nrow(tr)) - 1L
    tr$acceptance_rate <- fit@accepts / NABC
    tr
  }
  cat("eps", eps, "done at", (proc.time() - t_start)[["elapsed"]], "s\n")
  rungs[[length(rungs) + 1]] <- rung
  next_eps <- EPS_LADDER[match(eps, EPS_LADDER) + 1]
  if (!is.na(next_eps)) starts <- next_starts(rung, next_eps)
}
chains <- do.call(rbind, rungs)

elapsed <- proc.time() - t_start
cat("elapsed:", elapsed[["elapsed"]], "seconds\n")

keep <- c("chain", "epsilon", "iteration", BAYES_FREE, "acceptance_rate")
traces_df <- chains[, intersect(keep, colnames(chains))]

acceptance_df <- unique(traces_df[, c("chain", "epsilon", "acceptance_rate")])
acceptance_df$Nabc <- NABC
cat("chains that never moved, by epsilon:\n")
print(tapply(acceptance_df$acceptance_rate == 0, acceptance_df$epsilon, sum))

traces_df <- traces_df[traces_df$iteration %% TRACE_THIN == 0, ]

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
    epsilon_ladder = EPS_LADDER,
    trace_thin = TRACE_THIN,
    probes = names(scale_vec),
    probe_scale = as.list(scale_vec),
    free_params = BAYES_FREE,
    rw_sd_natural_scale = as.list(BAYES_ABC_RW_SD),
    execution_time = elapsed[["elapsed"]]
  )
)

cat("done\n")
