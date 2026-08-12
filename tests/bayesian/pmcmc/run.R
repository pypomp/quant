#' SIR: PMCMC posterior over (beta1, rho) using R pomp. The baseline that
#' run.py is compared against.
#'
#' This is the only leg of the suite sensitive to errors in the SIR model
#' translation -- the grid reference runs pypomp's own filter on pypomp's own
#' model and so cannot see them. A disagreement here is ambiguous between a
#' model error and a sampler error until the pfilter-logLik precondition in
#' report.qmd separates the two, which is why that section comes first.
#'
#' The proposal is NOT identical to pypomp's and cannot be: pypomp runs its
#' chain on the estimation scale, pomp perturbs params directly. See the note on
#' BAYES_RW_SD in ../model.R. Both are valid Metropolis-Hastings kernels for the
#' same posterior; only the stationary distributions are compared.

# --- SLURM CONFIG ---
# importance: medium
# description: "SIR: PMCMC posterior over (beta1, rho) (R pomp)"
# tags: [bayesian, sir, pmcmc, r-pomp, cpu]
# sbatch_args:
#   job-name: "bayesian pmcmc (R)"
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
#     sbatch_args: { time: "00:20:00" }
#   3:
#     sbatch_args: { time: "01:30:00" }
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
NCHAINS <- c(2, 4, 8, 12)[run_level]
NMCMC <- c(20, 2000, 20000, 50000)[run_level]
NP <- c(5, 100, 500, 500)[run_level]

cat("run level", run_level, ": chains", NCHAINS, "Nmcmc", NMCMC, "Np", NP, "\n")

cores <- as.integer(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = "2"))
registerDoParallel(cores = cores)
cat("registered", cores, "workers\n")

obj <- bayes_sir()
starts <- bayes_starts(NCHAINS)

#' The precondition check: pfilter logLik at the true theta. run.py computes the
#' same quantity, and report.qmd compares the two distributions before anything
#' else. If they disagree beyond Monte Carlo error the SIR translation is wrong
#' and no posterior comparison downstream is interpretable.
NP_PRECOND <- c(10, 500, 2000, 2000)[run_level]
NREPS_PRECOND <- c(2, 12, 24, 24)[run_level]

precond <- foreach(
  i = seq_len(NREPS_PRECOND),
  .packages = c("pomp"),
  .combine = rbind
) %dorng% {
  pf <- pfilter(obj, Np = NP_PRECOND, params = coef(obj))
  data.frame(replicate = i, logLik = logLik(pf), J = NP_PRECOND)
}
cat("precondition pfilter at truth: mean logLik", mean(precond$logLik), "\n")

t_start <- proc.time()

chains <- foreach(
  i = seq_len(NCHAINS),
  .packages = c("pomp"),
  .combine = rbind
) %dorng% {
  p <- coef(obj)
  for (nm in BAYES_FREE) p[[nm]] <- starts[[nm]][i]

  fit <- pmcmc(
    obj,
    Nmcmc = NMCMC,
    Np = NP,
    params = p,
    proposal = mvn_diag_rw(BAYES_RW_SD)
  )

  tr <- as.data.frame(traces(fit))
  tr$chain <- i
  tr$iteration <- seq_len(nrow(tr)) - 1L
  tr$acceptance_rate <- fit@accepts / NMCMC
  tr
}

elapsed <- proc.time() - t_start
cat("elapsed:", elapsed[["elapsed"]], "seconds\n")

keep <- c("chain", "iteration", "loglik", "log.prior", BAYES_FREE, "acceptance_rate")
keep <- intersect(keep, colnames(chains))
traces_df <- chains[, keep]
names(traces_df)[names(traces_df) == "loglik"] <- "logLik"
names(traces_df)[names(traces_df) == "log.prior"] <- "log_prior"

acceptance_df <- unique(traces_df[, c("chain", "acceptance_rate")])
acceptance_df$Np <- NP
acceptance_df$Nmcmc <- NMCMC

save_run(
  out_dir = file.path("results", "R"),
  tables = list(
    traces.csv.gz = traces_df,
    acceptance.csv = acceptance_df,
    pfilter_logliks.csv = precond,
    timings.csv = proc_time_frame(elapsed)
  ),
  run_config = list(
    kind = "pmcmc",
    model = "sir",
    RUN_LEVEL = run_level,
    NCHAINS = NCHAINS,
    Nmcmc = NMCMC,
    Np = NP,
    free_params = BAYES_FREE,
    rw_sd_natural_scale = as.list(BAYES_RW_SD),
    execution_time = elapsed[["elapsed"]]
  )
)

cat("done\n")
