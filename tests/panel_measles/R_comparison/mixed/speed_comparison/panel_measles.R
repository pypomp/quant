# --- SLURM CONFIG ---
# sbatch_args:
#   job-name: "panel measles speed comparison (R panelPomp)"
#   partition: standard
#   nodes: 1
#   ntasks-per-node: 36
#   cpus-per-task: 1
#   mem-per-cpu: 2GB
#   output: "results/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "00:02:00" }
#   2:
#     sbatch_args: { time: "00:30:00" }
#   3:
#     sbatch_args: { time: "02:00:00" }
#   4:
#     sbatch_args: { time: "08:00:00" }
# setup: |
#   module load R/4.4.0
# command: |
#   R CMD BATCH --no-restore --no-save panel_measles.R results/logs/panel_measles.Rout
# --- END SLURM CONFIG ---

stopifnot(getRversion() >= "4.1")
library(tidyverse)
library(pomp)
library(panelPomp)
library(doParallel)
library(foreach)
library(doRNG)

RUN_LEVEL <- as.numeric(Sys.getenv("RUN_LEVEL", unset = 1))
NP_FITR <- switch(RUN_LEVEL, 2, 500, 5000, 5000)
NFITR <- switch(RUN_LEVEL, 2, 10, 100, 100)
NREPS_FITR <- switch(RUN_LEVEL, 2, 3, 36, 36)
NP_EVAL <- switch(RUN_LEVEL, 2, 1000, 5000, 5000)
NREPS_EVAL <- switch(RUN_LEVEL, 2, 5, 36, 36)

source("../panel_measles_shared.R")


# Setup Parallel
cores <- as.numeric(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = NA))
if (is.na(cores)) {
  cores <- detectCores()
  if (is.na(cores)) cores <- 1
}
registerDoParallel(cores)
registerDoRNG(594709947L)

# Read starting parameters from Python
starting_parameters <- read.csv(
  "../parameter_comparison/starting_parameters.csv",
  stringsAsFactors = FALSE
)

shared_names <- c("R0", "sigma", "gamma", "sigmaSE", "cohort", "amplitude")
specific_names <- c("iota", "rho", "psi", "S_0", "E_0", "I_0", "R_0")

DEFAULT_SD <- 0.02
IVP_DEFAULT_SD <- DEFAULT_SD * 12
INITIAL_RW_SD <- rw_sd(
  S_0 = ivp(IVP_DEFAULT_SD),
  E_0 = ivp(IVP_DEFAULT_SD),
  I_0 = ivp(IVP_DEFAULT_SD),
  R_0 = ivp(IVP_DEFAULT_SD),
  R0 = DEFAULT_SD * 0.25,
  sigma = DEFAULT_SD * 0.25,
  gamma = DEFAULT_SD * 0.5,
  iota = DEFAULT_SD,
  rho = DEFAULT_SD * 0.5,
  sigmaSE = DEFAULT_SD,
  psi = DEFAULT_SD * 0.25,
  cohort = DEFAULT_SD * 0.5,
  amplitude = DEFAULT_SD * 0.5
)

# --- RUN BENCHMARK ---
cat("\nPhase 1: Timing MIF...\n")
t_mif_start <- Sys.time()

mif_out <- foreach(
  i = 1:NREPS_FITR,
  .packages = c("pomp", "panelPomp"),
  .options.multicore = list(set.seed = TRUE)
) %dopar%
  {
    rep_data <- starting_parameters %>% filter(replicate == (i - 1))

    # Extract shared parameters for starting point
    shared_data <- rep_data %>% filter(unit == "shared")
    shared_start <- shared_data$value
    names(shared_start) <- shared_data$param

    # Extract unit-specific parameters for starting point
    specific_data <- rep_data %>% filter(unit != "shared")
    specific_start <- matrix(
      NA,
      nrow = length(specific_names),
      ncol = length(units),
      dimnames = list(specific_names, units)
    )
    for (u in units) {
      for (p in specific_names) {
        specific_start[p, u] <- specific_data %>%
          filter(unit == u, param == p) %>%
          pull(value)
      }
    }

    # Run mif2
    mif2(
      panel_obj,
      shared.start = shared_start,
      specific.start = specific_start,
      Np = NP_FITR,
      Nmif = NFITR,
      rw.sd = INITIAL_RW_SD,
      cooling.fraction.50 = 0.5,
      block = TRUE
    )
  }

t_mif_end <- Sys.time()
mif_time_total <- as.numeric(t_mif_end - t_mif_start, units = "secs")

cat("Phase 2: Timing Pfilter...\n")
t_pf_start <- Sys.time()

# Run particle filter on the first mif result across replicates
pf_out <- foreach(
  i = 1:NREPS_EVAL,
  .packages = c("pomp", "panelPomp"),
  .options.multicore = list(set.seed = TRUE)
) %dopar%
  {
    # Take the first mif result parameter set for pfilter timing
    pfilter(mif_out[[1]], Np = NP_EVAL)
  }

t_pf_end <- Sys.time()
pf_time_total <- as.numeric(t_pf_end - t_pf_start, units = "secs")

# Save results
dir.create("results", recursive = TRUE, showWarnings = FALSE)
timings_df <- data.frame(
  phase = c("mif", "pfilter"),
  time_seconds = c(mif_time_total, pf_time_total)
)
write.csv(timings_df, "results/r_pomp_timings.csv", row.names = FALSE)

print(sprintf("MIF Total Time:    %.2f s", mif_time_total))
print(sprintf("Pfilter Total Time: %.2f s", pf_time_total))
