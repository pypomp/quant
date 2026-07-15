# --- SLURM CONFIG ---
# sbatch_args:
#   job-name: "panel measles loglik comparison (R panelPomp)"
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
NP_EVAL <- switch(RUN_LEVEL, 2, 1000, 5000, 5000)
NREPS_EVAL <- switch(RUN_LEVEL, 2, 300, 300, 3600)

source("../panel_measles_shared.R")


# Load MLE parameters and build panel parameters
mle_params <- read.csv("../../../../measles/R_comparison/data/AK_mles.csv", stringsAsFactors = FALSE)
mles <- mle_params %>% filter(town %in% units)

shared_names <- c("R0", "sigma", "gamma", "sigmaSE", "cohort", "amplitude")
specific_names <- c("iota", "rho", "psi", "S_0", "E_0", "I_0", "R_0")

shared_params <- sapply(shared_names, function(p) mean(mles[[p]]))

specific_params_matrix <- matrix(
  NA,
  nrow = length(specific_names),
  ncol = length(units),
  dimnames = list(specific_names, units)
)
for (u in units) {
  unit_mle <- mles %>% filter(town == u)
  for (p in specific_names) {
    specific_params_matrix[p, u] <- unit_mle[[p]]
  }
}

# Assign to panelPomp object
shared(panel_obj) <- shared_params
specific(panel_obj) <- specific_params_matrix

# Setup Parallel
cores <- as.numeric(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = NA))
if (is.na(cores)) {
  cores <- detectCores()
  if (is.na(cores)) cores <- 1
}
registerDoParallel(cores)
registerDoRNG(594709947L)

# Run Panel Pfilter
print(sprintf("Running %d panel pfilters with Np = %d...", NREPS_EVAL, NP_EVAL))

t_pf_start <- Sys.time()
all_logliks <- foreach(
  i = 1:NREPS_EVAL,
  .packages = c("pomp", "panelPomp"),
  .combine = rbind,
  .options.multicore = list(set.seed = TRUE)
) %dopar% {
  pf_res <- pfilter(panel_obj, Np = NP_EVAL)
  ull <- unitLogLik(pf_res)
  data.frame(
    replicate = i,
    unit = names(ull),
    logLik = as.numeric(ull)
  )
}
t_pf_end <- Sys.time()
pf_time_total <- as.numeric(t_pf_end - t_pf_start, units = "secs")

dir.create("results", recursive = TRUE, showWarnings = FALSE)
saveRDS(all_logliks, "results/pfilter_logliks_f64.rds")
print("Saved results/pfilter_logliks_f64.rds")

# Save timing to CSV
timings_df <- data.frame(
  phase = c("pfilter"),
  time_seconds = c(pf_time_total)
)
write.csv(timings_df, "results/r_pomp_time.csv", row.names = FALSE)
print(sprintf("Pfilter Total Time: %.2f s", pf_time_total))
