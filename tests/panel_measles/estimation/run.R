#' Panel measles: distribution of block-IF2 parameter estimates using R
#' panelPomp. The baseline run.py is measured against, from the same committed
#' starting points.

# --- SLURM CONFIG ---
# importance: low
# description: "Panel measles: distribution of block-IF2 parameter estimates from a global search (R panelPomp)"
# tags: [estimation, panel_measles, r-pomp, cpu]
# sbatch_args:
#   job-name: "panel measles estimation (R panelPomp)"
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
#     sbatch_args: { time: "36:00:00" }
# setup: |
#   module load R/4.4.0
# command: |
#   R CMD BATCH --no-restore --no-save run.R results/R/logs/run.Rout
# --- END SLURM CONFIG ---

library(doParallel)
library(foreach)
library(doRNG)
library(dplyr)

source("../../utils.R")
source("../model.R")

run_level <- as.numeric(Sys.getenv("RUN_LEVEL", unset = "1"))
NP_FITR <- c(2, 500, 5000, 5000)[run_level]
NFITR <- c(2, 10, 100, 100)[run_level]
NSTARTS <- c(2, 3, 36, 360)[run_level]
NP_EVAL <- c(2, 1000, 5000, 5000)[run_level]
NREPS_EVAL <- c(2, 5, 36, 36)[run_level]

cores <- panel_cores()
registerDoParallel(cores)
registerDoRNG(PANEL_MAIN_SEED)

cat(sprintf("Running at level %d\n", run_level))
cat(sprintf(
  "Np: %d, Nmif: %d, Nstarts: %d, cores: %d\n",
  NP_FITR,
  NFITR,
  NSTARTS,
  cores
))

panel_obj <- panel_measles_objects()
starts <- panel_measles_starts(NSTARTS)

shared_params <- c("R0", "sigma", "gamma", "sigmaSE", "cohort", "amplitude")
specific_params <- c("iota", "rho", "psi", "S_0", "E_0", "I_0", "R_0")

dir.create("results/R/logs", recursive = TRUE, showWarnings = FALSE)

t_mif <- system.time({
  res_list <- foreach(
    i = 1:NSTARTS,
    .packages = c("pomp", "panelPomp"),
    .options.multicore = list(set.seed = TRUE)
  ) %dopar%
    {
      mif_out <- mif2(
        panel_obj,
        shared.start = starts[[i]]$shared,
        specific.start = starts[[i]]$specific,
        Np = NP_FITR,
        Nmif = NFITR,
        rw.sd = PANEL_RW_SD,
        cooling.fraction.50 = PANEL_COOLING_FRACTION_50,
        block = TRUE
      )

      cf <- coef(mif_out)
      coef_df <- data.frame(
        replicate = i,
        name = names(cf),
        value = as.numeric(cf)
      )

      tr_mat <- traces(mif_out)
      unit_traces <- list()

      df_shared <- data.frame(
        theta_idx = i - 1,
        unit = "shared",
        iteration = seq_len(nrow(tr_mat)) - 1L,
        method = "mif",
        logLik = if ("loglik" %in% colnames(tr_mat)) tr_mat[, "loglik"] else NA_real_,
        se = NA_real_
      )
      for (sp in shared_params) {
        if (sp %in% colnames(tr_mat)) {
          df_shared[[sp]] <- tr_mat[, sp]
        }
      }
      for (sp in specific_params) {
        df_shared[[sp]] <- NA_real_
      }
      unit_traces[["shared"]] <- df_shared

      for (u in PANEL_UNITS) {
        df_u <- data.frame(
          theta_idx = i - 1,
          unit = u,
          iteration = seq_len(nrow(tr_mat)) - 1L,
          method = "mif",
          logLik = if (paste0("loglik[", u, "]") %in% colnames(tr_mat)) tr_mat[, paste0("loglik[", u, "]")] else tr_mat[, "loglik"],
          se = NA_real_
        )
        for (sp in shared_params) {
          if (sp %in% colnames(tr_mat)) {
            df_u[[sp]] <- tr_mat[, sp]
          }
        }
        for (sp in specific_params) {
          colname <- paste0(sp, "[", u, "]")
          if (colname %in% colnames(tr_mat)) {
            df_u[[sp]] <- tr_mat[, colname]
          }
        }
        unit_traces[[u]] <- df_u
      }
      trace_df <- do.call(rbind, unit_traces)

      pf_rows <- list()
      for (j in 1:NREPS_EVAL) {
        ull <- unitLogLik(pfilter(mif_out, Np = NP_EVAL))
        pf_rows[[j]] <- data.frame(
          theta_idx = i - 1,
          unit = names(ull),
          replicate = j,
          logLik = as.numeric(ull)
        )
      }
      pfilter_df <- do.call(rbind, pf_rows)

      list(coefs = coef_df, traces = trace_df, logliks = pfilter_df)
    }

  all_coefs <- do.call(rbind, lapply(res_list, function(x) x$coefs))
  all_traces <- do.call(rbind, lapply(res_list, function(x) x$traces))
  all_logliks <- do.call(rbind, lapply(res_list, function(x) x$logliks))
})

all_coefs <- all_coefs %>%
  mutate(
    is_shared = !grepl("\\[", name),
    param = if_else(is_shared, name, sub("\\[.*\\]", "", name)),
    unit = if_else(is_shared, "shared", sub(".*\\[(.*)\\]", "\\1", name))
  ) %>%
  select(replicate, unit, param, value)

save_run(
  out_dir = file.path("results", "R"),
  tables = list(
    mif_coefs.csv = all_coefs,
    mif_traces.csv.gz = all_traces,
    pfilter_logliks.csv = all_logliks,
    timings.csv = proc_time_frame(t_mif)
  ),
  run_config = list(
    kind = "estimation",
    model = "panel_measles",
    RUN_LEVEL = run_level,
    NP_FITR = NP_FITR,
    NFITR = NFITR,
    NSTARTS = NSTARTS,
    NP_EVAL = NP_EVAL,
    NREPS_EVAL = NREPS_EVAL,
    units = PANEL_UNITS
  ),
  raw = all_coefs,
  raw_name = "mif_coefs.rds"
)
