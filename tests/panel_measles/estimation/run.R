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

source("../../utils.R")
source("../model.R")

run_level <- as.numeric(Sys.getenv("RUN_LEVEL", unset = "1"))
NP_FITR <- c(2, 500, 5000, 5000)[run_level]
NFITR <- c(2, 10, 100, 100)[run_level]
NSTARTS <- c(2, 3, 36, 360)[run_level]

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

all_coefs <- foreach(
  i = 1:NSTARTS,
  .packages = c("pomp", "panelPomp"),
  .combine = rbind,
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
    data.frame(
      replicate = i,
      name = names(cf),
      value = as.numeric(cf)
    )
  }

# panelPomp names a unit-specific coefficient "param[unit]" and a shared one
# just "param"; split that back into the columns the report joins on.
all_coefs <- all_coefs %>%
  mutate(
    is_shared = !grepl("\\[", name),
    param = if_else(is_shared, name, sub("\\[.*\\]", "", name)),
    unit = if_else(is_shared, "shared", sub(".*\\[(.*)\\]", "\\1", name))
  ) %>%
  select(replicate, unit, param, value)

save_run(
  out_dir = file.path("results", "R"),
  tables = list(mif_coefs.csv = all_coefs),
  run_config = list(
    kind = "estimation",
    model = "panel_measles",
    RUN_LEVEL = run_level,
    NP_FITR = NP_FITR,
    NFITR = NFITR,
    NSTARTS = NSTARTS,
    units = PANEL_UNITS
  ),
  raw = all_coefs,
  raw_name = "mif_coefs.rds"
)
