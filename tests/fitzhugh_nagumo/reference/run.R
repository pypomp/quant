#!/usr/bin/env Rscript
# FHN: execute the pinned author simulators, not their ABC inference programs.

# --- SLURM CONFIG ---
# importance: medium
# description: "FHN: original author C++/R Strang simulator reference"
# tags: [reference, fitzhugh_nagumo, R, cpu]
# command: Rscript run.R
# sbatch_args:
#   job-name: "fhn R reference"
#   partition: standard
#   cpus-per-task: 1
#   mem: 6GB
#   time: "00:05:00"
#   output: "results/logs/slurm-%j.out"
# run_levels:
#   1:
#     sbatch_args: { time: "00:05:00" }
#   2:
#     sbatch_args: { time: "00:15:00" }
#   3:
#     sbatch_args: { time: "00:15:00" }
#   4:
#     sbatch_args: { time: "00:15:00" }
# --- END SLURM CONFIG ---

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
here <- dirname(normalizePath(script, mustWork = TRUE))
model_dir <- dirname(here)
tests_dir <- dirname(model_dir)
source(file.path(tests_dir, "utils.R"))
stopifnot(requireNamespace("jsonlite", quietly = TRUE))
level <- as.integer(Sys.getenv("RUN_LEVEL", "1"))
if (is.na(level) || !level %in% 1:4) stop("RUN_LEVEL must be 1, 2, 3, or 4")
cli <- commandArgs(trailingOnly = TRUE)
if (length(cli) && (length(cli) != 2L || cli[1] != "--output-dir")) {
  stop("Usage: run.R [--output-dir RESULTS_ROOT]")
}
results <- if (length(cli)) cli[2] else file.path(here, "results")
dir.create(results, recursive = TRUE, showWarnings = FALSE)
results <- normalizePath(results)
out_dir <- file.path(results, if (level == 1L) "smoke/R" else "R")

source(file.path(model_dir, "model.R"))
elapsed <- system.time(fhn_reference(
  sources_dir = file.path(model_dir, "data"), out_dir = out_dir,
  fixtures_only = level == 1L))

read_record <- function(name) jsonlite::fromJSON(file.path(out_dir, name), simplifyVector = FALSE)
reference <- read_record("reference_results.json")
fixtures <- read_record("fixtures.json")$scenarios
coupled <- do.call(rbind, lapply(fixtures, function(x) data.frame(
  name = x$name, dt = x$dt, n_steps = x$n_steps,
  authors_voltage_max_abs_difference = x$authors_voltage_max_abs_difference)))
endpoints <- data.frame(name = character(), dt = numeric(), path = integer(), V = numeric(), U = numeric())
if (level != 1L) {
  distributions <- read_record("distribution_samples.json")$scenarios
  endpoints <- do.call(rbind, lapply(distributions, function(x) {
    states <- do.call(rbind, lapply(x$endpoint_samples, unlist))
    data.frame(name = x$name, dt = x$dt, path = seq_len(nrow(states)) - 1L,
               V = states[, 1], U = states[, 2])
  }))
}

# utils.R obtains git metadata from cwd; the producing repository is explicit.
setwd(dirname(tests_dir))
meta <- save_run(out_dir = out_dir,
  tables = list(coupled.csv = coupled, endpoints.csv = endpoints,
                timings.csv = proc_time_frame(elapsed)),
  run_config = list(kind = "reference", model = "fitzhugh_nagumo", RUN_LEVEL = level,
    validation_scope = if (level == 1L) "coupled-only smoke" else "full author simulation",
    scope = "Strang simulator only; no likelihood, IF2, IFAD, ABC fit, or 2019 experiment reproduction.",
    n_paths_per_dt = if (level == 1L) 0L else 4096L,
    author_reference = reference$metadata, output_sha256 = reference$output_sha256))
meta$author_reference <- reference$metadata
meta$sources <- reference$sources
meta$Rcpp_version <- as.character(packageVersion("Rcpp"))
meta$jsonlite_version <- as.character(packageVersion("jsonlite"))
meta$reference_record_sha256 <- strsplit(system2("shasum", c("-a", "256",
  shQuote(file.path(out_dir, "reference_results.json"))), stdout = TRUE), " ")[[1]][1]
jsonlite::write_json(meta, file.path(out_dir, "latest.json"), auto_unbox = TRUE,
                     pretty = TRUE, null = "null", digits = NA)
cat(if (level == 1L) "Coupled-only smoke complete; no distribution checks.\n" else "Full author simulator reference complete.\n")
