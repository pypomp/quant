#' Blowflies1: R pomp baseline at the fixed default parameter vector.
#' Uses the same run levels as run.py; no optimization is performed.

# --- SLURM CONFIG ---
# importance: low
# description: "Blowflies1: R pomp fixed-parameter likelihood baseline"
# tags: [loglik, blowflies, r-pomp, cpu]
# command: "Rscript run.R"
# sbatch_args:
#   partition: standard
#   cpus-per-task: 1
#   mem: 6GB
#   time: "00:05:00"
#   output: "results/R/logs/slurm-%j.out"
# setup: |
#   module load R/4.4.0
# run_levels:
#   1:
#     sbatch_args: { time: "00:05:00" }
#   2:
#     sbatch_args: { time: "00:10:00" }
#   3:
#     sbatch_args: { time: "00:15:00" }
#   4:
#     sbatch_args: { time: "00:20:00" }
# --- END SLURM CONFIG ---

script_arg <- commandArgs(FALSE)
script_arg <- script_arg[startsWith(script_arg, "--file=")]
here <- dirname(normalizePath(sub("^--file=", "", script_arg)))
setwd(here)
source("../../utils.R")
source("../model.R")
stopifnot(requireNamespace("jsonlite", quietly=TRUE))

level <- as.integer(Sys.getenv("RUN_LEVEL", "1"))
stopifnot(length(level) == 1L, level %in% 1:4)
particles <- c(64L,1000L,5000L,5000L)[level]
reps <- c(2L,20L,40L,100L)[level]
n_observations <- as.integer(Sys.getenv("BLOWFLIES_NOBS", "192"))
out_dir <- Sys.getenv("BLOWFLIES_OUT_DIR", "")
if (n_observations != 192L && !nzchar(out_dir)) {
  stop("A shortened diagnostic requires BLOWFLIES_OUT_DIR")
}
if (!nzchar(out_dir)) {
  out_dir <- if (level == 1L) file.path("results","smoke","R") else file.path("results","R")
}
object <- blowflies_obj(n_observations)
logliks <- numeric(reps)
seeds <- BLOWFLIES_MAIN_SEED + 10000L + seq_len(reps)
RNGkind(kind="Mersenne-Twister", normal.kind="Inversion", sample.kind="Rejection")
elapsed <- system.time({
  for (i in seq_len(reps)) {
    set.seed(seeds[i])
    logliks[i] <- as.numeric(logLik(pfilter(object, Np=particles,
      pred.mean=FALSE, pred.var=FALSE, filter.mean=FALSE,
      filter.traj=FALSE, save.states="no")))
  }
})
aggregate <- pomp::logmeanexp(logliks, se=TRUE)
result <- as.data.frame(as.list(BLOWFLIES_THETA), check.names=FALSE)
result$logLik <- aggregate[1]
result$se <- aggregate[2]
metadata <- save_run(
  out_dir=out_dir,
  tables=list(
    pfilter_logliks.csv=data.frame(theta_idx=0L, replicate=seq_len(reps),
                                  seed=seeds, logLik=logliks),
    results.csv=result,
    timings.csv=proc_time_frame(elapsed)
  ),
  run_config=list(
    kind="loglik", model="blowflies", RUN_LEVEL=level,
    MAIN_SEED=BLOWFLIES_MAIN_SEED, NP_EVAL=particles, NREPS_EVAL=reps,
    NOBS=n_observations, theta=as.list(BLOWFLIES_THETA),
    seed_rule="MAIN_SEED + 10000 + replicate (one-based)",
    RNG_kind=RNGkind(), USE_64BIT=TRUE,
    model_source_commit="7cfb3f9aa84c85de687b82d71b081016b9cc5762",
    data_sha256="aed36213ce260b7b92787d4d802c2dfb15fbee66e358431f769a2ddd456c05fb",
    optimization_performed=FALSE
  )
)
# Shared utils.R defaults to four decimal places. Preserve exact parameter
# values for the cross-language metadata comparison without changing utils.R.
writeLines(jsonlite::toJSON(metadata, auto_unbox=TRUE, pretty=TRUE,
                           null="null", digits=NA),
           file.path(out_dir, "latest.json"))
print(result)
