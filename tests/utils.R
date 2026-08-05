#' R mirror of tests/utils.py. A test writes its own committed record at the
#' end of its own run: tidy CSVs plus latest.json into results/R/. Any bulky
#' .rds alongside them is gitignored and exists only to reopen a run.
#'
#'     source("../../utils.R")   # adjust depth to reach tests/utils.R
#'     save_run(
#'       out_dir = file.path("results", "R"),
#'       tables = list(timings.csv = timings_df),
#'       run_config = list(kind = "timing", model = "spx", RUN_LEVEL = run_level)
#'     )

.pkg_version <- function(name) {
  tryCatch(as.character(utils::packageVersion(name)), error = function(e) NULL)
}


.git_sha <- function() {
  tryCatch(
    {
      sha <- suppressWarnings(
        system2("git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE)
      )
      if (length(sha) == 0 || !nzchar(sha[1])) NULL else sha[1]
    },
    error = function(e) NULL
  )
}


.hardware <- function() {
  cpu_model <- tryCatch(
    {
      info <- readLines("/proc/cpuinfo", warn = FALSE)
      line <- grep("^model name", info, value = TRUE)
      if (length(line) == 0) NULL else trimws(sub("^model name\\s*:\\s*", "", line[1]))
    },
    error = function(e) NULL
  )

  cores <- Sys.getenv("SLURM_CPUS_PER_TASK", unset = "")
  if (!nzchar(cores)) {
    cores <- Sys.getenv("SLURM_NTASKS_PER_NODE", unset = "")
  }

  out <- list(
    nodelist = Sys.getenv("SLURMD_NODENAME", unset = Sys.info()[["nodename"]]),
    cpu_model = cpu_model,
    cores = if (nzchar(cores)) as.integer(cores) else parallel::detectCores()
  )
  out[!vapply(out, is.null, logical(1))]
}


#' R's 5-element proc_time vector as the labelled frame the reports expect.
proc_time_frame <- function(t) {
  data.frame(
    stage = c("user.self", "sys.self", "elapsed", "user.child", "sys.child"),
    seconds = as.numeric(t)[1:5]
  )
}


#' Which pomp, on what hardware, from which commit, with which knobs.
run_metadata <- function(run_config = list()) {
  slurm_vars <- c(
    job_id = "SLURM_JOB_ID",
    partition = "SLURM_JOB_PARTITION",
    cpus = "SLURM_CPUS_PER_TASK",
    gres = "SLURM_JOB_GRES",
    gpus = "SLURM_GPUS"
  )
  slurm <- lapply(slurm_vars, function(v) {
    val <- Sys.getenv(v, unset = "")
    if (nzchar(val)) val else NULL
  })
  slurm <- slurm[!vapply(slurm, is.null, logical(1))]

  list(
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    pomp_version = .pkg_version("pomp"),
    panelPomp_version = .pkg_version("panelPomp"),
    quant_git_sha = .git_sha(),
    hardware = .hardware(),
    slurm = slurm,
    run_config = run_config
  )
}


#' Write `tables` (named by output filename, *.csv or *.csv.gz) and latest.json
#' into `out_dir`; `raw` is saveRDS'd as the gitignored fallback.
save_run <- function(out_dir,
                     tables,
                     run_config = list(),
                     raw = NULL,
                     raw_name = "raw.rds") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  if (is.null(names(tables)) || any(!nzchar(names(tables)))) {
    stop("`tables` must be a fully named list; names become output filenames.")
  }

  for (name in names(tables)) {
    path <- file.path(out_dir, name)
    df <- tables[[name]]
    if (grepl("\\.csv\\.gz$", name)) {
      con <- gzfile(path, "w")
      write.csv(df, con, row.names = FALSE)
      close(con)
    } else if (grepl("\\.csv$", name)) {
      write.csv(df, path, row.names = FALSE)
    } else {
      stop(sprintf("table '%s' must be named *.csv or *.csv.gz", name))
    }
    cat(sprintf("  wrote %s (%d rows)\n", path, nrow(df)))
  }

  if (!is.null(raw)) {
    saveRDS(raw, file.path(out_dir, raw_name))
  }

  meta <- run_metadata(run_config)
  writeLines(
    jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE, null = "null"),
    file.path(out_dir, "latest.json")
  )
  cat(sprintf("  wrote %s/latest.json\n", out_dir))

  invisible(meta)
}
