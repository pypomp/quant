#!/usr/bin/env Rscript
#
# Extract the frozen reference data out of the SPX global-search .rda.
#
# That file holds a 360-element mif2List of S4 pomp objects and is ~66 MB, which
# is why it lives in a gitignored `_hidden` directory and why nobody but its
# author can render the SPX report. pyreadr cannot read S4 objects and traces()
# needs pomp's class definitions, so this extraction has to happen in R.
#
# Everything the report actually consumes is small: the likelihood estimates,
# the IF2 traces, and three proc_time vectors.
#
# Called by scripts/freeze_r_results.py; run directly as:
#   Rscript scripts/extract_spx_search360.R <path/to/1d_global_search360.rda> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("usage: extract_spx_search360.R <rda_path> <out_dir>")
}
rda_path <- args[1]
out_dir <- args[2]

# Allow a scratch library (R_LIBS_USER) to supply pomp when the project's renv
# library is unavailable.
suppressMessages(library(pomp))
# fwrite is used over write.csv for speed on the 72k-row trace table. Both emit
# ~15 significant digits, so the frozen values agree with the originals to a
# relative 5e-15 -- irrelevant next to the Monte Carlo error these numbers
# carry, but worth knowing before treating them as bit-exact. data.table comes
# in with pomp.
suppressMessages(library(data.table))
cat("pomp version used for extraction:", as.character(packageVersion("pomp")), "\n")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

e <- new.env()
load(rda_path, envir = e)

# --- likelihood estimates: a 360 x 2 matrix of (est, se) ---------------------
L_box <- get("L.box", envir = e)
logliks <- as.data.frame(L_box)
colnames(logliks) <- c("logLik", "se")
logliks$replicate <- seq_len(nrow(logliks))
logliks <- logliks[, c("replicate", "logLik", "se")]
fwrite(logliks, file.path(out_dir, "pfilter_logliks.csv"))
cat("pfilter_logliks.csv:", nrow(logliks), "rows\n")

# --- IF2 traces: one block per replicate ------------------------------------
if_box <- get("if.box", envir = e)
traces_list <- lapply(seq_along(if_box), function(i) {
    tr <- as.data.frame(traces(if_box[[i]]))
    tr$iteration <- seq_len(nrow(tr)) - 1L # iteration 0 is the starting value
    tr$replicate <- i
    tr
})
traces_df <- do.call(rbind, traces_list)
# Put the identifiers first; leave the parameter columns in their pomp order.
id_cols <- c("replicate", "iteration")
traces_df <- traces_df[, c(id_cols, setdiff(colnames(traces_df), id_cols))]

# gzip: ~72k rows x 9 columns is several MB as plain text but well under 2 MB
# compressed, and both readr and pandas read .csv.gz transparently.
fwrite(traces_df, file.path(out_dir, "mif_traces.csv.gz"), compress = "gzip")
cat("mif_traces.csv.gz:", nrow(traces_df), "rows,", ncol(traces_df), "cols\n")

# --- timings: three proc_time vectors ---------------------------------------
# t.if.box is the IF2 search, t.L.box the likelihood evaluation, t.box the total.
proc_rows <- function(name, label) {
    tv <- get(name, envir = e)
    data.frame(
        stage = label,
        metric = names(tv),
        seconds = as.numeric(tv),
        stringsAsFactors = FALSE
    )
}
timings <- rbind(
    proc_rows("t.if.box", "mif"),
    proc_rows("t.L.box", "pfilter"),
    proc_rows("t.box", "total")
)
fwrite(timings, file.path(out_dir, "timings.csv"))
cat("timings.csv:", nrow(timings), "rows\n")

cat("replicates:", length(if_box), "\n")
