#' Shared definition of the SIR Bayesian benchmark for R pomp.
#'
#' Source this from a kind directory (pmcmc/, abc/):
#'     source("../model.R")
#'
#' Paths below are relative to the kind directory, since the test runner cd's
#' into the directory of the script it is running. All kinds sit at the same
#' depth, so one path works for each.
#'
#' R is the generator of record for the dataset. `pomp::sir()` and pypomp's
#' `sir()` each simulate their own data at construction, so one side has to win
#' or nothing downstream is comparable. R wins because rebuilding an R pomp
#' around an external CSV means reusing compiled pomp_fun slots, whereas
#' pypomp's Pomp() takes `ys` as a plain DataFrame. Regenerate with
#' `Rscript -e 'source("model.R"); bayes_write_data("data")'` from this
#' directory; the CSVs are committed.

library(pomp)

BAYES_SIM_SEED <- 329343545L # the pomp::sir() default; data-generating seed
BAYES_MAIN_SEED <- 3141593L # everything else

#' The two estimated parameters. Everything else is fixed at its true value.
#' beta1/rho is the classic weakly-identified transmission/reporting pair, so
#' the posterior has a curved ridge rather than being a near-Gaussian blob.
BAYES_FREE <- c("beta1", "rho")

#' Flat bounded priors: reproducible exactly across R and Python with no
#' distributional-convention mismatch, and they make the grid reference simply
#' the normalized likelihood over the box.
#' Width is set by what the grid reference can resolve; see the note on
#' PRIOR_BOX in model.py. Roughly +/-9 posterior sd in each direction.
BAYES_PRIOR_LOWER <- c(beta1 = 300, rho = 0.45)
BAYES_PRIOR_UPPER <- c(beta1 = 500, rho = 0.80)

#' Random-walk proposal scales, on the NATURAL scale, as pomp's mvn_diag_rw
#' expects.
#'
#' These are not the same numbers as RW_SD in model.py, and cannot be. pypomp's
#' pmcmc runs its chain on the estimation scale (log beta1, logit rho) and takes
#' its rw_sd there; pomp perturbs `params` directly. The values below are the
#' local linearization of model.py's {beta1: 0.02, rho: 0.05} at the true theta:
#'
#'     beta1: 400 * 0.02                = 8.0
#'     rho:   0.6 * (1 - 0.6) * 0.05    = 0.012
#'
#' so the two kernels take comparably sized steps near the mode. They remain
#' different kernels -- both valid Metropolis-Hastings, targeting the same
#' posterior. Only the stationary distributions are compared; acceptance rate
#' and mixing are expected to differ and are reported separately per language.
BAYES_RW_SD <- c(beta1 = 8.0, rho = 0.012)

BAYES_DATA_PATH <- "../data/sir_data.csv"
BAYES_TRUE_PATH <- "../data/true_theta.csv"
BAYES_SCALE_PATH <- "../data/probe_scale.csv"


#' Log-density of the flat box prior, as a Csnippet.
#'
#' Returns the proper normalized density inside the box and -Inf outside, so
#' proposals leaving the box are rejected. pomp calls this on the natural scale.
bayes_dprior <- Csnippet("
  double lp;
  if (beta1 > 300.0 && beta1 < 500.0 && rho > 0.45 && rho < 0.80) {
    lp = -log(200.0) - log(0.35);
  } else {
    lp = R_NegInf;
  }
  lik = (give_log) ? lp : exp(lp);
")


#' The benchmark pomp object: pomp::sir() with the box prior attached.
#'
#' `pomp::sir()` simulates its data from `seed`, so this is deterministic and
#' matches the committed CSV by construction.
bayes_sir <- function(seed = BAYES_SIM_SEED) {
  s <- pomp::sir(seed = seed)
  pomp(s, dprior = bayes_dprior, paramnames = BAYES_FREE)
}


#' Write the canonical dataset, the simulating parameters, and the observation
#' times that pypomp needs to rebuild the same covariate basis.
bayes_write_data <- function(out_dir = "data", seed = BAYES_SIM_SEED) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  s <- pomp::sir(seed = seed)

  data_df <- data.frame(
    time = as.numeric(time(s)),
    reports = as.numeric(obs(s, "reports"))
  )
  write.csv(data_df, file.path(out_dir, "sir_data.csv"), row.names = FALSE)

  theta <- coef(s)
  theta_df <- data.frame(
    parameter = names(theta),
    value = as.numeric(theta)
  )
  write.csv(theta_df, file.path(out_dir, "true_theta.csv"), row.names = FALSE)

  invisible(list(data = data_df, theta = theta_df))
}


#' Starting points for the chains, drawn from the prior on a fixed seed so the
#' two languages disperse their chains over the same region. Written to CSV by
#' whichever side runs first is *not* the design -- each side draws its own,
#' because the chains are independent and only their stationary distribution is
#' being compared.
bayes_starts <- function(n, seed = BAYES_MAIN_SEED) {
  set.seed(seed)
  out <- matrix(NA_real_, nrow = n, ncol = length(BAYES_FREE))
  colnames(out) <- BAYES_FREE
  for (p in BAYES_FREE) {
    out[, p] <- runif(n, BAYES_PRIOR_LOWER[[p]], BAYES_PRIOR_UPPER[[p]])
  }
  as.data.frame(out)
}


#' The ABC probe set.
#'
#' Kept to statistics that are expressible identically in pure JAX. The JAX
#' twins live in model.py and are checked against these numerically by the
#' precondition section of abc/report.qmd -- probe_acf in particular dispatches
#' to a C routine whose centring and divisor conventions are not safe to assume.
bayes_probes <- function() {
  list(
    mean = probe_mean("reports"),
    sd = probe_sd("reports"),
    acf1 = probe_acf("reports", lags = 1, type = "correlation")
  )
}


#' Probe scales: the sd of each probe across simulations at the true theta.
#'
#' Normalizes the ABC distance so no single probe dominates. Computed once and
#' committed so both languages weight the distance identically.
bayes_write_probe_scale <- function(out_dir = "data", nsim = 500,
                                    seed = BAYES_MAIN_SEED) {
  obj <- bayes_sir()
  set.seed(seed)
  pb <- probe(obj, probes = bayes_probes(), nsim = nsim)
  sim_vals <- as.data.frame(pb@simvals)

  # pomp mangles probe names on the way out -- probe_acf returns "acf1.acf[1]"
  # rather than "acf1" -- but pypomp keys `scale` by the names of the `probes`
  # dict, so the canonical names are reasserted here. The ordering is the
  # ordering of bayes_probes(), and each probe contributes exactly one column.
  stopifnot(ncol(sim_vals) == length(bayes_probes()))
  scale_df <- data.frame(
    probe = names(bayes_probes()),
    scale = vapply(sim_vals, sd, numeric(1))
  )
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  write.csv(scale_df, file.path(out_dir, "probe_scale.csv"), row.names = FALSE)
  invisible(scale_df)
}
