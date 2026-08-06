#' Shared definition of the Dhaka cholera benchmark for R pomp.
#'
#' Source this from a kind directory (timing/, loglik/):
#'     source("../model.R")
#'
#' Paths below are relative to the kind directory, since the test runner cd's
#' into the directory of the script it is running. All kinds sit at the same
#' depth, so one path works for each.

library(pomp)

DACCA_COOLING_FRACTION_50 <- 0.8
DACCA_DEFAULT_SD <- 0.02
DACCA_DEFAULT_IVP_SD <- DACCA_DEFAULT_SD * 8

DACCA_STARTS_PATH <- "../starting_parameters.csv"

#' pomp's dacca() names several parameters differently from pypomp's. The
#' committed starting points carry the pypomp names, so they are translated on
#' the way in.
DACCA_NAME_MAP <- c(
  epsilon = "eps",
  m = "deltaI",
  c = "clin",
  sigma = "sd_beta",
  bs1 = "logbeta1",
  bs2 = "logbeta2",
  bs3 = "logbeta3",
  bs4 = "logbeta4",
  bs5 = "logbeta5",
  bs6 = "logbeta6",
  omegas1 = "logomega1",
  omegas2 = "logomega2",
  omegas3 = "logomega3",
  omegas4 = "logomega4",
  omegas5 = "logomega5",
  omegas6 = "logomega6"
)

#' The perturbation sizes must match RW_SD in model.py. rho, clin, alpha and
#' delta are not perturbed and so are simply absent here; Y_0 likewise.
DACCA_RW_SD <- rw_sd(
  gamma = DACCA_DEFAULT_SD,
  deltaI = DACCA_DEFAULT_SD,
  eps = DACCA_DEFAULT_SD,
  beta_trend = DACCA_DEFAULT_SD,
  sd_beta = DACCA_DEFAULT_SD,
  tau = DACCA_DEFAULT_SD,
  logbeta1 = DACCA_DEFAULT_SD,
  logbeta2 = DACCA_DEFAULT_SD,
  logbeta3 = DACCA_DEFAULT_SD,
  logbeta4 = DACCA_DEFAULT_SD,
  logbeta5 = DACCA_DEFAULT_SD,
  logbeta6 = DACCA_DEFAULT_SD,
  logomega1 = DACCA_DEFAULT_SD,
  logomega2 = DACCA_DEFAULT_SD,
  logomega3 = DACCA_DEFAULT_SD,
  logomega4 = DACCA_DEFAULT_SD,
  logomega5 = DACCA_DEFAULT_SD,
  logomega6 = DACCA_DEFAULT_SD,
  S_0 = ivp(DACCA_DEFAULT_IVP_SD),
  I_0 = ivp(DACCA_DEFAULT_IVP_SD),
  R1_0 = ivp(DACCA_DEFAULT_IVP_SD),
  R2_0 = ivp(DACCA_DEFAULT_IVP_SD),
  R3_0 = ivp(DACCA_DEFAULT_IVP_SD)
)


#' The built-in Dhaka model. pomp's dacca() uses dt = 1/240 on observations
#' spaced 1/12 of a year apart, the same discretisation as nstep = 20 in
#' model.py.
dacca_obj <- function() {
  dacca()
}


#' The first `n` committed starting points, under pomp's parameter names.
dacca_starts <- function(n, path = DACCA_STARTS_PATH) {
  if (!file.exists(path)) {
    stop(sprintf("starting parameters not found at %s", path))
  }
  starts <- read.csv(path)[seq_len(n), , drop = FALSE]
  for (old_name in names(DACCA_NAME_MAP)) {
    new_name <- DACCA_NAME_MAP[[old_name]]
    names(starts)[names(starts) == old_name] <- new_name
  }
  starts
}
