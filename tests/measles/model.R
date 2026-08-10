#' Shared definition of the UK measles benchmark for R pomp.
#'
#' Source this from a kind directory (loglik/, estimation/, timing/):
#'     source("../model.R")
#'
#' This is the Csnippet counterpart of model 001b in model.py. The two must
#' stay in step -- the whole point of these tests is that a difference in the
#' numbers means a difference in the frameworks, not in the models.
#'
#' Paths below are relative to the kind directory, since the test runner cd's
#' into the directory of the script it is running. All kinds sit at the same
#' depth, so one path works for each.

library(tidyverse)
library(pomp)

stopifnot(getRversion() >= "4.1")
stopifnot(packageVersion("pomp") >= "4.6")

MEASLES_MAIN_SEED <- 594709947L

#' London is the largest unit and Halesworth among the smallest, so between
#' them they bracket the range of population sizes the model has to cope with.
MEASLES_UNITS <- c("London", "Halesworth")

MEASLES_DATA_DIR <- "../data"
MEASLES_STARTS_PATH <- "../starting_parameters.csv"

MEASLES_COOLING_FRACTION_50 <- 0.5
MEASLES_DEFAULT_SD <- 0.02
MEASLES_DEFAULT_IVP_SD <- MEASLES_DEFAULT_SD * 12

#' The parameters carried by the model, in the order the estimates report them.
MEASLES_PARAM_NAMES <- c(
  "R0",
  "sigma",
  "gamma",
  "iota",
  "rho",
  "sigmaSE",
  "psi",
  "cohort",
  "amplitude",
  "S_0",
  "E_0",
  "I_0",
  "R_0"
)

#' The perturbation sizes must match RW_SD in model.py.
MEASLES_RW_SD <- rw_sd(
  R0 = MEASLES_DEFAULT_SD,
  sigma = MEASLES_DEFAULT_SD,
  gamma = MEASLES_DEFAULT_SD * 0.5,
  iota = MEASLES_DEFAULT_SD,
  rho = MEASLES_DEFAULT_SD * 0.5,
  sigmaSE = MEASLES_DEFAULT_SD,
  psi = MEASLES_DEFAULT_SD * 0.25,
  cohort = MEASLES_DEFAULT_SD * 0.5,
  amplitude = MEASLES_DEFAULT_SD * 0.5,
  S_0 = ivp(MEASLES_DEFAULT_IVP_SD),
  E_0 = ivp(MEASLES_DEFAULT_IVP_SD),
  I_0 = ivp(MEASLES_DEFAULT_IVP_SD),
  R_0 = ivp(MEASLES_DEFAULT_IVP_SD)
)

#' The global search box. Must match BOX in model.py.
# fmt: skip
MEASLES_BOX <- tibble::tribble(
  ~param,       ~lower,        ~upper,
  "R0",             10,            60,
  "rho",           0.1,           0.9,
  "sigmaSE",      0.04,           0.1,
  "amplitude",     0.1,           0.6,
  "S_0",          0.01,          0.07,
  "E_0",      0.000004,        0.0001,
  "I_0",      0.000003,         0.001,
  "R_0",           0.9,          0.99,
  "sigma",          25,           100,
  "iota",        0.004,             3,
  "psi",          0.05,             3,
  "cohort",        0.1,           0.7,
  "gamma",          25,           320
)


measles_rproc <- Csnippet(
  "
  double beta, br, seas, foi, dw, births;
  double rate[6], trans[6];
  double mu = 0.02;

  // cohort effect
  if (fabs(t-floor(t)-251.0/365.0) < 0.5*dt)
    br = cohort*birthrate/dt + (1-cohort)*birthrate;
  else
    br = (1.0-cohort)*birthrate;

  // term-time seasonality
  t = (t-floor(t))*365.25;
  if ((t>=7 && t<=100) ||
      (t>=115 && t<=199) ||
      (t>=252 && t<=300) ||
      (t>=308 && t<=356))
      seas = 1.0+amplitude*0.2411/0.7589;
  else
      seas = 1.0-amplitude;

  // transmission rate
  beta = R0 * seas * (1.0 - exp(-(gamma+mu) * dt)) / dt;

  // expected force of infection
  foi = beta*(I+iota)/pop;

  // white noise (extrademographic stochasticity)
  dw = rgammawn(sigmaSE,dt);

  rate[0] = foi*dw/dt;  // stochastic force of infection
  rate[1] = mu;         // natural S death
  rate[2] = sigma;      // rate of ending of latent stage
  rate[3] = mu;         // natural E death
  rate[4] = gamma;      // recovery
  rate[5] = mu;         // natural I death

  // Poisson births
  births = rpois(br*dt);

  // transitions between classes
  reulermultinom(2, S, &rate[0], dt, &trans[0]);
  reulermultinom(2, E, &rate[2], dt, &trans[2]);
  reulermultinom(2, I, &rate[4], dt, &trans[4]);

  S += births   - trans[0] - trans[1];
  E += trans[0] - trans[2] - trans[3];
  I += trans[2] - trans[4] - trans[5];
  R = pop - S - E - I;
  W += (dw - dt)/sigmaSE;  // standardized i.i.d. white noise
  C += trans[4];           // true incidence
"
)

measles_rinit <- Csnippet(
  "
  double m = pop/(S_0+E_0+I_0+R_0);
  S = nearbyint(m*S_0);
  E = nearbyint(m*E_0);
  I = nearbyint(m*I_0);
  R = nearbyint(m*R_0);
  W = 0;
  C = 0;
"
)

measles_dmeas <- Csnippet(
  "
  double m = rho*C;
  double v = m*(1.0-rho+psi*psi*m);
  double tol = 1.0e-18;
  if (cases > 0.0) {
    lik = pnorm(cases+0.5,m,sqrt(v)+tol,1,0)
           - pnorm(cases-0.5,m,sqrt(v)+tol,1,0) + tol;
  } else {
    lik = pnorm(cases+0.5,m,sqrt(v)+tol,1,0) + tol;
  }
  if (give_log) lik = log(lik);
"
)

measles_rmeas <- Csnippet(
  "
  double m = rho*C;
  double v = m*(1.0-rho+psi*psi*m);
  double tol = 1.0e-18;
  cases = rnorm(m,sqrt(v)+tol);
  if (cases > 0.0) {
    cases = nearbyint(cases);
  } else {
    cases = 0.0;
  }
"
)


#' The pomp object for one unit: 1950--1963 weekly cases, with population and
#' birth rate splined onto a monthly covariate grid. The four-year lag on the
#' birth rate is the delay between birth and entry into the susceptible pool.
measles_pomp <- function(unit_name, measles_data, demog_data) {
  measles_data |>
    mutate(year = as.integer(format(date, "%Y"))) |>
    filter(unit == unit_name & year >= 1950 & year < 1964) |>
    mutate(
      time = (julian(date, origin = as.Date("1950-01-01"))) / 365.25 + 1950
    ) |>
    filter(time > 1950 & time < 1964) |>
    select(time, cases) -> dat

  demog_data |>
    filter(unit == unit_name) |>
    select(-unit) -> demogUnit

  demogUnit |>
    summarize(
      time = seq(from = min(year), to = max(year), by = 1 / 12),
      pop = predict(smooth.spline(x = year, y = pop), x = time)$y,
      birthrate = predict(
        smooth.spline(x = year + 0.5, y = births),
        x = time - 4
      )$y
    ) -> covar

  pt <- parameter_trans(
    log = c("sigma", "gamma", "sigmaSE", "psi", "R0", "iota"),
    logit = c("cohort", "amplitude", "rho"),
    barycentric = c("S_0", "E_0", "I_0", "R_0")
  )

  dat |>
    pomp(
      t0 = with(dat, 2 * time[1] - time[2]),
      times = "time",
      rprocess = euler(measles_rproc, delta.t = 1 / 365.25),
      rinit = measles_rinit,
      dmeasure = measles_dmeas,
      rmeasure = measles_rmeas,
      partrans = pt,
      covar = covariate_table(covar, times = "time"),
      accumvars = c("C", "W"),
      statenames = c("S", "E", "I", "R", "C", "W"),
      paramnames = MEASLES_PARAM_NAMES
    )
}


#' A named list of pomp objects, one per unit.
measles_objects <- function(units = MEASLES_UNITS) {
  measles <- read.csv(file.path(MEASLES_DATA_DIR, "measles.csv"))
  measles$date <- as.Date(measles$date)
  demog <- read.csv(file.path(MEASLES_DATA_DIR, "demog.csv"))

  objs <- lapply(units, function(u) measles_pomp(u, measles, demog))
  names(objs) <- units
  objs
}


#' The He et al. (2010) estimates for `unit_name`, as a named numeric vector.
measles_mle <- function(unit_name) {
  mles <- read.csv(
    file.path(MEASLES_DATA_DIR, "AK_mles.csv"),
    stringsAsFactors = FALSE
  )
  row <- mles[mles$town == unit_name, ]
  if (nrow(row) == 0) {
    stop(sprintf("no MLE parameters for unit %s in AK_mles.csv", unit_name))
  }
  out <- as.numeric(row[1, MEASLES_PARAM_NAMES])
  names(out) <- MEASLES_PARAM_NAMES
  out
}


#' The first `n` committed starting points, as a data frame of parameter rows.
#' Both halves of `estimation/` and `timing/` read this same file, so the two
#' frameworks are given exactly the same work.
measles_starts <- function(n, path = MEASLES_STARTS_PATH) {
  if (!file.exists(path)) {
    stop(sprintf("starting parameters not found at %s", path))
  }
  starts <- read.csv(path)
  if (nrow(starts) < n) {
    stop(sprintf("%s holds %d starts, %d requested", path, nrow(starts), n))
  }
  starts[seq_len(n), MEASLES_PARAM_NAMES, drop = FALSE]
}


#' Regenerate the committed starting points. Run deliberately and commit the
#' result -- rerunning it invalidates every estimate already recorded against
#' the old design.
#'
#'     Rscript -e 'source("model.R"); measles_write_starts()'
measles_write_starts <- function(n = 360, path = "starting_parameters.csv") {
  set.seed(MEASLES_MAIN_SEED)
  lower <- MEASLES_BOX$lower
  upper <- MEASLES_BOX$upper
  names(lower) <- MEASLES_BOX$param
  names(upper) <- MEASLES_BOX$param
  starts <- runif_design(lower = lower, upper = upper, nseq = n)
  write.csv(starts, file = path, row.names = FALSE)
  cat(sprintf("wrote %d starting parameter vectors to %s\n", n, path))
}


#' Workers to register with doParallel: SLURM_NTASKS_PER_NODE, not
#' cpus-per-task, which is 1 in these jobs.
measles_cores <- function() {
  cores <- as.numeric(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = NA))
  if (is.na(cores)) {
    cores <- parallel::detectCores()
  }
  if (is.na(cores)) 1 else cores
}
