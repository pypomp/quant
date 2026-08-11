#' Shared definition of the UK panel measles benchmark for R panelPomp.
#'
#' Source this from a kind directory (loglik/, estimation/, timing/):
#'     source("../model.R")
#'
#' This is the Csnippet counterpart of model.py. The two must stay in step --
#' the whole point of these tests is that a difference in the numbers means a
#' difference in the frameworks, not in the models.
#'
#' Paths below are relative to the kind directory, since the test runner cd's
#' into the directory of the script it is running. All kinds sit at the same
#' depth, so one path works for each.

library(tidyverse)
library(pomp)
library(panelPomp)

stopifnot(getRversion() >= "4.1")

PANEL_MAIN_SEED <- 594709947L

PANEL_UNITS <- c("London", "Halesworth", "Hastings", "Cardiff")

PANEL_SHARED_NAMES <- c("R0", "sigma", "gamma", "sigmaSE", "cohort", "amplitude")
PANEL_SPECIFIC_NAMES <- c("iota", "rho", "psi", "S_0", "E_0", "I_0", "R_0")
PANEL_PARAM_NAMES <- c(PANEL_SHARED_NAMES, PANEL_SPECIFIC_NAMES)

PANEL_DATA_DIR <- "../../measles/data"
PANEL_STARTS_PATH <- "../starting_parameters.csv"
PANEL_COVARS_PATH <- "../R_covariates.csv"

PANEL_COOLING_FRACTION_50 <- 0.5
PANEL_DEFAULT_SD <- 0.02
PANEL_DEFAULT_IVP_SD <- PANEL_DEFAULT_SD * 12

#' The perturbation sizes must match RW_SD in model.py.
PANEL_RW_SD <- rw_sd(
  R0 = PANEL_DEFAULT_SD * 0.25,
  sigma = PANEL_DEFAULT_SD * 0.25,
  gamma = PANEL_DEFAULT_SD * 0.5,
  iota = PANEL_DEFAULT_SD,
  rho = PANEL_DEFAULT_SD * 0.5,
  sigmaSE = PANEL_DEFAULT_SD,
  psi = PANEL_DEFAULT_SD * 0.25,
  cohort = PANEL_DEFAULT_SD * 0.5,
  amplitude = PANEL_DEFAULT_SD * 0.5,
  S_0 = ivp(PANEL_DEFAULT_IVP_SD),
  E_0 = ivp(PANEL_DEFAULT_IVP_SD),
  I_0 = ivp(PANEL_DEFAULT_IVP_SD),
  R_0 = ivp(PANEL_DEFAULT_IVP_SD)
)


panel_rproc <- Csnippet(
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

panel_rinit <- Csnippet(
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

panel_dmeas <- Csnippet(
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

panel_rmeas <- Csnippet(
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


#' Population and birth rate splined onto the monthly covariate grid. The
#' four-year lag on the birth rate is the delay between birth and entry into
#' the susceptible pool.
panel_covariates <- function(unit_name, demog_data) {
  demog_data |>
    filter(unit == unit_name) |>
    select(-unit) -> demog_unit

  demog_unit |>
    reframe(
      time = seq(from = min(year), to = max(year), by = 1 / 12),
      pop = predict(smooth.spline(x = year, y = pop), x = time)$y,
      birthrate = predict(
        smooth.spline(x = year + 0.5, y = births),
        x = time - 4
      )$y
    )
}


panel_measles_pomp <- function(unit_name, measles_data, demog_data) {
  measles_data |>
    mutate(year = as.integer(format(date, "%Y"))) |>
    filter(unit == unit_name & year >= 1950 & year < 1964) |>
    mutate(
      time = (julian(date, origin = as.Date("1950-01-01"))) / 365.25 + 1950
    ) |>
    filter(time > 1950 & time < 1964) |>
    select(time, cases) -> dat

  covar <- panel_covariates(unit_name, demog_data)

  pt <- parameter_trans(
    log = c("sigma", "gamma", "sigmaSE", "psi", "R0", "iota"),
    logit = c("cohort", "amplitude", "rho"),
    barycentric = c("S_0", "E_0", "I_0", "R_0")
  )

  zeroed <- setNames(rep(0, length(PANEL_PARAM_NAMES)), PANEL_PARAM_NAMES)

  dat |>
    pomp(
      t0 = with(dat, 2 * time[1] - time[2]),
      times = "time",
      rprocess = euler(panel_rproc, delta.t = 1 / 365.25),
      rinit = panel_rinit,
      dmeasure = panel_dmeas,
      rmeasure = panel_rmeas,
      partrans = pt,
      covar = covariate_table(covar, times = "time"),
      accumvars = c("C", "W"),
      statenames = c("S", "E", "I", "R", "C", "W"),
      paramnames = PANEL_PARAM_NAMES,
      params = zeroed
    )
}


panel_measles_objects <- function(units = PANEL_UNITS) {
  measles <- read.csv(file.path(PANEL_DATA_DIR, "measles.csv"))
  measles$date <- as.Date(measles$date)
  demog <- read.csv(file.path(PANEL_DATA_DIR, "demog.csv"))

  objs <- lapply(units, function(u) panel_measles_pomp(u, measles, demog))
  names(objs) <- units
  panelPomp(objs)
}


#' The He et al. (2010) estimates, shared parameters averaged across units.
panel_measles_mles <- function(units = PANEL_UNITS) {
  mles <- read.csv(
    file.path(PANEL_DATA_DIR, "AK_mles.csv"),
    stringsAsFactors = FALSE
  )
  mles <- mles %>% filter(town %in% units)

  shared <- sapply(PANEL_SHARED_NAMES, function(p) mean(mles[[p]]))

  specific <- matrix(
    NA_real_,
    nrow = length(PANEL_SPECIFIC_NAMES),
    ncol = length(units),
    dimnames = list(PANEL_SPECIFIC_NAMES, units)
  )
  for (u in units) {
    row <- mles %>% filter(town == u)
    for (p in PANEL_SPECIFIC_NAMES) {
      specific[p, u] <- row[[p]]
    }
  }

  list(shared = shared, specific = specific)
}


#' The first `n` committed starting points, one `list(shared, specific)` per
#' replicate. Both halves of estimation/ and timing/ read this same file, so
#' the two frameworks are given exactly the same work.
panel_measles_starts <- function(n,
                                 path = PANEL_STARTS_PATH,
                                 units = PANEL_UNITS) {
  if (!file.exists(path)) {
    stop(sprintf("starting parameters not found at %s", path))
  }
  starts <- read.csv(path, stringsAsFactors = FALSE)

  reps <- sort(unique(starts$replicate))
  if (length(reps) < n) {
    stop(sprintf("%s holds %d starts, %d requested", path, length(reps), n))
  }

  lapply(reps[seq_len(n)], function(j) {
    rep_data <- starts %>% filter(replicate == j)

    shared_data <- rep_data %>% filter(unit == "shared")
    shared <- shared_data$value
    names(shared) <- shared_data$param
    shared <- shared[PANEL_SHARED_NAMES]

    specific_data <- rep_data %>% filter(unit != "shared")
    specific <- matrix(
      NA_real_,
      nrow = length(PANEL_SPECIFIC_NAMES),
      ncol = length(units),
      dimnames = list(PANEL_SPECIFIC_NAMES, units)
    )
    for (u in units) {
      for (p in PANEL_SPECIFIC_NAMES) {
        specific[p, u] <- specific_data %>%
          filter(unit == u, param == p) %>%
          pull(value)
      }
    }

    list(shared = shared, specific = specific)
  })
}


#' Regenerate the committed covariate grid that model.py's `align_covariates`
#' reads. Run deliberately and commit the result -- it is the table that makes
#' the pypomp and pomp likelihoods comparable.
panel_write_covariates <- function(path = PANEL_COVARS_PATH,
                                   units = PANEL_UNITS) {
  demog <- read.csv(file.path(PANEL_DATA_DIR, "demog.csv"))
  covars <- lapply(units, function(u) {
    out <- panel_covariates(u, demog)
    out$unit <- u
    out
  })
  write.csv(do.call(rbind, covars), path, row.names = FALSE)
  cat(sprintf("wrote covariate grid for %d units to %s\n", length(units), path))
}


#' Workers to register with doParallel: SLURM_NTASKS_PER_NODE, not
#' cpus-per-task, which is 1 in these jobs.
panel_cores <- function() {
  cores <- as.numeric(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = NA))
  if (is.na(cores)) {
    cores <- parallel::detectCores()
  }
  if (is.na(cores)) 1 else cores
}
