## ----prelims,include=FALSE,cache=FALSE-----------------------------------
stopifnot(getRversion() >= "4.1")
stopifnot(packageVersion("pomp") >= "4.6")

set.seed(594709947L)
library(tidyverse)
library(pomp)
library(doParallel)
library(foreach)
library(doRNG)

RUN_LEVEL <- as.numeric(Sys.getenv("RUN_LEVEL", unset = 1))

NP_FITR <- switch(RUN_LEVEL, 2, 500, 5000, 5000)
NFITR <- switch(RUN_LEVEL, 2, 10, 100, 100)
NREPS_FITR <- switch(RUN_LEVEL, 2, 3, 36, 360)

## ----rproc-------------------------------------------------
rproc <- Csnippet(
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

## ----rinit-------------------------------------------------
rinit <- Csnippet(
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

## ----dmeasure-------------------------------------------------
dmeas <- Csnippet(
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

## ----rmeasure-------------------------------------------------
rmeas <- Csnippet(
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

## ----load-data-------------------------------------------------
measles <- read.csv("measles/R/data/measles.csv")
measles$date <- as.Date(measles$date)
demog <- read.csv("measles/R/data/demog.csv")

## ----units-vector-------------------------------------------------
units <- c("London", "Halesworth") # Specify units to process

## ----create-pomp-function---------------------------------------------
create_pomp_for_unit <- function(unit_name, measles_data, demog_data) {
    ## ----unit-data-------------------------------------------------
    measles_data |>
        mutate(year = as.integer(format(date, "%Y"))) |>
        filter(unit == unit_name & year >= 1950 & year < 1964) |>
        mutate(
            time = (julian(date, origin = as.Date("1950-01-01"))) /
                365.25 +
                1950
        ) |>
        filter(time > 1950 & time < 1964) |>
        select(time, cases) -> dat

    demog_data |>
        filter(unit == unit_name) |>
        select(-unit) -> demogUnit

    ## ----prep-covariates-------------------------------------------------
    demogUnit |>
        summarize(
            time = seq(from = min(year), to = max(year), by = 1 / 12),
            pop = predict(smooth.spline(x = year, y = pop), x = time)$y,
            birthrate = predict(
                smooth.spline(x = year + 0.5, y = births),
                x = time - 4
            )$y,
            birthrate1 = predict(
                smooth.spline(x = year + 0.5, y = births),
                x = time
            )$y
        ) -> covar1

    covar1 |>
        select(-birthrate1) -> covar

    ## ----pomp-construction-----------------------------------------------
    pt <- parameter_trans(
        log = c("sigma", "gamma", "sigmaSE", "psi", "R0", "iota"),
        logit = c("cohort", "amplitude", "rho"),
        barycentric = c("S_0", "E_0", "I_0", "R_0")
    )

    dat |>
        pomp(
            t0 = with(dat, 2 * time[1] - time[2]),
            times = "time",
            rprocess = euler(rproc, delta.t = 1 / 365.25),
            rinit = rinit,
            dmeasure = dmeas,
            rmeasure = rmeas,
            partrans = pt,
            covar = covariate_table(covar, times = "time"),
            accumvars = c("C", "W"),
            statenames = c("S", "E", "I", "R", "C", "W"),
            paramnames = c(
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
        )
}

## ----create-pomp-objects-------------------------------------------------
pomp_objects <- lapply(units, function(unit_name) {
    create_pomp_for_unit(unit_name, measles, demog)
})
names(pomp_objects) <- units

## ----extract-first-pomp-------------------------------------------------
m1 <- pomp_objects[[1]] # For backward compatibility

## ----run-methods-----------------------------------------------

DEFAULT_SD = 0.02
IVP_DEFAULT_SD = DEFAULT_SD * 12
INITIAL_RW_SD = rw_sd(
    S_0 = ivp(IVP_DEFAULT_SD),
    E_0 = ivp(IVP_DEFAULT_SD),
    I_0 = ivp(IVP_DEFAULT_SD),
    R_0 = ivp(IVP_DEFAULT_SD),
    R0 = DEFAULT_SD,
    sigmaSE = DEFAULT_SD,
    amplitude = DEFAULT_SD * 0.5,
    rho = DEFAULT_SD * 0.5,
    gamma = DEFAULT_SD * 0.5,
    psi = DEFAULT_SD * 0.25,
    iota = DEFAULT_SD,
    sigma = DEFAULT_SD,
    cohort = DEFAULT_SD * 0.5
)

# fmt: skip
specific_bounds = tibble::tribble(
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
lower = specific_bounds$lower
upper = specific_bounds$upper
names(lower) = specific_bounds$param
names(upper) = specific_bounds$param

starting_parameters = runif_design(
    lower = lower,
    upper = upper,
    nseq = NREPS_FITR
)
write.csv(
    starting_parameters,
    file = "measles/R/parameter_comparison/starting_parameters.csv"
)

## ----setup-parallel-------------------------------------------------
cores <- as.numeric(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = NA))
if (is.na(cores)) {
    cores <- 36
}
registerDoParallel(cores)
registerDoRNG(594709947L)

## ----run-pfilter-for-all-units-------------------------------------------------
bake(file = sprintf("measles/R/parameter_comparison/mif_coefs.rds"), {
    time0 <- Sys.time()

    # Initialize list to store results
    all_coefs <- list()

    # Process each unit
    for (unit_name in units) {
        cat(sprintf("Processing unit: %s\n", unit_name))

        # Get the pomp object for this unit
        pomp_obj <- pomp_objects[[unit_name]]

        # Extract parameters (excluding town, loglik, loglik.sd, mu, delay, alpha)
        param_names <- c(
            "sigma",
            "gamma",
            "rho",
            "R0",
            "amplitude",
            "iota",
            "cohort",
            "psi",
            "S_0",
            "E_0",
            "I_0",
            "R_0",
            "sigmaSE"
        )

        unit_coefs <- foreach(
            i = 1:NREPS_FITR,
            .packages = "pomp",
            .combine = c,
            .options.multicore = list(set.seed = TRUE)
        ) %dopar%
            {
                unit_params = starting_parameters[i, ]
                names(unit_params) = colnames(starting_parameters)
                coef(mif2(
                    pomp_obj,
                    params = unit_params,
                    Np = NP_FITR,
                    Nmif = NFITR,
                    rw.sd = INITIAL_RW_SD,
                    cooling.fraction.50 = 0.5
                ))
            }

        # Store results
        all_coefs[[unit_name]] <- data.frame(
            unit = unit_name,
            replicate = 1:NREPS_FITR,
            coef = unit_coefs,
            names = names(unit_coefs)
        )
    }

    # Combine all results
    results_df <- do.call(rbind, all_coefs)

    time1 <- Sys.time()
    cat(sprintf("Total time: %s\n", time1 - time0))

    results_df
})
