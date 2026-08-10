library(tidyverse)
library(pomp)
library(panelPomp)

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

# Load demographic and case data (relative path to measles/data)
measles <- read.csv("../../../../measles/data/measles.csv")
measles$date <- as.Date(measles$date)
demog <- read.csv("../../../../measles/data/demog.csv")

units <- c("London", "Halesworth", "Hastings", "Cardiff")

create_pomp_for_unit <- function(unit_name, measles_data, demog_data) {
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
    reframe(
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
        "R0", "sigma", "gamma", "iota", "rho", "sigmaSE", "psi",
        "cohort", "amplitude", "S_0", "E_0", "I_0", "R_0"
      ),
      params = c(
        R0 = 0, sigma = 0, gamma = 0, iota = 0, rho = 0, sigmaSE = 0, psi = 0,
        cohort = 0, amplitude = 0, S_0 = 0, E_0 = 0, I_0 = 0, R_0 = 0
      )
    )
}

# Construct panelPomp
pomp_objects <- lapply(units, function(unit_name) {
  create_pomp_for_unit(unit_name, measles, demog)
})
names(pomp_objects) <- units
panel_obj <- panelPomp(pomp_objects)

# Extract and save covariates for all units to match Python
covar_list <- list()
for (u in units) {
  demogUnit <- demog[demog$unit == u, ]
  covar_u <- data.frame(
    time = seq(from = min(demogUnit$year), to = max(demogUnit$year), by = 1 / 12)
  )
  covar_u$pop <- predict(smooth.spline(x = demogUnit$year, y = demogUnit$pop), x = covar_u$time)$y
  covar_u$birthrate <- predict(
    smooth.spline(x = demogUnit$year + 0.5, y = demogUnit$births),
    x = covar_u$time - 4
  )$y
  covar_u$unit <- u
  covar_list[[u]] <- covar_u
}
combined_covars <- do.call(rbind, covar_list)
write.csv(combined_covars, "../R_covariates.csv", row.names = FALSE)
print("Saved R_covariates.csv")
