#' Shared definition of the SPX stochastic volatility benchmark for R pomp.
#'
#' This is the R counterpart of model.py. The model definition below was
#' previously duplicated verbatim in both performance/test.R and
#' pfilter_check/eval.R -- about 130 identical lines in each, which is exactly
#' the kind of thing that drifts silently between two copies.
#'
#' Source this from a kind directory (timing/, estimation/, loglik/):
#'     source("../model.R")
#'
#' Paths are relative to the kind directory, since the test runner cd's into
#' the directory of the script it is running. All kinds sit at the same depth,
#' so the data path below is the same for each.

library(pomp)
library(tidyverse)

SPX_DATA_PATH <- "../../../data/SPX.csv"

# The Sun (2024) estimates. Must match SUN2024_THETA in model.py -- the whole
# point of the loglik kind is that both languages evaluate at the same theta.
SPX_SUN2024_THETA <- c(
  mu = 3.68e-4,
  kappa = 3.14e-2,
  theta = 1.12e-4,
  xi = 2.27e-3,
  rho = -7.38e-1,
  V_0 = 7.66e-3^2
)

# Global search box. Must match BOX in model.py.
SPX_BOX <- rbind(
  mu = c(1e-6, 1e-4),
  theta = c(0.000075, 0.0002),
  kappa = c(1e-8, 0.1),
  xi = c(1e-8, 1e-2),
  rho = c(1e-8, 1),
  V_0 = c(1e-10, 1e-4)
)

SPX_RW_SD_RP <- 0.02
SPX_RW_SD_IVP <- 0.1
SPX_COOLING_FRACTION_50 <- 0.5

SPX_RW_SD <- rw_sd(
  mu = SPX_RW_SD_RP,
  theta = SPX_RW_SD_RP,
  kappa = SPX_RW_SD_RP,
  xi = SPX_RW_SD_RP,
  rho = SPX_RW_SD_RP,
  V_0 = ivp(SPX_RW_SD_IVP)
)


#' Load and difference the SPX series into log returns.
spx_data <- function(path = SPX_DATA_PATH) {
  read.csv(path) %>%
    mutate(date = as.Date(Date)) %>%
    mutate(diff_days = difftime(date, min(date), units = "day")) %>%
    mutate(time = as.numeric(diff_days)) %>%
    mutate(y = log(Close / lag(Close))) %>%
    select(time, y) %>%
    drop_na()
}


#' Build the SPX filtering pomp object.
spx_filt <- function(sp500 = spx_data()) {
  statenames <- c("V", "S")
  rp_names <- c("mu", "kappa", "theta", "xi", "rho")
  ivp_names <- c("V_0")
  parameters <- c(rp_names, ivp_names)
  covarnames <- "covaryt"

  rproc1 <- "
    double dWv, dZ, dWs, rt;

    rt=covaryt;
    dWs = (rt-mu+0.5*V)/(sqrt(V));
    dZ = rnorm(0, 1);

    dWv = rho * dWs + sqrt(1 - rho * rho) * dZ;

    S += S * (mu + sqrt(fmax(V, 0.0)) * dWs);
    V += kappa*(theta - V) + xi*sqrt(V)*dWv;

    if (V<=0) {
      V=1e-32;
    }
  "

  rinit <- "
    V = V_0; // V_0 is a parameter as well
    S = 1105; // 1105 is the starting price
  "

  rmeasure_filt <- "
    y=exp(covaryt);
  "

  dmeasure <- "
     lik=dnorm(y, mu-0.5*V, sqrt(V), give_log);
  "

  to_trans <- "
       T_xi = log(xi);
       T_kappa = log(kappa);
       T_theta = log(theta);
       T_V_0 = log(V_0);
       T_mu = log(mu);
       T_rho = log((rho + 1) / (1 - rho));
    "

  from_trans <- "
      kappa = exp(T_kappa);
      theta = exp(T_theta);
      xi = exp(T_xi);
      V_0 = exp(T_V_0);
      mu = exp(T_mu);
      rho = -1 + 2 / (1 + exp(-T_rho));
    "

  pomp(
    data = data.frame(
      y = sp500$y,
      time = 1:length(sp500$y)
    ),
    statenames = statenames,
    paramnames = parameters,
    covarnames = covarnames,
    times = "time",
    t0 = 0,
    covar = covariate_table(
      time = 0:length(sp500$y),
      covaryt = c(0, sp500$y),
      times = "time"
    ),
    rmeasure = Csnippet(rmeasure_filt),
    dmeasure = Csnippet(dmeasure),
    rprocess = discrete_time(step.fun = Csnippet(rproc1), delta.t = 1),
    rinit = Csnippet(rinit),
    partrans = parameter_trans(
      toEst = Csnippet(to_trans),
      fromEst = Csnippet(from_trans)
    )
  )
}


#' Draw starting parameter vectors from SPX_BOX, respecting Feller's condition.
#'
#' xi is redrawn conditional on kappa and theta so that 2*kappa*theta > xi^2;
#' model.py does the same.
spx_starts <- function(n) {
  starts <- pomp::runif_design(
    lower = SPX_BOX[, 1],
    upper = SPX_BOX[, 2],
    nseq = n
  )
  starts$xi <- runif(
    n = nrow(starts),
    min = 0,
    max = sqrt(starts$kappa * starts$theta * 2)
  )
  starts
}
