library(pomp)
library(mvtnorm)
library(doParallel)
library(foreach)
library(doRNG)
library(tidyverse)

cores <- as.numeric(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = NA))
run_level <- as.numeric(Sys.getenv("run_level", unset = NA))

if (is.na(cores)) {
  cores <- detectCores()
}
registerDoParallel(cores)
registerDoRNG(34118892)


# Data Manipulation -------------------------------------------------------

sp500_raw <- read.csv("data/SPX.csv")
sp500 <- sp500_raw %>%
  mutate(date = as.Date(Date)) %>%
  mutate(diff_days = difftime(date, min(date), units = "day")) %>%
  mutate(time = as.numeric(diff_days)) %>%
  mutate(y = log(Close / lag(Close))) %>%
  select(time, y) %>%
  drop_na()


# Name of States and Parmeters --------------------------------------------

sp500_statenames <- c("V", "S")
sp500_rp_names <- c("mu", "kappa", "theta", "xi", "rho")
sp500_ivp_names <- c("V_0")
sp500_parameters <- c(sp500_rp_names, sp500_ivp_names)
sp500_covarnames <- "covaryt"


# rprocess ----------------------------------------------------------------
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
# Initialization Model ----------------------------------------------------
sp500_rinit <- "
  V = V_0; // V_0 is a parameter as well
  S = 1105; // 1105 is the starting price
"
# rmeasure ----------------------------------------------------------------
sp500_rmeasure_filt <- "
  y=exp(covaryt);
"
sp500_rmeasure_sim <- "
  y = (mu - 0.5 * V) + sqrt(V);
"
# dmeasure ----------------------------------------------------------------
sp500_dmeasure <- "
   lik=dnorm(y, mu-0.5*V, sqrt(V), give_log);
"
# Parameter Transformation ------------------------------------------------
my_ToTrans <- "
     T_xi = log(xi);
     T_kappa = log(kappa);
     T_theta = log(theta);
     T_V_0 = log(V_0);
     T_mu = log(mu);
     T_rho = log((rho + 1) / (1 - rho));
  "
my_FromTrans <- "
    kappa = exp(T_kappa);
    theta = exp(T_theta);
    xi = exp(T_xi);
    V_0 = exp(T_V_0);
    mu = exp(T_mu);
    rho = -1 + 2 / (1 + exp(-T_rho));
  "
sp500_partrans <- parameter_trans(
  toEst = Csnippet(my_ToTrans),
  fromEst = Csnippet(my_FromTrans)
)

# Construct Filter Object -------------------------------------------------

sp500.filt <- pomp(
  data = data.frame(
    y = sp500$y,
    time = 1:length(sp500$y)
  ),
  statenames = sp500_statenames,
  paramnames = sp500_parameters,
  covarnames = sp500_covarnames,
  times = "time",
  t0 = 0,
  covar = covariate_table(
    time = 0:length(sp500$y),
    covaryt = c(0, sp500$y),
    times = "time"
  ),
  rmeasure = Csnippet(sp500_rmeasure_filt),
  dmeasure = Csnippet(sp500_dmeasure),
  rprocess = discrete_time(step.fun = Csnippet(rproc1), delta.t = 1),
  rinit = Csnippet(sp500_rinit),
  partrans = sp500_partrans
)


# Filter POMP to data ----------------------------------------------------

# sp500_Np <- switch(run_level,
#   100,
#   200,
#   500,
#   1000
# )
# # sp500_Nreps_eval <- switch(run_level,
# #   4,
# #   7,
# #   10,
# #   360
# # )

theta <- c(
  mu = 3.68e-4,
  kappa = 3.14e-2,
  theta = 1.12e-4,
  xi = 2.27e-3,
  rho = -7.38e-1,
  V_0 = 7.66e-3^2
)
stew(file = sprintf("spx/pfilter_check/R_results/spx_results_eval.rda"), {
  t.box <- system.time({
    L.box <- foreach(
      i = 1:3600,
      .packages = "pomp",
      .combine = rbind,
      .options.multicore = list(set.seed = TRUE)
    ) %dopar%
      {
        logLik(pfilter(sp500.filt, params = theta, Np = 1000))
      }
  })
})
