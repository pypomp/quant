library(pomp)
library(mvtnorm)
library(doParallel)
library(foreach)
library(doRNG)
library(tidyverse)

cores <- as.numeric(Sys.getenv("SLURM_NTASKS_PER_NODE", unset = NA))
run_level <- as.numeric(Sys.getenv("RUN_LEVEL", unset = NA))

if (is.na(cores)) cores <- detectCores()
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
    y = sp500$y, time = 1:length(sp500$y)
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

# Fitting POMP to data
sp500_rw.sd_rp <- 0.02
sp500_rw.sd_ivp <- 0.1
sp500_cooling.fraction50 <- 0.5

# Filter POMP to data ----------------------------------------------------

sp500_rw.sd <- rw_sd(
  mu = sp500_rw.sd_rp,
  theta = sp500_rw.sd_rp,
  kappa = sp500_rw.sd_rp,
  xi = sp500_rw.sd_rp,
  rho = sp500_rw.sd_rp,
  V_0 = ivp(sp500_rw.sd_ivp)
)

sp500_Np <- switch(run_level, 100, 200, 500, 1000)
sp500_Nmif <- switch(run_level,
  10,
  25,
  50,
  200
)
sp500_Nreps_eval <- switch(run_level,
  4,
  7,
  10,
  24
)
sp500_Nreps_local <- switch(run_level,
  10,
  15,
  20,
  24
)
sp500_Nreps_global <- switch(run_level,
  10,
  15,
  20,
  120*3
)

sp500_box <- rbind(
  mu = c(1e-6, 1e-4),
  theta = c(0.000075, 0.0002),
  kappa = c(1e-8, 0.1),
  xi = c(1e-8, 1e-2),
  rho = c(1e-8, 1),
  V_0 = c(1e-10, 1e-4)
)

global_starts <- pomp::runif_design(
  lower = sp500_box[, 1],
  upper = sp500_box[, 2],
  nseq = sp500_Nreps_global
)

# Feller's Condition
global_starts$xi <- runif(
  n = nrow(global_starts),
  min = 0,
  max = sqrt(global_starts$kappa * global_starts$theta * 2)
) # kappa<2*xi*theta

stew(file = sprintf("spx/R/1d_global_search360.rda"), {
    t.box <- system.time({
        t.if.box <- system.time({
            if.box <- foreach(
                i = 1:sp500_Nreps_global,
                .packages = 'pomp',
                .combine = c,
                .options.multicore = list(set.seed = TRUE)
            ) %dopar%
                {
                    mif2(
                        sp500.filt,
                        Nmif = sp500_Nmif,
                        rw.sd = sp500_rw.sd,
                        cooling.fraction.50 = sp500_cooling.fraction50,
                        Np = sp500_Np,
                        params = unlist(global_starts[i, ])
                    )
                } # if.box contains all estimates of parameters (list of mif objects)
        })
        t.L.box <- system.time({
            L.box <- foreach(
                i = 1:sp500_Nreps_global,
                .packages = 'pomp',
                .combine = rbind,
                .options.multicore = list(set.seed = TRUE)
            ) %dopar%
                {
                    replicate(
                        sp500_Nreps_eval,
                        logLik(pfilter(
                            sp500.filt,
                            params = coef(if.box[[i]]),
                            Np = sp500_Np
                        ))
                    ) |>
                        logmeanexp(se = TRUE)
                } # matrix containing logLik and SE for each time
        })
    })
})
