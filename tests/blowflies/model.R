#' Shared canonical R pomp::blowflies1() benchmark definition.
#' The frozen data/defaults are checked before each new baseline run.

suppressPackageStartupMessages(library(pomp))
BLOWFLIES_MODEL_DIR <- dirname(normalizePath(sys.frame(1)$ofile))
BLOWFLIES_MAIN_SEED <- 4517000L
BLOWFLIES_THETA <- c(P=3.2838, delta=0.16073, N0=679.94,
                    sigma.P=1.3512, sigma.d=0.74677, sigma.y=0.026649)

blowflies_obj <- function(n_observations=192L) {
  if (length(n_observations) != 1L || is.na(n_observations) ||
      n_observations != as.integer(n_observations) ||
      n_observations < 1L || n_observations > 192L) {
    stop("n_observations must be an integer from 1 through 192")
  }
  object <- pomp::blowflies1()
  frozen <- read.csv2(file.path(BLOWFLIES_MODEL_DIR, "data", "nicholson_population_I.csv"))
  raw <- read.csv2(text=get("blowfly_dat", asNamespace("pomp")))
  observed <- as.data.frame(object)
  stopifnot(identical(raw, frozen),
            identical(as.numeric(observed$day), seq(16,398,by=2)),
            identical(as.numeric(observed$y), as.numeric(frozen$y[frozen$day>14])),
            identical(coef(object)[names(BLOWFLIES_THETA)], BLOWFLIES_THETA),
            object@t0 == 14, object@rprocess@delta.t == 1)
  initial <- slot(object@rinit, "R.fun")(
    params=coef(object), t0=object@t0, y.init=object@userdata$y.init)
  stopifnot(identical(as.numeric(initial[seq_len(15)]),
    c(397,450.5,504,590,676,738.5,801,829.5,858,884.5,911,926.5,942,945,948)))
  if (n_observations < 192L) object <- window(object, end=14 + 2*n_observations)
  object
}
