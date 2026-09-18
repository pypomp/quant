# Original author Strang simulators, with fixed-innovation reference checks.
# Author code and licenses are retained under data/upstream.

fhn_reference <- function(sources_dir, out_dir, fixtures_only = FALSE) {
  sources_dir <- normalizePath(sources_dir, mustWork = TRUE)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_dir <- normalizePath(out_dir)
  pkg <- file.path(sources_dir, "upstream", "SMCABCFHN")
  experiment <- file.path(sources_dir, "upstream", "SMC-ABC_FHN")
  stopifnot(requireNamespace("Rcpp", quietly = TRUE),
            requireNamespace("jsonlite", quietly = TRUE))

  sha256 <- function(path) {
    strsplit(system2("shasum", c("-a", "256", shQuote(path)), stdout = TRUE), " ")[[1]][1]
  }
  frozen <- c(
    "SMCABCFHN/src/FHN.cpp" = "c3f7a6d760c85da12eee7b1f7afca07e2fb765f7f7663726670548db8e55f7f8",
    "SMCABCFHN/src/FHN_S.h" = "d50c118abf594a81a5b291f15b45e10562b0093d1e58de494dda776d65a2a481",
    "SMC-ABC_FHN/functions_SMC_ABC_FHN.R" = "0c8a01d44587612bebcfdc1da8a798e7dbafea3d654c07e85d8273643e9ccc5d",
    "SMC-ABC_FHN/SplittingStochasticFHN_1.0.tar.gz" = "1ec450b0371aefd784625ac373f85ba6cd5653c98fb8e0b466bb030f9956c5c4",
    "SMC-ABC_FHN/SplittingStochasticFHN/src/FHN_Strang_Cpp.cpp" = "a06b7ca1b7b2973f06dd6b5ad45819cd9788f36dc6e7249c1d403939c7811bca"
  )
  for (f in names(frozen)) {
    stopifnot(identical(sha256(file.path(sources_dir, "upstream", f)), unname(frozen[[f]])))
  }

  # Both full original C++ files were read before sourceCpp. Build in R's temporary
  # directory. The original prior exports compile but are never called.
  Rcpp::sourceCpp(file.path(pkg, "src", "FHN.cpp"), env = environment())
  Rcpp::sourceCpp(file.path(experiment, "SplittingStochasticFHN", "src", "FHN_Strang_Cpp.cpp"), env = environment())
  # This original file only defines functions. No ABC functions are called.
  source(file.path(experiment, "functions_SMC_ABC_FHN.R"), local = environment())
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")

  # FHN.cpp calls a dynamically looked-up R function named rmvn. The original
  # package depends on mvnfast, which is unavailable here. For coupled fixtures,
  # replace only this callback with a prescribed additive Gaussian-innovation
  # matrix. All covariance calculations and state updates remain original C++.
  # Independent stochastic samples use the other authors' original C++ simulator
  # and its original Rcpp::rnorm calls, with no callback replacement.
  callback_noise <- NULL
  callback_covariance <- NULL
  rmvn <- function(N, mu, Sigma) {
    stopifnot(is.matrix(callback_noise), nrow(callback_noise) == N,
              ncol(callback_noise) == 2L, all(mu == 0), all(is.finite(Sigma)))
    callback_covariance <<- Sigma
    callback_noise
  }

  had_rmvn <- exists("rmvn", envir = .GlobalEnv, inherits = FALSE)
  if (had_rmvn) previous_rmvn <- get("rmvn", envir = .GlobalEnv)
  assign("rmvn", rmvn, envir = .GlobalEnv)
  on.exit(if (had_rmvn) assign("rmvn", previous_rmvn, envir = .GlobalEnv)
          else rm("rmvn", envir = .GlobalEnv), add = TRUE)

  parameter_list <- function(theta) setNames(as.list(theta), c("epsilon", "gamma", "beta", "sigma"))
  original_matrices <- function(theta, dt) {
    transition <- exp_mat_SDEharmosc(dt, theta[1], theta[2])
    covariance <- cov_mat_SDEharmosc(dt, theta[4], theta[1], theta[2])
    list(transition = transition, covariance = covariance,
         cholesky = t(chol(covariance)))
  }

  make_scenario <- function(name, theta, dt, initial, seed = NULL, z = NULL, n = 30L) {
    mats <- original_matrices(theta, dt)
    if (!is.null(seed)) {
      # The original C++ draws iter normals for each coordinate and discards the
      # first column. Reconstruct its exact stream, then reset to run that code.
      set.seed(seed)
      randarr <- rbind(rnorm(n + 1L), rnorm(n + 1L))
      z <- t(randarr[, -1L, drop = FALSE])
      set.seed(seed)
      states <- t(FHN_Strang_Cpp_((0:n) * dt, dt, initial,
                                mats$transition, mats$cholesky, theta[1], theta[3]))
      method <- "unmodified FHN_Strang_Cpp_ full loop; exact native RNG replay"
    } else {
      n <- nrow(z)
      states <- matrix(0, n + 1L, 2L)
      states[1, ] <- initial
      for (i in seq_len(n)) {
        # Invoke the original exported C++ subflow functions, with prescribed z.
        x <- nonlinODE_FHN_Cpp_(states[i, ], dt / 2, theta[1], theta[3])
        x <- linSDE_FHN_Cpp_(x, mats$transition, mats$cholesky, z[i, ])
        states[i + 1L, ] <- nonlinODE_FHN_Cpp_(x, dt / 2, theta[1], theta[3])
      }
      method <- "unmodified exported C++ subflow functions; harness supplies fixed z"
    }
    callback_noise <<- z %*% t(mats$cholesky)
    voltage <- as.numeric(FHN_model_(theta, dt, initial, n))
    discrepancy <- max(abs(voltage - states[, 1]))
    stopifnot(all(is.finite(states)), discrepancy < 1e-11)
    list(name = name, parameters = parameter_list(theta), dt = dt,
         initial_state = initial, seed = seed, n_steps = n,
         standard_normals = z, innovations = callback_noise,
         author_V = voltage, author_states = states,
         author_states_method = method,
         author_covariance = callback_covariance,
         experiment_covariance = mats$covariance,
         experiment_transition = mats$transition,
         experiment_cholesky = mats$cholesky,
         authors_voltage_max_abs_difference = discrepancy)
  }

  parameters <- list(c(0.1, 1.5, 0.8, 0.3), c(0.25, 0.8, 0.4, 0.15), c(0.02, 0.5, 1.5, 0.7))
  initials <- list(c(0, 0), c(-0.5, -0.6), c(0.7, 0.2))
  step_sizes <- c(0.0001, 0.002, 0.02)
  fixed_z <- rbind(c(0, 0), c(1, 0), c(0, 1), c(-0.75, 0.5), c(2, -1))
  scenarios <- list()
  for (p in seq_along(parameters)) {
    for (j in seq_along(step_sizes)) {
      for (k in seq_along(initials)) {
        label <- sprintf("theta%d_dt%s_state%d", p, format(step_sizes[j], scientific = FALSE, trim = TRUE), k)
        scenarios[[length(scenarios) + 1L]] <- make_scenario(
          paste0(label, "_fixed"), parameters[[p]], step_sizes[j], initials[[k]], z = fixed_z)
        scenarios[[length(scenarios) + 1L]] <- make_scenario(
          paste0(label, "_native"), parameters[[p]], step_sizes[j], initials[[k]],
          seed = 17000L + p * 100L + j * 10L + k)
      }
    }
  }
  scenarios[[length(scenarios) + 1L]] <- make_scenario(
    "baseline_dt0.02_state1_native_T20", parameters[[1]], 0.02, initials[[1]],
    seed = 19001L, n = 1000L)

  distribution_scenarios <- list()
  if (!fixtures_only) for (j in seq_along(step_sizes)) {
    theta <- parameters[[1]]
    dt <- step_sizes[j]
    mats <- original_matrices(theta, dt)
    horizon <- 0.2
    n_steps <- as.integer(round(horizon / dt))
    n_paths <- 4096L
    seed <- 91800L + j
    endpoint <- matrix(0, n_paths, 2L)
    obs_indices <- as.integer(round((0:10) * 0.02 / dt)) + 1L
    observations <- matrix(0, n_paths, length(obs_indices))
    set.seed(seed)
    for (i in seq_len(n_paths)) {
      states <- FHN_Strang_Cpp_((0:n_steps) * dt, dt, initials[[1]],
                               mats$transition, mats$cholesky, theta[1], theta[3])
      endpoint[i, ] <- states[, n_steps + 1L]
      observations[i, ] <- states[1, obs_indices]
    }
    stopifnot(all(is.finite(endpoint)), all(is.finite(observations)))
    distribution_scenarios[[j]] <- list(
      name = sprintf("native_dt%s", format(dt, scientific = FALSE, trim = TRUE)),
      parameters = parameter_list(theta), dt = dt, initial_state = initials[[1]],
      horizon = horizon, n_steps = n_steps, n_paths = n_paths, seed = seed,
      endpoint_samples = endpoint, endpoint_mean = colMeans(endpoint),
      endpoint_covariance = cov(endpoint),
      voltage_observation_times = (0:10) * 0.02,
      voltage_observation_mean = colMeans(observations),
      voltage_observation_sd = apply(observations, 2, sd),
      finite_fraction = mean(is.finite(endpoint)),
      method = "unmodified FHN_Strang_Cpp_ and original R covariance/transition functions")
  }

  metadata <- list(
    schema_version = 1L,
    model = "dV=(V-V^3-U)/epsilon dt; dU=(gamma V-U+beta)dt + sigma dW",
    state_order = c("V", "U"), noise = "U only; s=0", observation = "exact V; no measurement noise",
    source_package_commit = "447d46a132d1df088d4826ae9f7369834bde8a42",
    experiment_commit = "55588cacf3f25e669f7238e1b527527781f208b5",
    source_sha256 = as.list(frozen),
    R = R.version.string, Rcpp = as.character(packageVersion("Rcpp")),
    jsonlite = as.character(packageVersion("jsonlite")),
    rng_kind = RNGkind(), platform = R.version$platform,
    author_cpp_modified = FALSE,
    callback_instrumentation = "global rmvn returns prescribed additive innovations and records Sigma; only SMCABCFHN coupled fixtures use this callback",
    native_rng_note = "Experiment C++ draws 2*(n_steps+1) standard normals, dimension-major, and discards each dimension's first draw",
    scope = "Local simulator validation only. No ABC, parameter optimization, cluster job, or paper-level reproduction.")

  jsonlite::write_json(list(metadata = metadata, scenarios = scenarios),
                       file.path(out_dir, "fixtures.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
  if (!fixtures_only) {
    jsonlite::write_json(list(metadata = metadata, scenarios = distribution_scenarios),
                         file.path(out_dir, "distribution_samples.json"), pretty = FALSE, auto_unbox = TRUE, digits = NA)
  }

  source_files <- list.files(file.path(sources_dir, "upstream"), recursive = TRUE, full.names = TRUE)
  source_entries <- lapply(source_files, function(path) {
    relative <- substring(path, nchar(file.path(sources_dir, "upstream")) + 2L)
    if (startsWith(relative, "SMCABCFHN/")) {
      url <- paste0("https://raw.githubusercontent.com/massimilianotamborrino/SMCABCFHN/",
                    metadata$source_package_commit, "/", sub("^SMCABCFHN/", "", relative))
    } else if (startsWith(relative, "SMC-ABC_FHN/SplittingStochasticFHN/")) {
      url <- paste0("https://raw.githubusercontent.com/IreneTubikanec/SMC-ABC_FHN/",
                    metadata$experiment_commit, "/SplittingStochasticFHN_1.0.tar.gz")
    } else {
      url <- paste0("https://raw.githubusercontent.com/IreneTubikanec/SMC-ABC_FHN/",
                    metadata$experiment_commit, "/", sub("^SMC-ABC_FHN/", "", relative))
    }
    list(path = paste0("upstream/", relative), source_url = url, sha256 = sha256(path),
         extracted_from_archive = startsWith(relative, "SMC-ABC_FHN/SplittingStochasticFHN/"))
  })
  summary <- list(metadata = metadata, sources = source_entries,
                  coupled_scenarios = length(scenarios),
                  native_rng_coupled_scenarios = sum(vapply(scenarios, function(x) !is.null(x$seed), logical(1))),
                  max_authors_voltage_difference = max(vapply(scenarios, `[[`, numeric(1), "authors_voltage_max_abs_difference")),
                  distribution_ran_this_invocation = !fixtures_only,
                  distribution_paths_per_dt = if (fixtures_only) 0L else 4096L,
                  distribution_horizon = 0.2,
                  output_sha256 = list(fixtures = sha256(file.path(out_dir, "fixtures.json")),
                                       distribution_samples = if (fixtures_only) NULL else sha256(file.path(out_dir, "distribution_samples.json"))))
  jsonlite::write_json(summary, file.path(out_dir, "reference_results.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
  cat(sprintf("Wrote %d coupled scenarios; maximum package/experiment V difference %.17g\n", length(scenarios), summary$max_authors_voltage_difference))
  if (!fixtures_only) cat("Wrote 4096 independent native-author paths per dt, T=0.2, dt=0.0001,0.002,0.02\n")
  invisible(summary)
}
