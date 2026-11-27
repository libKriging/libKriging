library(testthat)

# The global settings are exported (no `rlibkriging:::`), as the Optim /
# LinearAlgebra static methods of the Python, Octave / MATLAB and Julia bindings.

optim_names <- c(
  "is_reparametrized", "use_reparametrize",
  "get_theta_lower_factor", "set_theta_lower_factor",
  "get_theta_upper_factor", "set_theta_upper_factor",
  "variogram_bounds_heuristic_used", "use_variogram_bounds_heuristic",
  "get_log_level", "set_log_level",
  "get_max_iteration", "set_max_iteration",
  "get_max_restart", "set_max_restart",
  "get_gradient_tolerance", "set_gradient_tolerance",
  "get_objective_rel_tolerance", "set_objective_rel_tolerance")
# moved to parallel_* in 2.0 (#468)
removed_optim_names <- c(
  "get_thread_start_delay_ms", "set_thread_start_delay_ms",
  "get_thread_pool_size", "set_thread_pool_size",
  "get_num_threads_percent", "set_num_threads_percent", "get_num_threads")
linalg_names <- c(
  "get_chol_jitter", "set_chol_jitter",
  "get_num_nugget", "set_num_nugget",
  "set_chol_warning", "check_chol_rcond", "chol_rcond_checked")

test_that("Optim and LinearAlgebra settings are exported", {
  exported <- getNamespaceExports("rlibkriging")
  for (f in c(paste0("optim_", optim_names), paste0("linalg_", linalg_names)))
    expect_true(f %in% exported, info = f)
  for (f in paste0("optim_", removed_optim_names))
    expect_false(f %in% exported, info = f)
})

test_that("The thread budget is set without :::", {
  library(rlibkriging)
  default_percent <- parallel_get_num_threads_percent()
  full <- parallel_get_num_threads()
  parallel_set_num_threads_percent(1e-6)  # at least 1 thread
  expect_equal(parallel_get_num_threads(), 1)
  parallel_set_num_threads_percent(default_percent)
  expect_equal(parallel_get_num_threads(), full)
})

parallel_names <- c(
  "get_num_threads_percent", "set_num_threads_percent", "get_num_threads",
  "get_thread_pool_size", "set_thread_pool_size",
  "get_thread_start_delay_ms", "set_thread_start_delay_ms",
  "get_blas_threads_min_n", "set_blas_threads_min_n",
  "get_omp_min_rows", "set_omp_min_rows",
  "blas_backend", "blas_num_threads", "omp_enabled")

test_that("Parallel settings are exported", {
  exported <- getNamespaceExports("rlibkriging")
  for (f in paste0("parallel_", parallel_names))
    expect_true(f %in% exported, info = f)
  library(rlibkriging)
  orig <- list(parallel_get_num_threads_percent(), parallel_get_thread_pool_size(),
               parallel_get_blas_threads_min_n(), parallel_get_omp_min_rows())
  parallel_set_num_threads_percent(0.5)
  expect_equal(parallel_get_num_threads_percent(), 0.5)
  parallel_set_thread_pool_size(3)
  expect_equal(parallel_get_thread_pool_size(), 3)
  parallel_set_blas_threads_min_n(0)
  expect_equal(parallel_get_blas_threads_min_n(), 0)
  parallel_set_omp_min_rows(300)
  expect_equal(parallel_get_omp_min_rows(), 300)
  expect_true(parallel_blas_backend() %in% c("mkl", "openblas", "accelerate", "unknown"))
  expect_true(parallel_blas_num_threads() >= 0)
  expect_true(is.logical(parallel_omp_enabled()))
  parallel_set_num_threads_percent(orig[[1]])
  parallel_set_thread_pool_size(orig[[2]])
  parallel_set_blas_threads_min_n(orig[[3]])
  parallel_set_omp_min_rows(orig[[4]])
})

# docs/Settings.md (#468): set / get round trip of every user setting, and the
# precedence rule (LK_* environment read once at load, a setter call wins)
test_that("Every user setting round-trips", {
  settings <- list(
    list("optim_set_max_restart", "optim_get_max_restart", 3L),
    list("optim_set_max_iteration", "optim_get_max_iteration", 57L),
    list("optim_set_gradient_tolerance", "optim_get_gradient_tolerance", 0.0125),
    list("optim_set_objective_rel_tolerance", "optim_get_objective_rel_tolerance", 0.0025),
    list("optim_set_theta_lower_factor", "optim_get_theta_lower_factor", 0.25),
    list("optim_set_theta_upper_factor", "optim_get_theta_upper_factor", 4.5),
    list("optim_use_variogram_bounds_heuristic", "optim_variogram_bounds_heuristic_used", FALSE),
    list("optim_use_reparametrize", "optim_is_reparametrized", FALSE),
    list("optim_set_log_level", "optim_get_log_level", 2L),
    list("parallel_set_num_threads_percent", "parallel_get_num_threads_percent", 0.5),
    list("parallel_set_thread_pool_size", "parallel_get_thread_pool_size", 2L),
    list("parallel_set_thread_start_delay_ms", "parallel_get_thread_start_delay_ms", 7L),
    list("parallel_set_blas_threads_min_n", "parallel_get_blas_threads_min_n", 250L),
    list("parallel_set_omp_min_rows", "parallel_get_omp_min_rows", 300L),
    list("linalg_set_chol_jitter", "linalg_get_chol_jitter", 1e-7))
  for (s in settings) {
    setter <- getExportedValue("rlibkriging", s[[1]])
    getter <- getExportedValue("rlibkriging", s[[2]])
    orig <- getter()
    setter(s[[3]])
    expect_equal(getter(), s[[3]], info = s[[1]])
    setter(orig)
    expect_equal(getter(), orig, info = s[[1]])
  }
  # a plain double holding an integer is accepted, as a user types it
  orig <- optim_get_max_restart()
  optim_set_max_restart(2)
  expect_equal(optim_get_max_restart(), 2)
  expect_error(optim_set_max_restart(-1))
  expect_equal(optim_get_max_restart(), 2)
  optim_set_max_restart(orig)
})

test_that("LK_* environment is read at load, a setter call wins", {
  skip_on_cran()
  rscript <- file.path(R.home("bin"), "Rscript")
  script <- tempfile(fileext = ".R")
  writeLines(c(
    "library(rlibkriging)",
    "stopifnot(optim_get_max_restart() == 4)",
    "stopifnot(optim_get_max_iteration() == 33)",
    "stopifnot(parallel_get_omp_min_rows() == 456)",
    "stopifnot(linalg_get_chol_jitter() == 1e-8)",
    "stopifnot(!optim_is_reparametrized())",
    "Sys.setenv(LK_MAX_RESTART = '9')",
    "stopifnot(optim_get_max_restart() == 4)",
    "optim_set_max_restart(1)",
    "stopifnot(optim_get_max_restart() == 1)",
    "cat('OK')"), script)
  # the child inherits the environment of this process: system2(env = ) is
  # not supported on Windows
  vars <- c(R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep),
            LK_MAX_RESTART = "4", LK_MAX_ITERATION = "33", LK_OMP_MIN_ROWS = "456",
            LK_CHOL_JITTER = "1e-8", LK_REPARAMETRIZE = "0")
  old <- Sys.getenv(names(vars), unset = NA)
  do.call(Sys.setenv, as.list(vars))
  on.exit({
    for (v in names(old))
      if (is.na(old[[v]])) Sys.unsetenv(v) else do.call(Sys.setenv, setNames(list(old[[v]]), v))
    unlink(script)
  })
  out <- system2(rscript, shQuote(script), stdout = TRUE, stderr = TRUE)
  expect_true(any(grepl("OK", out)), info = paste(out, collapse = "\n"))
})

test_that("An invalid LK_* value gives a warning line and the default", {
  skip_on_cran()
  rscript <- file.path(R.home("bin"), "Rscript")
  script <- tempfile(fileext = ".R")
  writeLines(c(
    "library(rlibkriging)",
    "stopifnot(optim_get_max_restart() == 10)",
    "stopifnot(parallel_get_thread_pool_size() == 0)",
    "cat('OK')"), script)
  vars <- c(R_LIBS = paste(.libPaths(), collapse = .Platform$path.sep),
            LK_MAX_RESTART = "abc", LK_THREAD_POOL_SIZE = "-2")
  old <- Sys.getenv(names(vars), unset = NA)
  do.call(Sys.setenv, as.list(vars))
  on.exit({
    for (v in names(old))
      if (is.na(old[[v]])) Sys.unsetenv(v) else do.call(Sys.setenv, setNames(list(old[[v]]), v))
    unlink(script)
  })
  out <- paste(system2(rscript, shQuote(script), stdout = TRUE, stderr = TRUE), collapse = "\n")
  expect_true(grepl("OK", out), info = out)
  expect_true(grepl("[WARNING] LK_MAX_RESTART=", out, fixed = TRUE), info = out)
  expect_true(grepl("[WARNING] LK_THREAD_POOL_SIZE=", out, fixed = TRUE), info = out)
})
