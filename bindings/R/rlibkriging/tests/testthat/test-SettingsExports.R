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
  "get_gradient_tolerance", "set_gradient_tolerance",
  "get_objective_rel_tolerance", "set_objective_rel_tolerance",
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
})

test_that("The thread budget is set without :::", {
  library(rlibkriging)
  default_percent <- optim_get_num_threads_percent()
  full <- optim_get_num_threads()
  optim_set_num_threads_percent(1e-6)  # at least 1 thread
  expect_equal(optim_get_num_threads(), 1)
  optim_set_num_threads_percent(default_percent)
  expect_equal(optim_get_num_threads(), full)
})
