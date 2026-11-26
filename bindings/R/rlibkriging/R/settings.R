#' Global settings of the optimizer
#'
#' Process-wide settings of libKriging's hyperparameter optimizer, the same as
#' the \code{Optim} static methods of the Python, Octave / MATLAB and Julia
#' bindings (\code{optim_<name>} in R for \code{Optim.<name>}). Set them before
#' fitting: changing one while a fit runs in another thread is a data race.
#' Defaults from the \code{LK_*} environment variables, read when the package is
#' loaded (\code{Sys.setenv} afterwards has no effect); a setter call overrides
#' them (docs/Settings.md). The thread settings are \code{parallel_*} (moved
#' from \code{optim_*} in 2.0).
#'
#' @name optim_settings
#' @return The getters return the current value; the setters return
#'   \code{NULL} invisibly.
#' @examples
#' optim_set_max_restart(5)
#' optim_get_max_restart()
#' optim_set_max_restart(10)
NULL

#' Global settings of the Cholesky factorizations
#'
#' Process-wide settings of libKriging's Cholesky factorizations, the same as
#' the \code{LinearAlgebra} static methods of the Python, Octave / MATLAB and
#' Julia bindings (\code{linalg_<name>} in R for \code{LinearAlgebra.<name>}).
#' \code{linalg_set_chol_jitter}: fixed jitter of noise-free fits, \code{< 0}
#' automatic per kernel (default), \code{0} disabled, \code{> 0} this value.
#' \code{linalg_set_num_nugget}: numerical nugget of the fallback factorization.
#' \code{linalg_set_chol_warning}: warn when the fallback adds that nugget.
#' \code{linalg_check_chol_rcond}: check the reciprocal condition number of the
#' factorizations.
#'
#' @name linalg_settings
#' @return The getters return the current value; the setters return
#'   \code{NULL} invisibly.
#' @examples
#' linalg_get_chol_jitter()
NULL

#' Thread budget of the library and its distribution
#'
#' Process-wide thread settings of libKriging, the same as the \code{Parallel}
#' static methods of the Python, Octave / MATLAB and Julia bindings
#' (\code{parallel_<name>} in R for \code{Parallel.<name>}); see
#' docs/dev/parallel.md. One budget bounds the multistart workers, the OpenMP
#' loops and the BLAS threads together; the other settings only shape how it is
#' distributed (these settings were also \code{optim_*} before 2.0). Set
#' them before the calls.
#'
#' \code{parallel_set_num_threads_percent(x)}: share of the physical cores,
#' \code{x <= 1} a ratio, \code{x > 1} a percentage, \code{0} automatic
#' (100 \%); \code{parallel_get_num_threads()} the resulting number of threads.
#' \code{parallel_set_thread_pool_size}: multistart workers (0 automatic).
#' \code{parallel_set_blas_threads_min_n}: small-problem BLAS rule (-1
#' automatic, 0 off, n > 0 threshold). \code{parallel_set_omp_min_rows}: rows
#' from which the covariance / gradient loops run on OpenMP threads.
#' \code{parallel_blas_backend()}, \code{parallel_blas_num_threads()},
#' \code{parallel_omp_enabled()}: read only, the state of each level.
#' Socket workers (\code{parallel::makeCluster(type = "PSOCK")},
#' \code{future::multisession}) each use the whole budget: give k workers
#' \code{1 / k} each. Forked children (\code{parallel::mclapply}) start with
#' one thread.
#'
#' @name parallel_settings
#' @return The getters return the current value; the setters return
#'   \code{NULL} invisibly.
#' @examples
#' parallel_set_num_threads_percent(0.5)
#' parallel_get_num_threads()
#' parallel_blas_backend()
#' parallel_set_num_threads_percent(0)
NULL
