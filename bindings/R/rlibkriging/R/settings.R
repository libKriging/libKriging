#' Global settings of the optimizer
#'
#' Process-wide settings of libKriging's hyperparameter optimizer and thread
#' budget, the same as the \code{Optim} static methods of the Python, Octave /
#' MATLAB and Julia bindings (\code{optim_<name>} in R for \code{Optim.<name>}).
#' Set them before fitting: changing one while a fit runs in another thread is
#' a data race.
#'
#' Thread budget: \code{optim_set_num_threads_percent(x)} sets the share of the
#' physical cores used by libKriging (multistart workers, OpenMP loops and
#' BLAS), \code{x <= 1} a ratio (\code{0.5}), \code{x > 1} a percentage
#' (\code{50}), \code{0} automatic (100 \%). Default from the
#' \code{LK_NUM_THREADS_PERCENT} environment variable.
#' \code{optim_get_num_threads()} returns the resulting number of threads.
#' Socket workers (\code{parallel::makeCluster(type = "PSOCK")},
#' \code{future::multisession}) each use the whole budget: give k workers
#' \code{1 / k} each. Forked children (\code{parallel::mclapply}) start with
#' one thread.
#'
#' @name optim_settings
#' @return The getters return the current value; the setters return
#'   \code{NULL} invisibly.
#' @examples
#' optim_set_num_threads_percent(0.5)
#' optim_get_num_threads()
#' optim_set_num_threads_percent(0)
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
