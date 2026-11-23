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
#' Iterative method (\code{method = "iterative(...)"}):
#' \code{linalg_set_cg_warning}: warn when a conjugate-gradient solve stops
#' before its tolerance; \code{linalg_set_cg_auto_precond}: automatic Nystrom
#' preconditioner of the iterative predictor;
#' \code{linalg_set_iterative_dense_max_mb}: host-memory budget (MiB) of the
#' dense correlation matrix, \code{0} matrix-free (default 6144).
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
#' distributed. The \code{optim_*} thread settings are the same settings. Set
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

#' GPU backend of the iterative method
#'
#' Process-wide settings of the GPU acceleration of \code{method =
#' "iterative(...)"}, the same as the \code{Gpu} static methods of the Python,
#' Octave / MATLAB and Julia bindings (\code{gpu_<name>} in R for
#' \code{Gpu.<name>}). The default (CRAN) package is CPU only: no backend is
#' compiled in and the settings have no effect; the "-gpu" source package is
#' built with the backends found on the installing machine.
#'
#' \code{gpu_compiled_backends()}, \code{gpu_available()},
#' \code{gpu_active_backend()}: read only (\code{"none"} on the CPU path).
#' \code{gpu_set_enabled}: use the GPU when available (default \code{TRUE},
#' \code{FALSE} when \code{LK_ITERATIVE_GPU=0}). \code{gpu_set_dense_max_mb}:
#' device-memory budget (MiB) of the dense correlation matrix, \code{0}
#' matrix-free. \code{gpu_set_mixed_precision}: TF32 / fp32 products (CUDA).
#' \code{gpu_set_verbose}: diagnostic of each GPU CG call.
#'
#' @name gpu_settings
#' @return The getters return the current value; the setters return
#'   \code{NULL} invisibly.
#' @examples
#' gpu_compiled_backends()
#' gpu_active_backend()
NULL
