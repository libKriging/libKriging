#' GPU acceleration of the iterative path
#'
#' Query or switch the GPU backend used by \code{objective = "LLIterative"}
#' fits and \code{predictIterative}. Every other method (exact \code{"LL"},
#' Vecchia, Nystrom, \code{NestedKriging}) always runs on the CPU.
#'
#' A GPU backend is only present if libKriging was built with one
#' (\code{-DENABLE_CUDA_ITERATIVE=AUTO}, the default, enables CUDA when a CUDA
#' toolkit is found at build time); binary packages (e.g. from CRAN) are
#' built without. When present, it is enabled by default as soon as a usable
#' device is found, unless the environment variable \code{LK_ITERATIVE_GPU}
#' is set to \code{0} before the first GPU query or iterative call.
#'
#' @return \code{gpu_compiled_backends}: comma-separated names of the
#'     compiled-in backends (\code{"cuda"}, \code{"hip"}, \code{"sycl"},
#'     \code{"metal"}), or \code{""}. \code{gpu_available}: \code{TRUE} iff a
#'     compiled-in backend found a usable device. \code{gpu_backend}: the
#'     backend currently used, or \code{"none"} (CPU). \code{gpu_enabled}:
#'     \code{gpu_backend() != "none"}. \code{set_gpu_enabled}: invisible
#'     \code{gpu_enabled()} after the change.
#'
#' @examples
#' gpu_backend()
#' set_gpu_enabled(FALSE)
#' gpu_enabled()
#'
#' @name gpu
NULL

#' @rdname gpu
#' @export
gpu_compiled_backends <- function() lk_gpu_compiled_backends()

#' @rdname gpu
#' @export
gpu_available <- function() lk_gpu_available()

#' @rdname gpu
#' @export
gpu_backend <- function() lk_gpu_backend()

#' @rdname gpu
#' @export
gpu_enabled <- function() lk_gpu_enabled()

#' @rdname gpu
#' @param value \code{Logical}. \code{TRUE} to use the GPU when one is
#'     available (ignored otherwise), \code{FALSE} to force the CPU path.
#' @export
set_gpu_enabled <- function(value) {
  lk_set_gpu_enabled(isTRUE(as.logical(value)))
  invisible(gpu_enabled())
}
