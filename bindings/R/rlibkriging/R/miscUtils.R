warnOnDots <- function(dots) {
    warning("Arguments ",
            paste0(names(dots), "=", dots, collapse = ","),
            " are ignored.")
}


## XXXY use 'identical' with formula objects seesm cleaner
formula2regmodel = function(form) {
    if (format(form) == "~0")
        return("none")
    else if (format(form) == "~1")
        return("constant")
    else if (format(form) == "~.")
        return("linear")
    else if (format(form) == "~.^2")
        return("interactive")
    else stop("Unsupported formula ", form)
}


regmodel2formula = function(regmodel) {
  if (regmodel == "none")
    return(~0)
  else if (regmodel == "constant")
    return(~1)
  else if (regmodel == "linear")
    return(~.)
  else if (regmodel == "interactive")
    return(~.^2)
  else stop("Unsupported regmodel ",regmodel)
}

as_numeric_matrix = function(x) {
  if (is.matrix(x) & is.numeric(x))
    return(x)
  if (is.vector(x) & is.numeric(x))
    return(matrix(x,nrow=1))
  if (is.data.frame(x))
    return(data.matrix(x))
  stop(paste0("Data no convertible to numeric matrix: ", typeof(x)))
}


#' Self-test of the linear algebra backend
#'
#' Calls the linear algebra routines libKriging relies on (crossprod,
#' Cholesky, triangular and general solves, inverse, rcond, QR, Woodbury
#' solve, symmetric eigen decomposition) and compares each result with an
#' oracle computed in long double without the BLAS, from one thread then from
#' \code{n_threads} concurrent threads. The BLAS / LAPACK backend (OpenBLAS,
#' MKL, Accelerate, BLIS, FlexiBLAS, reference, ...) is detected, not chosen.
#'
#' @param n_threads Number of concurrent threads (0: hardware concurrency, at
#'   least 2).
#' @return A list: \code{ok} (logical), \code{backend} (detected backend and
#'   library libKriging's dgemm is bound to), \code{report} (one line per check, then the
#'   workaround of a backend known to be faulty).
#' @examples
#' c <- linearalgebra_check()
#' cat(c$report)
#' @export
linearalgebra_check <- function(n_threads = 0) {
  linalg_check(as.integer(n_threads))
}
