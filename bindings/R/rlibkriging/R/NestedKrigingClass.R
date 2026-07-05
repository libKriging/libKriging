## ****************************************************************************
## This file contains stuff related to the S3 class "NestedKriging".
## Same S3 pattern as KrigingClass.R.
## ****************************************************************************

#' Shortcut to provide functions to the S3 class "NestedKriging"
#' @param nk A pointer to a C++ object of class "NestedKriging"
#' @return An object of class "NestedKriging" with methods to access and manipulate the data
classNestedKriging <- function(nk) {
    class(nk) <- "NestedKriging"
    for (f in c('predict', 'print', 'show')) {
        eval(parse(text = paste0(
            "nk$", f, " <- function(...) ", f, "(nk,...)"
            )))
    }
    for (d in c('kernel', 'aggregation', 'nb_groups', 'groups', 'theta', 'sigma2', 'beta0', 'X', 'y')) {
        eval(parse(text = paste0(
            "nk$", d, " <- function() nestedkriging_", d, "(nk)"
            )))
    }
    nk
}

#' Create an object with S3 class \code{"NestedKriging"} using
#' the \pkg{libKriging} library.
#'
#' Divide-and-conquer Kriging for large designs: the data are partitioned in
#' \code{nb_groups} groups, one \code{\link{Kriging}} submodel is fitted per
#' group (then all share a common prior), and predictions are aggregated.
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param y Numeric vector of response values.
#' @param X Numeric matrix of input design.
#' @param kernel Character defining the covariance model:
#'     \code{"exp"}, \code{"gauss"}, \code{"matern3_2"}, \code{"matern5_2"}.
#' @param nb_groups Number of submodels (each of size ~ nrow(X)/nb_groups).
#' @param aggregation Character: \code{"NK"} (optimal nested-kriging
#'     aggregation of Rulliere et al. 2018, default), \code{"PoE"},
#'     \code{"gPoE"}, \code{"BCM"} or \code{"rBCM"}.
#' @param partition Character: \code{"kmeans"} (default) or \code{"random"}.
#' @param seed Integer seed for the partition.
#' @param regmodel Linear trend; \code{"NK"} aggregation requires
#'     \code{"constant"}.
#' @param optim Character, hyper-parameter optimization method of the
#'     submodels: \code{"BFGS"} or \code{"none"}.
#' @param objective Character: \code{"LL"}, \code{"LOO"} or \code{"LMP"}.
#' @param parameters Initial or fixed values for the hyper-parameters
#'     (named list with \code{"sigma2"}, \code{"theta"}, \code{"beta"}).
#'
#' @return An object with S3 class \code{"NestedKriging"}, to be used with
#'     its \code{predict} method.
#'
#' @export
#'
#' @examples
#' f <- function(X) apply(X, 1, function(x) sin(3 * x[1]) + cos(5 * x[2]))
#' set.seed(123)
#' X <- matrix(runif(2 * 400), ncol = 2)
#' y <- f(X)
#' k <- NestedKriging(y, X, kernel = "matern5_2", nb_groups = 8)
#' x <- matrix(runif(2 * 100), ncol = 2)
#' p <- predict(k, x)
NestedKriging <- function(y = NULL,
                          X = NULL,
                          kernel = NULL,
                          nb_groups = NULL,
                          aggregation = "NK",
                          partition = "kmeans",
                          seed = 123,
                          regmodel = "constant",
                          optim = "BFGS",
                          objective = "LL",
                          parameters = NULL) {
    stopifnot(!is.null(y), !is.null(X), !is.null(kernel), !is.null(nb_groups))
    nk <- new_NestedKrigingFit(y, X, kernel, nb_groups,
                               aggregation, partition, seed,
                               regmodel, optim, objective, parameters)
    classNestedKriging(nk)
}

#' Predict from a \code{NestedKriging} object.
#'
#' @param object S3 NestedKriging object.
#' @param x Input points (matrix) where to predict.
#' @param return_stdev Logical, return standard deviation (default TRUE).
#' @param ... Ignored.
#'
#' @return A list with elements \code{mean} and (optionally) \code{stdev}.
#'
#' @method predict NestedKriging
#' @export
predict.NestedKriging <- function(object, x, return_stdev = TRUE, ...) {
    if (!is.matrix(x)) x <- matrix(x, ncol = ncol(nestedkriging_X(object)))
    nestedkriging_predict(object, x, return_stdev)
}

#' Print a \code{NestedKriging} object.
#'
#' @param x S3 NestedKriging object.
#' @param ... Ignored.
#'
#' @method print NestedKriging
#' @export
print.NestedKriging <- function(x, ...) {
    cat(nestedkriging_summary(x))
    invisible(x)
}
