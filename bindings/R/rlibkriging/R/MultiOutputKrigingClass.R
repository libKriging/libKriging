## ****************************************************************************
## This file contains stuff related to the S3 class "MultiOutputKriging".
## Same S3 pattern as KrigingClass.R.
## ****************************************************************************

#' Shortcut to provide functions to the S3 class "MultiOutputKriging"
#' @param mo A pointer to a C++ object of class "MultiOutputKriging"
#' @return An object of class "MultiOutputKriging" with methods to access and manipulate the data
classMultiOutputKriging <- function(mo) {
    class(mo) <- "MultiOutputKriging"
    for (f in c('fit', 'predict', 'print', 'show', 'simulate', 'update', 'update_simulate',
                'leaveOneOut', 'logLikelihood', 'logLikelihoodFun', 'leaveOneOutFun')) {
        eval(parse(text = paste0(
            "mo$", f, " <- function(...) ", f, "(mo,...)"
            )))
    }
    for (d in c('kernel', 'output_model', 'nb_outputs', 'X', 'Y', 'output_coordinates',
                'regmodel', 'normalize', 'optim', 'objective', 'centerY', 'scaleY',
                'theta', 'sigma2', 'beta', 'output_cov',
                'nb_components', 'pca_basis', 'pca_explained', 'pca_residual',
                'leaveOneOutMat')) {
        eval(parse(text = paste0(
            "mo$", d, " <- function() multioutputkriging_", d, "(mo)"
            )))
    }
    mo$component <- function(k) multioutputkriging_component_R(mo, k)
    mo$predictCovFactors <- function(x) multioutputkriging_predictCovFactors_R(mo, x)
    mo$set_output_coordinates <- function(t) multioutputkriging_set_output_coordinates(mo, as.matrix(t))
    mo
}

.mo_as_X <- function(object, x) {
    if (is.data.frame(x)) x <- data.matrix(x)
    if (!is.matrix(x)) x <- matrix(x, ncol = ncol(multioutputkriging_X(object)))
    x
}

.mo_as_Y <- function(object, Y) {
    if (is.data.frame(Y)) Y <- data.matrix(Y)
    if (!is.matrix(Y)) Y <- matrix(Y, ncol = multioutputkriging_nb_outputs(object))
    Y
}

#' Create an object with S3 class \code{"MultiOutputKriging"} using
#' the \pkg{libKriging} library.
#'
#' Kriging of several outputs observed at the same design points
#' (isotopic design). The outputs are the columns of \code{Y}, the
#' observations its rows, as for \code{X}. The \code{output_model} links
#' the outputs:
#' \itemize{
#'   \item \code{"pca"}, \code{"pca(K)"}, \code{"pca(v)"}: Karhunen-Loeve
#'     reduction of \code{Y} (Higdon et al. 2008), one independent
#'     \code{\link{Kriging}} per principal component score. \code{K} is an
#'     integer number of components, \code{v} (between 0 and 1) a fraction of
#'     explained variance; \code{"pca"} is \code{"pca(0.99)"}.
#'   \item \code{"shared"}: one correlation (theta) shared by all outputs,
#'     each with its own trend and variance (parallel partial Gaussian
#'     process, Gu & Berger 2016).
#'   \item \code{"separable"}: intrinsic coregionalization model,
#'     \eqn{Cov(vec Y) = \Sigma \otimes R_\theta} with a free q x q
#'     output covariance \eqn{\Sigma} (Conti & O'Hagan 2010).
#' }
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param Y Numeric matrix of outputs (n x q): one row per observation, one
#'     column per output.
#' @param X Numeric matrix of input design (n x d).
#' @param kernel Character defining the covariance model:
#'     \code{"exp"}, \code{"gauss"}, \code{"matern3_2"}, \code{"matern5_2"}.
#' @param output_model Character, see above. Default \code{"pca"}.
#' @param regmodel Universal Kriging linear trend, the same for all outputs:
#'     \code{"constant"}, \code{"linear"}, \code{"interactive"},
#'     \code{"quadratic"}.
#' @param normalize Logical. If \code{TRUE}, inputs and outputs are
#'     normalized (\code{"pca"}: each output scaled by its standard
#'     deviation before the PCA).
#' @param optim Character: \code{"BFGS"}, \code{"BFGS<k>"} (k random
#'     starts) or \code{"none"}.
#' @param objective Character: \code{"LL"} (default) or \code{"LOO"};
#'     \code{"pca"} forwards it to each latent \code{Kriging} (so any
#'     \code{Kriging} objective is allowed there).
#' @param parameters Optional named list with \code{theta} (matrix, one row
#'     per starting point) and \code{is_theta_estim}.
#' @param output_coordinates Optional output coordinates (q x d_t), e.g. the
#'     time steps of curve outputs.
#'
#' @return An object with S3 class \code{"MultiOutputKriging"}.
#'
#' @export
#'
#' @examples
#' f <- function(x, t) sin(2 * pi * (x + t)) * exp(-t)
#' t <- seq(0, 1, length.out = 20)
#' set.seed(123)
#' X <- matrix(runif(15), ncol = 1)
#' Y <- outer(X[, 1], t, f)
#' mo <- MultiOutputKriging(Y, X, kernel = "matern5_2", output_model = "pca(0.999)")
#' print(mo)
#' x <- matrix(seq(0, 1, length.out = 50), ncol = 1)
#' p <- predict(mo, x)
#' matplot(t, t(p$mean[c(10, 40), ]), type = "l", ylab = "Y(x, t)")
MultiOutputKriging <- function(Y = NULL,
                               X = NULL,
                               kernel = NULL,
                               output_model = "pca",
                               regmodel = "constant",
                               normalize = FALSE,
                               optim = "BFGS",
                               objective = "LL",
                               parameters = NULL,
                               output_coordinates = NULL) {
    stopifnot(!is.null(kernel))
    if (is.null(Y) && is.null(X)) {
        mo <- new_MultiOutputKriging(kernel, output_model)
        if (!is.null(output_coordinates))
            multioutputkriging_set_output_coordinates(mo, as.matrix(output_coordinates))
        return(classMultiOutputKriging(mo))
    }
    stopifnot(!is.null(Y), !is.null(X))
    if (is.data.frame(X)) X <- data.matrix(X)
    if (!is.matrix(X)) X <- matrix(X, ncol = 1)
    if (is.data.frame(Y)) Y <- data.matrix(Y)
    if (!is.matrix(Y)) Y <- matrix(Y, ncol = 1)
    if (!is.null(output_coordinates)) output_coordinates <- as.matrix(output_coordinates)
    mo <- new_MultiOutputKrigingFit(Y, X, kernel, output_model, regmodel, normalize,
                                    optim, objective, parameters, output_coordinates)
    classMultiOutputKriging(mo)
}

#' Fit a \code{MultiOutputKriging} object on new data.
#'
#' @param object S3 MultiOutputKriging object.
#' @param Y Numeric matrix of outputs (n x q).
#' @param X Numeric matrix of input design (n x d).
#' @param regmodel,normalize,optim,objective,parameters See
#'     \code{\link{MultiOutputKriging}}.
#' @param ... Ignored.
#'
#' @method fit MultiOutputKriging
#' @export
fit.MultiOutputKriging <- function(object, Y, X, regmodel = "constant", normalize = FALSE,
                                   optim = "BFGS", objective = "LL", parameters = NULL, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    if (is.data.frame(X)) X <- data.matrix(X)
    if (!is.matrix(X)) X <- matrix(X, ncol = 1)
    if (is.data.frame(Y)) Y <- data.matrix(Y)
    if (!is.matrix(Y)) Y <- matrix(Y, ncol = 1)
    multioutputkriging_fit(object, Y, X, regmodel, normalize, optim, objective, parameters)
    invisible(object)
}

#' Predict from a \code{MultiOutputKriging} object.
#'
#' @param object S3 MultiOutputKriging object.
#' @param x Input points (m x d matrix) where to predict.
#' @param return_stdev Logical, return the standard deviations.
#' @param return_cov Logical, return the joint covariance (mq x mq, over
#'     \code{vec(Y)}: the m points of output 1, then output 2, ...).
#' @param return_deriv Logical, return the derivatives of the mean.
#' @param ... Ignored.
#'
#' @return A list with \code{mean} (m x q), and when requested
#'     \code{stdev} (m x q), \code{cov} (mq x mq) and \code{mean_deriv}
#'     (array m x d x q).
#'
#' @method predict MultiOutputKriging
#' @export
predict.MultiOutputKriging <- function(object, x, return_stdev = TRUE, return_cov = FALSE,
                                       return_deriv = FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    multioutputkriging_predict(object, .mo_as_X(object, x), return_stdev, return_cov, return_deriv)
}

#' Simulate joint conditional paths of a \code{MultiOutputKriging} object.
#'
#' @param object S3 MultiOutputKriging object.
#' @param nsim Number of simulations.
#' @param seed Random seed.
#' @param x Input points (m x d matrix) where to simulate.
#' @param will_update Logical, keep what \code{update_simulate} needs.
#' @param ... Ignored.
#'
#' @return An array m x q x nsim: \code{sims[, , s]} is one joint draw of all
#'     outputs.
#'
#' @importFrom stats simulate
#' @method simulate MultiOutputKriging
#' @export
simulate.MultiOutputKriging <- function(object, nsim = 1, seed = 123, x, will_update = FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    multioutputkriging_simulate(object, as.integer(nsim), as.integer(seed), .mo_as_X(object, x),
                                as.logical(will_update))
}

#' Condition the last simulations of a \code{MultiOutputKriging} object on
#' new observations, without changing the model.
#'
#' @param object S3 MultiOutputKriging object, after
#'     \code{simulate(..., will_update = TRUE)}.
#' @param Y_u New outputs (n_u x q matrix).
#' @param X_u New input points (n_u x d matrix).
#' @param ... Ignored.
#'
#' @return An array m x q x nsim of updated paths.
#'
#' @method update_simulate MultiOutputKriging
#' @export
update_simulate.MultiOutputKriging <- function(object, Y_u, X_u, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    multioutputkriging_update_simulate(object, .mo_as_Y(object, Y_u), .mo_as_X(object, X_u))
}

#' Add observations to a \code{MultiOutputKriging} object.
#'
#' @param object S3 MultiOutputKriging object.
#' @param Y_u New outputs (n_u x q matrix).
#' @param X_u New input points (n_u x d matrix).
#' @param refit Logical. \code{TRUE} (default): refit on all data (PCA basis
#'     recomputed for \code{"pca"}). \code{FALSE}: keep theta (and the PCA
#'     basis), condition on the new data.
#' @param ... Ignored.
#'
#' @importFrom stats update
#' @method update MultiOutputKriging
#' @export
update.MultiOutputKriging <- function(object, Y_u, X_u, refit = TRUE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    multioutputkriging_update(object, .mo_as_Y(object, Y_u), .mo_as_X(object, X_u), as.logical(refit))
    invisible(object)
}

#' Leave-one-out mean squared error of a \code{MultiOutputKriging} object,
#' over all n x q outputs (use \code{object$leaveOneOutMat()} for the LOO
#' means and standard deviations).
#'
#' @param object S3 MultiOutputKriging object.
#' @param ... Ignored.
#'
#' @method leaveOneOut MultiOutputKriging
#' @export
leaveOneOut.MultiOutputKriging <- function(object, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    multioutputkriging_leaveOneOut(object)
}

#' Summed log-likelihood of a \code{"shared"} or \code{"separable"}
#' \code{MultiOutputKriging} object at its fitted theta.
#'
#' @param object S3 MultiOutputKriging object.
#' @param ... Ignored.
#'
#' @method logLikelihood MultiOutputKriging
#' @export
logLikelihood.MultiOutputKriging <- function(object, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    multioutputkriging_logLikelihood(object)
}

#' Summed concentrated log-likelihood of a \code{"shared"} or
#' \code{"separable"} \code{MultiOutputKriging} object.
#'
#' @param object S3 MultiOutputKriging object.
#' @param theta Correlation ranges: a vector of length d, or a matrix with
#'     one row per value of theta.
#' @param return_grad Logical, also return the gradient in theta.
#' @param ... Ignored.
#'
#' @return A list with \code{logLikelihood} and, when requested,
#'     \code{logLikelihoodGrad}, one row per row of \code{theta}.
#'
#' @method logLikelihoodFun MultiOutputKriging
#' @export
logLikelihoodFun.MultiOutputKriging <- function(object, theta, return_grad = FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    d <- ncol(multioutputkriging_X(object))
    if (!is.matrix(theta)) theta <- matrix(theta, ncol = d)
    out <- list(logLikelihood = matrix(NA, nrow = nrow(theta)),
                logLikelihoodGrad = matrix(NA, nrow = nrow(theta), ncol = d))
    for (i in seq_len(nrow(theta))) {
        ll <- multioutputkriging_logLikelihoodFun(object, theta[i, ], isTRUE(return_grad))
        out$logLikelihood[i] <- ll$logLikelihood
        if (isTRUE(return_grad)) out$logLikelihoodGrad[i, ] <- ll$logLikelihoodGrad
    }
    if (!isTRUE(return_grad)) out$logLikelihoodGrad <- NULL
    out
}

#' Leave-one-out mean squared error of a \code{"shared"} or
#' \code{"separable"} \code{MultiOutputKriging} object, summed over outputs
#' on the normalized scale, as a function of theta.
#'
#' @param object S3 MultiOutputKriging object.
#' @param theta Correlation ranges: a vector of length d, or a matrix with
#'     one row per value of theta.
#' @param return_grad Logical, also return the gradient in theta.
#' @param ... Ignored.
#'
#' @return A list with \code{leaveOneOut} and, when requested,
#'     \code{leaveOneOutGrad}, one row per row of \code{theta}.
#'
#' @method leaveOneOutFun MultiOutputKriging
#' @export
leaveOneOutFun.MultiOutputKriging <- function(object, theta, return_grad = FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    d <- ncol(multioutputkriging_X(object))
    if (!is.matrix(theta)) theta <- matrix(theta, ncol = d)
    out <- list(leaveOneOut = matrix(NA, nrow = nrow(theta)),
                leaveOneOutGrad = matrix(NA, nrow = nrow(theta), ncol = d))
    for (i in seq_len(nrow(theta))) {
        r <- multioutputkriging_leaveOneOutFun(object, theta[i, ], isTRUE(return_grad))
        out$leaveOneOut[i] <- r$leaveOneOut
        if (isTRUE(return_grad)) out$leaveOneOutGrad[i, ] <- r$leaveOneOutGrad
    }
    if (!isTRUE(return_grad)) out$leaveOneOutGrad <- NULL
    out
}

multioutputkriging_component_R <- function(object, k) {
    classKriging(multioutputkriging_component(object, as.integer(k)))
}

multioutputkriging_predictCovFactors_R <- function(object, x) {
    multioutputkriging_predictCovFactors(object, .mo_as_X(object, x))
}

#' Print a \code{MultiOutputKriging} object.
#'
#' @param x S3 MultiOutputKriging object.
#' @param ... Ignored.
#'
#' @method print MultiOutputKriging
#' @export
print.MultiOutputKriging <- function(x, ...) {
    cat(multioutputkriging_summary(x))
    invisible(x)
}

#' @title Accessors of a \code{MultiOutputKriging} object
#' @description Same generics as for \code{Kriging}: \code{kernel},
#'     \code{X}, \code{centerY} and \code{scaleY} (1 x q), \code{regmodel},
#'     \code{normalize}; for \code{"shared"} and \code{"separable"} also
#'     \code{theta} (d), \code{sigma2} (q) and \code{beta} (p x q).
#' @param object S3 MultiOutputKriging object.
#' @param ... Ignored.
#' @name MultiOutputKriging-accessors
NULL

#' @rdname MultiOutputKriging-accessors
#' @method kernel MultiOutputKriging
#' @export
kernel.MultiOutputKriging <- function(object, ...) multioutputkriging_kernel(object)
#' @rdname MultiOutputKriging-accessors
#' @method X MultiOutputKriging
#' @export
X.MultiOutputKriging <- function(object, ...) multioutputkriging_X(object)
#' @rdname MultiOutputKriging-accessors
#' @method centerY MultiOutputKriging
#' @export
centerY.MultiOutputKriging <- function(object, ...) multioutputkriging_centerY(object)
#' @rdname MultiOutputKriging-accessors
#' @method scaleY MultiOutputKriging
#' @export
scaleY.MultiOutputKriging <- function(object, ...) multioutputkriging_scaleY(object)
#' @rdname MultiOutputKriging-accessors
#' @method regmodel MultiOutputKriging
#' @export
regmodel.MultiOutputKriging <- function(object, ...) multioutputkriging_regmodel(object)
#' @rdname MultiOutputKriging-accessors
#' @method normalize MultiOutputKriging
#' @export
normalize.MultiOutputKriging <- function(object, ...) multioutputkriging_normalize(object)
#' @rdname MultiOutputKriging-accessors
#' @method theta MultiOutputKriging
#' @export
theta.MultiOutputKriging <- function(object, ...) multioutputkriging_theta(object)
#' @rdname MultiOutputKriging-accessors
#' @method sigma2 MultiOutputKriging
#' @export
sigma2.MultiOutputKriging <- function(object, ...) multioutputkriging_sigma2(object)
#' @rdname MultiOutputKriging-accessors
#' @method beta MultiOutputKriging
#' @export
beta.MultiOutputKriging <- function(object, ...) multioutputkriging_beta(object)
