## ****************************************************************************
## This file contains stuff related to the S3 class "Kriging".
## As an S3 class, it has no formal definition.
## ****************************************************************************

#' Shortcut to provide functions to the S3 class "Kriging"
#' @param nk A pointer to a C++ object of class "Kriging"
#' @return An object of class "Kriging" with methods to access and manipulate the data
classKriging <- function(nk) {
    class(nk) <- "Kriging"
    # This will allow to call methods (like in Python/Matlab/Octave) using `k$m(...)` as well as R-style `m(k, ...)`.
    for (f in c('as.list','copy','fit','save',
    'covMat','leaveOneOut','leaveOneOutFun','leaveOneOutVec',
    'logLikelihood','logLikelihoodFun','logMargPost','logMargPostFun',
    'predict','print','show','simulate','update', 'update_simulate')) {
        eval(parse(text=paste0(
            "nk$", f, " <- function(...) ", f, "(nk,...)"
            )))
    }
    # This will allow to access kriging data/props using `k$d()`
    for (d in c('kernel','optim','objective','X','centerX','scaleX','y','centerY','scaleY','regmodel','normalize','F','T','M','z','beta','is_beta_estim','theta','is_theta_estim','sigma2','is_sigma2_estim','noise_model','nugget','is_nugget_estim','noise')) {
        eval(parse(text=paste0(
            "nk$", d, " <- function() kriging_", d, "(nk)"
            )))
    }
    nk
}

#' Create an object with S3 class \code{"Kriging"} using
#' the \pkg{libKriging} library.
#'
#' The hyper-parameters (variance and vector of correlation ranges)
#' are estimated thanks to the optimization of a criterion given by
#' \code{objective}, using the method given in \code{optim}.
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param y Numeric vector of response values.
#' @param X Numeric matrix of input design.
#' @param kernel Character defining the covariance model:
#'     \code{"exp"}, \code{"gauss"}, \code{"matern3_2"}, \code{"matern5_2"}.
#' @param regmodel Universal Kriging linear trend: \code{"constant"}, 
#'     \code{"linear"}, \code{"interactive"}, \code{"quadratic"}.
#' @param normalize Logical. If \code{TRUE} both the input matrix
#'     \code{X} and the response \code{y} in normalized to take
#'     values in the interval \eqn{[0, 1]}.
#' @param optim Character giving the Optimization method used to fit
#'     hyper-parameters. Possible values are: \code{"BFGS"} and
#'     \code{"none"}, the later simply keeping the values given in
#'     \code{parameters}. The method \code{"BFGS"} uses the gradient
#'     of the objective (note that \code{"BFGS10"} means 10
#'     multi-start of BFGS).
#' @param objective Character giving the objective function to
#'     optimize. Possible values are: \code{"LL"} for the
#'     Log-Likelihood, \code{"LOO"} for the Leave-One-Out sum of
#'     squares and \code{"LMP"} for the Log-Marginal Posterior.
#' @param parameters Initial values for the hyper-parameters. When
#'     provided this must be named list with elements \code{"sigma2"}
#'     and \code{"theta"} containing the initial value(s) for the
#'     variance and for the range parameters. If \code{theta} is a
#'     matrix with more than one row, each row is used as a starting
#'     point for optimization.
#' @param noise Either a numeric vector of per-observation noise variances,
#'     or \code{"nugget"} to estimate a homogeneous nugget, or
#'     \code{NULL} (default) for noise-free interpolation.
#'
#' @return An object with S3 class \code{"Kriging"}. Should be used
#'     with its \code{predict}, \code{simulate}, \code{update}
#'     methods.
#'
#' @export
#' @useDynLib rlibkriging, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom utils methods
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#' ## fit and print
#' k <- Kriging(y, X, kernel = "matern3_2")
#' print(k)
#'
#' x <- as.matrix(seq(from = 0, to = 1, length.out = 101))
#' p <- predict(k, x = x, return_stdev = TRUE, return_cov = FALSE)
#'
#' plot(f)
#' points(X, y)
#' lines(x, p$mean, col = "blue")
#' polygon(c(x, rev(x)), c(p$mean - 2 * p$stdev, rev(p$mean + 2 * p$stdev)),
#' border = NA, col = rgb(0, 0, 1, 0.2))
#'
#' s <- simulate(k, nsim = 10, seed = 123, x = x)
#'
#' matlines(x, s, col = rgb(0, 0, 1, 0.2), type = "l", lty = 1)
Kriging <- function(y=NULL, X=NULL, kernel=NULL,
                    noise = NULL,
                    regmodel = c("constant", "linear", "interactive", "none"),
                    normalize = FALSE,
                    optim = c("BFGS", "none"),
                    objective = c("LL", "LOO", "LMP"),
                    parameters = NULL) {

    regmodel <- match.arg(regmodel)
    objective <- match.arg(objective)
    if (is.character(optim)) optim <- optim[1] #optim <- match.arg(optim) because we can use BFGS10 for 10 (multistart) BFGS

    # Determine noise_model from noise argument:
    #   "nugget"    -> "nugget" (estimated homoscedastic nugget)
    #   numeric     -> "heterogeneous" (known per-point noise variance)
    if (is.null(noise)) {
        noise_model <- "none"
        noise_vec <- NULL
    } else if (is.character(noise) && tolower(noise) == "nugget") {
        noise_model <- "nugget"
        noise_vec <- NULL
    } else if (is.numeric(noise)) {
        noise_model <- "heterogeneous"
        if (length(noise) == 1)
            noise_vec <- rep(noise, NROW(y))
        else
            noise_vec <- noise
    } else {
        stop("'noise' must be NULL, \"nugget\", or a numeric scalar/vector")
    }

    if (is.character(y) && is.null(X) && is.null(kernel)) # just first arg for kernel, without naming
        nk <- new_Kriging(kernel = y, noise_model = noise_model)
    else if (is.null(y) && is.null(X) && !is.null(kernel))
        nk <- new_Kriging(kernel = kernel, noise_model = noise_model)
    else
        nk <- new_KrigingFit(y = y, X = X, kernel = kernel,
                      noise_model = noise_model,
                      noise = noise_vec,
                      regmodel = regmodel,
                      normalize = normalize,
                      optim = optim,
                      objective = objective,
                      parameters = parameters)
    return(classKriging(nk))
}


#' Coerce a \code{Kriging} Object into a List
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param x An object with class \code{"Kriging"}.
#' @param ... Ignored
#'
#' @return A list with its elements copying the content of the
#'     \code{Kriging} object fields: \code{kernel}, \code{optim},
#'     \code{objective}, \code{theta} (vector of ranges),
#'     \code{sigma2} (variance), \code{X}, \code{centerX},
#'     \code{scaleX}, \code{y}, \code{centerY}, \code{scaleY},
#'     \code{regmodel}, \code{F}, \code{T}, \code{M}, \code{z},
#'     \code{beta}.
#'
#' @export
#' @method as.list Kriging
#' @aliases as.list,Kriging,Kriging-method
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x ) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, kernel = "matern3_2")
#'
#' l <- as.list(k)
#' cat(paste0(names(l), " =" , l, collapse = "\n"))
as.list.Kriging <- function(x, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    kriging_model(x)
}


#' Print the content of a \code{Kriging} object.
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param x A (S3) \code{Kriging} Object.
#' @param ... Ignored.
#'
#' @return String of printed object.
#'
#' @export
#' @method print Kriging
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, "matern3_2")
#'
#' print(k)
#' ## same thing
#' k
print.Kriging <- function(x, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    p = kriging_summary(x)
    cat(p)
    invisible(p)
}


#' Fit \code{Kriging} object on given data.
#'
#' The hyper-parameters (variance and vector of correlation ranges)
#' are estimated thanks to the optimization of a criterion given by
#' \code{objective}, using the method given in \code{optim}.
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object S3 Kriging object.
#' @param y Numeric vector of response values.
#' @param X Numeric matrix of input design.
#' @param regmodel Universal Kriging linear trend: \code{"constant"}, 
#'     \code{"linear"}, \code{"interactive"}, \code{"quadratic"}.
#' @param normalize Logical. If \code{TRUE} both the input matrix
#'     \code{X} and the response \code{y} in normalized to take
#'     values in the interval \eqn{[0, 1]}.
#' @param optim Character giving the Optimization method used to fit
#'     hyper-parameters. Possible values are: \code{"BFGS"} and
#'     \code{"none"}, the later simply keeping the values given in
#'     \code{parameters}. The method \code{"BFGS"} uses the gradient
#'     of the objective (note that \code{"BFGS10"} means 10 multi-start of BFGS).
#' @param objective Character giving the objective function to
#'     optimize. Possible values are: \code{"LL"} for the
#'     Log-Likelihood, \code{"LOO"} for the Leave-One-Out sum of
#'     squares and \code{"LMP"} for the Log-Marginal Posterior.
#' @param parameters Initial values for the hyper-parameters. When
#'     provided this must be named list with elements \code{"sigma2"}
#'     and \code{"theta"} containing the initial value(s) for the
#'     variance and for the range parameters. If \code{theta} is a
#'     matrix with more than one row, each row is used as a starting
#'     point for optimization.
#' @param noise Either a numeric vector of per-observation noise variances,
#'     or \code{"nugget"} to estimate a homogeneous nugget, or
#'     \code{NULL} (default) for noise-free interpolation.
#' @param ... Ignored.
#'
#' @return No return value. Kriging object argument is modified.
#'
#' @method fit Kriging
#' @export
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#' points(X, y, col = "blue", pch = 16)
#'
#' k <- Kriging("matern3_2")
#' print(k)
#'
#' fit(k,y,X)
#' print(k)
fit.Kriging <- function(object, y, X,
                    noise = NULL,
                    regmodel = c("constant", "linear", "interactive", "none"),
                    normalize = FALSE,
                    optim = c("BFGS", "none"),
                    objective = c("LL", "LOO", "LMP"),
                    parameters = NULL, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)

    regmodel <- match.arg(regmodel)
    objective <- match.arg(objective)
    if (is.character(optim)) optim <- optim[1] #optim <- match.arg(optim) because we can use BFGS10 for 10 (multistart) BFGS

    kriging_fit(object, y, X,
                    noise = noise,
                    regmodel,
                    normalize,
                    optim ,
                    objective,
                    parameters)

    invisible(NULL)
}


#' Predict from a \code{Kriging} object.
#'
#' Given "new" input points, the method compute the expectation,
#' variance and (optionnally) the covariance of the corresponding
#' stochastic process, conditional on the values at the input points
#' used when fitting the model.
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object S3 Kriging object.
#' @param x Input points where the prediction must be computed.
#' @param return_stdev \code{Logical}. If \code{TRUE} the standard deviation
#'     is returned.
#' @param return_cov \code{Logical}. If \code{TRUE} the covariance matrix of
#'     the predictions is returned.
#' @param return_deriv \code{Logical}. If \code{TRUE} the derivatives of mean and sd
#'     of the predictions are returned.
#' @param ... Ignored.
#'
#' @return A list containing the element \code{mean} and possibly
#'     \code{stdev} and  \code{cov}.
#'
#' @note The names of the formal arguments differ from those of the
#'     \code{predict} methods for the S4 classes \code{"km"} and
#'     \code{"KM"}. The formal \code{x} corresponds to
#'     \code{newdata}, \code{stdev} corresponds to \code{se.compute}
#'     and \code{cov} to \code{cov.compute}. These names are chosen
#'     \pkg{Python} and \pkg{Octave} interfaces to \pkg{libKriging}.
#'
#' @method predict Kriging
#' @export
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#' points(X, y, col = "blue", pch = 16)
#'
#' k <- Kriging(y, X, "matern3_2")
#'
#' x <-seq(from = 0, to = 1, length.out = 101)
#' p <- predict(k, x)
#'
#' lines(x, p$mean, col = "blue")
#' polygon(c(x, rev(x)), c(p$mean - 2 * p$stdev, rev(p$mean + 2 * p$stdev)),
#'  border = NA, col = rgb(0, 0, 1, 0.2))
predict.Kriging <- function(object, x, return_stdev = TRUE, return_cov = FALSE, return_deriv = FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    ## manage the data frame case. Ideally we should then warn
    if (is.data.frame(x)) x = data.matrix(x)
    if (!is.matrix(x)) x=matrix(x,ncol=ncol(object$X()))
    return(kriging_predict(object, x, return_stdev, return_cov, return_deriv))
}


#' Simulation from a \code{Kriging} model object.
#'
#' This method draws paths of the stochastic process at new input
#' points conditional on the values at the input points used in the
#' fit.
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object S3 Kriging object.
#' @param nsim Number of simulations to perform.
#' @param seed Random seed used.
#' @param x Points in model input space where to simulate.
#' @param with_noise Logical or numeric specification controlling whether
#'     observation noise is included in the simulated paths. Use
#'     \code{TRUE} to include fitted noise when available, \code{FALSE}
#'     to exclude nugget noise, or \code{NULL} for the default behaviour.
#' @param will_update Set to \code{TRUE} if you plan to call
#'     \code{update_simulate()} later.
#' @param ... Ignored.
#'
#' @return a matrix with \code{nrow(x)} rows and \code{nsim}
#'     columns containing the simulated paths at the inputs points
#'     given in \code{x}.
#'
#' @note The names of the formal arguments differ from those of the
#'     \code{simulate} methods for the S4 classes \code{"km"} and
#'     \code{"KM"}. The formal \code{x} corresponds to
#'     \code{newdata}. These names are chosen \pkg{Python} and
#'     \pkg{Octave} interfaces to \pkg{libKriging}.
#'
#'
#' @importFrom stats runif
#' @method simulate Kriging
#' @export
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#' points(X, y, col = "blue")
#'
#' k <- Kriging(y, X, kernel = "matern3_2")
#'
#' x <- seq(from = 0, to = 1, length.out = 101)
#' s <- simulate(k, nsim = 3, x = x)
#'
#' lines(x, s[ , 1], col = "blue")
#' lines(x, s[ , 2], col = "blue")
#' lines(x, s[ , 3], col = "blue")
simulate.Kriging <- function(object, nsim = 1, seed = 123, x, with_noise = NULL, will_update = FALSE, ...) {
    args <- list(...)
    # Accept with_nugget as alias for with_noise (backward compat for NuggetKriging)
    if (!is.null(args$with_nugget)) {
        if (is.null(with_noise)) with_noise <- args$with_nugget
        args$with_nugget <- NULL
    }
    is_nugget <- kriging_noise_model(object) == "nugget"
    if (is.logical(with_noise) && !with_noise && is_nugget) {
        # with_noise=FALSE for a nugget model: simulate without nugget noise.
        # Pass NULL so C++ dispatches to the nugget-correct no-noise overload
        # (which uses alpha-scaled correlations).  Passing FALSE would trigger
        # the hetero path with noise=0 and wrong (non-alpha) correlation factors.
        with_noise <- NULL
    } else if (is.null(with_noise) && is_nugget) {
        # For nugget models, default to including the nugget noise
        # (backward compat: old simulate.NuggetKriging defaulted to with_nugget=TRUE)
        with_noise <- TRUE
    }
    if (length(args) > 0) warnOnDots(args)
    ## manage the data frame case. Ideally we should then warn
    if (is.data.frame(x)) x = data.matrix(x)
    if (!is.matrix(x)) x=matrix(x,ncol=ncol(object$X()))
    if (is.null(seed)) seed <- floor(runif(1) * 99999)
    return(kriging_simulate(object, nsim = nsim, seed = seed, X_n = x, with_noise = with_noise, will_update = will_update))
}

#' Update previous simulation of a \code{Kriging} model object.
#'
#' This method draws paths of the stochastic process conditional on the values at the input points used in the
#' fit, plus the new input points and their values given as argument (knonw as 'update' points).
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object S3 Kriging object.
#' @param y_u Numeric vector of new responses (output).
#' @param X_u Numeric matrix of new input points.
#' @param noise_u Optional numeric vector of observation noise variances
#'     attached to \code{y_u}.
#' @param ... Ignored.
#'
#' @return a matrix with \code{nrow(x)} rows and \code{nsim}
#'     columns containing the simulated paths at the inputs points
#'     given in \code{x}.
#'
#' @method update_simulate Kriging
#' @export
#' 
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#' points(X, y, col = "blue")
#' 
#' k <- Kriging(y, X, kernel = "matern3_2")
#' 
#' x <- seq(from = 0, to = 1, length.out = 101)
#' s <- k$simulate(nsim = 3, x = x, will_update = TRUE)
#' 
#' lines(x, s[ , 1], col = "blue")
#' lines(x, s[ , 2], col = "blue")
#' lines(x, s[ , 3], col = "blue")
#' 
#' X_u <- as.matrix(runif(3))
#' y_u <- f(X_u)
#' points(X_u, y_u, col = "red")
#' 
#' su <- k$update_simulate(y_u, X_u)
#' 
#' lines(x, su[ , 1], col = "blue", lty=2)
#' lines(x, su[ , 2], col = "blue", lty=2)
#' lines(x, su[ , 3], col = "blue", lty=2)
update_simulate.Kriging <- function(object, y_u, ..., X_u = NULL, noise_u = NULL) {
    # Support two calling conventions matching C++ overloads:
    #   update_simulate(y_u, X_u)            - no noise (Nugget / pure Kriging)
    #   update_simulate(y_u, noise_u, X_u)   - with known observation noise
    # Named arguments also accepted: update_simulate(y_u, noise_u=..., X_u=...)
    args <- list(...)
    if (is.null(X_u) && (!is.null(args$X_u) || !is.null(args$noise_u))) {
        X_u <- args$X_u
        noise_u <- if (is.null(noise_u)) args$noise_u else noise_u
    } else if (is.null(X_u) && length(args) >= 2) {
        noise_u <- args[[1]]
        X_u <- args[[2]]
    } else if (is.null(X_u) && length(args) == 1) {
        X_u <- args[[1]]
    } else if (is.null(X_u)) {
        stop("update_simulate: X_u is required")
    }
    extra <- args[!names(args) %in% c("X_u", "noise_u")]
    if (length(extra) > 0) warnOnDots(extra)
    if (is.data.frame(X_u)) X_u = data.matrix(X_u)
    if (!is.matrix(X_u)) X_u <- matrix(X_u, ncol = ncol(object$X()))
    if (is.data.frame(y_u)) y_u = data.matrix(y_u)
    if (!is.matrix(y_u)) y_u <- matrix(y_u, ncol = 1)
    return(kriging_update_simulate(object, y_u, noise_u = noise_u, X_u))
}

#' Update a \code{Kriging} model object with new points
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object S3 Kriging object.
#' @param y_u Numeric vector of new responses (output).
#' @param X_u Numeric matrix of new input points.
#' @param noise_u Optional numeric vector of observation noise variances
#'     attached to \code{y_u}.
#' @param refit Logical. If \code{TRUE} the model is refitted (default is \code{TRUE}).
#' @param ... Ignored.
#'
#' @return No return value. Kriging object argument is modified.
#'
#' @section Caution: The method \emph{does not return the updated
#'     object}, but instead changes the content of
#'     \code{object}. This behaviour is quite unusual in R and
#'     differs from the behaviour of \code{\link[DiceKriging]{update.km}}
#'     in \pkg{DiceKriging} and the \code{update} method for class
#'     \code{KM}.
#'
#' @method update Kriging
#' @export
#'
#' @examples
#' f <- function(x) 1- 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x)*x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#' points(X, y, col = "blue")
#'
#' k <- Kriging(y, X, "matern3_2")
#'
#' x <- seq(from = 0, to = 1, length.out = 101)
#' p <- predict(k, x)
#' lines(x, p$mean, col = "blue")
#' polygon(c(x, rev(x)), c(p$mean - 2 * p$stdev, rev(p$mean + 2 * p$stdev)),
#'  border = NA, col = rgb(0, 0, 1, 0.2))
#'
#' X_u <- as.matrix(runif(3))
#' y_u <- f(X_u)
#' points(X_u, y_u, col = "red")
#'
#' ## change the content of the object 'k'
#' update(k, y_u, X_u)
#'
#' ## include design points to see interpolation
#' x <- sort(c(X,X_u,seq(from = 0, to = 1, length.out = 101)))
#' p2 <- predict(k, x)
#' lines(x, p2$mean, col = "red")
#' polygon(c(x, rev(x)), c(p2$mean - 2 * p2$stdev, rev(p2$mean + 2 * p2$stdev)),
#'  border = NA, col = rgb(1, 0, 0, 0.2))
update.Kriging <- function(object, y_u, ..., X_u = NULL, noise_u = NULL, refit = TRUE) {
    # Support multiple calling conventions matching C++ overloads:
    #   update(y_u, X_u)                    - no noise, refit defaults TRUE
    #   update(y_u, X_u, refit)             - no noise, positional refit (logical)
    #   update(y_u, noise_u, X_u)           - with noise, refit defaults TRUE
    #   update(y_u, noise_u, X_u, refit)    - with noise, positional refit
    # Named arguments are also accepted.
    args <- list(...)
    arg_names <- names(args)
    named_mask <- if (is.null(arg_names)) rep(FALSE, length(args)) else nzchar(arg_names)
    pos_args <- args[!named_mask]
    named_args <- args[named_mask]

    refit <- if (!is.null(named_args$refit)) named_args$refit else refit
    noise_u <- if (is.null(noise_u)) named_args$noise_u else noise_u
    X_u <- if (is.null(X_u)) named_args$X_u else X_u

    if (is.null(X_u)) {
        n_pos <- length(pos_args)
        if (n_pos == 1) {
            X_u <- pos_args[[1]]
        } else if (n_pos == 2 && is.logical(pos_args[[2]])) {
            X_u <- pos_args[[1]]
            refit <- pos_args[[2]]
        } else if (n_pos == 2) {
            noise_u <- pos_args[[1]]
            X_u <- pos_args[[2]]
        } else if (n_pos == 3) {
            noise_u <- pos_args[[1]]
            X_u <- pos_args[[2]]
            refit <- pos_args[[3]]
        } else {
            stop("update: X_u is required")
        }
    }
    extra <- named_args[!names(named_args) %in% c("refit", "noise_u", "X_u")]
    if (length(extra) > 0) warnOnDots(extra)

    if (is.data.frame(X_u)) X_u = data.matrix(X_u)
    if (!is.matrix(X_u)) X_u <- matrix(X_u, ncol = ncol(object$X()))
    if (is.data.frame(y_u)) y_u = data.matrix(y_u)
    if (!is.matrix(y_u)) y_u <- matrix(y_u, ncol = 1)
    kriging_update(object, y_u, X_u, noise_u = noise_u, refit)

    invisible(NULL)
}

#' Save a Kriging Model to a file storage
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object An S3 Kriging object.
#' @param filename File name to save in.
#' @param ... Not used.
#'
#' @return The loaded Kriging object.
#'
#' @method save Kriging
#' @export
#' @aliases save,Kriging,Kriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, kernel = "matern3_2", objective="LMP")
#' print(k)
#'
#' outfile = tempfile("k.json") 
#' save(k,outfile)
save.Kriging <- function(object, filename, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    if (!is.character(filename))
        stop("'filename' must be a string")

    kriging_save(object, filename)

    invisible(NULL)
}

#' Load a Kriging Model from a file storage
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param filename File name to load from.
#' @param ... Not used.
#'
#' @return The loaded Kriging object.
#'
#' @export
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, kernel = "matern3_2", objective="LMP")
#' print(k)
#'
#' outfile = tempfile("k.json")
#' save(k,outfile)
#'
#' print(load.Kriging(outfile)) 
load.Kriging <- function(filename, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    if (!is.character(filename))
        stop("'filename' must be a string")
    return(classKriging( kriging_load(filename)))
}

#' Compute Covariance Matrix of Kriging Model
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object An S3 Kriging object.
#' @param x1 Numeric matrix of input points.
#' @param x2 Numeric matrix of input points.
#' @param ... Not used.
#' 
#' @return A matrix of the covariance matrix of the Kriging model.
#' 
#' @method covMat Kriging
#' @export
#' @aliases covMat,Kriging,Kriging-method
#' 
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, kernel = "gauss")
#' 
#' x1 = runif(10)
#' x2 = runif(10)
#' 
#' covMat(k, x1, x2)
covMat.Kriging <- function(object, x1, x2, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    if (is.data.frame(x1)) x1 = data.matrix(x1)
    if (is.data.frame(x2)) x2 = data.matrix(x2)
    if (!is.matrix(x1)) x1 = matrix(x1, ncol = ncol(object$X()))
    if (!is.matrix(x2)) x2 = matrix(x2, ncol = ncol(object$X()))
    return(kriging_covMat(object, x1, x2))
}

#' Compute Log-Likelihood of Kriging Model
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object An S3 Kriging object.
#' @param theta A numeric vector of (positive) range parameters at
#'     which the log-likelihood will be evaluated.
#' @param return_grad Logical. Should the function return the gradient?
#' @param return_hess Logical. Should the function return Hessian?
#' @param bench Logical. Should the function display benchmarking output?
#' @param ... Not used.
#'
#' @return The log-Likelihood computed for given
#'     \eqn{\boldsymbol{theta}}{\theta}.
#'
#' @method logLikelihoodFun Kriging
#' @export
#' @aliases logLikelihoodFun,Kriging,Kriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, kernel = "matern3_2")
#' print(k)
#'
#' ll <- function(theta) logLikelihoodFun(k, theta)$logLikelihood
#'
#' t <- seq(from = 0.001, to = 2, length.out = 101)
#' plot(t, ll(t), type = 'l')
#' abline(v = k$theta(), col = "blue")
logLikelihoodFun.Kriging <- function(object, theta,
                                  return_grad = FALSE, return_hess = FALSE, bench=FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    if (is.data.frame(theta)) theta = data.matrix(theta)
    # Noise/nugget models have an extra parameter (sigma2 or alpha) beyond theta
    nparams <- ncol(object$X()) +
      if (kriging_noise_model(object) %in% c("nugget", "heterogeneous")) 1L else 0L
    if (!is.matrix(theta)) theta <- matrix(theta, ncol = nparams)
    d <- ncol(theta)
    n <- nrow(theta)
    out <- list(logLikelihood = matrix(NA, nrow = n),
                logLikelihoodGrad = matrix(NA, nrow = n, ncol = d))
    if (isTRUE(return_hess)) out$logLikelihoodHess <- array(NA, dim = c(n, d, d))
    for (i in 1:n) {
        ll <- kriging_logLikelihoodFun(object, theta[i, ],
                                    return_grad = isTRUE(return_grad),
                                    return_hess = isTRUE(return_hess),
                                    bench = isTRUE(bench))
        out$logLikelihood[i] <- ll$logLikelihood
        if (isTRUE(return_grad)) out$logLikelihoodGrad[i, ] <- ll$logLikelihoodGrad
    }
    if (!isTRUE(return_grad)) out$logLikelihoodGrad <- NULL
    if (!isTRUE(return_hess)) out$logLikelihoodHess <- NULL

    return(out)
}


#' Get Log-Likelihood of Kriging Model
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object An S3 Kriging object.
#' @param ... Not used.
#'
#' @return The log-Likelihood computed for fitted
#'     \eqn{\boldsymbol{theta}}{\theta}.
#'
#' @method logLikelihood Kriging
#' @export
#' @aliases logLikelihood,Kriging,Kriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, kernel = "matern3_2", objective="LL")
#' print(k)
#'
#' logLikelihood(k)
logLikelihood.Kriging <- function(object, ...) {
  if (length(L <- list(...)) > 0) warnOnDots(L)
  return(kriging_logLikelihood(object))
}


#' Compute Leave-One-Out (LOO) error for an object with S3 class
#' \code{"Kriging"} representing a kriging model.
#'
#' The returned value is the sum of squares \eqn{\sum_{i=1}^n [y_i -
#' \hat{y}_{i,(-i)}]^2} where \eqn{\hat{y}_{i,(-i)}} is the
#' prediction of \eqn{y_i}{y[i]} based on the the observations \eqn{y_j}{y[j]}
#' with \eqn{j \neq i}{j != i}.
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object A \code{Kriging} object.
#' @param theta A numeric vector of range parameters at which the LOO
#'     will be evaluated.
#'
#' @param return_grad Logical. Should the gradient (w.r.t. \code{theta}) be
#'     returned?
#' @param bench Logical. Should the function display benchmarking output
#' @param ... Not used.
#'
#' @return The leave-One-Out value computed for the given vector
#'     \eqn{\boldsymbol{\theta}}{\theta} of correlation ranges.
#'
#' @method leaveOneOutFun Kriging
#' @export
#' @aliases leaveOneOutFun,Kriging,Kriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, kernel = "matern3_2", objective = "LOO", optim="BFGS")
#' print(k)
#'
#' loo <-  function(theta) leaveOneOutFun(k, theta)$leaveOneOut
#' t <-  seq(from = 0.001, to = 2, length.out = 101)
#' plot(t, loo(t), type = "l")
#' abline(v = k$theta(), col = "blue")
leaveOneOutFun.Kriging <- function(object, theta, return_grad = FALSE, bench=FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    if (is.data.frame(theta)) theta = data.matrix(theta)
    if (!is.matrix(theta)) theta <- matrix(theta, ncol = ncol(object$X()))
    out <- list(leaveOneOut = matrix(NA, nrow = nrow(theta)),
                leaveOneOutGrad = matrix(NA, nrow = nrow(theta),
                                         ncol = ncol(theta)))
    for (i in 1:nrow(theta)) {
        loo <- kriging_leaveOneOutFun(object,theta[i,], isTRUE(return_grad), bench = isTRUE(bench))
        out$leaveOneOut[i] <- loo$leaveOneOut
        if (isTRUE(return_grad)) out$leaveOneOutGrad[i, ] <- loo$leaveOneOutGrad
    }
    if (!isTRUE(return_grad)) out$leaveOneOutGrad <- NULL
    return(out)
}

#' Compute Leave-One-Out (LOO) vector error for an object with S3 class
#' \code{"Kriging"} representing a kriging model.
#'
#' The returned value is the mean and stdev of \eqn{\hat{y}_{i,(-i)}}, the
#' prediction of \eqn{y_i}{y[i]} based on the the observations \eqn{y_j}{y[j]}
#' with \eqn{j \neq i}{j != i}.
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object A \code{Kriging} object.
#' @param theta A numeric vector of range parameters at which the LOO
#'     will be evaluated.
#' @param ... Not used.
#'
#' @return The leave-One-Out vector computed for the given vector
#'     \eqn{\boldsymbol{\theta}}{\theta} of correlation ranges.
#'
#' @method leaveOneOutVec Kriging
#' @export
#' @aliases leaveOneOutVec,Kriging,Kriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(c(0.0, 0.25, 0.5, 0.75, 1.0))
#' y <- f(X)
#'
#' k <- Kriging(y, X, kernel = "matern3_2")
#' print(k)
#'
#' x <- as.matrix(seq(0, 1, , 101))
#' p <- predict(k, x, TRUE, FALSE)
#' 
#' plot(f)
#' points(X, y)
#' lines(x, p$mean, col = 'blue')
#' polygon(c(x, rev(x)), c(p$mean - 2 * p$stdev, rev(p$mean + 2 * p$stdev)),
#'         border = NA, col = rgb(0, 0, 1, 0.2))
#' 
#' # Compute leave-one-out (no range re-estimate) on 2nd point
#' X_no2 = X[-2,,drop=FALSE]
#' y_no2 = f(X_no2)
#' k_no2 = Kriging(y_no2, X_no2, "matern3_2", optim = "none", parameters = list(theta = k$theta()))
#' print(k_no2)
#' 
#' p_no2 <- predict(k_no2, x, TRUE, FALSE)
#' lines(x, p_no2$mean, col = 'red')
#' polygon(c(x, rev(x)), c(p_no2$mean - 2 * p_no2$stdev, rev(p_no2$mean + 2 * p_no2$stdev)), 
#'         border = NA, col = rgb(1, 0, 0, 0.2))
#' 
#' # Use leaveOneOutVec to get the same
#' loov = k$leaveOneOutVec(matrix(k$theta()))
#' points(X[2],loov$mean[2],col='red')
#' lines(rep(X[2],2),loov$mean[2]+2*c(-loov$stdev[2],loov$stdev[2]),col='red')
leaveOneOutVec.Kriging <- function(object, theta, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- kriging_model(object)
    if (!is.array(theta)) 
        stop("Input theta must be 1-dimensional")
    if (length(theta) != ncol(k$X))
        stop("Input theta must have ", ncol(k$X), " values (instead of ",
             length(theta),")")
    return( kriging_leaveOneOutVec(object,theta) )
}


#' Get leaveOneOut of Kriging Model
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object An S3 Kriging object.
#' @param ... Not used.
#'
#' @return The leaveOneOut computed for fitted
#'     \eqn{\boldsymbol{theta}}{\theta}.
#'
#' @method leaveOneOut Kriging
#' @export
#' @aliases leaveOneOut,Kriging,Kriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, kernel = "matern3_2", objective="LOO")
#' print(k)
#'
#' leaveOneOut(k)
leaveOneOut.Kriging <- function(object, ...) {
  if (length(L <- list(...)) > 0) warnOnDots(L)
  return(kriging_leaveOneOut(object))
}

#' Compute the log-marginal posterior of a kriging model, using the
#' prior XXXY.
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object S3 Kriging object.
#' @param theta Numeric vector of correlation range parameters at
#'     which the function is to be evaluated.
#' @param return_grad Logical. Should the function return the gradient
#'     (w.r.t theta)?
#' @param bench Logical. Should the function display benchmarking output?
#' @param ... Not used.
#'
#' @return The value of the log-marginal posterior computed for the
#'     given vector theta.
#'
#' @method logMargPostFun Kriging
#' @export
#' @aliases logMargPostFun,Kriging,Kriging-method
#'
#' @references
#' XXXY A reference describing the model (prior, ...)
#'
#' @seealso \code{\link[RobustGaSP]{rgasp}} in the RobustGaSP package.
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, "matern3_2", objective="LMP")
#' print(k)
#'
#' lmp <- function(theta) logMargPostFun(k, theta)$logMargPost
#'
#' t <- seq(from = 0.01, to = 2, length.out = 101)
#' plot(t, lmp(t), type = "l")
#' abline(v = k$theta(), col = "blue")
logMargPostFun.Kriging <- function(object, theta, return_grad = FALSE, bench=FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    if (is.data.frame(theta)) theta = data.matrix(theta)
    # Nugget models have an extra parameter (alpha) beyond theta
    nparams <- ncol(object$X()) +
      if (kriging_noise_model(object) == "nugget") 1L else 0L
    if (!is.matrix(theta)) theta <- matrix(theta, ncol = nparams)
    out <- list(logMargPost = matrix(NA, nrow = nrow(theta)),
                logMargPostGrad = matrix(NA, nrow = nrow(theta),
                                         ncol = ncol(theta)))
    for (i in 1:nrow(theta)) {
        lmp <- kriging_logMargPostFun(object, theta[i, ], return_grad = isTRUE(return_grad), bench = isTRUE(bench))
        out$logMargPost[i] <- lmp$logMargPost
        if (isTRUE(return_grad)) out$logMargPostGrad[i, ] <- lmp$logMargPostGrad
    }
    if (!isTRUE(return_grad)) out$logMargPostGrad <- NULL
    return(out)
}


#' Get logMargPost of Kriging Model
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object An S3 Kriging object.
#' @param ... Not used.
#'
#' @return The logMargPost computed for fitted
#'     \eqn{\boldsymbol{theta}}{\theta}.
#'
#' @method logMargPost Kriging
#' @export
#' @aliases logMargPost,Kriging,Kriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, kernel = "matern3_2", objective="LMP")
#' print(k)
#'
#' logMargPost(k)
logMargPost.Kriging <- function(object, ...) {
  if (length(L <- list(...)) > 0) warnOnDots(L)
  return(kriging_logMargPost(object))
}


#' Duplicate a Kriging Model
#'
#' @author Yann Richet \email{yann.richet@asnr.fr}
#'
#' @param object An S3 Kriging object.
#' @param ... Not used.
#'
#' @return The copy of object.
#'
#' @method copy Kriging
#' @export
#' @aliases copy,Kriging,Kriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#'
#' k <- Kriging(y, X, kernel = "matern3_2", objective="LMP")
#' print(k)
#'
#' print(copy(k))
copy.Kriging <- function(object, ...) {
  if (length(L <- list(...)) > 0) warnOnDots(L)
  return(classKriging(kriging_copy(object)))
}
