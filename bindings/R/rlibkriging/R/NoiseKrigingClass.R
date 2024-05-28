## ****************************************************************************
## This file contains stuff related to the S3 class "NoiseKriging".
## As an S3 class, it has no formal definition.
## ****************************************************************************

#' Shortcut to provide functions to the S3 class "NoiseKriging"
#' @param nk A pointer to a C++ object of class "NoiseKriging"
#' @return An object of class "NoiseKriging" with methods to access and manipulate the data
classNoiseKriging <- function(nk) {
    class(nk) <- "NoiseKriging"
    # This will allow to call methods (like in Python/Matlab/Octave) using `k$m(...)` as well as R-style `m(k, ...)`.
    for (f in c('as.km','as.list','copy','fit','save',
    'covMat','logLikelihood','logLikelihoodFun',
    'predict','print','show','simulate','update','update_simulate')) {
        eval(parse(text=paste0(
            "nk$", f, " <- function(...) ", f, "(nk,...)"
            )))
    }
    # This will allow to access kriging data/props using `k$d()`
    for (d in c('kernel','optim','objective','X','centerX','scaleX','y','noise','centerY','scaleY','regmodel','F','T','M','z','beta','is_beta_estim','theta','is_theta_estim','sigma2','is_sigma2_estim')) {
        eval(parse(text=paste0(
            "nk$", d, " <- function() noisekriging_", d, "(nk)"
            )))
    }
    nk
}

#' Create an object with S3 class \code{"NoiseKriging"} using
#' the \pkg{libKriging} library.
#'
#' The hyper-parameters (variance and vector of correlation ranges)
#' are estimated thanks to the optimization of a criterion given by
#' \code{objective}, using the method given in \code{optim}.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param y Numeric vector of response values.
#' @param noise Numeric vector of response variances.
#' @param X Numeric matrix of input design.
#' @param kernel Character defining the covariance model:
#'     \code{"exp"}, \code{"gauss"}, \code{"matern3_2"}, \code{"matern5_2"}.
#' @param regmodel Universal NoiseKriging linear trend.
#' @param normalize Logical. If \code{TRUE} both the input matrix
#'     \code{X} and the response \code{y} in normalized to take
#'     values in the interval \eqn{[0, 1]}.
#' @param optim Character giving the Optimization method used to fit
#'     hyper-parameters. Possible values are: \code{"BFGS"} and \code{"none"},
#'     the later simply keeping
#'     the values given in \code{parameters}. The method
#'     \code{"BFGS"} uses the gradient of the objective.
#' @param objective Character giving the objective function to
#'     optimize. Possible values are: \code{"LL"} for the
#'     Log-Likelihood.
#' @param parameters Initial values for the hyper-parameters. When
#'     provided this must be named list with elements \code{"sigma2"}
#'     and \code{"theta"} containing the initial value(s) for the
#'     variance and for the range parameters. If \code{theta} is a
#'     matrix with more than one row, each row is used as a starting
#'     point for optimization.
#'
#' @return An object with S3 class \code{"NoiseKriging"}. Should be used
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
#' y <- f(X) + X/10 * rnorm(nrow(X)) # add noise dep. on X
#' ## fit and print
#' k <- NoiseKriging(y, noise=(X/10)^2, X, kernel = "matern3_2")
#' print(k)
#'
#' x <- as.matrix(seq(from = 0, to = 1, length.out = 101))
#' p <- predict(k,x = x, stdev = TRUE, cov = FALSE)
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
NoiseKriging <- function(y=NULL, noise=NULL, X=NULL, kernel=NULL,
                    regmodel = c("constant", "linear", "interactive", "none"),
                    normalize = FALSE,
                    optim = c("BFGS", "none"),
                    objective = c("LL"),
                    parameters = NULL) {

    regmodel <- match.arg(regmodel)
    objective <- match.arg(objective)
    if (is.character(optim)) optim <- optim[1] #optim <- match.arg(optim) because we can use BFGS10 for 10 (multistart) BFGS
    if (is.character(y) && is.null(X) && is.null(noise) && is.null(kernel)) # just first arg for kernel, without naming
        nk <- new_NoiseKriging(kernel = y)
    else if (is.null(y) && is.null(X) && !is.null(kernel))
        nk <- new_NoiseKriging(kernel = kernel)
    else
        nk <- new_NoiseKrigingFit(y = y, noise = noise, X = X, kernel = kernel,
                      regmodel = regmodel,
                      normalize = normalize,
                      optim = optim,
                      objective = objective,
                      parameters = parameters)
    return(classNoiseKriging(nk))
}


#' Coerce a \code{NoiseKriging} Object into a List
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param x An object with class \code{"NoiseKriging"}.
#' @param ... Ignored
#'
#' @return A list with its elements copying the content of the
#'     \code{NoiseKriging} object fields: \code{kernel}, \code{optim},
#'     \code{objective}, \code{theta} (vector of ranges),
#'     \code{sigma2} (variance), \code{X}, \code{centerX},
#'     \code{scaleX}, \code{y}, \code{centerY}, \code{scaleY},
#'     \code{regmodel}, \code{F}, \code{T}, \code{M}, \code{z},
#'     \code{beta}.
#'
#' @export
#' @method as.list NoiseKriging
#' @aliases as.list,NoiseKriging,NoiseKriging-method
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10 * rnorm(nrow(X)) # add noise dep. on X
#'
#' k <- NoiseKriging(y, noise=(X/10)^2, X, kernel = "matern3_2")
#'
#' l <- as.list(k)
#' cat(paste0(names(l), " =" , l, collapse = "\n"))
as.list.NoiseKriging <- function(x, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    noisekriging_model(x)
}


#' Coerce a \code{NoiseKriging} object into the \code{"km"} class of the
#' \pkg{DiceKriging} package.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param x An object with S3 class \code{"NoiseKriging"}.
#' @param .call Force the \code{call} slot to be filled in the
#'     returned \code{km} object.
#' @param ... Not used.
#'
#' @return An object of having the S4 class \code{"KM"} which extends
#'     the \code{"km"} class of the \pkg{DiceKriging} package and
#'     contains an extra \code{NoiseKriging} slot.
#'
#' @importFrom methods new
#' @importFrom stats model.matrix
#' @export
#' @method as.km NoiseKriging
#' @aliases as.km,NoiseKriging,NoiseKriging-method
#'
#' @examples
#'f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#'set.seed(123)
#'X <- as.matrix(runif(10))
#'y <- f(X) + X/10 * rnorm(nrow(X)) # add noise dep. on X
#'## fit and print
#'k <- NoiseKriging(y, noise=(X/10)^2, X, kernel = "matern3_2")
#'print(k)
#'
#' k_km <- as.km(k)
#' print(k_km)
as.km.NoiseKriging <- function(x, .call = NULL, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    ## loadDiceKriging()
    ## if (! "DiceKriging" %in% installed.packages())
    ##     stop("DiceKriging must be installed to use its wrapper from libKriging.")

    if (!requireNamespace("DiceKriging", quietly = TRUE))
        stop("Package \"DiceKriging\" not found")

    model <- new("NoiseKM")
    model@NoiseKriging <- x

    if (is.null(.call))
        model@call <- match.call()
    else
        model@call <- .call

    m <- noisekriging_model(x)
    data <- data.frame(m$X)
    model@trend.formula <- regmodel2formula(m$regmodel)
    model@trend.coef <- as.numeric(m$beta)
    model@X <- m$X
    model@y <- m$y
    model@d <- ncol(m$X)
    model@n <- nrow(m$X)
    model@F <- m$F
    colnames(model@F) <- colnames(model.matrix(model@trend.formula,data))
    model@p <- ncol(m$F)
    model@noise.flag <- TRUE
    model@noise.var <- as.numeric(m$noise)

    model@case <- "LLconcentration_beta"

    isTrend = !m$is_beta_estim
    isCov = !m$is_theta_estim
    isVar = !m$is_sigma2_estim
    if (isCov) {
        known.covparam <- "All"
    } else {
        known.covparam <- "None"
    }
    model@param.estim <- NA
    model@method <- m$objective
    model@optim.method <- m$optim

    model@penalty <- list()
    model@lower <- 0
    model@upper <- Inf
    model@control <- list()

    model@gr <- FALSE

    model@T <- t(m$T) * sqrt(m$sigma2)
    model@z <- as.numeric(m$z) / sqrt(m$sigma2)
    model@M <- m$M / sqrt(m$sigma2)

    covStruct <-  new("covTensorProduct", d = model@d, name = m$kernel,
                      sd2 = m$sigma2, var.names = names(data),
                      nugget = 0, nugget.flag = FALSE, nugget.estim = FALSE,
                      known.covparam = known.covparam)

    if (isTrend && isCov && isVar) {
        model@known.param <- "All"
    } else if ((isTrend) && ((!isCov) || (!isVar))) {
        model@known.param <- "Trend"
    } else if ((!isTrend) && isCov && isVar) {
        model@known.param <- "CovAndVar"
    } else {    # In the other cases: All parameters are estimated (at this stage)
        model@known.param <- "None"
    }

    covStruct@range.names <- "theta"
    covStruct@paramset.n <- as.integer(1)
    covStruct@param.n <- as.integer(model@d)
    covStruct@range.n <- as.integer(model@d)
    covStruct@range.val <- as.numeric(m$theta)
    model@covariance <- covStruct

    return(model)
}


#' Print the content of a \code{NoiseKriging} object.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param x A (S3) \code{NoiseKriging} Object.
#' @param ... Ignored.
#'
#' @return String of printed object.
#'
#' @export
#' @method print NoiseKriging
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10 * rnorm(nrow(X)) # add noise dep. on X
#'
#' k <- NoiseKriging(y, noise=(X/10)^2, X, kernel = "matern3_2")
#'
#' print(k)
#' ## same thing
#' k
print.NoiseKriging <- function(x, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    p = noisekriging_summary(x)
    cat(p)
    invisible(p)
}


#' Fit \code{NoiseKriging} object on given data.
#'
#' The hyper-parameters (variance and vector of correlation ranges)
#' are estimated thanks to the optimization of a criterion given by
#' \code{objective}, using the method given in \code{optim}.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object S3 NoiseKriging object.
#' @param y Numeric vector of response values.
#' @param noise Numeric vector of response variances.
#' @param X Numeric matrix of input design.
#' @param regmodel Universal NoiseKriging linear trend.
#' @param normalize Logical. If \code{TRUE} both the input matrix
#'     \code{X} and the response \code{y} in normalized to take
#'     values in the interval \eqn{[0, 1]}.
#' @param optim Character giving the Optimization method used to fit
#'     hyper-parameters. Possible values are: \code{"BFGS"} and \code{"none"},
#'     the later simply keeping
#'     the values given in \code{parameters}. The method
#'     \code{"BFGS"} uses the gradient of the objective.
#' @param objective Character giving the objective function to
#'     optimize. Possible values are: \code{"LL"} for the
#'     Log-Likelihood.
#' @param parameters Initial values for the hyper-parameters. When
#'     provided this must be named list with elements \code{"sigma2"}
#'     and \code{"theta"} containing the initial value(s) for the
#'     variance and for the range parameters. If \code{theta} is a
#'     matrix with more than one row, each row is used as a starting
#'     point for optimization.
#' @param ... Ignored.
#'
#' @return No return value. NoiseKriging object argument is modified.
#'
#' @method fit NoiseKriging
#' @export
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10 * rnorm(nrow(X)) # add noise dep. on X
#' points(X, y, col = "blue", pch = 16)
#'
#' k <- NoiseKriging("matern3_2")
#' print(k)
#'
#' fit(k,y,noise=(X/10)^2,X)
#' print(k)
fit.NoiseKriging <- function(object, y, noise, X,
                    regmodel = c("constant", "linear", "interactive", "none"),
                    normalize = FALSE,
                    optim = c("BFGS", "none"),
                    objective = c("LL"),
                    parameters = NULL, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    regmodel <- match.arg(regmodel)
    objective <- match.arg(objective)
    if (is.character(optim)) optim <- optim[1] #optim <- match.arg(optim) because we can use BFGS10 for 10 (multistart) BFGS

    noisekriging_fit(object, y, noise, X,
                    regmodel,
                    normalize,
                    optim ,
                    objective,
                    parameters)

    invisible(NULL)
}


#' Predict from a \code{NoiseKriging} object.
#'
#' Given "new" input points, the method compute the expectation,
#' variance and (optionnally) the covariance of the corresponding
#' stochastic process, conditional on the values at the input points
#' used when fitting the model.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object S3 NoiseKriging object.
#' @param x Input points where the prediction must be computed.
#' @param stdev \code{Logical}. If \code{TRUE} the standard deviation
#'     is returned.
#' @param cov \code{Logical}. If \code{TRUE} the covariance matrix of
#'     the predictions is returned.
#' @param deriv \code{Logical}. If \code{TRUE} the derivatives of mean and sd
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
#' @method predict NoiseKriging
#' @export
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10 * rnorm(nrow(X))
#' points(X, y, col = "blue", pch = 16)
#'
#' k <- NoiseKriging(y, (X/10)^2, X, "matern3_2")
#'
#' x <-seq(from = 0, to = 1, length.out = 101)
#' p <- predict(k, x)
#'
#' lines(x, p$mean, col = "blue")
#' polygon(c(x, rev(x)), c(p$mean - 2 * p$stdev, rev(p$mean + 2 * p$stdev)),
#'  border = NA, col = rgb(0, 0, 1, 0.2))
predict.NoiseKriging <- function(object, x, stdev = TRUE, cov = FALSE, deriv = FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- noisekriging_model(object)
    ## manage the data frame case. Ideally we should then warn
    if (is.data.frame(x)) x = data.matrix(x)
    if (!is.matrix(x)) x=matrix(x,ncol=ncol(k$X))
    if (ncol(x) != ncol(k$X))
        stop("Input x must have ", ncol(k$X), " columns (instead of ",
             ncol(x), ")")
    return(noisekriging_predict(object, x, stdev, cov, deriv))
}


#' Simulation from a \code{NoiseKriging} model object.
#'
#' This method draws paths of the stochastic process at new input
#' points conditional on the values at the input points used in the
#' fit.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object S3 NoiseKriging object.
#' @param nsim Number of simulations to perform.
#' @param seed Random seed used.
#' @param x Points in model input space where to simulate.
#' @param will_update Compute useful data for future update_simulate call
#' @param ... Ignored.
#'
#' @return a matrix with \code{length(x)} rows and \code{nsim}
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
#' @method simulate NoiseKriging
#' @export
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10 * rnorm(nrow(X))
#' points(X, y, col = "blue")
#'
#' k <- NoiseKriging(y, (X/10)^2, X, kernel = "matern3_2")
#'
#' x <- seq(from = 0, to = 1, length.out = 101)
#' s <- simulate(k, nsim = 3, x = x)
#'
#' lines(x, s[ , 1], col = "blue")
#' lines(x, s[ , 2], col = "blue")
#' lines(x, s[ , 3], col = "blue")
simulate.NoiseKriging <- function(object, nsim = 1, seed = 123, x, with_noise = NULL, will_update = FALSE,  ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- noisekriging_model(object)
    if (is.data.frame(x)) x = data.matrix(x)
    if (!is.matrix(x)) x = matrix(x, ncol = ncol(k$X))
    if (ncol(x) != ncol(k$X))
        stop("Input x must have ", ncol(k$X), " columns (instead of ",
             ncol(x),")")
    if (is.null(seed)) seed <- floor(runif(1) * 99999)
    if (is.null(with_noise)) with_noise = rep(0, nrow(x))
    return(noisekriging_simulate(object, nsim = nsim, seed = seed, X = x, with_noise, will_update = will_update))
}

#' Update previous simulation of a \code{NoiseKriging} model object.
#'
#' This method draws paths of the stochastic process conditional on the values at the input points used in the
#' fit, plus the new input points and their values given as argument (knonw as 'update' points).
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object S3 NoiseKriging object.
#' @param y_u Numeric vector of new responses (output).
#' @param noise_u Numeric vector of new noise variances (output).
#' @param X_u Numeric matrix of new input points.
#' @param ... Ignored.
#'
#' @return a matrix with \code{length(x)} rows and \code{nsim}
#'     columns containing the simulated paths at the inputs points
#'     given in \code{x}.
#'
#' @method update_simulate NoiseKriging
#' @export
update_simulate.NoiseKriging <- function(object, y_u, noise_u, X_u, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- noisekriging_model(object)
    if (is.data.frame(X_u)) X_u = data.matrix(X_u)
    if (!is.matrix(X_u)) X_u <- matrix(X_u, ncol = ncol(k$X))
    if (is.data.frame(y_u)) y_u = data.matrix(y_u)
    if (!is.matrix(y_u)) y_u <- matrix(y_u, ncol = ncol(k$y))
    if (ncol(X_u) != ncol(k$X))
        stop("Object 'X_u' must have ", ncol(k$X), " columns (instead of ",
             ncol(X_u), ")")
    if (nrow(y_u) != nrow(X_u))
        stop("Objects 'X_u' and 'y_u' must have the same number of rows.")

    ## Modify 'object' in the parent environment
    return(noisekriging_update_simulate(object, y_u, noise_u, X_u))
}

#' Update a \code{NoiseKriging} model object with new points
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object S3 NoiseKriging object.
#' @param y_u Numeric vector of new responses (output).
#' @param noise_u Numeric vector of new noise variances (output).
#' @param X_u Numeric matrix of new input points.
#' @param refit Logical. If \code{TRUE} the model is refitted (default is FALSE).
#' @param ... Ignored.
#'
#' @return No return value. NoiseKriging object argument is modified.
#'
#' @section Caution: The method \emph{does not return the updated
#'     object}, but instead changes the content of
#'     \code{object}. This behaviour is quite unusual in R and
#'     differs from the behaviour of the methods
#'     \code{\link[DiceKriging]{update.km}} in \pkg{DiceKriging} and
#'     \code{\link{update,KM-method}}.
#'
#' @method update NoiseKriging
#' @export
#'
#' @examples
#' f <- function(x) 1- 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x)*x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10 * rnorm(nrow(X))
#' points(X, y, col = "blue")
#'
#' k <- NoiseKriging(y, (X/10)^2, X, "matern3_2")
#'
#' x <- seq(from = 0, to = 1, length.out = 101)
#' p <- predict(k, x)
#' lines(x, p$mean, col = "blue")
#' polygon(c(x, rev(x)), c(p$mean - 2 * p$stdev, rev(p$mean + 2 * p$stdev)),
#'  border = NA, col = rgb(0, 0, 1, 0.2))
#'
#' X_u <- as.matrix(runif(3))
#' y_u <- f(X_u) + 0.1 * rnorm(nrow(X_u))
#' points(X_u, y_u, col = "red")
#'
#' ## change the content of the object 'k'
#' update(k, y_u, rep(0.1^2,3), X_u)
#'
#' ## include design points to see interpolation
#' x <- sort(c(X,newX,seq(from = 0, to = 1, length.out = 101)))
#' p2 <- predict(k, x)
#' lines(x, p2$mean, col = "red")
#' polygon(c(x, rev(x)), c(p2$mean - 2 * p2$stdev, rev(p2$mean + 2 * p2$stdev)),
#'  border = NA, col = rgb(1, 0, 0, 0.2))
update.NoiseKriging <- function(object, y_u, noise_u, X_u, refit=TRUE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- noisekriging_model(object)
    if (is.data.frame(X_u)) X_u = data.matrix(X_u)
    if (!is.matrix(X_u)) X_u <- matrix(X_u, ncol = ncol(k$X))
    if (is.data.frame(y_u)) y_u = data.matrix(y_u)
    if (!is.matrix(y_u)) y_u <- matrix(y_u, ncol = ncol(k$y))
    if (!is.matrix(noise_u)) noise_u <- matrix(noise_u, ncol = ncol(k$y))
    if (ncol(X_u) != ncol(k$X))
        stop("Object 'X_u' must have ", ncol(k$X), " columns (instead of ",
             ncol(X_u), ")")
    if (nrow(y_u) != nrow(X_u))
        stop("Objects 'X_u' and 'y_u' must have the same number of rows.")
    if (nrow(noise_u) != nrow(X_u))
        stop("Objects 'noise_u' and 'y_u' must have the same number of rows.")

    ## Modify 'object' in the parent environment
    noisekriging_update(object, y_u, noise_u, X_u, refit)

    invisible(NULL)
}


#' Save a NoiseKriging Model to a file storage
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object An S3 NoiseKriging object.
#' @param filename File name to save in.
#' @param ... Not used.
#'
#' @return The loaded NoiseKriging object.
#'
#' @method save NoiseKriging
#' @export
#' @aliases save,NoiseKriging,NoiseKriging-method
#'
#' @examples
#' f <- function(x) 1- 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x)*x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10 * rnorm(nrow(X))
#'
#' k <- NoiseKriging(y, (X/10)^2, X, "matern3_2")
#' print(k)
#'
#' outfile = tempfile("k.json") 
#' save(k,outfile)
save.NoiseKriging <- function(object, filename, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    if (!is.character(filename))
        stop("'filename' must be a string")

    noisekriging_save(object, filename)

    invisible(NULL)
}


#' Load a NoiseKriging Model from a file storage
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param filename File name to load from.
#' @param ... Not used.
#'
#' @return The loaded NoiseKriging object.
#'
#' @export
#'
#' @examples
#' f <- function(x) 1- 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x)*x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10 * rnorm(nrow(X))
#' points(X, y, col = "blue")
#'
#' k <- NoiseKriging(y, (X/10)^2, X, "matern3_2")
#' print(k)
#'
#' outfile = tempfile("k.json")
#' save(k,outfile)
#'
#' print(load.NoiseKriging(outfile)) 
load.NoiseKriging <- function(filename, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    if (!is.character(filename))
        stop("'filename' must be a string")
    return(classNoiseKriging(noisekriging_load(filename)))
}

#' Compute Covariance Matrix of NoiseKriging Model
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object An S3 NoiseKriging object.
#' @param x1 Numeric matrix of input points.
#' @param x2 Numeric matrix of input points.
#' @param ... Not used.
#' 
#' @return A matrix of the covariance matrix of the NoiseKriging model.
#' 
#' @method covMat NoiseKriging
#' @export
#' @aliases covMat,NoiseKriging,NoiseKriging-method
#' 
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10 * rnorm(nrow(X))
#'
#' k <- NoiseKriging(y, (X/10)^2, X, "matern3_2")
#' 
#' x1 = runif(10)
#' x2 = runif(10)
#' 
#' covMat(k, x1, x2)
covMat.NoiseKriging <- function(object, x1, x2, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- kriging_model(object)
    if (is.data.frame(x1)) x1 = data.matrix(x1)
    if (is.data.frame(x2)) x2 = data.matrix(x2)
    if (!is.matrix(x1)) x1 = matrix(x1, ncol = ncol(k$X))
    if (!is.matrix(x2)) x2 = matrix(x2, ncol = ncol(k$X))
    if (ncol(x1) != ncol(k$X))
        stop("Input x1 must have ", ncol(k$X), " columns (instead of ",
             ncol(x1), ")")
    if (ncol(x2) != ncol(k$X))
        stop("Input x2 must have ", ncol(k$X), " columns (instead of ",
             ncol(x2), ")")
    return(noisekriging_covMat(object, x1, x2))
}

#' Compute Log-Likelihood of NoiseKriging Model
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object An S3 NoiseKriging object.
#' @param theta_sigma2 A numeric vector of (positive) range parameters and variance at
#'     which the log-likelihood will be evaluated.
#' @param grad Logical. Should the function return the gradient?
#' @param bench Logical. Should the function display benchmarking output
#' @param ... Not used.
#'
#' @return The log-Likelihood computed for given
#'     \eqn{\boldsymbol{theta_sigma2}}{\theta,\sigma^2}.
#'
#' @method logLikelihoodFun NoiseKriging
#' @export
#' @aliases logLikelihoodFun,NoiseKriging,NoiseKriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10  *rnorm(nrow(X))
#'
#' k <- NoiseKriging(y, (X/10)^2, X, kernel = "matern3_2")
#' print(k)
#'
#' theta0 = k$theta()
#' ll_sigma2 <- function(sigma2) logLikelihoodFun(k, cbind(theta0,sigma2))$logLikelihood
#' s2 <- seq(from = 0.001, to = 1, length.out = 101)
#' plot(s2, Vectorize(ll_sigma2)(s2), type = 'l')
#' abline(v = k$sigma2(), col = "blue")
#'
#' sigma20 = k$sigma2()
#' ll_theta <- function(theta) logLikelihoodFun(k, cbind(theta,sigma20))$logLikelihood
#' t <- seq(from = 0.001, to = 2, length.out = 101)
#' plot(t, Vectorize(ll_theta)(t), type = 'l')
#' abline(v = k$theta(), col = "blue")
#'
#' ll <- function(theta_sigma2) logLikelihoodFun(k, theta_sigma2)$logLikelihood
#' s2 <- seq(from = 0.001, to = 1, length.out = 31)
#' t <- seq(from = 0.001, to = 2, length.out = 31)
#' contour(t,s2,matrix(ncol=length(s2),ll(expand.grid(t,s2))),xlab="theta",ylab="sigma2")
#' points(k$theta(),k$sigma2(),col='blue')
logLikelihoodFun.NoiseKriging <- function(object, theta_sigma2,
                                  grad = FALSE, bench=FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- noisekriging_model(object)
    if (is.data.frame(theta_sigma2)) theta_sigma2 = data.matrix(theta_sigma2)
    if (!is.matrix(theta_sigma2)) theta_sigma2 <- matrix(theta_sigma2, ncol = ncol(k$X)+1)
    if (ncol(theta_sigma2) != ncol(k$X)+1)
        stop("Input theta_sigma2 must have ", ncol(k$X)+1, " columns (instead of ",
             ncol(theta_sigma2),")")
    out <- list(logLikelihood = matrix(NA, nrow = nrow(theta_sigma2)),
                logLikelihoodGrad = matrix(NA,nrow=nrow(theta_sigma2),
                                           ncol = ncol(theta_sigma2)))
    for (i in 1:nrow(theta_sigma2)) {
        ll <- noisekriging_logLikelihoodFun(object, theta_sigma2[i, ],
                                    grad = isTRUE(grad), bench = isTRUE(bench))
        out$logLikelihood[i] <- ll$logLikelihood
        if (isTRUE(grad)) out$logLikelihoodGrad[i, ] <- ll$logLikelihoodGrad
    }
    if (!isTRUE(grad)) out$logLikelihoodGrad <- NULL

    return(out)
}



#' Get logLikelihood of NoiseKriging Model
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object An S3 NoiseKriging object.
#' @param ... Not used.
#'
#' @return The logLikelihood computed for fitted
#'     \eqn{\boldsymbol{theta_sigma2}}{theta,\sigma^2}.
#'
#' @method logLikelihood NoiseKriging
#' @export
#' @aliases logLikelihood,NoiseKriging,NoiseKriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10 * rnorm(nrow(X))
#'
#' k <- NoiseKriging(y, (X/10)^2, X, kernel = "matern3_2", objective="LL")
#' print(k)
#'
#' logLikelihood(k)
logLikelihood.NoiseKriging <- function(object, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    return(noisekriging_logLikelihood(object))
}

#' Duplicate a NoiseKriging Model
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object An S3 NoiseKriging object.
#' @param ... Not used.
#'
#' @return The copy of object.
#'
#' @method copy NoiseKriging
#' @export
#' @aliases copy,NoiseKriging,NoiseKriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + X/10 * rnorm(nrow(X))
#'
#' k <- NoiseKriging(y, (X/10)^2, X, kernel = "matern3_2", objective="LL")
#' print(k)
#'
#' print(copy(k))
copy.NoiseKriging <- function(object, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    return(classNoiseKriging(noisekriging_copy(object)))
}
