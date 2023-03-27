## ****************************************************************************
## This file contains stuff related to the S3 class "NuggetKriging".
## As an S3 class, it has no formal definition.
## ****************************************************************************




#' Create an object with S3 class \code{"NuggetKriging"} using
#' the \pkg{libKriging} library.
#'
#' The hyper-parameters (variance and vector of correlation ranges)
#' are estimated thanks to the optimization of a criterion given by
#' \code{objective}, using the method given in \code{optim}.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param y Numeric vector of response values.
#'
#' @param X Numeric matrix of input design.
#'
#' @param kernel Character defining the covariance model:
#'     \code{"exp"}, \code{"gauss"}, \code{"matern3_2"}, \code{"matern5_2"}.
#'
#' @param regmodel Universal NuggetKriging linear trend.
#'
#' @param normalize Logical. If \code{TRUE} both the input matrix
#'     \code{X} and the response \code{y} in normalized to take
#'     values in the interval \eqn{[0, 1]}.
#'
#' @param optim Character giving the Optimization method used to fit
#'     hyper-parameters. Possible values are: \code{"BFGS"} and \code{"none"},
#'     the later simply keeping
#'     the values given in \code{parameters}. The method
#'     \code{"BFGS"} uses the gradient of the objective.
#'
#' @param objective Character giving the objective function to
#'     optimize. Possible values are: \code{"LL"} for the
#'     Log-Likelihood and \code{"LMP"} for the Log-Marginal Posterior.
#'
#' @param parameters Initial values for the hyper-parameters. When provided this
#'     must be named list with some elements \code{"sigma2"}, \code{"theta"}, \code{"nugget"}
#'     containing the initial value(s) for the variance, range and nugget
#'     parameters. If \code{theta} is a matrix with more than one row,
#'     each row is used as a starting point for optimization.
#'
#' @return An object with S3 class \code{"NuggetKriging"}. Should be used
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
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#' ## fit and print
#' k <- NuggetKriging(y, X, kernel = "matern3_2")
#' print(k)
#'
#' x <- sort(c(X,as.matrix(seq(from = 0, to = 1, length.out = 101))))
#' p <- predict(k, x = x, stdev = TRUE, cov = FALSE)
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
NuggetKriging <- function(y=NULL, X=NULL, kernel=NULL,
                    regmodel = c("constant", "linear", "interactive"),
                    normalize = FALSE,
                    optim = c("BFGS", "none"),
                    objective = c("LL", "LMP"),
                    parameters = NULL) {

    regmodel <- match.arg(regmodel)
    objective <- match.arg(objective)
    if (is.character(optim)) optim <- optim[1] #optim <- match.arg(optim) because we can use BFGS10 for 10 (multistart) BFGS
    if (is.character(y) && is.null(X) && is.null(kernel)) # just first arg for kernel, without naming
        nk <- new_NuggetKriging(kernel = y)
    else if (is.null(y) && is.null(X) && !is.null(kernel))
        nk <- new_NuggetKriging(kernel = kernel)
    else
        nk <- new_NuggetKrigingFit(y = y, X = X, kernel = kernel,
                      regmodel = regmodel,
                      normalize = normalize,
                      optim = optim,
                      objective = objective,
                      parameters = parameters)
    class(nk) <- "NuggetKriging"
    # This will allow to call methods (like in Python/Matlab/Octave) using `k$m(...)` as well as R-style `m(k, ...)`.
    for (f in c('as.km','as.list','copy','fit','logLikelihood','logLikelihoodFun','logMargPost','logMargPostFun','predict','print','show','simulate','update')) {
        eval(parse(text=paste0(
            "nk$", f, " <- function(...) ", f, "(nk,...)"
            )))
    }
    # This will allow to access kriging data/props using `k$d()`
    for (d in c('kernel','optim','objective','X','centerX','scaleX','y','centerY','scaleY','regmodel','F','T','M','z','beta','is_beta_estim','theta','is_theta_estim','sigma2','is_sigma2_estim','nugget','is_nugget_estim')) {
        eval(parse(text=paste0(
            "nk$", d, " <- function() nuggetkriging_", d, "(nk)"
            )))
    }
    nk
}


#' Coerce a \code{NuggetKriging} Object into a List
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param x An object with class \code{"NuggetKriging"}.
#' @param ... Ignored
#'
#' @return A list with its elements copying the content of the
#'     \code{NuggetKriging} object fields: \code{kernel}, \code{optim},
#'     \code{objective}, \code{theta} (vector of ranges),
#'     \code{sigma2} (variance), \code{X}, \code{centerX},
#'     \code{scaleX}, \code{y}, \code{centerY}, \code{scaleY},
#'     \code{regmodel}, \code{F}, \code{T}, \code{M}, \code{z},
#'     \code{beta}.
#'
#' @export
#' @method as.list NuggetKriging
#' @aliases as.list,NuggetKriging,NuggetKriging-method
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#'
#' k <- NuggetKriging(y, X, kernel = "matern3_2")
#'
#' l <- as.list(k)
#' cat(paste0(names(l), " =" , l, collapse = "\n"))
as.list.NuggetKriging <- function(x, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    nuggetkriging_model(x)
}


#' Coerce a \code{NuggetKriging} object into the \code{"km"} class of the
#' \pkg{DiceKriging} package.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param x An object with S3 class \code{"NuggetKriging"}.
#'
#' @param .call Force the \code{call} slot to be filled in the
#'     returned \code{km} object.
#'
#' @param ... Not used.
#'
#' @return An object of having the S4 class \code{"KM"} which extends
#'     the \code{"km"} class of the \pkg{DiceKriging} package and
#'     contains an extra \code{NuggetKriging} slot.
#'
#' @importFrom methods new
#' @importFrom stats model.matrix
#' @export
#' @method as.km NuggetKriging
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#'
#' k <- NuggetKriging(y, X, "matern3_2")
#' print(k)
#'
#' k_km <- as.km(k)
#' print(k_km)
as.km.NuggetKriging <- function(x, .call = NULL, ...) {

    ## loadDiceKriging()
    ## if (! "DiceKriging" %in% installed.packages())
    ##     stop("DiceKriging must be installed to use its wrapper from libKriging.")

    if (!requireNamespace("DiceKriging", quietly = TRUE))
        stop("Package \"DiceKriging\" not found")

    model <- new("NuggetKM")
    model@NuggetKriging <- x

    if (is.null(.call))
        model@call <- match.call()
    else
        model@call <- .call

    m <- nuggetkriging_model(x)
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
    model@noise.flag <- FALSE
    model@noise.var <- 0

    model@case <- "LLconcentration_beta_v_alpha"

    model@known.param <- "None"
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
                      nugget = m$nugget, nugget.flag = TRUE, nugget.estim = TRUE,
                      known.covparam = "")

    covStruct@range.names <- "theta"
    covStruct@paramset.n <- as.integer(1)
    covStruct@param.n <- as.integer(model@d)
    covStruct@range.n <- as.integer(model@d)
    covStruct@range.val <- as.numeric(m$theta)
    model@covariance <- covStruct

    return(model)
}


#' Print the content of a \code{NuggetKriging} object.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param x A (S3) \code{NuggetKriging} Object.
#' @param ... Ignored.
#'
#' @return String of printed object.
#'
#' @export
#' @method print NuggetKriging
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#'
#' k <- NuggetKriging(y, X, "matern3_2")
#'
#' print(k)
#' ## same thing
#' k
print.NuggetKriging <- function(x, ...) {
    if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
    p = nuggetkriging_summary(x)
    cat(p)
    invisible(p)
}


#' Fit \code{NuggetKriging} object on given data.
#'
#' The hyper-parameters (variance and vector of correlation ranges)
#' are estimated thanks to the optimization of a criterion given by
#' \code{objective}, using the method given in \code{optim}.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object S3 NuggetKriging object.
#'
#' @param y Numeric vector of response values.
#'
#' @param X Numeric matrix of input design.
#'
#' @param regmodel Universal NuggetKriging linear trend.
#'
#' @param normalize Logical. If \code{TRUE} both the input matrix
#'     \code{X} and the response \code{y} in normalized to take
#'     values in the interval \eqn{[0, 1]}.
#'
#' @param optim Character giving the Optimization method used to fit
#'     hyper-parameters. Possible values are: \code{"BFGS"} and \code{"none"},
#'     the later simply keeping
#'     the values given in \code{parameters}. The method
#'     \code{"BFGS"} uses the gradient of the objective.
#'
#' @param objective Character giving the objective function to
#'     optimize. Possible values are: \code{"LL"} for the
#'     Log-Likelihood and \code{"LMP"} for the Log-Marginal Posterior.
#'
#' @param parameters Initial values for the hyper-parameters. When provided this
#'     must be named list with some elements \code{"sigma2"}, \code{"theta"}, \code{"nugget"}
#'     containing the initial value(s) for the variance, range and nugget
#'     parameters. If \code{theta} is a matrix with more than one row,
#'     each row is used as a starting point for optimization.
#'
#' @param ... Ignored.
#'
#' @return No return value. NuggetKriging object argument is modified.
#'
#' @method fit NuggetKriging
#' @export
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#' points(X, y, col = "blue", pch = 16)
#'
#' k <- NuggetKriging("matern3_2")
#' print(k)
#'
#' fit(k,y,X)
#' print(k)
fit.NuggetKriging <- function(object, y, X,
                    regmodel = c("constant", "linear", "interactive"),
                    normalize = FALSE,
                    optim = c("BFGS", "none"),
                    objective = c("LL", "LMP"),
                    parameters = NULL, ...) {

    regmodel <- match.arg(regmodel)
    objective <- match.arg(objective)
    if (is.character(optim)) optim <- optim[1] #optim <- match.arg(optim) because we can use BFGS10 for 10 (multistart) BFGS

    nuggetkriging_fit(object, y, X,
                    regmodel,
                    normalize,
                    optim ,
                    objective,
                    parameters)

    invisible(NULL)
}


#' Predict from a \code{NuggetKriging} object.
#'
#' Given "new" input points, the method compute the expectation,
#' variance and (optionnally) the covariance of the corresponding
#' stochastic process, conditional on the values at the input points
#' used when fitting the model.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object S3 NuggetKriging object.
#'
#' @param x Input points where the prediction must be computed.
#'
#' @param stdev \code{Logical}. If \code{TRUE} the standard deviation
#'     is returned.
#'
#' @param cov \code{Logical}. If \code{TRUE} the covariance matrix of
#'     the predictions is returned.
#'
#' @param deriv \code{Logical}. If \code{TRUE} the derivatives of mean and sd
#'     of the predictions are returned.
#'
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
#' @method predict NuggetKriging
#' @export
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#' points(X, y, col = "blue", pch = 16)
#'
#' k <- NuggetKriging(y, X, "matern3_2")
#'
#' ## include design points to see interpolation
#' x <- sort(c(X,seq(from = 0, to = 1, length.out = 101)))
#' p <- predict(k, x)
#'
#' lines(x, p$mean, col = "blue")
#' polygon(c(x, rev(x)), c(p$mean - 2 * p$stdev, rev(p$mean + 2 * p$stdev)),
#'  border = NA, col = rgb(0, 0, 1, 0.2))
predict.NuggetKriging <- function(object, x, stdev = TRUE, cov = FALSE, deriv = FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- nuggetkriging_model(object)
    ## manage the data frame case. Ideally we should then warn
    if (is.data.frame(x)) x = data.matrix(x)
    if (!is.matrix(x)) x=matrix(x,ncol=ncol(k$X))
    if (ncol(x) != ncol(k$X))
        stop("Input x must have ", ncol(k$X), " columns (instead of ",
             ncol(x), ")")
    return(nuggetkriging_predict(object, x, stdev, cov, deriv))
}

#' Simulation from a \code{NuggetKriging} model object.
#'
#' This method draws paths of the stochastic process at new input
#' points conditional on the values at the input points used in the
#' fit.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object S3 NuggetKriging object.
#' @param nsim Number of simulations to perform.
#' @param seed Random seed used.
#' @param x Points in model input space where to simulate.
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
#' @method simulate NuggetKriging
#' @export
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + 0.1  *rnorm(nrow(X))
#' points(X, y, col = "blue")
#'
#' k <- NuggetKriging(y, X, kernel = "matern3_2")
#'
#' x <- seq(from = 0, to = 1, length.out = 101)
#' s <- simulate(k, nsim = 3, x = x)
#'
#' lines(x, s[ , 1], col = "blue")
#' lines(x, s[ , 2], col = "blue")
#' lines(x, s[ , 3], col = "blue")
simulate.NuggetKriging <- function(object, nsim = 1, seed = 123, x,  ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- nuggetkriging_model(object)
    if (is.data.frame(x)) x = data.matrix(x)
    if (!is.matrix(x)) x = matrix(x, ncol = ncol(k$X))
    if (ncol(x) != ncol(k$X))
        stop("Input x must have ", ncol(k$X), " columns (instead of ",
             ncol(x),")")
    ## XXXY
    if (is.null(seed)) seed <- floor(runif(1) * 99999)
    return(nuggetkriging_simulate(object, nsim = nsim, seed = seed, X = x))
}


#' Update a \code{NuggetKriging} model object with new points
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object S3 NuggetKriging object.
#'
#' @param newy Numeric vector of new responses (output).
#'
#' @param newX Numeric matrix of new input points.
#'
#' @param ... Ignored.
#'
#' @return No return value. NuggetKriging object argument is modified.
#'
#' @section Caution: The method \emph{does not return the updated
#'     object}, but instead changes the content of
#'     \code{object}. This behaviour is quite unusual in R and
#'     differs from the behaviour of the methods
#'     \code{\link[DiceKriging]{update.km}} in \pkg{DiceKriging} and
#'     \code{\link{update,KM-method}}.
#'
#' @method update NuggetKriging
#' @export
#'
#' @examples
#' f <- function(x) 1- 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x)*x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#' points(X, y, col = "blue")
#'
#' k <- NuggetKriging(y, X, "matern3_2")
#'
#' ## include design points to see interpolation
#' x <- sort(c(X,seq(from = 0, to = 1, length.out = 101)))
#' p <- predict(k, x)
#' lines(x, p$mean, col = "blue")
#' polygon(c(x, rev(x)), c(p$mean - 2 * p$stdev, rev(p$mean + 2 * p$stdev)),
#'  border = NA, col = rgb(0, 0, 1, 0.2))
#'
#' newX <- as.matrix(runif(3))
#' newy <- f(newX) + 0.1 * rnorm(nrow(newX))
#' points(newX, newy, col = "red")
#'
#' ## change the content of the object 'k'
#' update(k, newy, newX)
#'
#' ## include design points to see interpolation
#' x <- sort(c(X,newX,seq(from = 0, to = 1, length.out = 101)))
#' p2 <- predict(k, x)
#' lines(x, p2$mean, col = "red")
#' polygon(c(x, rev(x)), c(p2$mean - 2 * p2$stdev, rev(p2$mean + 2 * p2$stdev)),
#'  border = NA, col = rgb(1, 0, 0, 0.2))
update.NuggetKriging <- function(object, newy, newX, ...) {

    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- nuggetkriging_model(object)
    if (is.data.frame(newX)) newX = data.matrix(newX)
    if (!is.matrix(newX)) newX <- matrix(newX, ncol = ncol(k$X))
    if (is.data.frame(newy)) newy = data.matrix(newy)
    if (!is.matrix(newy)) newy <- matrix(newy, ncol = ncol(k$y))
    if (ncol(newX) != ncol(k$X))
        stop("Object 'newX' must have ", ncol(k$X), " columns (instead of ",
             ncol(newX), ")")
    if (nrow(newy) != nrow(newX))
        stop("Objects 'newX' and 'newy' must have the same number of rows.")

    ## Modify 'object' in the parent environment
    nuggetkriging_update(object, newy, newX)

    invisible(NULL)
}


#' Compute Log-Likelihood of NuggetKriging Model
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object An S3 NuggetKriging object.
#' @param theta_alpha A numeric vector of (positive) range parameters and variance over variance plus nugget at
#'     which the log-likelihood will be evaluated.
#' @param grad Logical. Should the function return the gradient?
#' @param ... Not used.
#'
#' @return The log-Likelihood computed for given
#'     \eqn{\boldsymbol{theta_alpha}}{\frac{\sigma^2}{\sigma^2+nugget}}.
#'
#' @method logLikelihoodFun NuggetKriging
#' @export
#' @aliases logLikelihoodFun,NuggetKriging,NuggetKriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#'
#' k <- NuggetKriging(y, X, kernel = "matern3_2")
#' print(k)
#'
#' theta0 = k$theta()
#' ll_alpha <- function(alpha) logLikelihoodFun(k,cbind(theta0,alpha))$logLikelihood
#' a <- seq(from = 0.9, to = 1.0, length.out = 101)
#' plot(a, Vectorize(ll_alpha)(a), type = "l",xlim=c(0.9,1))
#' abline(v = k$sigma2()/(k$sigma2()+k$nugget()), col = "blue")
#'
#' alpha0 = k$sigma2()/(k$sigma2()+k$nugget())
#' ll_theta <- function(theta) logLikelihoodFun(k,cbind(theta,alpha0))$logLikelihood
#' t <- seq(from = 0.001, to = 2, length.out = 101)
#' plot(t, Vectorize(ll_theta)(t), type = 'l')
#' abline(v = k$theta(), col = "blue")
#'
#' ll <- function(theta_alpha) logLikelihoodFun(k,theta_alpha)$logLikelihood
#' a <- seq(from = 0.9, to = 1.0, length.out = 31)
#' t <- seq(from = 0.001, to = 2, length.out = 101)
#' contour(t,a,matrix(ncol=length(a),ll(expand.grid(t,a))),xlab="theta",ylab="sigma2/(sigma2+nugget)")
#' points(k$theta(),k$sigma2()/(k$sigma2()+k$nugget()),col='blue')
logLikelihoodFun.NuggetKriging <- function(object, theta_alpha,
                                  grad = FALSE, ...) {
    k <- nuggetkriging_model(object)
    if (is.data.frame(theta_alpha)) theta_alpha = data.matrix(theta_alpha)
    if (!is.matrix(theta_alpha)) theta_alpha <- matrix(theta_alpha, ncol = ncol(k$X)+1)
    if (ncol(theta_alpha) != ncol(k$X)+1)
        stop("Input theta_alpha must have ", ncol(k$X)+1, " columns (instead of ",
             ncol(theta_alpha),")")
    out <- list(logLikelihood = matrix(NA, nrow = nrow(theta_alpha)),
                logLikelihoodGrad = matrix(NA,nrow=nrow(theta_alpha),
                                           ncol = ncol(theta_alpha)))
    for (i in 1:nrow(theta_alpha)) {
        ll <- nuggetkriging_logLikelihoodFun(object, theta_alpha[i, ],
                                    grad = isTRUE(grad))
        out$logLikelihood[i] <- ll$logLikelihood
        if (isTRUE(grad)) out$logLikelihoodGrad[i, ] <- ll$logLikelihoodGrad
    }
    if (!isTRUE(grad)) out$logLikelihoodGrad <- NULL

    return(out)
}



#' Get logLikelihood of NuggetKriging Model
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object An S3 NuggetKriging object.
#' @param ... Not used.
#'
#' @return The logLikelihood computed for fitted
#'     \eqn{\boldsymbol{theta_alpha}}{\theta,\frac{\sigma^2}{\sigma^2+nugget}}.
#'
#' @method logLikelihood NuggetKriging
#' @export
#' @aliases logLikelihood,NuggetKriging,NuggetKriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#'
#' k <- NuggetKriging(y, X, kernel = "matern3_2", objective="LL")
#' print(k)
#'
#' logLikelihood(k)
logLikelihood.NuggetKriging <- function(object, ...) {
  return(nuggetkriging_logLikelihood(object))
}

#' Compute the log-marginal posterior of a kriging model, using the
#' prior XXXY.
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object S3 NuggetKriging object.
#' @param theta_alpha Numeric vector of correlation range and variance over variance plus nugget parameters at
#'     which the function is to be evaluated.
#' @param grad Logical. Should the function return the gradient
#'     (w.r.t theta_alpha)?
#' @param ... Not used.
#'
#' @return The value of the log-marginal posterior computed for the
#'     given vector \eqn{\boldsymbol{theta_alpha}}{\theta,\frac{\sigma^2}{\sigma^2+nugget}}.
#'
#' @method logMargPostFun NuggetKriging
#' @export
#' @aliases logMargPostFun,NuggetKriging,NuggetKriging-method
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
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#'
#' k <- NuggetKriging(y, X, "matern3_2", objective="LMP")
#' print(k)
#'
#' theta0 = k$theta()
#' lmp_alpha <- function(alpha) k$logMargPostFun(cbind(theta0,alpha))$logMargPost
#' a <- seq(from = 0.9, to = 1.0, length.out = 101)
#' plot(a, Vectorize(lmp_alpha)(a), type = "l",xlim=c(0.9,1))
#' abline(v = k$sigma2()/(k$sigma2()+k$nugget()), col = "blue")
#'
#' alpha0 = k$sigma2()/(k$sigma2()+k$nugget())
#' lmp_theta <- function(theta) k$logMargPostFun(cbind(theta,alpha0))$logMargPost
#' t <- seq(from = 0.001, to = 2, length.out = 101)
#' plot(t, Vectorize(lmp_theta)(t), type = 'l')
#' abline(v = k$theta(), col = "blue")
#'
#' lmp <- function(theta_alpha) k$logMargPostFun(theta_alpha)$logMargPost
#' t <- seq(from = 0.4, to = 0.6, length.out = 51)
#' a <- seq(from = 0.9, to = 1, length.out = 51)
#' contour(t,a,matrix(ncol=length(t),lmp(expand.grid(t,a))),
#'  nlevels=50,xlab="theta",ylab="sigma2/(sigma2+nugget)")
#' points(k$theta(),k$sigma2()/(k$sigma2()+k$nugget()),col='blue')
logMargPostFun.NuggetKriging <- function(object, theta_alpha, grad = FALSE, ...) {
    k <- nuggetkriging_model(object)
    if (is.data.frame(theta_alpha)) theta_alpha = data.matrix(theta_alpha)
    if (!is.matrix(theta_alpha)) theta_alpha <- matrix(theta_alpha,ncol=ncol(k$X)+1)
    if (ncol(theta_alpha) != ncol(k$X)+1)
        stop("Input theta_alpha must have ", ncol(k$X)+1, " columns (instead of ",
             ncol(theta_alpha), ")")
    out <- list(logMargPost = matrix(NA, nrow = nrow(theta_alpha)),
                logMargPostGrad = matrix(NA, nrow = nrow(theta_alpha),
                                         ncol = ncol(theta_alpha)))
    for (i in 1:nrow(theta_alpha)) {
        lmp <- nuggetkriging_logMargPostFun(object, theta_alpha[i, ], grad = isTRUE(grad))
        out$logMargPost[i] <- lmp$logMargPost
        if (isTRUE(grad)) out$logMargPostGrad[i, ] <- lmp$logMargPostGrad
    }
    if (!isTRUE(grad)) out$logMargPostGrad <- NULL
    return(out)
}


#' Get logMargPost of NuggetKriging Model
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object An S3 NuggetKriging object.
#' @param ... Not used.
#'
#' @return The logMargPost computed for fitted
#'     \eqn{\boldsymbol{theta_alpha}}{\theta,\frac{\sigma^2}{\sigma^2+nugget}}.
#'
#' @method logMargPost NuggetKriging
#' @export
#' @aliases logMargPost,NuggetKriging,NuggetKriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#'
#' k <- NuggetKriging(y, X, kernel = "matern3_2", objective="LMP")
#' print(k)
#'
#' logMargPost(k)
logMargPost.NuggetKriging <- function(object, ...) {
  return(nuggetkriging_logMargPost(object))
}


#' Duplicate a NuggetKriging Model
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param object An S3 NuggetKriging object.
#' @param ... Not used.
#'
#' @return The copy of object.
#'
#' @method copy NuggetKriging
#' @export
#' @aliases copy,NuggetKriging,NuggetKriging-method
#'
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X) + 0.1 * rnorm(nrow(X))
#'
#' k <- NuggetKriging(y, X, kernel = "matern3_2", objective="LMP")
#' print(k)
#'
#' print(copy(k))
copy.NuggetKriging <- function(object, ...) {
  return(nuggetkriging_copy(object))
}
