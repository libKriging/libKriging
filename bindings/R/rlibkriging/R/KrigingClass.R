## ****************************************************************************
## This file contains stuff related to the S3 class "Kriging".
## As an S3 class, it has no formal definition. 
## ****************************************************************************


## ****************************************************************************
#' Create an object with S3 class \code{"Kriging"} using
#' the \pkg{libKriging} library.
#'
#' The hyper-parameters (variance and vector of correlation ranges)
#' are estimated thanks to the optimization of a criterion given by
#' \code{objective}, using the method given in \code{optim}.
#'
#' @title Create a \code{Kriging} Object using \pkg{libKriging}
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
#' @param regmodel Universal Kriging linear trend.
#'
#' @param normalize Logical. If \code{TRUE} both the input matrix
#'     \code{X} and the response \code{y} in normalized to take
#'     values in the interval \eqn{[0, 1]}.
#'
#' @param optim Character giving the Optimization method used to fit
#'     hyper-parameters. Possible values are: \code{"BFGS"},
#'     \code{"Newton"} and \code{"none"}, the later simply keeping
#'     the values given in \code{parameters}. The method
#'     \code{"BFGS"} uses the gradient of the objective. The method
#'     \code{"Newton"} uses both the gradient and the Hessian of the
#'     objective.
#'
#' @param objective Character giving the objective function to
#'     optimize. Possible values are: \code{"LL"} for the
#'     Log-Likelihood, \code{"LOO"} for the Leave-One-Out sum of
#'     squares and \code{"LMP"} for the Log-Marginal Posterior.
#' 
#' @param parameters Initial values for the hyper-parameters. When
#'     provided this must be named list with elements \code{"sigma2"}
#'     and \code{"theta"} containing the initial value(s) for the
#'     variance and for the range parameters. If \code{theta} is a
#'     matrix with more than one row, each row is used as a starting
#'     point for optimization.
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
Kriging <- function(y, X, kernel,
                    regmodel = c("constant", "linear", "interactive"),
                    normalize = FALSE,
                    optim = c("BFGS", "Newton", "none"),
                    objective = c("LL", "LOO", "LMP"),
                    parameters = NULL) {

    regmodel <- match.arg(regmodel)
    objective <- match.arg(objective)
    if (is.character(optim)) optim <- optim[1] #optim <- match.arg(optim) because we can use BFGS10 for 10 (multistart) BFGS
    nk <- new_Kriging(y = y, X = X, kernel = kernel,
                      regmodel = regmodel,
                      normalize = normalize,
                      optim = optim,
                      objective = objective,
                      parameters = parameters)
    class(nk) <- "Kriging"
    # This will allow to call methods (like in Python/Matlab/Octave) using `k$m(...)` as well as R-style `m(k, ...)`.
    for (f in methods(class=class(nk))) {
        if (regexec(paste0(".",class(nk)),f)[[1]]>0) {
            f_anon = sub(paste0(".",class(nk)),"",fixed=TRUE,f)
            eval(parse(text=paste0(
                "nk$", f_anon, " <- function(...) ", f_anon, "(nk,...)"
                )))
        }
    }
    # This will allow to access kriging data/props using `k$d()`
    for (d in c('kernel','optim','objective','X','centerX','scaleX','y','centerY','scaleY','regmodel','F','T','M','z','beta','is_beta_estim','theta','is_theta_estim','sigma2','is_sigma2_estim')) {
        eval(parse(text=paste0(
            "nk$", d, " <- function() kriging_", d, "(nk)"
            )))
    }
    nk
}

## ****************************************************************************
#' Coerce a \code{Kriging} Object into a List
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
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

## setMethod("as.list", "Kriging", as.list.Kriging)

## ****************************************************************************
#' Coerce a \code{Kriging} object into the \code{"km"} class of the
#' \pkg{DiceKriging} package.
#'
#' @title Coerce a \code{Kriging} Object into the Class \code{"km"}
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param x An object with S3 class \code{"Kriging"}.
#' 
#' @param .call Force the \code{call} slot to be filled in the
#'     returned \code{km} object.
#'
#' @param ... Not used.
#' 
#' @return An object of having the S4 class \code{"KM"} which extends
#'     the \code{"km"} class of the \pkg{DiceKriging} package and
#'     contains an extra \code{Kriging} slot.
#'
#' @importFrom methods new
#' @importFrom stats model.matrix
#' @export
#' @method as.km Kriging
#' 
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' set.seed(123)
#' X <- as.matrix(runif(10))
#' y <- f(X)
#' 
#' k <- Kriging(y, X, "matern3_2")
#' print(k)
#' 
#' k_km <- as.km(k)
#' print(k_km)
as.km.Kriging <- function(x, .call = NULL, ...) {
    
    ## loadDiceKriging()
    ## if (! "DiceKriging" %in% installed.packages())
    ##     stop("DiceKriging must be installed to use its wrapper from libKriging.")

    if (!requireNamespace("DiceKriging", quietly = TRUE))
        stop("Package \"DiceKriging\" not found")
    
    model <- new("KM")
    model@Kriging <- x
    
    if (is.null(.call))
        model@call <- match.call()
    else
        model@call <- .call
    
    m <- kriging_model(x)
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
    
    model@case <- "LLconcentration_beta_sigma2"
    
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
                      nugget = 0, nugget.flag = FALSE, nugget.estim = FALSE,
                      known.covparam = "")
    
    covStruct@range.names <- "theta" 
    covStruct@paramset.n <- as.integer(1)
    covStruct@param.n <- as.integer(model@d)
    covStruct@range.n <- as.integer(model@d)
    covStruct@range.val <- as.numeric(m$theta)
    model@covariance <- covStruct 
    
    return(model)
}

## ****************************************************************************
#' Print the content of a \code{Kriging} object.
#'
#' @title Print a \code{Kriging} object
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @param x A (S3) \code{Kriging} Object.
#' @param ... Ignored.
#'
#' @return NULL
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
    if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
    k=kriging_model(x)
    p = paste0("Kriging model:\n\n",kriging_summary(x),"\n")
    cat(p)
    ## return(p)
}

## setMethod("print", "Kriging", print.Kriging)


## ****************************************************************************
#' Predict from a \code{Kriging} object.
#'
#' Given "new" input points, the method compute the expectation,
#' variance and (optionnally) the covariance of the corresponding
#' stochastic process, conditional on the values at the input points
#' used when fitting the model.
#'
#' @title Prediction Method  for a \code{Kriging} Object
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param object S3 Kriging object.
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
#' polygon(c(x, rev(x)), c(p$mean - 2 * p$stdev, rev(p$mean + 2 * p$stdev)), border = NA, col = rgb(0, 0, 1, 0.2))
predict.Kriging <- function(object, x, stdev = TRUE, cov = FALSE, deriv = FALSE, ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- kriging_model(object)
    ## manage the data frame case. Ideally we should then warn
    if (is.data.frame(x)) x = data.matrix(x)
    if (!is.matrix(x)) x=matrix(x,ncol=ncol(k$X))
    if (ncol(x) != ncol(k$X))
        stop("Input x must have ", ncol(k$X), " columns (instead of ",
             ncol(x), ")")
    return(kriging_predict(object, x, stdev, cov, deriv))
}

## predict <- function (...) UseMethod("predict")
## setMethod("predict", "Kriging", predict.Kriging)

## ****************************************************************************
#' Simulation from a \code{Kriging} model object.
#'
#' This method draws paths of the stochastic process at new input
#' points conditional on the values at the input points used in the
#' fit.
#'
#' @title Simulation from a \code{Kriging} Object
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param object S3 Kriging object.
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
simulate.Kriging <- function(object, nsim = 1, seed = 123, x,  ...) {
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- kriging_model(object) 
    if (is.data.frame(x)) x = data.matrix(x)
    if (!is.matrix(x)) x = matrix(x, ncol = ncol(k$X))
    if (ncol(x) != ncol(k$X))
        stop("Input x must have ", ncol(k$X), " columns (instead of ",
             ncol(x),")")
    ## XXXY
    if (is.null(seed)) seed <- floor(runif(1) * 99999)
    return(kriging_simulate(object, nsim = nsim, seed = seed, X = x))
}

## simulate <- function (...) UseMethod("simulate")
## setMethod("simulate", "Kriging", simulate.Kriging)


## removed by Yves @aliases update,Kriging,Kriging-method
## *****************************************************************************
#' Update a \code{Kriging} model object with new points
#' 
#' @title Update a \code{Kriging} Object with New Points
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' 
#' @param object S3 Kriging object.
#'
#' @param newy Numeric vector of new responses (output).
#'
#' @param newX Numeric matrix of new input points.
#' 
#' @param ... Ignored.
#' 
#' @section Caution: The method \emph{does not return the updated
#'     object}, but instead changes the content of
#'     \code{object}. This behaviour is quite unusual in R and
#'     differs from the behaviour of the methods
#'     \code{\link[DiceKriging]{update.km}} in \pkg{DiceKriging} and
#'     \code{\link{update,KM-method}}.
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
#' polygon(c(x, rev(x)), c(p$mean - 2 * p$stdev, rev(p$mean + 2 * p$stdev)), border = NA, col = rgb(0, 0, 1, 0.2))
#' 
#' newX <- as.matrix(runif(3))
#' newy <- f(newX)
#' points(newX, newy, col = "red")
#' 
#' ## change the content of the object 'k'
#' update(k, newy, newX)
#' 
#' x <- seq(from = 0, to = 1, length.out = 101)
#' p2 <- predict(k, x)
#' lines(x, p2$mean, col = "red")
#' polygon(c(x, rev(x)), c(p2$mean - 2 * p2$stdev, rev(p2$mean + 2 * p2$stdev)), border = NA, col = rgb(1, 0, 0, 0.2))
update.Kriging <- function(object, newy, newX, ...) {
    
    if (length(L <- list(...)) > 0) warnOnDots(L)
    k <- kriging_model(object)
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
    kriging_update(object, newy, newX)
    
    invisible(NULL)
    
}

## update <- function(...) UseMethod("update")
## setMethod("update", "Kriging", update.Kriging)
## setGeneric(name = "update", def = function(...) standardGeneric("update"))

## ****************************************************************************
#' Compute Log-Likelihood of Kriging Model
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param object An S3 Kriging object.
#' @param theta A numeric vector of (positive) range parameters at
#'     which the log-likelihood will be evaluated.
#' @param grad Logical. Should the function return the gradient?
#' @param hess Logical. Should the function return Hessian?
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
                                  grad = FALSE, hess = FALSE, ...) {
    k <- kriging_model(object)
    if (is.data.frame(theta)) theta = data.matrix(theta)
    if (!is.matrix(theta)) theta <- matrix(theta, ncol = ncol(k$X))
    if (ncol(theta) != ncol(k$X))
        stop("Input theta must have ", ncol(k$X), " columns (instead of ",
             ncol(theta),")")
    out <- list(logLikelihood = matrix(NA, nrow = nrow(theta)),
                logLikelihoodGrad = matrix(NA,nrow=nrow(theta),
                                           ncol = ncol(theta)),
                logLikelihoodHess = array(NA, dim = c(nrow(theta), ncol(theta),
                                                      ncol(theta))))
    for (i in 1:nrow(theta)) {
        ll <- kriging_logLikelihoodFun(object, theta[i, ],
                                    grad = isTRUE(grad), hess = isTRUE(hess))
        out$logLikelihood[i] <- ll$logLikelihood
        if (isTRUE(grad)) out$logLikelihoodGrad[i, ] <- ll$logLikelihoodGrad
        if (isTRUE(hess)) out$logLikelihoodHess[i, , ] <- ll$logLikelihoodHess
    }
    if (!isTRUE(grad)) out$logLikelihoodGrad <- NULL
    if (!isTRUE(hess)) out$logLikelihoodHess <- NULL
  
    return(out)
}

## ****************************************************************************
#' Get Log-Likelihood of Kriging Model
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
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
  return(kriging_logLikelihood(object))
}

## setMethod("logLikelihood", "Kriging", logLikelihoodFun.Kriging)

## ****************************************************************************
#' Compute Leave-One-Out (LOO) error for an object with S3 class
#' \code{"Kriging"} representing a kriging model.
#'
#' The returned value is the sum of squares \eqn{\sum_{i=1}^n [y_i -
#' \hat{y}_{i,(-i)}]^2} where \eqn{\hat{y}_{i,(-i)}} is the
#' prediction of \eqn{y_i}{y[i]} based on the the observations \eqn{y_j}{y[j]}
#' with \eqn{j \neq i}{j != i}.
#'
#' @title Leave-One-Out Method for the S3 class \code{"Kriging"}
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param object A \code{Kriging} object.
#' @param theta A numeric vector of range parameters at which the LOO
#'     will be evaluated.
#' 
#' @param grad Logical. Should the gradient (w.r.t. \code{theta}) be
#'     returned?
#'
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
leaveOneOutFun.Kriging <- function(object, theta, grad = FALSE, ...) {
    k <- kriging_model(object) 
    if (is.data.frame(theta)) theta = data.matrix(theta)
    if (!is.matrix(theta)) theta <- matrix(theta,ncol=ncol(k$X))
    if (ncol(theta) != ncol(k$X))
        stop("Input theta must have ", ncol(k$X), " columns (instead of ",
             ncol(theta),")")
    out <- list(leaveOneOut = matrix(NA, nrow = nrow(theta)),
                leaveOneOutGrad = matrix(NA, nrow = nrow(theta),
                                         ncol = ncol(theta)))
    for (i in 1:nrow(theta)) {
        loo <- kriging_leaveOneOutFun(object,theta[i,], isTRUE(grad))
        out$leaveOneOut[i] <- loo$leaveOneOut
        if (isTRUE(grad)) out$leaveOneOutGrad[i, ] <- loo$leaveOneOutGrad
    }    
    if (!isTRUE(grad)) out$leaveOneOutGrad <- NULL
    return(out)
}

## ****************************************************************************
#' Get leaveOneOut of Kriging Model
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
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
  return(kriging_leaveOneOut(object))
}

##****************************************************************************
#' Compute the log-marginal posterior of a kriging model, using the
#' prior XXXY.
#'
#' @title Compute Log-Marginal-Posterior of Kriging Model
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param object S3 Kriging object.
#' @param theta Numeric vector of correlation range parameters at
#'     which the function is to be evaluated.
#' @param grad Logical. Should the function return the gradient
#'     (w.r.t theta)?
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
logMargPostFun.Kriging <- function(object, theta, grad = FALSE, ...) {
    k <- kriging_model(object)
    if (is.data.frame(theta)) theta = data.matrix(theta)
    if (!is.matrix(theta)) theta <- matrix(theta,ncol=ncol(k$X))
    if (ncol(theta) != ncol(k$X))
        stop("Input theta must have ", ncol(k$X), " columns (instead of ",
             ncol(theta), ")")
    out <- list(logMargPost = matrix(NA, nrow = nrow(theta)),
                logMargPostGrad = matrix(NA, nrow = nrow(theta),
                                         ncol = ncol(theta)))
    for (i in 1:nrow(theta)) {
        lmp <- kriging_logMargPostFun(object, theta[i, ], grad = isTRUE(grad))
        out$logMargPost[i] <- lmp$logMargPost
        if (isTRUE(grad)) out$logMargPostGrad[i, ] <- lmp$logMargPostGrad
    }
    if (!isTRUE(grad)) out$logMargPostGrad <- NULL
    return(out)
}

## ****************************************************************************
#' Get logMargPost of Kriging Model
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
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
  return(kriging_logMargPost(object))
}
