## ****************************************************************************
## This file contains stuff related to the generic functions, be they true
## generic function (for S4 methods) or S3 generic.
##
## Note that the DiceKriging function defines 'predict', 'simulate'
## and 'update' as generic, keeping their original signature in the
## 'stats' package.
##
## ****************************************************************************

## True generic
setGeneric(name = "logLikelihoodFun",
           def = function(object, ...) standardGeneric("logLikelihoodFun"))

setGeneric(name = "logMargPostFun",
           def = function(object, ...) standardGeneric("logMargPostFun"))

setGeneric(name = "leaveOneOutFun",
           def = function(object, ...) standardGeneric("leaveOneOutFun"))

## *****************************************************************************
##' Coerce an object into an object with S4 class \code{"km"} from the
##' \pkg{DiceKriging} package.
##'
##' Such a coercion is typically used to compare the performance of
##' the methods implemented in the current \pkg{rlibkriging} package to
##' those which are available in the \pkg{DiceKriging} package.
##'
##' @title Coerce an Object into a \code{km} Object
##'
##' @param x Object to be coerced.
##' @param ... Further arguments for methods.
##' @return An object with S4 class \code{"km"}.
##' 
##' @export 
as.km <- function(x, ...) {
    UseMethod("as.km")
}

## *****************************************************************************
##' Compute the leave-One-Out error of a model given in \code{object},
##' at a different value of the parameters.
##'
##' @title Leave-One-Out function
##' 
##' @param object An object representing a fitted model.
##' @param ... Further arguments of function (eg. range).
##'
##' @return The Leave-One-Out sum of squares.
##' @export
leaveOneOutFun <- function (object, ...) {
    UseMethod("leaveOneOutFun")
}

## *****************************************************************************
##' Compute the log-Likelihood of a model given in \code{object},
##' at a different value of the parameters.
##'
##' @title Log-Likelihood function
##'
##' @param object An object representing a fitted model.
##' @param ... Further arguments of function (eg. range).
##'
##' @return The log-likelihood.
##' @export
logLikelihoodFun <- function (object, ...) {
    UseMethod("logLikelihoodFun")
}

## *****************************************************************************
##' Compute the log-Marginal Posterior of a model given in
##' \code{object}, at a different value of the parameters.
##'
##' @title log-Marginal Posterior function
##'
##' @param object An object representing a fitted model.
##' @param ... Further arguments of function (eg. range).
##'
##' @return The log-marginal posterior.
##' @export
logMargPostFun <- function (object, ...) {
    UseMethod("logMargPostFun")
}


## True generic
setGeneric(name = "logLikelihood",
           def = function(object, ...) standardGeneric("logLikelihood"))

setGeneric(name = "logMargPost",
           def = function(object, ...) standardGeneric("logMargPost"))

setGeneric(name = "leaveOneOut",
           def = function(object, ...) standardGeneric("leaveOneOut"))

## *****************************************************************************
##' Compute the leave-One-Out error of a model given in \code{object}.
##'
##' @title Compute Leave-One-Out
##' 
##' @param object An object representing a fitted model.
##' @param ... Ignored.
##'
##' @return The Leave-One-Out sum of squares.
##' @export
leaveOneOut <- function (object, ...) {
    UseMethod("leaveOneOut")
}

## *****************************************************************************
##' Compute the log-Likelihood of a model given in \code{object}.
##'
##' @title Compute Log-Likelihood
##'
##' @param object An object representing a fitted model.
##' @param ... Ignored.
##'
##' @return The log-likelihood.
##' @export
logLikelihood <- function (object, ...) {
    UseMethod("logLikelihood")
}

## *****************************************************************************
##' Compute the log-Marginal Posterior of a model given in
##' \code{object}.
##'
##' @title Compute log-Marginal Posterior
##'
##' @param object An object representing a fitted model.
##' @param ... Ignored.
##'
##' @return The log-marginal posterior.
##' @export
logMargPost <- function (object, ...) {
    UseMethod("logMargPost")
}
