## ****************************************************************************
## This file contains stuff related to the generic functions, be they true
## generic function (for S4 methods) or S3 generic.
##
## Note that the DiceKriging function defines 'predict', 'simulate'
## and 'update' as generic, keeping their original signature in the
## 'stats' package.
##
## ****************************************************************************

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

## True generic
setGeneric(name = "covMat",
           def = function(object, ...) standardGeneric("covMat"))

## *****************************************************************************
##' Compute the covariance matrix of a model given in \code{object},
##' between given set of points.
##'
##' @title covariance function
##'
##' @param object An object representing a fitted model.
##' @param ... Further arguments of function (eg. points, range).
##'
##' @return The covariance matrix.
##' @export
covMat <- function(object, ...) {
    UseMethod("covMat")
}


setGeneric(name = "logLikelihoodFun",
           def = function(object, ...) standardGeneric("logLikelihoodFun"))

setGeneric(name = "logMargPostFun",
           def = function(object, ...) standardGeneric("logMargPostFun"))

setGeneric(name = "leaveOneOutFun",
           def = function(object, ...) standardGeneric("leaveOneOutFun"))

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
leaveOneOutFun <- function(object, ...) {
    UseMethod("leaveOneOutFun")
}

## *****************************************************************************
##' Compute the leave-One-Out vector error of a model given in \code{object},
##' at a different value of the parameters.
##'
##' @title Leave-One-Out vector
##' 
##' @param object An object representing a fitted model.
##' @param ... Further arguments of function (eg. range).
##'
##' @return The Leave-One-Out errors (mean and stdev) for each conditional point.
##' @export
leaveOneOutVec <- function(object, ...) {
    UseMethod("leaveOneOutVec")
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
logLikelihoodFun <- function(object, ...) {
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
logMargPostFun <- function(object, ...) {
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
leaveOneOut <- function(object, ...) {
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
logLikelihood <- function(object, ...) {
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
logMargPost <- function(object, ...) {
    UseMethod("logMargPost")
}

## *****************************************************************************
##' Update previous simulate of a model given in
##' \code{object}.
##'
##' @title Update simulation of model on data.
##'
##' @param object An object representing a fitted model.
##' @param ... Further arguments of function
##'
##' @return Updated simulation of model output.
##' @export
update_simulate <- function(object, ...) {
    UseMethod("update_simulate")
}

## *****************************************************************************
##' Duplicate a model given in
##' \code{object}.
##'
##' @title Duplicate object.
##'
##' @param object An object representing a fitted model.
##' @param ... Ignored.
##'
##' @return The copied object.
##' @export
copy <- function(object, ...) {
    UseMethod("copy")
}

## *****************************************************************************
##' Fit a model given in
##' \code{object}.
##'
##' @title Fit model on data.
##'
##' @param object An object representing a fitted model.
##' @param ... Further arguments of function
##'
##' @return No return value. Kriging object argument is modified.
##' @export
fit <- function(object, ...) {
    UseMethod("fit")
}

## *****************************************************************************
##' Save a Kriging Model inside a file. Back to base::save if argument 
##' is not a Kriging object.
##' 
##' @author Yann Richet \email{yann.richet@irsn.fr}
##' 
##' @param ... An object representing a model.
##' @param filename A file holding the object.
##' 
##' @export
save <- function(..., filename=NULL) {
    if (is.null(filename) ||
        !is.character(filename) ||
        endsWith(filename,"Rdata") ||
        endsWith(filename,"RData") ||
        endsWith(filename,"rdata") ||
        endsWith(filename,"Rds") ||
        endsWith(filename,"rds")
    ) {# back to base::save
        # warning("Using base::save")
        if (! "envir" %in% names(list(...)))
          envir = parent.frame(n=1)
        else
          envir = list(...)$envir
        if (is.null(filename) || !is.character(filename))
            base::save(...,envir=envir)
        else 
            base::save(list=as.character(substitute(list(...))), file=filename,envir=envir)
    } else {
        if (is.null(filename) || !is.character(filename))
            stop("'filename' must be a string")
        if (length(L <- list(...)) > 1) stop("Too many arguments: ",L)
        object = list(...)[[1]]
        k_class = class(object)
        if (is.null(k_class))
            stop("No class for: ",as.character(substitute(object)))
        
        if (k_class=="Kriging")
            return(save.Kriging(object, filename))
        else if (k_class=="NuggetKriging")
            return(save.NuggetKriging(object, filename))
        else if (k_class=="NoiseKriging")
            return(save.NoiseKriging(object, filename))
        else 
            stop("Unknown Kriging class: ",k_class)
    }
}

## *****************************************************************************
##' Load any Kriging Model from a file storage. Back to base::load if not a Kriging object.
##'
##' @author Yann Richet \email{yann.richet@irsn.fr}
##'
##' @param filename A file holding any Kriging object.
##' @param ... options if base::load is used.
##'
##' @return The loaded "*"Kriging object, or nothing if base::load is used (update parent environment).
##'
##' @export
##' 
##' @examples
##' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
##' set.seed(123)
##' X <- as.matrix(runif(10))
##' y <- f(X)
##'
##' k <- Kriging(y, X, kernel = "matern3_2", objective="LMP")
##' print(k)
##' 
##' outfile = tempfile("k.json") 
##' save(k,outfile)
##'
##' print(load(outfile))
load <- function(filename, ...) {
    if (is.null(filename) ||
        !is.character(filename) ||
        endsWith(filename,"Rdata") ||
        endsWith(filename,"RData") ||
        endsWith(filename,"rdata") ||
        endsWith(filename,"Rds") ||
        endsWith(filename,"rds")
        ) {# back to base::load
        # warning("Using base::load")
        if (! "envir" %in% names(list(...))) {
            envir = parent.frame(n=2)
        if (is.null(filename) || !is.character(filename))
            base::load(...,envir=envir)
        else 
            base::load(file=filename,...,envir=envir)
        } else {
          if (is.null(filename) || !is.character(filename))
            base::load(...)
          else 
            base::load(file=filename,...)
        }
    } else {
        if (length(L <- list(...)) > 0) warnOnDots(L)
        k_class = NULL
        base::try(k_class <- class_saved(filename))
        if (is.null(k_class)) {# back to base::load
            # warning("Using base::load")
          if (! "envir" %in% names(list(...))) {
            envir = parent.frame(n=2)
            if (is.null(filename) || !is.character(filename))
              base::load(...,envir=envir)
            else 
              base::load(file=filename,...,envir=envir)
          } else {
            if (is.null(filename) || !is.character(filename))
              base::load(...)
            else 
              base::load(file=filename,...)
          }
        } else
            if (k_class=="Kriging")
                return(load.Kriging(filename))
            else if (k_class=="NuggetKriging")
                return(load.NuggetKriging(filename))
            else if (k_class=="NoiseKriging")
                return(load.NoiseKriging(filename))
            else 
                stop("Unknown Kriging class: ",k_class)
    }
}
