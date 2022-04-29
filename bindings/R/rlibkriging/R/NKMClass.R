## *****************************************************************************
## This file contains stuff related to the S4 class "NKM" including its
## definition as a class extending "km" from the DiceKriging package.
## ****************************************************************************
    
#if (!requireNamespace("DiceKriging", quietly = TRUE)) {
#    stop("Package \"DiceKriging\" not found")
#}

## Register the S3 class "NuggetKriging" to define the class of the @NuggetKriging
## slot in a `NKM` object
setOldClass("NuggetKriging")

## *****************************************************************************
#' @title S4 class for NuggetKriging Models Extending the \code{"km"} Class
#' 
#' @description This class is intended to be used either by using its
#'     own dedicated S4 methods or by using the S4 methods inherited
#'     from the \code{"km"} class of the \pkg{libKriging} package.
#'
#'
#' @slot d,n,X,y,p,F Number of (numeric) inputs, number of
#'     observations, design matrix, response vector, number of trend
#'     variables, trend matrix.
#' 
#' @slot trend.formula,trend.coef Formula used for the trend, vector
#' \eqn{\hat{\boldsymbol{\beta}}}{betaHat} of estimated (or fixed)
#' trend coefficients with length \eqn{p}.
#'
#' @slot covariance A S4 object with class \code{"covTensorProduct"}
#' representing a covariance kernel.
#' 
#' @slot noise.flag,noise.var Logical flag and numeric value for an
#'     optional noise term.
#'
#' @slot known.param A character code indicating what parameters are
#'     known.
#'
#' @slot lower,upper Bounds on the correlation range parameters.
#'
#' @slot method,penalty,optim.method,control,gr,parinit Objects
#'     defining the estimation criterion, the optimization.
#' 
#' @slot T,M,z Auxiliary variables (matrices and vectors) that can be
#'     used in several computations.
#'
#' @slot case The possible concentration (a.k.a. profiling) of the
#'     likelihood.
#' 
#' @slot param.estim Logical. Is an estimation used?
#'
#' @slot NuggetKriging A copy of the \code{NuggetKriging} object used to create
#'     the current \code{NKM} object.
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
#'
#' @rdname NKM-class
#'
#' @seealso \code{\link[DiceKriging]{km-class}} in the
#'     \pkg{DiceKriging} package. The creator \code{\link{NKM}}.
#' 
#' @export
#' 
if (requireNamespace("DiceKriging", quietly = TRUE))
  setClass("NKM", slots = c("NuggetKriging" = "NuggetKriging"), contains = "km")

## *****************************************************************************
#' Create an object of S4 class \code{"NKM"} similar to a
#' \code{km} object in the \pkg{DiceKriging} package.
#' 
#' The class \code{"NKM"} extends the \code{"km"} class of the
#' \pkg{DiceKriging} package, hence has all slots of \code{"km"}. It
#' also has an extra slot \code{"NuggetKriging"} slot which contains a copy
#' of the original object. 
#'
#' @title Create an \code{NKM} Object
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param formula R formula object to setup the linear trend in
#'     Universal NuggetKriging. Supports \code{~ 1}, ~. and \code{~ .^2}.
#'
#' @param design Data frame. The design of experiments.
#'
#' @param response Vector of output values.
#'
#' @param covtype Covariance structure. For now all the kernels are
#'     tensor product kernels.
#' 
#' @param coef.trend Optional value for a fixed vector of trend
#'     coefficients.  If given, no optimization is done.
#'
#' @param coef.cov Optional value for a fixed correlation range
#'     value. If given, no optimization is done.
#'
#' @param coef.var Optional value for a fixed variance. If given, no
#'     optimization is done.
#'
#' @param nugget,nugget.estim,noise.var Not implemented yet. 
#'
#' @param estim.method Estimation criterion. \code{"MLE"} for
#'     Maximum-Likelihood or \code{"LOO"} for Leave-One-Out
#'     cross-validation.
#' 
#' @param penalty Not implemented yet.
#'
#' @param optim.method Optimization algorithm used in the
#'     optimization of the objective given in
#'     \code{estim.method}. Supports \code{"BFGS"}.
#'
#' @param lower,upper Not implemented yet. 
#'
#' @param parinit Initial values for the correlation ranges which
#'     will be optimized using \code{optim.method}.
#'
#' @param multistart,control,gr,iso Not implemented yet. 
#'
#' @param scaling,knots,kernel, Not implemented yet. 
#'
#' @param ... Ignored.
#'
#' @return A NKM object. See \bold{Details}.
#'
#' @seealso \code{\link[DiceKriging]{km}} in the \pkg{DiceKriging}
#'     package for more details on the slots.
#'
#' @export NKM
#' @examples
#' # a 16-points factorial design, and the corresponding response
#' d <- 2; n <- 16
#' design.fact <- as.matrix(expand.grid(x1 = seq(0, 1, length = 4),
#'                                      x2 = seq(0, 1, length = 4)))
#' y <- apply(design.fact, 1, DiceKriging::branin) + rnorm(nrow(design.fact))
#' 
#' # Using `km` from DiceKriging and a similar `NKM` object 
#' # kriging model 1 : matern5_2 covariance structure, no trend, no nugget effect
#' km1 <- DiceKriging::km(design = design.fact, response = y, covtype = "gauss",
#'                        nugget.estim=TRUE,
#'                        parinit = c(.5, 1), control = list(trace = FALSE))
#' KM1 <- NKM(design = design.fact, response = y, covtype = "gauss",
#'           parinit = c(.5, 1))
#' 
NKM <- function(formula = ~1, design, response,
               covtype = c("matern5_2", "gauss", "matern3_2", "exp"),
               coef.trend = NULL, coef.cov = NULL, coef.var = NULL,
               nugget = NULL, nugget.estim = TRUE, noise.var = NULL,
               estim.method = c("MLE", "LOO"), penalty = NULL,
               optim.method = "BFGS",
               lower = NULL, upper = NULL, parinit = NULL,
               multistart = 1, control = NULL,
               gr = TRUE, iso = FALSE, scaling = FALSE,
               knots = NULL, kernel = NULL,
               ...) {

    covtype <- match.arg(covtype)
    estim.method <- match.arg(estim.method)
    formula <- formula2regmodel(formula)

    ## get rid of unimplemented formals.
    if (!is.null(penalty)) {
        stop("The formal arg 'penalty' can not be used for now.")
    }
    if (!nugget.estim) {
        stop("The formal args 'nugget.estim=FALSE' ",
             "can only be used with KM()")
    }
    if (!is.null(nugget) || !is.null(noise.var)) {
        stop("The formal args 'nugget' and 'noise.var' ",
             "can not be used for now.")
    }
    if (!is.null(lower) || !is.null(upper)) {
        stop("The formal args 'lower', 'upper' and 'parinit' ",
             "can not be used for now.")
    }
    if ((multistart != 1) || !is.null(control) || !gr || iso) {
         stop("The formal args 'multistart', 'control', 'gr' ",
              "and 'iso' can not be used for now.")
    }
    if (scaling || !is.null(knots) || !is.null(kernel)) {
        stop("The formal args 'scaling', 'knots', 'kernel' ",
             "can not be used for now.")
    }

    ## check the design and response 
    if (!is.matrix(design)) design <- as.matrix(design)
    response <- as.matrix(response)
    if (!is.numeric(response) || (length(response) != nrow(design))) {
        stop("bad 'response'. Must be coercible to a numeric column ",
             "matrix with ", nrow(design), " rows")
    }
    
    if (estim.method == "MLE") estim.method <- "LL"
    else if (estim.method == "LOO") estim.method <- "LOO"
    
    if (optim.method != "BFGS")
        warning("Cannot setup optim.method ", optim.method,". Ignored.")

    ## Make the parameter list. These are coped by their name "sigma",
    ## 'theta' and 'beta'.
    
    parameters <- list()
    if (!is.null(coef.var))
        parameters <- c(parameters, list(sigma2 = coef.var))
    if (!is.null(coef.cov)) {
        parameters <- c(parameters,
                        list(theta = matrix(coef.cov, ncol = ncol(design))))
        optim.method <- "none"
        ## XXXY 
        warning("Since 'coef.cov' is provided 'optim.method' is set to ",
                "\"none\"")
    }  
    if (!is.null(coef.trend)) {
        parameters <- c(parameters, list(beta = matrix(coef.trend)))
    }
    if (!is.null(parinit)) {
        parameters <- c(parameters,
                        list(theta = matrix(parinit, ncol = ncol(design))))
    }
    if (!is.null(nugget)) {
        parameters <- c(parameters,
                        list(nugget = nugget))
    }
    if (length(parameters) == 0) parameters <- NULL
    
    r <- rlibkriging::NuggetKriging(y = response, X = design, kernel = covtype,
                              regmodel = formula,
                              normalize = FALSE,
                              objective = estim.method, optim = optim.method,
                              parameters = parameters)
    
    return(as.km.NuggetKriging(r, .call = match.call()))
}

## *****************************************************************************
## 'predict' S4 method We no longer export 'predict.NKM'
## *****************************************************************************

predict.NKM <- function(object, newdata, type = "UK",
                       se.compute = TRUE,
                       cov.compute = FALSE,
                       light.return = TRUE,
                       bias.correct = FALSE, checkNames = FALSE,...) {
    
    if (length(L <- list(...)) > 0) warnOnDots(L)
    
    if (isTRUE(checkNames)) stop("'checkNames = TRUE' unsupported.")
    if (isTRUE(bias.correct)) stop("'bias.correct = TRUE' unsupported.")
    if (!isTRUE(light.return)) stop("'light.return = FALSE' unsupported.")
    if (type != "UK") stop("'type != UK' unsupported.")
    
    y.predict <- predict.NuggetKriging(object@NuggetKriging, x = newdata,
                                 stdev = se.compute, cov = cov.compute)
    
    output.list <- list()
    ## output.list$trend <- y.predict.trend
    output.list$mean <- y.predict$mean
    
    if (se.compute) {		
        s2.predict <- y.predict$stdev^2
        q95 <- qt(0.975, object@n - object@p)
        
        lower95 <- y.predict$mean - q95 * sqrt(s2.predict)
        upper95 <- y.predict$mean + q95 * sqrt(s2.predict)
        
        output.list$sd <- sqrt(s2.predict)
        output.list$lower95 <- lower95
        output.list$upper95 <- upper95
    }
    
    if (cov.compute) {		
        output.list$cov <- y.predict$cov
    }
    
    F.newdata <- model.matrix(object@trend.formula, data = data.frame(newdata))
    output.list$trend <- F.newdata %*% object@trend.coef
    
    return(output.list)
}

## *****************************************************************************
#' Compute predictions for the response at new given input
#' points. These conditional mean, the conditional standard deviation
#' and confidence limits at the 95\% level. Optionnally the
#' conditional covariance can be returned as well.
#'
#' Without a dedicated \code{predict} method for the class
#' \code{"NKM"}, this method would have been inherited from the
#' \code{"km"} class. The dedicated method is expected to run faster.
#' A comparison can be made by coercing a \code{NKM} object to a
#' \code{km} object with \code{\link{as.km}} before calling
#' \code{predict}.
#' 
#' @title Prediction Method for a \code{NKM} Object
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param object \code{NKM} object.
#' @param newdata Matrix of "new" input points where to perform
#'     prediction.
#' @param type character giving the kriging type. For now only
#'     \code{"UK"} is possible.
#' @param se.compute Logical. Should the standard error be computed?
#' @param cov.compute Logical. Should the covariance matrix between
#'     newdata points be computed?
#' @param light.return Logical. If \code{TRUE}, no auxiliary results
#'     will be returned (such as the Cholesky root of the correlation
#'     matrix).
#' @param bias.correct Logical. If \code{TRUE} the UK variance and
#'     covariance are .
#' @param checkNames Logical to check the consistency of the column
#'     names between the design stored in \code{object@X} and the new
#'     one given \code{newdata}.
#' @param ... Ignored.
#'
#' @return A named list. The elements are the conditional mean and
#'     standard deviation (\code{mean} and \code{sd}), the predicted
#'     trend (\code{trend}) and the confidence limits (\code{lower95}
#'     and \code{upper95}). Optionnally, the conditional covariance matrix
#'     is returned in \code{cov}.
#' 
#' @importFrom stats qt
#' @method predict NKM
#' @exportMethod predict
#' @aliases predict,NKM-method
#'
#' @examples
#' ## a 16-points factorial design, and the corresponding response
#' d <- 2; n <- 16
#' design.fact <- expand.grid(x1 = seq(0, 1, length = 4), x2 = seq(0, 1, length = 4))
#' y <- apply(design.fact, 1, DiceKriging::branin) + rnorm(nrow(design.fact))
#' 
#' ## library(DiceKriging)
#' ## kriging model 1 : matern5_2 covariance structure, no trend, no nugget
#' ## m1 <- km(design = design.fact, response = y, covtype = "gauss",
#'             nugget.estim=TRUE,
#' ##          parinit = c(.5, 1), control = list(trace = FALSE))
#' KM1 <- NKM(design = design.fact, response = y, covtype = "gauss",
#'                parinit = c(.5, 1))
#' Pred <- predict(KM1, newdata = matrix(.5,ncol = 2), type = "UK",
#'                 checkNames = FALSE, light.return = TRUE)
#' 
setMethod("predict", "NKM", predict.NKM)


## *****************************************************************************
## 'simulate' S4 method We no longer export 'simulate.NKM'
## *****************************************************************************

simulate.NKM <- function(object, nsim = 1, seed = NULL, newdata,
                           cond = TRUE, nugget.sim = 0,
                           checkNames = FALSE, ...) {
  if (length(L <- list(...)) > 0) warnOnDots(L)
  if (isTRUE(checkNames)) stop("'checkNames = TRUE' unsupported.")
  if (!isTRUE(cond)) stop("'cond = FALSE' unsupported.")
  if (nugget.sim!=0) stop("'nugget.sim != 0' unsupported.")
  
  return(simulate.NuggetKriging(object = object@NuggetKriging,
                          x = newdata,nsim = nsim, seed = seed))
}

## *****************************************************************************
#' The \code{simulate} method is used to simulate paths from the
#' kriging model described in \code{object}.
#'
#' Without a dedicated \code{simulate} method for the class
#' \code{"NKM"}, this method would have been inherited from the
#' \code{"km"} class. The dedicated method is expected to run faster.
#' A comparison can be made by coercing a \code{NKM} object to a
#' \code{km} object with \code{\link{as.km}} before calling
#' \code{simulate}.
#'
#' @title Simulation from a \code{NKM} Object
#' 
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param object A \code{NKM} object.
#'
#' @param nsim Integer: number of response vectors to simulate.
#'
#' @param seed Random seed.
#' 
#' @param newdata Numeric matrix with it rows giving the points where
#'     the simulation is to be performed.
#'
#' @param cond Logical telling wether the simulation is conditional
#'     or not. Only \code{TRUE} is accepted for now.
#'
#' @param nugget.sim Numeric. A postive nugget effect used to avoid
#'     numerical instability.
#'
#' @param checkNames Check consistency between the design data
#'     \code{X} within \code{object} and \code{newdata}. The default
#'     is \code{FALSE}. XXXY Not used!!!
#'
#' @param ... Ignored.
#'
#' @return A numeric matrix with \code{nrow(newdata)} rows and
#'     \code{nsim} columns containing as its columns the simulated
#'     paths at the input points given in \code{newdata}.
#' 
#' XXX method simulate NKM
#' @export
#' @aliases simulate,NKM-method
#' @exportMethod simulate
#'
#' @examples
#' f <-  function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X) + 0.01*rnorm(nrow(X))
#' points(X, y, col = 'blue')
#' k <- NKM(design = X, response = y, covtype = "gauss")
#' x <- seq(from = 0, to = 1, length.out = 101)
#' s_x <- simulate(k, nsim = 3, newdata = x)
#' lines(x, s_x[ , 1], col = 'blue')
#' lines(x, s_x[ , 2], col = 'blue')
#' lines(x, s_x[ , 3], col = 'blue')
#' 
setMethod("simulate", "NKM", simulate.NKM)

## *****************************************************************************
## 'update' S4 method We no longer export 'update.NKM'
## *****************************************************************************

update.NKM <- function(object,
                      newX,
                      newy,
                      newX.alreadyExist =  FALSE,
                      cov.reestim = TRUE,trend.reestim = cov.reestim,
                      nugget.reestim = FALSE,
                      newnoise.var = NULL,
                      kmcontrol = NULL, newF = NULL,
                      ...) {
    
    if (length(list(...)) > 0) warnOnDots()
    
    if (isTRUE(newX.alreadyExist))
        stop("'newX.alreadyExist = TRUE' unsupported.")
    if (!is.null(newnoise.var))
        stop("'newnoise.var != NULL' unsupported.")
    if (!is.null(kmcontrol)) stop("'kmcontrol != NULL' unsupported.")
    if (!is.null(newF)) stop("'newF != NULL' unsupported.")

    ## duplicate to avoid changing 'object' in an inconsistent
    ## way (
    obK <- object@NuggetKriging
    update.NuggetKriging(obK, newy, newX)
    
    return(as.km(obK))
    
}

## *****************************************************************************

#' The \code{update} method is used when new observations are added
#' to a fitted kriging model. Rather than fitting the model from
#' scratch with the updated observations added, the results of the
#' fit as stored in \code{object} are used to achieve some savings.
#'
#' Without a dedicated \code{update} method for the class
#' \code{"NKM"}, this would have been inherited from the class
#' \code{"km"}. The dedicated method is expected to run faster.  A
#' comparison can be made by coercing a \code{NKM} object to a
#' \code{km} object with \code{\link{as.km}} before calling
#' \code{update}.
#'
#' @title Update a \code{NKM} Object with New Points
#'
#' @author Yann Richet \email{yann.richet@irsn.fr}
#' 
#' @param object A NKM object.
#' @param newX A numeric matrix containing the new design points. It
#'     must have \code{object@d} columns in correspondence with those
#'     of the design matrix used to fit the model which is stored as
#'     \code{object@X}.
#' @param newy A numeric vector of new response values, in
#'     correspondence with the rows of \code{newX}.
#' @param newX.alreadyExist Logical. If TRUE, \code{newX} can contain
#'     some input points that are already in \code{object@X}.
#' @param cov.reestim Logical. If \code{TRUE}, the vector
#'     \code{theta} of correlation ranges will be re-estimated using
#'     the new observations as well as the observations already used
#'     when fitting \code{object}. Only \code{TRUE} can be used for
#'     now.
#' @param trend.reestim Logical. If \code{TRUE} the vector
#'     \code{beta} of trend coefficients will be re-estimated using
#'     all the observations. Only \code{TRUE} can be used for now.
#' @param nugget.reestim Logical. If \code{TRUE} the nugget effect
#'     will be re-estimated using all the observations. Only
#'     \code{FALSE} can be used for now.
#' @param newnoise.var Optional variance of an additional noise on
#'     the new response.
#' @param kmcontrol A list of options to tune the fit. Not available
#'     yet.
#' @param newF New trend matrix. XXXY?
#' @param ... Ignored.
#'
#' @return The updated \code{NKM} object.
#' 
#' @seealso \code{\link{as.km}} to coerce a \code{NKM} object to the
#'     class \code{"km"}.
#'
#' @export
#' @exportMethod update
#' @aliases update,NKM-method
#' 
#' @examples
#' f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
#' plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X) + 0.01*rnorm(nrow(X))
#' points(X, y, col = "blue")
#' KMobj <- NKM(design = X, response = y,covtype = "gauss")
#' x <-  seq(from = 0, to = 1, length.out = 101)
#' p_x <- predict(KMobj, x)
#' lines(x, p_x$mean, col = "blue")
#' lines(x, p_x$lower95, col = "blue")
#' lines(x, p_x$upper95, col = "blue")
#' newX <- as.matrix(runif(3))
#' newy <- f(newX)
#' points(newX, newy, col = "red")
#' 
#' ## replace the object by its udated version
#' KMobj <- update(KMobj, newy, newX)
#'
#' x <- seq(from = 0, to = 1, length.out = 101)
#' p2_x <- predict(KMobj, x)
#' lines(x, p2_x$mean, col = "red")
#' lines(x, p2_x$lower95, col = "red")
#' lines(x, p2_x$upper95, col = "red")
#' 
setMethod("update", "NKM", update.NKM)

