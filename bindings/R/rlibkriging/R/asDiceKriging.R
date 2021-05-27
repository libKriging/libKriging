# loadDiceKriging <- function() {
#' @importFrom utils installed.packages
if (! "DiceKriging" %in% utils::installed.packages())
  warning("DiceKriging must be installed to use its wrapper from libKriging.")
if (!"DiceKriging" %in% loadedNamespaces())
  try(library(DiceKriging))
if (!isClass("as_km"))
  try(setClass("as_km",slots = "Kriging",contains="km"))
# }


#' Convert a "Kriging" object to a DiceKriging::km one.
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param k "Kriging" object 
#' @param .call Force the "call" filed in km object 
#'
#' @return as_km object, extends DiceKriging::km plus contains "Kriging" field
#'
#' @importFrom utils installed.packages
#' @importFrom methods new
#' @importFrom stats model.matrix
#' @export as_km
#' @method as_km Kriging
#' @aliases as_km,Kriging,Kriging-method
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#' r <- Kriging(y, X, "gauss")
#' print(r)
#' k <- as_km(r)
#' print(k)
as_km.Kriging <- function(k,.call=NULL) {
  # loadDiceKriging()
  if (! "DiceKriging" %in% installed.packages())
    stop("DiceKriging must be installed to use its wrapper from libKriging.")
  
  model <- new("as_km")
  model@Kriging = k
  
  if (is.null(.call))
    model@call <- match.call()
  else
    model@call <- .call
  
  m = kriging_model(k)
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
  model@method = m$objective
  model@optim.method <- m$optim
  
  model@penalty = list()
  model@lower = 0
  model@upper = Inf
  model@control = list()
  
  model@gr = FALSE
  
  model@T = m$T
  model@z = as.numeric(m$z)
  model@M = m$M
  
  covStruct =  new("covTensorProduct", d=model@d, name=m$kernel, 
                   sd2 = m$sigma2, var.names=names(data), 
                   nugget = 0, nugget.flag=FALSE, nugget.estim=FALSE, known.covparam="") 
  covStruct@range.names  = "theta"
  covStruct@paramset.n <- as.integer(1)
  covStruct@param.n <- as.integer(model@d)
  covStruct@range.n <- as.integer(model@d)
  covStruct@range.val = as.numeric(m$theta)
  model@covariance = covStruct 
  
  return(model)
}

#' Build a DiceKriging "km" like object.
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param formula R formula object to setup the linear trend (aka Universal Kriging). Supports ~1, ~. and ~.^2
#' @param design data.frame of design of experiments
#' @param response array of output values
#' @param covtype covariance structure. Supports "gauss", "exp", ...
#' @param coef.cov fixed covariance range value (so will not optimize if given)
#' @param coef.var fixed variance value (so will not estimate if given)
#' @param coef.trend fixed trend value (so will not estimate if given)
#' @param estim.method estimation criterion. Supports "MLE" or "LOO"
#' @param optim.method optimization algorithm used on estim.method objective. Supports "BFGS"
#' @param parinit initial values of covariance range which will be optimzed using optim.method
#' @param ... Ignored
#'
#' @return as_km object, extends DiceKriging::km (plus contains a "Kriging" field which contains original object)
#'
#' @export as_km
#' @method as_km default
#' @examples
#' # a 16-points factorial design, and the corresponding response
#' d <- 2; n <- 16
#' design.fact <- expand.grid(x1=seq(0,1,length=4), x2=seq(0,1,length=4))
#' y <- apply(design.fact, 1, DiceKriging::branin) 
#' 
#' #library(DiceKriging)
#' # kriging model 1 : matern5_2 covariance structure, no trend, no nugget effect
#' #m1 <- km(design=design.fact, response=y,covtype = "gauss",parinit = c(.5,1),control = list(trace=F))
#' as_m1 <- as_km(design=design.fact, response=y,covtype = "gauss",parinit = c(.5,1))
as_km.default <- function(formula=~1, design, response, covtype="matern5_2",
                  coef.cov = NULL, coef.var = NULL, coef.trend = NULL,
                  estim.method="MLE", optim.method = "BFGS", parinit = NULL, ...) {
  formula = formula2regmodel(formula)
  
  if (!is.matrix(design)) design = as.matrix(design)
  if (!is.matrix(response)) response = matrix(response, nrow=nrow(design))
  
  if (estim.method=="MLE")
    estim.method = "LL"
  else if (estim.method=="LOO")
    estim.method = "LOO"
  else stop("Unsupported estim.method ",estim.method)
  
  if (!(covtype=="gauss" | covtype=="exp" | covtype=="matern3_2" | covtype=="matern5_2"))
    stop("Unsupported covtype ",covtype)
  
  if (optim.method!="BFGS")
    warning("Cannot setup optim.method ",optim.method,". Ignored.")
  
  parameters=list()
  if (!is.null(coef.var))
    parameters = c(parameters,list(sigma2=coef.var))
  if (!is.null(coef.cov)) {
    parameters = c(parameters,list(theta=matrix(coef.cov,ncol=ncol(design))))
    optim.method = "none"
  }  
  if (!is.null(coef.trend)) {
    parameters = c(parameters,list(beta=matrix(coef.trend)))
  }
  if (!is.null(parinit)) {
    parameters = c(parameters,list(theta=matrix(parinit,ncol=ncol(design))))
  }
  if (length(parameters)==0) 
    parameters=NULL
  
  r = rlibkriging::Kriging(y=response, X=design, kernel = covtype, regmodel = formula,
                           normalize = F, 
                           objective = estim.method, optim = optim.method, parameters = parameters)
  
  return(as_km.Kriging(r,.call=match.call()))
}

#' Build a "as_km" object, which extends DiceKriging::km S4 class.
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param ... args
#'
#' @return as_km/km object
#' @export
as_km <- function(...) UseMethod("as_km")
setMethod("as_km", "Kriging", as_km.Kriging)
setGeneric(name = "as_km", def = function(...) standardGeneric("as_km"))


#' Overload DiceKriging::predict.km for as_km objects (expected faster).
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object as_km object
#' @param newdata matrix of points where to perform prediction
#' @param type kriging family ("UK")
#' @param se.compute compute standard error (TRUE by default)
#' @param cov.compute compute covariance matrix between newdata points (FALSE by default)
#' @param light.return return no other intermediate objects (like T matrix) (default is TRUE)
#' @param bias.correct fix UK variance and covaariance (defualt is FALSE)
#' @param checkNames check consistency between object design data: X and newdata (default is FALSE) 
#' @param ... Ignored
#'
#' @return list of predict data: mean, sd, trend, cov, upper95 and lower95 quantiles.
#' 
#' @importFrom stats qt
#' @method predict as_km
#' @export predict
#' @aliases predict,as_km,as_km-method
#'
#' @examples
#' # a 16-points factorial design, and the corresponding response
#' d <- 2; n <- 16
#' design.fact <- expand.grid(x1=seq(0,1,length=4), x2=seq(0,1,length=4))
#' y <- apply(design.fact, 1, DiceKriging::branin) 
#' 
#' #library(DiceKriging)
#' # kriging model 1 : matern5_2 covariance structure, no trend, no nugget effect
#' #m1 <-      km(design=design.fact, response=y,covtype = "gauss",parinit = c(.5,1),control = list(trace=F))
#' as_m1 <- as_km(design=design.fact, response=y,covtype = "gauss",parinit = c(.5,1))
#' as_p = predict(as_m1,newdata=matrix(.5,ncol=2),type="UK",checkNames=FALSE,light.return=TRUE)
predict.as_km <- function(object, newdata, type="UK",
                          se.compute = TRUE, cov.compute = FALSE, light.return = TRUE,
                          bias.correct = FALSE, checkNames = FALSE,...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  if (isTRUE(checkNames)) stop("checkNames=TRUE unsupported.")
  if (isTRUE(bias.correct)) stop("bias.correct=TRUE unsupported.")
  if (!isTRUE(light.return)) stop("light.return=FALSE unsupported.")
  if (type!="UK") stop("type!=UK unsupported.")
  
  y.predict = predict.Kriging(object@Kriging, x = newdata, stdev=se.compute, cov=cov.compute)
  
  output.list <- list()
  # output.list$trend <- y.predict.trend
  output.list$mean <- y.predict$mean
  
  if (se.compute) {		
    s2.predict <- y.predict$stdev^2
    q95 <- qt(0.975, object@n - object@p)
    
    lower95 <- y.predict$mean - q95*sqrt(s2.predict)
    upper95 <- y.predict$mean + q95*sqrt(s2.predict)
    
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

setMethod("predict", "as_km", predict.as_km)


#' Overload DiceKriging::simulate.km for as_km objects (expected faster).
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object as_km object
#' @param nsim number of response vector to simulate
#' @param seed random seed
#' @param newdata matrix of points where to perform prediction
#' @param cond simulate conditional samples (only TRUE accepted)
#' @param nugget.sim numercial ngget ,effect to avoid numerical unstabilities
#' @param checkNames check consistency between object design data: X and newdata (default is FALSE) 
#' @param ... Ignored
#'
#' @return length(x) x nsim matrix containing simulated path at newdata points
#' 
#' @method simulate as_km
#' @export simulate
#' @aliases simulate,as_km,as_km-method
#'
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#'   plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#'   points(X,y,col='blue')
#' k <- as_km(design=X, response=y,covtype = "gauss")
#' x = seq(0,1,,101)
#' s_x = simulate(k, nsim=3, newdata=x)
#'   lines(x,s_x[,1],col='blue')
#'   lines(x,s_x[,2],col='blue')
#'   lines(x,s_x[,3],col='blue')
simulate.as_km <- function(object, nsim = 1, seed = NULL, newdata,
                           cond = TRUE, nugget.sim = 0, checkNames = FALSE, ...) {
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  if (isTRUE(checkNames)) stop("checkNames=TRUE unsupported.")
  if (!isTRUE(cond)) stop("cond=FALSE unsupported.")
  if (nugget.sim!=0) stop("nugget.sim!=0 unsupported.")
  
  return(simulate.Kriging(object = object@Kriging,x = newdata,nsim = nsim, seed=seed))
}

setMethod("simulate", "as_km", simulate.as_km)


#' Overload DiceKriging::update.km methd for as_km objects (expected faster).
#'
#' @author Yann Richet (yann.richet@irsn.fr)
#' 
#' @param object as_km object
#' @param newX new design points: matrix of object@d columns
#' @param newy new response points
#' @param newX.alreadyExist if TRUE, newX contains some ppoints already in object@X
#' @param cov.reestim fit object to newdata: estimate theta (only supports TRUE)
#' @param trend.reestim fit object to newdata: estimate beta (only supports TRUE)
#' @param nugget.reestim fit object to newdata: estimate nugget effect (only support FALSE)
#' @param newnoise.var add noise to newy response
#' @param kmcontrol parametrize fit (unsupported)
#' @param newF 
#' @param ... Ignored
#'
#' @method update as_km
#' @export update
#' @aliases update,as_km,as_km-method
#' @examples
#' f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
#'   plot(f)
#' set.seed(123)
#' X <- as.matrix(runif(5))
#' y <- f(X)
#'   points(X,y,col='blue')
#' k <- as_km(design=X, response=y,covtype = "gauss")
#' x = seq(0,1,,101)
#' p_x = predict(k, x)
#'   lines(x,p_x$mean,col='blue')
#'   lines(x,p_x$lower95,col='blue')
#'   lines(x,p_x$upper95,col='blue')
#' newX <- as.matrix(runif(3))
#' newy <- f(newX)
#'   points(newX,newy,col='red')
#' update(k,newy,newX)
#' x = seq(0,1,,101)
#' p2_x = predict(k, x)
#'   lines(x,p2_x$mean,col='red')
#'   lines(x,p2_x$lower95,col='red')
#'   lines(x,p2_x$upper95,col='red')
update.as_km <- function(object,
                         newX,
                         newy,
                         newX.alreadyExist =  FALSE,
                         cov.reestim = TRUE,trend.reestim = cov.reestim, nugget.reestim=FALSE,
                         newnoise.var = NULL, kmcontrol = NULL, newF = NULL, ...){
  if (length(list(...))>0) warning("Arguments ",paste0(names(list(...)),"=",list(...),collapse=",")," are ignored.")
  if (isTRUE(newX.alreadyExist)) stop("newX.alreadyExist=TRUE unsupported.")
  if (!is.null(newnoise.var)) stop("newnoise.var!=NULL unsupported.")
  if (!is.null(kmcontrol)) stop("kmcontrol!=NULL unsupported.")
  if (!is.null(newF)) stop("newF!=NULL unsupported.")
  
  update.Kriging(object@Kriging,newy,newX)
  
  return(object)
}

setMethod("update", "as_km", update.as_km)


formula2regmodel = function(form) {
  if (format(form) == "~1")
    return("constant")
  else if (format(form) == "~.")
    return("linear")
  else if (format(form) == "~.^2")
    return("interactive")
  else stop("Unsupported formula ",form)
}


regmodel2formula = function(regmodel) {
  if (regmodel == "constant")
    return(~1)
  else if (regmodel == "linear")
    return(~.)
  else if (regmodel == "interactive")
    return(~.^2)
  else stop("Unsupported regmodel ",regmodel)
}
